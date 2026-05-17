"""
Convert EgoSound (Ego4D or EgoBlind split) into VERL TTRL format.

EgoSound is an egocentric audio-visual QA benchmark (CVPR 2026). Each record
includes a video_path, question, gold answer, context, and question_type.

For TTRL semantics, train.json == test.json — the model adapts to the
test set at test time, so the same prompts are used for both rollouts
and offline eval.

Source JSON: {timestamp, context, question_type, question, answer, video_path, question_id}
VERL format: {prompt, answer, source, id, video_file, audio_file?, question_type}

Audio: EgoSound's video_path .mp4 files contain the audio track. The
multimodal pipeline can either extract audio from the video file directly
(use_audio_in_video=True) or expect a separate .wav. We use the video
file as the audio_file, letting the data loader extract.

Usage:
  python verl/scripts/build_egosound_ttrl.py \\
      --src /data/sls/scratch/mvideet/datasets/EgoSound/ego4d.json \\
      --video_root /data/sls/scratch/mvideet/datasets/EgoSound \\
      --out-dir verl/data/EgoSound-Ego4d-TTRL
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="ego4d.json or egoblind.json")
    p.add_argument("--video_root", required=True, help="Root dir containing the extracted Ego4d/ or EgoBlind/ folder")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None, help="Cap n records (default: all)")
    p.add_argument("--require_video", action="store_true",
                   help="Drop records whose video file doesn't exist on disk")
    args = p.parse_args()

    with open(args.src) as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} records from {args.src}")

    records = []
    dropped_missing_video = 0
    for r in raw:
        question = (r.get("question") or "").strip()
        answer = (r.get("answer") or "").strip()
        vp = r.get("video_path") or ""
        if not question or not answer:
            continue
        full_video_path = str(Path(args.video_root) / vp)
        if args.require_video and not os.path.exists(full_video_path):
            dropped_missing_video += 1
            continue
        records.append({
            "prompt": question,
            "answer": answer,
            "source": "egosound",
            "id": r.get("question_id", str(len(records))),
            "video_file": full_video_path,
            "audio_file": full_video_path,  # extract audio from video
            "question_type": r.get("question_type", "unknown"),
            "timestamp": r.get("timestamp", ""),
            "context": r.get("context", ""),
        })
        if args.limit and len(records) >= args.limit:
            break

    print(f"Kept {len(records)} records ({dropped_missing_video} dropped: missing video files)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TTRL semantics: train == test.
    for name, data in [("train", records), ("test", records), ("sanity", records[:30])]:
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  wrote {len(data):>5} rows -> {path}")

    if records:
        print("\nSchema preview (first record):")
        r = records[0]
        print(json.dumps({**r, "prompt": r["prompt"][:200],
                          "answer": r["answer"][:200],
                          "context": r["context"][:200]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
