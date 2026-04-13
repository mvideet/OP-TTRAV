#!/usr/bin/env python3
"""Download OmniBench from HuggingFace and convert to our eval JSON format.

Creates:
  - test.json: all 1114 samples with local file paths
  - mm_data/audio/*.mp3: audio files
  - mm_data/image/*.{png,jpg}: image files

Usage:
    python verl/data/OmniBench/prepare.py [--out-dir verl/data/OmniBench]
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "mm_data" / "audio"
    image_dir = out_dir / "mm_data" / "image"
    audio_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OmniBench from HuggingFace...")
    from datasets import load_dataset
    # Disable auto-decoding of audio/image (torchcodec not available)
    ds = load_dataset("m-a-p/OmniBench", split="train")
    ds = ds.cast_column("audio", ds.features["audio"])  # keep as-is
    # Load raw — avoid torchcodec requirement by using with_format
    try:
        # Try iterating; if audio decode fails, reload without decoding
        _ = ds[0]
    except ImportError:
        print("  Audio auto-decode not available, loading raw bytes...")
        from datasets import Audio, Image
        ds = load_dataset("m-a-p/OmniBench", split="train")
        # Disable automatic decoding
        ds = ds.cast_column("audio", Audio(decode=False))
        ds = ds.cast_column("image", Image(decode=False))
    print(f"Loaded {len(ds)} samples")
    print(f"Columns: {ds.column_names}")

    samples = []
    for i, row in enumerate(ds):
        # Save audio file — handle both decoded (dict with array) and raw (dict with bytes/path)
        audio_path_rel = row.get("audio_path", f"audio_{i}.mp3")
        audio_basename = os.path.basename(audio_path_rel)
        audio_out = audio_dir / audio_basename
        audio_obj = row.get("audio")
        if not audio_out.exists() and audio_obj is not None:
            if isinstance(audio_obj, dict) and "bytes" in audio_obj and audio_obj["bytes"]:
                audio_out.write_bytes(audio_obj["bytes"])
            elif isinstance(audio_obj, dict) and "path" in audio_obj and audio_obj["path"]:
                import shutil
                src = audio_obj["path"]
                if os.path.exists(src):
                    shutil.copy2(src, str(audio_out))
            elif isinstance(audio_obj, dict) and "array" in audio_obj:
                import soundfile as sf
                wav_out = str(audio_out).replace('.mp3', '.wav')
                sf.write(wav_out, audio_obj["array"], audio_obj["sampling_rate"])
                audio_out = Path(wav_out)
            elif isinstance(audio_obj, bytes):
                audio_out.write_bytes(audio_obj)

        # Save image file — handle both PIL Image and raw bytes
        image_path_rel = row.get("image_path", f"image_{i}.png")
        image_basename = os.path.basename(image_path_rel)
        image_out = image_dir / image_basename
        image_obj = row.get("image")
        if not image_out.exists() and image_obj is not None:
            if isinstance(image_obj, dict) and "bytes" in image_obj and image_obj["bytes"]:
                image_out.write_bytes(image_obj["bytes"])
            elif isinstance(image_obj, dict) and "path" in image_obj and image_obj["path"]:
                import shutil
                src = image_obj["path"]
                if os.path.exists(src):
                    shutil.copy2(src, str(image_out))
            elif hasattr(image_obj, "save"):
                image_obj.save(str(image_out))
            elif isinstance(image_obj, bytes):
                image_out.write_bytes(image_obj)

        # Build question with MCQ options
        options = row.get("options", [])
        question_text = row["question"]
        if options:
            letters = ["A", "B", "C", "D"]
            for j, opt in enumerate(options):
                if j < len(letters):
                    question_text += f"\n{letters[j]}. {opt}"

        # Resolve actual audio file path (may be .wav if converted)
        actual_audio = str(audio_out)
        if not os.path.exists(actual_audio):
            wav_alt = actual_audio.replace('.mp3', '.wav')
            if os.path.exists(wav_alt):
                actual_audio = wav_alt

        sample = {
            "id": str(row.get("index", i)),
            "question": question_text,
            "answer": row["answer"],
            "image_file": str(image_out),
            "audio_file": actual_audio,
            "task_type": row.get("task_type", row.get("task type", "unknown")),
            "audio_type": row.get("audio_type", row.get("audio type", "unknown")),
            "audio_content": row.get("audio_content", ""),
            "image_content": row.get("image_content", ""),
            "source": "OmniBench",
        }
        samples.append(sample)

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(ds)}")

    # Write JSON
    out_path = out_dir / "test.json"
    with open(out_path, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(samples)} samples to {out_path}")

    # Verify a few samples
    print("\nSample 0:")
    s = samples[0]
    for k in ["id", "question", "answer", "image_file", "audio_file", "task_type"]:
        val = s[k]
        if len(str(val)) > 120:
            val = str(val)[:120] + "..."
        print(f"  {k}: {val}")
    print(f"  image exists: {os.path.exists(s['image_file'])}")
    print(f"  audio exists: {os.path.exists(s['audio_file'])}")

    # Category distribution
    from collections import Counter
    cats = Counter(s["task_type"] for s in samples)
    print(f"\nCategories:")
    for k, v in sorted(cats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
