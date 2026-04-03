#!/usr/bin/env python3
"""Convert MMAU parquet to OmniVideo-compatible JSON + extracted WAV files.

Usage:
    python convert_mmau.py \
        --parquet /data/sls/scratch/mvideet/datasets/MMAU-test-mini/test_mini.parquet \
        --audio-dir /data/sls/scratch/mvideet/Audio/MMAU \
        --out-json /data/sls/r/u/mvideet/TTRL/verl/data/MMAU/test_mini.json
"""

import argparse
import json
import os
import re

import pandas as pd


def extract_letter(answer_str: str) -> str:
    """'(A) Man' -> 'A'"""
    m = re.match(r"\(([A-Da-d])\)", answer_str.strip())
    return m.group(1).upper() if m else answer_str.strip()


def format_choices(choices) -> str:
    """['(A) Man', '(B) Woman', ...] -> 'A. Man\\nB. Woman\\n...'"""
    lines = []
    for c in choices:
        m = re.match(r"\(([A-Da-d])\)\s*(.*)", str(c).strip())
        if m:
            lines.append(f"{m.group(1).upper()}. {m.group(2)}")
        else:
            lines.append(str(c))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Path to MMAU parquet file")
    parser.add_argument("--audio-dir", required=True, help="Directory to write extracted .wav files")
    parser.add_argument("--out-json", required=True, help="Output JSON file path")
    parser.add_argument("--split-ratio", type=float, default=None,
                        help="If set, split into train/test by this ratio (e.g. 0.8 = 80%% train)")
    args = parser.parse_args()

    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} rows from {args.parquet}")

    records = []
    for idx, row in df.iterrows():
        attrs = {}
        if isinstance(row.get("other_attributes"), str):
            try:
                attrs = json.loads(row["other_attributes"])
            except json.JSONDecodeError:
                pass

        sample_id = attrs.get("id", str(idx))

        # Extract audio bytes to WAV file
        ctx = row["context"]
        audio_bytes = ctx["bytes"] if isinstance(ctx, dict) else ctx
        wav_path = os.path.join(args.audio_dir, f"{sample_id}.wav")
        if not os.path.exists(wav_path):
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)

        # Build question: instruction + formatted choices
        instruction = row["instruction"]
        choices = list(row["choices"])
        question = f"{instruction}\n{format_choices(choices)}"

        answer_letter = extract_letter(row["answer"])

        record = {
            "id": sample_id,
            "question": question,
            "audio_file": wav_path,
            "answer": answer_letter,
            "source": "mmau",
            "dataset": attrs.get("dataset", "MMAU"),
            "category": attrs.get("category", ""),
            "sub_category": attrs.get("sub-category", ""),
            "difficulty": attrs.get("difficulty", ""),
            "task": attrs.get("task", ""),
        }
        records.append(record)

    with open(args.out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records)} records to {args.out_json}")
    print(f"Audio files in {args.audio_dir}")

    if args.split_ratio is not None:
        import random
        random.seed(42)
        random.shuffle(records)
        split_idx = int(len(records) * args.split_ratio)
        train_records = records[:split_idx]
        test_records = records[split_idx:]

        base, ext = os.path.splitext(args.out_json)
        train_path = f"{base}_train{ext}"
        test_path = f"{base}_test{ext}"

        with open(train_path, "w") as f:
            json.dump(train_records, f, indent=2)
        with open(test_path, "w") as f:
            json.dump(test_records, f, indent=2)
        print(f"Split: {len(train_records)} train -> {train_path}")
        print(f"Split: {len(test_records)} test  -> {test_path}")


if __name__ == "__main__":
    main()
