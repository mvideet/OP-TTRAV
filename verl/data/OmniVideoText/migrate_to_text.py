#!/usr/bin/env python3
"""Migrate OmniVideo datasets to text-only (no audio/video paths)."""

import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "OmniVideo"
DST_DIR = Path(__file__).resolve().parent

# Keys to exclude from each entry (audio/video paths)
EXCLUDE_KEYS = {"audio_file", "video_file"}


def to_text_only(entry: dict) -> dict:
    """Return a copy of the entry without audio/video file paths."""
    return {k: v for k, v in entry.items() if k not in EXCLUDE_KEYS}


def migrate_file(filename: str) -> int:
    """Migrate a single JSON file. Returns number of entries."""
    src = SRC_DIR / filename
    dst = DST_DIR / filename
    if not src.exists():
        print(f"  Skip {filename} (not found)")
        return 0
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    text_data = [to_text_only(entry) for entry in data]
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(text_data, f, indent=2, ensure_ascii=False)
    print(f"  {filename}: {len(text_data)} entries")
    return len(text_data)


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    files = ["train.json", "test.json", "train_sanity.json", "test_sanity.json"]
    total = 0
    for f in files:
        total += migrate_file(f)
    print(f"Total: {total} entries migrated to {DST_DIR}")


if __name__ == "__main__":
    main()
