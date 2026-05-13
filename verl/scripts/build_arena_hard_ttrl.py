"""
Convert Arena-Hard v2.0 questions into VERL TTRL train.json/test.json.

For TTRL semantics, train.json == test.json — the model adapts to the test
set at test time, so the same prompts are used for both rollouts and eval.

Source: lmarena-ai/arena-hard-auto, data/arena-hard-v2.0/question.jsonl
Schema (per record): {uid, category, subcategory, language, prompt}

Output schema (matches AIME-TTT for drop-in VERL compatibility):
  [{"prompt": str, "answer": "", "source": "arena-hard-v2.0", "id": str}, ...]

The "answer" field is empty — Arena-Hard doesn't provide gold answers
(it's evaluated via pairwise judge against a reference model). We won't
have a ground_truth diagnostic, only the training reward and offline
GPT-judge eval.

Usage:
  python verl/scripts/build_arena_hard_ttrl.py \\
      --src /data/sls/scratch/mvideet/datasets/arena-hard-v2.0/data/arena-hard-v2.0/question.jsonl \\
      --out-dir verl/data/ArenaHard-v2.0-TTRL
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None, help="Cap n records (default: all 750)")
    p.add_argument("--lang", default=None, help="Filter by language (e.g., 'en'). Default: keep all.")
    args = p.parse_args()

    records = []
    with open(args.src) as f:
        for line in f:
            r = json.loads(line)
            if args.lang and r.get("language") != args.lang:
                continue
            prompt = (r.get("prompt") or "").strip()
            if not prompt:
                continue
            records.append({
                "prompt": prompt,
                "answer": "",  # no gold; Arena-Hard uses pairwise judge
                "source": "arena-hard-v2.0",
                "id": r.get("uid", str(len(records))),
                "category": r.get("category", "unknown"),
                "subcategory": r.get("subcategory", "unknown"),
            })
            if args.limit and len(records) >= args.limit:
                break

    print(f"Loaded {len(records)} records from {args.src}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TTRL semantics: train == test (model adapts to the test set).
    train_path = out_dir / "train.json"
    test_path = out_dir / "test.json"
    sanity_path = out_dir / "sanity.json"

    for path, data in [(train_path, records), (test_path, records), (sanity_path, records[:30])]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  wrote {len(data):>4} rows -> {path}")

    print("\nSchema preview (first record):")
    r = records[0]
    print(json.dumps({**r, "prompt": r["prompt"][:200] + ("…" if len(r["prompt"]) > 200 else "")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
