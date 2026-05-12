"""
Convert UltraFeedback into VERL-style train.json / test.json for TTRL on a
text-only base model.

For TTRL the *labels* (the 4 GPT-4 ratings) are discarded — we only keep
the instruction. The highest-rated of the 4 responses is preserved as the
"answer" field, but it's used only for the offline aux GPT-judge monitor
(BLEU/ROUGE/aux_gpt_judge_mean), NEVER as a training signal. Set
`--no-reference` to drop it entirely.

Output schema mirrors AIME-TTT for drop-in compatibility:
  [{"prompt": str, "answer": str, "source": "ultrafeedback", "id": str}, ...]

Usage:
  python verl/scripts/build_ultrafeedback_ttrl.py \\
      --src /data/sls/scratch/mvideet/datasets/UltraFeedback \\
      --out-dir verl/data/UltraFeedback-TTRL \\
      --train-n 4000 --test-n 500 --sanity-n 50 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def _pick_best_response(row):
    """Pick the highest-overall-rated of the 4 completions. Returns "" if none."""
    completions = row.get("completions") or []
    if not completions:
        return ""
    best = ""
    best_score = -1.0
    for c in completions:
        if not isinstance(c, dict):
            continue
        try:
            avg = float(c.get("overall_score") or 0.0)
        except (TypeError, ValueError):
            avg = 0.0
        if avg > best_score and c.get("response"):
            best_score = avg
            best = c["response"]
    return best


def _is_clean(prompt: str, response: str, min_p=10, max_p=2000, max_r=4000):
    if not prompt or len(prompt) < min_p or len(prompt) > max_p:
        return False
    if response and len(response) > max_r:
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="HF datasets snapshot path (load_from_disk)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-n", type=int, default=4000)
    p.add_argument("--test-n", type=int, default=500)
    p.add_argument("--sanity-n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-reference", action="store_true",
                   help="Drop the reference response (answer field becomes empty)")
    args = p.parse_args()

    from datasets import load_from_disk
    print(f"Loading UltraFeedback from {args.src}...")
    ds = load_from_disk(args.src)
    print(f"Loaded {len(ds)} rows. Columns: {ds.column_names}")

    # Pick instruction column. UltraFeedback usually has "instruction".
    instr_key = None
    for cand in ["instruction", "prompt", "question"]:
        if cand in ds.column_names:
            instr_key = cand
            break
    assert instr_key is not None, f"could not find instruction column; have {ds.column_names}"
    print(f"Using {instr_key!r} as the instruction field.")

    records = []
    skipped = 0
    for i, row in enumerate(ds):
        prompt = (row.get(instr_key) or "").strip()
        if args.no_reference:
            response = ""
        else:
            response = _pick_best_response(row).strip()
        if not _is_clean(prompt, response):
            skipped += 1
            continue
        records.append({
            "prompt": prompt,
            "answer": response,
            "source": "ultrafeedback",
            "id": str(i),
        })
    print(f"Kept {len(records)} / {len(ds)} records (skipped {skipped} too-short / too-long)")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    n_need = args.train_n + args.test_n + args.sanity_n
    assert len(records) >= n_need, f"only {len(records)} clean records, need {n_need}"

    sanity = records[: args.sanity_n]
    test = records[args.sanity_n : args.sanity_n + args.test_n]
    train = records[args.sanity_n + args.test_n : args.sanity_n + args.test_n + args.train_n]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train), ("test", test), ("sanity", sanity)]:
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        print(f"  wrote {len(split):>5} rows -> {path}")

    print("\nSchema preview (first train record):")
    print(json.dumps({**train[0], "prompt": train[0]["prompt"][:200] + ("…" if len(train[0]["prompt"]) > 200 else "")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
