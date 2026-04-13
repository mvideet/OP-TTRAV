#!/usr/bin/env python3
# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Strip multiple-choice options from OmniVideo questions to produce an
open-ended variant of the dataset for open-ended TTRL.

Input: train.json / test.json with samples like
    {
      "question": "Which audio event occurred...?\nA. Sustained gunfire\nB. ...\nC. ...\nD. ...",
      "answer":   "C",
      ...
    }

Output: train_open.json / test_open.json with samples like
    {
      "question":    "Which audio event occurred...?",
      "answer":      "C",                                   # kept for backward-compat eval
      "answer_text": "Male speaker emphasizes 'new release...'",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},  # for reference eval
      ...
    }

The original train.json/test.json files are NOT touched. The traditional MCQ
TTRL pipeline continues to read those files unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Optional

# Match "A. text", "B) text", "(A) text", etc., as section headers.
# We split on the four canonical letters (A, B, C, D); other layouts (E, ...)
# fall through unchanged.
_OPTION_LINE_RE = re.compile(
    r"(?m)^\s*[\(\[]?\s*([A-D])[\)\]\.]\s+(.+?)\s*$"
)


def parse_options(question_text: str) -> tuple[str, dict[str, str]]:
    """
    Split the question text into (stem, options).

    Returns:
        stem: question text without the option block.
        options: dict mapping letter -> option text. Empty if parsing fails.
    """
    matches = list(_OPTION_LINE_RE.finditer(question_text))
    if not matches:
        return question_text.strip(), {}

    # The first option's start marks the end of the stem.
    stem_end = matches[0].start()
    stem = question_text[:stem_end].rstrip()

    options: dict[str, str] = {}
    for m in matches:
        letter = m.group(1)
        text = m.group(2).strip()
        options[letter] = text
    return stem, options


def reformulate_sample(sample: dict, drop_unparseable: bool) -> Optional[dict]:
    """
    Convert one MCQ sample to its open-ended form. Returns None if parsing
    fails and drop_unparseable=True.
    """
    raw_q = sample.get("question", "")
    answer_letter = (sample.get("answer") or "").strip().upper()

    stem, options = parse_options(raw_q)

    if not options or answer_letter not in options:
        if drop_unparseable:
            return None
        # Keep the sample but with an empty answer_text; the TTRL training
        # path doesn't need answer_text (it self-supervises via medoid voting),
        # so this is only a problem for reference-based eval.
        out = dict(sample)
        out["question"] = stem if stem else raw_q
        out["answer_text"] = ""
        out["options"] = options
        return out

    out = dict(sample)
    out["question"] = stem
    out["answer_text"] = options[answer_letter]
    out["options"] = options
    return out


def reformulate_file(in_path: str, out_path: str, drop_unparseable: bool) -> tuple[int, int, int]:
    with open(in_path) as f:
        data = json.load(f)
    n_total = len(data)
    out_samples = []
    n_dropped = 0
    n_no_answer_text = 0
    for s in data:
        new = reformulate_sample(s, drop_unparseable=drop_unparseable)
        if new is None:
            n_dropped += 1
            continue
        if not new.get("answer_text"):
            n_no_answer_text += 1
        out_samples.append(new)
    with open(out_path, "w") as f:
        json.dump(out_samples, f, ensure_ascii=False, indent=2)
    return n_total, len(out_samples), n_dropped


def make_subset(in_path: str, out_path: str, n: int, seed: int) -> None:
    with open(in_path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    if n >= len(data):
        subset = list(data)
    else:
        subset = rng.sample(data, n)
    with open(out_path, "w") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    print(f"  subset {n} -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="OmniVideo data directory containing train.json and test.json",
    )
    p.add_argument("--drop-unparseable", action="store_true",
                   help="drop samples whose options cannot be parsed (default: keep)")
    p.add_argument("--val-subset-n", type=int, default=20,
                   help="size of the periodic validation subset (default: 20)")
    p.add_argument("--sanity-n", type=int, default=40,
                   help="size of the sanity-check subset (default: 40)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_in = os.path.join(args.data_dir, "train.json")
    test_in = os.path.join(args.data_dir, "test.json")
    train_out = os.path.join(args.data_dir, "train_open.json")
    test_out = os.path.join(args.data_dir, "test_open.json")
    val20_out = os.path.join(args.data_dir, "test_open_val20.json")
    sanity_train_out = os.path.join(args.data_dir, "train_open_sanity.json")
    sanity_test_out = os.path.join(args.data_dir, "test_open_sanity.json")

    print(f"[strip_mcq] reading {train_in}")
    t_total, t_kept, t_dropped = reformulate_file(train_in, train_out, args.drop_unparseable)
    print(f"  train: total={t_total} kept={t_kept} dropped={t_dropped} -> {train_out}")

    print(f"[strip_mcq] reading {test_in}")
    s_total, s_kept, s_dropped = reformulate_file(test_in, test_out, args.drop_unparseable)
    print(f"  test:  total={s_total} kept={s_kept} dropped={s_dropped} -> {test_out}")

    print(f"[strip_mcq] making subsets")
    make_subset(test_out, val20_out, args.val_subset_n, args.seed)
    make_subset(train_out, sanity_train_out, args.sanity_n, args.seed)
    make_subset(test_out, sanity_test_out, args.sanity_n, args.seed)

    print(f"[strip_mcq] done")
    print()
    print("Sanity check on first sample:")
    with open(train_out) as f:
        first = json.load(f)[0]
    print(f"  question:    {first.get('question', '')[:200]}")
    print(f"  answer:      {first.get('answer', '')}")
    print(f"  answer_text: {first.get('answer_text', '')[:200]}")
    print(f"  options:     {first.get('options', {})}")


if __name__ == "__main__":
    sys.exit(main())
