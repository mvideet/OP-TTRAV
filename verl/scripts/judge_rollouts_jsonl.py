#!/usr/bin/env python3
"""
Run an LLM-as-judge over a saved rollouts JSONL.

Reads records produced by dump_rollouts.py (or eval_open_ended_judge.py's
per-sample dump) and judges each rollout against `gold` using one of:
  --judge-mode local    : load a local model as judge (default)
  --judge-mode anthropic: use Anthropic API (requires ANTHROPIC_API_KEY)
  --judge-mode openai   : use OpenAI API (requires OPENAI_API_KEY)

The point of this script: hold the JUDGE FIXED across multiple checkpoints
to remove the self-bias confound from eval_open_ended_judge.py. Generate
rollouts once with dump_rollouts.py for both step 0 and step 200, then
run this script with a single judge model — gives a clean apples-to-apples
score difference attributable only to rollout content.

Output:
  - augments the input JSONL by writing a new file with judge_score_external
    column added per record
  - aggregate CSV per step + per-category CSV

Usage:
  # Local judge (e.g. base Qwen as fixed judge):
  python verl/scripts/judge_rollouts_jsonl.py \\
      --rollouts rollouts_judge_v2.jsonl \\
      --judge-mode local \\
      --judge-model /data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B \\
      --output rollouts_judge_v2_basejudge.jsonl

  # API judge (cleanest, no self-bias possible):
  ANTHROPIC_API_KEY=sk-ant-... python verl/scripts/judge_rollouts_jsonl.py \\
      --rollouts rollouts_judge_v2.jsonl \\
      --judge-mode anthropic \\
      --judge-model claude-sonnet-4-6 \\
      --output rollouts_judge_v2_claude.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

JUDGE_TEMPLATE = (
    "You are evaluating answers to a video understanding question.\n\n"
    "Question: {question}\n\n"
    "Reference answer: {reference}\n\n"
    "Candidate answer: {candidate}\n\n"
    "How well does the candidate match the reference in factual content "
    "(ignore phrasing, focus on whether the same facts are stated)?\n\n"
    "Output a single integer from 0 (completely different) to 10 (identical "
    "in factual content). No explanation. Just the number.\n\n"
    "Score:"
)

_NUM_RE = re.compile(r"\b(\d{1,2})\b")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True, help="Path to rollouts JSONL")
    p.add_argument("--output", required=True, help="Output JSONL with judge scores added")
    p.add_argument("--csv-output", default=None, help="Aggregate CSV (default: derived from --output)")
    p.add_argument("--judge-mode", choices=["local", "anthropic", "openai"], default="local")
    p.add_argument("--judge-model", required=True,
                    help="local: path to model; anthropic: model id (e.g. claude-sonnet-4-6); openai: model id")
    p.add_argument("--judge-max-new-tokens", type=int, default=8)
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None, help="Limit to first N records (for testing)")
    return p.parse_args()


def parse_judge_score(text: str) -> tuple[float, bool]:
    if not text:
        return 0.5, False
    m = _NUM_RE.search(text)
    if not m:
        return 0.5, False
    raw = max(0, min(10, int(m.group(1))))
    return raw / 10.0, True


# -------------------------------------------------------------------------
# Local judge — uses verl's loader for FSDP-compatibility
# -------------------------------------------------------------------------

def make_local_judge(model_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_mmau_offline import load_model_with_checkpoint
    import torch

    thinker, processor = load_model_with_checkpoint(model_path, state_dict=None)

    def judge(question, gold, rollout, max_new_tokens=8):
        prompt = JUDGE_TEMPLATE.format(
            question=(question or "")[:4000],
            reference=(gold or "[empty]")[:4000],
            candidate=(rollout or "[empty]")[:4000],
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            try:
                out = thinker.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, temperature=1.0, top_p=1.0, use_cache=True,
                )
            except Exception:
                return "", 0.5, False
        in_len = inputs["input_ids"].shape[1]
        decoded = processor.decode(out[0, in_len:], skip_special_tokens=True)
        sc, ok = parse_judge_score(decoded)
        return decoded, sc, ok

    return judge


# -------------------------------------------------------------------------
# Anthropic API judge
# -------------------------------------------------------------------------

def make_anthropic_judge(model_id):
    import anthropic
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    def judge(question, gold, rollout, max_new_tokens=8):
        prompt = JUDGE_TEMPLATE.format(
            question=(question or "")[:4000],
            reference=(gold or "[empty]")[:4000],
            candidate=(rollout or "[empty]")[:4000],
        )
        try:
            msg = client.messages.create(
                model=model_id,
                max_tokens=max_new_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text if msg.content else ""
        except Exception as e:
            print(f"  anthropic API err: {e}", file=sys.stderr)
            return "", 0.5, False
        sc, ok = parse_judge_score(text)
        return text, sc, ok

    return judge


# -------------------------------------------------------------------------
# OpenAI API judge
# -------------------------------------------------------------------------

def make_openai_judge(model_id):
    from openai import OpenAI
    client = OpenAI()  # uses OPENAI_API_KEY env var

    def judge(question, gold, rollout, max_new_tokens=8):
        prompt = JUDGE_TEMPLATE.format(
            question=(question or "")[:4000],
            reference=(gold or "[empty]")[:4000],
            candidate=(rollout or "[empty]")[:4000],
        )
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_completion_tokens=max_new_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  openai API err: {e}", file=sys.stderr)
            return "", 0.5, False
        sc, ok = parse_judge_score(text)
        return text, sc, ok

    return judge


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.judge_mode == "local":
        judge_fn = make_local_judge(args.judge_model)
    elif args.judge_mode == "anthropic":
        judge_fn = make_anthropic_judge(args.judge_model)
    elif args.judge_mode == "openai":
        judge_fn = make_openai_judge(args.judge_model)
    else:
        raise ValueError(args.judge_mode)

    # Read rollouts JSONL
    records = []
    with open(args.rollouts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"Loaded {len(records)} rollout records from {args.rollouts}")

    if args.limit:
        records = records[: args.limit]
        print(f"Limited to first {len(records)} records")

    # Aggregation buffers
    per_step = defaultdict(lambda: {"scores": [], "by_cat": defaultdict(list), "parse_ok": 0, "n_judged": 0})

    # Stream output
    out_path = Path(args.output)
    out_writer = open(out_path, "w")

    t0 = time.time()
    for i, rec in enumerate(tqdm(records, desc=f"  judging via {args.judge_mode}:{args.judge_model}")):
        rollouts = rec.get("rollouts", [])
        gold = rec.get("gold", "")
        question = rec.get("question", "")
        step = rec.get("step", -1)
        category = rec.get("category", "unknown")

        scores, raws = [], []
        for r in rollouts:
            raw, sc, ok = judge_fn(question, gold, r, max_new_tokens=args.judge_max_new_tokens)
            scores.append(sc)
            raws.append(raw)
            if ok:
                per_step[step]["parse_ok"] += 1

        agg = float(np.mean(scores)) if scores else 0.0
        new_rec = dict(rec)
        new_rec["judge_external_scores"] = scores
        new_rec["judge_external_raws"] = raws
        new_rec["judge_external_agg"] = agg
        new_rec["judge_external_mode"] = args.judge_mode
        new_rec["judge_external_model"] = args.judge_model
        out_writer.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
        out_writer.flush()

        per_step[step]["scores"].append(agg)
        per_step[step]["by_cat"][category].append(agg)
        per_step[step]["n_judged"] += len(rollouts)

        if i < 3:
            print(f"\n  === SANITY {i} step={step} ===")
            print(f"  id: {rec.get('id', i)}")
            print(f"  question: {question[:160]}")
            print(f"  gold:     {gold[:160]}")
            print(f"  rollout0: {(rollouts[0] if rollouts else '<EMPTY>')[:160]}")
            print(f"  judge_raw0: {raws[0]!r}  scores: {scores}  agg: {agg:.3f}")

    out_writer.close()
    elapsed = time.time() - t0

    # Aggregate CSV
    csv_path = Path(args.csv_output) if args.csv_output else out_path.with_suffix(".csv")
    cat_csv_path = csv_path.parent / (csv_path.stem + "_by_category" + csv_path.suffix)
    with open(csv_path, "w", newline="") as csvf, open(cat_csv_path, "w", newline="") as catf:
        w = csv.writer(csvf)
        w.writerow(["step", "judge_mode", "judge_model", "judge_mean", "judge_std",
                    "frac_zero", "frac_full", "parse_rate", "n_records", "time_s"])
        cw = csv.writer(catf)
        cw.writerow(["step", "category", "judge_mean", "n"])

        print(f"\n{'=' * 70}")
        print(f"  judge: {args.judge_mode}:{args.judge_model}")
        print(f"{'=' * 70}")
        print(f"  {'step':>5}  {'mean':>6}  {'std':>5}  {'frac0':>6}  {'frac1':>6}  {'parse':>6}  {'n':>5}")
        for step in sorted(per_step):
            arr = np.array(per_step[step]["scores"])
            n = len(arr)
            mean = float(arr.mean()) if n else 0.0
            std = float(arr.std()) if n > 1 else 0.0
            fz = float((arr == 0.0).mean()) if n else 0.0
            ff = float((arr == 1.0).mean()) if n else 0.0
            parse_rate = per_step[step]["parse_ok"] / max(1, per_step[step]["n_judged"])
            print(f"  {step:>5}  {mean:>6.3f}  {std:>5.3f}  {fz:>6.3f}  {ff:>6.3f}  {parse_rate:>6.3f}  {n:>5}")
            w.writerow([step, args.judge_mode, args.judge_model,
                        f"{mean:.4f}", f"{std:.4f}",
                        f"{fz:.4f}", f"{ff:.4f}",
                        f"{parse_rate:.4f}", n, f"{elapsed:.1f}"])
            for cat, scores in sorted(per_step[step]["by_cat"].items()):
                cw.writerow([step, cat, f"{float(np.mean(scores)):.4f}", len(scores)])

    print(f"\n  augmented JSONL: {out_path}")
    print(f"  aggregate CSV:   {csv_path}")
    print(f"  per-cat CSV:     {cat_csv_path}")


if __name__ == "__main__":
    main()
