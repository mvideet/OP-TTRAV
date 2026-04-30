#!/usr/bin/env python3
"""
LLM-as-judge eval for open-ended TTRL checkpoints on OmniVideo.

Workflow per checkpoint:
  1. Load FSDP-merged thinker (same as eval_mmau_offline / eval_open_ended_bleu).
  2. For each test sample, generate one greedy rollout against (video, audio, question).
  3. Build a judge prompt:
       "Question: {Q}
        Reference answer: {gold answer_text}
        Candidate answer: {rollout}
        Score 0-10 on factual content match."
  4. Run the SAME loaded thinker as judge (text-only, no media re-encode).
  5. Parse integer score, normalize to [0, 1], aggregate per category.

This is the eval-time analog of training-time apply_ttrl_judge_gt: the
trained policy doubles as judge against the gold answer (vs against the
medoid of peer rollouts, which is what training did). At step 0 this
gives a base-model score; at step 200 it gives a trained-model score.
A real improvement should show up as a higher mean judge score.

Output: results_judge_eval_step{0,N}_<date>.csv with columns:
  step, judge_mean, judge_std, frac_zero, frac_full, total
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from eval_mmau_offline import (  # noqa: E402
    merge_fsdp_shards,
    load_model_with_checkpoint,
    _prepare_inputs,
    _generate_one,
)


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
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--base-model", default="/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B")
    p.add_argument("--steps", nargs="*", type=int, default=None)
    p.add_argument("--output", default="results_judge_eval.csv")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--judge-max-new-tokens", type=int, default=8)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--max-audio-duration", type=float, default=30.0)
    p.add_argument("--video-fps", type=float, default=1.0)
    p.add_argument("--video-max-frames", type=int, default=32)
    p.add_argument("--use-audio-in-video", action="store_true")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-baseline", action="store_true")
    p.add_argument("--eval-n", type=int, default=1,
                    help="Generate N rollouts per question; judge takes mean.")
    p.add_argument("--eval-temperature", type=float, default=0.6)
    p.add_argument("--eval-top-p", type=float, default=0.95)
    p.add_argument("--suffix-prompt", type=str,
                    default="\nExplain your reasoning step by step, then give a concise answer to the question in 1-3 complete sentences.",
                    help="OE-style suffix prompt.")
    p.add_argument("--gold-key", type=str, default="answer_text")
    p.add_argument("--category-key", type=str, default=None)
    return p.parse_args()


def _parse_judge_score(text: str) -> tuple[float, bool]:
    if not text:
        return 0.5, False
    m = _NUM_RE.search(text)
    if not m:
        return 0.5, False
    raw = max(0, min(10, int(m.group(1))))
    return raw / 10.0, True


def _judge_with_thinker(thinker, processor, question: str, reference: str, candidate: str,
                        max_new_tokens: int = 8) -> tuple[str, float, bool]:
    """Build judge prompt, run thinker text-only, parse score."""
    q_s = (question or "")[:4000]
    r_s = (reference or "[empty]")[:4000]
    c_s = (candidate or "[empty]")[:4000]
    prompt = JUDGE_TEMPLATE.format(question=q_s, reference=r_s, candidate=c_s)

    # Build a text-only message and tokenize via the processor (same path
    # as _prepare_inputs but without any video/audio).
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
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
            )
        except Exception:
            return "", 0.5, False

    in_len = inputs["input_ids"].shape[1]
    decoded = processor.decode(out[0, in_len:], skip_special_tokens=True)
    score, ok = _parse_judge_score(decoded)
    return decoded, score, ok


def evaluate_checkpoint(thinker, processor, test_data, args):
    cat_key = args.category_key
    if cat_key is None and test_data:
        for k in ["question_type", "source", "content_parent_category"]:
            if k in test_data[0]:
                cat_key = k
                break

    cat_scores = defaultdict(list)
    judge_scores = []
    parse_ok_count = 0
    total = len(test_data)
    desc = f"  judge_eval (N={args.eval_n})"

    for i, raw in enumerate(tqdm(test_data, desc=desc)):
        sample = dict(raw)
        gold = sample.get(args.gold_key, "") or ""
        sample["answer"] = gold
        category = (sample.get(cat_key, "unknown") if cat_key else "unknown")
        if isinstance(category, str):
            category = category.strip().title().replace("Av ", "AV ")

        question = sample.get("question", "")
        inputs, _ = _prepare_inputs(sample, processor, args, i)
        if inputs is None:
            judge_scores.append(0.0)
            cat_scores[category].append(0.0)
            continue

        # 1. Generate rollout(s)
        rollouts = []
        if args.eval_n <= 1:
            r = _generate_one(thinker, inputs, processor, args, do_sample=False, temperature=1.0)
            if r is not None:
                rollouts.append(r)
        else:
            for _ in range(args.eval_n):
                r = _generate_one(thinker, inputs, processor, args, do_sample=True,
                                  temperature=args.eval_temperature)
                if r is not None:
                    rollouts.append(r)

        if not rollouts:
            judge_scores.append(0.0)
            cat_scores[category].append(0.0)
            continue

        # 2. Judge each rollout against the gold reference
        per_rollout_scores = []
        per_rollout_raws = []
        for cand in rollouts:
            raw_judge, sc, ok = _judge_with_thinker(
                thinker, processor, question, gold, cand,
                max_new_tokens=args.judge_max_new_tokens,
            )
            per_rollout_scores.append(sc)
            per_rollout_raws.append(raw_judge)
            if ok:
                parse_ok_count += 1

        agg = float(np.mean(per_rollout_scores)) if per_rollout_scores else 0.0
        judge_scores.append(agg)
        cat_scores[category].append(agg)

        if i < 3:
            print(f"\n  === SANITY sample {i} ===")
            print(f"  id: {sample.get('id', i)}")
            print(f"  question: {question[:200]}")
            print(f"  gold:     {gold[:200]}")
            print(f"  rollout0: {rollouts[0][:200]}")
            print(f"  judge_raw0: {per_rollout_raws[0]!r}  per_score: {per_rollout_scores}")
            print(f"  agg: {agg:.3f}")
        elif (i + 1) % 10 == 0 or (i + 1) == total:
            running_mean = float(np.mean(judge_scores))
            print(f"  [{i+1}/{total}] running_mean={running_mean:.3f}", flush=True)

    arr = np.array(judge_scores)
    summary = {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "std": float(arr.std()) if len(arr) > 1 else 0.0,
        "frac_zero": float((arr == 0.0).mean()) if len(arr) else 0.0,
        "frac_full": float((arr == 1.0).mean()) if len(arr) else 0.0,
        "parse_rate": parse_ok_count / max(1, total * args.eval_n),
        "total": total,
        "cat_summary": {
            cat: {
                "mean": float(np.mean(s)) if s else 0.0,
                "n": len(s),
            }
            for cat, s in cat_scores.items()
        },
    }

    print(f"\n  {'Category':<30} {'Judge':>8} {'N':>5}")
    print(f"  {'-' * 48}")
    for cat in sorted(summary["cat_summary"].keys()):
        s = summary["cat_summary"][cat]
        print(f"  {cat:<30} {s['mean']:>8.3f} {s['n']:>5}")
    print(f"  {'-' * 48}")
    print(f"  {'TOTAL':<30} {summary['mean']:>8.3f} {summary['total']:>5}")
    print(f"  judge_parse_rate: {summary['parse_rate']:.3f}")
    return summary


def main():
    args = parse_args()

    with open(args.test_file) as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples from {args.test_file}")

    if args.n_samples and args.n_samples < len(test_data):
        import random as _rng
        _rng.seed(args.seed)
        test_data = _rng.sample(test_data, args.n_samples)
        print(f"Subsampled to {len(test_data)} samples (seed={args.seed})")

    ckpt_dir = Path(args.ckpt_dir)
    if args.steps:
        step_dirs = [(s, ckpt_dir / f"global_step_{s}") for s in sorted(args.steps)]
    else:
        step_dirs = []
        for d in sorted(ckpt_dir.iterdir()):
            if d.name.startswith("global_step_") and d.is_dir():
                step = int(d.name.split("_")[-1])
                step_dirs.append((step, d))
        step_dirs.sort(key=lambda x: x[0])

    if args.eval_baseline:
        step_dirs = [(s, d) for s, d in step_dirs if s != 0]
        step_dirs.insert(0, (0, None))

    print(f"Will evaluate {len(step_dirs)} checkpoints: {[s for s, _ in step_dirs]}")

    csv_path = Path(args.output)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "eval_n", "judge_mean", "judge_std",
                     "frac_zero", "frac_full", "parse_rate", "total", "time_s"])
    csv_file.flush()

    cat_csv = csv_path.parent / (csv_path.stem + "_by_category" + csv_path.suffix)
    cat_csv_file = open(cat_csv, "w", newline="")
    cat_writer = csv.writer(cat_csv_file)
    cat_writer.writerow(["step", "eval_n", "category", "judge_mean", "n"])
    cat_csv_file.flush()

    for step, step_dir in step_dirs:
        print(f"\n{'=' * 60}")
        print(f"Step {step}")
        print(f"{'=' * 60}")
        t0 = time.time()

        state_dict = None
        if step_dir is not None:
            actor_dir = step_dir / "actor"
            if not actor_dir.exists():
                print(f"  SKIP: {actor_dir} not found")
                continue
            print(f"  Merging FSDP shards from {actor_dir}...")
            state_dict = merge_fsdp_shards(str(actor_dir))
            print(f"  Merged {len(state_dict)} params")

        thinker, processor = load_model_with_checkpoint(args.base_model, state_dict)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        summary = evaluate_checkpoint(thinker, processor, test_data, args)
        elapsed = time.time() - t0
        print(f"\n  Step {step}: judge_mean={summary['mean']:.4f} "
              f"frac_zero={summary['frac_zero']:.3f} "
              f"frac_full={summary['frac_full']:.3f} "
              f"parse_rate={summary['parse_rate']:.3f} "
              f"({summary['total']} samples, {elapsed:.1f}s)")

        writer.writerow([
            step, args.eval_n,
            f"{summary['mean']:.4f}",
            f"{summary['std']:.4f}",
            f"{summary['frac_zero']:.4f}",
            f"{summary['frac_full']:.4f}",
            f"{summary['parse_rate']:.4f}",
            summary["total"],
            f"{elapsed:.1f}",
        ])
        csv_file.flush()

        for cat, cat_data in sorted(summary["cat_summary"].items()):
            cat_writer.writerow([
                step, args.eval_n, cat,
                f"{cat_data['mean']:.4f}",
                cat_data["n"],
            ])
        cat_csv_file.flush()

        del thinker, processor
        gc.collect()
        torch.cuda.empty_cache()

    csv_file.close()
    cat_csv_file.close()
    print(f"\nResults written to {csv_path}")
    print(f"Per-category results: {cat_csv}")


if __name__ == "__main__":
    main()
