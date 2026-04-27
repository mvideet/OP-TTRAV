#!/usr/bin/env python3
"""
Offline BLEU / ROUGE-L / keyword-hit evaluation of TTRL-judge checkpoints
on the open-ended val split.

Mirrors verl/scripts/eval_mmau_offline.py (FSDP shard merging, multimodal
input prep, generation), but:
  - Treats responses as raw free-text (no \\boxed{} extraction).
  - Scores against `answer_text` with three metrics:
        * bleu1        : NLTK BLEU-1 with smoothing function 1
        * rouge_l_f1   : F1 of the longest common subsequence
        * keyword_hit  : binary 1/0 — every content word in the gold
                         (length >= 3, lowercased, stripped) appears in
                         the rollout
  - Default suffix prompt is the OE training-time prompt
    ("Explain your reasoning... 1-3 complete sentences.").

Usage:
  python verl/scripts/eval_open_ended_bleu.py \\
      --ckpt-dir /data/sls/scratch/.../saved/judge_0426 \\
      --test-file verl/data/OmniVideo/test_open_val20.json \\
      --base-model /data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B \\
      --steps 300 \\
      --eval-baseline \\
      --output results_judge_bleu.csv
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

# Reuse the multimodal + model loading helpers from the MCQ eval script.
sys.path.insert(0, str(Path(__file__).parent))
from eval_mmau_offline import (  # noqa: E402
    merge_fsdp_shards,
    load_model_with_checkpoint,
    _prepare_inputs,
    _generate_one,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--base-model", default="/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B")
    p.add_argument("--steps", nargs="*", type=int, default=None)
    p.add_argument("--output", default="results_open_ended_bleu.csv")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--max-audio-duration", type=float, default=30.0)
    p.add_argument("--video-fps", type=float, default=1.0)
    p.add_argument("--video-max-frames", type=int, default=32)
    p.add_argument("--use-audio-in-video", action="store_true")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-baseline", action="store_true",
                    help="Also evaluate the base model (step 0)")
    p.add_argument("--eval-n", type=int, default=1,
                    help="Sampled responses per question. >1 averages metrics across rollouts.")
    p.add_argument("--eval-temperature", type=float, default=0.6)
    p.add_argument("--eval-top-p", type=float, default=0.95)
    p.add_argument("--suffix-prompt", type=str,
                    default="\nExplain your reasoning step by step, then give a concise answer to the question in 1-3 complete sentences.",
                    help="OE-style suffix prompt (matches do_judge training).")
    p.add_argument("--gold-key", type=str, default="answer_text",
                    help="Sample key for the open-ended gold reference.")
    p.add_argument("--category-key", type=str, default=None,
                    help="JSON key for per-category breakdown. Auto-detected if not set.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Free-text scoring
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "with", "this",
    "that", "from", "have", "has", "was", "were", "they", "them", "their",
    "his", "her", "hers", "its", "into", "than", "then", "there", "what",
    "when", "where", "which", "who", "why", "how", "all", "any", "can",
    "could", "would", "should", "will", "shall", "may", "might", "must",
    "did", "does", "done", "doing", "been", "being", "very", "too",
}


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _content_words(tokens: list[str], min_len: int = 3) -> list[str]:
    return [t for t in tokens if len(t) >= min_len and t not in _STOPWORDS]


def _bleu1(reference: str, candidate: str) -> float:
    """NLTK BLEU-1 with smoothing function 1, falls back to a hand-rolled
    unigram precision + brevity penalty if NLTK is unavailable."""
    ref_toks = _tokenize(reference)
    cand_toks = _tokenize(candidate)
    if not ref_toks or not cand_toks:
        return 0.0
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        sm = SmoothingFunction().method1
        return float(sentence_bleu(
            [ref_toks], cand_toks,
            weights=(1.0, 0.0, 0.0, 0.0),
            smoothing_function=sm,
        ))
    except Exception:
        # Fallback: unigram precision with brevity penalty.
        from collections import Counter
        ref_count = Counter(ref_toks)
        match = 0
        cand_count = Counter(cand_toks)
        for w, c in cand_count.items():
            match += min(c, ref_count.get(w, 0))
        precision = match / len(cand_toks) if cand_toks else 0.0
        bp = 1.0 if len(cand_toks) >= len(ref_toks) else \
            float(np.exp(1 - len(ref_toks) / max(1, len(cand_toks))))
        return precision * bp


def _rouge_l_f1(reference: str, candidate: str) -> float:
    """ROUGE-L F1: longest common subsequence-based F1."""
    ref_toks = _tokenize(reference)
    cand_toks = _tokenize(candidate)
    if not ref_toks or not cand_toks:
        return 0.0
    n, m = len(ref_toks), len(cand_toks)
    # Memory-light LCS DP (rolling arrays)
    prev = [0] * (m + 1)
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_toks[i - 1] == cand_toks[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev, cur = cur, prev
        for k in range(m + 1):
            cur[k] = 0
    lcs = prev[m]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)


def _keyword_hit(reference: str, candidate: str) -> float:
    """Binary 1.0 iff every content word from reference appears in candidate."""
    ref_words = set(_content_words(_tokenize(reference)))
    if not ref_words:
        return 0.0
    cand_set = set(_tokenize(candidate))
    return 1.0 if ref_words.issubset(cand_set) else 0.0


def score_open_ended(response: str, gt_text: str) -> dict:
    return {
        "bleu1": _bleu1(gt_text, response or ""),
        "rouge_l": _rouge_l_f1(gt_text, response or ""),
        "keyword_hit": _keyword_hit(gt_text, response or ""),
        "pred": (response or "")[:300],
        "gt": (gt_text or "")[:300],
    }


def _aggregate_eval_n(per_rollout: list[dict]) -> dict:
    """Average bleu1/rouge_l/keyword_hit across N rollouts; pred is the longest one."""
    if not per_rollout:
        return {"bleu1": 0.0, "rouge_l": 0.0, "keyword_hit": 0.0, "pred": "", "n_responses": 0}
    bleu1 = float(np.mean([r["bleu1"] for r in per_rollout]))
    rouge = float(np.mean([r["rouge_l"] for r in per_rollout]))
    kw = float(np.mean([r["keyword_hit"] for r in per_rollout]))
    longest_pred = max(per_rollout, key=lambda r: len(r.get("pred") or ""))["pred"]
    return {
        "bleu1": bleu1,
        "rouge_l": rouge,
        "keyword_hit": kw,
        "pred": longest_pred,
        "n_responses": len(per_rollout),
    }


# ---------------------------------------------------------------------------
# Sample wrapper: the OE val split uses `answer_text`, not `answer`.
# ---------------------------------------------------------------------------

def _wrap_sample_for_oe(sample, gold_key):
    """Return a copy of `sample` with `answer` set to the open-ended gold text.

    `_prepare_inputs` from eval_mmau_offline pulls `sample["answer"]` for its
    return value (we discard it here) and `sample["question"]` for the prompt.
    No format extraction happens; we generate raw text and score it ourselves.
    """
    s = dict(sample)
    gold = s.get(gold_key, "")
    if not gold and "answer" in s and not s.get("answer", "").strip().upper() in {"A", "B", "C", "D"}:
        gold = s["answer"]
    s["answer"] = gold or ""
    s["_oe_gold"] = gold or ""
    return s


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(thinker, processor, test_data, args):
    cat_key = args.category_key
    if cat_key is None and test_data:
        for k in ["question_type", "source", "content_parent_category"]:
            if k in test_data[0]:
                cat_key = k
                break

    cat_metrics = defaultdict(lambda: {"bleu1": [], "rouge_l": [], "keyword_hit": []})
    total = len(test_data)
    bleu1_all, rouge_all, kw_all = [], [], []
    results = []

    eval_n = max(1, getattr(args, "eval_n", 1))
    eval_temp = getattr(args, "eval_temperature", 0.6)
    desc = f"  eval (N={eval_n})" if eval_n > 1 else "  eval"

    for i, raw_sample in enumerate(tqdm(test_data, desc=desc)):
        sample = _wrap_sample_for_oe(raw_sample, args.gold_key)
        gt_text = sample["_oe_gold"]
        category = (sample.get(cat_key, "unknown") if cat_key else "unknown")
        category = category.strip().title().replace("Av ", "AV ") if isinstance(category, str) else "unknown"

        inputs, _ = _prepare_inputs(sample, processor, args, i)
        if inputs is None:
            results.append({"id": sample.get("id", i), "category": category, "bleu1": 0.0,
                            "rouge_l": 0.0, "keyword_hit": 0.0, "pred": "ERROR", "gt": gt_text})
            cat_metrics[category]["bleu1"].append(0.0)
            cat_metrics[category]["rouge_l"].append(0.0)
            cat_metrics[category]["keyword_hit"].append(0.0)
            bleu1_all.append(0.0)
            rouge_all.append(0.0)
            kw_all.append(0.0)
            continue

        per_rollout = []
        if eval_n <= 1:
            response = _generate_one(thinker, inputs, processor, args,
                                     do_sample=False, temperature=1.0)
            if response is not None:
                per_rollout.append(score_open_ended(response, gt_text))
        else:
            for _ in range(eval_n):
                resp = _generate_one(thinker, inputs, processor, args,
                                     do_sample=True, temperature=eval_temp)
                if resp is not None:
                    per_rollout.append(score_open_ended(resp, gt_text))

        agg = _aggregate_eval_n(per_rollout)
        cat_metrics[category]["bleu1"].append(agg["bleu1"])
        cat_metrics[category]["rouge_l"].append(agg["rouge_l"])
        cat_metrics[category]["keyword_hit"].append(agg["keyword_hit"])
        bleu1_all.append(agg["bleu1"])
        rouge_all.append(agg["rouge_l"])
        kw_all.append(agg["keyword_hit"])

        results.append({
            "id": sample.get("id", i),
            "category": category,
            "bleu1": agg["bleu1"],
            "rouge_l": agg["rouge_l"],
            "keyword_hit": agg["keyword_hit"],
            "pred": agg["pred"][:200],
            "gt": gt_text[:200],
        })

        if i < 3:
            print(f"\n  === SANITY sample {i} ===")
            print(f"  id: {sample.get('id', i)}")
            print(f"  question: {sample.get('question', '')[:200]}")
            print(f"  gold:     {gt_text[:200]}")
            print(f"  pred[0]:  {agg['pred'][:200]}")
            print(f"  bleu1={agg['bleu1']:.3f} rouge_l={agg['rouge_l']:.3f} keyword_hit={agg['keyword_hit']:.2f}")
        elif (i + 1) % 10 == 0 or (i + 1) == total:
            running_b = float(np.mean(bleu1_all))
            running_r = float(np.mean(rouge_all))
            running_k = float(np.mean(kw_all))
            print(f"  [{i+1}/{total}] bleu1={running_b:.3f} rouge_l={running_r:.3f} kw_hit={running_k:.3f}",
                  flush=True)

    summary = {
        "bleu1_mean": float(np.mean(bleu1_all)) if bleu1_all else 0.0,
        "rouge_l_mean": float(np.mean(rouge_all)) if rouge_all else 0.0,
        "keyword_hit_mean": float(np.mean(kw_all)) if kw_all else 0.0,
        "total": total,
        "results": results,
        "cat_summary": {
            cat: {
                "bleu1": float(np.mean(d["bleu1"])) if d["bleu1"] else 0.0,
                "rouge_l": float(np.mean(d["rouge_l"])) if d["rouge_l"] else 0.0,
                "keyword_hit": float(np.mean(d["keyword_hit"])) if d["keyword_hit"] else 0.0,
                "n": len(d["bleu1"]),
            }
            for cat, d in cat_metrics.items()
        },
    }

    print(f"\n  {'Category':<30} {'BLEU-1':>8} {'ROUGE-L':>8} {'KW-hit':>8} {'N':>5}")
    print(f"  {'-' * 64}")
    for cat in sorted(summary["cat_summary"].keys()):
        s = summary["cat_summary"][cat]
        print(f"  {cat:<30} {s['bleu1']:>8.3f} {s['rouge_l']:>8.3f} {s['keyword_hit']:>8.3f} {s['n']:>5}")
    print(f"  {'-' * 64}")
    print(f"  {'TOTAL':<30} {summary['bleu1_mean']:>8.3f} {summary['rouge_l_mean']:>8.3f} {summary['keyword_hit_mean']:>8.3f} {summary['total']:>5}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    eval_n = args.eval_n
    print(f"Will evaluate {len(step_dirs)} checkpoints: {[s for s, _ in step_dirs]}")
    if eval_n > 1:
        print(f"Using mean@{eval_n} sampled (T={args.eval_temperature}, top_p={args.eval_top_p})")
    else:
        print("Using greedy decoding (single rollout per sample)")

    csv_path = Path(args.output)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "eval_n", "bleu1", "rouge_l", "keyword_hit", "total", "time_s"])
    csv_file.flush()

    cat_csv_path = csv_path.parent / (csv_path.stem + "_by_category" + csv_path.suffix)
    cat_csv_file = open(cat_csv_path, "w", newline="")
    cat_writer = csv.writer(cat_csv_file)
    cat_writer.writerow(["step", "eval_n", "category", "bleu1", "rouge_l", "keyword_hit", "n"])
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
        print(f"\n  Step {step}: bleu1={summary['bleu1_mean']:.4f} "
              f"rouge_l={summary['rouge_l_mean']:.4f} "
              f"keyword_hit={summary['keyword_hit_mean']:.4f} "
              f"({summary['total']} samples, {elapsed:.1f}s)")

        writer.writerow([
            step, eval_n,
            f"{summary['bleu1_mean']:.4f}",
            f"{summary['rouge_l_mean']:.4f}",
            f"{summary['keyword_hit_mean']:.4f}",
            summary["total"],
            f"{elapsed:.1f}",
        ])
        csv_file.flush()

        for cat, cat_data in sorted(summary["cat_summary"].items()):
            cat_writer.writerow([
                step, eval_n, cat,
                f"{cat_data['bleu1']:.4f}",
                f"{cat_data['rouge_l']:.4f}",
                f"{cat_data['keyword_hit']:.4f}",
                cat_data["n"],
            ])
        cat_csv_file.flush()

        del thinker, processor
        gc.collect()
        torch.cuda.empty_cache()

    csv_file.close()
    cat_csv_file.close()
    print(f"\nResults written to {csv_path}")
    print(f"Per-category results: {cat_csv_path}")


if __name__ == "__main__":
    main()
