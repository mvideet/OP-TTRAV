# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
LLM-as-judge open-ended TTRL reward.

Train-time:
    apply_ttrl_judge_gt precomputes a per-rollout score in [0, 1] (parsed
    from the policy-as-judge output) and stashes a JSON dict
    {response_text -> score} in reward_model["ground_truth"]. reward_func
    looks up the precomputed score for the rollout being graded.

Eval-time (val_kwargs.n=1):
    ground_truth is the plain reference text (e.g. answer_text). We fall
    back to BGE cosine similarity (the same metric ttrl_open_ended uses
    for eval) so eval still computes a defined-but-degraded signal.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

# Reuse open-ended primitives for eval-time fallback grading.
from verl.utils.reward_score.ttrl_open_ended import (
    extract_answer,
    grade,
)

_DEBUG = os.environ.get("TTRL_DEBUG", "0") == "1"


def _try_parse_score_map(gt):
    if not isinstance(gt, str):
        return None
    s = gt.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    for v in obj.values():
        if not isinstance(v, (int, float)):
            return None
    return obj


def _lookup_score(solution_str, score_map):
    if solution_str in score_map:
        return float(score_map[solution_str])
    stripped = solution_str.strip() if solution_str else ""
    if stripped in score_map:
        return float(score_map[stripped])
    for k, v in score_map.items():
        if k.strip() == stripped:
            return float(v)
    return 0.0


def compute_score(model_response, gt_answer):
    if isinstance(gt_answer, (list, tuple)):
        gt_answer = gt_answer[0] if gt_answer else ""
    gt_str = str(gt_answer) if gt_answer is not None else ""

    score_map = _try_parse_score_map(gt_str)
    if score_map is not None:
        score = _lookup_score(model_response or "", score_map)
        return {
            "score": float(score),
            "sim": float(score),
            "acc": float(score),
            "mode": "train_lookup",
            "extracted_gt": "[judge_score_map]",
            "pred": extract_answer(model_response or ""),
        }

    # Eval-time fallback: BGE cosine sim against the plain reference text.
    sim = grade(model_response or "", gt_str)
    return {
        "score": float(sim),
        "sim": float(sim),
        "acc": float(sim),
        "mode": "eval_bge",
        "extracted_gt": gt_str,
        "pred": extract_answer(model_response or ""),
    }


def reward_func(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    try:
        res = compute_score(solution_str, ground_truth)

        if not hasattr(reward_func, "_call_count"):
            reward_func._call_count = 0
        reward_func._call_count += 1
        if _DEBUG and (reward_func._call_count <= 5 or reward_func._call_count % 100 == 0):
            print(
                f"[JUDGE_REWARD] call={reward_func._call_count} mode={res['mode']} "
                f"score={res['score']:.3f} pred_len={len(res.get('pred') or '')}",
                file=sys.stderr,
                flush=True,
            )
        return res
    except Exception as e:
        print(f"[ERROR] Error in ttrl_judge.reward_func: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        return {
            "score": 0.0,
            "sim": 0.0,
            "acc": 0.0,
            "mode": "error",
            "extracted_gt": "",
            "pred": str(solution_str)[:200] if solution_str else "",
        }
