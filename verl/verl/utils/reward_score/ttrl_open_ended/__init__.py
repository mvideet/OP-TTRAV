# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Open-ended TTRL reward module.

Replaces the MCQ-style extract_answer/grade pair with a semantic-similarity
scorer backed by a frozen sentence encoder (BGE-small by default).

Public functions match the existing TTRL interface so this module can be
slotted into ttrl_utils.py via TTRL_TASK_TYPE=open_ended_video and into
the trainer via custom_reward_function.path.
"""

from __future__ import annotations

import os
import re
import sys
import traceback

from .embedding import cosine, encode_cached, get_cached, put_cached

_DEBUG = os.environ.get("TTRL_OE_DEBUG", "0") == "1"

# Tunables (env-overridable so the user can ablate without code edits).
#
# BGE-small produces a high baseline similarity (~0.55-0.65) even for unrelated
# English sentences. The natural [-1, 1] cosine range is squashed into roughly
# [0.4, 1.0] in practice. We clamp to [0, 1] without rescaling so GRPO sees
# the raw discriminative signal; group-relative advantage normalization in
# GRPO handles the compressed scale.
#
# Length floor: a simple multiplicative penalty for trivially short replies
# (e.g. "yes."). Disabled by default for v1; if we observe collapse to short
# answers we can re-enable via TTRL_OE_MIN_LEN_TOKENS > 0.
_MIN_LEN_TOKENS = int(os.environ.get("TTRL_OE_MIN_LEN_TOKENS", "0"))
_MIN_LEN_PENALTY = float(os.environ.get("TTRL_OE_MIN_LEN_PENALTY", "0.7"))


def extract_answer(response: str) -> str:
    """
    Identity extractor for open-ended responses.

    Strips a few common artifacts (Chinese fences, leading/trailing whitespace,
    leftover \\boxed{} wrappers if the model still emits them) but otherwise
    returns the full response. Returns "" for empty/None.
    """
    if not response:
        return ""
    text = response.strip()
    # If the model still emits \boxed{X}, lift the contents (handles graceful
    # mode-switching from MCQ-style training to open-ended).
    boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        # Replace the boxed wrapper with its contents but keep the surrounding
        # reasoning so similarity is computed against the full answer.
        text = text.replace(boxed.group(0), boxed.group(1))
    return text.strip()


def grade(model_answer: str, gt_answer: str) -> float:
    """
    Continuous semantic similarity in [0, 1] between an open-ended model answer
    and a reference text (the medoid rollout chosen by majority voting).

    Returns 0.0 for empty inputs. Uses cached embeddings when available
    (the vote step pre-populates the cache, so this is usually free).
    """
    if not model_answer or not gt_answer:
        return 0.0

    pred_text = extract_answer(model_answer)
    ref_text = extract_answer(gt_answer) if isinstance(gt_answer, str) else str(gt_answer)
    if not pred_text or not ref_text:
        return 0.0

    # Cache-first path: vote step pre-encoded everything.
    pe = get_cached(pred_text)
    re_ = get_cached(ref_text)
    if pe is None or re_ is None:
        # Fall back to lazy encoding (e.g., during eval where vote isn't run).
        embs = encode_cached([pred_text, ref_text])
        pe, re_ = embs[0], embs[1]
        put_cached(pred_text, pe)
        put_cached(ref_text, re_)

    sim = cosine(pe, re_)
    # Clamp negatives to 0 (rare with BGE) but do NOT rescale (sim+1)/2; that
    # mapping inflates BGE's natural ~0.6 baseline up to 0.8 and destroys the
    # discriminative signal in the high-similarity range that we care about.
    sim01 = max(0.0, min(1.0, sim))
    return sim01


def _length_penalty(text: str) -> float:
    """Multiplicative penalty for trivially short responses. Disabled when _MIN_LEN_TOKENS=0."""
    if _MIN_LEN_TOKENS <= 0:
        return 1.0
    n_words = len(text.split())
    if n_words >= _MIN_LEN_TOKENS:
        return 1.0
    return _MIN_LEN_PENALTY


def compute_score(model_response: str, gt_answer: str) -> dict:
    """
    Per-rollout score for open-ended TTRL.

    Args:
        model_response: full text of one rollout.
        gt_answer: reference text. During TTRL this is the medoid rollout
            chosen by apply_ttrl_open_ended_gt; during eval this is a
            reference answer text from the dataset.

    Returns:
        dict with score, sim, length_penalty, acc, extracted_gt, pred. The
        `acc` field mirrors `score` so existing logging dashboards continue
        to plot something meaningful.
    """
    pred = extract_answer(model_response)
    sim = grade(pred, gt_answer)
    lp = _length_penalty(pred)
    score = sim * lp

    if _DEBUG:
        print(
            f"[OE_GRADE] sim={sim:.3f} lp={lp:.2f} score={score:.3f} "
            f"pred='{pred[:80]}' ref='{(gt_answer or '')[:80]}'",
            file=sys.stderr,
            flush=True,
        )

    return {
        "score": float(score),
        "sim": float(sim),
        "length_penalty": float(lp),
        "acc": float(sim),
        "extracted_gt": gt_answer if isinstance(gt_answer, str) else str(gt_answer),
        "pred": pred,
    }


def reward_func(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    """
    Open-ended TTRL reward function.

    Interface mirrors verl.utils.reward_score.ttrl_video_qa.reward_func so
    this module can be selected via custom_reward_function.path without
    other code changes.

    During TTRL training, `ground_truth` will be the medoid rollout text
    set by apply_ttrl_open_ended_gt. During eval, it should be a reference
    answer string from the dataset (e.g. answer_text field).
    """
    try:
        if isinstance(ground_truth, (list, tuple)):
            ground_truth = ground_truth[0] if ground_truth else ""
        gt_str = str(ground_truth) if ground_truth is not None else ""
        res = compute_score(solution_str, gt_str)

        # Periodic sanity logging (every 100th call)
        if not hasattr(reward_func, "_call_count"):
            reward_func._call_count = 0
        reward_func._call_count += 1
        if reward_func._call_count <= 5 or reward_func._call_count % 100 == 0:
            pred_snippet = (res.get("pred", "")[:80] + "...") if len(res.get("pred", "")) > 80 else res.get("pred", "")
            gt_snippet = (gt_str[:80] + "...") if len(gt_str) > 80 else gt_str
            print(
                f"[OE_REWARD] call={reward_func._call_count} "
                f"score={res['score']:.3f} sim={res['sim']:.3f} "
                f"pred='{pred_snippet}' gt='{gt_snippet}'",
                file=sys.stderr,
                flush=True,
            )

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in ttrl_open_ended.reward_func: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        # Don't crash training on a single bad sample.
        return {
            "score": 0.0,
            "sim": 0.0,
            "length_penalty": 0.0,
            "acc": 0.0,
            "extracted_gt": str(ground_truth) if ground_truth is not None else "",
            "pred": str(solution_str)[:200] if solution_str else "",
        }
