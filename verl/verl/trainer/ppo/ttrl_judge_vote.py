# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
LLM-as-judge open-ended TTRL voting.

For each prompt:
  1. Decode all N rollouts.
  2. Pick a pseudo-GT via BGE-medoid voting (same as apply_ttrl_open_ended_gt).
  3. For each rollout j, build a templated judge prompt asking the SAME policy
     to score the rollout's factual alignment with the medoid on a 0-10 scale.
  4. Run the policy as a judge in a single batched generate_sequences call
     (text-only path: no multi_modal_inputs in the judge DataProto, so the
     rollout layer skips video/audio re-encoding).
  5. Parse the integer score from each judge output, normalize to [0, 1],
     stash {rollout_text -> score} as JSON in reward_model["ground_truth"]
     for the per-rollout reward_func to look up.

Why same-policy: keeps the experiment "test-time" -- no extra frozen model
weights live in GPU memory; the policy doubles as judge using the existing
FSDP summon-then-generate path. Cost: one extra ~8-token forward pass per
rollout per training step (~64 prompts at batch=4 / N=16; micro_batched).
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import List, Tuple

import numpy as np
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.ttrl_open_ended_vote import _batch_semantic_vote

_DEBUG = os.environ.get("TTRL_DEBUG", "0") == "1"
_JUDGE_DEBUG = os.environ.get("TTRL_JUDGE_DEBUG", "0") == "1"

_JUDGE_MAX_NEW_TOKENS = int(os.environ.get("TTRL_JUDGE_MAX_NEW_TOKENS", "8"))
_JUDGE_MAX_PROMPT_LEN = int(os.environ.get("TTRL_JUDGE_MAX_PROMPT_LEN", "4096"))
_JUDGE_NEUTRAL_FALLBACK = float(os.environ.get("TTRL_JUDGE_NEUTRAL_FALLBACK", "0.5"))

_JUDGE_TEMPLATE = (
    "You are evaluating answers to a video understanding question.\n\n"
    "Question: {question}\n\n"
    "Reference answer (from majority vote): {reference}\n\n"
    "Candidate answer: {candidate}\n\n"
    "How well does the candidate match the reference in factual content "
    "(ignore phrasing, focus on whether the same facts are stated)?\n\n"
    "Output a single integer from 0 (completely different) to 10 (identical "
    "in factual content). No explanation. Just the number.\n\n"
    "Score:"
)


_NUM_RE = re.compile(r"\b(\d{1,2})\b")


def _parse_judge_score(text: str) -> Tuple[float, bool]:
    """Return (score in [0, 1], parsed_ok). Falls back to neutral if no number."""
    if not text:
        return _JUDGE_NEUTRAL_FALLBACK, False
    m = _NUM_RE.search(text)
    if not m:
        return _JUDGE_NEUTRAL_FALLBACK, False
    raw = int(m.group(1))
    raw = max(0, min(10, raw))
    return raw / 10.0, True


def _build_judge_prompts(
    questions: List[str],
    references: List[str],
    candidates: List[str],
    max_chars: int = 4000,
) -> List[str]:
    """Build per-rollout judge prompt strings. Inputs aligned 1:1."""
    assert len(questions) == len(references) == len(candidates), (
        f"length mismatch: q={len(questions)} r={len(references)} c={len(candidates)}"
    )
    prompts = []
    for q, r, c in zip(questions, references, candidates):
        # Cap individual fields so the judge prompt stays in a reasonable length budget.
        q_s = (q or "")[:max_chars]
        r_s = (r or "[empty]")[:max_chars]
        c_s = (c or "[empty]")[:max_chars]
        prompts.append(_JUDGE_TEMPLATE.format(question=q_s, reference=r_s, candidate=c_s))
    return prompts


def _build_judge_dataproto(prompts: List[str], tokenizer) -> DataProto:
    """Tokenize judge prompts (left-padded, text-only) into a DataProto for generate_sequences."""
    saved_pad = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_JUDGE_MAX_PROMPT_LEN,
        )
    finally:
        if saved_pad is not None:
            tokenizer.padding_side = saved_pad

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    # Position ids: standard text-only causal positions, zero for left-pad slots.
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    dp = DataProto.from_dict(
        tensors={
            "input_ids": input_ids.to(torch.long),
            "attention_mask": attention_mask.to(torch.long),
            "position_ids": position_ids.to(torch.long),
        },
        non_tensors={},
        meta_info={
            "do_sample": False,
            "temperature": 0.0,
            "response_length": _JUDGE_MAX_NEW_TOKENS,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        },
    )
    return dp


def _decode_judge_outputs(judge_dp: DataProto, tokenizer) -> List[str]:
    """Extract decoded judge response strings from the generate_sequences output."""
    decoded: List[str] = []
    n = len(judge_dp)
    for k in range(n):
        item = judge_dp[k]
        prompts_ids = item.batch["prompts"]
        prompt_length = prompts_ids.shape[-1]
        responses_ids = item.batch["responses"]
        # response-side attention mask is the suffix of the full attention mask.
        valid_response_length = item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = responses_ids[:valid_response_length]
        text = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        decoded.append(text)
    return decoded


def apply_ttrl_judge_gt(batch, gen_batch_output, n, tokenizer, actor_rollout_wg):
    """
    LLM-as-judge analog of apply_ttrl_open_ended_gt.

    Signature differs from apply_ttrl_open_ended_gt: requires actor_rollout_wg
    (the policy's worker-group handle) so we can run the same policy as judge
    via generate_sequences. ttrl_utils.apply_ttrl_gt plumbs this through.
    """
    assert actor_rollout_wg is not None, (
        "apply_ttrl_judge_gt requires actor_rollout_wg; ray_trainer must pass it through "
        "apply_ttrl_gt(batch, gen_batch_output, n, tokenizer, actor_rollout_wg)"
    )
    assert len(gen_batch_output) % n == 0, (
        f"gen_batch_output length {len(gen_batch_output)} not divisible by n={n}"
    )
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, (
        f"batch length {len(batch)} != num_prompts {num_prompts}"
    )

    # ----- 1. Decode all rollouts (same path as open_ended_vote) -----
    model_outputs: List[str] = []
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    # ----- 2. BGE-medoid voting picks pseudo-GT per prompt -----
    medoid_list, sim_list, vote_stats_list = _batch_semantic_vote(model_outputs, n)

    # ----- 3. Build per-rollout judge prompts -----
    questions: List[str] = []
    references: List[str] = []
    candidates: List[str] = []
    for i in range(num_prompts):
        nb = batch[i].non_tensor_batch
        q_text = str(nb.get("question", "") or "")
        ref_text = medoid_list[i] or ""
        for j in range(n):
            cand_text = model_outputs[i * n + j]
            questions.append(q_text)
            references.append(ref_text)
            candidates.append(cand_text)

    judge_prompts = _build_judge_prompts(questions, references, candidates)

    # ----- 4. Run policy-as-judge via existing rollout path -----
    judge_input_dp = _build_judge_dataproto(judge_prompts, tokenizer)
    judge_output_dp = actor_rollout_wg.generate_sequences(judge_input_dp)
    raw_judge_outputs = _decode_judge_outputs(judge_output_dp, tokenizer)

    # ----- 5. Parse scores -----
    scores: List[float] = []
    n_parsed_ok = 0
    for txt in raw_judge_outputs:
        s, ok = _parse_judge_score(txt)
        scores.append(s)
        if ok:
            n_parsed_ok += 1
    assert len(scores) == num_prompts * n, (
        f"score count {len(scores)} != expected {num_prompts * n}"
    )

    # ----- 6. Stash per-prompt {text -> score} JSON maps + bookkeeping -----
    if not hasattr(apply_ttrl_judge_gt, "_call_count"):
        apply_ttrl_judge_gt._call_count = 0
    apply_ttrl_judge_gt._call_count += 1
    step = apply_ttrl_judge_gt._call_count
    verbose = step <= 3 or step % 10 == 0

    scores_arr = np.array(scores, dtype=float)
    print(
        f"\n[JUDGE_TTRL] step={step} apply_ttrl_judge_gt: "
        f"{num_prompts} prompts, {n} rollouts each | "
        f"judge parse_ok={n_parsed_ok}/{len(scores)} "
        f"({100.0 * n_parsed_ok / max(1, len(scores)):.1f}%)",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[JUDGE_HEALTH] step={step} | scores: mean={scores_arr.mean():.3f} "
        f"std={scores_arr.std():.3f} min={scores_arr.min():.3f} max={scores_arr.max():.3f} | "
        f"frac_zero={(scores_arr == 0.0).mean():.2f} frac_full={(scores_arr == 1.0).mean():.2f} | "
        f"unparseable_frac={1.0 - n_parsed_ok / max(1, len(scores)):.2f}",
        file=sys.stderr,
        flush=True,
    )

    per_prompt_score_means: List[float] = []
    n_zero_var_groups = 0
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"].get("ground_truth", "")
        group_outputs = model_outputs[i * n : (i + 1) * n]
        group_scores = scores[i * n : (i + 1) * n]
        group_arr = np.array(group_scores)
        per_prompt_score_means.append(float(group_arr.mean()))
        if group_arr.std() < 1e-6:
            n_zero_var_groups += 1

        # Build text -> score map. Identical rollouts map to the same score, fine.
        score_map = {}
        for text, sc in zip(group_outputs, group_scores):
            score_map[text.strip()] = float(sc)
        gt_json = json.dumps(score_map, ensure_ascii=False)

        data_item.non_tensor_batch["reward_model"]["ground_truth"] = gt_json
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = medoid_list[i].strip()
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

        nb = data_item.non_tensor_batch
        sample_id = nb.get("id", nb.get("index", i))
        st = vote_stats_list[i]
        print(
            f"[JUDGE_TTRL INPUT] step={step} prompt {i}/{num_prompts} | id={sample_id}"
            f" | mean_pair={st.get('mean_pairwise_sim', 0.0):.3f} "
            f"unique={st.get('unique_responses', 0)}/{n}"
            f" | judge: mean={group_arr.mean():.3f} std={group_arr.std():.3f}"
            f" min={group_arr.min():.3f} max={group_arr.max():.3f}"
            f"{' ZERO_VAR!' if group_arr.std() < 1e-6 else ''}",
            file=sys.stderr,
            flush=True,
        )

        if verbose or _JUDGE_DEBUG:
            print(
                f"  question: {str(nb.get('question', 'N/A'))[:200]}\n"
                f"  original_gt: {str(original_gt)[:120]}\n"
                f"  medoid (ref): {medoid_list[i][:200]}",
                file=sys.stderr,
                flush=True,
            )
            for j in range(min(n, 4 if not _JUDGE_DEBUG else n)):
                snippet = (group_outputs[j].strip()[:100] + "...") \
                    if len(group_outputs[j].strip()) > 100 else group_outputs[j].strip()
                judge_raw = raw_judge_outputs[i * n + j].strip()[:24]
                print(
                    f"    r{j}: score={group_scores[j]:.2f} judge_raw={judge_raw!r} | {snippet}",
                    file=sys.stderr,
                    flush=True,
                )

    print(
        f"[JUDGE_REWARD_SUMMARY] step={step} | "
        f"prompt_score_mean={float(np.mean(per_prompt_score_means)):.4f} | "
        f"zero_var_groups={n_zero_var_groups}/{num_prompts}"
        f"{' (NO GRADIENT - all groups collapsed)' if n_zero_var_groups == num_prompts else ''}",
        file=sys.stderr,
        flush=True,
    )
    if n_zero_var_groups > num_prompts * 0.5:
        print(
            f"[JUDGE_HEALTH] *** WARNING: {n_zero_var_groups}/{num_prompts} groups have "
            f"zero judge-score variance — judge may be collapsing to constant output ***",
            file=sys.stderr,
            flush=True,
        )

    # majority_ratio_list: drop in mean per-prompt judge score for the existing
    # wandb dashboard (it expects a per-prompt confidence-ish quantity).
    batch.non_tensor_batch["majority_ratio_list"] = np.array(per_prompt_score_means, dtype=float)
    batch.non_tensor_batch["vote_stats_list"] = np.array(vote_stats_list, dtype=object)

    return batch
