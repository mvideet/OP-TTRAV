# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Open-ended TTRL voting.

For each prompt we:
  1. Encode all N rollouts with a frozen sentence encoder.
  2. Build the N x N cosine similarity matrix.
  3. Pick the medoid (rollout with the highest mean similarity to its peers).
  4. Use the medoid response text as the pseudo ground truth for that prompt.

The reward function then grades each rollout against the medoid via cosine
similarity (continuous in [0, 1]).

We also compute several collapse diagnostics:
  * mean_pairwise_sim          : average off-diagonal similarity within the group
  * max_pairwise_sim           : max off-diagonal similarity (1.0 = duplicates)
  * sim_std                    : std of the off-diagonal similarities
  * medoid_mean_sim            : mean similarity of the medoid to the rest
  * frac_high_sim              : fraction of pairs with sim > 0.95 (collapse signal)
  * unique_responses           : number of distinct response texts in the group
  * unparseable                : number of empty / blank responses

These are returned per-prompt as a dict and surfaced via existing
ttrl_metrics aggregation paths.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np

from verl.utils.reward_score.ttrl_open_ended.embedding import (
    cosine_matrix,
    encode_cached,
    put_cached,
)

_DEBUG = os.environ.get("TTRL_DEBUG", "0") == "1"
_OE_DEBUG = os.environ.get("TTRL_OE_DEBUG", "0") == "1"


def _semantic_majority_vote(
    model_outputs: List[str],
) -> Tuple[str, float, dict]:
    """
    Semantic analog of ttrl_utils._majority_vote.

    Returns:
        (medoid_text, medoid_mean_sim, vote_stats_dict)
    """
    n = len(model_outputs)
    assert n > 0, "empty rollout group"

    # Treat None and pure-whitespace as unparseable but keep them in the
    # group with a near-zero embedding so the matrix dimensions stay aligned.
    cleaned = [(o or "").strip() for o in model_outputs]
    n_unparseable = sum(1 for c in cleaned if not c)

    if n_unparseable == n:
        # All rollouts empty - emit a zero-similarity stat block; downstream
        # advantage normalization will produce zero gradients which is the
        # right behavior here.
        stats = {
            "n_total": n,
            "n_unparseable": n,
            "unique_responses": 0,
            "mean_pairwise_sim": 0.0,
            "max_pairwise_sim": 0.0,
            "sim_std": 0.0,
            "medoid_mean_sim": 0.0,
            "frac_high_sim": 0.0,
            "unanimous": False,
        }
        return "", 0.0, stats

    # Encode in one batched call (cache-aware).
    # Empty strings get a placeholder so they don't influence the medoid.
    enc_inputs = [c if c else " " for c in cleaned]
    E = encode_cached(enc_inputs)  # [n, d], L2-normalized

    # Pre-populate the reward function's cache with the (response, embedding)
    # pairs we just computed so the per-rollout grade() calls are free lookups.
    for text, vec in zip(cleaned, E):
        if text:
            put_cached(text, vec)

    # NxN similarity, then mask diagonal for "mean to others".
    S = cosine_matrix(E)  # [n, n]
    mask = ~np.eye(n, dtype=bool)
    off_diag = S[mask]
    mean_sims = S.sum(axis=1) - np.diag(S)  # exclude self
    if n > 1:
        mean_sims = mean_sims / (n - 1)
    else:
        mean_sims = mean_sims  # n=1 trivial case

    # Penalize empty responses so they can never be chosen as medoid.
    for i, c in enumerate(cleaned):
        if not c:
            mean_sims[i] = -1.0

    medoid_idx = int(np.argmax(mean_sims))
    medoid_text = cleaned[medoid_idx]
    medoid_mean_sim = float(mean_sims[medoid_idx])

    # Diagnostics.
    if off_diag.size > 0:
        mean_pair = float(off_diag.mean())
        max_pair = float(off_diag.max())
        sim_std = float(off_diag.std())
        frac_high = float((off_diag > 0.95).mean())
    else:
        mean_pair = max_pair = sim_std = frac_high = 0.0
    unique_responses = len(set(cleaned)) - (1 if "" in set(cleaned) else 0)
    unanimous = unique_responses <= 1 and n_unparseable == 0

    stats = {
        "n_total": n,
        "n_unparseable": n_unparseable,
        "unique_responses": unique_responses,
        "mean_pairwise_sim": mean_pair,
        "max_pairwise_sim": max_pair,
        "sim_std": sim_std,
        "medoid_mean_sim": medoid_mean_sim,
        "frac_high_sim": frac_high,
        "unanimous": bool(unanimous),
    }

    if _DEBUG or _OE_DEBUG:
        print(
            f"[OE_VOTE] n={n} unique={unique_responses} "
            f"mean_pair={mean_pair:.3f} max_pair={max_pair:.3f} "
            f"sim_std={sim_std:.3f} frac_high={frac_high:.2f} "
            f"medoid_mean_sim={medoid_mean_sim:.3f} medoid_idx={medoid_idx}",
            file=sys.stderr,
            flush=True,
        )
        if _OE_DEBUG:
            for i, c in enumerate(cleaned):
                marker = " <- MEDOID" if i == medoid_idx else ""
                snippet = (c[:140] + "...") if len(c) > 140 else c
                print(
                    f"  [OE_VOTE]   r{i} sim_to_others={mean_sims[i]:+.3f}{marker} | {snippet}",
                    file=sys.stderr,
                )

    return medoid_text, medoid_mean_sim, stats


def _batch_semantic_vote(
    model_outputs: List[str], n: int
) -> Tuple[List[str], List[float], List[dict]]:
    """
    Apply _semantic_majority_vote per prompt.

    Args:
        model_outputs: flat list of length num_prompts * n.
        n: number of rollouts per prompt.

    Returns:
        (medoid_text_list, medoid_mean_sim_list, stats_list) of length num_prompts.
    """
    assert len(model_outputs) % n == 0, (
        f"model_outputs length {len(model_outputs)} not divisible by n={n}"
    )
    num_prompts = len(model_outputs) // n

    medoid_list: List[str] = []
    sim_list: List[float] = []
    stats_list: List[dict] = []
    for i in range(num_prompts):
        group = model_outputs[i * n : (i + 1) * n]
        medoid, mean_sim, stats = _semantic_majority_vote(group)
        medoid_list.append(medoid)
        sim_list.append(mean_sim)
        stats_list.append(stats)

    return medoid_list, sim_list, stats_list


def apply_ttrl_open_ended_gt(batch, gen_batch_output, n, tokenizer):
    """
    Open-ended analog of ttrl_utils.apply_ttrl_gt.

    Decodes all rollouts, runs semantic majority voting, replaces each
    prompt's ground_truth with the medoid response text, and stores
    diagnostics on the batch for later metric aggregation.

    Signature matches apply_ttrl_gt so the trainer can dispatch on
    TTRL_TASK_TYPE without changing call sites.
    """
    assert len(gen_batch_output) % n == 0, (
        f"gen_batch_output length {len(gen_batch_output)} not divisible by n={n}"
    )
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, (
        f"batch length {len(batch)} != num_prompts {num_prompts}"
    )

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

    medoid_list, sim_list, stats_list = _batch_semantic_vote(model_outputs, n)

    # Track call count for periodic verbose logging
    if not hasattr(apply_ttrl_open_ended_gt, "_call_count"):
        apply_ttrl_open_ended_gt._call_count = 0
    apply_ttrl_open_ended_gt._call_count += 1
    step = apply_ttrl_open_ended_gt._call_count
    verbose = step <= 3 or step % 10 == 0  # detailed logs for first 3 steps + every 10th

    print(
        f"\n[OE_TTRL] step={step} apply_ttrl_open_ended_gt: "
        f"{num_prompts} prompts, {n} rollouts each, "
        f"{len(model_outputs)} total outputs",
        file=sys.stderr,
        flush=True,
    )

    # Batch-level collapse summary
    all_mean_pair = [s["mean_pairwise_sim"] for s in stats_list]
    all_max_pair = [s["max_pairwise_sim"] for s in stats_list]
    all_unique = [s["unique_responses"] for s in stats_list]
    all_frac_high = [s["frac_high_sim"] for s in stats_list]
    n_collapse_risk = sum(1 for m in all_mean_pair if m > 0.92)
    print(
        f"[OE_HEALTH] step={step} | "
        f"mean_pair: avg={np.mean(all_mean_pair):.3f} min={np.min(all_mean_pair):.3f} max={np.max(all_mean_pair):.3f} | "
        f"max_pair: avg={np.mean(all_max_pair):.3f} | "
        f"unique_responses: avg={np.mean(all_unique):.1f}/{n} | "
        f"frac_high_sim: avg={np.mean(all_frac_high):.3f} | "
        f"collapse_risk_groups={n_collapse_risk}/{num_prompts}",
        file=sys.stderr,
        flush=True,
    )
    if n_collapse_risk > num_prompts * 0.5:
        print(
            f"[OE_HEALTH] *** WARNING: {n_collapse_risk}/{num_prompts} groups have "
            f"mean_pairwise_sim > 0.92 — possible mode collapse! ***",
            file=sys.stderr,
            flush=True,
        )

    # Compute per-rollout rewards preview (cosine to medoid) for sanity check
    from verl.utils.reward_score.ttrl_open_ended import grade as oe_grade
    all_rewards = []

    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"].get("ground_truth", "")
        medoid_text = medoid_list[i] or ""
        # Store the medoid as ground_truth (the reward fn computes cos_sim against it).
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = medoid_text
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = medoid_text
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

        nb = data_item.non_tensor_batch
        video_file = nb.get("video_file", "N/A")
        audio_file = nb.get("audio_file", "N/A")
        image_file = nb.get("image_file", "N/A")
        question_text = nb.get("question", "N/A")
        sample_id = nb.get("id", nb.get("index", i))
        st = stats_list[i]

        # Compute per-rollout reward for this group
        group_outputs = model_outputs[i * n : (i + 1) * n]
        group_rewards = [oe_grade(o, medoid_text) for o in group_outputs]
        all_rewards.extend(group_rewards)
        reward_arr = np.array(group_rewards)
        reward_std = float(reward_arr.std()) if len(reward_arr) > 1 else 0.0
        zero_var = reward_std < 1e-6

        # Always log the summary line
        print(
            f"[OE_TTRL INPUT] step={step} prompt {i}/{num_prompts} | id={sample_id}"
            f" | mean_pair={st['mean_pairwise_sim']:.3f}"
            f" max_pair={st['max_pairwise_sim']:.3f}"
            f" unique={st['unique_responses']}/{n}"
            f" medoid_sim={st['medoid_mean_sim']:.3f}"
            f" | reward: mean={reward_arr.mean():.3f} std={reward_std:.3f}"
            f" min={reward_arr.min():.3f} max={reward_arr.max():.3f}"
            f"{' ZERO_VAR!' if zero_var else ''}",
            file=sys.stderr,
            flush=True,
        )

        # Verbose: full details for first 3 steps or every 10th
        if verbose:
            modality = []
            if video_file and video_file != "N/A": modality.append("video")
            if audio_file and audio_file != "N/A": modality.append("audio")
            if image_file and image_file != "N/A": modality.append("image")
            print(
                f"  modalities: {'+'.join(modality) if modality else 'text-only'}"
                f"\n  question: {str(question_text)[:250]}"
                f"\n  original_gt: {str(original_gt)[:100]}"
                f"\n  medoid ({len(medoid_text)} chars): {medoid_text[:250]}",
                file=sys.stderr,
                flush=True,
            )
            # Show individual rollout rewards and snippets
            for j, (resp, rew) in enumerate(zip(group_outputs, group_rewards)):
                marker = " <- MEDOID" if resp.strip() == medoid_text.strip() else ""
                snippet = (resp.strip()[:120] + "...") if len(resp.strip()) > 120 else resp.strip()
                print(
                    f"    r{j}: reward={rew:.3f}{marker} | {snippet}",
                    file=sys.stderr,
                    flush=True,
                )

    # Batch reward summary
    all_rewards_arr = np.array(all_rewards)
    n_zero_var_groups = sum(
        1 for i in range(num_prompts)
        if np.std(all_rewards[i * n : (i + 1) * n]) < 1e-6
    )
    print(
        f"[OE_REWARD_SUMMARY] step={step} | "
        f"all_rewards: mean={all_rewards_arr.mean():.4f} std={all_rewards_arr.std():.4f} "
        f"min={all_rewards_arr.min():.4f} max={all_rewards_arr.max():.4f} | "
        f"zero_var_groups={n_zero_var_groups}/{num_prompts} "
        f"({'NO GRADIENT' if n_zero_var_groups == num_prompts else 'gradient OK'})",
        file=sys.stderr,
        flush=True,
    )

    # Stash list-shaped diagnostics on the batch so compute_ttrl_metrics can
    # aggregate them. Use the same field names the MCQ code path uses where
    # possible so existing wandb dashboards keep working.
    batch.non_tensor_batch["majority_ratio_list"] = np.array(sim_list, dtype=float)
    batch.non_tensor_batch["vote_stats_list"] = np.array(stats_list, dtype=object)

    return batch
