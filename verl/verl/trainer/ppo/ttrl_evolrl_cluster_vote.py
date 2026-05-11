# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
EVOL-RL with continuous-vector cluster voting.

Pipeline per prompt (N=16 rollouts):
  1. Encode every rollout with the configured encoder (BGE or Qwen3-Embedding-4B
     selected via TTRL_OE_ENCODER).
  2. K-means cluster the L2-normalized embeddings (K auto-detected in
     [2, K_MAX] by silhouette, defaulting to K_MAX=4). Empty / unparseable
     responses are excluded from clustering and assigned r = -1.
  3. Modal cluster = the largest cluster. Its centroid (mean of cluster
     embeddings) is the pseudo-ground-truth direction; the rollout closest
     to that centroid is the anchor *text* (used for eval-time fallback and
     human-readable logs).
  4. Per-rollout EVOL-RL reward (Zhou et al., arXiv 2509.15194):

         in modal cluster:    r_i = 0.5 + 0.5 * u_i   in [0.5, 1.0]
         not in modal cluster: r_i = -1.0 + 0.5 * u_i in [-1.0, -0.5]
         invalid response:     r_i = -1.0

     where u_i is the per-rollout novelty in [0, 1] (min-max normalized
     within its correctness band):

         novelty_i = 1 - 0.5 * mean_{j in same band, j != i} S[i,j]
                       - 0.5 * max_{j != i} S[i,j]
         u_i       = (novelty_i - min_band) / (max_band - min_band)

  5. Stash {response_text -> reward} as a JSON dict in
     reward_model["ground_truth"]; reuse ttrl_judge.reward_func at grade
     time to look up the precomputed reward.

The score map handoff matches ttrl_judge so we don't need a new reward
module — the judge_open_ended reward_func parses the JSON dict and looks
up by response text.

Diagnostics surfaced via the `vote_stats_list` non-tensor batch field so
existing wandb aggregation continues to work; we add EVOL-RL-specific keys
(modal_cluster_size_frac, n_clusters, novelty_mean, novelty_std,
intra_modal_sim, inter_cluster_sim, frac_invalid).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

from verl.utils.reward_score.ttrl_open_ended.embedding import (
    cosine_matrix,
    encode_cached,
    put_cached,
)

_DEBUG = os.environ.get("TTRL_DEBUG", "0") == "1"
_OE_DEBUG = os.environ.get("TTRL_OE_DEBUG", "0") == "1"

# Cluster hyperparameters (env-overridable).
_K_MAX = int(os.environ.get("TTRL_CLUSTER_K_MAX", "4"))
_K_MIN = int(os.environ.get("TTRL_CLUSTER_K_MIN", "2"))
_KMEANS_RANDOM_STATE = int(os.environ.get("TTRL_CLUSTER_SEED", "0"))


def _kmeans_auto_k(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    K-means with K auto-selected in [_K_MIN, _K_MAX] by silhouette.

    Args:
        E: [N, d] L2-normalized embeddings (N >= 2).

    Returns:
        labels: [N] int cluster index in [0, K)
        centroids: [K, d] cluster centroids (NOT renormalized)
        K: chosen K
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = E.shape[0]
    if n <= _K_MIN:
        # Trivial: each point is its own cluster (n=2) or all same (n=1).
        labels = np.arange(n, dtype=int)
        centroids = E.copy()
        return labels, centroids, n

    k_high = min(_K_MAX, n - 1)
    if k_high < _K_MIN:
        # Degenerate; force single cluster.
        labels = np.zeros(n, dtype=int)
        centroids = E.mean(axis=0, keepdims=True)
        return labels, centroids, 1

    best_k, best_labels, best_centroids, best_score = 1, np.zeros(n, dtype=int), E.mean(0, keepdims=True), -2.0
    for k in range(_K_MIN, k_high + 1):
        try:
            km = KMeans(n_clusters=k, n_init=4, random_state=_KMEANS_RANDOM_STATE)
            labels = km.fit_predict(E)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(E, labels, metric="cosine")
        except Exception:
            continue
        if score > best_score:
            best_k, best_labels, best_centroids, best_score = k, labels, km.cluster_centers_, score

    return best_labels, best_centroids, best_k


def _evolrl_cluster_vote(
    model_outputs: List[str],
) -> Tuple[str, Dict[str, float], dict, List[float]]:
    """
    Cluster-mode voting + EVOL-RL banded reward + novelty bonus.

    Returns:
        anchor_text: text of the rollout closest to modal-cluster centroid
        score_map:   {response_text -> reward in [-1, 1]}
        stats:       diagnostics dict (modal_cluster_size_frac, n_clusters, ...)
        per_rollout_rewards: list[float] of length len(model_outputs)
    """
    n = len(model_outputs)
    assert n > 0, "empty rollout group"

    cleaned = [(o or "").strip() for o in model_outputs]
    valid_idx = [i for i, c in enumerate(cleaned) if c]
    n_invalid = n - len(valid_idx)

    if not valid_idx:
        # All empty: zero-gradient batch.
        stats = {
            "n_total": n,
            "n_invalid": n,
            "n_clusters": 0,
            "modal_cluster_size": 0,
            "modal_cluster_size_frac": 0.0,
            "novelty_mean": 0.0,
            "novelty_std": 0.0,
            "intra_modal_sim": 0.0,
            "inter_cluster_sim": 0.0,
            "frac_invalid": 1.0,
        }
        per_rollout = [-1.0] * n
        return "", {}, stats, per_rollout

    # Encode valid rollouts only; cache populated for downstream lookups.
    valid_texts = [cleaned[i] for i in valid_idx]
    E = encode_cached(valid_texts)  # [m, d], L2-normalized
    for text, vec in zip(valid_texts, E):
        put_cached(text, vec)
    m = E.shape[0]

    # Cluster (auto-K via silhouette).
    labels, centroids, K = _kmeans_auto_k(E)

    # Modal cluster = largest by member count; tie-break by highest mean
    # intra-cluster similarity.
    cluster_sizes = np.bincount(labels, minlength=K)
    largest = int(cluster_sizes.max())
    candidates = np.where(cluster_sizes == largest)[0]
    if len(candidates) == 1:
        modal = int(candidates[0])
    else:
        # Tie-break: pick cluster with highest mean cosine-to-centroid among its members.
        best_modal, best_mean = int(candidates[0]), -2.0
        for c in candidates:
            mask = labels == c
            cent = centroids[c] / (np.linalg.norm(centroids[c]) + 1e-12)
            mean_sim = float((E[mask] @ cent).mean())
            if mean_sim > best_mean:
                best_modal, best_mean = int(c), mean_sim
        modal = best_modal

    modal_centroid = centroids[modal] / (np.linalg.norm(centroids[modal]) + 1e-12)

    # Anchor text = valid rollout with highest cosine to modal centroid.
    sims_to_centroid = E @ modal_centroid  # [m]
    anchor_local = int(np.argmax(sims_to_centroid))
    anchor_text = valid_texts[anchor_local]

    # Pairwise similarity matrix on valid rollouts (for novelty).
    S = cosine_matrix(E)  # [m, m] in [-1, 1] (L2-normed inputs)
    np.fill_diagonal(S, -2.0)  # exclude self from neighbor stats

    in_modal = labels == modal  # [m] bool

    # Per-rollout EVOL-RL novelty: 1 - 0.5*intra_band_avg - 0.5*global_max
    # Band = same correctness label (in modal vs out modal).
    novelty_raw = np.zeros(m, dtype=float)
    for i in range(m):
        global_max = float(np.max(S[i])) if m > 1 else 0.0
        same_band = in_modal == in_modal[i]
        same_band[i] = False  # exclude self
        if same_band.any():
            intra_avg = float(S[i, same_band].mean())
        else:
            intra_avg = 0.0
        novelty_raw[i] = 1.0 - 0.5 * intra_avg - 0.5 * global_max

    # Min-max normalize novelty within each band into u_i in [0, 1].
    u = np.zeros(m, dtype=float)
    for band_mask, _ in ((in_modal, "modal"), (~in_modal, "outside")):
        if band_mask.sum() == 0:
            continue
        band_vals = novelty_raw[band_mask]
        lo, hi = float(band_vals.min()), float(band_vals.max())
        if hi - lo < 1e-9:
            u[band_mask] = 0.5  # all tied -> middle-of-band
        else:
            u[band_mask] = (band_vals - lo) / (hi - lo)

    # Banded reward: [0.5, 1.0] for modal, [-1.0, -0.5] for outside.
    rewards_valid = np.where(in_modal, 0.5 + 0.5 * u, -1.0 + 0.5 * u)

    # Reassemble per-original-rollout reward; invalid -> -1.
    per_rollout_rewards = [-1.0] * n
    for local_i, orig_i in enumerate(valid_idx):
        per_rollout_rewards[orig_i] = float(rewards_valid[local_i])

    # Map response_text -> reward (for ttrl_judge reward_func lookup).
    score_map: Dict[str, float] = {}
    for i, c in enumerate(cleaned):
        if c:
            # Map both raw and stripped variants to the same reward to
            # survive whitespace differences between rollout encoding paths.
            score_map[cleaned[i]] = per_rollout_rewards[i]

    # Diagnostics.
    intra_modal_sim = float(S[in_modal][:, in_modal].mean()) if in_modal.sum() > 1 else 0.0
    if K > 1:
        inter_block = S.copy()
        np.fill_diagonal(inter_block, 0.0)
        same_label = labels[:, None] == labels[None, :]
        inter_mask = ~same_label & (inter_block > -1.5)
        inter_cluster_sim = float(inter_block[inter_mask].mean()) if inter_mask.any() else 0.0
    else:
        inter_cluster_sim = 0.0

    stats = {
        "n_total": n,
        "n_invalid": n_invalid,
        "n_clusters": int(K),
        "modal_cluster_size": int(in_modal.sum()),
        "modal_cluster_size_frac": float(in_modal.sum() / max(1, m)),
        "novelty_mean": float(novelty_raw.mean()) if m else 0.0,
        "novelty_std": float(novelty_raw.std()) if m > 1 else 0.0,
        "intra_modal_sim": intra_modal_sim,
        "inter_cluster_sim": inter_cluster_sim,
        "frac_invalid": float(n_invalid / n),
    }

    if _DEBUG or _OE_DEBUG:
        print(
            f"[EVOLRL_VOTE] n={n} m_valid={m} K={K} modal={modal} "
            f"|modal|={int(in_modal.sum())}/{m} "
            f"intra_modal_sim={intra_modal_sim:.3f} inter_cluster_sim={inter_cluster_sim:.3f} "
            f"novelty_mean={stats['novelty_mean']:.3f}",
            file=sys.stderr,
            flush=True,
        )

    return anchor_text, score_map, stats, per_rollout_rewards


def _simple_cluster_vote(
    model_outputs: List[str],
) -> Tuple[str, Dict[str, float], dict, List[float]]:
    """
    Stripped-down cluster vote with BINARY reward (no novelty, no banding).

    Per-rollout reward:
        in modal cluster: 1.0
        elsewhere:        0.0
        invalid:          0.0

    Same k-means and embedding pipeline as _evolrl_cluster_vote, just with
    none of the EVOL-RL/DAPO machinery on top. This is the rung-1 ablation
    of "does plain cluster voting beat medoid voting?"

    Returns same shape as _evolrl_cluster_vote (anchor, score_map, stats,
    per_rollout_rewards) so the dispatch code is parallel.
    """
    n = len(model_outputs)
    assert n > 0, "empty rollout group"

    cleaned = [(o or "").strip() for o in model_outputs]
    valid_idx = [i for i, c in enumerate(cleaned) if c]
    n_invalid = n - len(valid_idx)

    if not valid_idx:
        stats = {
            "n_total": n,
            "n_invalid": n,
            "n_clusters": 0,
            "modal_cluster_size": 0,
            "modal_cluster_size_frac": 0.0,
            "intra_modal_sim": 0.0,
            "inter_cluster_sim": 0.0,
            "frac_invalid": 1.0,
        }
        per_rollout = [0.0] * n
        return "", {}, stats, per_rollout

    valid_texts = [cleaned[i] for i in valid_idx]
    E = encode_cached(valid_texts)
    for text, vec in zip(valid_texts, E):
        put_cached(text, vec)
    m = E.shape[0]

    labels, centroids, K = _kmeans_auto_k(E)

    cluster_sizes = np.bincount(labels, minlength=K)
    largest = int(cluster_sizes.max())
    candidates = np.where(cluster_sizes == largest)[0]
    if len(candidates) == 1:
        modal = int(candidates[0])
    else:
        # Tie-break: highest mean cosine-to-centroid among members.
        best_modal, best_mean = int(candidates[0]), -2.0
        for c in candidates:
            mask = labels == c
            cent = centroids[c] / (np.linalg.norm(centroids[c]) + 1e-12)
            mean_sim = float((E[mask] @ cent).mean())
            if mean_sim > best_mean:
                best_modal, best_mean = int(c), mean_sim
        modal = best_modal

    modal_centroid = centroids[modal] / (np.linalg.norm(centroids[modal]) + 1e-12)
    sims_to_centroid = E @ modal_centroid
    anchor_local = int(np.argmax(sims_to_centroid))
    anchor_text = valid_texts[anchor_local]

    in_modal = labels == modal

    # Reward shape. Default = binary (1.0 in modal cluster, 0.0 else).
    # If TTRL_CLUSTER_CONTINUOUS=1, use continuous cosine-sim-to-medoid mapped
    # to [0, 1]: r = (cos(emb, anchor_emb) + 1) / 2. Restores gradient slope
    # between "almost in modal" and "far from modal" — useful when modal_frac
    # is high and binary reward collapses GRPO advantage to ~0.
    if os.environ.get("TTRL_CLUSTER_CONTINUOUS", "0") == "1":
        anchor_emb = E[anchor_local]
        anchor_emb_norm = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-12)
        E_norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        E_unit = E / E_norms
        sims_to_anchor = E_unit @ anchor_emb_norm  # (m,), in [-1, 1]
        rewards_valid = (sims_to_anchor + 1.0) / 2.0  # in [0, 1]
    else:
        rewards_valid = in_modal.astype(float)

    per_rollout_rewards = [0.0] * n
    for local_i, orig_i in enumerate(valid_idx):
        per_rollout_rewards[orig_i] = float(rewards_valid[local_i])

    score_map: Dict[str, float] = {}
    for i, c in enumerate(cleaned):
        if c:
            score_map[cleaned[i]] = per_rollout_rewards[i]

    # Diagnostics (subset of EVOL-RL stats).
    S = cosine_matrix(E)
    np.fill_diagonal(S, -2.0)
    intra_modal_sim = float(S[in_modal][:, in_modal].mean()) if in_modal.sum() > 1 else 0.0
    if K > 1:
        inter_block = S.copy()
        np.fill_diagonal(inter_block, 0.0)
        same_label = labels[:, None] == labels[None, :]
        inter_mask = ~same_label & (inter_block > -1.5)
        inter_cluster_sim = float(inter_block[inter_mask].mean()) if inter_mask.any() else 0.0
    else:
        inter_cluster_sim = 0.0

    stats = {
        "n_total": n,
        "n_invalid": n_invalid,
        "n_clusters": int(K),
        "modal_cluster_size": int(in_modal.sum()),
        "modal_cluster_size_frac": float(in_modal.sum() / max(1, m)),
        "intra_modal_sim": intra_modal_sim,
        "inter_cluster_sim": inter_cluster_sim,
        "frac_invalid": float(n_invalid / n),
    }

    if _DEBUG or _OE_DEBUG:
        print(
            f"[SIMPLE_CLUSTER_VOTE] n={n} m_valid={m} K={K} modal={modal} "
            f"|modal|={int(in_modal.sum())}/{m} "
            f"intra={intra_modal_sim:.3f} inter={inter_cluster_sim:.3f}",
            file=sys.stderr,
            flush=True,
        )

    return anchor_text, score_map, stats, per_rollout_rewards


def apply_ttrl_simple_cluster_gt(batch, gen_batch_output, n, tokenizer):
    """
    Rung-1 ablation: plain cluster vote with binary {0, 1} reward.

    Same dispatch shape as apply_ttrl_evolrl_cluster_gt; uses
    _simple_cluster_vote which strips the novelty / banding machinery.
    Reward path still routes through ttrl_judge.reward_func via the JSON
    score map stash.
    """
    assert len(gen_batch_output) % n == 0, (
        f"gen_batch_output length {len(gen_batch_output)} not divisible by n={n}"
    )
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts

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

    if not hasattr(apply_ttrl_simple_cluster_gt, "_call_count"):
        apply_ttrl_simple_cluster_gt._call_count = 0
    apply_ttrl_simple_cluster_gt._call_count += 1
    step = apply_ttrl_simple_cluster_gt._call_count
    verbose = step <= 3 or step % 10 == 0

    print(
        f"\n[SIMPLE_CLUSTER_TTRL] step={step}: {num_prompts} prompts, {n} rollouts each",
        file=sys.stderr,
        flush=True,
    )

    anchor_list: List[str] = []
    stats_list: List[dict] = []
    sim_list: List[float] = []
    score_map_list: List[Dict[str, float]] = []
    all_rewards: List[float] = []

    for i in range(num_prompts):
        group_outputs = model_outputs[i * n : (i + 1) * n]
        anchor, score_map, stats, group_rewards = _simple_cluster_vote(group_outputs)
        anchor_list.append(anchor)
        stats_list.append(stats)
        sim_list.append(stats["modal_cluster_size_frac"])
        score_map_list.append(score_map)
        all_rewards.extend(group_rewards)

    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"].get("ground_truth", "")
        gt_json = json.dumps(score_map_list[i], ensure_ascii=False)
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = gt_json
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = anchor_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

        st = stats_list[i]
        nb = data_item.non_tensor_batch
        sample_id = nb.get("id", nb.get("index", i))
        question_text = nb.get("question", "N/A")
        group_outputs = model_outputs[i * n : (i + 1) * n]
        group_rewards = all_rewards[i * n : (i + 1) * n]
        reward_arr = np.array(group_rewards)

        print(
            f"[SIMPLE_CLUSTER INPUT] step={step} prompt {i}/{num_prompts} | id={sample_id}"
            f" | K={st['n_clusters']} modal_frac={st['modal_cluster_size_frac']:.2f}"
            f" intra_modal={st['intra_modal_sim']:.3f} inter={st['inter_cluster_sim']:.3f}"
            f" | reward: mean={reward_arr.mean():.3f} std={reward_arr.std():.3f}"
            f" min={reward_arr.min():.3f} max={reward_arr.max():.3f}",
            file=sys.stderr,
            flush=True,
        )

        if verbose:
            print(
                f"  question: {str(question_text)[:200]}"
                f"\n  original_gt: {str(original_gt)[:100]}"
                f"\n  anchor ({len(anchor_list[i])} chars): {anchor_list[i][:200]}",
                file=sys.stderr,
                flush=True,
            )
            for j, (resp, rew) in enumerate(zip(group_outputs, group_rewards)):
                marker = " <- ANCHOR" if resp.strip() == anchor_list[i].strip() else ""
                snippet = (resp.strip()[:120] + "...") if len(resp.strip()) > 120 else resp.strip()
                print(
                    f"    r{j}: reward={rew:.0f}{marker} | {snippet}",
                    file=sys.stderr,
                    flush=True,
                )

    all_rewards_arr = np.array(all_rewards)
    n_zero_var = sum(
        1 for i in range(num_prompts)
        if np.std(all_rewards[i * n : (i + 1) * n]) < 1e-6
    )
    avg_modal_frac = float(np.mean([s["modal_cluster_size_frac"] for s in stats_list]))
    avg_K = float(np.mean([s["n_clusters"] for s in stats_list]))
    avg_intra = float(np.mean([s["intra_modal_sim"] for s in stats_list]))
    avg_inter = float(np.mean([s["inter_cluster_sim"] for s in stats_list]))
    print(
        f"[SIMPLE_CLUSTER_HEALTH] step={step} | "
        f"avg_K={avg_K:.2f} avg_modal_frac={avg_modal_frac:.3f} "
        f"avg_intra_modal={avg_intra:.3f} avg_inter_cluster={avg_inter:.3f} | "
        f"reward: mean={all_rewards_arr.mean():.3f} std={all_rewards_arr.std():.3f} "
        f"zero_var_groups={n_zero_var}/{num_prompts}",
        file=sys.stderr,
        flush=True,
    )

    # Auxiliary monitoring metrics (BLEU / ROUGE-L / exact-match / optional
    # GPT-4o-mini judge) against the *real* gold. Does not affect training —
    # only logged via vote_stats_list -> compute_ttrl_metrics. Gated by
    # TTRL_AUX_DETERMINISTIC=1 (default on) and TTRL_AUX_GPT_JUDGE=1 (opt-in).
    try:
        questions = []
        golds = []
        for i in range(num_prompts):
            nb = batch[i].non_tensor_batch
            q = nb.get("question") or nb.get("prompt") or ""
            g = nb["reward_model"].get("original_gt", "")
            if isinstance(g, (list, tuple)):
                g = g[0] if g else ""
            questions.append(str(q))
            golds.append(str(g))
        from verl.trainer.ppo.ttrl_aux_metrics import compute_aux_metrics_for_batch
        compute_aux_metrics_for_batch(
            batch=batch,
            model_outputs=model_outputs,
            num_prompts=num_prompts,
            n=n,
            stats_list=stats_list,
            questions=questions,
            golds=golds,
        )
        if stats_list and "aux_bleu_mean" in stats_list[0]:
            agg = lambda k: float(np.mean([s.get(k, 0.0) for s in stats_list]))
            extra = (
                f" | aux_bleu={agg('aux_bleu_mean'):.3f}"
                f" aux_rouge_l={agg('aux_rouge_l_mean'):.3f}"
                f" aux_em={agg('aux_exact_match_mean'):.3f}"
            )
            if "aux_gpt_judge_mean" in stats_list[0]:
                extra += f" aux_gpt={agg('aux_gpt_judge_mean'):.3f}"
            print(f"[SIMPLE_CLUSTER_AUX] step={step}{extra}", file=sys.stderr, flush=True)
    except Exception as _e:
        print(f"[SIMPLE_CLUSTER_AUX] skipped ({type(_e).__name__}: {_e})",
              file=sys.stderr, flush=True)

    batch.non_tensor_batch["majority_ratio_list"] = np.array(sim_list, dtype=float)
    batch.non_tensor_batch["vote_stats_list"] = np.array(stats_list, dtype=object)
    return batch


def apply_ttrl_evolrl_cluster_gt(batch, gen_batch_output, n, tokenizer):
    """
    EVOL-RL cluster voting analog of apply_ttrl_open_ended_gt /
    apply_ttrl_judge_gt.

    For each prompt, runs cluster voting + EVOL-RL banded reward +
    novelty bonus, then stashes a JSON {response_text -> reward} map in
    reward_model["ground_truth"]. The reward_func from ttrl_judge looks up
    the precomputed reward by exact response text match.

    Signature matches apply_ttrl_gt so the trainer's dispatcher can call
    this without changing call sites.
    """
    assert len(gen_batch_output) % n == 0, (
        f"gen_batch_output length {len(gen_batch_output)} not divisible by n={n}"
    )
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, (
        f"batch length {len(batch)} != num_prompts {num_prompts}"
    )

    # Decode every rollout.
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

    # Track call count for periodic verbose logging
    if not hasattr(apply_ttrl_evolrl_cluster_gt, "_call_count"):
        apply_ttrl_evolrl_cluster_gt._call_count = 0
    apply_ttrl_evolrl_cluster_gt._call_count += 1
    step = apply_ttrl_evolrl_cluster_gt._call_count
    verbose = step <= 3 or step % 10 == 0

    print(
        f"\n[EVOLRL_TTRL] step={step} apply_ttrl_evolrl_cluster_gt: "
        f"{num_prompts} prompts, {n} rollouts each, "
        f"{len(model_outputs)} total outputs",
        file=sys.stderr,
        flush=True,
    )

    # Per-prompt: cluster + reward; collect for diagnostics.
    anchor_list: List[str] = []
    stats_list: List[dict] = []
    sim_list: List[float] = []  # repurposed: modal_cluster_size_frac (for compat)
    score_map_list: List[Dict[str, float]] = []
    all_rewards: List[float] = []

    for i in range(num_prompts):
        group_outputs = model_outputs[i * n : (i + 1) * n]
        anchor, score_map, stats, group_rewards = _evolrl_cluster_vote(group_outputs)
        anchor_list.append(anchor)
        stats_list.append(stats)
        sim_list.append(stats["modal_cluster_size_frac"])
        score_map_list.append(score_map)
        all_rewards.extend(group_rewards)

    # Stash per-prompt: ground_truth = JSON score map (for train-time);
    # also expose anchor + original_gt for eval-time fallback.
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"].get("ground_truth", "")
        # Encode score map as JSON; ttrl_judge.compute_score will parse it.
        gt_json = json.dumps(score_map_list[i], ensure_ascii=False)
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = gt_json
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = anchor_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

        st = stats_list[i]
        nb = data_item.non_tensor_batch
        sample_id = nb.get("id", nb.get("index", i))
        question_text = nb.get("question", "N/A")
        group_outputs = model_outputs[i * n : (i + 1) * n]
        group_rewards = all_rewards[i * n : (i + 1) * n]
        reward_arr = np.array(group_rewards)

        print(
            f"[EVOLRL INPUT] step={step} prompt {i}/{num_prompts} | id={sample_id}"
            f" | K={st['n_clusters']} modal_frac={st['modal_cluster_size_frac']:.2f}"
            f" intra_modal={st['intra_modal_sim']:.3f} inter={st['inter_cluster_sim']:.3f}"
            f" novelty_mean={st['novelty_mean']:.3f}"
            f" | reward: mean={reward_arr.mean():.3f} std={reward_arr.std():.3f}"
            f" min={reward_arr.min():.3f} max={reward_arr.max():.3f}",
            file=sys.stderr,
            flush=True,
        )

        if verbose:
            print(
                f"  question: {str(question_text)[:200]}"
                f"\n  original_gt: {str(original_gt)[:100]}"
                f"\n  anchor ({len(anchor_list[i])} chars): {anchor_list[i][:200]}",
                file=sys.stderr,
                flush=True,
            )
            for j, (resp, rew) in enumerate(zip(group_outputs, group_rewards)):
                marker = " <- ANCHOR" if resp.strip() == anchor_list[i].strip() else ""
                snippet = (resp.strip()[:120] + "...") if len(resp.strip()) > 120 else resp.strip()
                print(
                    f"    r{j}: reward={rew:+.3f}{marker} | {snippet}",
                    file=sys.stderr,
                    flush=True,
                )

    # Batch-level health summary.
    all_rewards_arr = np.array(all_rewards)
    n_zero_var = sum(
        1 for i in range(num_prompts)
        if np.std(all_rewards[i * n : (i + 1) * n]) < 1e-6
    )
    avg_modal_frac = float(np.mean([s["modal_cluster_size_frac"] for s in stats_list]))
    avg_K = float(np.mean([s["n_clusters"] for s in stats_list]))
    avg_novelty = float(np.mean([s["novelty_mean"] for s in stats_list]))
    avg_intra = float(np.mean([s["intra_modal_sim"] for s in stats_list]))
    avg_inter = float(np.mean([s["inter_cluster_sim"] for s in stats_list]))
    frac_ambiguous = float(np.mean([1.0 if s["modal_cluster_size_frac"] < 0.5 else 0.0 for s in stats_list]))
    print(
        f"[EVOLRL_HEALTH] step={step} | "
        f"avg_K={avg_K:.2f} avg_modal_frac={avg_modal_frac:.3f} "
        f"avg_novelty={avg_novelty:.3f} "
        f"avg_intra_modal={avg_intra:.3f} avg_inter_cluster={avg_inter:.3f} "
        f"frac_ambiguous={frac_ambiguous:.3f} | "
        f"reward: mean={all_rewards_arr.mean():.3f} std={all_rewards_arr.std():.3f} "
        f"zero_var_groups={n_zero_var}/{num_prompts}",
        file=sys.stderr,
        flush=True,
    )

    # Stash list-shaped diagnostics on the batch so compute_ttrl_metrics can
    # aggregate them into wandb. Reuse the "majority_ratio" name slot for
    # modal_cluster_size_frac so existing dashboards keep working.
    batch.non_tensor_batch["majority_ratio_list"] = np.array(sim_list, dtype=float)
    batch.non_tensor_batch["vote_stats_list"] = np.array(stats_list, dtype=object)
    return batch
