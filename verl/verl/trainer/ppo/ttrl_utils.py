# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
from collections import Counter
import math
import random
import sys
import torch
import numpy as np
import os

# Control which answer extraction/grading to use via environment variable
# Set TTRL_TASK_TYPE="video_qa" for OmniVideo/video QA tasks  
# Set TTRL_TASK_TYPE="math" for math tasks (default)
TTRL_TASK_TYPE = os.environ.get("TTRL_TASK_TYPE", "video_qa")  # Default to video_qa for now

if TTRL_TASK_TYPE == "video_qa":
    from verl.utils.reward_score.ttrl_video_qa import extract_answer, grade
    # Video QA doesn't need simplify_expression_string, use identity function
    def simplify_expression_string(s):
        return s if s else ""
    print(f"[TTRL] Using ttrl_video_qa extract_answer and grade functions (TTRL_TASK_TYPE={TTRL_TASK_TYPE})")
else:
    from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade
    print(f"[TTRL] Using ttrl_math extract_answer and grade functions (TTRL_TASK_TYPE={TTRL_TASK_TYPE})")

def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]


# === Ground Truth Manipulation ===


def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch


def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []  
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

    majority_gt_list, majority_ratio_list, vote_stats_list = _batch_majority_vote(model_outputs, n)
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    TTRL_DEBUG = os.environ.get("TTRL_DEBUG", "0") == "1"
    if TTRL_DEBUG:
        print(f"\n[TTRL DEBUG] apply_ttrl_gt: {num_prompts} prompts, {n} votes each", file=sys.stderr)
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        if i == 0:
            print(f"[TTRL] apply_ttrl_gt: original_gt type={type(original_gt).__name__}, is_list={isinstance(original_gt, list)}, repr={repr(original_gt)[:80]}", file=sys.stderr)
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt
        
        # Log input files and question for each prompt
        nb = data_item.non_tensor_batch
        video_file = nb.get("video_file", "N/A")
        audio_file = nb.get("audio_file", "N/A")
        question_text = nb.get("question", "N/A")
        sample_id = nb.get("id", nb.get("index", i))
        q_type = nb.get("question_type", nb.get("source", "N/A"))
        print(
            f"[TTRL INPUT] prompt {i}/{num_prompts} | id={sample_id} | type={q_type}"
            f" | gt={original_gt} -> majority={majority_gt_list[i]} (ratio={majority_ratio_list[i]:.2f})"
            f"\n  video: {video_file}"
            f"\n  audio: {audio_file}"
            f"\n  question: {question_text[:200]}{'...' if len(str(question_text)) > 200 else ''}",
            file=sys.stderr, flush=True,
        )
        
        if TTRL_DEBUG and i < 3:
            print(f"  Prompt {i}: original_gt type={type(original_gt).__name__} value='{original_gt}' -> majority_gt='{majority_gt_list[i]}' (ratio={majority_ratio_list[i]:.2f})", file=sys.stderr)

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["vote_stats_list"] = np.array(vote_stats_list, dtype=object)
    return batch


def _batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float], List[dict]]:
    """
    Used to generate the ground truth for TTRL.
    Returns:
        majority_gt_list: list of str
        majority_ratio_list: list of float
        vote_stats_list: list of dict with per-prompt vote distribution stats
    """
    majority_gt_list = []
    majority_ratio_list = []
    vote_stats_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        prompt_majority_gt, prompt_majority_ratio, vote_stats = _majority_vote(prompt_outputs)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        vote_stats_list.append(vote_stats)
        
    return majority_gt_list, majority_ratio_list, vote_stats_list


def _vote_entropy(counter: Counter, total: int, n_choices: int = 4) -> float:
    """Normalized Shannon entropy of vote distribution. 0 = unanimous, 1 = uniform over n_choices."""
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    max_entropy = math.log2(n_choices)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _majority_vote(model_outputs: List[str]) -> tuple[str, float, dict]:
    assert len(model_outputs) > 0
    n = len(model_outputs)
    
    TTRL_DEBUG = os.environ.get("TTRL_DEBUG", "0") == "1"
    if TTRL_DEBUG:
        print(f"\n[TTRL DEBUG] _majority_vote called with {n} outputs")
        for idx, out in enumerate(model_outputs):
            print(f"  Output {idx}: {out}")

    raw_model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    
    if TTRL_DEBUG:
        print(f"[TTRL DEBUG] Raw extracted answers: {raw_model_answers}")
    
    n_unparseable = sum(1 for a in raw_model_answers if a is None)
    model_answers = [answer for answer in raw_model_answers if answer is not None]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    
    if TTRL_DEBUG:
        print(f"[TTRL DEBUG] After filtering None and simplify: {model_answers}")
    
    if len(model_answers) == 0:
        fallback = random.choice(["A", "B", "C", "D"])
        if TTRL_DEBUG:
            print(f"[TTRL DEBUG] WARNING: All answers were None! Random fallback: {fallback}")
        vote_stats = {
            "n_total": n,
            "n_unparseable": n_unparseable,
            "n_unique_answers": 0,
            "vote_entropy": 0.0,
            "unanimous": False,
        }
        return fallback, 0.0, vote_stats
    
    counter = Counter(model_answers)
    majority_answer, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / n
    
    vote_stats = {
        "n_total": n,
        "n_unparseable": n_unparseable,
        "n_unique_answers": len(counter),
        "vote_entropy": _vote_entropy(counter, len(model_answers)),
        "unanimous": len(counter) == 1 and n_unparseable == 0,
    }
    
    if TTRL_DEBUG:
        print(f"[TTRL DEBUG] Majority answer: {majority_answer}, ratio: {majority_ratio}, stats: {vote_stats}")
    
    return majority_answer, majority_ratio, vote_stats


# === Metrics Computation ===


def _get_batch_index(data_item):
    non_tensor = data_item.non_tensor_batch
    if "extra_info" in non_tensor and isinstance(non_tensor["extra_info"], dict):
        return non_tensor["extra_info"].get("index", 0)
    return non_tensor.get("index", 0)


def _label_to_hashable(label):
    """Normalize ground-truth label so Counter() can hash it (lists are unhashable)."""
    if isinstance(label, list):
        return tuple(label)
    return label


def _label_to_str(label):
    """Convert label to str for grade(); dataset may store ground_truth as a list."""
    if isinstance(label, (list, tuple)):
        return label[0] if label else ""
    return label if label is not None else ""


def compute_ttrl_metrics(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    idx = sorted(range(len(batch)), key=lambda x: _get_batch_index(batch[x]))

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        raw_maj = data_item.non_tensor_batch["reward_model"]["majority_gt"]
        raw_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        if i == 0:
            print(f"[TTRL] compute_ttrl_metrics: first item raw majority_gt type={type(raw_maj).__name__}, original_gt type={type(raw_gt).__name__}", file=sys.stderr)
            print(f"[TTRL]   -> after _label_to_hashable: majority hashable={type(_label_to_hashable(raw_maj)).__name__}, gt hashable={type(_label_to_hashable(raw_gt)).__name__}", file=sys.stderr)
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(_label_to_hashable(raw_maj))
        gt_label.append(_label_to_hashable(raw_gt))

    ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n)
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
    ttrl_metrics["majority_ratio"] = majority_ratio

    # Aggregate vote distribution stats across prompts.
    # vote_stats_list is repeated n times (once per sample), deduplicate by stride n.
    vote_stats_list = batch.non_tensor_batch.get("vote_stats_list", None)
    if vote_stats_list is not None and len(vote_stats_list) > 0:
        unique_stats = [vote_stats_list[i] for i in range(0, len(vote_stats_list), n)]
        num_p = len(unique_stats)
        avg_entropy = sum(s["vote_entropy"] for s in unique_stats) / num_p
        avg_unique = sum(s["n_unique_answers"] for s in unique_stats) / num_p
        frac_unanimous = sum(1 for s in unique_stats if s["unanimous"]) / num_p
        total_unparseable = sum(s["n_unparseable"] for s in unique_stats)
        total_votes = sum(s["n_total"] for s in unique_stats)
        frac_unparseable = total_unparseable / total_votes if total_votes > 0 else 0.0

        ttrl_metrics["vote_entropy"] = avg_entropy
        ttrl_metrics["vote_n_unique_answers"] = avg_unique
        ttrl_metrics["vote_frac_unanimous"] = frac_unanimous
        ttrl_metrics["vote_frac_unparseable"] = frac_unparseable

    return ttrl_metrics


def _batch_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    n: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
    assert len(majority_reward) % n == 0
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label_str = _label_to_str(prompt_majority_label[0])
        prompt_gt_label_str = _label_to_str(prompt_gt_label[0])

        ttrl_metric = _prompt_compute_ttrl_metrics(prompt_majority_reward, prompt_gt_reward, prompt_majority_label_str, prompt_gt_label_str)
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    ):    
    assert len(majority_reward) == len(gt_reward)

    grade_result = grade(majority_label, gt_label)
    hit_rate = 1.0 if grade_result else 0.0
    
    # DEBUG: Show label comparison
    TTRL_DEBUG = os.environ.get("TTRL_DEBUG", "0") == "1"
    if TTRL_DEBUG:
        print(f"\n[TTRL DEBUG] _prompt_compute_ttrl_metrics:", file=sys.stderr)
        print(f"  majority_label: '{majority_label}'", file=sys.stderr)
        print(f"  gt_label (original): '{gt_label}'", file=sys.stderr)
        print(f"  grade(majority, gt) = {grade_result} -> label_accuracy = {hit_rate}", file=sys.stderr)
        print(f"  majority_rewards: {majority_reward}", file=sys.stderr)
        print(f"  gt_rewards: {gt_reward}", file=sys.stderr)
    
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric