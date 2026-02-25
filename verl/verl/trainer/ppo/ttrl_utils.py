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

    majority_gt_list, majority_ratio_list = _batch_majority_vote(model_outputs, n)
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    if os.environ.get("TTRL_DEBUG", "0") == "1":
        print(f"\n[TTRL DEBUG] apply_ttrl_gt: {num_prompts} prompts, {n} votes each", file=sys.stderr)
    
    # Debug: show type of ground_truth from dataset (can be list -> causes Counter to fail if not normalized)
    TTRL_DEBUG = os.environ.get("TTRL_DEBUG", "0") == "1"
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        if i == 0:
            print(f"[TTRL] apply_ttrl_gt: original_gt type={type(original_gt).__name__}, is_list={isinstance(original_gt, list)}, repr={repr(original_gt)[:80]}", file=sys.stderr)
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt
        
        if TTRL_DEBUG and i < 3:  # Show first 3 prompts
            print(f"  Prompt {i}: original_gt type={type(original_gt).__name__} value='{original_gt}' -> majority_gt='{majority_gt_list[i]}' (ratio={majority_ratio_list[i]:.2f})", file=sys.stderr)

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch


def _batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n: int
    Output:
        majority_gt_list: list of str
        majority_ratio_list: list of float
    """
    majority_gt_list = []
    majority_ratio_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n] # indexing: [0, n-1], [n, 2n-1], ...
        prompt_majority_gt, prompt_majority_ratio = _majority_vote(prompt_outputs)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        
    return majority_gt_list, majority_ratio_list


def _majority_vote(model_outputs: List[str]) -> tuple[str, float]:
    assert len(model_outputs) > 0
    
    # DEBUG: Show raw model outputs (first 200 chars each)
    TTRL_DEBUG = os.environ.get("TTRL_DEBUG", "0") == "1"
    if TTRL_DEBUG:
        print(f"\n[TTRL DEBUG] _majority_vote called with {len(model_outputs)} outputs")
        for idx, out in enumerate(model_outputs):
            print(f"  Output {idx}: {out[:200]}..." if len(out) > 200 else f"  Output {idx}: {out}")
    

    print(f"================================================\n")
    raw_model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    
    if TTRL_DEBUG:
        print(f"[TTRL DEBUG] Raw extracted answers: {raw_model_answers}")
    
    model_answers = [answer for answer in raw_model_answers if answer is not None]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    
    if TTRL_DEBUG:
        print(f"[TTRL DEBUG] After filtering None and simplify: {model_answers}")
    
    if len(model_answers) == 0:
        if TTRL_DEBUG:
            print(f"[TTRL DEBUG] WARNING: All answers were None! Returning 'None', 0.0")
        return "None", 0.0
    
    counter = Counter(model_answers)
    
    majority_answer, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / len(model_outputs)
    
    if TTRL_DEBUG:
        print(f"[TTRL DEBUG] Majority answer: {majority_answer}, ratio: {majority_ratio}")
    
    return majority_answer, majority_ratio


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

        if i == 0:
            print(f"[TTRL] _batch_compute_ttrl_metrics: first prompt group (n={n})", file=sys.stderr)
            print(f"[TTRL]   prompt_gt_label types: {[type(x).__name__ for x in prompt_gt_label]}", file=sys.stderr)
            print(f"[TTRL]   prompt_majority_label types: {[type(x).__name__ for x in prompt_majority_label]}", file=sys.stderr)

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label_str = _label_to_str(prompt_majority_label[0])
        prompt_gt_label_str = _label_to_str(prompt_gt_label[0])
        if i == 0:
            m_preview = (prompt_majority_label_str[:60] + "...") if len(prompt_majority_label_str) > 60 else prompt_majority_label_str
            g_preview = (prompt_gt_label_str[:60] + "...") if len(prompt_gt_label_str) > 60 else prompt_gt_label_str
            print(f"[TTRL]   after _label_to_str: majority_gt='{m_preview}', original_gt='{g_preview}'", file=sys.stderr)

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