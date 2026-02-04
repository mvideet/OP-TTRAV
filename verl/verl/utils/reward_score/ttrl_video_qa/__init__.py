# Copyright 2024 PRIME team and/or its affiliates
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

"""Provides video QA answer grading for multiple choice questions.
Designed for OmniVideo-style audio-visual reasoning tasks.
"""

import re
import traceback


def extract_answer(response: str) -> str:
    """
    Extract the answer choice (A, B, C, D, etc.) from model response.
    
    Looks for patterns like:
    - "The answer is A"
    - "Answer: B"  
    - "(C)"
    - "\\boxed{D}"
    - Just the letter at the end
    """
    if not response:
        return None
    
    response = response.strip()
    
    # Try to find boxed answer first (for models that use LaTeX formatting)
    boxed_match = re.search(r'\\boxed\{([A-Da-d])\}', response)
    if boxed_match:
        return boxed_match.group(1).upper()
    
    # Look for "answer is X" pattern
    answer_is_match = re.search(r'(?:answer|choice|option)\s*(?:is|:)\s*[(\[]?\s*([A-Da-d])\s*[)\]]?', response, re.IGNORECASE)
    if answer_is_match:
        return answer_is_match.group(1).upper()
    
    # Look for standalone letter in parentheses or brackets at end
    paren_match = re.search(r'[(\[]\s*([A-Da-d])\s*[)\]]\s*$', response)
    if paren_match:
        return paren_match.group(1).upper()
    
    # Look for "X." or "X:" pattern (like "A." or "B:")
    letter_match = re.search(r'\b([A-Da-d])\s*[.):]\s*$', response)
    if letter_match:
        return letter_match.group(1).upper()
    
    # Last resort: find any standalone A/B/C/D in the last part of response
    # Look at last 100 chars
    last_part = response[-100:] if len(response) > 100 else response
    last_letter_match = re.search(r'\b([A-Da-d])\b', last_part)
    if last_letter_match:
        return last_letter_match.group(1).upper()
    
    return None


def grade(model_answer: str, gt_answer: str) -> bool:
    """
    Grade whether the model answer matches the ground truth.
    
    Args:
        model_answer: Extracted answer from model (A, B, C, D)
        gt_answer: Ground truth answer (A, B, C, D)
        
    Returns:
        True if correct, False otherwise
    """
    if model_answer is None or gt_answer is None:
        return False
    
    # Normalize both answers
    model_answer = model_answer.strip().upper()
    gt_answer = gt_answer.strip().upper()
    
    return model_answer == gt_answer


def compute_score(model_response: str, gt_answer: str) -> dict:
    """
    Compute score for video QA multiple choice response.
    
    Args:
        model_response: Full model response text
        gt_answer: Ground truth answer (A, B, C, or D)
        
    Returns:
        Dictionary with score, format_score, acc, extracted_gt, pred
    """
    model_answer = extract_answer(model_response)
    
    if model_answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": "",
        }
    
    is_correct = grade(model_answer, gt_answer)
    
    if is_correct:
        return {
            "score": 1.0,
            "format_score": 1.0,
            "acc": True,
            "extracted_gt": gt_answer,
            "pred": model_answer,
        }
    else:
        return {
            "score": 0.0,
            "format_score": 1.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": model_answer,
        }


def reward_func(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """
    Reward function for TTRL video QA task.
    
    Args:
        data_source: Data source identifier (unused but required by interface)
        solution_str: Model's response text
        ground_truth: Ground truth answer (A, B, C, or D)
        extra_info: Optional extra information (unused)
        sandbox_fusion_url: Optional sandbox URL (unused)
        concurrent_semaphore: Optional semaphore (unused)
        
    Returns:
        Dictionary with score information or float score
    """
    try:
        res = compute_score(solution_str, str(ground_truth))
        
        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in reward_func for video QA: {str(e)}")
        traceback.print_exc()
        raise
