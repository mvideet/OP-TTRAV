#!/usr/bin/env python3
"""
Oneshot pseudo-labeling for OmniVideo SFT dataset.

For each question, runs a single inference with Qwen2.5-Omni,
parses the answer letter (A/B/C/D) + option text, and formats
the result as a LlamaFactory-compatible SFT dataset.

Usage:
    python oneshot.py \
        --model_path /data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B \
        --data_path  /data/sls/r/u/mvideet/TTRL/verl/data/OmniVideo/train.json \
        --output_path ./data/omnivideo_oneshot_sft.json \
        --num_gpus 4
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# Answer / option helpers (shared with majority_voting.py)
# ---------------------------------------------------------------------------

def parse_options(question: str) -> dict[str, str]:
    """Parse multiple-choice options from question text.

    Returns e.g. {"A": "Sustained rapid gunfire", "B": "Speech describing ..."}
    """
    pattern = r'\b([A-D])\.\s*(.+?)(?=\n[A-D]\.|$)'
    matches = re.findall(pattern, question, re.DOTALL)
    return {letter.upper(): text.strip() for letter, text in matches}


def extract_answer(response: str) -> str | None:
    """Extract the answer choice (A-D) from a model response."""
    if not response:
        return None
    response = response.strip()

    # \\boxed{X}
    m = re.search(r'\\boxed\{([A-Da-d])\}', response)
    if m:
        return m.group(1).upper()
    # "answer/choice/option is X"
    m = re.search(r'(?:answer|choice|option)\s*(?:is|:)\s*[(\[]?\s*([A-Da-d])\s*[)\]]?', response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # "X." or "X:" at start
    m = re.match(r'^([A-Da-d])\s*[.:)\]]\s*', response)
    if m:
        return m.group(1).upper()
    # (X) at end
    m = re.search(r'[(\[]\s*([A-Da-d])\s*[)\]]\s*$', response)
    if m:
        return m.group(1).upper()
    # X. at end
    m = re.search(r'\b([A-Da-d])\s*[.):]\s*$', response)
    if m:
        return m.group(1).upper()
    # first standalone letter in first 50 chars
    first = response[:50]
    m = re.search(r'\b([A-Da-d])\b', first)
    if m:
        return m.group(1).upper()
    # last standalone letter in last 100 chars
    last = response[-100:]
    m = re.search(r'\b([A-Da-d])\b', last)
    if m:
        return m.group(1).upper()
    return None


def format_answer(letter: str | None, options: dict[str, str], raw_response: str) -> str:
    """Format answer as 'B. option text'. Falls back to raw response."""
    if letter and letter in options:
        return f"{letter}. {options[letter]}"
    if letter:
        return f"{letter}."
    # No parseable letter – return raw response (truncated)
    return raw_response.strip()[:512]


# ---------------------------------------------------------------------------
# SFT output helpers
# ---------------------------------------------------------------------------

def build_sft_entry(item: dict, assistant_response: str) -> dict:
    """Build a LlamaFactory sharegpt-style SFT entry with video + audio."""
    question = item["question"]
    video_file = item.get("video_file", "")
    audio_file = item.get("audio_file", "")

    entry = {
        "messages": [
            {"role": "user",      "content": f"<video>{question}"},
            {"role": "assistant", "content": assistant_response},
        ],
        "videos": [video_file],
    }
    # Include audio path if available
    if audio_file:
        entry["messages"][0]["content"] = f"<video><audio>{question}"
        entry["audios"] = [audio_file]

    return entry


# ---------------------------------------------------------------------------
# Inference worker (one per GPU)
# ---------------------------------------------------------------------------

def worker(gpu_id: int, shard: list[dict], args, output_file: str):
    """Run inference on a single GPU for a shard of the dataset."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    print(f"[GPU {gpu_id}] Loading model from {args.model_path} ...")
    from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info

    # Fix: talker config may be missing pad_token_id in some transformers versions
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(config, "talker_config"):
        tc = config.talker_config
        if not hasattr(tc, "pad_token_id") or tc.pad_token_id is None:
            tc.pad_token_id = getattr(config, "pad_token_id", 0) or 0

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if not args.no_flash_attn else "eager",
    )
    # We only need text output; drop the talker to save GPU memory
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model.eval()
    print(f"[GPU {gpu_id}] Model loaded. Processing {len(shard)} items ...")

    results = []
    for idx, item in enumerate(shard):
        try:
            # Build messages in Qwen2.5-Omni format
            messages = [
                {"role": "user", "content": [
                    {"type": "video", "video": item["video_file"]},
                    {"type": "text",  "text": item["question"]},
                ]}
            ]

            text_prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(
                messages, use_audio_in_video=args.use_audio_in_video
            )
            inputs = processor(
                text=text_prompt,
                audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True,
                use_audio_in_video=args.use_audio_in_video,
            )
            inputs = inputs.to(device).to(model.dtype)

            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    use_audio_in_video=args.use_audio_in_video,
                    return_audio=False,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature if args.temperature > 0 else None,
                    do_sample=args.temperature > 0,
                )

            # Strip the prompt tokens -- only decode the generated part
            generated_ids = output_ids[:, input_len:]
            response = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Parse answer
            options = parse_options(item["question"])
            letter = extract_answer(response)
            formatted = format_answer(letter, options, response)

            sft_entry = build_sft_entry(item, formatted)
            # Attach metadata for debugging / filtering
            sft_entry["_meta"] = {
                "id": item.get("id", ""),
                "gt_answer": item.get("answer", ""),
                "pred_letter": letter,
                "raw_response": response[:500],
            }
            results.append(sft_entry)

        except Exception as e:
            print(f"[GPU {gpu_id}] Error on item {idx} ({item.get('id','')}): {e}")
            traceback.print_exc()
            continue

        if (idx + 1) % 50 == 0 or idx == len(shard) - 1:
            print(f"[GPU {gpu_id}] {idx+1}/{len(shard)} done")
            # Incremental save so we can resume
            with open(output_file, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # Final save
    with open(output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[GPU {gpu_id}] Finished. {len(results)} entries saved to {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Oneshot pseudo-label generation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path",  type=str, required=True, help="OmniVideo JSON")
    parser.add_argument("--output_path", type=str, default="./data/omnivideo_oneshot_sft.json")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--use_audio_in_video", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N items (for testing)")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading data from {args.data_path} ...")
    with open(args.data_path) as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]
    print(f"Loaded {len(data)} items.")

    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    # Split into shards
    num_gpus = min(args.num_gpus, len(data))
    shards = [data[i::num_gpus] for i in range(num_gpus)]
    shard_files = [
        args.output_path.replace(".json", f"_shard{i}.json") for i in range(num_gpus)
    ]

    print(f"Launching {num_gpus} workers ...")
    mp.set_start_method("spawn", force=True)

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, shards[gpu_id], args, shard_files[gpu_id]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge shards (interleave back to original order)
    print("Merging shards ...")
    shard_data = []
    for sf in shard_files:
        if os.path.exists(sf):
            with open(sf) as f:
                shard_data.append(json.load(f))
        else:
            shard_data.append([])

    # Interleave: shard 0 has indices [0, num_gpus, 2*num_gpus, ...], etc.
    merged = []
    max_len = max(len(s) for s in shard_data) if shard_data else 0
    for j in range(max_len):
        for i in range(num_gpus):
            if j < len(shard_data[i]):
                merged.append(shard_data[i][j])

    with open(args.output_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Clean up shard files
    for sf in shard_files:
        if os.path.exists(sf):
            os.remove(sf)

    # Print summary
    pred_letters = [e.get("_meta", {}).get("pred_letter") for e in merged]
    gt_letters   = [e.get("_meta", {}).get("gt_answer") for e in merged]
    correct = sum(1 for p, g in zip(pred_letters, gt_letters) if p and g and p == g)
    parsed  = sum(1 for p in pred_letters if p is not None)

    print(f"\n=== Summary ===")
    print(f"Total: {len(merged)}")
    print(f"Parsed answer: {parsed}/{len(merged)} ({100*parsed/max(len(merged),1):.1f}%)")
    print(f"Correct vs GT: {correct}/{len(merged)} ({100*correct/max(len(merged),1):.1f}%)")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
