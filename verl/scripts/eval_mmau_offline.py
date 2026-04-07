#!/usr/bin/env python3
"""Offline evaluation of MMAU checkpoints from TTRL training.

Loads FSDP-sharded checkpoints, merges them in memory, runs generation
on the MMAU test set, and reports accuracy per checkpoint step.

Usage:
    python verl/scripts/eval_mmau_offline.py \
        --ckpt-dir /path/to/TTRL-MMAU-grpo-.../  \
        --test-file verl/data/MMAU/test_mini_test.json \
        --base-model /data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B \
        --steps 5 50 100 150 195 \
        --output results_mmau.csv
"""

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True,
                    help="Base checkpoint dir containing global_step_* subdirs")
    p.add_argument("--test-file", required=True,
                    help="MMAU test JSON file")
    p.add_argument("--base-model", default="/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B",
                    help="Base Qwen2.5-Omni model path (for processor + architecture)")
    p.add_argument("--steps", nargs="*", type=int, default=None,
                    help="Specific steps to evaluate (default: all)")
    p.add_argument("--output", default="results_mmau.csv",
                    help="Output CSV file")
    p.add_argument("--batch-size", type=int, default=1,
                    help="Batch size for generation")
    p.add_argument("--max-new-tokens", type=int, default=512,
                    help="Max new tokens for generation")
    p.add_argument("--sample-rate", type=int, default=16000,
                    help="Audio sample rate")
    p.add_argument("--max-audio-duration", type=float, default=30.0,
                    help="Max audio duration in seconds")
    p.add_argument("--suffix-prompt", type=str,
                    default="\nExplain your reasoning step by step in detail, then give your final answer as exactly one of: \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}.",
                    help="Suffix prompt appended to each question")
    p.add_argument("--video-fps", type=float, default=1.0,
                    help="Video frame sampling rate")
    p.add_argument("--video-max-frames", type=int, default=32,
                    help="Max video frames")
    p.add_argument("--use-audio-in-video", action="store_true",
                    help="Extract audio from video files (for OmniVideo)")
    p.add_argument("--n-samples", type=int, default=None,
                    help="Randomly subsample N test samples (for large datasets)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for subsampling")
    p.add_argument("--eval-baseline", action="store_true",
                    help="Also evaluate the base model (step 0)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Answer extraction / scoring (matches ttrl_video_qa reward function)
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> str:
    if not response:
        return None
    boxed_match = re.search(r'\\boxed\{([A-Da-d])\}', response)
    if boxed_match:
        return boxed_match.group(1).upper()
    return None


def score_response(response: str, gt_answer: str) -> dict:
    pred = extract_answer(response)
    format_ok = pred is not None
    if pred is None:
        pred = "NONE"
    is_correct = pred.strip().upper() == gt_answer.strip().upper()
    return {
        "correct": is_correct,
        "format_ok": format_ok,
        "pred": pred,
        "gt": gt_answer,
    }


# ---------------------------------------------------------------------------
# FSDP shard merging (adapted from verl.model_merger.fsdp_model_merger)
# ---------------------------------------------------------------------------

def merge_fsdp_shards(actor_dir: str) -> dict:
    """Load FSDP shards and merge into a single state_dict on CPU."""
    from concurrent.futures import ThreadPoolExecutor

    actor_dir = Path(actor_dir)
    fsdp_config = json.load(open(actor_dir / "fsdp_config.json"))
    world_size = fsdp_config["world_size"]

    shards = [None] * world_size

    def load_shard(rank):
        path = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        sd = torch.load(path, map_location="cpu", weights_only=False)
        shards[rank] = sd

    with ThreadPoolExecutor(max_workers=min(4, world_size)) as ex:
        list(ex.map(load_shard, range(world_size)))

    # Check if DTensor
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor

    merged = {}
    keys = list(shards[0].keys())
    for key in keys:
        tensors = []
        placement = None
        for rank in range(world_size):
            t = shards[rank].pop(key)
            if isinstance(t, DTensor):
                tensors.append(t._local_tensor.to(torch.bfloat16))
                if placement is None:
                    placement = t.placements[-1]  # last dim is FSDP shard dim
            else:
                tensors.append(t.to(torch.bfloat16))

        if placement is not None and hasattr(placement, 'is_shard') and placement.is_shard():
            merged[key] = torch.cat(tensors, dim=placement.dim).contiguous()
        elif placement is not None and hasattr(placement, 'is_replicate') and placement.is_replicate():
            merged[key] = tensors[0]
        else:
            # fallback: concatenate along dim 0
            merged[key] = torch.cat(tensors, dim=0).contiguous()

    del shards
    gc.collect()
    return merged


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_with_checkpoint(base_model_path: str, state_dict: dict = None):
    """Load Qwen2.5-Omni full model, extract thinker, optionally load checkpoint weights."""
    from transformers import AutoProcessor, AutoConfig

    # Load the full Omni model class (Qwen2_5OmniForConditionalGeneration)
    # which is what fsdp_workers.py uses
    from transformers import Qwen2_5OmniForConditionalGeneration

    print("  Loading processor...")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    print("  Loading base model...")
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    config.enable_audio_output = False
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        config=config,
        trust_remote_code=True,
    )
    model.disable_talker()
    thinker = model.thinker

    if state_dict is not None:
        print(f"  Loading checkpoint weights ({len(state_dict)} params)...")
        missing, unexpected = thinker.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

    thinker = thinker.to("cuda").eval()

    # Free the outer model shell (talker already disabled)
    gc.collect()
    torch.cuda.empty_cache()

    return thinker, processor


# ---------------------------------------------------------------------------
# Media loading
# ---------------------------------------------------------------------------

def load_audio(audio_path: str, target_sr: int = 16000, max_duration: float = 30.0):
    """Load and resample audio file, truncate to max_duration (matches training code)."""
    import librosa
    waveform, _ = librosa.load(audio_path, sr=target_sr)
    max_samples = int(max_duration * target_sr)
    waveform = waveform[:max_samples]
    # Pad very short audio to avoid avg_pool1d crash in audio_tower
    min_samples = int(target_sr * 1.0)
    if len(waveform) < min_samples:
        waveform = np.pad(waveform, (0, min_samples - len(waveform)))
    return waveform, target_sr


def load_video(video_path: str, fps: float = 1.0, max_frames: int = 32):
    """Load video frames using VERL's vision_utils (matches training code)."""
    from verl.utils.dataset.vision_utils import process_video
    video_tensor = process_video(
        {"video": video_path},
        fps=fps,
        fps_max_frames=max_frames,
    )
    return video_tensor.numpy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(thinker, processor, test_data, args):
    """Run generation on all test samples and compute accuracy."""
    results = []
    correct = 0
    format_ok_count = 0
    total = len(test_data)

    for i, sample in enumerate(tqdm(test_data, desc="    Evaluating")):
        question = sample["question"]
        gt_answer = sample["answer"]
        audio_file = sample.get("audio_file", "")
        video_file = sample.get("video_file", "")
        # Fall back to extracting audio from video if no separate audio file
        if not audio_file and video_file and args.use_audio_in_video:
            audio_file = video_file

        # Build conversation message content
        content = []
        audios = None
        videos = None

        has_video = video_file and os.path.exists(video_file)
        has_audio = audio_file and os.path.exists(audio_file)

        if has_video:
            # Pre-process video with same fps/max_frames as training
            content.append({"type": "video", "video": video_file})
            try:
                video_np = load_video(video_file, fps=args.video_fps, max_frames=args.video_max_frames)
                videos = [video_np]
            except Exception as e:
                print(f"    WARNING: Failed to load video for sample {i}: {e}")
                has_video = False
                videos = None
            # Load audio alongside video when use_audio_in_video is set
            if has_audio and args.use_audio_in_video:
                try:
                    audio_data, sr = load_audio(audio_file, args.sample_rate, args.max_audio_duration)
                    audios = [audio_data]
                except Exception as e:
                    print(f"    WARNING: Failed to load audio for sample {i}: {e}")
                    audios = None
        elif has_audio:
            # Audio-only samples (MMAU): include audio in message
            try:
                audio_data, sr = load_audio(audio_file, args.sample_rate, args.max_audio_duration)
                content.append({"type": "audio", "audio": audio_data})
                audios = [audio_data]
            except Exception as e:
                print(f"    WARNING: Failed to load audio for sample {i}: {e}")
                audios = None
                has_audio = False

        content.append({"type": "text", "text": question + args.suffix_prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": content},
        ]

        # Process inputs — pass pre-processed numpy frames (not file paths)
        # to match training behavior in rl_omni_dataset.py
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if has_video and videos is not None:
                inputs = processor(
                    text=[text], videos=videos, audio=audios,
                    return_tensors="pt", padding=True,
                    use_audio_in_video=args.use_audio_in_video,
                )
            elif audios is not None:
                inputs = processor(
                    text=[text], audio=audios, return_tensors="pt", padding=True,
                    use_audio_in_video=False,
                )
            else:
                inputs = processor(text=[text], return_tensors="pt", padding=True)
        except Exception as e:
            import traceback
            print(f"    WARNING: Failed to process sample {i} ({sample.get('id',i)}): {e}")
            traceback.print_exc()
            results.append({"correct": False, "format_ok": False, "pred": "ERROR", "gt": gt_answer})
            continue

        # Diagnostic: confirm modalities in processor output
        if i < 3:
            _has_vid = any('video' in k or 'pixel_values_video' in k for k in inputs)
            _has_aud = any('input_features' in k or 'feature_attention_mask' in k for k in inputs)
            _shapes = {k: (v.shape if hasattr(v, 'shape') else type(v).__name__) for k, v in inputs.items()}
            print(f"    [MM_VERIFY] video={_has_vid} audio={_has_aud} | keys={_shapes}")

        # Move to GPU
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            try:
                output_ids = thinker.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    use_cache=True,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"    WARNING: OOM for sample {i} ({sample.get('id',i)}), skipping")
                results.append({"correct": False, "format_ok": False, "pred": "OOM", "gt": gt_answer})
                continue
            except Exception as e:
                import traceback
                print(f"    WARNING: Generation failed for sample {i} ({sample.get('id',i)}): {e}")
                traceback.print_exc()
                results.append({"correct": False, "format_ok": False, "pred": "ERROR", "gt": gt_answer})
                continue

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        response_ids = output_ids[0, input_len:]
        response = processor.decode(response_ids, skip_special_tokens=True)

        # Score
        result = score_response(response, gt_answer)
        results.append(result)

        if result["correct"]:
            correct += 1
        if result["format_ok"]:
            format_ok_count += 1

        # Sanity check: print full details for first few samples
        if i < 3:
            print(f"\n    === SANITY CHECK sample {i} ===")
            print(f"    ID: {sample.get('id', i)}")
            print(f"    Question: {question[:200]}")
            print(f"    GT answer: {gt_answer}")
            print(f"    Input tokens: {inputs['input_ids'].shape[1]}")
            print(f"    Response tokens: {len(response_ids)}")
            print(f"    --- FULL RESPONSE ---")
            print(f"    {response}")
            print(f"    --- END RESPONSE ---")
            print(f"    Extracted pred: {result['pred']} | Format OK: {result['format_ok']} | Correct: {result['correct']}")
        elif (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"    [{i+1}/{total}] running_acc={correct/(i+1):.3f} | gt={gt_answer} pred={result['pred']} correct={result['correct']}")

    accuracy = correct / total if total > 0 else 0
    format_rate = format_ok_count / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "format_rate": format_rate,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load test data
    with open(args.test_file) as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples from {args.test_file}")

    # Subsample if requested
    if args.n_samples and args.n_samples < len(test_data):
        import random as _rng
        _rng.seed(args.seed)
        test_data = _rng.sample(test_data, args.n_samples)
        print(f"Subsampled to {len(test_data)} samples (seed={args.seed})")

    # Find checkpoints
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
        # Remove step 0 from checkpoint list if present, then add baseline
        step_dirs = [(s, d) for s, d in step_dirs if s != 0]
        step_dirs.insert(0, (0, None))  # baseline = base model

    print(f"Will evaluate {len(step_dirs)} checkpoints: {[s for s, _ in step_dirs]}")

    # Prepare output CSV
    csv_path = Path(args.output)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "accuracy", "format_rate", "correct", "total", "time_s"])
    csv_file.flush()

    for step, step_dir in step_dirs:
        print(f"\n{'='*60}")
        print(f"Step {step}")
        print(f"{'='*60}")
        t0 = time.time()

        # Merge FSDP shards
        state_dict = None
        if step_dir is not None:
            actor_dir = step_dir / "actor"
            if not actor_dir.exists():
                print(f"  SKIP: {actor_dir} not found")
                continue
            print(f"  Merging FSDP shards from {actor_dir}...")
            state_dict = merge_fsdp_shards(str(actor_dir))
            print(f"  Merged {len(state_dict)} params")

        # Load model
        thinker, processor = load_model_with_checkpoint(args.base_model, state_dict)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate
        metrics = evaluate_checkpoint(thinker, processor, test_data, args)

        elapsed = time.time() - t0
        print(f"\n  Step {step}: accuracy={metrics['accuracy']:.4f} "
              f"format_rate={metrics['format_rate']:.4f} "
              f"({metrics['correct']}/{metrics['total']}) "
              f"time={elapsed:.1f}s")

        writer.writerow([
            step,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['format_rate']:.4f}",
            metrics['correct'],
            metrics['total'],
            f"{elapsed:.1f}",
        ])
        csv_file.flush()

        # Cleanup
        del thinker, processor
        gc.collect()
        torch.cuda.empty_cache()

    csv_file.close()
    print(f"\nResults saved to {csv_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Step':>6} | {'Accuracy':>8} | {'Format':>6} | {'Correct':>7}/{' Total'}")
    print(f"{'-'*60}")
    csv_file = open(csv_path, "r")
    reader = csv.DictReader(csv_file)
    for row in reader:
        print(f"{row['step']:>6} | {row['accuracy']:>8} | {row['format_rate']:>6} | "
              f"{row['correct']:>7}/{row['total']:>6}")
    csv_file.close()


if __name__ == "__main__":
    main()
