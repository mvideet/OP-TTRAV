#!/usr/bin/env python3
"""
Dump rollouts for offline re-judging.

For each (checkpoint, test_sample), generates eval_n rollouts and writes
them to JSONL. NO judging — just the raw rollouts. Cheaper and faster
than eval_open_ended_judge.py since it skips the per-rollout judge call.

Output JSONL schema (one record per sample, per checkpoint):
  {
    "step": int,                 # 0 for baseline, ckpt step otherwise
    "id": str,                   # sample id from dataset
    "category": str,             # auto-detected from sample
    "question": str,
    "gold": str,                 # answer_text or configured gold key
    "rollouts": [str, ...],      # length = eval_n
    "rollout_temperature": float,
    "rollout_top_p": float,
    "rollout_do_sample": bool,
  }

Usage:
  python verl/scripts/dump_rollouts.py \\
      --ckpt-dir /data/.../saved/judge_v2_0430 \\
      --test-file verl/data/OmniVideo/test_open.json \\
      --base-model /data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B \\
      --steps 200 --eval-baseline \\
      --n-samples 100 --eval-n 1 \\
      --output rollouts_judge_v2.jsonl

Re-judging downstream: read the JSONL, run any judge (local model, API,
human, etc.) over `rollouts` against `gold`. See judge_rollouts_jsonl.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from eval_mmau_offline import (  # noqa: E402
    merge_fsdp_shards,
    load_model_with_checkpoint,
    _prepare_inputs,
    _generate_one,
)


def _is_text_only_model(base_model_path: str) -> bool:
    """Detect non-Omni text-only base models (e.g. Qwen2.5-Math-1.5B)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    return not hasattr(cfg, "thinker_config")


def load_text_model_with_checkpoint(base_model_path: str, state_dict: dict | None = None):
    """Load AutoModelForCausalLM + tokenizer for text-only models."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    print("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if state_dict is not None:
        print(f"  Loading checkpoint weights ({len(state_dict)} params)...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}...")
    return model.to("cuda").eval(), tokenizer


def _prepare_text_inputs(sample, tokenizer, args, sample_idx):
    """Build tokenized inputs for one text-only sample."""
    question = sample.get("question") or sample.get("prompt") or ""
    gt_answer = sample["answer"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question + args.suffix_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    if sample_idx < 3:
        print(f"    [TEXT_VERIFY] input_ids={inputs['input_ids'].shape}")
    return inputs, gt_answer


def _generate_text_one(model, inputs, tokenizer, args, do_sample, temperature):
    """Generate one text response."""
    prompt_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        try:
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=args.eval_top_p if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        except Exception:
            return None
    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--base-model", default="/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B")
    p.add_argument("--steps", nargs="*", type=int, default=None)
    p.add_argument("--output", default="rollouts.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--max-audio-duration", type=float, default=30.0)
    p.add_argument("--video-fps", type=float, default=1.0)
    p.add_argument("--video-max-frames", type=int, default=32)
    p.add_argument("--use-audio-in-video", action="store_true")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-baseline", action="store_true")
    p.add_argument("--eval-n", type=int, default=1)
    p.add_argument("--eval-temperature", type=float, default=0.6)
    p.add_argument("--eval-top-p", type=float, default=0.95)
    p.add_argument("--suffix-prompt", type=str,
                    default="\nExplain your reasoning step by step, then give a concise answer to the question in 1-3 complete sentences.")
    p.add_argument("--gold-key", type=str, default="answer_text")
    p.add_argument("--category-key", type=str, default=None)
    p.add_argument("--text-only", action="store_true",
                   help="Force text-only mode (AutoModelForCausalLM); auto-detected from base-model config otherwise.")
    return p.parse_args()


def dump_rollouts_for_checkpoint(thinker, processor, test_data, args, jsonl_writer, step, text_only=False):
    cat_key = args.category_key
    if cat_key is None and test_data:
        for k in ["question_type", "source", "content_parent_category"]:
            if k in test_data[0]:
                cat_key = k
                break

    do_sample = args.eval_n > 1
    desc = f"  rollouts step={step} (n={args.eval_n} {'sampled' if do_sample else 'greedy'})"
    n_dumped = 0

    for i, raw in enumerate(tqdm(test_data, desc=desc)):
        sample = dict(raw)
        gold = sample.get(args.gold_key, "") or ""
        sample["answer"] = gold
        category = (sample.get(cat_key, "unknown") if cat_key else "unknown")
        if isinstance(category, str):
            category = category.strip().title().replace("Av ", "AV ")
        question = sample.get("question") or sample.get("prompt") or ""

        if text_only:
            inputs, _ = _prepare_text_inputs(sample, processor, args, i)
            gen_fn = _generate_text_one
        else:
            inputs, _ = _prepare_inputs(sample, processor, args, i)
            gen_fn = _generate_one
        rollouts = []
        if inputs is None:
            # write a record with empty rollouts so downstream can skip
            pass
        elif args.eval_n <= 1:
            r = gen_fn(thinker, inputs, processor, args, do_sample=False, temperature=1.0)
            if r is not None:
                rollouts.append(r)
        else:
            for _ in range(args.eval_n):
                r = gen_fn(thinker, inputs, processor, args,
                           do_sample=True, temperature=args.eval_temperature)
                if r is not None:
                    rollouts.append(r)

        record = {
            "step": step,
            "id": sample.get("id", i),
            "category": category,
            "question": question,
            "gold": gold,
            "rollouts": rollouts,
            "rollout_temperature": args.eval_temperature if do_sample else 0.0,
            "rollout_top_p": args.eval_top_p if do_sample else 1.0,
            "rollout_do_sample": do_sample,
        }
        jsonl_writer.write(json.dumps(record, ensure_ascii=False) + "\n")
        jsonl_writer.flush()
        n_dumped += 1

        if i < 3:
            print(f"\n  === SANITY sample {i} step={step} ===")
            print(f"  id: {sample.get('id', i)}")
            print(f"  question: {question[:200]}")
            print(f"  gold:     {gold[:200]}")
            print(f"  rollout0: {(rollouts[0] if rollouts else '<EMPTY>')[:200]}")
        elif (i + 1) % 25 == 0:
            print(f"  [step={step}] {i+1}/{len(test_data)} dumped", flush=True)

    return n_dumped


def main():
    args = parse_args()

    text_only = args.text_only or _is_text_only_model(args.base_model)
    if text_only:
        print(f"Text-only mode (base model has no thinker_config).")

    with open(args.test_file) as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples from {args.test_file}")

    if args.n_samples and args.n_samples < len(test_data):
        import random as _rng
        _rng.seed(args.seed)
        test_data = _rng.sample(test_data, args.n_samples)
        print(f"Subsampled to {len(test_data)} samples (seed={args.seed})")

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
        step_dirs = [(s, d) for s, d in step_dirs if s != 0]
        step_dirs.insert(0, (0, None))

    print(f"Will dump rollouts for {len(step_dirs)} checkpoints: {[s for s, _ in step_dirs]}")

    jsonl_path = Path(args.output)
    jsonl_writer = open(jsonl_path, "w")

    for step, step_dir in step_dirs:
        print(f"\n{'=' * 60}")
        print(f"Step {step}")
        print(f"{'=' * 60}")
        t0 = time.time()

        state_dict = None
        if step_dir is not None:
            actor_dir = step_dir / "actor"
            if not actor_dir.exists():
                print(f"  SKIP: {actor_dir} not found")
                continue
            print(f"  Merging FSDP shards from {actor_dir}...")
            state_dict = merge_fsdp_shards(str(actor_dir))
            print(f"  Merged {len(state_dict)} params")

        if text_only:
            thinker, processor = load_text_model_with_checkpoint(args.base_model, state_dict)
        else:
            thinker, processor = load_model_with_checkpoint(args.base_model, state_dict)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        n_dumped = dump_rollouts_for_checkpoint(
            thinker, processor, test_data, args, jsonl_writer, step, text_only=text_only,
        )

        elapsed = time.time() - t0
        print(f"  Step {step}: {n_dumped} rollouts dumped, time={elapsed:.1f}s")

        del thinker, processor
        gc.collect()
        torch.cuda.empty_cache()

    jsonl_writer.close()
    print(f"\nRollouts written to {jsonl_path}")
    print(f"Records: {sum(1 for _ in open(jsonl_path))}")


if __name__ == "__main__":
    main()
