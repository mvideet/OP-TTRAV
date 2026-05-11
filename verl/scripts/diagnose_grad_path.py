"""
Diagnose audio-visual training path on a single A5 GPU.

Three independent tests. Each gets its own fresh CUDA state so a failure
in one can't cascade.

  T1 — forward determinism on multimodal input
       Same input twice → identical logits (within bf16 noise). Catches any
       hidden state / RNG / KV-cache leakage in the multimodal forward.

  T2 — gradients flow & params actually move element-wise
       Forward + backward + optimizer.step on a small fixed loss. Verifies:
         (a) no NaN/Inf in gradients
         (b) per-component grad norms are non-zero where expected
         (c) param TENSORS (not just .norm) actually change element-wise
         (d) log-prob on the same input visibly changes after the step

  T3 — response-mask vs prompt-mask gradient norms
       Loss masked to last 64 tokens ("response") produces non-zero,
       finite gradient. Same model under prompt-only mask should also
       produce a finite gradient. This catches dead branches in the loss
       computation.

Usage:
  sbatch slurm/diag_grad.sh
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def banner(s: str):
    bar = "=" * 70
    print(f"\n{bar}\n{s}\n{bar}", flush=True)


def report(name: str, ok: bool, detail: str = ""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""), flush=True)


def load_model(base_model_path: str):
    from transformers import AutoConfig, AutoProcessor, Qwen2_5OmniForConditionalGeneration

    print(f"Loading processor from {base_model_path}...", flush=True)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"Loading model (bf16)...", flush=True)
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    config.enable_audio_output = False
    full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, config=config, trust_remote_code=True,
    )
    full_model.disable_talker()
    thinker = full_model.thinker.to("cuda")
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    # Gradient checkpointing — required to fit a 3B Omni model + bf16
    # activations + AdamW state on a single 24GB A5. Mirrors the training
    # config (`enable_gradient_checkpointing=True`).
    try:
        thinker.gradient_checkpointing_enable()
        thinker.config.use_cache = False
        print("  gradient checkpointing enabled", flush=True)
    except Exception as _e:
        print(f"  gradient checkpointing not available: {_e}", flush=True)
    return thinker, processor


def build_one_sample(processor, video_path: str, audio_path: str, question: str,
                     audio_sr: int = 8000, video_fps: float = 0.5):
    import librosa
    from verl.utils.dataset.vision_utils import process_video

    print(f"  loading video {video_path} ...", flush=True)
    video_tensor = process_video({"video": video_path}, fps=video_fps, fps_max_frames=32)
    video_np = video_tensor.numpy()

    print(f"  loading audio {audio_path} ...", flush=True)
    waveform, _ = librosa.load(audio_path, sr=audio_sr)
    waveform = waveform[: int(audio_sr * 30.0)]
    if len(waveform) < int(audio_sr * 1.0):
        waveform = np.pad(waveform, (0, int(audio_sr * 1.0) - len(waveform)))

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": question},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        videos=[video_np],
        audio=[waveform],
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    return inputs


def move_to(d: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def split_for_thinker(inputs: Dict[str, Any], device: str):
    base_ids = inputs["input_ids"].to(device)
    base_mask = inputs["attention_mask"].to(device)
    mm_kwargs = move_to({k: v for k, v in inputs.items()
                          if k not in ("input_ids", "attention_mask")}, device)
    mm_kwargs["use_audio_in_video"] = torch.tensor(True, device=device)
    return base_ids, base_mask, mm_kwargs


# ---------------------------------------------------------------------------
# T1 — forward determinism
# ---------------------------------------------------------------------------


def t1_determinism(thinker, sample_inputs):
    banner("T1: forward determinism on multimodal input")
    device = next(thinker.parameters()).device
    base_ids, base_mask, mm_kwargs = split_for_thinker(sample_inputs, device)

    thinker.eval()
    with torch.no_grad():
        out1 = thinker(input_ids=base_ids, attention_mask=base_mask, **mm_kwargs, return_dict=True)
        logits1 = out1.logits.float()
        out2 = thinker(input_ids=base_ids, attention_mask=base_mask, **mm_kwargs, return_dict=True)
        logits2 = out2.logits.float()

    diff = (logits1 - logits2).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    print(f"  logits shape: {tuple(logits1.shape)}", flush=True)
    print(f"  max|Δ|={max_d:.4e}, mean|Δ|={mean_d:.4e}", flush=True)
    # bf16 + no_grad two forwards should be bit-identical or near so.
    ok = max_d < 1e-3
    report("forward is deterministic (same input → same logits)", ok,
           f"max|Δ|={max_d:.4e}  (tolerance 1e-3)")
    if not ok:
        print("  CONCERN: forward is non-deterministic. Could be dropout-in-eval, "
              "KV-cache leakage, or stochastic attention. Investigate.", flush=True)


# ---------------------------------------------------------------------------
# T2 — gradients + element-wise param movement + log-prob shift after step
# ---------------------------------------------------------------------------


def t2_grad_and_param_update(thinker, sample_inputs):
    banner("T2: gradient flow + element-wise param update + log-prob shift")
    device = next(thinker.parameters()).device
    base_ids, base_mask, mm_kwargs = split_for_thinker(sample_inputs, device)

    # Pick representative parameters to snapshot ELEMENT-WISE.
    candidate_keys = ["embed_tokens.weight", "lm_head.weight",
                       "layers.0.self_attn.q_proj", "layers.10.mlp.up_proj"]
    snapshots_before: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    thinker.train()
    optim = torch.optim.AdamW(
        [p for p in thinker.parameters() if p.requires_grad],
        lr=2e-6, betas=(0.9, 0.95),
    )

    # --- Snapshot params + run pre-step forward ---
    print("  snapshotting params + pre-step forward ...", flush=True)
    for name, p in thinker.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in candidate_keys) and len(snapshots_before) < 4:
            flat = p.data.flatten()
            n_sample = min(1024, flat.numel())
            # Use float64 to avoid float32 precision loss when numel >> 1e7.
            idx = torch.linspace(0, flat.numel() - 1, n_sample, dtype=torch.float64).long()
            idx = idx.clamp(0, flat.numel() - 1).to(flat.device)
            snapshots_before[name] = (idx, flat[idx].detach().float().cpu().clone())

    target_tok = base_ids[0, -1].item()
    with torch.no_grad():
        out_before = thinker(input_ids=base_ids, attention_mask=base_mask,
                              **mm_kwargs, return_dict=True)
        # Only keep the slice we need; free the rest before backward.
        last_slice = out_before.logits[0, -2, :].float().detach().clone()
    lp_before = torch.log_softmax(last_slice, dim=-1)[target_tok].item()
    del out_before, last_slice
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  log-prob of last token before step: {lp_before:.4f}", flush=True)

    # --- Backward step ---
    optim.zero_grad(set_to_none=True)
    print("  forward + backward ...", flush=True)
    try:
        out = thinker(input_ids=base_ids, attention_mask=base_mask, **mm_kwargs, return_dict=True)
        shift_logits = out.logits[:, :-1, :].float()
        shift_labels = base_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        print(f"  loss={loss.item():.4f}", flush=True)
        loss.backward()
    except Exception as e:
        report("forward+backward", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return

    # --- Grad diagnostics ---
    components: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    n_with_grad = 0
    n_zero_grad = 0
    nan_or_inf = 0
    total_grad_norm = 0.0
    for name, p in thinker.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        if torch.isnan(g).any() or torch.isinf(g).any():
            nan_or_inf += 1
            continue
        gn = g.norm().item()
        if gn == 0.0:
            n_zero_grad += 1
            continue
        n_with_grad += 1
        total_grad_norm += gn * gn
        for kw in ["embed", "lm_head", "visual", "audio", "self_attn", "mlp", "norm"]:
            if kw in name:
                components[kw] = components.get(kw, 0.0) + gn * gn
                counts[kw] = counts.get(kw, 0) + 1
                break
        else:
            components["other"] = components.get("other", 0.0) + gn * gn
            counts["other"] = counts.get("other", 0) + 1
    total_grad_norm = total_grad_norm ** 0.5
    print("  per-component grad L2 norms:", flush=True)
    for k in sorted(components):
        print(f"    {k:10s} n={counts[k]:4d}  ||grad||={components[k]**0.5:.4e}", flush=True)
    report("gradients populated", n_with_grad > 0,
           f"{n_with_grad} non-zero, {n_zero_grad} zero, {nan_or_inf} NaN/Inf, "
           f"total_grad_norm={total_grad_norm:.4f}")
    report("no NaN/Inf grads", nan_or_inf == 0)

    # --- Optimizer step ---
    print("  optimizer.step() ...", flush=True)
    optim.step()

    # --- Element-wise param-change verification ---
    print("  element-wise param-change check (sampling 1024 elements per tracked param):", flush=True)
    any_changed = False
    max_overall = 0.0
    for name, (idx, before_vals) in snapshots_before.items():
        p = dict(thinker.named_parameters())[name]
        after_vals = p.data.flatten()[idx].detach().float().cpu()
        diff = (after_vals - before_vals).abs()
        max_d = diff.max().item()
        mean_d = diff.mean().item()
        n_changed = int((diff > 0).sum().item())
        changed = n_changed > 0
        any_changed = any_changed or changed
        max_overall = max(max_overall, max_d)
        status = "PASS" if changed else "FAIL"
        print(f"    [{status}] {name}: {n_changed}/{idx.numel()} elements changed, "
              f"max|Δ|={max_d:.3e}, mean|Δ|={mean_d:.3e}", flush=True)
    report("params actually changed after optimizer.step()", any_changed,
           f"max element delta across tracked params: {max_overall:.3e}")

    # --- Post-step forward: log-prob should shift ---
    optim.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        out_after = thinker(input_ids=base_ids, attention_mask=base_mask,
                             **mm_kwargs, return_dict=True)
        last_slice = out_after.logits[0, -2, :].float().detach().clone()
    lp_after = torch.log_softmax(last_slice, dim=-1)[target_tok].item()
    del out_after, last_slice
    delta_lp = lp_after - lp_before
    print(f"  log-prob of last token after  step: {lp_after:.4f}  (Δ={delta_lp:+.4e})", flush=True)
    # We trained on next-token prediction, so log-prob of the actual continuation
    # token at this position should go UP (or at least not collapse).
    report("model output changed after optimizer step", abs(delta_lp) > 1e-5,
           f"|Δlog_prob|={abs(delta_lp):.4e}  (>1e-5 = real movement)")


# ---------------------------------------------------------------------------
# T3 — response-mask vs prompt-mask gradient sanity
# ---------------------------------------------------------------------------


def t3_mask_sanity(thinker, sample_inputs):
    banner("T3: response-vs-prompt mask sanity")
    device = next(thinker.parameters()).device
    base_ids, base_mask, mm_kwargs = split_for_thinker(sample_inputs, device)
    thinker.train()

    seq_len = base_ids.shape[1]
    resp_len = min(64, max(8, seq_len // 4))
    if seq_len <= resp_len + 4:
        report("mask sanity", False, f"seq_len={seq_len} too short for resp_len={resp_len}")
        return

    def grad_norm_under_mask(mask_label: str, loss_mask: torch.Tensor) -> Tuple[float, bool]:
        thinker.zero_grad(set_to_none=True)
        try:
            out = thinker(input_ids=base_ids, attention_mask=base_mask,
                          **mm_kwargs, return_dict=True)
            shift_logits = out.logits[:, :-1, :].float()
            shift_labels = base_ids[:, 1:]
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            per_tok_ll = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            mask = loss_mask[:, 1:].float()
            n_active = mask.sum().clamp(min=1)
            loss = -(per_tok_ll * mask).sum() / n_active
            loss.backward()
        except Exception as e:
            print(f"  [{mask_label}] forward/backward failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return 0.0, False

        total = 0.0
        nan_inf = False
        for name, p in thinker.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            if torch.isnan(g).any() or torch.isinf(g).any():
                nan_inf = True
                continue
            total += g.norm().item() ** 2
        return total ** 0.5, not nan_inf

    resp_mask = torch.zeros_like(base_mask)
    resp_mask[:, -resp_len:] = base_mask[:, -resp_len:]

    prompt_mask = base_mask.clone()
    prompt_mask[:, -resp_len:] = 0

    print(f"  response-only mask covers last {resp_len} tokens", flush=True)
    gn_resp, ok_resp = grad_norm_under_mask("response", resp_mask)
    gn_prompt, ok_prompt = grad_norm_under_mask("prompt", prompt_mask)
    print(f"  ||grad||_resp   = {gn_resp:.4e}", flush=True)
    print(f"  ||grad||_prompt = {gn_prompt:.4e}", flush=True)

    report("response-only mask produces finite non-zero grad", ok_resp and gn_resp > 1e-6,
           f"||grad||={gn_resp:.4e}")
    report("prompt-only mask produces finite grad", ok_prompt,
           f"||grad||={gn_prompt:.4e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B")
    p.add_argument("--test-file", default="/data/sls/u/urop/mvideet/TTRL/verl/data/OmniVideo/test_open_val20.json")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--skip-t1", action="store_true")
    p.add_argument("--skip-t2", action="store_true")
    p.add_argument("--skip-t3", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(True)

    with open(args.test_file) as f:
        test_data = json.load(f)
    sample = test_data[args.sample_index]
    print(f"Using sample {args.sample_index}: id={sample.get('id', '?')}", flush=True)
    print(f"  video: {sample.get('video_file')}", flush=True)

    thinker, processor = load_model(args.base_model)

    video_path = sample.get("video_file")
    audio_path = sample.get("audio_file") or video_path
    question = sample.get("question", "Describe this video.")
    sample_inputs = build_one_sample(processor, video_path, audio_path, question)
    print(f"  input_ids shape: {sample_inputs['input_ids'].shape}", flush=True)
    print(f"  mm keys: {[k for k in sample_inputs if k not in ('input_ids', 'attention_mask')]}",
          flush=True)

    tests = [
        ("T1", args.skip_t1, lambda: t1_determinism(thinker, sample_inputs)),
        ("T2", args.skip_t2, lambda: t2_grad_and_param_update(thinker, sample_inputs)),
        ("T3", args.skip_t3, lambda: t3_mask_sanity(thinker, sample_inputs)),
    ]
    for tag, skip, fn in tests:
        if skip:
            continue
        try:
            fn()
        except Exception as e:
            print(f"{tag} crashed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        # Fresh CUDA state for next test
        gc.collect()
        torch.cuda.empty_cache()

    banner("DONE")


if __name__ == "__main__":
    main()
