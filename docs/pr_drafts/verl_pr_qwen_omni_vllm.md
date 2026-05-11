# PR Draft: volcengine/verl — enable vLLM rollout for Qwen2.5-Omni-3B (audio+video)

## Title

`[Feature] Qwen2.5-Omni vLLM rollout: end-to-end audio+video support with use_audio_in_video=True`

## Summary

Adds the missing pieces to make verl's vLLM rollout work for Qwen2.5-Omni-3B (the audio+video Omni thinker), achieving ~30% step-time speedup vs the existing HF rollout fallback on a batched FSDP+GRPO+TTRL configuration with N=16 rollouts and `data.use_audio_in_video=True`.

Closes/supersedes #3241 (which never merged) and is the audio+video analog of the in-progress #6277 for Qwen3-Omni Thinker.

## What works after this PR

End-to-end:
- `actor_rollout_ref.rollout.name=vllm` with Qwen2.5-Omni-3B
- N=16 rollouts/prompt, `data.use_audio_in_video=True`
- video+audio interleaved tokens (M-RoPE)
- FSDP + vLLM coexistence on a single node, 4 GPUs
- TTRL voting + GRPO actor update via the existing recipe

Smoke test recipe (added in this PR): `slurm/test_vllm_omni.sh` runs 5 training steps end-to-end and saves the step-5 checkpoint.

## Required vLLM patches (separate vllm-project/vllm PRs)

This verl PR depends on three vLLM-side fixes for Qwen2.5-Omni's `use_audio_in_video=True` codepath. Track them in:
- vllm-project/vllm#XXXX — `merge_interleaved_embeddings` greedy categorization fix
- vllm-project/vllm#XXXX — `_apply_hf_processor_main` dummy-text bypass for tokenized prompts
- vllm-project/vllm#XXXX — `_maybe_apply_prompt_updates` `use_audio_in_video` detection from cached splits

Until those land, users need to apply the patches as a local override (we'll document this in `docs/qwen2_5_omni_vllm.md`).

## Verl-side changes

### 1. `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

**a) Forward `use_audio_in_video` per-request to `mm_processor_kwargs`**

Mirrors the existing pattern in `verl/workers/rollout/hf_rollout.py:111`. When the dataset stashes `use_audio_in_video` inside `non_tensor_batch["multi_modal_inputs"][i]`, attach it as `mm_processor_kwargs={"use_audio_in_video": True}` on each vLLM request:

```python
mm_proc_kwargs_list: List[dict] = []
if "multi_modal_inputs" in non_tensor_batch:
    for inp in non_tensor_batch.get("multi_modal_inputs"):
        kw = {}
        if isinstance(inp, dict) and "use_audio_in_video" in inp:
            kw["use_audio_in_video"] = bool(inp["use_audio_in_video"])
        mm_proc_kwargs_list.append(kw)

# Per-request:
item = {"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data}
if mm_proc_kwargs_list[i]:
    item["mm_processor_kwargs"] = mm_proc_kwargs_list[i]
```

**b) Repeat `multi_modal_inputs` for `n>1` sampling**

Existing code repeats `tools_kwargs`, `interaction_kwargs`, `raw_prompt` when `sampling_params.n > 1`, but skips `multi_modal_inputs`. Result: bs×n batch with multi_modal_inputs of length bs → `DataProto` validation fails. Add the missing repeat:

```python
if "multi_modal_inputs" in non_tensor_batch.keys():
    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
        non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
    )
```

**c) Drop multimodal keys from returned `non_tensor_batch`**

The return path was including `multi_modal_inputs` / `multi_modal_data` / `raw_prompt_ids` in the gen_batch_output's non_tensor_batch. The trainer's later `batch.union(gen_batch_output)` then trips on object-dtype numpy arrays of dicts with tensor values (pandas tensor-equality assertion). The original prompts batch already carries these keys; gen output doesn't need to re-include them. Mirrors `hf_rollout.py` which returns `DataProto(batch=batch)` only.

```python
for _mm_key in ("multi_modal_inputs", "multi_modal_data", "raw_prompt_ids"):
    non_tensor_batch.pop(_mm_key, None)
return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
```

**d) Qwen2.5-Omni nested config support**

Omni's HF config nests the LLM config two levels deep (`config.thinker_config.text_config.max_position_embeddings`), unlike the `llm_config` / `text_config` paths the existing fallback already handles. Add the new branch:

```python
elif (
    hasattr(model_hf_config, "thinker_config")
    and hasattr(model_hf_config.thinker_config, "text_config")
    and hasattr(model_hf_config.thinker_config.text_config, "max_position_embeddings")
):
    max_position_embeddings = model_hf_config.thinker_config.text_config.max_position_embeddings
```

**e) `limit_audios` / `limit_videos` config knobs**

Symmetric with existing `limit_images`. Lets users configure `limit_mm_per_prompt={"audio":1, "video":1, "image":0}` via Hydra:

```python
mm_limits = dict(engine_kwargs.pop("limit_mm_per_prompt", {}) or {})
if config.get("limit_images", None) is not None:
    mm_limits.setdefault("image", config.get("limit_images"))
if config.get("limit_audios", None) is not None:
    mm_limits.setdefault("audio", config.get("limit_audios"))
if config.get("limit_videos", None) is not None:
    mm_limits.setdefault("video", config.get("limit_videos"))
if mm_limits:
    engine_kwargs["limit_mm_per_prompt"] = mm_limits
```

### 2. `verl/examples/ttrl/Qwen2.5-Omni/daily_omni*.sh`

Two existing scripts hard-set environment variables that conflict with vLLM rollout:

- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — incompatible with vLLM's cumem allocator (pytorch/pytorch#147851)
- `export TTRL_TASK_TYPE=judge_open_ended` — clobbers callers' settings

Change both to `${VAR-default}` form so callers can override:

```diff
-export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
+export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}"

-export TTRL_TASK_TYPE=judge_open_ended
+export TTRL_TASK_TYPE="${TTRL_TASK_TYPE:-judge_open_ended}"
```

### 3. `slurm/test_vllm_omni.sh` (new)

Smoke-test recipe — 5 training steps end-to-end with vLLM rollout + Qwen2.5-Omni-3B + use_audio_in_video=True. Documented as the canonical "is vLLM Omni working in your env?" test:

```bash
#SBATCH --partition=a6
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00

export VLLM_USE_V1=0                 # cumem allocator stable on V0 with FSDP
export PYTORCH_CUDA_ALLOC_CONF=      # vLLM incompatible with expandable_segments
export TTRL_TASK_TYPE=open_ended_video
# ...
bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh \
  trainer.total_training_steps=5 \
  trainer.save_freq=999 \
  trainer.test_freq=-1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.max_model_len=12000 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  +actor_rollout_ref.rollout.limit_audios=1 \
  +actor_rollout_ref.rollout.limit_videos=1 \
  +actor_rollout_ref.rollout.limit_images=0
```

### 4. `docs/qwen2_5_omni_vllm.md` (new)

Tutorial doc covering:
- Required vLLM version (≥0.16.x with our patches)
- Required env variables (`VLLM_USE_V1=0`, clear `PYTORCH_CUDA_ALLOC_CONF`)
- Required Hydra knobs (`free_cache_engine=False`, `max_model_len`, `max_num_batched_tokens`, `limit_audios/videos/images`)
- Smoke-test command
- Known limitations (vLLM 0.16's audio-in-video patches required)

## Validation

Tested on:
- vllm 0.16.0 (with our 3 vLLM-side patches applied locally)
- 4× A6000 (48GB each)
- Qwen2.5-Omni-3B
- daily_omni dataset (OmniVideo train_open.json, 9k-token prompts with video+audio)
- N=16 rollouts/prompt, max_response_length=1024, T=1.0

Observed: 5 training steps complete cleanly, step time ~13min vs ~17min with HF rollout. Step-1 reward signal is meaningful (gt-comparison rewards in [0.05, 0.34], cluster-vote rewards in {0.0, 1.0}). Larger 300-step run in flight.

## Files changed

```
verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py    | +50 lines (5 patches)
verl/examples/ttrl/Qwen2.5-Omni/daily_omni_judge.sh       | 1 line
verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh             | 1 line
slurm/test_vllm_omni.sh                                   | + new file (~80 lines)
docs/qwen2_5_omni_vllm.md                                 | + new file (~150 lines)
```

## Open follow-ups

- Once vllm-project/vllm patches land, drop the local-override docs.
- Add CI integration test on vLLM-rollout path (currently no Omni-specific CI exists).
- Bigger speedup (currently 1.3×) likely possible by enabling `enable_sleep_mode` once cumem-with-FSDP issues are addressed in a future vLLM release.
