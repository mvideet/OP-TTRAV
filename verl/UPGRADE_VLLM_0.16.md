# Upgrading to vLLM 0.16+

This guide covers upgrading from vLLM 0.8.x to vLLM 0.16+ for improved Qwen2.5-Omni support (including `use_audio_in_video`).

## Steps

### 1. Install vLLM 0.16+

```bash
# In your conda environment (e.g. verl310)
pip uninstall vllm -y
pip install "vllm>=0.16.0"
```

Or with uv (recommended by vLLM):
```bash
uv pip install vllm==0.16.0 --torch-backend=auto
```

### 2. Remove any site-packages patches

If you previously patched vLLM files in site-packages (e.g. `qwen2_5_omni_thinker.py`, `rotary_embedding.py`), those are overwritten by the new install. vLLM 0.16 includes fixes for `use_audio_in_video` and placeholder validation, so patches should no longer be needed.

### 3. Verify installation

```bash
python -c "import vllm; print(vllm.__version__)"
# Should print 0.16.x or higher
```

### 4. Run your job

```bash
# From repo root
bash slurm/daily_omni.sh
# or
bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh
```

## Changes made in this repo

- **verl/requirements.txt**: `vllm>=0.16.0`
- **verl/setup.py**: `VLLM_REQUIRES` updated to `vllm>=0.16.0`
- **verl/verl/third_party/vllm/__init__.py**: Minimum version 0.16.0
- **verl/verl/trainer/main_ppo.py**: Version check updated to 0.16.0
- **verl/verl/utils/vllm_utils.py**: Added `get_model_runner_from_engine()` for vLLM 0.16+ engine structure
- **verl/verl/workers/sharding_manager/fsdp_vllm.py**: Uses version-adaptive model_runner path
- **verl/verl/workers/sharding_manager/megatron_vllm.py**: Same
- **verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh**: Removed `VLLM_USE_V1=0` (0.16 uses V1 by default with fixes)

## Potential issues

1. **PyTorch version**: vLLM 0.16 may require PyTorch 2.10+. Check vLLM release notes.
2. **Import errors**: If `vllm.lora.*` or `vllm.model_executor.*` paths changed, vllm_utils.py may need updates.
3. **model_runner path**: If you see "Could not find model_runner in vLLM engine", the engine structure may have changed further; open an issue with your vLLM version.
