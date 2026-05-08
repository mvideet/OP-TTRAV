#!/bin/bash
#SBATCH -J test_vllm_omni
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/test_vllm_omni_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/test_vllm_omni_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --exclude=sls-a6-1,sls-a6-3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --requeue

# Smoke test: vLLM rollout for Qwen2.5-Omni-3B on daily_omni.
# 5 training steps, vanilla GRPO + medoid voting (NOT cluster vote — we
# want to isolate "does vLLM rollout work?" from "does cluster vote help?").
#
# If this completes 5 steps without M-RoPE / audio-token errors, vLLM
# rollout is viable and we'll switch the EVOL-RL run to use it.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export WANDB_MODE=offline   # smoke test, no need to log

# vLLM 0.16 should default to V1 with audio fixes per UPGRADE_VLLM_0.16.md.
# If we see audio-token-count assertions, set VLLM_USE_V1=0 and rerun.
# export VLLM_USE_V1=0

# vLLM's memory pool is incompatible with expandable_segments
# (pytorch/pytorch#147851). Disable for vLLM rollout runs.
export PYTORCH_CUDA_ALLOC_CONF=

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Use the existing open_ended_video task type (BGE-medoid voting, simplest).
export TTRL_TASK_TYPE=open_ended_video
export TTRL_OE_ENCODER=bge   # don't pull in Qwen3 embed for the smoke test

export TTRL_DEBUG=1
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400

# Tiny train horizon: 5 steps, just want to see vLLM rollout actually work.
export EPISODE=1
export TOTAL_TRAINING_STEPS=5
export SAVE_FREQ=999  # don't save
export TEST_FREQ=-1   # skip val
export VAL_BEFORE_TRAIN=false

export VIDEO_FPS=0.5
export AUDIO_SAMPLE_RATE=8000

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

# Override the launcher's hf rollout to vllm + Omni mm kwargs.
bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh \
  trainer.total_training_steps=5 \
  trainer.save_freq=999 \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  data.max_response_length=512 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10512 \
  +actor_rollout_ref.rollout.limit_audios=1 \
  +actor_rollout_ref.rollout.limit_videos=1 \
  +actor_rollout_ref.rollout.limit_images=0 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.max_model_len=12000 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  ttrl.n_votes_per_prompt=8 \
  ttrl.n_samples_per_prompt=4
