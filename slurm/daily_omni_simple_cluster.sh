#!/bin/bash
#SBATCH -J do_simple
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/do_simple_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/do_simple_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --exclude=sls-a6-1,sls-a6-3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --requeue

# Rung 1 of the ablation ladder: plain cluster vote with binary reward.
#
# In modal cluster: r=1.0
# Else:             r=0.0
# Invalid:          r=0.0
#
# No novelty bonus, no DAPO Clip-Higher, no token-level entropy reg, no
# actor-side KL. Only changes vs vanilla GRPO are:
#  - reward function = cluster vote (Qwen3-Embedding-4B + k-means K=2..4)
#  - ttrl pipeline (rollout.n=16, gen samples=4, vote-then-train)
#
# If this beats judge_v2's +3 GPT-4o-mini, the cluster idea works alone
# and we add EVOL-RL machinery on top. If not, the underlying issue is
# probably the format-vs-content cluster pathology, and we need to embed
# only the answer span (the next ablation).
#
# 300 steps, save_freq=100, walltime 72h.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export WANDB_MODE=online

export TTRL_DEBUG=1
export TTRL_OE_DEBUG=0
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400

# Task / encoder.
export TTRL_TASK_TYPE=simple_cluster
export TTRL_OE_ENCODER=qwen3
export QWEN3_EMBED_PATH=/data/sls/scratch/mvideet/models/Qwen3-Embedding-4B
export TTRL_OE_DEVICE=cuda
export TTRL_OE_MAX_LEN=1024

# Cluster knobs.
export TTRL_CLUSTER_K_MAX="${TTRL_CLUSTER_K_MAX:-4}"
export TTRL_CLUSTER_K_MIN="${TTRL_CLUSTER_K_MIN:-2}"
export TTRL_CLUSTER_SEED=0

export TTRL_CG_ENABLE=0

# Train horizon: 300 steps, save 100/200/300.
export EPISODE="${EPISODE:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export TEST_FREQ="${TEST_FREQ:-25}"
export VAL_BEFORE_TRAIN=true
export VAL_DO_SAMPLE=true
export VAL_N=1
export VAL_TEMPERATURE=0.6
export VAL_TOP_P=0.95

export VIDEO_FPS=0.5
export AUDIO_SAMPLE_RATE=8000

# vLLM rollout: V0 engine + clear expandable_segments (cumem allocator
# incompatibilities surface in V1 + FSDP coexistence).
export VLLM_USE_V1=0
export PYTORCH_CUDA_ALLOC_CONF=

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

# Hydra overrides:
#   - rollout.n=16
#   - rollout.temperature=1.0  (89299 lever)
#   - max_response_length=1024
#   - DEFAULT symmetric clipping (clip_ratio_low=high=0.20 — vanilla GRPO)
#   - entropy_coeff=0           (no entropy reg)
#   - use_kl_loss=False         (no actor-side KL)
#   - kl_ctrl.kl_coef=0.0       (no reward-side KL)
bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni_judge.sh \
  trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.val_before_train=$VAL_BEFORE_TRAIN \
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
  +actor_rollout_ref.rollout.limit_images=0 \
  data.use_audio_in_video=False \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  data.max_response_length=1024 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=11024 \
  actor_rollout_ref.actor.clip_ratio_low=0.20 \
  actor_rollout_ref.actor.clip_ratio_high=0.20 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  algorithm.kl_ctrl.kl_coef=0.0 \
  ttrl.n_votes_per_prompt=16 \
  ttrl.n_samples_per_prompt=4
