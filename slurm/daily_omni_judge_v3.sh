#!/bin/bash
#SBATCH -J do_judge_v3
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/do_judge_v3_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/do_judge_v3_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --exclude=sls-a6-1,sls-a6-3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --requeue

# judge_v3: judge_v2's policy-as-judge reward + vLLM rollout + aux GPT
# monitoring + clean wandb panels. This is the version that targets the
# diagnosed failure mode: cluster vote partitions by style, not content;
# self-judge skips embedding entirely and asks the policy "given the
# question, score this response 0-10."
#
# Differences vs daily_omni_judge_v2.sh:
#   * vLLM rollout (faster than HF, all patches landed)
#   * TTRL_AUX_DETERMINISTIC + TTRL_AUX_GPT_JUDGE for monitoring
#   * 300 steps to match the cluster-vote ablation
#   * SAVE_FREQ=100 (steps 100/200/300), no in-training eval (offline)
#   * Clean wandb filter

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# vLLM rollout - V0 engine, clear expandable_segments.
export VLLM_USE_V1=0
export PYTORCH_CUDA_ALLOC_CONF=

export TTRL_DEBUG=1
export TTRL_JUDGE_DEBUG=1
export TTRL_OE_DEBUG=0
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400

# Self-judge reward path.
export TTRL_TASK_TYPE=judge_open_ended

# Judge knobs.
export TTRL_JUDGE_MAX_NEW_TOKENS="${TTRL_JUDGE_MAX_NEW_TOKENS:-8}"
export TTRL_JUDGE_MAX_PROMPT_LEN="${TTRL_JUDGE_MAX_PROMPT_LEN:-4096}"
export TTRL_JUDGE_NEUTRAL_FALLBACK="${TTRL_JUDGE_NEUTRAL_FALLBACK:-0.5}"

# Encoder for the BGE-medoid pre-vote (cheap). Mpnet for content/style sep.
export TTRL_OE_ENCODER="${TTRL_OE_ENCODER:-mpnet}"
export MPNET_EMBED_PATH=/data/sls/scratch/mvideet/models/paraphrase-mpnet-base-v2
export TTRL_OE_DEVICE=cpu
export TTRL_OE_MAX_LEN=384

# Confidence gate OFF.
export TTRL_CG_ENABLE=0

# Train horizon.
export EPISODE="${EPISODE:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
# No in-train eval; GPT eval is run offline on saved ckpts.
export TEST_FREQ="${TEST_FREQ:--1}"
export VAL_BEFORE_TRAIN=false
export VAL_DO_SAMPLE=true
export VAL_N=1
export VAL_TEMPERATURE=0.6
export VAL_TOP_P=0.95

export VIDEO_FPS=0.5
export AUDIO_SAMPLE_RATE=8000

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

# OpenAI key for monitoring-only GPT-4o-mini judge (TTRL_AUX_GPT_JUDGE).
if [ -f /data/sls/r/u/mvideet/home/.openai_key ]; then
  source /data/sls/r/u/mvideet/home/.openai_key
fi

# Auxiliary monitoring metrics (training is unaffected; diagnostic only).
export TTRL_AUX_DETERMINISTIC="${TTRL_AUX_DETERMINISTIC:-1}"
export TTRL_AUX_GPT_JUDGE="${TTRL_AUX_GPT_JUDGE:-1}"
export TTRL_AUX_GPT_MODEL="${TTRL_AUX_GPT_MODEL:-gpt-4o-mini-2024-07-18}"
export TTRL_AUX_GPT_CONCURRENCY="${TTRL_AUX_GPT_CONCURRENCY:-8}"

# wandb noise filter - keep only the metrics that matter for the research.
export TTRL_LOG_DROP_PATTERNS="${TTRL_LOG_DROP_PATTERNS:-global_seqlen,timing_s/,timing_per_token_ms/,perf/,critic/,actor/pg_clipfrac,actor/ppo_kl,reward/sim,reward/acc,prompt_length/clip_ratio,response_length/clip_ratio,prompt_length/min,prompt_length/max,response_length/min,response_length/max}"

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

# Hydra overrides matching the cluster-vote vLLM run (89923 / 89887) so the
# two are directly comparable except for the reward source.
bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni_judge.sh \
  trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.val_before_train=$VAL_BEFORE_TRAIN \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.max_model_len=12000 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  +actor_rollout_ref.rollout.limit_audios=1 \
  +actor_rollout_ref.rollout.limit_videos=1 \
  +actor_rollout_ref.rollout.limit_images=0 \
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
