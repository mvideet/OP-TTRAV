#!/bin/bash
#SBATCH -J uf_cluster
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/uf_cluster_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/uf_cluster_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --exclude=sls-a6-1,sls-a6-3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --requeue

# TTRL on Qwen2.5-3B base + UltraFeedback prompts.
# Text-only, much faster iteration than daily_omni.
#
# Method: simple_cluster vote with continuous medoid reward (smooth [0,1])
# + paraphrase-MPNet encoder + aux GPT-4o-mini judge monitoring.
#
# Compare against: vanilla SFT-from-UltraFeedback baselines (~30% LC win
# rate on AlpacaEval) and DPO (~40-50%). Base model alone is ~5-15%.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
export PYTORCH_CUDA_ALLOC_CONF=

export TTRL_DEBUG=1
export TTRL_OE_DEBUG=0

# Reward path
export TTRL_TASK_TYPE=simple_cluster
export TTRL_CLUSTER_CONTINUOUS=1        # smooth [0,1] medoid reward
export TTRL_OE_ENCODER=bge              # apples-to-apples with 90015 (medoid)
export BGE_MODEL_PATH=/data/sls/scratch/mvideet/models/bge-small-en-v1.5
export TTRL_OE_DEVICE=cuda
export TTRL_OE_MAX_LEN=512

# Dedup duplicates before GRPO (same as 90015)
export TTRL_DEDUP_SAMPLES=1
export TTRL_CLUSTER_K_MIN=2
export TTRL_CLUSTER_K_MAX=4
export TTRL_CG_ENABLE=0

# Train horizon
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ=-1                     # offline eval only
export VAL_BEFORE_TRAIN=false

unset ROCR_VISIBLE_DEVICES || true
export ROCR_VISIBLE_DEVICES=

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

# OpenAI key for monitoring-only GPT-4o-mini judge.
if [ -f /data/sls/r/u/mvideet/home/.openai_key ]; then
  source /data/sls/r/u/mvideet/home/.openai_key
fi

# Aux monitoring metrics (no training-time effect).
export TTRL_AUX_DETERMINISTIC="${TTRL_AUX_DETERMINISTIC:-1}"
export TTRL_AUX_GPT_JUDGE="${TTRL_AUX_GPT_JUDGE:-1}"
export TTRL_AUX_GPT_MODEL="${TTRL_AUX_GPT_MODEL:-gpt-4o-mini-2024-07-18}"
export TTRL_AUX_GPT_CONCURRENCY="${TTRL_AUX_GPT_CONCURRENCY:-8}"

# Drop noisy wandb panels.
export TTRL_LOG_DROP_PATTERNS="${TTRL_LOG_DROP_PATTERNS:-global_seqlen,timing_s/,timing_per_token_ms/,perf/,critic/,actor/pg_clipfrac,actor/ppo_kl,reward/sim,reward/acc,prompt_length/clip_ratio,response_length/clip_ratio,prompt_length/min,prompt_length/max,response_length/min,response_length/max}"

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="UltraFeedback-TTRL"
BACKBONE="Qwen2.5-3B"
ADVANTAGE="grpo"

MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=1024
DATA_TRAIN_BATCH_SIZE=8
N_VOTES_PER_PROMPT=16
N_SAMPLES_PER_PROMPT=4
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=1

DATA_LOCAL_DIR="$(pwd)/verl/data"
BACKBONE_PATH="/data/sls/scratch/mvideet/models/${BACKBONE}"

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="TTRL-UF-SimpleClusterContinuous"
WANDB_PROJECT="TTRL-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="/data/sls/scratch/mvideet/TTRL/verl/checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}"

python -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \
  data.train_files=["${DATA_LOCAL_DIR}/${TASK}/train.json"] \
  data.val_files=["${DATA_LOCAL_DIR}/${TASK}/test.json"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.question_key=prompt \
  data.answer_key=answer \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.20 \
  actor_rollout_ref.actor.clip_ratio_high=0.20 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
  actor_rollout_ref.rollout.n=$N_VOTES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  custom_reward_function.path="./verl/verl/utils/reward_score/ttrl_judge/__init__.py" \
  custom_reward_function.name=reward_func \
  ttrl.enable=True \
  ttrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  ttrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.val_before_train=$VAL_BEFORE_TRAIN \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
  trainer.total_epochs=20 "$@"

echo "Output directory: $OUTPUT_DIR"
