#!/bin/bash
#SBATCH -J aime_simple
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/aime_simple_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/aime_simple_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --requeue

# Companion to daily_omni_simple_cluster.sh: rung-1 ablation (binary
# cluster vote, no novelty / no clip-higher / no entropy reg) on AIME
# math benchmark instead of open-ended video QA.
#
# Goal: see if the simple cluster vote idea generalizes from open-ended
# multimodal to math (where a deterministic answer extractor exists).
# Same Qwen3-Embedding-4B, same N=16, similar lr/horizon to do_simple.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# vLLM rollout - text-only, no Omni audio-in-video issues
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF=

export TTRL_DEBUG=1
export TTRL_OE_DEBUG=0

# Task / encoder.
export TTRL_TASK_TYPE=simple_cluster
export TTRL_OE_ENCODER=qwen3
export QWEN3_EMBED_PATH=/data/sls/scratch/mvideet/models/Qwen3-Embedding-4B
export TTRL_OE_DEVICE=cuda
export TTRL_OE_MAX_LEN=1024

# Cluster knobs.
export TTRL_CLUSTER_K_MAX=4
export TTRL_CLUSTER_K_MIN=2
export TTRL_CLUSTER_SEED=0

export TTRL_CG_ENABLE=0

# Train horizon: 300 steps for direct comparison with do_simple.
export TOTAL_TRAINING_STEPS=300
export SAVE_FREQ=100
export TEST_FREQ=25
export VAL_BEFORE_TRAIN=true

unset ROCR_VISIBLE_DEVICES || true
export ROCR_VISIBLE_DEVICES=

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="AIME-TTT"
BACKBONE="Qwen2.5-7B"
ADVANTAGE="grpo"

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=3072
DATA_TRAIN_BATCH_SIZE=8
N_VOTES_PER_PROMPT=16
N_SAMPLES_PER_PROMPT=4
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=1

DATA_LOCAL_DIR="$(pwd)/verl/data"
BACKBONE_PATH="/data/sls/scratch/mvideet/models/${BACKBONE}"

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="TTRL-AIME-SimpleCluster"
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
  +data.suffix_prompt='"\nPlease reason step by step, and put your final answer within \\boxed{}."' \
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
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
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
  trainer.total_epochs=80 "$@"

echo "Output directory: $OUTPUT_DIR"
