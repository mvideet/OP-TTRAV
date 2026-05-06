#!/bin/bash
# TTRL AMC on Qwen2.5-Omni-3B thinker (text-only math).
#
# KEY: Must use data.use_qwen2_5_omni=True so the Omni processor is loaded
# and RLOMNIDataset computes 3D M-RoPE position_ids.  Without this the
# Qwen2_5OmniRotaryEmbedding receives 2D position_ids and crashes.
# The dataset's use_omnivideo_text=True avoids loading any AV features.

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TTRL_TASK_TYPE=math
export VAL_DEBUG=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
if [[ ! -d "${REPO_ROOT}/verl/verl" && -n "${SLURM_SUBMIT_DIR}" && -d "${SLURM_SUBMIT_DIR}/verl/verl" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"

unset ROCR_VISIBLE_DEVICES || true
export ROCR_VISIBLE_DEVICES=
export XDG_RUNTIME_DIR="/data/sls/scratch/mvideet/xdg_runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}_${SLURM_JOB_ID:-manual}}"
mkdir -p "$RAY_TMPDIR"

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="AMC-TTT"
BACKBONE="Qwen2.5-Omni-3B"
ADVANTAGE="grpo"

K=3
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=$((1024 * K))

EPISODE="${EPISODE:-50}"
DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-8}"
N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"

TRAIN_ON_GT_LABELS="${TRAIN_ON_GT_LABELS:-0}"
if [[ "${TRAIN_ON_GT_LABELS}" != "0" ]]; then
  TTRL_ENABLE=false
  ROLLOUT_N="${ROLLOUT_N:-$N_SAMPLES_PER_PROMPT}"
  TRAIN_MODE_DESC="ground-truth labels"
else
  TTRL_ENABLE=true
  ROLLOUT_N="${ROLLOUT_N:-$N_VOTES_PER_PROMPT}"
  TRAIN_MODE_DESC="TTRL majority-voted labels"
fi

DATA_LOCAL_DIR="${REPO_ROOT}/verl/data"
BACKBONE_PATH="/data/sls/scratch/mvideet/models/${BACKBONE}"

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="TTRL-AMC-Omni"

WANDB_PROJECT="TTRL-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}"

cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"
echo "Training mode: ${TRAIN_MODE_DESC} (TRAIN_ON_GT_LABELS=${TRAIN_ON_GT_LABELS}, ttrl.enable=${TTRL_ENABLE}, rollout.n=${ROLLOUT_N})"

python -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train.json"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test.json"] \
  data.val_batch_size=32 \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.trust_remote_code=True \
  data.use_qwen2_5_omni=True \
  data.use_omnivideo_text=True \
  data.question_key='prompt' \
  data.answer_key='answer' \
  +data.suffix_prompt='"\nPlease reason step by step, and put your final answer within \\boxed{}."' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_activation_offload=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.005 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.micro_batch_size=8 \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  +actor_rollout_ref.rollout.num_return_sequences_batch_size=8 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.model.enable_activation_offload=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  critic.use_dynamic_bsz=True \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  custom_reward_function.path="./verl/verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  ttrl.enable=$TTRL_ENABLE \
  ttrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  ttrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=${N_GPUS:-4} \
  trainer.nnodes=${NNODES:-1} \
  trainer.save_freq=${SAVE_FREQ:-20} \
  trainer.test_freq=${TEST_FREQ:-2} \
  trainer.val_before_train=True \
  trainer.max_actor_ckpt_to_keep=${MAX_CKPT:-0} \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "Output directory: $OUTPUT_DIR"
