#!/bin/bash
# TTRL fine-tuning of Qwen2.5-Omni model on MMAU (audio-only) dataset

# ------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TTRL_TASK_TYPE=video_qa
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
if [[ ! -d "${REPO_ROOT}/verl/verl" && -n "${SLURM_SUBMIT_DIR}" && -d "${SLURM_SUBMIT_DIR}/verl/verl" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"

# Cluster: writable runtime dirs for Ray
unset ROCR_VISIBLE_DEVICES || true
export ROCR_VISIBLE_DEVICES=
export XDG_RUNTIME_DIR="/data/sls/scratch/mvideet/xdg_runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}_${SLURM_JOB_ID:-manual}}"
mkdir -p "$RAY_TMPDIR"


DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="MMAU"
BACKBONE="Qwen2.5-Omni-3B"
ADVANTAGE="grpo"

# Audio-only: shorter prompts than video (no vision tokens, just audio features)
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=512

EPISODE="${EPISODE:-50}"
DATA_TRAIN_BATCH_SIZE=4
N_VOTES_PER_PROMPT=16
N_SAMPLES_PER_PROMPT=4
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=1

# TTRL majority-voting mode (not GT labels)
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

# Validation
VAL_N="${VAL_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-false}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
TEST_FREQ="${TEST_FREQ:-2}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"

# Audio: native Whisper sample rate for Qwen2.5-Omni
AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-16000}"

DATA_LOCAL_DIR="${REPO_ROOT}/verl/data/${TASK}"
BACKBONE_PATH="/data/sls/scratch/mvideet/models/${BACKBONE}"

N_GPUS="${N_GPUS:-4}"
# TTRL: train on the full test set itself (majority voting, not GT labels)
# Validation on a small 20-sample subset for periodic monitoring
TRAIN_FILES="${DATA_LOCAL_DIR}/test_mini.json"
VAL_FILES="${DATA_LOCAL_DIR}/test_mini_val20.json"

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="TTRL-MMAU"

WANDB_PROJECT="TTRL-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/sls/scratch/mvideet/TTRL/verl/checkpoints/TTRL-verl/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}}"

cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"
echo "Training mode: ${TRAIN_MODE_DESC} (TRAIN_ON_GT_LABELS=${TRAIN_ON_GT_LABELS}, ttrl.enable=${TTRL_ENABLE}, rollout.n=${ROLLOUT_N})"
echo "Validation: test_freq=${TEST_FREQ}, val_before_train=${VAL_BEFORE_TRAIN}, val_n=${VAL_N}, val_do_sample=${VAL_DO_SAMPLE}, val_temperature=${VAL_TEMPERATURE}"
echo "Audio: audio_sample_rate=${AUDIO_SAMPLE_RATE}"

python -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \
  data.train_files=["$TRAIN_FILES"] \
  data.val_files=["$VAL_FILES"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.val_batch_size=8 \
  data.filter_overlong_prompts=False \
  data.truncation='error' \
  +data.suffix_prompt='"\nExplain your reasoning step by step in detail, then give your final answer as exactly one of: \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."' \
  +data.collate_fn=verl.utils.dataset.collate_fn.default_collate_fn \
  data.trust_remote_code=True \
  data.use_qwen2_5_omni=True \
  data.video_file_key='video_file' \
  +data.audio_file_key='audio_file' \
  data.question_key='question' \
  data.answer_key='answer' \
  data.use_audio_in_video=False \
  +data.audio_sample_rate=${AUDIO_SAMPLE_RATE} \
  +data.max_audio_duration=${MAX_AUDIO_DURATION:-30.0} \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_epochs=4 \
  actor_rollout_ref.actor.optim.lr=2e-6 \
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
  actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
  actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
  actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
  actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
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
  custom_reward_function.path="./verl/verl/utils/reward_score/ttrl_video_qa/__init__.py" \
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
  trainer.test_freq=$TEST_FREQ \
  trainer.val_before_train=$VAL_BEFORE_TRAIN \
  trainer.max_actor_ckpt_to_keep=${MAX_CKPT:-1} \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "Output directory: $OUTPUT_DIR"
