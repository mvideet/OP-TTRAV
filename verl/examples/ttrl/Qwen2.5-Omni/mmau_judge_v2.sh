#!/bin/bash
# LLM-as-judge OE TTRL on MMAU (audio-only) with judge_v2 hparams.
#
# Replicates the do_judge_v2 (89391) recipe that gave +9 LLM-judge on
# OmniVideo, but on MMAU's audio-only open-ended split.
# - TTRL_TASK_TYPE=judge_open_ended (BGE-medoid + LLM-as-judge reward)
# - rollout.temperature=1.0  (the 89299 lever; rollout diversity)
# - max_response_length=1024 (room for full answer)
# - LR=2e-6, warmup=0.005    (validated)
# - N=16 votes, batch=4
#
# train data: test_mini_open.json (1000 audio MCQ converted to free-text)
# val data:   test_mini_open_val20.json (20 samples)

# ------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TTRL_TASK_TYPE=judge_open_ended
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

export TTRL_OE_DEBUG="${TTRL_OE_DEBUG:-0}"
export TTRL_JUDGE_DEBUG="${TTRL_JUDGE_DEBUG:-1}"
export TTRL_CG_ENABLE=0
export TTRL_JUDGE_MAX_NEW_TOKENS="${TTRL_JUDGE_MAX_NEW_TOKENS:-8}"
export TTRL_JUDGE_MAX_PROMPT_LEN="${TTRL_JUDGE_MAX_PROMPT_LEN:-4096}"
export TTRL_JUDGE_NEUTRAL_FALLBACK="${TTRL_JUDGE_NEUTRAL_FALLBACK:-0.5}"

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

TASK="MMAU"
BACKBONE="Qwen2.5-Omni-3B"
ADVANTAGE="grpo"

# Audio-only: short prompts but allow long CoT for v2.
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=1024

# 1000 train samples / batch=4 = 250 steps/epoch. Cap at 200 steps to match
# do_judge_v2 (89391) horizon and overnight walltime budget.
EPISODE="${EPISODE:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
DATA_TRAIN_BATCH_SIZE=4
N_VOTES_PER_PROMPT=16
N_SAMPLES_PER_PROMPT=4
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=1

TTRL_ENABLE=true
ROLLOUT_N="${ROLLOUT_N:-$N_VOTES_PER_PROMPT}"
TRAIN_MODE_DESC="open-ended TTRL (semantic medoid)"

VAL_N="${VAL_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-false}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-16000}"

DATA_LOCAL_DIR="${REPO_ROOT}/verl/data/${TASK}"
BACKBONE_PATH="/data/sls/scratch/mvideet/models/${BACKBONE}"

N_GPUS="${N_GPUS:-4}"
TRAIN_FILES="${DATA_LOCAL_DIR}/test_mini_open.json"
VAL_FILES="${DATA_LOCAL_DIR}/test_mini_open_val20.json"

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="TTRL-MMAU-JudgeV2"

WANDB_PROJECT="TTRL-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/sls/scratch/mvideet/TTRL/verl/checkpoints/TTRL-verl/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}}"

cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"
echo "Training mode: ${TRAIN_MODE_DESC} (TTRL_TASK_TYPE=${TTRL_TASK_TYPE}, ttrl.enable=${TTRL_ENABLE}, rollout.n=${ROLLOUT_N})"
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
  +data.suffix_prompt='"\nExplain your reasoning step by step, then give a concise answer to the question in 1-3 complete sentences."' \
  +data.collate_fn=verl.utils.dataset.collate_fn.default_collate_fn \
  data.trust_remote_code=True \
  data.use_qwen2_5_omni=True \
  data.video_file_key='video_file' \
  +data.audio_file_key='audio_file' \
  data.question_key='question' \
  data.answer_key='answer_text' \
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
  actor_rollout_ref.rollout.temperature=1.0 \
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
  custom_reward_function.path="./verl/verl/utils/reward_score/ttrl_judge/__init__.py" \
  custom_reward_function.name=reward_func \
  ttrl.enable=$TTRL_ENABLE \
  ttrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  ttrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=${N_GPUS:-4} \
  trainer.nnodes=${NNODES:-1} \
  trainer.save_freq=${SAVE_FREQ:-50} \
  trainer.test_freq=$TEST_FREQ \
  trainer.val_before_train=$VAL_BEFORE_TRAIN \
  trainer.max_actor_ckpt_to_keep=${MAX_CKPT:-3} \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
  trainer.total_epochs=$EPISODE "$@"

echo "Output directory: $OUTPUT_DIR"
