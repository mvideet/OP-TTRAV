#!/bin/bash
# TTRL: Qwen2.5-Omni on the OmniVideo (daily) JSON, text-only — same questions as full multimodal
# runs but no video/audio loaded. Set data.use_omnivideo_text=True (see RLOMNIDataset).

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

unset ROCR_VISIBLE_DEVICES || true
export ROCR_VISIBLE_DEVICES=
export XDG_RUNTIME_DIR="/data/sls/scratch/mvideet/xdg_runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export RAY_TMPDIR="/data/sls/scratch/mvideet/ray_tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$RAY_TMPDIR"

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="OmniVideo"
BACKBONE="Qwen2.5-Omni-3B"
ADVANTAGE="grpo"

# Text-only: short prompts (no vision tokens)
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=512

EPISODE=10
DATA_TRAIN_BATCH_SIZE=4
N_VOTES_PER_PROMPT=16
N_SAMPLES_PER_PROMPT=4
MINI_BATCH_SIZE=2
MICRO_BATCH_SIZE=1

DATA_LOCAL_DIR="${REPO_ROOT}/verl/data/${TASK}"
BACKBONE_PATH="/data/sls/scratch/mvideet/models/${BACKBONE}"

N_SANITY="${N_SANITY:-40}"
N_GPUS="${N_GPUS:-4}"
if [[ -n "${SANITY_CHECK}" && "${SANITY_CHECK}" != "0" ]]; then
  SANITY_DIR="${DATA_LOCAL_DIR}"
  CAT_ARGS=()
  [[ -n "${CONTENT_PARENT_CATEGORY}" ]] && CAT_ARGS=(--content-parent-category "${CONTENT_PARENT_CATEGORY}")
  python "${REPO_ROOT}/verl/data/${TASK}/sample_by_category.py" \
    --train "${DATA_LOCAL_DIR}/train.json" \
    --test "${DATA_LOCAL_DIR}/test.json" \
    --out-dir "${SANITY_DIR}" \
    --n-total "${N_SANITY}" \
    "${CAT_ARGS[@]}" \
    --suffix sanity
  TRAIN_FILES="${SANITY_DIR}/train_sanity.json"
  VAL_FILES="${SANITY_DIR}/test_sanity.json"
  DATA_TRAIN_BATCH_SIZE=$(( (DATA_TRAIN_BATCH_SIZE + N_GPUS - 1) / N_GPUS * N_GPUS ))
  CAT_DESC="${CONTENT_PARENT_CATEGORY:-all categories}"
  echo "Sanity check: ${TRAIN_FILES} / ${VAL_FILES} (content_parent_category=${CAT_DESC}, n_total=${N_SANITY})"
else
  TRAIN_FILES="${DATA_LOCAL_DIR}/train.json"
  VAL_FILES="${DATA_LOCAL_DIR}/test.json"
fi

MODEL="${TASK}-${BACKBONE}-TextOnly"
EXPERIMENT="TTRL-Omni-TextOnly"

WANDB_PROJECT="TTRL-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="/data/sls/scratch/mvideet/TTRL/verl/checkpoints/TTRL-verl/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}"

cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"

python -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \
  data.train_files=["$TRAIN_FILES"] \
  data.val_files=["$VAL_FILES"] \
  data.use_omnivideo_text=True \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.val_batch_size=8 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  +data.suffix_prompt='"\nExplain your reasoning step by step in detail, then give your final answer as exactly one of: \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."' \
  +data.collate_fn=verl.utils.dataset.collate_fn.default_collate_fn \
  data.trust_remote_code=True \
  data.use_qwen2_5_omni=True \
  data.video_file_key='video_file' \
  data.question_key='question' \
  data.answer_key='answer' \
  data.use_audio_in_video=True \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_activation_offload=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.ppo_epochs=4 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
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
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.n=$N_VOTES_PER_PROMPT \
  +actor_rollout_ref.rollout.num_return_sequences_batch_size=8 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
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
  custom_reward_function.path="./verl/verl/utils/reward_score/ttrl_video_qa/__init__.py" \
  custom_reward_function.name=reward_func \
  ttrl.enable=True \
  ttrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  ttrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=${N_GPUS:-4} \
  trainer.nnodes=${NNODES:-1} \
  trainer.save_freq=${SAVE_FREQ:-5} \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.max_actor_ckpt_to_keep=${MAX_CKPT:-3} \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "Output directory: $OUTPUT_DIR"
