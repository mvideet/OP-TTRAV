#!/bin/bash
# Qwen3-Omni TTRL training script for Audio-Visual reasoning
# Uses HF rollout (not vLLM) for multimodal generation

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

# Task and model configuration
TASK="OmniVideo"
BACKBONE="Qwen3-Omni-7B"
ADVANTAGE="grpo"

# Sequence length settings (shorter for multimodal due to memory)
K=2
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=$((512 * $K))

# Validation samples (reduced for 4-GPU setup)
if [ "$K" -gt 4 ]; then
  N=2
else
  N=4
fi

# Training hyperparameters (adjusted for 4-GPU setup)
EPISODE=10
DATA_TRAIN_BATCH_SIZE=8
N_VOTES_PER_PROMPT=16
N_SAMPLES_PER_PROMPT=8
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=1

# Paths - UPDATE THESE FOR YOUR SETUP
DATA_LOCAL_DIR="/data/sls/r/u/mvideet/TTRL/verl/data"
# Set BACKBONE_PATH via environment variable or update this default
BACKBONE_PATH="${BACKBONE_PATH:-/data/sls/u/urop/mvideet/TTRL/verl/Qwen3-Omni-30B-A3B-Instruct}"

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="TTRL-Len@${K}k"

WANDB_PROJECT="TTRL-verl-omni"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}"

# ------------------------------------------------------------
# Training command
# ------------------------------------------------------------
python -m verl.trainer.main_ppo \
--config-name='ppo_trainer_ttrl.yaml' \
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train.json"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test.json"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.trust_remote_code=True \
  data.use_qwen3_omni=True \
  data.video_file_key='video_file' \
  data.question_key='question' \
  data.answer_key='answer' \
  data.use_audio_in_video=True \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.n=$N \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$N \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=False \
  critic.model.path=$BACKBONE_PATH \
  critic.model.trust_remote_code=True \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  custom_reward_function.path="./verl/utils/reward_score/ttrl_video_qa/__init__.py" \
  custom_reward_function.name=reward_func \
  ttrl.enable=True \
  ttrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  ttrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=${N_GPUS:-4} \
  trainer.nnodes=1 \
  ray_init.num_cpus=${RAY_NUM_CPUS:-28} \
  trainer.save_freq=2000000 \
  trainer.test_freq=2 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "Output directory: $OUTPUT_DIR"
