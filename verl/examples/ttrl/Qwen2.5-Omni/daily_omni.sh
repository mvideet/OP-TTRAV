#!/bin/bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The LLM-Tuner Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script is for TTRL fine-tuning of Qwen2.5-Omni-7B model on OmniVideo dataset

set -x

# Ensure runtime/tmp dirs are writable before anything else
unset ROCR_VISIBLE_DEVICES || true
export ROCR_VISIBLE_DEVICES=

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/data/sls/scratch/mvideet/ray_files}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

export RAY_TMPDIR="${RAY_TMPDIR:-/data/sls/scratch/mvideet/ray_tmp/${SLURM_JOB_ID:-manual}}"
mkdir -p "$RAY_TMPDIR"

export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

TASK="OmniVideo"
BACKBONE="Qwen2.5-Omni-3B"
MAX_RESPONSE_LENGTH=128   # Reduced from 256 to save memory (longer = more KV cache + activations)
N=16                      # Total rollouts, used to be 128

DATA_TRAIN_BATCH_SIZE=4    # Batch size per rollout (must be divisible by n_gpus_per_node)
N_VOTES_PER_PROMPT=16      # Reduced from 32 to fit GPU memory (fewer sequences per generation)
N_SAMPLES_PER_PROMPT=8     # Reduced from 16 (fewer samples for PPO = less memory in actor/critic)
# PPO mini-batch size must be <= train_batch_size (validated in RayPPOTrainer)
MINI_BATCH_SIZE=${DATA_TRAIN_BATCH_SIZE}
MICRO_BATCH_SIZE=1         # Reduced from 2 to lower peak memory during log_prob and PPO steps

# Paths - use the verl/data directory in the repo
DATA_LOCAL_DIR=/data/sls/r/u/mvideet/TTRL/verl/data/${TASK}
# Load 7B from HF Hub
BACKBONE_PATH="/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B"

# NOTE: Adjust based on your dataset
DATA_TRAIN_FILES=(
    "${DATA_LOCAL_DIR}/train.json"
)

DATA_VAL_FILES=(
    "${DATA_LOCAL_DIR}/test.json"
)

# NOTE: Update these paths to your actual file locations
train_files_str=$(IFS=, ; echo "${DATA_TRAIN_FILES[*]}")
val_files_str=$(IFS=, ; echo "${DATA_VAL_FILES[*]}")

# Create output directory
OUTPUT_DIR="outputs/ttrl/${TASK}/${BACKBONE}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# Run training with VERL
python3 -m verl.trainer.main_ppo \
    --config-name='ppo_trainer_ttrl.yaml' \
    actor_rollout_ref.model.path=${BACKBONE_PATH} \
    critic.model.path=${BACKBONE_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    critic.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    critic.model.use_remove_padding=False \
    data.train_files=${train_files_str} \
    data.val_files=${val_files_str} \
    data.max_prompt_length=10000 \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${DATA_TRAIN_BATCH_SIZE} \
    data.val_batch_size=${DATA_TRAIN_BATCH_SIZE} \
    +data.collate_fn=verl.utils.dataset.collate_fn.default_collate_fn \
    data.trust_remote_code=True \
    data.use_qwen2_5_omni=True \
    data.video_file_key='video_file' \
    data.question_key='question' \
    data.answer_key='answer' \
    data.use_audio_in_video=True \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.n=${N_VOTES_PER_PROMPT} \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="./verl/verl/utils/reward_score/ttrl_video_qa/__init__.py" \
    custom_reward_function.name=reward_func \
    ttrl.enable=True \
    ttrl.n_votes_per_prompt=${N_VOTES_PER_PROMPT} \
    ttrl.n_samples_per_prompt=${N_SAMPLES_PER_PROMPT} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.n_gpus_per_node=${N_GPUS:-4} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=1000 \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name="ttrl_qwen25_omni" \
    trainer.experiment_name="${TASK}_${BACKBONE}" \
    "$@"
