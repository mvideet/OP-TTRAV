#!/bin/bash
#SBATCH -J es_kde
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/es_kde_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/es_kde_%j.err
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

# Task / encoder. Reward = KDE density (fixes saddle pathology) + EVOL-TTRL
# novelty bonus (anti-mode-collapse without entropy regularization).
# Switched from qwen3 (broken for paraphrase sim) to bge encoder.
export TTRL_TASK_TYPE=simple_cluster
export TTRL_OE_ENCODER=bge
export BGE_MODEL_PATH=/data/sls/scratch/mvideet/models/bge-small-en-v1.5
export TTRL_OE_DEVICE=cuda
export TTRL_OE_MAX_LEN=512

# KDE density reward + EVOL novelty bonus (the winning combo on Arena-Hard).
export TTRL_CLUSTER_KDE=1
export TTRL_CLUSTER_KDE_TAU=0.2
export TTRL_NOVELTY_BETA=0.1
export TTRL_NOVELTY_BUFFER_SIZE=100
export TTRL_CLUSTER_CONTINUOUS=0         # KDE supersedes continuous medoid

# Cluster knobs (still computed for diagnostics even if KDE drives reward).
export TTRL_CLUSTER_K_MAX="${TTRL_CLUSTER_K_MAX:-4}"
export TTRL_CLUSTER_K_MIN="${TTRL_CLUSTER_K_MIN:-2}"
export TTRL_CLUSTER_SEED=0

export TTRL_CG_ENABLE=0

# Train horizon: 200 steps, save once at end. Multimodal step time ~10 min
# so 200 steps = ~33h walltime (within 72h limit). Halved from 300 due to
# step-50 ghost-stop pattern we kept hitting with intermediate saves.
export EPISODE="${EPISODE:-3}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
export SAVE_FREQ="${SAVE_FREQ:-200}"
# No in-training eval; user will run GPT-4o-mini judge offline on saved ckpts.
export TEST_FREQ="${TEST_FREQ:--1}"
export VAL_BEFORE_TRAIN=false
export VAL_DO_SAMPLE=true
export VAL_N=1
export VAL_TEMPERATURE=0.6
export VAL_TOP_P=0.95

export VIDEO_FPS=0.25                    # halved from 0.5 (EgoSound clips can be 60-120s; 0.5 fps → >30K tokens)
export VIDEO_MAX_FRAMES=16               # cap (was default 32); each frame ~640 multimodal tokens
export AUDIO_SAMPLE_RATE=8000
export MAX_AUDIO_DURATION=30.0           # clip audio at 30s

# vLLM rollout: V0 engine + clear expandable_segments (cumem allocator
# incompatibilities surface in V1 + FSDP coexistence).
export VLLM_USE_V1=0
export PYTORCH_CUDA_ALLOC_CONF=

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

# OpenAI key for monitoring-only GPT-4o-mini judge (TTRL_AUX_GPT_JUDGE).
if [ -f /data/sls/r/u/mvideet/home/.openai_key ]; then
  source /data/sls/r/u/mvideet/home/.openai_key
fi

# Auxiliary monitoring metrics (no training-time effect):
#   TTRL_AUX_DETERMINISTIC=1  BLEU/ROUGE-L/exact-match (always cheap)
#   TTRL_AUX_GPT_JUDGE=1      GPT-4o-mini side-channel judge (~$4.50 / 300-step run)
export TTRL_AUX_DETERMINISTIC="${TTRL_AUX_DETERMINISTIC:-1}"
export TTRL_AUX_GPT_JUDGE="${TTRL_AUX_GPT_JUDGE:-1}"
export TTRL_AUX_GPT_MODEL="${TTRL_AUX_GPT_MODEL:-gpt-4o-mini-2024-07-18}"
export TTRL_AUX_GPT_CONCURRENCY="${TTRL_AUX_GPT_CONCURRENCY:-8}"

# wandb noise filter: drop debugging/perf series so the dashboard shows
# only the metrics that matter for the cluster-vote research story.
# Keeps: actor/{entropy,pg_loss,grad_norm,lr}, train/{label_accuracy,
# ground_truth_reward,majority_voting_reward,pass@4,cluster_*,aux_*},
# training/global_step. Drops: global_seqlen, timing, perf, critic mirrors,
# pg_clipfrac, ppo_kl, reward/{sim,acc} (aliases of reward/score),
# prompt/response length clip_ratio.
export TTRL_LOG_DROP_PATTERNS="${TTRL_LOG_DROP_PATTERNS:-global_seqlen,timing_s/,timing_per_token_ms/,perf/,critic/,actor/pg_clipfrac,actor/ppo_kl,reward/sim,reward/acc,prompt_length/clip_ratio,response_length/clip_ratio,prompt_length/min,prompt_length/max,response_length/min,response_length/max}"

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
  data.train_files=[/data/sls/u/urop/mvideet/TTRL/verl/data/EgoSound-Ego4d-TTRL/train.json] \
  data.val_files=[/data/sls/u/urop/mvideet/TTRL/verl/data/EgoSound-Ego4d-TTRL/test.json] \
  data.question_key=prompt \
  data.answer_key=answer \
  data.video_file_key=video_file \
  +data.audio_file_key=audio_file \
  data.filter_overlong_prompts=True \
  data.max_prompt_length=12000 \
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
