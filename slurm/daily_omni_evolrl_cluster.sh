#!/bin/bash
#SBATCH -J do_evolrl
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/do_evolrl_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/do_evolrl_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --exclude=sls-a6-1,sls-a6-3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --requeue

# EVOL-RL with continuous-vector cluster voting on daily_omni (OmniVideo).
#
# Recipe combines:
#  1. User's "discrete cluster space" idea — replace BGE-medoid with k-means
#     cluster mode in Qwen3-Embedding-4B latent space; modal cluster centroid
#     is the pseudo-GT direction; reward = membership in modal cluster.
#  2. EVOL-RL (Zhou et al., arXiv 2509.15194) banded reward:
#        modal cluster:  r = 0.5 + 0.5 * u_i  in [0.5, 1.0]
#        outside:        r = -1.0 + 0.5 * u_i in [-1.0, -0.5]
#        invalid:        r = -1.0
#     where u_i is min-max-normalized novelty within each band, and novelty
#     itself is 1 - 0.5*intra_band_avg_sim - 0.5*global_max_sim.
#  3. DAPO Clip-Higher (Yu et al., arXiv 2503.14476): asymmetric clipping
#     ε_low=0.20, ε_high=0.28 to allow promising-novel updates through.
#  4. Token-level entropy regularizer (EVOL-RL Eq. 2): -λ_ent * mean H(π).
#  5. KL on actor side: kl_loss_coef=0.001 (EVOL-RL keeps this; DAPO disables).
#
# Goal: beat current judge_v2 +3 GPT-4o-mini gain on daily_omni open-ended
# eval by replacing the noisy medoid+judge step with cluster-mode voting +
# anti-collapse mechanisms.
#
# Weekend run: 500 steps, save every 125, walltime 72h.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export WANDB_MODE=online

# Debug knobs (small, periodic).
export TTRL_DEBUG=1
export TTRL_OE_DEBUG=0
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400

# Task / encoder.
export TTRL_TASK_TYPE=evolrl_cluster
export TTRL_OE_ENCODER=qwen3
export QWEN3_EMBED_PATH=/data/sls/scratch/mvideet/models/Qwen3-Embedding-4B
export TTRL_OE_DEVICE=cuda
export TTRL_OE_MAX_LEN=1024

# Cluster knobs.
export TTRL_CLUSTER_K_MAX="${TTRL_CLUSTER_K_MAX:-4}"
export TTRL_CLUSTER_K_MIN="${TTRL_CLUSTER_K_MIN:-2}"
export TTRL_CLUSTER_SEED=0

# Confidence gate off (orthogonal mechanism, not used here).
export TTRL_CG_ENABLE=0

# Train horizon: 500 steps, save every 125 (so 4 ckpts: step 125/250/375/500).
export EPISODE="${EPISODE:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
export SAVE_FREQ="${SAVE_FREQ:-125}"
export TEST_FREQ="${TEST_FREQ:-25}"
export VAL_BEFORE_TRAIN=true
export VAL_DO_SAMPLE=true
export VAL_N=1
export VAL_TEMPERATURE=0.6
export VAL_TOP_P=0.95

# Audio/video settings (match daily_omni).
export VIDEO_FPS=0.5
export AUDIO_SAMPLE_RATE=8000

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

# Reuse daily_omni_judge.sh as launcher; it points custom_reward_function at
# ttrl_judge/__init__.py which parses the JSON score map written by our
# evolrl_cluster vote module — no launcher edit needed. We override:
#   - rollout.n  = 16  (already default)
#   - clip_ratio_low / clip_ratio_high  (DAPO Clip-Higher)
#   - clip_ratio_c                      (dual-clip)
#   - entropy_coeff                     (EVOL-RL ent reg)
#   - kl_loss_coef                      (EVOL-RL keeps actor-side KL)
#   - max_response_length, temperature  (89299 lever; longer + hotter)
bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni_judge.sh \
  trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.val_before_train=$VAL_BEFORE_TRAIN \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  data.max_response_length=1024 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=11024 \
  actor_rollout_ref.actor.clip_ratio_low=0.20 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  algorithm.kl_ctrl.kl_coef=0.0 \
  ttrl.n_votes_per_prompt=16 \
  ttrl.n_samples_per_prompt=4
