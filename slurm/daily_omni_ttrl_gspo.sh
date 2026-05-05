#!/bin/bash
#SBATCH -J do_ttrl_gspo
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/do_ttrl_gspo_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/do_ttrl_gspo_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --exclude=sls-a6-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=40:00:00
#SBATCH --requeue

# Vanilla TTRL (string-counting majority vote, TTRL_TASK_TYPE=video_qa)
# on OmniVideo train.json (2018 samples), from base weights, 300 steps,
# with GSPO (length-normalized sequence-level importance ratio) replacing
# GRPO's token-level ratio in the policy loss. Advantage estimator stays
# grpo (group-relative), only the loss function changes.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export WANDB_MODE=online

export TTRL_DEBUG=1
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400

# MCQ majority-vote TTRL (not TTRV).
export TTRL_TASK_TYPE=video_qa

# TTRL on (majority-voted pseudo labels, not ground-truth labels).
export TRAIN_ON_GT_LABELS=0

export EPISODE="${EPISODE:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"

export VIDEO_FPS=0.5
export AUDIO_SAMPLE_RATE=8000
export TEST_FREQ="${TEST_FREQ:-10}"
export VAL_BEFORE_TRAIN=true
export VAL_DO_SAMPLE=false
export VAL_N=1
export SAVE_FREQ="${SAVE_FREQ:-20}"

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

# Inject GSPO + total_training_steps as extra Hydra overrides appended to the
# daily_omni.sh python invocation via "$@".
bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  trainer.total_training_steps=$TOTAL_TRAINING_STEPS
