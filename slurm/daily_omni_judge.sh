#!/bin/bash
#SBATCH -J do_judge
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/do_judge_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/do_judge_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --exclude=sls-a6-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --requeue

# LLM-as-judge open-ended TTRL on the full OmniVideo train_open.json
# (2018 free-text samples), 500 steps from base weights. The same policy
# acts as judge after BGE-medoid voting selects the pseudo-GT per prompt.
# Confidence gate disabled per experiment spec.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export WANDB_MODE=online

export TTRL_DEBUG=1
export TTRL_JUDGE_DEBUG=1
export TTRL_OE_DEBUG=0
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400

export TTRL_TASK_TYPE=judge_open_ended

# Confidence gate OFF (different experiment, isolated to judge-as-reward).
export TTRL_CG_ENABLE=0

# Judge knobs.
export TTRL_JUDGE_MAX_NEW_TOKENS="${TTRL_JUDGE_MAX_NEW_TOKENS:-8}"
export TTRL_JUDGE_MAX_PROMPT_LEN="${TTRL_JUDGE_MAX_PROMPT_LEN:-4096}"
export TTRL_JUDGE_NEUTRAL_FALLBACK="${TTRL_JUDGE_NEUTRAL_FALLBACK:-0.5}"

# Train horizon.
export EPISODE="${EPISODE:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"

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

bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni_judge.sh \
  trainer.total_training_steps=$TOTAL_TRAINING_STEPS
