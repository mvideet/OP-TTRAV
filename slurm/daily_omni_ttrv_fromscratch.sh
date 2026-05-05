#!/bin/bash
#SBATCH -J do_ttrv_fs
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/do_ttrv_fs_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/do_ttrv_fs_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --exclude=sls-a6-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=40:00:00
#SBATCH --requeue

# TTRV full-dataset run on OmniVideo train.json (2018 samples), from base
# weights (step 0). Uses the same launch pipeline as daily_omni.sh but via
# the TTRV freq+entropy reward (TTRL_TASK_TYPE=mcq_freq_entropy).

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export WANDB_MODE=online

export TTRL_DEBUG=1
export TTRV_DEBUG=1
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400

# Full dataset (no SANITY_CHECK) -> 2018 train samples from verl/data/OmniVideo/train.json.
export RESUME_PATH=""

# Full dataset: 2018 samples / batch=4 = 504 steps/epoch. Cap at 280
# training steps (matches the step-280 checkpoint from the noreason run).
export EPISODE="${EPISODE:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-280}"

export VIDEO_FPS=0.5
export AUDIO_SAMPLE_RATE=8000
export TEST_FREQ="${TEST_FREQ:-10}"
export VAL_BEFORE_TRAIN=true
export VAL_DO_SAMPLE=false
export VAL_N=1
export SAVE_FREQ="${SAVE_FREQ:-20}"

export MCQ_FREQ_ALPHA="${MCQ_FREQ_ALPHA:-0.1}"

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni_ttrv_fromscratch.sh || bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni_ttrv.sh
