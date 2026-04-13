#!/bin/bash
#SBATCH -J daily_omni_open
#SBATCH -o slurm/out/daily_omni_open_%j.out
#SBATCH -e slurm/err/daily_omni_open_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --requeue

mkdir -p slurm/out slurm/err

export WANDB_MODE=online

# Verbose: TTRL vote dumps + open-ended reward debug + per-sample Omni metadata
export TTRL_DEBUG=1
export TTRL_OE_DEBUG=1
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400

# Open-ended TTRL is always self-supervised; no GT label toggle.

export VIDEO_FPS=0.5
export AUDIO_SAMPLE_RATE=8000

# Periodic eval cadence + baseline
export TEST_FREQ=5
export VAL_BEFORE_TRAIN=true
export VAL_DO_SAMPLE=false
export VAL_N=1

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni_open.sh
