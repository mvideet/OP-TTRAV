#!/bin/bash
#SBATCH -J doo_sanity
#SBATCH -o slurm/out/doo_sanity_%j.out
#SBATCH -e slurm/err/doo_sanity_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --requeue

mkdir -p slurm/out slurm/err

export WANDB_MODE=online

# Verbose for sanity check: dump every group's pairwise sims and medoid choice
export TTRL_DEBUG=1
export TTRL_OE_DEBUG=1
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400

# 40-sample sanity subset
export SANITY_CHECK=1
export N_SANITY=40

# Stop early - sanity is just for collapse/health verification, not full training
export EPISODE=2

# Frequent val so we can see if collapse starts
export VIDEO_FPS=0.5
export AUDIO_SAMPLE_RATE=8000
export TEST_FREQ=2
export VAL_BEFORE_TRAIN=true
export VAL_DO_SAMPLE=false
export VAL_N=1
export SAVE_FREQ=999  # don't save during sanity

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni_open.sh
