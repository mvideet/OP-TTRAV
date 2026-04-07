#!/bin/bash
#SBATCH -J daily_omni
#SBATCH -o slurm/out/daily_omni_%j.out
#SBATCH -e slurm/err/daily_omni_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --requeue

mkdir -p slurm/out slurm/err

export WANDB_MODE=online

export TTRL_DEBUG=1
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400
export TRAIN_ON_GT_LABELS=0
export TEST_FREQ=-1
export VAL_BEFORE_TRAIN=false
export VIDEO_FPS=0.5
export AUDIO_SAMPLE_RATE=8000
export VAL_DO_SAMPLE=false
export VAL_N=1

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh
