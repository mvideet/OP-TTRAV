#!/bin/bash
#SBATCH -J mmau_omni
#SBATCH -o slurm/out/mmau_omni_%j.out
#SBATCH -e slurm/err/mmau_omni_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --requeue

mkdir -p slurm/out slurm/err

export WANDB_MODE=online

# Verbose debugging: full TTRL vote dumps + per-sample Omni metadata
export TTRL_DEBUG=1
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400

# TTRL majority-voting (not GT labels)
export TRAIN_ON_GT_LABELS=0

# Audio: native Whisper sample rate
export AUDIO_SAMPLE_RATE=16000

# Eval every 2 steps + baseline before training
export TEST_FREQ=2
export VAL_BEFORE_TRAIN=true
export VAL_DO_SAMPLE=false
export VAL_N=1

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

bash verl/examples/ttrl/Qwen2.5-Omni/mmau_omni.sh
