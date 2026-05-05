#!/bin/bash
#SBATCH -J do_cwa
#SBATCH -o slurm/out/daily_omni_cwa_%j.out
#SBATCH -e slurm/err/daily_omni_cwa_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --requeue

# Daily OmniVideo MCQ TTRL with enhancements:
#   1. Confidence-weighted advantage (exp)
#   2. Multiple-attempts sampling

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
export EPISODE=1
export SAVE_FREQ=300
export MAX_CKPT=1

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh \
  algorithm.confidence_weighted_advantage.enable=true \
  ttrl.multi_attempt_sampling.enable=true \
  ttrl.multi_attempt_sampling.max_attempts=3 \
  trainer.total_training_steps=300
