#!/bin/bash
#SBATCH -J daily_omni_txt_omni
#SBATCH -o slurm/out/daily_omni_text_omni_%j.out
#SBATCH -e slurm/err/daily_omni_text_omni_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --requeue

# Qwen2.5-Omni, text-only, same OmniVideo JSON as daily_omni (video paths ignored).

mkdir -p slurm/out slurm/err

export WANDB_MODE=online
export TTRL_DEBUG=1

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni_text_omni.sh
