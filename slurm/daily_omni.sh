#!/bin/bash
#SBATCH -J daily_omni
#SBATCH -o slurm/out/daily_omni_%j.out
#SBATCH -e slurm/err/daily_omni_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --requeue

# Ensure log dirs exist (some clusters do not create them)
mkdir -p slurm/out slurm/err

# W&B: online mode for cloud sync. Set WANDB_API_KEY (e.g. export WANDB_API_KEY=xxx or wandb login).
export WANDB_MODE=online

export SANITY_CHECK=1
export CONTENT_PARENT_CATEGORY=Education
export N_SANITY=320
export TTRL_DEBUG=1
export VAL_DEBUG=1

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

# Clear stale .pyc caches so restored/edited source files are always used
find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

# Run from repo root so paths in the training script resolve (e.g. custom_reward_function.path)
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni.sh
