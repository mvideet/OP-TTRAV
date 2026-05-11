#!/bin/bash
#SBATCH -J diag_grad
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/diag_grad_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/diag_grad_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00

# Single-GPU gradient/forward-pass diagnostics for the Qwen2.5-Omni training path.
# Tests:
#   T1: log_probs match between dense and use_remove_padding=True forward
#   T2: forward+backward produces real gradients + params actually move
#   T3: response-only loss mask produces meaningful gradient

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

cd /data/sls/u/urop/mvideet/TTRL || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

python verl/scripts/diagnose_grad_path.py \
    --base-model /data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B \
    --test-file verl/data/OmniVideo/test_open_val20.json \
    --sample-index 0
