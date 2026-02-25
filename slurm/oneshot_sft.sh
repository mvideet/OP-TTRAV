#!/bin/bash
#SBATCH -J oneshot_sft
#SBATCH -o slurm/out/oneshot_sft_%j.out
#SBATCH -e slurm/err/oneshot_sft_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --requeue

mkdir -p slurm/out slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# LlamaFactory requires accelerate<=1.11.0; env has 1.12.0 (needed by verl). Skip version check.
export DISABLE_VERSION_CHECK=1

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl310

# SLURM_SUBMIT_DIR = cwd when you ran sbatch. Run: cd /path/to/TTRL && sbatch slurm/oneshot_sft.sh
cd "${SLURM_SUBMIT_DIR:?Must run sbatch from repo root (cd /path/to/TTRL && sbatch slurm/oneshot_sft.sh)}" || exit 1

cd LlamaFactory || { echo "LlamaFactory not found in $(pwd)"; exit 1; }

# Use python -m since llamafactory-cli may not be installed in env
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python -m llamafactory.cli train examples/train_lora/qwen2.5_omni_3b_lora_sft_oneshot.yaml
