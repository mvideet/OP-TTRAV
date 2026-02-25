#!/bin/bash
#SBATCH -J aime_ttrl
#SBATCH -o slurm/out/aime_ttrl_%j.out
#SBATCH -e slurm/err/aime_ttrl_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --requeue

mkdir -p slurm/out slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TTRL_TASK_TYPE=math

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl310

# Run from repo root
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

# Override GPU count to match allocation
export N_GPUS=8
export NNODES=1

bash verl/examples/ttrl/Qwen2.5-Math/aime.sh
