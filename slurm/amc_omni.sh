#!/bin/bash
#SBATCH -J amc_omni
#SBATCH -o slurm/out/amc_omni_%j.out
#SBATCH -e slurm/err/amc_omni_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --requeue

mkdir -p slurm/out slurm/err

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Math task: ttrl_math (required for AMC)
export TTRL_TASK_TYPE=math
export VAL_DEBUG=1

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

# TTRL math reward requires latex2sympy2-extended and math-verify
pip install -q latex2sympy2-extended math-verify

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ ! -d "${REPO_ROOT}/verl/verl" && -n "${SLURM_SUBMIT_DIR}" && -d "${SLURM_SUBMIT_DIR}/verl/verl" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
cd "${REPO_ROOT}" || exit 1
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"

unset ROCR_VISIBLE_DEVICES || true
export ROCR_VISIBLE_DEVICES=
export XDG_RUNTIME_DIR="/data/sls/scratch/mvideet/xdg_runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export RAY_TMPDIR="/data/sls/scratch/mvideet/ray_tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$RAY_TMPDIR"

export N_GPUS=4
export NNODES=1

bash verl/examples/ttrl/Qwen2.5-Omni/amc.sh "$@"
