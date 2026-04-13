#!/bin/bash
#SBATCH -J eval_omnibench
#SBATCH -o slurm/out/eval_omnibench_%j.out
#SBATCH -e slurm/err/eval_omnibench_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --requeue

mkdir -p slurm/out slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

# OmniBench: image + audio QA, 1142 samples
TEST_FILE="${TEST_FILE:-verl/data/OmniBench/test.json}"
BASE_MODEL="${BASE_MODEL:-/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B}"
OUTPUT="${OUTPUT:-results_omnibench_$(date +%m%d_%H%M).csv}"

# For baseline-only eval, set CKPT_DIR to a dummy path and STEPS to empty
CKPT_DIR="${CKPT_DIR:-/data/sls/scratch/mvideet/TTRL/verl/checkpoints/TTRL-verl/dummy}"
STEPS="${STEPS:-}"

echo "Test file: ${TEST_FILE}"
echo "Output: ${OUTPUT}"
echo "Eval N: ${EVAL_N:-4}"

python verl/scripts/eval_mmau_offline.py \
    --ckpt-dir "${CKPT_DIR}" \
    --test-file "${TEST_FILE}" \
    --base-model "${BASE_MODEL}" \
    ${STEPS:+--steps ${STEPS}} \
    --output "${OUTPUT}" \
    --eval-baseline \
    --max-new-tokens 512 \
    --sample-rate 16000 \
    --max-audio-duration 30.0 \
    --eval-n ${EVAL_N:-4} \
    --eval-temperature ${EVAL_TEMPERATURE:-0.6} \
    --category-key task_type

echo "Done. Results in ${OUTPUT}"
