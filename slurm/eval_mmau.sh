#!/bin/bash
#SBATCH -J eval_mmau
#SBATCH -o slurm/out/eval_mmau_%j.out
#SBATCH -e slurm/err/eval_mmau_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --requeue

mkdir -p slurm/out slurm/err

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

CKPT_DIR="${CKPT_DIR:-/data/sls/scratch/mvideet/TTRL/verl/checkpoints/TTRL-verl/MMAU-Qwen2.5-Omni-3B/0405/TTRL-MMAU-grpo-020022}"
TEST_FILE="${TEST_FILE:-verl/data/MMAU/test_mini_test.json}"
BASE_MODEL="${BASE_MODEL:-/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B}"
OUTPUT="${OUTPUT:-results_mmau_$(date +%m%d_%H%M).csv}"

# Evaluate every 10th step + first and last
STEPS="${STEPS:-0 5 10 20 30 40 50 60 70 80 90 100 120 140 160 180 195}"

echo "Checkpoint dir: ${CKPT_DIR}"
echo "Test file: ${TEST_FILE}"
echo "Output: ${OUTPUT}"
echo "Steps: ${STEPS}"

python verl/scripts/eval_mmau_offline.py \
    --ckpt-dir "${CKPT_DIR}" \
    --test-file "${TEST_FILE}" \
    --base-model "${BASE_MODEL}" \
    --steps ${STEPS} \
    --output "${OUTPUT}" \
    --eval-baseline \
    --max-new-tokens 512 \
    --sample-rate 16000 \
    --max-audio-duration 30.0

echo "Done. Results in ${OUTPUT}"
