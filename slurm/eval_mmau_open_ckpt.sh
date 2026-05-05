#!/bin/bash
#SBATCH -J eval_mmau_oe
#SBATCH -o slurm/out/eval_mmau_oe_%j.out
#SBATCH -e slurm/err/eval_mmau_oe_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --requeue

# Evaluate the open-ended TTRL checkpoint on MCQ MMAU benchmark.
# Uses the freeze-copied step 9350 (= real step 350) to avoid race conditions.

mkdir -p slurm/out slurm/err

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

CKPT_DIR="/data/sls/scratch/mvideet/TTRL/verl/checkpoints/TTRL-verl/MMAU-Qwen2.5-Omni-3B/0412/TTRL-MMAU-Open-grpo-215504"
TEST_FILE="verl/data/MMAU/test_mini.json"
BASE_MODEL="/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B"
OUTPUT="results_mmau_open_ckpt_$(date +%m%d_%H%M).csv"

# Evaluate freeze-copied step 9350 (real step 350) + baseline
STEPS="9350"

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
    \
    --max-new-tokens 512 \
    --sample-rate 16000 \
    --max-audio-duration 30.0 \
    --eval-n ${EVAL_N:-4} \
    --eval-temperature ${EVAL_TEMPERATURE:-0.6}

echo "Done. Results in ${OUTPUT}"
