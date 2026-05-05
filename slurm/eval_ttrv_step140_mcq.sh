#!/bin/bash
#SBATCH -J eval_ttrv140
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/eval_ttrv140_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/eval_ttrv140_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --requeue

# MCQ mean@4 eval of the TTRV-trained checkpoint at step 140 (job 88897).
# Step 0 baseline already in results_daily_omni_noreason_baseline_0416_1259.csv
# (overall 0.5818) — we skip re-running it.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

CKPT_DIR="${CKPT_DIR:-/data/sls/scratch/mvideet/TTRL/verl/checkpoints/saved/ttrv_0422}"
TEST_FILE="${TEST_FILE:-verl/data/OmniVideo/test.json}"
BASE_MODEL="${BASE_MODEL:-/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B}"
OUTPUT="${OUTPUT:-results_ttrv_step140_mcq_$(date +%m%d_%H%M).csv}"

STEPS="${STEPS:-140}"
EVAL_N="${EVAL_N:-4}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"

echo "Checkpoint dir: ${CKPT_DIR}"
echo "Test file: ${TEST_FILE}"
echo "Output: ${OUTPUT}"
echo "Steps: ${STEPS}"
echo "Mean@N: ${EVAL_N} at T=${EVAL_TEMPERATURE}"

python verl/scripts/eval_mmau_offline.py \
    --ckpt-dir "${CKPT_DIR}" \
    --test-file "${TEST_FILE}" \
    --base-model "${BASE_MODEL}" \
    --steps ${STEPS} \
    --output "${OUTPUT}" \
    --max-new-tokens 512 \
    --sample-rate 16000 \
    --max-audio-duration 30.0 \
    --video-fps 0.5 \
    --video-max-frames 32 \
    --use-audio-in-video \
    --eval-n ${EVAL_N} \
    --eval-temperature ${EVAL_TEMPERATURE}

echo "Done. Results in ${OUTPUT}"
