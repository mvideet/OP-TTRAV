#!/bin/bash
#SBATCH -J eval_daily_full
#SBATCH -o slurm/out/eval_daily_full_%j.out
#SBATCH -e slurm/err/eval_daily_full_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
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

CKPT_DIR="${CKPT_DIR:-/data/sls/scratch/mvideet/TTRL/verl/checkpoints/TTRL-verl/OmniVideo-Qwen2.5-Omni-3B/0408/TTRL-Omni-grpo-221124}"
TEST_FILE="${TEST_FILE:-verl/data/OmniVideo/test.json}"
BASE_MODEL="${BASE_MODEL:-/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B}"
OUTPUT="${OUTPUT:-results_daily_omni_full_$(date +%m%d_%H%M).csv}"

# Full eval: baseline + latest checkpoint, all 2018 samples
# At ~6s/sample, 2018 samples × 2 checkpoints = ~6.7h total
STEPS="${STEPS:-140}"

echo "Checkpoint dir: ${CKPT_DIR}"
echo "Test file: ${TEST_FILE}"
echo "Output: ${OUTPUT}"
echo "Steps: ${STEPS}"
echo "N samples: FULL (2018)"

python verl/scripts/eval_mmau_offline.py \
    --ckpt-dir "${CKPT_DIR}" \
    --test-file "${TEST_FILE}" \
    --base-model "${BASE_MODEL}" \
    --steps ${STEPS} \
    --output "${OUTPUT}" \
    --eval-baseline \
    --max-new-tokens 512 \
    --sample-rate 16000 \
    --max-audio-duration 30.0 \
    --video-fps 0.5 \
    --video-max-frames 32 \
    --use-audio-in-video \
    --eval-n ${EVAL_N:-4} \
    --eval-temperature ${EVAL_TEMPERATURE:-0.6}

echo "Done. Results in ${OUTPUT}"
