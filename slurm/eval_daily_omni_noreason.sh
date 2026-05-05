#!/bin/bash
#SBATCH -J eval_do_nr
#SBATCH -o slurm/out/eval_do_noreason_%j.out
#SBATCH -e slurm/err/eval_do_noreason_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --requeue

# Eval no-reasoning Daily Omni checkpoint on full test set, mean@4

mkdir -p slurm/out slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

CKPT_DIR="/data/sls/scratch/mvideet/TTRL/verl/checkpoints/TTRL-verl/OmniVideo-Qwen2.5-Omni-3B/0414/TTRL-Omni-grpo-141622"
TEST_FILE="verl/data/OmniVideo/test.json"
BASE_MODEL="/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B"
OUTPUT="results_daily_omni_noreason_baseline_$(date +%m%d_%H%M).csv"

STEPS="0"

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
    --suffix-prompt $'\nGive your final answer as exactly one of: \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}.' \
    --max-new-tokens 512 \
    --sample-rate 16000 \
    --max-audio-duration 30.0 \
    --video-fps 0.5 \
    --video-max-frames 32 \
    --use-audio-in-video \
    --eval-n 4 \
    --eval-temperature 0.6

echo "Done. Results in ${OUTPUT}"
