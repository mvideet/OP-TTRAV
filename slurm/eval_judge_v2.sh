#!/bin/bash
#SBATCH -J eval_jdgv2
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/eval_jdgv2_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/eval_jdgv2_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --requeue

# LLM-as-judge eval of do_judge_v2 (89391) checkpoints. Compares
# step 0 (base) vs the trained checkpoint(s) on test_open_val20 (or
# a 200-sample slice of test_open). Judge = the same Qwen2.5-Omni-3B
# model used for training, run text-only against gold answer_text.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

CKPT_DIR="${CKPT_DIR:-/data/sls/scratch/mvideet/TTRL/verl/checkpoints/saved/judge_v2_0430}"
TEST_FILE="${TEST_FILE:-verl/data/OmniVideo/test_open.json}"
BASE_MODEL="${BASE_MODEL:-/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B}"
OUTPUT="${OUTPUT:-results_judge_v2_eval_$(date +%m%d_%H%M).csv}"
STEPS="${STEPS:-200}"
N_SAMPLES="${N_SAMPLES:-100}"
EVAL_BASELINE="${EVAL_BASELINE:---eval-baseline}"

echo "Checkpoint dir: ${CKPT_DIR}"
echo "Test file: ${TEST_FILE}"
echo "Steps: ${STEPS} (+baseline=${EVAL_BASELINE})"
echo "N samples: ${N_SAMPLES}"
echo "Output: ${OUTPUT}"

python verl/scripts/eval_open_ended_judge.py \
    --ckpt-dir "${CKPT_DIR}" \
    --test-file "${TEST_FILE}" \
    --base-model "${BASE_MODEL}" \
    --steps ${STEPS} \
    ${EVAL_BASELINE} \
    --output "${OUTPUT}" \
    --max-new-tokens 1024 \
    --judge-max-new-tokens 8 \
    --sample-rate 16000 \
    --max-audio-duration 30.0 \
    --video-fps 0.5 \
    --video-max-frames 32 \
    --use-audio-in-video \
    --eval-n 1 \
    --eval-temperature 0.6 \
    --n-samples ${N_SAMPLES}

echo "Done. Results in ${OUTPUT}"
