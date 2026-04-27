#!/bin/bash
#SBATCH -J eval_judge_bleu
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/eval_judge_bleu_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/eval_judge_bleu_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --requeue

# Open-ended BLEU/ROUGE-L/keyword-hit eval of the do_judge run on
# OmniVideo's open-ended val (test_open_val20 by default).
#
# Required: a safe-copy parent dir containing global_step_* subdirs for
# every checkpoint to evaluate. Step 0 (base) is added via --eval-baseline.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

# Defaults (override via env when calling sbatch).
CKPT_DIR="${CKPT_DIR:-/data/sls/scratch/mvideet/TTRL/verl/checkpoints/saved/judge_0426}"
TEST_FILE="${TEST_FILE:-verl/data/OmniVideo/test_open_val20.json}"
BASE_MODEL="${BASE_MODEL:-/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B}"
OUTPUT="${OUTPUT:-results_judge_bleu_$(date +%m%d_%H%M).csv}"
STEPS="${STEPS:-300}"
N_SAMPLES="${N_SAMPLES:-200}"      # 200-sample slice of test_open by default; ignored if test_file is val20
EVAL_N="${EVAL_N:-1}"              # greedy single rollout
EVAL_BASELINE="${EVAL_BASELINE:---eval-baseline}"

echo "Checkpoint dir: ${CKPT_DIR}"
echo "Test file: ${TEST_FILE}"
echo "Steps: ${STEPS} (+baseline=${EVAL_BASELINE})"
echo "N samples cap: ${N_SAMPLES} | mean@${EVAL_N}"
echo "Output: ${OUTPUT}"

python verl/scripts/eval_open_ended_bleu.py \
    --ckpt-dir "${CKPT_DIR}" \
    --test-file "${TEST_FILE}" \
    --base-model "${BASE_MODEL}" \
    --steps ${STEPS} \
    ${EVAL_BASELINE} \
    --output "${OUTPUT}" \
    --max-new-tokens 512 \
    --sample-rate 16000 \
    --max-audio-duration 30.0 \
    --video-fps 0.5 \
    --video-max-frames 32 \
    --use-audio-in-video \
    --eval-n ${EVAL_N} \
    --eval-temperature 0.6 \
    --n-samples ${N_SAMPLES}

echo "Done. Results in ${OUTPUT}"
