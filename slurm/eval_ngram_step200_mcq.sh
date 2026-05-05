#!/bin/bash
#SBATCH -J eval_ngram200
#SBATCH -o slurm/out/eval_ngram200_%j.out
#SBATCH -e slurm/err/eval_ngram200_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --requeue

# MCQ eval of the n-gram-trained checkpoint at step 200.
# (Step 0 baseline was already produced by
#  results_daily_omni_noreason_baseline_0416_1259.csv — we do not re-run it.)
#
# The checkpoint was trained open-ended (free-text answers), so we force MCQ
# format at eval time via the suffix prompt and grade by \boxed{[A-D]} exact
# match (eval_mmau_offline.extract_answer / score_response).

mkdir -p slurm/out slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

# Safe copy of the live training checkpoint (preserves step 200 against
# max_actor_ckpt_to_keep pruning / continued training overwrites).
CKPT_DIR="${CKPT_DIR:-/data/sls/scratch/mvideet/TTRL/verl/checkpoints/saved/ngram_0416}"
TEST_FILE="${TEST_FILE:-verl/data/OmniVideo/test.json}"
BASE_MODEL="${BASE_MODEL:-/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B}"
OUTPUT="${OUTPUT:-results_ngram_step200_mcq_$(date +%m%d_%H%M).csv}"

STEPS="${STEPS:-200}"
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
