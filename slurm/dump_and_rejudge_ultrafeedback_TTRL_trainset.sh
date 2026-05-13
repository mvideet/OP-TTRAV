#!/bin/bash
#SBATCH -J uf_eval
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/uf_eval_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/uf_eval_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --requeue

# UltraFeedback TTRL-native eval (same prompts model was trained on). Parametrized by:
#   CKPT_DIR    - the training run's ckpt directory (required)
#   EVAL_STEP   - the global_step_X to evaluate (required)
#   EVAL_TAG    - short label for output filenames (default: timestamp)
#   N_SAMPLES   - test-set size (default: 200 of 500)

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

if [ -f /data/sls/r/u/mvideet/home/.openai_key ]; then
  source /data/sls/r/u/mvideet/home/.openai_key
fi

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

: "${CKPT_DIR:?CKPT_DIR env var must be set}"
: "${EVAL_STEP:?EVAL_STEP env var must be set}"
EVAL_TAG="${EVAL_TAG:-$(date +%m%d_%H%M)}"
N_SAMPLES="${N_SAMPLES:-200}"

TEST_FILE=verl/data/UltraFeedback-TTRL/train.json
BASE_MODEL=/data/sls/scratch/mvideet/models/Qwen2.5-3B

ROLLOUTS_JSONL="rollouts_uf_${EVAL_TAG}.jsonl"
JUDGED_JSONL="rollouts_uf_gpt4omini_${EVAL_TAG}.jsonl"
JUDGED_CSV="results_uf_gpt4omini_${EVAL_TAG}.csv"

echo "=========================================="
echo "STEP A: dump rollouts"
echo "  ckpt_dir: $CKPT_DIR"
echo "  eval_step: $EVAL_STEP (+ step 0 baseline)"
echo "  test_file: $TEST_FILE"
echo "  n_samples: $N_SAMPLES of 500"
echo "=========================================="
python verl/scripts/dump_rollouts.py \
    --ckpt-dir "$CKPT_DIR" \
    --test-file "$TEST_FILE" \
    --base-model "$BASE_MODEL" \
    --steps "$EVAL_STEP" \
    --eval-baseline \
    --n-samples "$N_SAMPLES" \
    --output "$ROLLOUTS_JSONL" \
    --max-new-tokens 1024 \
    --eval-n 1 \
    --eval-temperature 0.6 \
    --gold-key answer

if [ ! -s "$ROLLOUTS_JSONL" ]; then
  echo "ERROR: rollouts JSONL is empty, aborting"
  exit 1
fi
echo "Rollouts dumped: $(wc -l < $ROLLOUTS_JSONL) records (expect ~$(($N_SAMPLES * 2)))"

echo ""
echo "=========================================="
echo "STEP B: judge with GPT-4o-mini"
echo "=========================================="
if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY not set"
  exit 1
fi

python verl/scripts/judge_rollouts_jsonl.py \
    --rollouts "$ROLLOUTS_JSONL" \
    --judge-mode openai \
    --judge-model gpt-4o-mini-2024-07-18 \
    --output "$JUDGED_JSONL" \
    --csv-output "$JUDGED_CSV" \
    --judge-max-new-tokens 8

echo ""
echo "Done."
echo "  rollouts:    $ROLLOUTS_JSONL"
echo "  judged:      $JUDGED_JSONL"
echo "  aggregate:   $JUDGED_CSV"
