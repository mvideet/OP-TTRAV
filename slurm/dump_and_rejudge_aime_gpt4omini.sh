#!/bin/bash
#SBATCH -J dump_rejdg_aime
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/dump_rejdg_aime_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/dump_rejdg_aime_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --requeue

# AIME GPT-4o-mini eval: dump rollouts at step 0 (base) + step 100/200/300
# (cluster-vote trained checkpoints) on the full 30-sample AIME test set,
# then judge with GPT-4o-mini. Mirrors dump_and_rejudge_mmau_gpt4omini.sh
# which gave the +3.2 result on MMAU.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

if [ -f /data/sls/r/u/mvideet/home/.openai_key ]; then
  source /data/sls/r/u/mvideet/home/.openai_key
fi

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

CKPT_DIR=/data/sls/scratch/mvideet/TTRL/verl/checkpoints/TTRL-verl/AIME-TTT-Qwen2.5-Math-1.5B/0509/TTRL-AIME-SimpleCluster-grpo-191323
TEST_FILE=verl/data/AIME-TTT/test.json
BASE_MODEL=/data/sls/scratch/mvideet/models/Qwen2.5-Math-1.5B
TS=$(date +%m%d_%H%M)
ROLLOUTS_JSONL="rollouts_aime_simple_cluster_${TS}.jsonl"
JUDGED_JSONL="rollouts_aime_simple_cluster_gpt4omini_${TS}.jsonl"
JUDGED_CSV="results_aime_simple_cluster_gpt4omini_${TS}.csv"

echo "=========================================="
echo "STEP A: dump rollouts (step 0 + 100/200/300 on AIME 30-sample)"
echo "=========================================="
python verl/scripts/dump_rollouts.py \
    --ckpt-dir "$CKPT_DIR" \
    --test-file "$TEST_FILE" \
    --base-model "$BASE_MODEL" \
    --steps 100 200 300 \
    --eval-baseline \
    --output "$ROLLOUTS_JSONL" \
    --max-new-tokens 2048 \
    --eval-n 1 \
    --eval-temperature 0.6 \
    --gold-key answer \
    --suffix-prompt $'\nPlease reason step by step, and put your final answer within \\boxed{}.'

if [ ! -s "$ROLLOUTS_JSONL" ]; then
  echo "ERROR: rollouts JSONL is empty, aborting"
  exit 1
fi
echo "Rollouts dumped: $(wc -l < $ROLLOUTS_JSONL) records (expect ~120 = 30 prompts × 4 steps)"

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
