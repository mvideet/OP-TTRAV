#!/bin/bash
#SBATCH -J ah_pair
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/ah_pair_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/ah_pair_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

# Arena-Hard pairwise eval using GPT-4o-mini judge with the official
# arena-hard-auto judge prompt. Compares (our_model_step0, our_model_stepX)
# vs the GPT-4.1 baseline. No GPU needed - just API calls.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

if [ -f /data/sls/r/u/mvideet/home/.openai_key ]; then
  source /data/sls/r/u/mvideet/home/.openai_key
fi

cd /data/sls/u/urop/mvideet/TTRL || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

# Inputs (env-overridable)
: "${ROLLOUTS_JSONL:=/data/sls/u/urop/mvideet/TTRL/rollouts_uf_ah_medoid_step300.jsonl}"
: "${BASELINE_JSONL:=/data/sls/scratch/mvideet/datasets/arena-hard-v2.0/data/arena-hard-v2.0/model_answer/gpt-4.1.jsonl}"
: "${QUESTIONS_JSON:=/data/sls/u/urop/mvideet/TTRL/verl/data/ArenaHard-v2.0-TTRL/test.json}"
: "${JUDGE_MODEL:=gpt-4o-mini-2024-07-18}"
: "${EVAL_TAG:=$(date +%m%d_%H%M)}"

echo "=========================================="
echo "Arena-Hard PAIRWISE eval (canonical methodology, GPT-4o-mini judge)"
echo "  model rollouts: $ROLLOUTS_JSONL"
echo "  baseline:       $BASELINE_JSONL"
echo "  questions:      $QUESTIONS_JSON"
echo "  judge:          $JUDGE_MODEL"
echo "=========================================="

for STEP in 0 300; do
  echo ""
  echo "--- Step $STEP ---"
  python verl/scripts/judge_arena_hard_pairwise.py \
    --model_rollouts "$ROLLOUTS_JSONL" \
    --baseline_rollouts "$BASELINE_JSONL" \
    --questions "$QUESTIONS_JSON" \
    --judge_model "$JUDGE_MODEL" \
    --step $STEP \
    --our_label "qwen2.5-3b-base+TTRL_step$STEP" \
    --baseline_label "gpt-4.1" \
    --output "ah_pairwise_step${STEP}_${EVAL_TAG}.json"
done

echo ""
echo "Done. Output files:"
ls -la ah_pairwise_step*_${EVAL_TAG}.json
