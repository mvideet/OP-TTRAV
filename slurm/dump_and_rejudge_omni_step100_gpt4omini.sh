#!/bin/bash
#SBATCH -J dump_rejdg_omni
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/dump_rejdg_omni_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/dump_rejdg_omni_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --requeue

# daily_omni GPT-4o-mini eval: dump rollouts at step 0 (base) + step 100
# (89923 cluster-vote ckpt) on the 20-sample val set, then judge with
# GPT-4o-mini. Companion to dump_and_rejudge_aime_gpt4omini.sh.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

if [ -f /data/sls/r/u/mvideet/home/.openai_key ]; then
  source /data/sls/r/u/mvideet/home/.openai_key
fi

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

CKPT_DIR=/data/sls/scratch/mvideet/TTRL/verl/checkpoints/TTRL-verl/OmniVideo-Qwen2.5-Omni-3B/0510/TTRL-Omni-Judge-grpo-182601
TEST_FILE=verl/data/OmniVideo/test_open_val20.json
BASE_MODEL=/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B
TS=$(date +%m%d_%H%M)
ROLLOUTS_JSONL="rollouts_omni_simple_cluster_${TS}.jsonl"
JUDGED_JSONL="rollouts_omni_simple_cluster_gpt4omini_${TS}.jsonl"
JUDGED_CSV="results_omni_simple_cluster_gpt4omini_${TS}.csv"

echo "=========================================="
echo "STEP A: dump rollouts (step 0 + 100 on Omni val 20-sample)"
echo "=========================================="
python verl/scripts/dump_rollouts.py \
    --ckpt-dir "$CKPT_DIR" \
    --test-file "$TEST_FILE" \
    --base-model "$BASE_MODEL" \
    --steps 100 \
    --eval-baseline \
    --output "$ROLLOUTS_JSONL" \
    --max-new-tokens 1024 \
    --eval-n 1 \
    --eval-temperature 0.6 \
    --use-audio-in-video \
    --gold-key answer_text \
    --sample-rate 8000 \
    --max-audio-duration 30.0 \
    --video-fps 0.5 \
    --video-max-frames 32 \
    --suffix-prompt $'\nExplain your reasoning step by step, then give a concise answer to the question in 1-3 complete sentences.'

if [ ! -s "$ROLLOUTS_JSONL" ]; then
  echo "ERROR: rollouts JSONL is empty, aborting"
  exit 1
fi
echo "Rollouts dumped: $(wc -l < $ROLLOUTS_JSONL) records (expect ~40 = 20 prompts × 2 steps)"

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
