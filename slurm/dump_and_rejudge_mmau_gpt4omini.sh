#!/bin/bash
#SBATCH -J dump_rejdg_mmau
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/dump_rejdg_mmau_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/dump_rejdg_mmau_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --requeue

# MMAU final eval: dump rollouts for step 0 (base) + step 250 (judge_v2 final)
# on full 1000-sample MMAU test, then re-judge with GPT-4o-mini (external,
# no self-bias). Mirrors dump_and_rejudge_v2_full.sh that gave +3.0 on
# OmniVideo with same external judge.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

# load OpenAI key (chmod 600)
if [ -f /data/sls/r/u/mvideet/home/.openai_key ]; then
  source /data/sls/r/u/mvideet/home/.openai_key
fi

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

CKPT_DIR=/data/sls/scratch/mvideet/TTRL/verl/checkpoints/TTRL-verl/MMAU-Qwen2.5-Omni-3B/0502/TTRL-MMAU-JudgeV2-grpo-130612
TEST_FILE=verl/data/MMAU/test_mini_open.json
BASE_MODEL=/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B
TS=$(date +%m%d_%H%M)
ROLLOUTS_JSONL="rollouts_mmau_judge_v2_final_${TS}.jsonl"
JUDGED_JSONL="rollouts_mmau_judge_v2_final_gpt4omini_${TS}.jsonl"
JUDGED_CSV="results_mmau_judge_v2_final_gpt4omini_${TS}.csv"

echo "=========================================="
echo "STEP A: dump rollouts (step 0 + step 250 on MMAU 1000-sample)"
echo "=========================================="
python verl/scripts/dump_rollouts.py \
    --ckpt-dir "$CKPT_DIR" \
    --test-file "$TEST_FILE" \
    --base-model "$BASE_MODEL" \
    --steps 250 \
    --eval-baseline \
    --output "$ROLLOUTS_JSONL" \
    --max-new-tokens 1024 \
    --sample-rate 16000 \
    --max-audio-duration 30.0 \
    --eval-n 1 \
    --eval-temperature 0.6

if [ ! -s "$ROLLOUTS_JSONL" ]; then
  echo "ERROR: rollouts JSONL is empty, aborting"
  exit 1
fi
echo "Rollouts dumped: $(wc -l < $ROLLOUTS_JSONL) records (expect ~2000)"

echo ""
echo "=========================================="
echo "STEP B: judge with GPT-4o-mini (external, no self-bias)"
echo "=========================================="
if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY not set, aborting judge step"
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
