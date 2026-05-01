#!/bin/bash
#SBATCH -J dump_rejdg_full
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/dump_rejdg_full_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/dump_rejdg_full_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --requeue

# Lockdown number: full 2018-sample fixed-base-judge eval of judge_v2
# (do_judge_v2 89391 step 200) vs base. Tightens the +9 LLM-judge result
# from 100-sample (sigma ~0.05) to full set (sigma ~0.011).
#
# Step A) Dump rollouts (no judging) for step 0 and step 200 on FULL
#         test_open.json (2018 samples). Greedy, eval_n=1.
# Step B) Re-judge BOTH rollout sets with FIXED base Qwen-Omni-3B.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

CKPT_DIR=/data/sls/scratch/mvideet/TTRL/verl/checkpoints/saved/judge_v2_0430
TEST_FILE=verl/data/OmniVideo/test_open.json
BASE_MODEL=/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B
TS=$(date +%m%d_%H%M)
ROLLOUTS_JSONL="rollouts_judge_v2_full_${TS}.jsonl"
JUDGED_JSONL="rollouts_judge_v2_full_basejudge_${TS}.jsonl"
JUDGED_CSV="results_judge_v2_full_basejudge_${TS}.csv"

echo "=========================================="
echo "STEP A: dump rollouts (full 2018, step 0 + step 200)"
echo "=========================================="
python verl/scripts/dump_rollouts.py \
    --ckpt-dir "$CKPT_DIR" \
    --test-file "$TEST_FILE" \
    --base-model "$BASE_MODEL" \
    --steps 200 \
    --eval-baseline \
    --output "$ROLLOUTS_JSONL" \
    --max-new-tokens 1024 \
    --sample-rate 16000 \
    --max-audio-duration 30.0 \
    --video-fps 0.5 \
    --video-max-frames 32 \
    --use-audio-in-video \
    --eval-n 1 \
    --eval-temperature 0.6

if [ ! -s "$ROLLOUTS_JSONL" ]; then
  echo "ERROR: rollouts JSONL is empty, aborting"
  exit 1
fi
echo "Rollouts dumped: $(wc -l < $ROLLOUTS_JSONL) records (expect ~4036)"

echo ""
echo "=========================================="
echo "STEP B: re-judge with FIXED base Qwen-Omni-3B"
echo "=========================================="
python verl/scripts/judge_rollouts_jsonl.py \
    --rollouts "$ROLLOUTS_JSONL" \
    --judge-mode local \
    --judge-model "$BASE_MODEL" \
    --output "$JUDGED_JSONL" \
    --csv-output "$JUDGED_CSV" \
    --judge-max-new-tokens 8

echo ""
echo "Done."
echo "  rollouts:    $ROLLOUTS_JSONL"
echo "  judged:      $JUDGED_JSONL"
echo "  aggregate:   $JUDGED_CSV"
