#!/bin/bash
#SBATCH -J rejdg_mmau_base
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/rejdg_mmau_base_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/rejdg_mmau_base_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --requeue

# Re-judge the MMAU step-0/step-250 rollouts (already dumped) with FIXED base
# Qwen2.5-Omni-3B as judge. Companion to GPT-4o-mini result (+3.2 pts) — gives
# the in-family judge baseline parallel to OmniVideo's +10.6 base-Qwen number.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

ROLLOUTS_JSONL=rollouts_mmau_judge_v2_final_0504_1707.jsonl
BASE_MODEL=/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B
TS=$(date +%m%d_%H%M)
JUDGED_JSONL="rollouts_mmau_judge_v2_final_basejudge_${TS}.jsonl"
JUDGED_CSV="results_mmau_judge_v2_final_basejudge_${TS}.csv"

if [ ! -s "$ROLLOUTS_JSONL" ]; then
  echo "ERROR: $ROLLOUTS_JSONL not found / empty"
  exit 1
fi

echo "Judging $(wc -l < $ROLLOUTS_JSONL) records with base Qwen-Omni-3B"

python verl/scripts/judge_rollouts_jsonl.py \
    --rollouts "$ROLLOUTS_JSONL" \
    --judge-mode local \
    --judge-model "$BASE_MODEL" \
    --output "$JUDGED_JSONL" \
    --csv-output "$JUDGED_CSV" \
    --judge-max-new-tokens 8

echo ""
echo "Done."
echo "  judged:    $JUDGED_JSONL"
echo "  aggregate: $JUDGED_CSV"
