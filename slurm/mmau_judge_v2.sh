#!/bin/bash
#SBATCH -J mmau_jdg_v2
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/mmau_jdg_v2_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/mmau_jdg_v2_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --exclude=sls-a6-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=22:00:00
#SBATCH --requeue

# LLM-as-judge OE TTRL on MMAU (audio-only). Replicates do_judge_v2
# (89391) recipe — +9 LLM-judge on OmniVideo. If MMAU shows similar
# gain, the v2 recipe generalizes across modalities.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export WANDB_MODE=online

export TTRL_DEBUG=1
export TTRL_JUDGE_DEBUG=1
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400

export TTRL_TASK_TYPE=judge_open_ended
export TTRL_CG_ENABLE=0

export EPISODE="${EPISODE:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
export TEST_FREQ="${TEST_FREQ:-10}"
export VAL_BEFORE_TRAIN=true
export VAL_DO_SAMPLE=false
export VAL_N=1
export SAVE_FREQ="${SAVE_FREQ:-25}"
export AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-16000}"

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

bash verl/examples/ttrl/Qwen2.5-Omni/mmau_judge_v2.sh
