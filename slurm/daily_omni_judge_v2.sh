#!/bin/bash
#SBATCH -J do_judge_v2
#SBATCH -o /data/sls/scratch/mvideet/TTRL/slurm/out/do_judge_v2_%j.out
#SBATCH -e /data/sls/scratch/mvideet/TTRL/slurm/err/do_judge_v2_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --exclude=sls-a6-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=22:00:00
#SBATCH --requeue

# LLM-as-judge OE TTRL with refparams from the WORKING daily_omni TTRV
# 89299 (+20.6 val pts on MCQ).
#
# What changed vs original do_judge (89116):
#   * rollout.temperature   0.6 -> 1.0    (wider cluster spread; key 89299 win)
#   * data.max_response_len 512 -> 1024   (room for full reasoning chains)
#   * actor.optim.lr        unchanged at 2e-6 (was always this; works)
#   * lr_warmup_steps_ratio 0.005 unchanged
#
# Hypothesis: T=0.6 caused 89116's failure (too tight rollouts -> model
# converged to its own template -> judge bimodal scoring hardened
# template -> -29% BLEU). T=1.0 gives the judge actually-different
# candidates to grade so the gradient pushes toward gold-aligned
# content rather than just self-similarity.

mkdir -p /data/sls/scratch/mvideet/TTRL/slurm/out /data/sls/scratch/mvideet/TTRL/slurm/err

export WANDB_MODE=online

export TTRL_DEBUG=1
export TTRL_JUDGE_DEBUG=1
export TTRL_OE_DEBUG=0
export OMNI_INPUT_DEBUG=1
export OMNI_INPUT_LOG_LIMIT=0
export OMNI_INPUT_LOG_MAX_Q_CHARS=400

export TTRL_TASK_TYPE=judge_open_ended

# Confidence gate OFF (judge variant doesn't combine with CG).
export TTRL_CG_ENABLE=0

# Judge knobs.
export TTRL_JUDGE_MAX_NEW_TOKENS="${TTRL_JUDGE_MAX_NEW_TOKENS:-8}"
export TTRL_JUDGE_MAX_PROMPT_LEN="${TTRL_JUDGE_MAX_PROMPT_LEN:-4096}"
export TTRL_JUDGE_NEUTRAL_FALLBACK="${TTRL_JUDGE_NEUTRAL_FALLBACK:-0.5}"

# Train horizon — fits ~17-19h at ~5min/step.
export EPISODE="${EPISODE:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"

export VIDEO_FPS=0.5
export AUDIO_SAMPLE_RATE=8000
export TEST_FREQ="${TEST_FREQ:-10}"
export VAL_BEFORE_TRAIN=true
export VAL_DO_SAMPLE=false
export VAL_N=1
export SAVE_FREQ="${SAVE_FREQ:-25}"

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

find "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/verl" -name "*.pyc" -delete 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

# Hydra overrides for the four hparam swaps from 89299. The launcher
# hardcodes T=0.6 and max_response_length=512; we override here.
# Also need ppo_max_token_len_per_gpu since launcher computes it from
# bash MAX_RESPONSE_LENGTH=512.
bash verl/examples/ttrl/Qwen2.5-Omni/daily_omni_judge.sh \
  trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
  actor_rollout_ref.rollout.temperature=1.0 \
  data.max_response_length=1024 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=11024
