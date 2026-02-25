#!/bin/bash
#SBATCH -J pseudo_oneshot
#SBATCH -o slurm/out/pseudo_oneshot_%j.out
#SBATCH -e slurm/err/pseudo_oneshot_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --requeue

mkdir -p slurm/out slurm/err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl310

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

MODEL_PATH="/data/sls/scratch/mvideet/models/Qwen2.5-Omni-3B"
DATA_PATH="/data/sls/r/u/mvideet/TTRL/verl/data/OmniVideo/train.json"
OUTPUT_PATH="/data/sls/r/u/mvideet/TTRL/LlamaFactory/data/omnivideo_oneshot_sft.json"

python LlamaFactory/src/llamafactory/psuedolabel/oneshot.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --num_gpus 2 \
    --max_new_tokens 128 \
    --temperature 0.0
