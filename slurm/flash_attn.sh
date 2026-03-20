#!/bin/bash
#SBATCH -J flash_attn
#SBATCH -o slurm/out/flash_attn_%j.out
#SBATCH -e slurm/err/flash_attn_%j.err
#SBATCH --qos=regular
#SBATCH --partition=a5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --requeue

echo "Starting verl312 environment setup..."

# CUDA_HOME required for any package that builds flash-attn from source (e.g. liger-kernel)
if [ -z "$CUDA_HOME" ]; then
  for cuda in /usr/local/cuda /opt/cuda /usr/lib/cuda; do
    if [ -d "$cuda" ]; then
      export CUDA_HOME="$cuda"
      echo "Set CUDA_HOME=$CUDA_HOME"
      break
    fi
  done
  if [ -z "$CUDA_HOME" ]; then
    echo "WARNING: CUDA_HOME not set and no default path found. Builds may fail."
  fi
fi

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

REPO_ROOT="/data/sls/r/u/mvideet/TTRL"
cd "$REPO_ROOT"

# 1. PyTorch with CUDA 12.x
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Flash Attention (prebuilt wheel for torch 2.9 + Python 3.12 - avoids CUDA build)
echo "Installing Flash Attention..."
pip uninstall -y flash-attn 2>/dev/null || true
wget -nc -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl -P /tmp/
pip install /tmp/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 3. vLLM >= 0.16.0 (required for Qwen2.5-Omni, OmniVideoText with VLLM rollout)
echo "Installing vLLM..."
pip install "vllm>=0.16.0"

# 4. verl + geo. Use [geo] only; vllm extra pins vllm<=0.8.5, we already have vllm>=0.16.
echo "Installing verl..."
pip install -e ./verl[geo]

# 5. liger-kernel (may trigger flash-attn build if not satisfied; CUDA_HOME must be set)
echo "Installing liger-kernel..."
pip install liger-kernel

# 6. Verify
echo "Verifying installation..."
python - <<'EOF'
import torch
import flash_attn
import vllm
import verl
print("PyTorch:", torch.__version__)
print("FlashAttention: OK")
print("vLLM:", vllm.__version__)
print("verl: OK")
EOF

echo "Installation complete."