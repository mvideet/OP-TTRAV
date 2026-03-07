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

source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
conda activate verl312

REPO_ROOT="/data/sls/r/u/mvideet/TTRL"
cd "$REPO_ROOT"

# 1. PyTorch with CUDA 12.x
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Flash Attention (torch 2.9 wheel for Python 3.12 - must match PyTorch version)
echo "Installing Flash Attention..."
pip uninstall -y flash-attn 2>/dev/null || true
wget -nc -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl -P /tmp/
pip install /tmp/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 3. vLLM >= 0.16.0 (required for Qwen2.5-Omni use_audio_in_video)
echo "Installing vLLM..."
pip install "vllm>=0.16.0"

# 4. verl + extras (vllm, geo). Skip gpu extra to avoid pip overwriting our flash-attn wheel.
echo "Installing verl..."
pip install -e ./verl[vllm,geo]
pip install liger-kernel

# 5. Verify
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