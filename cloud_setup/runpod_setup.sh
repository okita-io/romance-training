#!/bin/bash
# RunPod GPU Setup Script
# Run this on a RunPod instance with RTX 4090 or A6000

set -e

echo "=== RunPod Training Environment Setup ==="
echo ""

# Update system
apt-get update
apt-get install -y git wget vim

# Install Python dependencies
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Install additional dependencies
pip install datasets transformers huggingface-hub tokenizers

# Clone the repository (if needed) or upload via runpodctl
echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your romance-factory directory using runpodctl or scp"
echo "2. cd /workspace/romance-factory"
echo "3. python3 train_qwen_unsloth.py"
echo ""
echo "Or use this one-liner to download from your repo:"
echo "  git clone YOUR_REPO_URL /workspace/romance-factory"
