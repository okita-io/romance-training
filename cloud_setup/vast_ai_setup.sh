#!/bin/bash
# Vast.ai GPU Setup Script
# For instances with PyTorch/CUDA pre-installed

set -e

echo "=== Vast.ai Training Environment Setup ==="
echo ""

# Check CUDA version
nvidia-smi
echo ""

# Install Unsloth and dependencies
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install datasets transformers huggingface-hub tokenizers

echo ""
echo "Setup complete!"
echo ""
echo "Upload your data:"
echo "  scp -r data/romance_corpus user@instance:/workspace/"
echo "  scp train_qwen_unsloth.py user@instance:/workspace/"
echo ""
echo "Then run training:"
echo "  python3 train_qwen_unsloth.py"
