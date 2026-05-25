#!/usr/bin/env bash
# Install Python dependencies for train_qwen_unsloth.py (Unsloth + PyTorch CUDA 12.1).
# Requires Python 3.10+ (Unsloth -> peft>=0.18).
# Usage:
#   chmod +x install_training_deps.sh
#   ./install_training_deps.sh
#   PYTHON=python3.10 ./install_training_deps.sh
# Create a venv at ./.venv and install into it:
#   CREATE_VENV=1 ./install_training_deps.sh
#   CREATE_VENV=1 VENV_PATH=.venv-trtllm PYTHON=python3.11 ./install_training_deps.sh
# Skip xformers (e.g. if wheels fail):
#   SKIP_XFORMERS=1 ./install_training_deps.sh
# PyTorch variant: default cu126 (pytorch.org stable). cu118 = CUDA 11.8 toolkit. cu121 legacy (avoid on Python 3.13+).
#   TORCH_CUDA=cu118 ./install_training_deps.sh
#   TORCH_CUDA=cu128 ./install_training_deps.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-.venv}"
VENV_ABS="$REPO_ROOT/$VENV_PATH"

if [[ "${CREATE_VENV:-0}" == "1" ]]; then
  BOOT_PY="${PYTHON:-python3}"
  if ! command -v "$BOOT_PY" &>/dev/null; then
    BOOT_PY="python"
  fi
  if [[ ! -x "$VENV_ABS/bin/python" ]]; then
    echo "Creating venv at $VENV_ABS (using $BOOT_PY)..."
    "$BOOT_PY" -m venv "$VENV_ABS"
  else
    echo "Using existing venv at $VENV_ABS"
  fi
  PY="$VENV_ABS/bin/python"
else
  PY="${PYTHON:-python3}"
  if ! command -v "$PY" &>/dev/null; then
    PY="python"
  fi
fi

pip_install() {
  "$PY" -m pip "$@"
}

if ! "$PY" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"; then
  echo ""
  echo "Unsloth needs Python 3.10 or newer (dependencies require peft>=0.18)."
  echo "Interpreter: $PY"
  "$PY" --version
  echo ""
  echo "Install Python 3.10+. If .venv was created with an old Python, remove it and run e.g.:"
  echo "  CREATE_VENV=1 PYTHON=python3.12 ./install_training_deps.sh"
  exit 1
fi

if ! "$PY" -c "import sys; raise SystemExit(0 if sys.version_info < (3, 13) else 1)"; then
  echo "WARN: Python 3.13+ is still maturing for some ML wheels. Prefer 3.12 if you hit NumPy/PyTorch errors." >&2
fi

echo "Using interpreter: $PY"
"$PY" --version

echo ""
echo "[1/6] Upgrading pip, setuptools, wheel..."
pip_install install --upgrade pip setuptools wheel

TORCH_CUDA="${TORCH_CUDA:-cu126}"
case "$TORCH_CUDA" in
  cu128) TORCH_URL="https://download.pytorch.org/whl/cu128" ;;
  cu126) TORCH_URL="https://download.pytorch.org/whl/cu126" ;;
  cu121) TORCH_URL="https://download.pytorch.org/whl/cu121" ;;
  cu118) TORCH_URL="https://download.pytorch.org/whl/cu118" ;;
  cpu)   TORCH_URL="https://download.pytorch.org/whl/cpu" ;;
  *)     TORCH_URL="https://download.pytorch.org/whl/cu126" ;;
esac
echo ""
echo "[2/6] Installing PyTorch (TORCH_CUDA=$TORCH_CUDA)..."
pip_install install torch torchvision torchaudio --index-url "$TORCH_URL"

echo ""
echo "[3/6] Installing Unsloth from Git..."
pip_install install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

echo ""
echo "[4/6] Installing accelerate & bitsandbytes (trl/peft stay on Unsloth's versions)..."
pip_install install accelerate bitsandbytes
echo "  Aligning trl with unsloth-zoo (>=0.18.2, <=0.24.0, !=0.19.0)..."
pip_install install "trl>=0.18.2,<=0.24.0,!=0.19.0"
if [[ "${SKIP_XFORMERS:-0}" == "1" ]]; then
  echo "  (Skipping xformers: SKIP_XFORMERS=1)"
else
  echo "  Trying optional xformers..."
  if ! pip_install install xformers; then
    echo "WARN: xformers not installed; training may still work without it." >&2
  fi
fi

echo ""
echo "[5/6] Installing Hugging Face datasets stack..."
pip_install install datasets transformers huggingface-hub tokenizers

echo ""
echo "[6/6] Ensuring NumPy and Pillow wheels match this Python..."
pip_install install --upgrade --force-reinstall "numpy>=2.1.3" pillow

echo ""
if [[ "${CREATE_VENV:-0}" == "1" ]]; then
  echo "Done. Activate:  source $VENV_PATH/bin/activate"
  echo "Then run:  python train_qwen_unsloth.py"
else
  echo "Done. Run: $PY train_qwen_unsloth.py"
fi
