#!/usr/bin/env bash
# Phase 4 on DGX Spark via Unsloth's GB10 image (Triton + CUDA 13 preconfigured).
#
# Gemma 4 26B-A4B QAT (recommended on Spark):
#   ./run_phase4_docker.sh --config train_config.gemma4_spark.toml
#
# Mistral-Nemo 12B (default train_config.toml):
#   ./run_phase4_docker.sh
set -euo pipefail

REPO_TRAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${UNSLOTH_DGX_IMAGE:-unsloth/unsloth:dgxspark-latest}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
# Persistent overlay for Gemma 4 only (transformers>=5.5 + newer Unsloth).
# Everything is installed with --no-deps so NVIDIA torch in the image is not shadowed.
GEMMA4_SITE="${ROMANCE_GEMMA4_SITE:-$HOME/.cache/romance-training/gemma4-py-packages}"

mkdir -p "$HF_CACHE" "$GEMMA4_SITE"

NEED_GEMMA4_STACK=0
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--config" && "$arg" == *gemma4* ]]; then
    NEED_GEMMA4_STACK=1
  fi
  if [[ "$arg" == --config=*gemma4* ]]; then
    NEED_GEMMA4_STACK=1
  fi
  prev="$arg"
done

if [[ "$NEED_GEMMA4_STACK" -eq 1 ]]; then
  exec docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --entrypoint bash \
    -v "$REPO_TRAIN:/workspace/train" \
    -v "$HF_CACHE:$HF_CACHE" \
    -v "$GEMMA4_SITE:/opt/gemma4-site" \
    -e HF_HOME="$HF_CACHE" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e PYTHONPATH="/opt/gemma4-site/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}" \
    -w /workspace/train \
    "$IMAGE" \
    -lc 'set -euo pipefail
      SITE=/opt/gemma4-site/lib/python3.12/site-packages
      mkdir -p "$SITE"
      # Never keep a torch/triton overlay — it shadows the image CUDA build.
      rm -rf "$SITE"/torch "$SITE"/torch-* "$SITE"/torchgen \
             "$SITE"/triton "$SITE"/triton-* \
             "$SITE"/torchvision "$SITE"/torchvision-* "$SITE"/torchao "$SITE"/torchao-* \
             "$SITE"/nvidia 2>/dev/null || true

      need_install=0
      if ! python -c "import transformers; assert transformers.__version__.startswith(\"5.5\")" >/dev/null 2>&1; then
        need_install=1
      fi
      if ! python -c "import unsloth; v=getattr(unsloth,\"__version__\",\"0\"); assert tuple(int(x) for x in v.split(\".\")[:2]) >= (2026, 7)" >/dev/null 2>&1; then
        need_install=1
      fi
      if [[ "$need_install" -eq 1 ]]; then
        echo "Installing Gemma 4 overlay (--no-deps; keeps image CUDA torch)..."
        pip install --target "$SITE" --upgrade --no-deps --quiet \
          "transformers==5.5.0" \
          "huggingface_hub>=1.5.0,<2.0" \
          "tokenizers>=0.22.0,<=0.23.0" \
          "unsloth" \
          "unsloth_zoo"
        # Strip any transitive wheels that slipped in
        rm -rf "$SITE"/torch "$SITE"/torch-* "$SITE"/torchgen \
               "$SITE"/triton "$SITE"/triton-* \
               "$SITE"/torchvision "$SITE"/torchvision-* "$SITE"/torchao "$SITE"/torchao-* \
               "$SITE"/nvidia 2>/dev/null || true
      fi
      if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        echo "ERROR: torch.cuda unavailable after Gemma 4 overlay." >&2
        python -c "import torch; print(torch.__file__); print(torch.__version__)" || true
        exit 1
      fi
      exec python train_qwen_unsloth.py "$@"
    ' -- "$@"
fi

exec docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  -v "$REPO_TRAIN:/workspace/train" \
  -v "$HF_CACHE:$HF_CACHE" \
  -e HF_HOME="$HF_CACHE" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -w /workspace/train \
  "$IMAGE" \
  train_qwen_unsloth.py "$@"
