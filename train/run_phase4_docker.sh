#!/usr/bin/env bash
# Phase 4 on DGX Spark via Unsloth's GB10 image (Triton + CUDA 13 preconfigured).
set -euo pipefail

REPO_TRAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${UNSLOTH_DGX_IMAGE:-unsloth/unsloth:dgxspark-latest}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$HF_CACHE"

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
