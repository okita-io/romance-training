#!/usr/bin/env bash
# Prefetch Gemma 4 26B-A4B QAT HF weights for Spark training.
# GGUF deploy pack (LM Studio): unsloth/gemma-4-26B-A4B-it-qat-GGUF
set -euo pipefail

MODEL_ID="unsloth/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"
DEST="${CACHE_ROOT}/gemma-4-26B-A4B-it-qat-q4_0-unquantized"

echo "Downloading ${MODEL_ID}"
echo "  -> ${DEST}"
echo "Requires: hf auth login + Gemma license accepted on Hugging Face"
echo

mkdir -p "$DEST"
hf download "$MODEL_ID" --local-dir "$DEST"

echo
echo "Done. Start training:"
echo "  cd $(dirname "$0")"
echo "  ./run_phase4_docker.sh --config train_config.gemma4_spark.toml"
