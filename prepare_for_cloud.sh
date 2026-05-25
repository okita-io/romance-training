#!/bin/bash
# Package everything needed for cloud training (romance-training repo)

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Preparing Romance Training for Cloud ==="
echo ""

DEPLOY_DIR="$REPO_ROOT/romance-training-deploy"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

echo "[1/5] Copying training data..."
cp -r "$REPO_ROOT/train/romance_corpus" "$DEPLOY_DIR/"

echo "[2/5] Copying training scripts..."
cp "$REPO_ROOT/train/train_qwen_unsloth.py" "$DEPLOY_DIR/"
cp "$REPO_ROOT/requirements-train.txt" "$DEPLOY_DIR/"
cp "$REPO_ROOT/cloud_setup/"*.sh "$DEPLOY_DIR/" 2>/dev/null || true

echo "[3/5] Copying config template..."
cp "$REPO_ROOT/train/train_config.example.toml" "$DEPLOY_DIR/train_config.toml"

echo "[4/5] Creating setup script..."
cat > "$DEPLOY_DIR/setup_and_train.sh" << 'INNER_EOF'
#!/bin/bash
# Quick setup and training script for cloud GPU

set -e

echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install datasets transformers huggingface-hub tokenizers

echo ""
echo "Starting training..."
python3 train_qwen_unsloth.py --config train_config.toml

echo ""
echo "Training complete! Check outputs in train_config [paths].output_dir / export dirs."
INNER_EOF
chmod +x "$DEPLOY_DIR/setup_and_train.sh"

echo "[5/5] Creating tarball..."
tar -czf "$REPO_ROOT/romance-training-deploy.tar.gz" -C "$REPO_ROOT" romance-training-deploy

echo ""
echo "✓ Package created: romance-training-deploy.tar.gz"
echo ""
echo "To deploy to cloud:"
echo "  1. Upload romance-training-deploy.tar.gz to your GPU instance"
echo "  2. Extract: tar -xzf romance-training-deploy.tar.gz"
echo "  3. cd romance-training-deploy"
echo "  4. bash setup_and_train.sh"
echo ""
echo "Or use cloud_setup/ scripts: runpod_setup.sh, vast_ai_setup.sh"
echo ""
echo "Data size: $(du -sh "$DEPLOY_DIR" | cut -f1)"
echo "Package size: $(du -sh "$REPO_ROOT/romance-training-deploy.tar.gz" | cut -f1)"
