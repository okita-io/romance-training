# Python dependencies by concern

Root **`.txt`** files are optional **pinned-style** installs.

| Concern | What to install | Notes |
|--------|-----------------|--------|
| **Train** (local Unsloth / torch stack) | `-r requirements-train.txt` or `train/install_training_deps.sh` | Heavy GPU stack. Install PyTorch with the CUDA index that matches your GPU first if needed. |
| **Dev / integration tests** | `pip install -r requirements-dev.txt` | Tests import `romance_factory.*`; install the sibling factory repo too (below). |
| **Generate (dependency)** | `pip install -e "../romance-factory[dev,v2]"` | Required for `train/tests/` — those tests exercise the LanceDB generate pipeline. |

## Typical setup

From this repo root, with `romance-factory` checked out as a sibling directory:

```bash
python -m venv .venv
source .venv/bin/activate

# Factory package + generate stack (for integration tests)
pip install -e "../romance-factory[dev,v2]"

# Training stack (GPU machine)
pip install -r requirements-train.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or use the helper script (creates .venv, installs CUDA PyTorch + Unsloth):
CREATE_VENV=1 ./train/install_training_deps.sh
```

## Layout

| Path | Purpose |
|------|---------|
| `train/train_qwen_unsloth.py` | Qwen LoRA fine-tuning + GGUF export (Unsloth) |
| `train/romance_corpus/` | JSONL training/validation splits and source texts |
| `train/tests/` | Integration tests against `romance_factory` |
| `train/install_training_deps.sh` | GPU training environment bootstrap |
| `tools/data_preparation/` | Corpus normalization (PG, Fiction-1B, combine) |

Corpus path reference: `docs/CORPUS_ORGANIZATION.md`, `docs/DATA_PATHS_QUICKREF.md`.
