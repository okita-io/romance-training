# Romance Training

Self-contained **training data, fine-tuning scripts, and integration tests** for models used with [Romance Factory](../romance-factory/) (`romance-factory` generates novels; this repo trains the models those runs can use).

Formerly `romance-factory/train/`. Training concerns now live here so the factory repo can focus on **collect** (corpus gathering) and **generate** (LanceDB RAG novel pipeline).

## Layout

```
train/
├── train_qwen_unsloth.py      # Qwen LoRA + GGUF export (Unsloth)
├── train_config.example.toml  # Copy → train_config.toml
├── romance_corpus/            # JSONL splits + Gutenberg sources
├── tests/                     # Integration tests (import romance_factory)
└── ...

tools/
└── data_preparation/          # Normalize PG / Fiction-1B / combine corpora
```

## Setup

### 1. Python environment

Python **3.10+** (3.12 recommended for ML wheels).

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Romance Factory (required for tests)

Integration tests import `romance_factory`. With both repos as siblings:

```bash
pip install -e "../romance-factory[dev,v2]"
export PYTHONPATH="../romance-factory/src${PYTHONPATH:+:$PYTHONPATH}"
```

### 3. Training stack (GPU machine)

```bash
pip install -r requirements-train.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

Or:

```bash
CREATE_VENV=1 ./train/install_training_deps.sh
```

See [`REQUIREMENTS.md`](REQUIREMENTS.md) for the full dependency map.

## Fine-tune Qwen (Unsloth)

```bash
cd train
cp train_config.example.toml train_config.toml   # edit model.base, paths, VRAM
python train_qwen_unsloth.py --config train_config.toml
```

Corpus defaults: `train/romance_corpus/train.jsonl` and `validation.jsonl`.

### Prepare / combine corpora

```bash
python tools/data_preparation/run_all_preparation.py
python tools/data_preparation/combine_all_datasets.py
```

Set `ROMANCE_CORPUS_ROOT` for a central corpus tree (default: `data/corpus/`). See `tools/data_preparation/README.md` and `docs/CORPUS_ORGANIZATION.md`.

## Tests

From this repo root (with factory installed as above):

```bash
python -m pytest train/tests/ -v
```

**Opt-in live LLM tests** (Ollama / LM Studio):

```bash
export ROMANCE_FACTORY_LOCAL_LLM_LIVE=1
python -m pytest train/tests/test_local_llm_live.py -v -m local_llm_live
```

## Relationship to Romance Factory

| Concern | Repo |
|---------|------|
| Novel generation (LanceDB RAG) | `romance-factory` |
| Corpus collection / browser harness | `romance-factory/pipeline/` |
| Training data + fine-tuning + train integration tests | **`romance-training`** (this repo) |

Factory docs: [`../romance-factory/docs/REPO_LAYOUT.md`](../romance-factory/docs/REPO_LAYOUT.md).

## Cloud training

Use `prepare_for_cloud.sh` (packages corpus + scripts) or the scripts under `cloud_setup/` for RunPod / Vast.ai helpers.
