## Cursor Cloud specific instructions

### Project overview

**Romance Training** is a sibling repo to Romance Factory. It holds training data, Unsloth fine-tuning scripts, and integration tests that exercise `romance_factory` (the generate pipeline lives in the factory repo).

**Factory repo:** [`../romance-factory/`](../romance-factory/) — collect + generate only.

### Development environment

- Python **3.10+** (3.12 recommended for GPU training wheels).
- Virtual environment at `.venv`.
- **Always install Romance Factory first** for tests:
  ```bash
  pip install -e "../romance-factory[dev,v2]"
  export PYTHONPATH="../romance-factory/src${PYTHONPATH:+:$PYTHONPATH}"
  ```
- Training deps: `pip install -r requirements-train.txt` or `CREATE_VENV=1 ./train/install_training_deps.sh` on a GPU machine.
- Pin files: [`REQUIREMENTS.md`](REQUIREMENTS.md).

### Key directories

| Directory | Purpose |
|-----------|---------|
| `train/` | Fine-tuning scripts, `romance_corpus/`, config TOML |
| `train/tests/` | Pytest suite (imports `romance_factory.*`) |
| `tools/data_preparation/` | Corpus normalization (PG, Fiction-1B, combine) |
| `cloud_setup/` | RunPod / Vast.ai helpers |

### Tests

```bash
python -m pytest train/tests/ -v
```

Live/network markers match factory (`openrouter_live`, `local_llm_live`) — see `pyproject.toml`.

### Fine-tuning

```bash
cd train
python train_qwen_unsloth.py --config train_config.toml
```

Copy `train_config.example.toml` first; adjust `model.base` for your VRAM.

### Data preparation

```bash
python tools/data_preparation/run_all_preparation.py
python tools/data_preparation/combine_all_datasets.py
```

Set `ROMANCE_CORPUS_ROOT` for an external corpus mount (default: `data/corpus/`).
