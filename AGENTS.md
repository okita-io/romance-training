## Project overview

**Style Classifier Training** — a pipeline to annotate prose with Leech & Short style metrics and fine-tune Mistral-Nemo 12B as a prose style judge, classifier, and rewriter.

See `README.md` for the full pipeline and run order.

## Development environment

- Python **3.12** recommended (3.10+ minimum).
- CUDA GPU for training (RTX 3090, 24 GB tested).
- Ollama running locally for LLM-based metrics.

```bash
pip install -r requirements-train.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m spacy download en_core_web_sm
```

## Key directories

| Directory | Purpose |
|-----------|---------|
| `source/` | `Style-in-Fiction.pdf` + generated `style_rubric.json` |
| `tools/style_extraction/` | Phase 1: PDF → rubric JSON |
| `tools/style_classification/` | Phase 2: classify corpus with style metrics |
| `tools/training_formats/` | Phase 3: generate multi-task instruction pairs |
| `train/` | Training script, config, corpus JSONL, generated outputs |
| `cloud_setup/` | RunPod / Vast.ai / Modal helpers |

## Run order

```bash
python tools/style_extraction/extract_rubric.py
python tools/style_classification/run_pipeline.py
python tools/training_formats/generate_instruction_pairs.py
python train/train_qwen_unsloth.py
```

## Active training config

`train/train_config.toml` — Mistral-Nemo 12B, QLoRA rank 32, data at `train/style_training/`.
