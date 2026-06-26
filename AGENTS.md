## Project overview

**Style Classifier Training** — a pipeline to annotate prose with Leech & Short style metrics and fine-tune Mistral-Nemo 12B as a prose style judge, classifier, and rewriter.

See `README.md` for the full pipeline, layout, and backlog.

**Fresh clone on the 3090:** `docs/GPU_RUNBOOK.md` — HF downloads, corpus conversion, resumable Phase 2, training.

## Development environment

- Python **3.12** recommended (3.10+ minimum).
- CUDA GPU for training (RTX 3090, 24 GB tested).
- **LM Studio** (default) or Ollama for LLM-based metrics and rubric extraction.
- **Vision LLM** (Qwen-VL / Qwen3.6) for Phase 1A PDF transcription.

```bash
pip install -r requirements-train.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m spacy download en_core_web_sm

# PDF for Phase 1 (bundled copy in style-guide/)
mkdir -p source && cp style-guide/Style-in-Fiction.pdf source/

# Training config (before Phase 4)
cp train/train_config.example.toml train/train_config.toml
```

## Key directories

| Directory | Purpose |
|-----------|---------|
| `source/` | `Style-in-Fiction.pdf`, generated `style_rubric.json` |
| `source/extracted/` | Vision markdown (`pages/`, `Style-in-Fiction.md`), `style_knowledge.jsonl` |
| `style-guide/` | Bundled PDF reference copy |
| `tools/style_extraction/` | Phase 1A–1C: vision PDF → RAG → rubric JSON |
| `tools/style_classification/` | Phase 2: sentence-aware chunking + style metrics |
| `tools/training_formats/` | Phase 3: generate multi-task instruction pairs |
| `train/` | Training script, config, corpus JSONL, generated outputs |
| `train/tests/test_style_fidelity.py` | Unit tests for chunking and knowledge retrieval |
| `cloud_setup/` | RunPod / Vast.ai / Modal helpers |

## Run order

Quality-first pipeline — complete Phase 1 before bulk classification with LLM metrics.

```bash
# Phase 1A — vision PDF → markdown (mermaid flowcharts)
export LLM_VISION_MODEL=your-vision-model
python tools/style_extraction/pdf_vision_harness.py --disable-thinking --concat

# Phase 1B — markdown → RAG knowledge base
python tools/style_extraction/build_style_knowledge.py

# Phase 1C — rubric already in repo; regenerate only if needed:
python tools/style_extraction/distill_style_system.py --force
# Legacy LLM path: extract_rubric.py --skip-pdf --use-knowledge

# Phase 2 — classify corpus (sentence-boundary chunks; RAG context when LLM enabled)
python tools/style_classification/run_pipeline.py

# Phase 3 — instruction pairs
python tools/training_formats/generate_instruction_pairs.py

# Phase 4 — fine-tune
python train/train_qwen_unsloth.py
```

Computable-only Phase 2 (fast, no LLM): `python tools/style_classification/run_pipeline.py --no-llm`

Re-chunk or re-classify from scratch: add `--no-resume` to Phase 2.

## Active training config

`train/train_config.toml` — Mistral-Nemo 12B, QLoRA rank 32, data at `train/style_training/`.
