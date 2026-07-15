## Project overview

**Style Classifier Training** — annotate prose with Leech & Short style metrics and fine-tune a prose style judge / classifier. **Active training target:** Gemma 4 26B-A4B QAT on DGX Spark.

See `README.md` for the full pipeline, layout, and backlog. Phase 5 plan: `docs/PHASE5_STYLE_STEERING.md`.

**Machine split:**
- **DGX Spark (`spark-4f07`, ~128 GB unified Blackwell + CUDA)** — **all Phase 4+ LoRA training** and GGUF export.
- **RTX 3090 (24 GB) + LM Studio** — **inference only** (quantized finished GGUFs), optional Phase 2 labeling. Do **not** fine-tune here.

**Data prep / Phase 2 on the 3090:** `docs/GPU_RUNBOOK.md`.

**Training on Spark:** `train/train_config.gemma4_spark.toml` + `./run_phase4_docker.sh`.

## Development environment

- Python **3.12** recommended (3.10+ minimum).
- **DGX Spark (`spark-4f07`)**: Phase 4 fine-tuning (Gemma 4 26B-A4B QAT; Mistral-Nemo legacy paused at checkpoint-1750).
- **RTX 3090** (24 GB): quantized GGUF inference in LM Studio; optional Phase 2 classification.
- **LM Studio** (default on 3090) or Ollama for LLM-based metrics and local eval of exported GGUFs.
- **Vision LLM** (Qwen-VL / Qwen3.6) for Phase 1A PDF transcription.

```bash
pip install -r requirements-train.txt
# Local tooling only — Phase 4 training uses Unsloth dgxspark Docker on Spark
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m spacy download en_core_web_sm

# PDF for Phase 1 (bundled copy in style-guide/)
mkdir -p source && cp style-guide/Style-in-Fiction.pdf source/

# Active training is on Spark with train_config.gemma4_spark.toml (not the 3090 example)
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

# Phase 3 — instruction pairs (prefer on Spark before training)
python tools/training_formats/generate_instruction_pairs.py

# Phase 4 — fine-tune on DGX Spark only (not the 3090)
cd train && ./run_phase4_docker.sh --config train_config.gemma4_spark.toml
```

Computable-only Phase 2 (fast, no LLM): `python tools/style_classification/run_pipeline.py --no-llm`

Re-chunk or re-classify from scratch: add `--no-resume` to Phase 2.

## Active training config

**Primary:** `train/train_config.gemma4_spark.toml` — Gemma 4 26B-A4B QAT on **spark-4f07** via `./run_phase4_docker.sh`.

**Legacy (paused):** `train/train_config.toml` — Mistral-Nemo 12B QLoRA at checkpoint-1750; resume on Spark only.