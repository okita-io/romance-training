# Style Classifier Training

A pipeline to annotate prose with **Leech & Short style metrics** and fine-tune **Mistral-Nemo 12B** as a prose style judge, classifier, and rewriter.

The trained model can:
- **Classify** any passage — outputs a structured style profile (register, POV, figurative density, sentence rhythm, etc.)
- **Judge** specific dimensions — "Analyze the verbosity of this passage", "What register is this written in?"
- **Rewrite** to a target style — "Rewrite this in a more formal register" (Phase 3B — requires paired training data, built separately)

## How it works

```
source/Style-in-Fiction.pdf
  └─[Phase 1]─▶ source/style_rubric.json          (Leech & Short taxonomy)
                  └─[Phase 2]─▶ gutenberg_styled.jsonl   (corpus + style_profile per chunk)
                                  └─[Phase 3]─▶ style_training/train.jsonl   (instruction pairs)
                                                  └─[Phase 4]─▶ mistral_style_lora/   (fine-tuned model)
```

## Layout

```
source/
├── Style-in-Fiction.pdf          # Leech & Short reference (rubric source)
└── style_rubric.json             # Generated — review before Phase 2

tools/
├── llm_client.py                 # Shared OpenAI-compatible LLM client (LM Studio / Ollama)
├── style_extraction/
│   ├── extract_rubric.py         # Phase 1: PDF → style_rubric.json
│   └── pdf_vision_harness.py     # Phase 1A: PDF pages → markdown via vision LLM
├── style_classification/
│   ├── metrics_computable.py     # spaCy + textstat (fast, no LLM)
│   ├── metrics_llm.py            # LM Studio semantic metrics (single-pass)
│   ├── classify_passage.py       # Combines both into a style_profile dict
│   └── run_pipeline.py           # Phase 2: bulk JSONL enrichment + auto-chunking
└── training_formats/
    └── generate_instruction_pairs.py   # Phase 3: multi-task JSONL

train/
├── train_qwen_unsloth.py         # LoRA + GGUF export via Unsloth (model set by config)
├── train_config.toml             # Active config → Mistral-Nemo 12B
├── train_config.example.toml     # Template — copy and adjust
├── romance_corpus/
│   ├── gutenberg_romance.jsonl   # Raw Gutenberg prose (full books — auto-chunked at runtime)
│   └── gutenberg_styled.jsonl    # Generated — 500-word chunks with style_profile
└── style_training/               # Generated — ready for fine-tuning
    ├── train.jsonl
    └── validation.jsonl
```

## Setup

Python **3.12** recommended. CUDA GPU required for training (tested on RTX 3090, 24 GB).

```bash
python -m venv .venv && source .venv/bin/activate

pip install -r requirements-train.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# spaCy model for computable metrics
python -m spacy download en_core_web_sm
```

## LLM backend

The pipeline uses an **OpenAI-compatible local server** for all LLM work. Both LM Studio and Ollama are supported.

**LM Studio** (recommended — default):
1. Load any instruct model (Mistral, Llama 3, etc.)
2. Enable the local server via the toggle in the top bar
3. Default endpoint: `http://localhost:1234/v1`

**Ollama**:
```bash
ollama pull llama3.1:8b
# then pass: --base-url http://localhost:11434/v1
```

Override via env vars (no flags needed):
```bash
export LLM_BASE_URL=http://localhost:1234/v1
export LLM_MODEL=your-model-name          # as shown in LM Studio
export LLM_VISION_MODEL=your-vision-model # optional override for pdf_vision_harness.py
```

## Phase 1 — Extract rubric

Converts `source/Style-in-Fiction.pdf` to a structured JSON taxonomy. Requires LM Studio running. Run once; review the output before proceeding.

### Option A — Vision transcription (recommended)

Feed each PDF page to a **Qwen3.6 vision** model (or any OpenAI-compatible VLM in LM Studio) and save per-page markdown:

```bash
pip install pymupdf   # included in requirements-train.txt

# Load a Qwen3.6 / Qwen-VL vision model in LM Studio, enable local server
export LLM_VISION_MODEL=your-model-id-as-shown-in-lm-studio

# Smoke test (first 3 pages)
python tools/style_extraction/pdf_vision_harness.py --limit 3 --disable-thinking

# Full run — resumable; re-run picks up unfinished pages
python tools/style_extraction/pdf_vision_harness.py --disable-thinking --concat
```

Outputs:
- `source/extracted/pages/page_0001.md`, `page_0002.md`, … (one file per page)
- `source/extracted/Style-in-Fiction.md` (merged, when `--concat` is passed)

Then extract the rubric from the merged markdown:

```bash
python tools/style_extraction/extract_rubric.py --skip-pdf
```

### Option B — marker-pdf (batch OCR, no vision model)

`marker-pdf` has heavy deps — install separately:

```bash
pip install marker-pdf
python tools/style_extraction/extract_rubric.py
```

With Ollama instead of LM Studio for rubric LLM steps:

```bash
python tools/style_extraction/extract_rubric.py --base-url http://localhost:11434/v1 --model llama3.1:8b
```

The rubric script checks the LLM connection on startup and lists available models before processing.

Edit `source/style_rubric.json` to correct or add dimensions before running Phase 2.

## Phase 2 — Classify corpus

Adds a `style_profile` to every chunk. Full books are auto-chunked into 500-word passages at runtime — no pre-processing step needed.

```bash
# Computable metrics only — no LLM needed, fast (~14 rec/s)
python tools/style_classification/run_pipeline.py --no-llm

# Full run with LLM semantic metrics (LM Studio must be running)
python tools/style_classification/run_pipeline.py

# LLM on 20% sample — good balance of speed and coverage at scale
python tools/style_classification/run_pipeline.py --llm-sample-rate 0.2

# Parallel workers for computable-only mode
python tools/style_classification/run_pipeline.py --no-llm --workers 8

# Test on a few records first
python tools/style_classification/run_pipeline.py --no-llm --limit 50
```

Resumable — interrupted runs pick up where they left off.
Output: `train/romance_corpus/gutenberg_styled.jsonl`

## Phase 3 — Generate instruction pairs

Converts the enriched corpus into multi-task training data.

```bash
python tools/training_formats/generate_instruction_pairs.py
# → train/style_training/train.jsonl + validation.jsonl
```

Two task types per record:
- **classification** — "Classify this passage" → JSON style profile
- **judgment** — "Analyze the [dimension]" → natural language explanation

## Phase 4 — Fine-tune

Mistral-Nemo 12B with QLoRA rank 32 — fits on a single RTX 3090 in 4-bit.

```bash
python train/train_qwen_unsloth.py
```

Config is already pointed at `train/style_training/` and `mistralai/Mistral-Nemo-Instruct-2407`. Outputs a LoRA adapter + three GGUF quantizations (F16, Q5, Q4).

To override the model without editing config:

```bash
ROMANCE_BASE_MODEL=mistralai/Mistral-Nemo-Instruct-2407 python train/train_qwen_unsloth.py
```

## Cloud training

See `cloud_setup/` for RunPod, Vast.ai, and Modal helpers.

## Gaps and tasks

Known gaps in the repo and concrete tasks to close them. Use this as a backlog when onboarding or planning work.

### First-run prerequisites

| Gap | Task |
|-----|------|
| `source/Style-in-Fiction.pdf` is not in the repo | Document where to obtain the PDF (Leech & Short, *Style in Fiction*) and add a `source/README.md` with placement instructions, or host a licensed copy link if permitted |
| `train/romance_corpus/gutenberg_romance.jsonl` is gitignored and not shipped | Add a corpus builder script (e.g. `tools/data_preparation/build_gutenberg_jsonl.py`) that reads bundled `train/romance_corpus/gutenberg/*.txt` and writes the expected JSONL schema |
| `train/train_config.toml` is gitignored; only `train_config.example.toml` exists | Add a **Prerequisites** section above Phase 1 with `cp train/train_config.example.toml train/train_config.toml`, and fix Phase 4 wording ("copy example config first") |
| No single **Quick start** block tying all phases together | Add a copy-paste run order from empty clone → trained GGUF, including prerequisite copies |

### Pipeline and tooling

| Gap | Task |
|-----|------|
| `tools/data_preparation/prepare_project_gutenberg.py` expects external `data/corpus/…` JSONL, not the in-repo `.txt` files | Either wire it to the bundled Gutenberg texts or deprecate it in favor of the new JSONL builder; update `tools/data_preparation/paths.py` docs |
| `classify_passage.py` is in the layout but not documented in any phase | Add a Phase 2 smoke-test example: `echo "…" \| python tools/style_classification/classify_passage.py` |
| Phase 3B (rewrite pairs) is mentioned in the intro but has no implementation | Design rewrite pair schema, add `generate_rewrite_pairs.py` (frontier LLM or paired corpus), and document in a Phase 3B section |
| No example `style_profile` JSON in the docs | Add a short sample output (computable + LLM fields) so training targets are visible without running the pipeline |
| `extract_rubric.py` references `os.environ` without `import os` | Fix the missing import (runtime bug on Phase 1) |

### Training and inference

| Gap | Task |
|-----|------|
| `train_qwen_unsloth.py` docstring still describes Qwen / romance generation | Update module docstring and GGUF size hints for Mistral-Nemo 12B |
| README omits `--resume`, `--export-only`, and `TRAIN_CONFIG_PATH` | Document training restart and export-only flows in Phase 4 |
| No post-training inference section | Add "Using the model" with LM Studio load steps and example classify / judge / rewrite prompts |
| Windows setup exists (`install_training_deps.ps1`, GGUF/CMake notes) but README is Linux-centric | Add a short Windows subsection under Setup pointing at the `.ps1` helpers |

### Docs and repo hygiene

| Gap | Task |
|-----|------|
| `REQUIREMENTS.md` references missing `docs/CORPUS_ORGANIZATION.md` and `docs/DATA_PATHS_QUICKREF.md` | Create those docs or remove stale links from `REQUIREMENTS.md` |
| README does not link to `REQUIREMENTS.md` | Add a Setup note: "Full dependency map → `REQUIREMENTS.md`" |
| `pyproject.toml` description says "Romance Factory models" while README focuses on style classification | Align project metadata with the style-classifier scope (or note the sibling `romance-factory` relationship explicitly) |
| `train/tests/` (50+ integration tests) is undocumented | Add a Testing section: `pip install -e "../romance-factory[dev,v2]"` + `pytest train/tests` |

### Cloud and deployment

| Gap | Task |
|-----|------|
| `cloud_setup/runpod_setup.sh` and `vast_ai_setup.sh` reference `romance-factory`, old `trl<0.9.0`, and `data/romance_corpus` paths | Rewrite scripts for `romance-training` layout (`train/style_training/`, current `requirements-train.txt` pins) |
| `cloud_setup/modal_train.py` mounts `data/romance_corpus` and targets Qwen training | Update Modal app name, image deps, mount paths, and entrypoint to match the style pipeline |
| Cloud section is a one-liner with no run instructions | Expand with per-provider steps once scripts are updated (upload data, copy config, run `train_qwen_unsloth.py`) |

### Data and quality (future)

| Gap | Task |
|-----|------|
| Rubric quality depends on LLM extraction — no validation suite | Add a schema test + spot-check script for `style_rubric.json` dimension counts and required fields |
| LLM metrics on a sample rate leave some records computable-only | Document tradeoffs; optionally backfill LLM fields in a second pass |
| No eval harness for the fine-tuned style judge | Add a small held-out eval script (dimension accuracy, JSON parse rate, judge coherence) |
