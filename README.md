# Style Classifier Training

A pipeline to annotate prose with **Leech & Short style metrics** and fine-tune **Mistral-Nemo 12B** as a prose style judge, classifier, and rewriter.

The trained model can:
- **Classify** any passage — outputs a structured style profile (register, POV, figurative density, sentence rhythm, etc.)
- **Judge** specific dimensions — "Analyze the verbosity of this passage", "What register is this written in?"
- **Rewrite** to a target style — "Rewrite this in a more formal register" (Phase 3B — requires paired training data, built separately)

## How it works

```
source/Style-in-Fiction.pdf
  └─[Phase 1A]─▶ source/extracted/pages/*.md + Style-in-Fiction.md   (vision LLM + mermaid diagrams)
                  └─[Phase 1B]─▶ source/extracted/style_knowledge.jsonl   (RAG chunks)
                                  └─[Phase 1C]─▶ source/style_rubric.json   (Leech & Short taxonomy)
                                                  └─[Phase 2]─▶ gutenberg_styled.jsonl   (sentence-aware chunks + style_profile)
                                                                  └─[Phase 3]─▶ style_training/train.jsonl
                                                                                  └─[Phase 4]─▶ mistral_style_lora/
```

## Layout

```
source/                           # Pipeline inputs and generated reference data
├── Style-in-Fiction.pdf          # Place PDF here (copy from style-guide/ if needed)
├── style_rubric.json             # Generated Phase 1C — review before Phase 2
└── extracted/                    # Generated Phase 1A–1B
    ├── pages/page_0001.md        # Per-page vision transcription
    ├── Style-in-Fiction.md       # Merged markdown (mermaid flowcharts)
    └── style_knowledge.jsonl     # RAG chunks for rubric + classification

style-guide/
└── Style-in-Fiction.pdf          # Bundled reference copy (symlink or copy → source/)

tools/
├── llm_client.py                 # Shared OpenAI-compatible LLM client (LM Studio / Ollama)
├── style_extraction/
│   ├── extract_rubric.py         # Phase 1C: knowledge/markdown → style_rubric.json
│   ├── build_style_knowledge.py  # Phase 1B: markdown → RAG JSONL
│   └── pdf_vision_harness.py     # Phase 1A: PDF pages → markdown (mermaid diagrams)
├── style_classification/
│   ├── metrics_computable.py     # spaCy + textstat (fast, no LLM)
│   ├── metrics_llm.py            # Rubric + RAG-grounded semantic metrics
│   ├── style_knowledge.py        # Retrieve Leech & Short chunks for classification
│   ├── chunk_text.py             # Sentence-boundary chunking
│   ├── classify_passage.py       # Combines both into a style_profile dict
│   └── run_pipeline.py           # Phase 2: bulk JSONL enrichment + auto-chunking
└── training_formats/
    └── generate_instruction_pairs.py   # Phase 3: multi-task JSONL

train/
├── train_qwen_unsloth.py         # LoRA + GGUF export via Unsloth (model set by config)
├── train_config.toml             # Active config → Mistral-Nemo 12B
├── train_config.example.toml     # Template — copy and adjust
├── tests/
│   └── test_style_fidelity.py    # Chunking + knowledge retrieval unit tests
├── romance_corpus/
│   ├── gutenberg_romance.jsonl   # Raw Gutenberg prose (full books — auto-chunked at runtime)
│   └── gutenberg_styled.jsonl    # Generated — sentence-boundary ~500-word chunks + style_profile
└── style_training/               # Generated — ready for fine-tuning
    ├── train.jsonl
    └── validation.jsonl
```

## Fresh clone on a GPU machine (RTX 3090)

After `git pull`, HF datasets are **not** in the repo — download and convert them locally, then run Phases 2–4. Phase 1 rubric/knowledge **is** already committed (`source/style_rubric.json`, `source/extracted/style_knowledge.jsonl`).

**Full step-by-step:** [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md) — HF auth, recommended corpora (Korshuk + Gothic + 32K blurbs), resumable multi-day classification, merge, train.

```bash
hf auth login
python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/romance-books
python tools/data_preparation/convert_romance_books_korshuk.py --chunk
python tools/style_classification/run_pipeline.py \
  --input source-data/processed/romance_books_korshuk/chunks.jsonl \
  --output train/romance_corpus/korshuk_styled.jsonl
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

Full dependency map → `REQUIREMENTS.md`.

Copy training config before Phase 4:

```bash
cp train/train_config.example.toml train/train_config.toml
```

Place the Leech & Short PDF where Phase 1 expects it (a copy ships in `style-guide/`):

```bash
mkdir -p source
cp style-guide/Style-in-Fiction.pdf source/
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

Override via env vars or a repo-root `.env` file (loaded automatically; does not override exported vars):

```bash
# .env example
OPENROUTER_API_KEY=sk-or-...
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=your-model-name          # as shown in LM Studio
LLM_VISION_MODEL=your-vision-model # optional override for pdf_vision_harness.py
```

Or export directly:

```bash
export LLM_BASE_URL=http://10.0.1.7:1234/v1
export LLM_VISION_MODEL=your-vision-model-id
```

## Phase 1 — Extract rubric (quality-first)

Phase 1 prioritizes **analytic fidelity** over speed. The PDF is transcribed by a vision LLM (with flowcharts converted to mermaid), chunked into a RAG knowledge base, then distilled into a structured rubric. That knowledge base is also retrieved during Phase 2 LLM classification.

### 1A — Vision transcription (recommended)

Feed each PDF page to a **Qwen3.6 vision** model (or any OpenAI-compatible VLM in LM Studio). Simple flowcharts are converted to fenced `mermaid` blocks.

```bash
pip install pymupdf   # included in requirements-train.txt

export LLM_VISION_MODEL=your-model-id-as-shown-in-lm-studio

python tools/style_extraction/pdf_vision_harness.py --limit 3 --disable-thinking
python tools/style_extraction/pdf_vision_harness.py --disable-thinking --concat
```

**LM Studio on another machine** (e.g. GPU server at `10.0.1.7`):

```bash
# Auto-connects to http://10.0.1.7:1234/v1 and picks a VL/vision model
python tools/style_extraction/pdf_vision_harness.py --lm-studio-remote --limit 3 --pdf style-guide/Style-in-Fiction.pdf

# Or set explicitly in .env:
# LLM_BASE_URL=http://10.0.1.7:1234/v1
# LLM_VISION_MODEL=qwen3-vl-...   # must support image input in LM Studio
```

The loaded model must accept **image** inputs. Plain instruct models (e.g. `gemma-4-12b-it-...` without VL) will fail with “does not support image inputs”. Load a vision/VL checkpoint in LM Studio on that host.

**Quality checks:** each page is validated before save (rejects reasoning dumps, repetition spam, and one-word outputs). On resume, bad pages are re-transcribed automatically.

```bash
# Audit existing conversions without calling the LLM
python tools/style_extraction/pdf_vision_harness.py --audit-only
```

Outputs:
- `source/extracted/pages/page_0001.md`, …
- `source/extracted/Style-in-Fiction.md` (merged)

**OpenRouter (cloud vision):** Nemotron tends to emit reasoning instead of clean transcription — prefer LM Studio with a VL model when possible. OpenRouter remains available with cooldown:

```bash
# OPENROUTER_API_KEY in repo-root .env is loaded automatically

# Smoke test — 3 pages, 10s cooldown (default with --openrouter)
python tools/style_extraction/pdf_vision_harness.py --openrouter --limit 3 --pdf style-guide/Style-in-Fiction.pdf

# Full run — resumable; re-run skips finished pages
python tools/style_extraction/pdf_vision_harness.py --openrouter --cooldown 10 --concat
```

Model: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`. Put `OPENROUTER_API_KEY` in repo-root `.env` (loaded automatically). On HTTP 429 the client retries with backoff.

### 1B — RAG knowledge base

```bash
python tools/style_extraction/build_style_knowledge.py
```

Output: `source/extracted/style_knowledge.jsonl` — section-aware chunks with category tags and mermaid metadata.

### 1C — Rubric extraction

```bash
python tools/style_extraction/extract_rubric.py --skip-pdf --use-knowledge
```

### Option B — marker-pdf (batch OCR, no vision model)

`marker-pdf` has heavy deps — install separately:

```bash
pip install marker-pdf
python tools/style_extraction/extract_rubric.py
python tools/style_extraction/build_style_knowledge.py
python tools/style_extraction/extract_rubric.py --skip-pdf --use-knowledge
```

With Ollama instead of LM Studio for rubric LLM steps:

```bash
python tools/style_extraction/extract_rubric.py --base-url http://localhost:11434/v1 --model llama3.1:8b
```

The rubric script checks the LLM connection on startup and lists available models before processing.

Edit `source/style_rubric.json` to correct or add dimensions before running Phase 2.

## Phase 2 — Classify corpus

Adds a `style_profile` to every chunk. Full books are auto-chunked into **~500-word sentence-boundary passages** (2-sentence overlap) at runtime — chunks never start or end mid-sentence.

When the LLM is enabled (`run_pipeline.py` without `--no-llm`), classification retrieves Leech & Short reference excerpts from `style_knowledge.jsonl` and rubric definitions from `style_rubric.json` for each passage. The `--no-llm` path computes spaCy/textstat metrics only and does not use the knowledge base.

Smoke-test a single passage:

```bash
echo "She wakened early, in the hour before dawn." | python tools/style_classification/classify_passage.py
```

Bulk classification:

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

Resumable — interrupted runs pick up where they left off. To regenerate after changing chunking or classification logic, pass `--no-resume` (deletes existing output and starts fresh).

Output: `train/romance_corpus/gutenberg_styled.jsonl`

Example `style_profile` fields (computable + LLM when enabled):

```json
{
  "lexical_density": 0.51,
  "sentence_length_mean": 18.4,
  "type_token_ratio": 0.54,
  "passive_rate": 0.09,
  "register": "neutral_narrative",
  "pov": "third_limited",
  "tone": "melancholic",
  "figurative_density": "low"
}
```

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

## Testing

Style-pipeline unit tests (no external LLM required):

```bash
PYTHONPATH=. python -m pytest train/tests/test_style_fidelity.py -q --noconftest
```

Other tests under `train/tests/` may require the sibling `romance-factory` package.

## Cloud training

See `cloud_setup/` for RunPod, Vast.ai, and Modal helpers.

## Gaps and tasks

Known gaps in the repo and concrete tasks to close them. Use this as a backlog when onboarding or planning work.

### First-run prerequisites

| Gap | Task |
|-----|------|
| PDF lives in `style-guide/` but pipeline reads `source/` | Documented above — `cp style-guide/Style-in-Fiction.pdf source/` |
| `train/romance_corpus/gutenberg_romance.jsonl` is gitignored and not shipped | Add a corpus builder script (e.g. `tools/data_preparation/build_gutenberg_jsonl.py`) that reads bundled `train/romance_corpus/gutenberg/*.txt` and writes the expected JSONL schema |
| No single **Quick start** block tying all phases together | See [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md) and run order in `AGENTS.md` |

### Pipeline and tooling

| Gap | Task |
|-----|------|
| `tools/data_preparation/prepare_project_gutenberg.py` expects external `data/corpus/…` JSONL, not the in-repo `.txt` files | Either wire it to the bundled Gutenberg texts or deprecate it in favor of the new JSONL builder; update `tools/data_preparation/paths.py` docs |
| Phase 3B (rewrite pairs) is mentioned in the intro but has no implementation | Design rewrite pair schema, add `generate_rewrite_pairs.py` (frontier LLM or paired corpus), and document in a Phase 3B section |

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
| README does not link to `REQUIREMENTS.md` | Done — see Setup section |
| `pyproject.toml` description says "Romance Factory models" while README focuses on style classification | Align project metadata with the style-classifier scope (or note the sibling `romance-factory` relationship explicitly) |
| `train/tests/` (50+ integration tests) is undocumented | Style pipeline tests documented in Testing section; romance-factory tests still need separate setup |

### Cloud and deployment

| Gap | Task |
|-----|------|
| `cloud_setup/runpod_setup.sh` and `vast_ai_setup.sh` reference `romance-factory`, old `trl<0.9.0`, and `data/romance_corpus` paths | Rewrite scripts for `romance-training` layout (`train/style_training/`, current `requirements-train.txt` pins) |
| `cloud_setup/modal_train.py` mounts `data/romance_corpus` and targets Qwen training | Update Modal app name, image deps, mount paths, and entrypoint to match the style pipeline |
| Cloud section is a one-liner with no run instructions | Expand with per-provider steps once scripts are updated (upload data, copy config, run `train_qwen_unsloth.py`) |

### Data and quality (future)

| Gap | Task |
|-----|------|
| Knowledge retrieval uses keyword overlap, not embeddings | Add optional embedding index over `style_knowledge.jsonl` for better passage→reference matching |
| Rubric quality depends on LLM extraction — no validation suite | Add a schema test + spot-check script for `style_rubric.json` dimension counts and required fields |
| LLM metrics on a sample rate leave some records computable-only | Document tradeoffs; optionally backfill LLM fields in a second pass |
| No eval harness for the fine-tuned style judge | Add a small held-out eval script (dimension accuracy, JSON parse rate, judge coherence) |
| `generate_instruction_pairs.py` uses hardcoded score thresholds | Read `low`/`mid`/`high` bands from `style_rubric.json` instead |

---

## Additional HF datasets

Beyond the three recommended corpora above, these datasets have manifests and converters in the repo:

| HF repo | Slug | Content | Convert |
|---------|------|---------|---------|
| `taozi555/literotica-stories` | `literotica_stories` | ~645k story texts (~10.8 GB) | `convert_hf_parquet.py --dataset literotica_stories` |
| `mrcedric98/fiction_books` | `fiction_books` | ~20k book chapters | `convert_hf_parquet.py --dataset fiction_books` |
| `AlekseyKorshuk/fiction-books` | `fiction_books_korshuk` | ~4.7k BookRix novels (gated) | `convert_hf_parquet.py --dataset fiction_books_korshuk --chunk` |
| `molbal/horror-novel-chunks` | `horror_novel_chunks` | ~5.5k pre-chunked horror | `convert_hf_parquet.py --dataset horror_novel_chunks` |
| `ppirli/Gutenberg-Fiction` | `gutenberg_fiction` | ~23k Gutenberg books (~4.8 GB) | `convert_hf_parquet.py --dataset gutenberg_fiction --chunk` |

```bash
# Download (repeat per repo)
python tools/data_preparation/download_hf_dataset.py taozi555/literotica-stories
python tools/data_preparation/download_hf_dataset.py mrcedric98/fiction_books
python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/fiction-books
python tools/data_preparation/download_hf_dataset.py molbal/horror-novel-chunks
python tools/data_preparation/download_hf_dataset.py ppirli/Gutenberg-Fiction

# Convert → source-data/processed/<slug>/chunks.jsonl
python tools/data_preparation/convert_hf_parquet.py --dataset horror_novel_chunks
python tools/data_preparation/convert_hf_parquet.py --dataset fiction_books_korshuk --chunk
```

`AlekseyKorshuk/fiction-books` is gated like `romance-books` — accept HF terms before downloading.
