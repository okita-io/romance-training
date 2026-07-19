# Style Classifier Training

A pipeline to annotate prose with **Leech & Short style metrics** and fine-tune a prose style judge / classifier (active target: **Gemma 4 26B-A4B QAT** on DGX Spark).

The trained model can:
- **Classify** any passage — outputs a structured style profile (register, POV, figurative density, sentence rhythm, etc.)
- **Judge** specific dimensions — "Analyze the verbosity of this passage", "What register is this written in?"
- **Rewrite** to a target style — "Rewrite this in a more formal register" (Phase 3B — requires paired training data, built separately)

## Machine setup (Spark trains; 3090 infers)

**All LoRA / Phase 4+ training runs on DGX Spark.** The RTX 3090 is for quantized inference (LM Studio), optional Phase 2 labeling, and local testing — not for fine-tuning.

| Machine | Host | Role | Phases |
|---------|------|------|--------|
| **DGX Spark** (`spark-4f07`, `10.0.1.4`) | `okita@spark-4f07` | **Training** — ~128 GB unified Blackwell + CUDA; Phase 3–4 LoRA + GGUF export | 3–4 (and later writer SFT) on Spark |
| **RTX 3090** (Windows, 24 GB) | dev box + LM Studio | **Inference** — run quantized GGUFs; optional Phase 2 bulk labeling | 1–2 locally; load finished Q4/Q5 in LM Studio |

**Current status (Jul 2026):**
- **Spark:** Active Phase 4 — **Gemma 4 26B-A4B QAT** via `train/train_config.gemma4_spark.toml`. Mistral-Nemo 12B LoRA paused at **checkpoint-1750** (`train/mistral_style_lora/`).
- **3090:** LM Studio for classification helpers and (after export) quantized Gemma evaluator GGUFs. Do **not** run Phase 4 training here.

**Data flow:** Styled `*_styled_seg_*.jsonl` (often labeled on the 3090) → `scp` to Spark (`train/romance_corpus/`) → merge + Phase 3 on Spark → **train on Spark**. See [`train/romance_corpus/README.md`](train/romance_corpus/README.md).

**North-star:** long-form fiction that holds a chosen voice, style, and tone across a full novel. Roadmap: [`docs/PHASE5_STYLE_STEERING.md`](docs/PHASE5_STYLE_STEERING.md) (Phases 5–6: evaluator → steering → bulk Gemma classify → novel merge).

**MoE style editor (target):** train a multi-grain editor (sentence / span / act experts) on top of the judge stack — current system, gaps, and completion tracks: [`docs/MOE_STYLE_EDITOR.md`](docs/MOE_STYLE_EDITOR.md).

**Training on Spark only:**

```bash
cd ~/git_repos/romance-training/train
hf auth login   # Gemma license accepted on Hugging Face
./download_gemma4_spark.sh      # one-time ~50 GB prefetch
./run_phase4_docker.sh --config train_config.gemma4_spark.toml
```

**Inference on 3090:** After GGUF export, copy `gemma4_style_q4/` (or q5) to the Windows box and load in LM Studio (temp 1.0, top_p 0.95, top_k 64, thinking off for JSON). Optional Phase 2: `run_pipeline.py --pass both` against a small instruct model in LM Studio.

## How it works

```
source/Style-in-Fiction.pdf
  └─[Phase 1A]─▶ source/extracted/pages/*.md + Style-in-Fiction.md   (vision LLM + mermaid diagrams)
                  └─[Phase 1B]─▶ source/extracted/style_knowledge.jsonl   (RAG chunks)
                                  └─[Phase 1C]─▶ source/style_rubric.json   (Leech & Short taxonomy)
                                                  └─[Phase 2]─▶ gutenberg_styled.jsonl   (sentence-aware chunks + style_profile)
                                                                  └─[Phase 3]─▶ style_training/train.jsonl
                                                                                  └─[Phase 4 on Spark]─▶ gemma4_style_lora/ + GGUF
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
├── run_phase4_docker.sh          # Spark: Unsloth dgxspark Docker wrapper
├── download_gemma4_spark.sh      # Prefetch Gemma 4 QAT HF weights on Spark
├── train_config.toml             # Mistral-Nemo 12B (paused at checkpoint-1750)
├── train_config.gemma4_spark.toml  # Active Spark config → Gemma 4 26B-A4B QAT
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

## Fresh clone — data prep (3090 or any workstation)

After `git pull`, HF datasets are **not** in the repo — download and convert them, then run Phase 2 labeling. Phase 1 rubric/knowledge **is** already committed (`source/style_rubric.json`, `source/extracted/style_knowledge.jsonl`).

**Full step-by-step:** [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md) — HF auth, corpora, resumable Phase 2, then **hand off to Spark for Phase 4**.

```bash
hf auth login
python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/romance-books
python tools/data_preparation/convert_romance_books_korshuk.py --chunk
python tools/style_classification/run_pipeline.py \
  --input source-data/processed/romance_books_korshuk/chunks.jsonl \
  --output train/romance_corpus/korshuk_styled.jsonl
```

Push styled JSONL to Spark before training (see `train/romance_corpus/README.md`).

## Setup

Python **3.12** recommended.

- **Training:** DGX Spark (`spark-4f07`) — Unsloth `dgxspark` Docker via `train/run_phase4_docker.sh` (~128 GB unified memory).
- **Inference / Phase 2:** RTX 3090 (24 GB) is enough for quantized GGUFs in LM Studio and optional classification; it is **not** the training host.

```bash
python -m venv .venv && source .venv/bin/activate

pip install -r requirements-train.txt
# Unsloth for local tooling only — Phase 4 training uses the Spark Docker image
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# spaCy model for computable metrics
python -m spacy download en_core_web_sm
```

Full dependency map → `REQUIREMENTS.md`.

Active Phase 4 config on Spark (do not rely on the 3090-era Mistral example for new runs):

```bash
# On spark-4f07:
cd ~/git_repos/romance-training/train
./run_phase4_docker.sh --config train_config.gemma4_spark.toml
```

Legacy Mistral template (paused): `cp train/train_config.example.toml train/train_config.toml`
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

# Parallel LLM requests when LM Studio allows multiple slots (e.g. 4)
python tools/style_classification/run_pipeline.py --workers 4

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

## Phase 4 — Fine-tune (**DGX Spark only**)

**Host:** `spark-4f07` (~128 GB unified Blackwell + CUDA). Do **not** fine-tune on the RTX 3090.

**Active target:** Gemma 4 26B-A4B QAT with 16-bit LoRA — train from HF weights, export GGUF for LM Studio on the 3090.

```bash
cd train
./run_phase4_docker.sh --config train_config.gemma4_spark.toml
```

Config: `train/train_config.gemma4_spark.toml` → local QAT weights under `~/.cache/huggingface/gemma-4-26B-A4B-it-qat-q4_0-unquantized`. Outputs LoRA adapter + GGUF quantizations under `gemma4_style_*`. Deploy the Q4 GGUF in LM Studio on the 3090 (match settings from the [QAT model card](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF): temp 1.0, top_p 0.95, top_k 64, thinking off for JSON tasks).

Resume or export-only:

```bash
./run_phase4_docker.sh --config train_config.gemma4_spark.toml --resume
./run_phase4_docker.sh --config train_config.gemma4_spark.toml --export-only
```

**Legacy (paused on Spark):** Mistral-Nemo 12B QLoRA — checkpoint at step 1750 in `mistral_style_lora/`. Resume with `train_config.toml` **on Spark**, not the 3090:

```bash
./run_phase4_docker.sh --config train_config.toml --resume
```

Override model without editing config:

```bash
ROMANCE_BASE_MODEL=unsloth/gemma-4-26B-A4B-it-qat-q4_0-unquantized \
  ./run_phase4_docker.sh --config train_config.gemma4_spark.toml
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
| `AlekseyKorshuk/erotic-books` | `erotic_books_korshuk` | ~646 BookRix novels | `convert_hf_parquet.py --dataset erotic_books_korshuk --chunk` |
| `AlekseyKorshuk/fantasy-books` | `fantasy_books_korshuk` | BookRix fantasy novels (gated) | `convert_hf_parquet.py --dataset fantasy_books_korshuk --chunk` |
| `molbal/horror-novel-chunks` | `horror_novel_chunks` | ~5.5k pre-chunked horror | `convert_hf_parquet.py --dataset horror_novel_chunks` |
| `ppirli/Gutenberg-Fiction` | `gutenberg_fiction` | ~23k Gutenberg books (~4.8 GB) | `convert_hf_parquet.py --dataset gutenberg_fiction --chunk` |

```bash
# Download (repeat per repo)
python tools/data_preparation/download_hf_dataset.py taozi555/literotica-stories
python tools/data_preparation/download_hf_dataset.py mrcedric98/fiction_books
python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/fiction-books
python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/erotic-books
python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/fantasy-books
python tools/data_preparation/download_hf_dataset.py molbal/horror-novel-chunks
python tools/data_preparation/download_hf_dataset.py ppirli/Gutenberg-Fiction

# Convert → source-data/processed/<slug>/chunks.jsonl
python tools/data_preparation/convert_hf_parquet.py --dataset horror_novel_chunks
python tools/data_preparation/convert_hf_parquet.py --dataset fiction_books_korshuk --chunk
python tools/data_preparation/convert_hf_parquet.py --dataset erotic_books_korshuk --chunk
python tools/data_preparation/convert_hf_parquet.py --dataset fantasy_books_korshuk --chunk
```

`AlekseyKorshuk/fiction-books`, `AlekseyKorshuk/fantasy-books`, and `AlekseyKorshuk/romance-books` are gated — accept HF terms before downloading.
