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
│   └── extract_rubric.py         # Phase 1: PDF → style_rubric.json
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

`marker-pdf` (Phase 1 PDF conversion) has heavy deps — install separately:

```bash
pip install marker-pdf
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
```

## Phase 1 — Extract rubric

Converts `source/Style-in-Fiction.pdf` to a structured JSON taxonomy. Requires LM Studio running and `marker-pdf` installed. Run once; review the output before proceeding.

```bash
python tools/style_extraction/extract_rubric.py
# → source/style_rubric.json

# With Ollama instead:
python tools/style_extraction/extract_rubric.py --base-url http://localhost:11434/v1 --model llama3.1:8b
```

The script checks the LLM connection on startup and lists available models before converting the PDF.

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
