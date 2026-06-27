# GPU machine runbook (RTX 3090)

Fresh-clone checklist for the PC that downloads HF corpora, runs multi-day Phase 2 classification, and fine-tunes Mistral-Nemo 12B.

## What ships in git vs what you download

| In git (pull only) | Downloaded locally (gitignored) |
|--------------------|----------------------------------|
| `source/style_rubric.json` (v2) | `source-data/hf/` — raw HF datasets |
| `source/extracted/style_knowledge.jsonl` | `source-data/processed/` — chunked JSONL |
| `source/Style-in-Fiction.parsed.md` | `train/romance_corpus/*_styled.jsonl` — Phase 2 output |
| `source-data/manifests/*.json` | `train/style_training/` — Phase 3 output |
| All `tools/` scripts | `mistral_style_lora/` — Phase 4 adapter |

**Phase 1 is already done** in the repo (parsed manuscript → rubric + knowledge base). You do not need to re-run vision PDF transcription unless you are changing the rubric.

---

## 1. Clone and Python environment

```bash
git clone https://github.com/okita-io/romance-training.git
cd romance-training

python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements-train.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m spacy download en_core_web_sm

cp train/train_config.example.toml train/train_config.toml
```

Verify CUDA is visible to PyTorch (after Unsloth install):

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 2. Hugging Face CLI and authentication

Install the HF CLI if needed:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Log in (required for **gated** datasets):

```bash
hf auth login
```

For `AlekseyKorshuk/romance-books` you must also:

1. Open https://huggingface.co/datasets/AlekseyKorshuk/romance-books
2. Accept the dataset terms while logged in
3. Ensure your token has **read access to gated repos**

---

## 3. Download datasets from Hugging Face

All downloads land in `source-data/hf/<Author>__<dataset>/`.

### Recommended training mix

| Priority | HF repo | Content | Chunks (approx.) |
|----------|---------|---------|----------------|
| **Primary** | `AlekseyKorshuk/romance-books` | Full BookRix romance novels | ~167k |
| **Secondary** | `Dwaraka/Training_Dataset_of_Project_Gutebberg_Gothic_Fiction` | 8 Gothic Gutenberg novels | ~2k |
| **Supplementary** | `diltdicker/romance_books_32K` | Romance blurbs + genre tags | ~24k |

```bash
# Primary — gated; accept terms on HF first
python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/romance-books

# Gothic literary prose
python tools/data_preparation/download_hf_dataset.py \
  Dwaraka/Training_Dataset_of_Project_Gutebberg_Gothic_Fiction

# Blurbs + metadata (open)
python tools/data_preparation/download_hf_dataset.py diltdicker/romance_books_32K
```

Optional additional sources (manifests in repo):

```bash
python tools/data_preparation/download_hf_dataset.py TristanBehrens/lovecraftcorpus
python tools/data_preparation/download_hf_dataset.py leftyfeep/Robot.E.Howard.v2
python tools/data_preparation/download_hf_dataset.py taozi555/literotica-stories
python tools/data_preparation/download_hf_dataset.py mrcedric98/fiction_books
python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/fiction-books
python tools/data_preparation/download_hf_dataset.py molbal/horror-novel-chunks
python tools/data_preparation/download_hf_dataset.py ppirli/Gutenberg-Fiction
```

Convert with `convert_hf_parquet.py --dataset <slug>` (add `--chunk` for full-book corpora). See `source-data/README.md`.

---

## 4. Convert HF downloads → chunked JSONL

Each converter writes English-only chunks by default. Skipped non-English rows go to `skipped_non_english.jsonl`.

```bash
# BookRix full novels (~30 min)
python tools/data_preparation/convert_romance_books_korshuk.py --chunk

# Gothic Gutenberg (~seconds)
python tools/data_preparation/convert_gutenberg_gothic.py --chunk

# 32K blurbs (~5 min)
python tools/data_preparation/split_romance_parquet.py --chunk --by-author
```

Outputs:

```
source-data/processed/romance_books_korshuk/chunks.jsonl
source-data/processed/gutenberg_gothic_fiction/chunks.jsonl
source-data/processed/romance_books_32k/chunks.jsonl
```

Smoke-test converters:

```bash
PYTHONPATH=. python -m pytest train/tests/test_language_filter.py \
  train/tests/test_convert_gutenberg_gothic.py \
  train/tests/test_split_romance_parquet.py -q --noconftest
```

---

## 5. LLM backend for Phase 2 (classification)

Phase 2 with full semantic labels needs an OpenAI-compatible server. The 3090 can host this locally while training runs later on the same GPU — typically run classification first, then fine-tune.

**LM Studio** (default, port 1234):

```bash
# .env in repo root (optional)
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=<model-id-as-shown-in-lm-studio>
```

Load an instruct model (e.g. Llama 3.1 8B, Mistral 7B) and enable the local server.

**Ollama** alternative:

```bash
ollama pull llama3.1:8b
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3.1:8b
```

Test connectivity:

```bash
echo "She wakened early." | python tools/style_classification/classify_passage.py
```

---

## 6. Phase 2 — Classify corpus (resumable, multi-day)

Adds `metadata.style_profile` to each chunk. **Runs are resumable** — stop and restart the same command; already-classified records are skipped.

```bash
mkdir -p train/romance_corpus

# Korshuk — largest; expect several days with full LLM
python tools/style_classification/run_pipeline.py \
  --input source-data/processed/romance_books_korshuk/chunks.jsonl \
  --output train/romance_corpus/korshuk_styled.jsonl

# Gothic
python tools/style_classification/run_pipeline.py \
  --input source-data/processed/gutenberg_gothic_fiction/chunks.jsonl \
  --output train/romance_corpus/gothic_styled.jsonl

# 32K blurbs
python tools/style_classification/run_pipeline.py \
  --input source-data/processed/romance_books_32k/chunks.jsonl \
  --output train/romance_corpus/romance_32k_styled.jsonl
```

### Mode options

| Flag | When to use |
|------|-------------|
| *(default)* | Full LLM labels — register, POV, tone, textual principles + computable metrics |
| `--no-llm --workers 8` | Fast baseline (~14 rec/s) — spaCy/textstat only |
| `--workers 4 --pass fast` | Pass 1 — small model, lexical/discourse/textual fields |
| `--workers 2 --pass deep` | Pass 2 — large model, tone + viewpoint (merge into same output) |
| `--workers 2` | Parallel LLM requests — match LM Studio concurrent slot count |
| `--llm-sample-rate 0.2` | LLM on 20% of chunks; rest computable-only |
| `--limit 50` | Smoke test |
| `--no-resume` | Rebuild output from scratch |

Full LLM on ~193k chunks at ~2–5 s/chunk is **days of runtime** — safe to interrupt; rerun the same command to continue.

---

## 7. Phase 3 — Instruction pairs

Merge styled corpora, then generate train/val JSONL:

```bash
cat train/romance_corpus/korshuk_styled.jsonl \
    train/romance_corpus/gothic_styled.jsonl \
    train/romance_corpus/romance_32k_styled.jsonl \
    > train/romance_corpus/combined_styled.jsonl

python tools/training_formats/generate_instruction_pairs.py \
  --input train/romance_corpus/combined_styled.jsonl \
  --output-dir train/style_training
```

Output: `train/style_training/train.jsonl` + `validation.jsonl`

---

## 8. Phase 4 — Fine-tune on RTX 3090

Mistral-Nemo 12B, QLoRA rank 32, 4-bit — fits 24 GB VRAM.

```bash
python train/train_qwen_unsloth.py
```

Config: `train/train_config.toml` (copy from `train_config.example.toml` if missing).

Outputs:

- LoRA adapter → `mistral_style_lora/`
- GGUF exports → `mistral_style_f16/`, `mistral_style_q5/`, `mistral_style_q4/`

Override model without editing config:

```bash
ROMANCE_BASE_MODEL=mistralai/Mistral-Nemo-Instruct-2407 python train/train_qwen_unsloth.py
```

---

## Quick reference — full pipeline order

```
git pull
  → pip install + spacy + unsloth
  → hf auth login (+ accept gated dataset terms)
  → download_hf_dataset.py (×3 recommended)
  → convert_* / split_romance_parquet.py --chunk
  → run_pipeline.py (per corpus, resumable)
  → cat *_styled.jsonl → combined_styled.jsonl
  → generate_instruction_pairs.py
  → train_qwen_unsloth.py
```

Phase 1 rubric/knowledge: **already in repo** — skip unless regenerating:

```bash
python tools/style_extraction/distill_style_system.py --force
```

---

## Disk space (rough)

| Item | Size |
|------|------|
| HF: Korshuk parquet | ~260 MB |
| HF: Gothic corpus | ~5 MB |
| HF: 32K parquet | ~50 MB |
| Processed chunks (all three) | ~500 MB–1 GB |
| Styled JSONL (full LLM, ~193k records) | ~2–4 GB |
| Mistral-Nemo 12B download (first train) | ~24 GB |
| LoRA + GGUF exports | ~10–20 GB |

Plan **~50 GB free** for a comfortable first run.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `hf CLI not found` | `curl -LsSf https://hf.co/cli/install.sh \| bash` |
| Gated dataset 403 | Accept terms on HF; `hf auth login` with gated-repo token |
| `spaCy model not found` | `python -m spacy download en_core_web_sm` |
| LLM connection refused | Start LM Studio server or Ollama; check `LLM_BASE_URL` |
| Phase 2 seems stuck | Normal for LLM mode — check progress lines every 100 records |
| Want to restart Phase 2 clean | Add `--no-resume` to delete output and start over |
| CUDA OOM during training | Lower `batch_size` in `train_config.toml` (try 1) |

More detail: `README.md`, `source-data/README.md`, `AGENTS.md`.
