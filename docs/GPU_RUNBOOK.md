# GPU machine runbook (RTX 3090 — inference & data prep)

Fresh-clone checklist for the **Windows RTX 3090** box: download HF corpora, run multi-day Phase 2 classification, and **run quantized finished models in LM Studio**.

**Training (Phase 4+) is not done here.** Fine-tuning runs on **DGX Spark (`spark-4f07`, ~128 GB unified Blackwell + CUDA)** — see README § Machine setup and `train/run_phase4_docker.sh` with `train_config.gemma4_spark.toml`.

## What ships in git vs what you download

| In git (pull only) | Downloaded locally (gitignored) |
|--------------------|----------------------------------|
| `source/style_rubric.json` (v2) | `source-data/hf/` — raw HF datasets |
| `source/extracted/style_knowledge.jsonl` | `source-data/processed/` — chunked JSONL |
| `source/Style-in-Fiction.parsed.md` | `train/romance_corpus/*_styled.jsonl` — Phase 2 output |
| `source-data/manifests/*.json` | `train/style_training/` — Phase 3 output (often built on Spark) |
| All `tools/` scripts | Exported GGUFs copied from Spark for LM Studio (`gemma4_style_q4/`, etc.) |

**Phase 1 is already done** in the repo (parsed manuscript → rubric + knowledge base). You do not need to re-run vision PDF transcription unless you are changing the rubric.

---

## 1. Clone and Python environment

```bash
git clone https://github.com/okita-io/romance-training.git
cd romance-training

python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements-train.txt
# Unsloth optional for local tooling — Phase 4 training uses Spark Docker, not this venv
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m spacy download en_core_web_sm
```

Verify CUDA is visible if you use local PyTorch tooling:

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

Phase 2 with full semantic labels needs an OpenAI-compatible server. On the 3090, host this in **LM Studio** (quantized instruct model). Keep VRAM free for inference — **do not** start Phase 4 fine-tuning on this GPU; training belongs on `spark-4f07`.

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
cat train/romance_corpus/*_styled_seg_*.jsonl \
    > train/style_training/combined_styled.jsonl

python tools/training_formats/generate_instruction_pairs.py \
  --input train/style_training/combined_styled.jsonl \
  --output-dir train/style_training
```

Output: `train/style_training/train.jsonl` + `validation.jsonl`

---

## 8. Hand off to DGX Spark for Phase 4 (training)

**Do not fine-tune on the RTX 3090.** Sync styled / Phase 3 JSONL to Spark, then train there.

```bash
# Example: push styled segments to spark-4f07
scp train/romance_corpus/*_styled_seg_*.jsonl \
  okita@spark-4f07:~/git_repos/romance-training/train/romance_corpus/
```

On **spark-4f07**:

```bash
cd ~/git_repos/romance-training/train
# merge + Phase 3 if not already done
./run_phase4_docker.sh --config train_config.gemma4_spark.toml
```

Active config: `train/train_config.gemma4_spark.toml` (Gemma 4 26B-A4B QAT, ~128 GB unified memory).

After export, copy GGUFs back to the 3090 for LM Studio:

- `gemma4_style_q4/` — recommended for day-to-day eval
- `gemma4_style_q5/` — higher fidelity
- `gemma4_style_f16/` — large; optional

Legacy Mistral-Nemo (`train_config.toml`, checkpoint-1750) is paused; if resumed, resume **on Spark** only.

See also: [`PHASE5_STYLE_STEERING.md`](PHASE5_STYLE_STEERING.md).

---

## Quick reference — full pipeline order

```
git pull
  → pip install + spacy (+ optional unsloth for local tools)
  → hf auth login (+ accept gated dataset terms)
  → download_hf_dataset.py (×3 recommended)
  → convert_* / split_romance_parquet.py --chunk
  → run_pipeline.py (per corpus, resumable)   # 3090 / LM Studio OK
  → scp styled JSONL → spark-4f07
  → on Spark: cat → combined_styled.jsonl → generate_instruction_pairs.py
  → on Spark: ./run_phase4_docker.sh --config train_config.gemma4_spark.toml
  → copy gemma4_style_q4/*.gguf → 3090 LM Studio
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
| Exported Gemma Q4 GGUF (from Spark) | ~14–17 GB |
| Exported Gemma Q5 GGUF (optional) | ~19 GB |

Plan **~40 GB free** on the 3090 for corpora + one quantized GGUF. Training weights live on Spark (~50 GB+ for Gemma QAT base + checkpoints).

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
