# Incremental training (segments, ledger, batches)

Split large corpora into ~50 MB JSONL segments, classify incrementally, and
train on mixed batches without waiting for full-corpus classification.

## Layout

```
train/incremental/
  corpora.json          # corpus registry (tracked)
  ledger.json           # segment + batch state (local, gitignored)
  segments/
    horror_novel_chunks/
      input/seg_000.jsonl   # unclassified chunks
      styled/seg_000.jsonl  # Phase 2 output
    literotica_stories/
    fiction_books/
    gutenberg_fiction/
    erotic_books_korshuk/
    fantasy_books_korshuk/
  batches/
    batch_001/
      manifest.json
      styled_combined.jsonl
      train.jsonl
      validation.jsonl
  logs/
    <output-stem>.events.jsonl # append-only Phase 2 run/interruption events
```

## Ledger states

| Stage | `classification_status` | `training_status` |
|-------|-------------------------|-------------------|
| Input segment created | `pending` | `unavailable` |
| Phase 2 running | `in_progress` | `unavailable` |
| Classified, not yet trained | `classified` | `available` |
| In a built batch | `classified` | `allocated` |
| Used in a training run | `classified` | `trained` |

## Typical workflow

```bash
# 1. Dashboard
python tools/incremental/manage.py status

# 2. Split source corpora (~50 MB input segments)
python tools/incremental/manage.py segment --all

# 3. Horror is already classified — import into ledger
python tools/incremental/manage.py import-styled --corpus horror_novel_chunks

# 4. Classify one ~50 MB segment at a time (repeat until status shows 0 pending)
python tools/incremental/manage.py classify-next \
  --corpus literotica_stories --pass both --workers 4
# Picks lowest pending seg_000, seg_001, …; processes entire segment file, then stops.
# Re-run the same command for the next segment. Interrupted runs resume the same segment.

# Manual run_pipeline on a segment? Check progress anytime:
python tools/incremental/manage.py classify-progress \
  --corpus literotica_stories --segment 0 --pass both

# Optional: reflect manual progress in ledger.json
python tools/incremental/manage.py classify-progress \
  --corpus literotica_stories --segment 0 --sync-ledger

# 5. Build a mixed batch: up to 50 MB styled per corpus
python tools/incremental/manage.py build-batch --max-mb 50

# 6. Train on DGX Spark (active target: Gemma 4 26B-A4B QAT).
#    Point paths.data_dir at the batch dir in your config, e.g.:
#    paths.data_dir = "train/incremental/batches/batch_001"
cd train && ./run_phase4_docker.sh --config train_config.gemma4_spark.toml
#    The Docker wrapper runs train_qwen_unsloth.py; batches are model-agnostic JSONL.
#    Legacy path: python train/train_qwen_unsloth.py with train_config.toml (Mistral-Nemo, paused).

# 7. Record the run so those segments are not reused
python tools/incremental/manage.py mark-trained --batch batch_001 --run run_001 \
  --model-base "your-model-id" --output-dir "gemma4_style_lora"
```

Re-run `build-batch` after more segments are classified to start the next
training iteration. The ledger prevents reusing segments already marked `trained`.

`run_pipeline.py` records run starts, resume counts, periodic progress,
interruptions, compactions, and completions in `train/incremental/logs/` by
default. Pass `--run-log <path>` for a custom JSONL log or `--no-run-log` to
disable event logging.

## Alternate base models (Silver Siren / abliterated Mistral, etc.)

Incremental batches are **model-agnostic JSONL**, so any base works through the same
`run_phase4_docker.sh` → `train_qwen_unsloth.py` path — set `model.base` in your
config to the target HF id (e.g. a Silver Siren 12B). The active target is
**Gemma 4 26B-A4B QAT** (`train_config.gemma4_spark.toml`); Mistral-Nemo 12B
(`train_config.toml`) is the paused legacy config.
