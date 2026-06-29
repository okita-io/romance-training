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
  batches/
    batch_001/
      manifest.json
      styled_combined.jsonl
      train.jsonl
      validation.jsonl
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

# 4. Classify one segment at a time (repeat per corpus)
python tools/incremental/manage.py classify-next \
  --corpus literotica_stories --pass fast --workers 4

# 5. Build a mixed batch: up to 50 MB styled per corpus
python tools/incremental/manage.py build-batch --max-mb 50

# 6. Train (Mistral-Nemo 12B, Silver Siren 12B, etc.)
#    Set paths.data_dir in train/train_config.toml to the batch dir, e.g.:
#    paths.data_dir = "train/incremental/batches/batch_001"
python train/train_qwen_unsloth.py

# 7. Record the run so those segments are not reused
python tools/incremental/manage.py mark-trained --batch batch_001 --run run_001 \
  --model-base "your-model-id" --output-dir "mistral_style_lora"
```

Re-run `build-batch` after more segments are classified to start the next
training iteration. The ledger prevents reusing segments already marked `trained`.

## Silver Siren / abliterated Mistral

Use the same `train_qwen_unsloth.py` path; set `model.base` in `train_config.toml`
to your Silver Siren 12B HF id. Incremental batches are model-agnostic JSONL.
