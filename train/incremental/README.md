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
      # Multigrain (additive; does not replace legacy ~500w trees above):
      # gutenberg_fiction/span/input/seg_000.jsonl
      # gutenberg_fiction/sentence/...
      # gutenberg_fiction/act/...
  batches/
    batch_001/
      manifest.json
      styled_combined.jsonl
      train.jsonl
      validation.jsonl
  logs/
    <output-stem>.events.jsonl # append-only Phase 2 run/interruption events
```

Legacy `segments/<corpus>/input/` stays the default ~500w classify path. Multigrain
trees live under `segments/<corpus>/<grain>/` and are registered separately in the ledger
(`grain` field; ids like `gutenberg_fiction/span/seg_000`).

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

`run_pipeline.py` records run starts, resume counts, periodic progress,
interruptions, compactions, and completions in `train/incremental/logs/` by
default. Pass `--run-log <path>` for a custom JSONL log or `--no-run-log` to
disable event logging.

## Silver Siren / abliterated Mistral

Use the same `train_qwen_unsloth.py` path; set `model.base` in `train_config.toml`
to your Silver Siren 12B HF id. Incremental batches are model-agnostic JSONL.

## Multigrain segments (sentence / span / act)

Build additive grains without touching existing ~500w segments.

**One-shot (recommended):** staging + ~50 MB pack for all mix corpora:

```bash
# Needs source-data/processed/<slug>/chunks.jsonl
python tools/data_preparation/prepare_bulk_segments.py --dry-run
python tools/data_preparation/prepare_bulk_segments.py --write          # default grain: span
python tools/data_preparation/prepare_bulk_segments.py --write --grain all --slug gutenberg_fiction
```

**Stratified council pilot** (interleave fiction / Literotica / Gutenberg / …):

```bash
python tools/data_preparation/build_council_mix.py --report-only
python tools/data_preparation/build_council_mix.py --write --per-corpus 40
# Then classify the emitted mix.jsonl with --llm-mode council --no-rechunk
```

**Manual two-step** (same as the orchestrator):

```bash
# 1. Materialize staging JSONL (~300w span, ~1000w act, plus sentences)
python tools/data_preparation/build_multigrain_chunks.py --write --slug gutenberg_fiction

# 2. Pack each grain into ~50 MB segments (same byte budget as legacy)
python tools/incremental/manage.py segment --corpus gutenberg_fiction --grain all
# or one grain: --grain span

# 3. Classify without re-chunking to 500w
python tools/incremental/manage.py classify-next \
  --corpus gutenberg_fiction --grain span --pass both --workers 4
# Writes train/romance_corpus/gutenberg_fiction_span_styled_seg_NNN.jsonl
```

Staging output: `train/staging/multigrain/<slug>/{sentence,span,act}.jsonl`  
Council mixes: `train/staging/council_mix/<name>/mix.jsonl`  
(`corpora.json` → `multigrain_staging`). Default classify path without `--grain` remains legacy ~500w.
