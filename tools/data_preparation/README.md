# Data Preparation Pipeline

Normalizes raw romance sources into **YouTube-compatible JSONL** for unified fine-tuning.

## Corpus layout

Set **`ROMANCE_CORPUS_ROOT`** to a central corpus tree (see `docs/CORPUS_ORGANIZATION.md`). Default when unset:

```
data/corpus/
├── sources/
│   ├── project_gutenberg/train.jsonl
│   └── external/fiction1b_enhanced/
└── training/processed/
    ├── youtube_combined_v3/
    ├── project_gutenberg_normalized/
    ├── fiction1b_normalized/
    └── final_combined/
```

**In-repo fallback:** if `sources/project_gutenberg/train.jsonl` is missing, `prepare_project_gutenberg.py` uses `train/romance_corpus/gutenberg_romance.jsonl`.

## Target sample format

```json
{
  "text": "chunk of ~500 words",
  "metadata": {
    "source": "project_gutenberg",
    "category": "classic",
    "heat_level": "moderate",
    "chunk_index": 0,
    "total_chunks": 1,
    "chunk_size": 500,
    "chunk_overlap": 50,
    "word_count": 500
  }
}
```

## Scripts

| Script | Purpose |
|--------|---------|
| `prepare_project_gutenberg.py` | Clean PG text, infer category/heat, re-chunk to 500 words |
| `prepare_fiction1b.py` | Normalize Fiction-1B metadata, filter by confidence |
| `combine_all_datasets.py` | Merge YouTube + PG + F1B, 90/10 split |
| `run_all_preparation.py` | Run PG + F1B prep sequentially |
| `enrich_metadata.py` | LLM metadata enrichment (requires `pip install -e ../romance-factory`) |

## Usage

From **romance-training** repo root:

```bash
# All normalization steps (PG + F1B)
python tools/data_preparation/run_all_preparation.py

# Or individually (run from repo root; scripts use their own directory as cwd)
python tools/data_preparation/prepare_project_gutenberg.py
python tools/data_preparation/prepare_fiction1b.py
python tools/data_preparation/combine_all_datasets.py
```

## Paths in Python

```python
from tools.data_paths import FINAL_COMBINED_TRAIN, FINAL_COMBINED_VAL, CORPUS_ROOT
```

## Training integration

Point Unsloth / AutoTrain at the combined output:

```yaml
data:
  path: /path/to/corpus/training/processed/final_combined
  train_split: train
  valid_split: validation
```

Or symlink/copy into `train/romance_corpus/` for `train_qwen_unsloth.py`.
