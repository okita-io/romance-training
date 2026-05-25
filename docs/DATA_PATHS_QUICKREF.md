# Quick Reference: Central Corpus Paths

Use these variables in romance-training scripts and configs. Set **`ROMANCE_CORPUS_ROOT`** to override the default (`data/corpus/` under the repo root).

## Python

Run from romance-training repo root (or add repo root to `PYTHONPATH`):

```python
from tools.data_paths import (
    DEFAULT_TRAIN,
    DEFAULT_VAL,
    FINAL_COMBINED_TRAIN,
    FINAL_COMBINED_VAL,
    FICTION1B_ROOT,
    HEAT_SUBSETS_ROOT,
    CORPUS_ROOT,
)
```

## YAML configs

```yaml
data:
  path: /path/to/corpus/training/processed/youtube_combined_v3
  train_split: train
  valid_split: validation
```

## Shell

```bash
export ROMANCE_CORPUS_ROOT="${ROMANCE_CORPUS_ROOT:-$PWD/data/corpus}"
TRAIN_DATA="$ROMANCE_CORPUS_ROOT/training/processed/youtube_combined_v3/train.jsonl"
```

See also [`CORPUS_ORGANIZATION.md`](CORPUS_ORGANIZATION.md) and [`../tools/data_preparation/README.md`](../tools/data_preparation/README.md).
