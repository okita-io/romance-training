# MOVED — this repo is romance-training (formerly romance-factory/train/).

Training data files are NOT checked into git (too large).

## Corpus location

```
train/romance_corpus/train.jsonl
train/romance_corpus/validation.jsonl
```

## Verify data

```bash
ls -lh train/romance_corpus/*.jsonl
wc -l train/romance_corpus/train.jsonl
```

Should show ~7534 lines in train.jsonl and ~837 in validation.jsonl when populated.

See README.md and train/romance_corpus/STATUS.md for collection status.

