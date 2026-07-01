# Styled training corpus (Phase 2 output)

Only **classified** JSONL files belong here — each row must have `metadata.style_profile`
with a complete LLM pass (`--pass both` or fast + deep merged).

## run
```powershell
$env:LLM_MODEL="llama-3.2-4x3b-moe-ultra-instruct-10b"
>> python tools/style_classification/run_pipeline.py  --pass both --workers 4 --input .\train\incremental\segments\literotica_stories\input\seg_000.jsonl --output .\train\romance_corpus\literotica_stories_deep_seg_000.jsonl
```

## Allowed files

| Pattern | Example | Purpose |
|---------|---------|---------|
| `{corpus}_styled.jsonl` | `horror_styled.jsonl` | Full corpus, pass 2 complete |
| `{corpus}_styled_seg_NNN.jsonl` | `fiction_books_styled_seg_000.jsonl` | Per-segment classified output |

## Do **not** put here

| Item | Use instead |
|------|-------------|
| Raw / unclassified chunks | `source-data/processed/<slug>/chunks.jsonl` |
| Pipeline-expanded input (~500w) | `train/staging/pipeline_chunks/<slug>_pipeline_chunks.jsonl` |
| Incremental input segments | `train/incremental/segments/<slug>/input/` |
| Incremental styled segments (default) | `train/incremental/segments/<slug>/styled/` |
| Backups (`.bak`, `.pre_strip*`) | `train/staging/backups/` |
| Scratch / temp files | Delete or use `train/staging/` |

Phase 3 reads styled JSONL from here (or from `build-batch` combined output under
`train/incremental/batches/`).

## Validate

```bash
python tools/data_preparation/validate_romance_corpus.py
python tools/data_preparation/validate_romance_corpus.py --strict
```

## Dedupe after interrupted runs

```bash
python tools/data_preparation/dedup_corpus_jsonl.py --input train/romance_corpus/horror_styled.jsonl --in-place
```
