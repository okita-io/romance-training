# Styled training corpus (Phase 2 output)

Only **classified** JSONL files belong here — each row must have `metadata.style_profile`
with a complete LLM pass (`--pass both` or fast + deep merged).

## run
```powershell
$env:LLM_MODEL="llama-3.2-4x3b-moe-ultra-instruct-10b"
python tools/style_classification/run_pipeline.py  --pass both --workers 4 --input .\train\incremental\segments\literotica_stories\input\seg_000.jsonl --output .\train\romance_corpus\literotica_stories_deep_seg_000.jsonl
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
| Merged combine files (`combined_styled.jsonl`) | `train/style_training/` (see Phase 3 below) |

Phase 3 reads styled JSONL from here (or from `build-batch` combined output under
`train/incremental/batches/`). All corpora in the mix must appear in
`train/incremental/corpora.json` → `training_mix_corpora` when using
`manage.py build-batch`; manual `cat` below includes every `*_styled_seg_*.jsonl`
in this directory regardless.

## Sync to DGX Spark (`spark-4f07` — training host)

These JSONL files are gitignored — copy them to Spark after Phase 2 (often run on
the Windows RTX 3090 with LM Studio). **Phase 4 LoRA training runs only on Spark**
(~128 GB unified Blackwell + CUDA); the 3090 loads exported quantized GGUFs later.

Requires SSH to `spark-4f07` / `10.0.1.4` (NVIDIA Sync alias `okita-pc` may map here)
and a hosts entry if mDNS does not resolve `spark-4f07.local`:

```
10.0.1.4 spark-4f07.local spark-4f07
```

Preferred filenames: `{corpus}_styled_seg_NNN.jsonl`. Legacy Phase 2 `--output`
paths may use `*_deep_seg_*.jsonl` — rename or let cleanup on the Spark normalize
them.

**Push** styled segments from Windows (PowerShell):

```powershell
scp "e:\git_repos\romance-training\train\romance_corpus\*_styled_seg_*.jsonl" okita-pc:~/git_repos/romance-training/train/romance_corpus/
# legacy names still work:
scp "e:\git_repos\romance-training\train\romance_corpus\*_deep_seg_*.jsonl" okita-pc:~/git_repos/romance-training/train/romance_corpus/
```

**Pull** cleaned corpus back from the Spark (after cleanup or Phase 2 there):

```powershell
scp okita-pc:~/git_repos/romance-training/train/romance_corpus/*_styled_seg_*.jsonl e:\git_repos\romance-training\train\romance_corpus\
```

**Verify** on the Spark:

```bash
ssh okita-pc "ls -lh ~/git_repos/romance-training/train/romance_corpus/*.jsonl"
ssh okita-pc "cd ~/git_repos/romance-training && python3 tools/data_preparation/validate_romance_corpus.py"
```

**Pull** Phase 3 output:

```powershell
scp -r okita-pc:~/git_repos/romance-training/train/style_training e:\git_repos\romance-training\train\
```

## Cleanup before Phase 3

On the Spark (or locally), strip junk chunks and incomplete rows, then rename to
`*_styled_seg_*.jsonl`:

```bash
python tools/data_preparation/clean_processed_corpora.py --styled --in-place
python tools/data_preparation/validate_romance_corpus.py --strict
```

Re-run Phase 3 after any corpus cleanup — instruction pairs are derived from the
styled JSONL and go stale when rows are removed.

## Phase 3 — instruction pairs

`validate_romance_corpus.py` rejects filenames containing `combined` in this
directory — build the merge file under `train/style_training/` instead.

Merge all styled segments, then generate train/val JSONL on the Spark (or locally):

```bash
cat train/romance_corpus/*_styled_seg_*.jsonl > train/style_training/combined_styled.jsonl

python tools/training_formats/generate_instruction_pairs.py \
  --input train/style_training/combined_styled.jsonl \
  --output-dir train/style_training
```

Ensure `train/incremental/corpora.json` lists every corpus slug in
`training_mix_corpora` (including `erotic_books_korshuk` and
`fantasy_books_korshuk`) if you use the incremental batch path:

```bash
python tools/incremental/manage.py build-batch --max-mb 50
```

Output: `train/style_training/train.jsonl` + `validation.jsonl`

## Validate

```bash
python tools/data_preparation/validate_romance_corpus.py
python tools/data_preparation/validate_romance_corpus.py --strict
```

## Dedupe after interrupted runs

```bash
python tools/data_preparation/dedup_corpus_jsonl.py --input train/romance_corpus/horror_styled.jsonl --in-place
```

## Run event logs

`run_pipeline.py` writes append-only progress/interruption events to
`train/incremental/logs/<output-stem>.events.jsonl` by default. These logs stay
outside `train/romance_corpus/` so this directory remains limited to styled
training JSONL.
