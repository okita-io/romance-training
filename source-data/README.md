# Source data

Raw fiction corpora downloaded from Hugging Face and other sources, plus scripts to merge them into one JSONL for the style pipeline.

## Layout

```
source-data/
├── manifests/          # Per-dataset field mapping (tracked in git)
│   ├── lovecraftcorpus.json
│   ├── Robot.E.Howard.v2.json
│   └── romance_books_32K.json
├── hf/                 # Raw HF downloads (gitignored)
│   ├── TristanBehrens__lovecraftcorpus/
│   ├── leftyfeep__Robot.E.Howard.v2/
│   └── diltdicker__romance_books_32K/
└── unified/            # Merged corpus for Phase 2 (gitignored)
    └── fiction_corpus.jsonl
```

Unified records match the style pipeline schema:

```json
{
  "text": "passage prose …",
  "metadata": {
    "source": "hf:lovecraftcorpus",
    "source_dataset": "TristanBehrens/lovecraftcorpus",
    "genres": ["horror", "weird_fiction"],
    "author": "H.P. Lovecraft",
    "source_file": "unnamable.txt",
    "record_index": 42,
    "word_count": 243
  }
}
```

## Download a dataset

```bash
python tools/data_preparation/download_hf_dataset.py TristanBehrens/lovecraftcorpus
python tools/data_preparation/download_hf_dataset.py leftyfeep/Robot.E.Howard.v2
python tools/data_preparation/download_hf_dataset.py diltdicker/romance_books_32K
```

Files land in `source-data/hf/<Author>__<name>/`. Add or edit a manifest in `source-data/manifests/` when field names differ from the defaults.

## Convert all HF sources to unified JSONL

```bash
python tools/data_preparation/convert_hf_sources.py
# → source-data/unified/fiction_corpus.jsonl
```

Then run Phase 2 on the merged corpus:

```bash
python tools/style_classification/run_pipeline.py \
  --input source-data/unified/fiction_corpus.jsonl \
  --output train/romance_corpus/hf_styled.jsonl \
  --no-llm
```
