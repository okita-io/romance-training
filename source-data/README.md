# Source data

Raw fiction corpora downloaded from Hugging Face and other sources, plus scripts to merge them into one JSONL for the style pipeline.

**GPU machine setup (clone → HF download → classify → train):** see [`docs/GPU_RUNBOOK.md`](../docs/GPU_RUNBOOK.md).

## Layout

```
source-data/
├── manifests/          # Per-dataset field mapping (tracked in git)
│   ├── lovecraftcorpus.json
│   ├── Robot.E.Howard.v2.json
│   ├── romance_books_32K.json
│   ├── romance_books_korshuk.json
│   ├── gutenberg_gothic_fiction.json
│   ├── literotica_stories.json
│   ├── fiction_books.json
│   ├── fiction_books_korshuk.json
│   ├── horror_novel_chunks.json
│   └── gutenberg_fiction.json
├── processed/          # Per-dataset extractions (gitignored)
│   ├── romance_books_korshuk/
│   ├── romance_books_32k/
│   └── gutenberg_gothic_fiction/
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
python tools/data_preparation/download_hf_dataset.py Dwaraka/Training_Dataset_of_Project_Gutebberg_Gothic_Fiction
python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/romance-books
python tools/data_preparation/download_hf_dataset.py taozi555/literotica-stories
python tools/data_preparation/download_hf_dataset.py mrcedric98/fiction_books
python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/fiction-books
python tools/data_preparation/download_hf_dataset.py molbal/horror-novel-chunks
python tools/data_preparation/download_hf_dataset.py ppirli/Gutenberg-Fiction
```

Files land in `source-data/hf/<Author>__<name>/`. Add or edit a manifest in `source-data/manifests/` when field names differ from the defaults.

Convert parquet datasets to Phase 2 input:

```bash
python tools/data_preparation/convert_hf_parquet.py --dataset horror_novel_chunks
python tools/data_preparation/convert_hf_parquet.py --dataset fiction_books
python tools/data_preparation/convert_hf_parquet.py --dataset fiction_books_korshuk --chunk
python tools/data_preparation/convert_hf_parquet.py --dataset gutenberg_fiction --chunk
python tools/data_preparation/convert_hf_parquet.py --dataset literotica_stories
```

## Corpus roles (romance + gothic)

| Dataset | Content | Size | Best for |
|---------|---------|------|----------|
| `diltdicker/romance_books_32K` | Book **blurbs** + genre tags | ~25k train | Register, tone, romance metadata |
| `AlekseyKorshuk/romance-books` | Likely **full text** (gated) | ~263 MB | Long-form romance prose — accept HF terms first |
| `Dwaraka/...Gothic_Fiction` | 12 Gothic **novels** (Gutenberg) | ~1M words | Syntax, cohesion, climax, mind style |
| `taozi555/literotica-stories` | Story texts | ~645k rows (~10.8 GB) | Contemporary fiction register, dialogue |
| `mrcedric98/fiction_books` | Book **chapters** | ~20k rows | Chapter-level narrative prose |
| `AlekseyKorshuk/fiction-books` | BookRix **full novels** (gated) | ~4.7k books | General fiction — same schema as romance-books |
| `molbal/horror-novel-chunks` | Pre-chunked Gutenberg horror | ~5.5k chunks | Horror register, atmosphere, pacing |
| `ppirli/Gutenberg-Fiction` | Gutenberg **full books** | ~23k books (~4.8 GB) | Broad literary fiction baseline |

`AlekseyKorshuk/romance-books` and `AlekseyKorshuk/fiction-books` are **gated**: log in at Hugging Face, accept the dataset terms, and enable gated repos on your token before downloading.

## Gothic Gutenberg corpus

```bash
python tools/data_preparation/download_hf_dataset.py \
  Dwaraka/Training_Dataset_of_Project_Gutebberg_Gothic_Fiction

python tools/data_preparation/convert_gutenberg_gothic.py --chunk
# → source-data/processed/gutenberg_gothic_fiction/stories.jsonl
# → source-data/processed/gutenberg_gothic_fiction/chunks.jsonl
```

## Korshuk romance books (BookRix full novels)

```bash
python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/romance-books
python tools/data_preparation/convert_romance_books_korshuk.py --chunk
# English-only by default; skipped rows → skipped_non_english.jsonl
# → source-data/processed/romance_books_korshuk/stories.jsonl
# → source-data/processed/romance_books_korshuk/chunks.jsonl
```

Schema: `url` + `text` (3548 full books, ~3.2k words median). Requires accepting gated terms on Hugging Face.

## Convert all HF sources to unified JSONL

```bash
python tools/data_preparation/convert_hf_sources.py
# → source-data/unified/fiction_corpus.jsonl
```

## Extract romance parquet by story / author

The `diltdicker/romance_books_32K` parquet has one row per book. The `description`
column is the prose blurb to classify (not full novel text). Extract and organize it
before Phase 2:

```bash
python tools/data_preparation/split_romance_parquet.py --chunk --by-author
# → source-data/processed/romance_books_32k/stories.jsonl
# → source-data/processed/romance_books_32k/chunks.jsonl
# → source-data/processed/romance_books_32k/by_author/<author_slug>/stories.jsonl
```

Then run Phase 2 on the chunked corpus:

```bash
python tools/style_classification/run_pipeline.py \
  --input source-data/processed/romance_books_32k/chunks.jsonl \
  --output train/romance_corpus/romance_32k_styled.jsonl \
  --no-llm
```

```bash
python tools/style_classification/run_pipeline.py \
  --input source-data/unified/fiction_corpus.jsonl \
  --output train/romance_corpus/hf_styled.jsonl \
  --no-llm
```
