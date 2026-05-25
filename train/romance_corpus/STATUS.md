# Romance Corpus Collection Status

## Completed ✓

### 1. Project Gutenberg Romance Collection
- **Status:** COMPLETE
- **Books Downloaded:** 25 classic romance novels
- **Raw Files:** data/romance_corpus/gutenberg/ (34MB total)
- **Processed JSONL:** data/romance_corpus/gutenberg_romance.jsonl (25 books)
- **Training Data Generated:** 
  - train.jsonl: 7,534 samples
  - validation.jsonl: 837 samples
- **Configuration:**
  - Chunk size: 2000 characters
  - Overlap: 200 characters  
  - Preamble: "You are writing commercial romance fiction."
  - Validation split: 10%

**Notable titles included:**
- Pride and Prejudice (Jane Austen)
- Jane Eyre (Charlotte Brontë)
- Sense and Sensibility (Jane Austen)
- Romeo and Juliet (Shakespeare)
- A Room with a View (E. M. Forster)

## In Progress ⏳

### 2. Fiction-1B Dataset (HuggingFace)
- **Status:** DOWNLOADING
- **Expected Size:** ~5GB download, filtered to romance subset
- **Script:** download_fiction1b.py
- **Next Step:** After download completes, merge with existing corpus

## Planned 📋

### 3. Additional HuggingFace Datasets
- ai-danger/spicyfiction (adult romance, chat format)
- AlekseyKorshuk/romance-books (gated, needs agreement)
- AlekseyKorshuk/fiction-books (general fiction)

### 4. Smashwords Public Books
- Use CoreWeave scraper (Go-based)
- Category: Western Romance (ID 1245)
- License: Check individual book terms

### 5. Romancely.com Structure Reference
- Use browser_harness with Chrome DevTools MCP
- Purpose: Story structure analysis only (no full text)
- Output: story_references.jsonl

## Next Steps

1. Monitor Fiction-1B download completion
2. Filter Fiction-1B for romance content
3. Merge all sources with collect_corpus.py
4. Test training run with current corpus
5. Evaluate if more data is needed
6. Consider modern sources (Smashwords, etc.)

## Training Corpus Summary (Current)

**Source:** Project Gutenberg Romance
**Total Samples:** 8,371
- Training: 7,534
- Validation: 837

**Estimated Word Count:** ~3-5 million words
**License:** Public Domain
**Ready for Training:** YES

## Commands for Next Steps

# Merge Fiction-1B when ready:
python3 pipeline/collect_corpus.py \
  --from-jsonl data/romance_corpus/gutenberg_romance.jsonl \
               data/romance_corpus/fiction1b/*.jsonl \
  --out-dir data/romance_corpus \
  --preamble "You are writing commercial romance fiction." \
  --chunk-chars 2000 \
  --chunk-overlap 200 \
  --validation-fraction 0.1

# Start training:
python3 pipeline/run_pipeline.py train --write-config-only
