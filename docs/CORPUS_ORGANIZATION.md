# Romance Corpus Organization

> **Central repository for all romance training data.** Override with `ROMANCE_CORPUS_ROOT` (default: `data/corpus/` under this repo).

## Directory Structure

```
romance-corpus/
├── sources/                    # Raw/original data sources
│   ├── youtube/
│   │   └── collection_markdown/     # 38 markdown files from YouTube transcripts
│   ├── project_gutenberg/
│   │   └── train.jsonl              # 8,371 samples (15MB) from Project Gutenberg
│   └── external/
│       └── fiction1b_enhanced/      # 4GB Fiction-1B dataset (unfiltered)
├── training/
│   ├── processed/
│   │   └── youtube_combined_v3/
│   │       ├── train.jsonl          # 4,679 samples (3.0M words)
│   │       ├── validation.jsonl     # 520 samples (416K words)
│   │       └── dataset_info.json    # Dataset statistics
│   └── subsets/
│       └── by_heat/                 # Heat-level filtered training sets
│           ├── train_all_4000.jsonl
│           ├── train_explicit.jsonl
│           ├── train_steamy.jsonl
│           ├── train_moderate.jsonl
│           ├── train_mild.jsonl
│           ├── train_sweet.jsonl
│           └── MANIFEST.json
└── manifests/
    └── initial_organization.json   # This organization's manifest
```

## Dataset Summary

| Dataset | Size | Samples | Words | Purpose |
|---------|------|---------|-------|---------|
| YouTube Combined | 18 MB | 5,199 | 3.45M | **Primary training data** - clean, heat-labeled |
| Project Gutenberg | 15 MB | 8,371 | ~? | Additional corpus (in repo) |
| Fiction-1B External | 4 GB | - | 500M+ | Raw source for filtering |
| Heat Subsets | 12 MB | 4,000+ | - | Heat-level specialized training |

## How to Use

### For Training (autotrain / Unsloth)

**Main training** (YouTube corpus):
```yaml
data:
  path: /Users/alexokita/romance-corpus/training/processed/youtube_combined_v3
  train_split: train
  valid_split: validation
```

**Quick smoke test** (Project Gutenberg):
```yaml
data:
  path: /Users/alexokita/romance-corpus/sources/project_gutenberg
  train_split: train
  valid_split: validation  # Note: may need to create validation split
```

### Adding New Data

1. **YouTube transcripts**: 
   ```bash
   python src/romance_factory/data_collection/youtube_romance_harvest.py --video-ids VID1 VID2
   python src/romance_factory/data_collection/convert_collection_to_training.py --recreate
   ```
   Output goes to `training/processed/youtube_combined_v3/`

2. **External datasets**:
   Place under `sources/external/` and update scripts accordingly.

## Migration Notes

All previous scattered locations have been consolidated:

| Old Location | New Location |
|--------------|--------------|
| `~/hermes/romance-training-data-combined-v3` | `training/processed/youtube_combined_v3` |
| `~/romance-data-external/fiction1b/enhanced-fast` | `sources/external/fiction1b_enhanced` |
| `~/romance-factory/data/romance_corpus` | `sources/project_gutenberg/` |
| `~/hermes/romance-collection-final` | `sources/youtube/collection_markdown/` |
| `.../training_by_heat` | `training/subsets/by_heat/` |

**Important**: All scripts have been updated to use new paths. If you encounter hardcoded paths, update them to reference the central corpus.

## LAN Transfer

This central location is optimized for fast LAN transfer to your PC:

```bash
# From Mac to PC (using rsync over network)
rsync -avz /Users/alexokita/romance-corpus/ user@pc-ip:/path/to/romance-corpus/
```

Large files ( Fiction-1B: 4GB) transfer faster on LAN than git push/pull.

Created: 2026-04-03
