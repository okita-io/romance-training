# Romance Corpus Collection Plan

## Overview
Hybrid approach combining public domain sources, licensed datasets, and modern structure references.

## Data Sources

### 1. Project Gutenberg Romance (Public Domain)
- **URL:** http://www.gutenberg.org/ebooks/bookshelf/639
- **Count:** 100+ classic romance novels
- **License:** Public domain
- **Status:** Ready to download
- **Priority:** HIGH - clean, safe to use

### 2. HuggingFace: SaladTechnologies/fiction-1b
- **URL:** https://huggingface.co/datasets/SaladTechnologies/fiction-1b
- **Size:** 1B+ words, ~20k fiction works
- **License:** Mix (check individual sources)
- **Priority:** HIGH - large volume, pre-processed

### 3. Smashwords Public Books
- **Tool:** github.com/coreweave/dataset-downloader
- **Category ID:** 1245 (Western Romance)
- **Priority:** MEDIUM - modern language, legal

## Collection Strategy - Phase 1: Quick Start (TODAY)
1. Download Fiction-1B dataset from HuggingFace
2. Filter for romance/narrative fiction
3. Run collect_corpus.py to create train.jsonl
EOF; __hermes_rc=$?; printf '__HERMES_FENCE_a9f7b3__'; exit $__hermes_rc
