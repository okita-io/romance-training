"""Corpus path configuration for the style classifier pipeline."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_REPO_CORPUS = REPO_ROOT / "train" / "romance_corpus"
STAGING_ROOT = REPO_ROOT / "train" / "staging"
PIPELINE_CHUNKS_DIR = STAGING_ROOT / "pipeline_chunks"
CORPUS_BACKUPS_DIR = STAGING_ROOT / "backups"

# Override with ROMANCE_CORPUS_ROOT to point at a larger external corpus tree.
CORPUS_ROOT = Path(os.environ.get("ROMANCE_CORPUS_ROOT", REPO_ROOT / "data" / "corpus"))

# --- Gutenberg source ---

GUTENBERG_SOURCE = CORPUS_ROOT / "sources" / "project_gutenberg" / "train.jsonl"


def project_gutenberg_source() -> Path:
    """Return the Gutenberg JSONL path, preferring external corpus over in-repo fallback."""
    candidates = (
        GUTENBERG_SOURCE,
        IN_REPO_CORPUS / "gutenberg_romance.jsonl",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


# --- Style pipeline paths ---

STYLE_RUBRIC = REPO_ROOT / "source" / "style_rubric.json"
STYLE_ANALYSIS_SYSTEM = REPO_ROOT / "source" / "extracted" / "style_analysis_system.json"
STYLE_KNOWLEDGE = REPO_ROOT / "source" / "extracted" / "style_knowledge.jsonl"
STYLE_MARKDOWN = REPO_ROOT / "source" / "extracted" / "Style-in-Fiction.md"
STYLE_PARSED = REPO_ROOT / "source" / "Style-in-Fiction.parsed.md"
GUTENBERG_STYLED = IN_REPO_CORPUS / "gutenberg_styled.jsonl"
STYLE_TRAINING_DIR = REPO_ROOT / "train" / "style_training"
STYLE_TRAIN = STYLE_TRAINING_DIR / "train.jsonl"
STYLE_VAL = STYLE_TRAINING_DIR / "validation.jsonl"

# --- HF source data ---

SOURCE_DATA = REPO_ROOT / "source-data"
HF_SOURCE_ROOT = SOURCE_DATA / "hf"
HF_MANIFESTS = SOURCE_DATA / "manifests"
UNIFIED_FICTION_CORPUS = SOURCE_DATA / "unified" / "fiction_corpus.jsonl"
ROMANCE_32K_PROCESSED = SOURCE_DATA / "processed" / "romance_books_32k"
ROMANCE_32K_STORIES = ROMANCE_32K_PROCESSED / "stories.jsonl"
ROMANCE_32K_CHUNKS = ROMANCE_32K_PROCESSED / "chunks.jsonl"


def unified_fiction_corpus() -> Path:
    """Return unified HF fiction corpus if built; else the path where it would be written."""
    return UNIFIED_FICTION_CORPUS


def gutenberg_styled_source() -> Path:
    """Return the style-enriched JSONL if it exists; fall back to plain Gutenberg."""
    if GUTENBERG_STYLED.exists():
        return GUTENBERG_STYLED
    return project_gutenberg_source()
