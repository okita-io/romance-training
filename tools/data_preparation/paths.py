"""Corpus path configuration for the style classifier pipeline."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_REPO_CORPUS = REPO_ROOT / "train" / "romance_corpus"

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
GUTENBERG_STYLED = IN_REPO_CORPUS / "gutenberg_styled.jsonl"
STYLE_TRAINING_DIR = REPO_ROOT / "train" / "style_training"
STYLE_TRAIN = STYLE_TRAINING_DIR / "train.jsonl"
STYLE_VAL = STYLE_TRAINING_DIR / "validation.jsonl"


def gutenberg_styled_source() -> Path:
    """Return the style-enriched JSONL if it exists; fall back to plain Gutenberg."""
    if GUTENBERG_STYLED.exists():
        return GUTENBERG_STYLED
    return project_gutenberg_source()
