"""Corpus path configuration for romance-training data preparation."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_REPO_CORPUS = REPO_ROOT / "train" / "romance_corpus"

# Override with ROMANCE_CORPUS_ROOT for a central corpus tree (see docs/CORPUS_ORGANIZATION.md).
CORPUS_ROOT = Path(os.environ.get("ROMANCE_CORPUS_ROOT", REPO_ROOT / "data" / "corpus"))

SOURCES = {
    "youtube_markdown": CORPUS_ROOT / "sources" / "youtube" / "collection_markdown",
    "project_gutenberg": CORPUS_ROOT / "sources" / "project_gutenberg" / "train.jsonl",
    "fiction1b_enhanced": CORPUS_ROOT / "sources" / "external" / "fiction1b_enhanced",
}

PROCESSED = CORPUS_ROOT / "training" / "processed"

TRAINING = {
    "youtube_combined": PROCESSED / "youtube_combined_v3",
    "project_gutenberg_normalized": PROCESSED / "project_gutenberg_normalized",
    "fiction1b_normalized": PROCESSED / "fiction1b_normalized",
    "final_combined": PROCESSED / "final_combined",
}

FINAL_COMBINED_TRAIN = TRAINING["final_combined"] / "train.jsonl"
FINAL_COMBINED_VAL = TRAINING["final_combined"] / "validation.jsonl"

# --- Style pipeline paths ---

# Rubric extracted from Style in Fiction (Leech & Short)
STYLE_RUBRIC = REPO_ROOT / "source" / "style_rubric.json"

# Gutenberg corpus enriched with style_profile metadata
GUTENBERG_STYLED = IN_REPO_CORPUS / "gutenberg_styled.jsonl"

# Multi-task instruction pairs ready for fine-tuning
STYLE_TRAINING_DIR = REPO_ROOT / "train" / "style_training"
STYLE_TRAIN = STYLE_TRAINING_DIR / "train.jsonl"
STYLE_VAL = STYLE_TRAINING_DIR / "validation.jsonl"


def project_gutenberg_source() -> Path:
    """Prefer central corpus; fall back to in-repo gutenberg_romance.jsonl."""
    candidates = (
        SOURCES["project_gutenberg"],
        IN_REPO_CORPUS / "gutenberg_romance.jsonl",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def gutenberg_styled_source() -> Path:
    """Prefer style-enriched JSONL; fall back to plain gutenberg_romance."""
    if GUTENBERG_STYLED.exists():
        return GUTENBERG_STYLED
    return project_gutenberg_source()


def fiction1b_source() -> Path:
    return SOURCES["fiction1b_enhanced"]
