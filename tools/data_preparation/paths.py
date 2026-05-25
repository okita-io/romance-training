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


def fiction1b_source() -> Path:
    return SOURCES["fiction1b_enhanced"]
