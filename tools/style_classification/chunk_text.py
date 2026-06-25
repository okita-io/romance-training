"""Sentence-boundary-aware text chunking for analytic fidelity."""

from __future__ import annotations

import re
from typing import Any

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences; keeps fragments when boundaries are unclear."""
    text = text.strip()
    if not text:
        return []

    parts = _SENT_SPLIT_RE.split(text)
    sents = [p.strip() for p in parts if p.strip()]
    return sents if sents else [text]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_by_sentences(
    text: str,
    *,
    target_words: int = 500,
    overlap_sentences: int = 2,
    min_words: int = 30,
) -> list[str]:
    """
    Build chunks from complete sentences, targeting ~target_words per chunk.

    Chunk boundaries align to sentence ends; consecutive chunks share
    overlap_sentences trailing/leading sentences.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    total_words = _word_count(text)
    if total_words <= int(target_words * 1.5):
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        end = start
        words = 0
        while end < len(sentences) and words < target_words:
            words += _word_count(sentences[end])
            end += 1

        chunk_text = " ".join(sentences[start:end]).strip()
        if _word_count(chunk_text) >= min_words:
            chunks.append(chunk_text)

        if end >= len(sentences):
            break
        start = max(start + 1, end - overlap_sentences)

    return chunks


def chunk_record(
    record: dict[str, Any],
    *,
    target_words: int = 500,
    overlap_sentences: int = 2,
) -> list[dict]:
    """Split a long corpus record into sentence-aware chunks with metadata."""
    text = record.get("text", "")
    if not text:
        return [record]

    pieces = chunk_by_sentences(
        text,
        target_words=target_words,
        overlap_sentences=overlap_sentences,
    )
    if len(pieces) <= 1:
        return [record]

    source = record.get("source") or record.get("metadata", {}).get("source", "unknown")
    base_meta = {k: v for k, v in record.items() if k != "text"}
    total = len(pieces)

    chunks: list[dict] = []
    for idx, piece in enumerate(pieces):
        chunks.append({
            "text": piece,
            "metadata": {
                "source": source,
                "chunk_index": idx,
                "total_chunks": total,
                "chunk_size": target_words,
                "chunk_overlap_sentences": overlap_sentences,
                "chunk_boundary": "sentence",
                "word_count": _word_count(piece),
                **{k: v for k, v in base_meta.items() if k not in ("metadata",)},
            },
        })
    return chunks
