"""Sentence-boundary-aware text chunking for analytic fidelity."""

from __future__ import annotations

import re
from typing import Any, Literal

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')

Grain = Literal["sentence", "span", "act"]
GRAINS: tuple[Grain, ...] = ("sentence", "span", "act")

# Kev / Jev training state budget (~384 English tokens).
KEV_STATE_CHARS = 1400
KEV_MIN_WORDS = 80


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


def take_char_window(
    sentences: list[str],
    start: int,
    *,
    max_chars: int = KEV_STATE_CHARS,
) -> str:
    """Take complete sentences from ``start`` until ``max_chars``."""
    kept: list[str] = []
    size = 0
    for sent in sentences[start:]:
        extra = len(sent) if not kept else len(sent) + 1
        if kept and size + extra > max_chars:
            break
        if not kept and extra > max_chars:
            return sent[: max_chars - 1].rstrip() + "…"
        kept.append(sent)
        size += extra
    return " ".join(kept).strip()


def random_kev_span(
    text: str,
    rng,
    *,
    max_chars: int = KEV_STATE_CHARS,
    min_words: int = KEV_MIN_WORDS,
) -> tuple[str, bool]:
    """
    Return a sentence-bounded span that fits Kev's training state.

    If the whole passage already fits, it is returned unchanged. Otherwise a
    random sentence start is tried until the window has at least ``min_words``.
    """
    text = (text or "").strip()
    if not text:
        return "", False
    if len(text) <= max_chars:
        return text, False
    sentences = split_sentences(text)
    if not sentences:
        return text[: max_chars - 1].rstrip() + "…", True
    starts = list(range(len(sentences)))
    rng.shuffle(starts)
    fallback = take_char_window(sentences, 0, max_chars=max_chars)
    for start in starts:
        chunk = take_char_window(sentences, start, max_chars=max_chars)
        if chunk and _word_count(chunk) >= min_words:
            return chunk, True
    return fallback, True


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
    ranges = _window_ranges(
        sentences,
        target_words=target_words,
        overlap_sentences=overlap_sentences,
        min_words=min_words,
    )
    return [" ".join(sentences[start:end]).strip() for start, end in ranges]


def _window_ranges(
    sentences: list[str],
    *,
    target_words: int,
    overlap_sentences: int = 2,
    min_words: int = 30,
) -> list[tuple[int, int]]:
    """Return (start, end) sentence-index windows (end exclusive)."""
    if not sentences:
        return []

    total_words = sum(_word_count(s) for s in sentences)
    if total_words <= int(target_words * 1.5):
        return [(0, len(sentences))]

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(sentences):
        end = start
        words = 0
        while end < len(sentences) and words < target_words:
            words += _word_count(sentences[end])
            end += 1

        chunk_words = sum(_word_count(sentences[i]) for i in range(start, end))
        if chunk_words >= min_words:
            ranges.append((start, end))

        if end >= len(sentences):
            break
        start = max(start + 1, end - overlap_sentences)

    return ranges


def _story_id(record: dict[str, Any]) -> str:
    meta = record.get("metadata") or {}
    for key in ("story_id", "book_id", "source", "title"):
        val = meta.get(key) or record.get(key)
        if val:
            return str(val)
    return "unknown"


def _legacy_chunk_index(record: dict[str, Any]) -> int | None:
    meta = record.get("metadata") or {}
    for key in ("chunk_index", "legacy_chunk_index"):
        if key in meta and meta[key] is not None:
            try:
                return int(meta[key])
            except (TypeError, ValueError):
                return None
    if "chunk_index" in record and record["chunk_index"] is not None:
        try:
            return int(record["chunk_index"])
        except (TypeError, ValueError):
            return None
    return None


def _make_row(
    *,
    text: str,
    source: str,
    story_id: str,
    grain: Grain,
    chunk_index: int,
    total_chunks: int,
    chunk_size: int,
    overlap_sentences: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "text": text,
        "metadata": {
            "source": source,
            "story_id": story_id,
            "grain": grain,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
            "chunk_overlap_sentences": overlap_sentences,
            "chunk_boundary": "sentence",
            "word_count": _word_count(text),
            **extra,
        },
    }


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


def chunk_record_multigrain(
    record: dict[str, Any],
    *,
    span_words: int = 300,
    act_words: int = 1000,
    overlap_sentences: int = 2,
) -> dict[Grain, list[dict[str, Any]]]:
    """
    Emit sentence / span / act rows from one source record with stable parent ids.

    Acts (~act_words) and spans (~span_words) share sentence boundaries. Each span
    links to the act containing its midpoint sentence; each sentence links to the
    first span that contains it. When the input is an existing ~500w chunk,
    ``legacy_chunk_index`` / ``parent_legacy_chunk_id`` are set for lineage.
    """
    text = (record.get("text") or "").strip()
    empty: dict[Grain, list[dict[str, Any]]] = {"sentence": [], "span": [], "act": []}
    if not text:
        return empty

    sentences = split_sentences(text)
    if not sentences:
        return empty

    source = record.get("source") or record.get("metadata", {}).get("source", "unknown")
    story_id = _story_id(record)
    legacy_idx = _legacy_chunk_index(record)
    legacy_extra: dict[str, Any] = {}
    if legacy_idx is not None:
        legacy_extra["legacy_chunk_index"] = legacy_idx
        legacy_extra["parent_legacy_chunk_id"] = f"{story_id}:legacy:{legacy_idx:04d}"

    act_ranges = _window_ranges(
        sentences,
        target_words=act_words,
        overlap_sentences=overlap_sentences,
        min_words=min(30, sum(_word_count(s) for s in sentences) or 1),
    )
    span_ranges = _window_ranges(
        sentences,
        target_words=span_words,
        overlap_sentences=overlap_sentences,
        min_words=min(30, sum(_word_count(s) for s in sentences) or 1),
    )

    act_ids = [f"{story_id}:act:{i:04d}" for i in range(len(act_ranges))]
    span_ids = [f"{story_id}:span:{i:04d}" for i in range(len(span_ranges))]

    def act_for_sentence(sent_idx: int) -> str | None:
        for act_i, (a0, a1) in enumerate(act_ranges):
            if a0 <= sent_idx < a1:
                return act_ids[act_i]
        return act_ids[-1] if act_ids else None

    def span_for_sentence(sent_idx: int) -> str | None:
        for span_i, (s0, s1) in enumerate(span_ranges):
            if s0 <= sent_idx < s1:
                return span_ids[span_i]
        return span_ids[-1] if span_ids else None

    acts: list[dict[str, Any]] = []
    for i, (a0, a1) in enumerate(act_ranges):
        piece = " ".join(sentences[a0:a1]).strip()
        act_id = act_ids[i]
        acts.append(
            _make_row(
                text=piece,
                source=source,
                story_id=story_id,
                grain="act",
                chunk_index=i,
                total_chunks=len(act_ranges),
                chunk_size=act_words,
                overlap_sentences=overlap_sentences,
                extra={
                    "act_id": act_id,
                    "sentence_start": a0,
                    "sentence_end": a1,
                    **legacy_extra,
                },
            )
        )

    spans: list[dict[str, Any]] = []
    for i, (s0, s1) in enumerate(span_ranges):
        piece = " ".join(sentences[s0:s1]).strip()
        mid = (s0 + s1 - 1) // 2
        parent_act = act_for_sentence(mid)
        span_id = span_ids[i]
        spans.append(
            _make_row(
                text=piece,
                source=source,
                story_id=story_id,
                grain="span",
                chunk_index=i,
                total_chunks=len(span_ranges),
                chunk_size=span_words,
                overlap_sentences=overlap_sentences,
                extra={
                    "span_id": span_id,
                    "act_id": parent_act,
                    "parent_act_id": parent_act,
                    "sentence_start": s0,
                    "sentence_end": s1,
                    **legacy_extra,
                },
            )
        )

    sent_rows: list[dict[str, Any]] = []
    for i, sent in enumerate(sentences):
        sentence_id = f"{story_id}:sent:{i:04d}"
        parent_span = span_for_sentence(i)
        parent_act = act_for_sentence(i)
        sent_rows.append(
            _make_row(
                text=sent,
                source=source,
                story_id=story_id,
                grain="sentence",
                chunk_index=i,
                total_chunks=len(sentences),
                chunk_size=1,
                overlap_sentences=0,
                extra={
                    "sentence_id": sentence_id,
                    "span_id": parent_span,
                    "parent_span_id": parent_span,
                    "act_id": parent_act,
                    "parent_act_id": parent_act,
                    **legacy_extra,
                },
            )
        )

    return {"sentence": sent_rows, "span": spans, "act": acts}
