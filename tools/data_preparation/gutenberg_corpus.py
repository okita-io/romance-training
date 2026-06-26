"""Split Project Gutenberg concatenated corpora into books and strip boilerplate."""

from __future__ import annotations

import re
from typing import Iterator

_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK (.+?)\s*\*\*\*",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THIS )?PROJECT GUTENBERG EBOOK .+?\s*\*\*\*",
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r"^Title:\s*(.+)$", re.MULTILINE)
_AUTHOR_RE = re.compile(r"^Author:\s*(.+)$", re.MULTILINE)


def title_slug(title: str) -> str:
    slug = title.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return re.sub(r"_+", "_", slug).strip("_") or "untitled"


def parse_title_author(block: str) -> tuple[str | None, str | None]:
    title_m = _TITLE_RE.search(block[:4000])
    author_m = _AUTHOR_RE.search(block[:4000])
    title = title_m.group(1).strip() if title_m else None
    author = author_m.group(1).strip() if author_m else None
    return title, author


def strip_gutenberg_boilerplate(text: str) -> str:
    """Keep prose between START and END markers when present."""
    start_m = _START_RE.search(text)
    if start_m:
        text = text[start_m.end() :]
    end_m = _END_RE.search(text)
    if end_m:
        text = text[: end_m.start()]
    return text.strip()


def split_gutenberg_corpus(text: str) -> list[dict[str, str]]:
    """
    Split a concatenated Gutenberg corpus into books.

    Uses *** START OF PROJECT GUTENBERG EBOOK *** markers. Falls back to a
    single book if no markers are found.
    """
    matches = list(_START_RE.finditer(text))
    if not matches:
        cleaned = strip_gutenberg_boilerplate(text)
        title, author = parse_title_author(text)
        return [{
            "title": title or "Unknown",
            "author": author,
            "text": cleaned,
        }]

    books: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        lookback = text[max(0, start - 4000) : start]
        fallback_title = match.group(1).strip().title()
        title, author = parse_title_author(lookback + block[:2000])
        body = strip_gutenberg_boilerplate(block)
        if not body:
            continue
        books.append({
            "title": title or fallback_title,
            "author": author,
            "text": body,
        })
    return books


def iter_book_chunks(
    books: list[dict[str, str]],
    *,
    target_words: int = 500,
    overlap_sentences: int = 2,
) -> Iterator[dict]:
    """Yield chunk records from parsed books using sentence-aware chunking."""
    from tools.style_classification.chunk_text import chunk_by_sentences

    for book_idx, book in enumerate(books):
        pieces = chunk_by_sentences(
            book["text"],
            target_words=target_words,
            overlap_sentences=overlap_sentences,
        )
        story_key = f"{title_slug(book['title'])}:{book_idx}"
        for chunk_idx, piece in enumerate(pieces):
            yield {
                "text": piece,
                "metadata": {
                    "title": book["title"],
                    "author": book.get("author"),
                    "title_slug": title_slug(book["title"]),
                    "story_key": story_key,
                    "story_id": book_idx,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(pieces),
                },
            }
