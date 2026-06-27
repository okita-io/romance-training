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

_LICENSE_PREAMBLE_RE = re.compile(
    r"\A(?:The )?Project Gutenberg EBook of .+?\n\n|"
    r"This eBook is for the use of anyone anywhere at no cost and with\n"
    r"almost no restrictions whatsoever\.\s+You may copy it, give it away or\n"
    r"re-use it under the terms of the Project Gutenberg License included\n"
    r"with this eBook or online at www\.gutenberg\.org\.\s*\n\n|"
    r"This eBook is for the use of anyone anywhere at no cost and with\n"
    r"almost no restrictions whatsoever\.\s+You may copy it, give it away or\n\n",
    re.IGNORECASE | re.DOTALL,
)
_METADATA_LINE_RE = re.compile(
    r"^(?:Title|Author|Translator|Editor|Release Date|Posting Date|Language|"
    r"Character set encoding|EBook #|EBook No\.|Credits|Updated)\s*:.*$|"
    r"^\[This file was .+\]$|"
    r"^\[eBook #\d+\]$",
    re.IGNORECASE | re.MULTILINE,
)
_TRANSCRIBER_RE = re.compile(
    r"^Produced by .+(?:\n(?![ \t]*\n)[^\n]+)*",
    re.IGNORECASE | re.MULTILINE,
)
_ILLUSTRATION_LINE_RE = re.compile(
    r"^[ \t]*\[(?:Illustration|Frontispiece|Picture|Image|Plate|Map|Diagram)"
    r"[^\]]*\][ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_ILLUSTRATION_INLINE_RE = re.compile(
    r"\[(?:Illustration|Frontispiece|Picture|Image|Plate|Map|Diagram)[^\]]*\]",
    re.IGNORECASE,
)
_TOC_START_RE = re.compile(
    r"^[ \t]*(?:Table of Contents|TABLE OF CONTENTS|Contents|CONTENTS(?:\s+OF\s+VOL[^\n]*)?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_EDITION_NOTICE_RE = re.compile(
    r"^[ \t]*(?:THE )?[A-Z][A-Z0-9 \.\-]{3,} EDITION(?:\s+[\d\.]+)?[ \t]*$",
    re.MULTILINE,
)
_TRAILING_LICENSE_RE = re.compile(
    r"(?:\*{3}\s*)?End of (?:the )?Project Gutenberg(?:'s)? .+$|"
    r"^.*?www\.gutenberg\.org/license.*?$",
    re.IGNORECASE | re.DOTALL,
)
_CREDIT_LINE_RE = re.compile(
    r"^(?:Transcriber(?:'s)? Notes?|Proofreading Team|Distributed Proofreading Team|"
    r"Online Distributed Proofreading Team|pgdp\.net).*$",
    re.IGNORECASE | re.MULTILINE,
)

_NOVEL_TITLE_RE = re.compile(
    r"\b(?:novel|novelized|romance|stories|tales|novelette)\b",
    re.IGNORECASE,
)
_PLAY_TITLE_STRONG_RES = (
    re.compile(r"\bThe Tragedy of\b", re.IGNORECASE),
    re.compile(r"\bThe Comedy of\b", re.IGNORECASE),
    re.compile(r": A Comedy\b", re.IGNORECASE),
    re.compile(r": A Tragedy\b", re.IGNORECASE),
    re.compile(r": A Play\b", re.IGNORECASE),
    re.compile(r"\bA Comedy for\b", re.IGNORECASE),
    re.compile(r"\b(?:Three|Four|Five)-Act (?:Play|Comedy|Tragedy|Drama)\b", re.IGNORECASE),
)
_PLAY_TITLE_WEAK_RE = re.compile(
    r"\b(?:play|drama|comedy|tragedy|farce|melodrama)\b",
    re.IGNORECASE,
)
_PLAY_BODY_RES = (
    re.compile(r"(?mi)^\s*DRAMATIS PERSONAE\s*$"),
    re.compile(r"(?mi)^\s*ACT\s+(?:I|1|ONE)(?:[\.\s:\-]|$)"),
    re.compile(r"(?mi)^\s*SCENE\s+(?:I|1|ONE)(?:[\.\s:\-]|$)"),
    re.compile(r"(?mi)^\s*\[(?:Enter|Exit|Exeunt|Aside)\b"),
)


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


def _normalize_gutenberg_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_table_of_contents(text: str) -> str:
    """Remove Contents blocks that precede the first chapter body."""
    chapter_body = re.compile(
        r"^[ \t]*CHAPTER\s+(?:"
        r"I(?:[\.\s:\-]|$)|"
        r"1(?:[\.\s:\-]|$)|"
        r"ONE\b"
        r")[^\n]*(?:\n[^\n]+)?\n\n+(?=[^\n]*[a-z])",
        re.IGNORECASE | re.MULTILINE,
    )
    while True:
        toc_start = _TOC_START_RE.search(text)
        if not toc_start:
            break
        body_start = chapter_body.search(text, toc_start.end())
        if not body_start:
            break
        text = text[: toc_start.start()] + text[body_start.start() :]
    return text


def clean_gutenberg_prose(text: str, *, strip_toc: bool = True) -> str:
    """
    Strip non-prose Gutenberg markup: license headers, transcriber credits,
    illustration tags, edition notices, and optional table-of-contents blocks.
    """
    text = strip_gutenberg_boilerplate(text)
    text = _LICENSE_PREAMBLE_RE.sub("", text, count=1)
    text = _METADATA_LINE_RE.sub("", text)
    text = _TRANSCRIBER_RE.sub("", text)
    text = _CREDIT_LINE_RE.sub("", text)
    text = _ILLUSTRATION_LINE_RE.sub("", text)
    text = _ILLUSTRATION_INLINE_RE.sub("", text)
    text = _EDITION_NOTICE_RE.sub("", text)
    if strip_toc:
        text = _strip_table_of_contents(text)
    text = _TRAILING_LICENSE_RE.sub("", text)
    return _normalize_gutenberg_whitespace(text)


def _play_body_signals(text: str, *, sample_chars: int = 15000) -> str | None:
    head = text[:sample_chars]
    hits = sum(1 for pattern in _PLAY_BODY_RES if pattern.search(head))
    if hits >= 3:
        return "body"
    stage_dirs = len(re.findall(r"^\s*\[(?:Enter|Exit|Exeunt|Aside)\b", head, re.IGNORECASE | re.MULTILINE))
    if stage_dirs >= 20:
        return "stage_directions"
    return None


def detect_gutenberg_play(title: str | None, text: str, *, sample_chars: int = 15000) -> str | None:
    """
    Return a reason string when a Gutenberg work looks like a play, else None.

    Novelizations and prose works with incidental drama vocabulary are kept.
    """
    title = title or ""
    if _NOVEL_TITLE_RE.search(title):
        return None

    for pattern in _PLAY_TITLE_STRONG_RES:
        if pattern.search(title):
            return "title"

    body_reason = _play_body_signals(text, sample_chars=sample_chars)
    if _PLAY_TITLE_WEAK_RE.search(title):
        return body_reason or None
    return body_reason


def is_gutenberg_play(title: str | None, text: str, *, sample_chars: int = 15000) -> bool:
    return detect_gutenberg_play(title, text, sample_chars=sample_chars) is not None


def split_gutenberg_corpus(text: str) -> list[dict[str, str]]:
    """
    Split a concatenated Gutenberg corpus into books.

    Uses *** START OF PROJECT GUTENBERG EBOOK *** markers. Falls back to a
    single book if no markers are found.
    """
    matches = list(_START_RE.finditer(text))
    if not matches:
        cleaned = clean_gutenberg_prose(text)
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
        body = clean_gutenberg_prose(block)
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
