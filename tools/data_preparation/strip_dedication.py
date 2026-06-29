"""Strip chapter dedications and BookRix/HF boilerplate from fiction chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WIKI_BLOCK_RE = re.compile(r"\[\[[^\]]*\]\]", re.DOTALL)

_CHAPTER_DEDICATION_OPENING_RE = re.compile(
    r"^this chapter is dedicated to\b",
    re.IGNORECASE,
)

_BOOK_DEDICATION_OPENING_RE = re.compile(
    r"^(?:this book is dedicated to|dedicated to my)\b",
    re.IGNORECASE,
)

_BOOKRIX_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"m\.b\. julien anthology complex bookrix|"
    r"bookrix gmbh|"
    r"this book was distributed courtesy of|"
    r"for your own unlimited reading and free ebooks|"
    r"free-ebooks\.net|"
    r"copyright information|"
    r"terms of service here:"
    r")",
    re.IGNORECASE,
)

_EBOOK_TOS_END_RE = re.compile(
    r"https?://(?:www\.)?free-ebooks\.net/tos\.html\s*",
    re.IGNORECASE,
)

_EBOOK_BOILERPLATE_TRAILER_START_RE = re.compile(
    r"\bthis book was distributed courtesy of\b",
    re.IGNORECASE,
)

_COMPOSITION_LABEL_RE = re.compile(
    r"^composition\s+\d+,\s+part\s+\d+\s*$",
    re.IGNORECASE,
)

_COLON_CHAPTER_TITLE_RE = re.compile(
    r"^:\s*(.+?)\s+"
    r"(?=last night|about a year|one |the |when |as |i had|i'm |it was |there )",
    re.IGNORECASE,
)

# Little Brother-style appendix: inline ``epilogue contents`` TOC + CC license + afterwords.
_BACK_MATTER_TOC_RE = re.compile(
    r"^epilogue contents\s*(?:-\s*about this book)?",
    re.IGNORECASE,
)

_BOOKRIX_PUBLICATION_TRAILER_RE = re.compile(
    r"\s*publication\s+date\s*:\s*.+?"
    r"https?://(?:www\.)?bookrix\.com/\S+"
    r"(?:\s*isbn\s*:\s*[\d-]+)?"
    r"\s*$",
    re.IGNORECASE | re.DOTALL,
)

_DEDICATION_HINTS_RE = re.compile(
    r"\b(?:"
    r"this chapter is dedicated|dedicated to|bookstore|bookseller|bookrix|"
    r"amazon\.com|borderlands|barnes and noble|secret headquarters|"
    r"science fiction bookstore|working at bakka|writers should work|"
    r"jeff bezos|shop there like crazy|anthology of stories|"
    r"made me the mutant|largest internet bookseller|independent science fiction|"
    r"distributed courtesy of|free ebooks today|unlimited reading|"
    r"copyright information|terms of service here|free-ebooks\.net|"
    r"share this ebook with anyone|show your appreciation to the author"
    r")\b",
    re.IGNORECASE,
)

_NARRATIVE_ANCHOR_RES = [
    re.compile(r"i'm a senior at cesar chavez", re.IGNORECASE),
    re.compile(r'"i\'m thinking of majoring in physics', re.IGNORECASE),
    re.compile(r"my name is marcus yallow", re.IGNORECASE),
]

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')


@dataclass(frozen=True)
class StripResult:
    text: str
    stripped: bool
    reason: str | None = None


def _word_count(text: str) -> int:
    return len(text.split())


def _normalize_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r" +", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _strip_wiki_blocks(text: str) -> str:
    return _normalize_ws(_WIKI_BLOCK_RE.sub(" ", text))


def _cut_at_narrative_anchor(text: str, *, min_offset: int = 80) -> str | None:
    best_idx: int | None = None
    for pat in _NARRATIVE_ANCHOR_RES:
        m = pat.search(text, min_offset)
        if m is not None and (best_idx is None or m.start() < best_idx):
            best_idx = m.start()
    if best_idx is None:
        return None
    return text[best_idx:].strip()


def _skip_embedded_duplicate(text: str) -> str:
    """
    HF ``chapter_text`` sometimes repeats dedication prose after a wiki link block.

    Find a mid-size word window that occurs twice near the start and resume after
    the second copy.
    """
    words = text.split()
    if len(words) < 120:
        return text
    lower = text.lower()
    for win in range(50, 25, -5):
        if 10 + win >= len(words) // 2:
            continue
        chunk = " ".join(words[10 : 10 + win])
        first = lower.find(chunk)
        if first < 0:
            continue
        second = lower.find(chunk, first + len(chunk))
        if 200 < second < len(text) * 0.35:
            return text[second + len(chunk) :].strip()
    return text


def _is_dedication_sentence(sentence: str) -> bool:
    return bool(_DEDICATION_HINTS_RE.search(sentence))


def _strip_leading_dedication_sentences(text: str) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text
    seen: set[str] = set()
    start = 0
    for i, sent in enumerate(sentences):
        key = re.sub(r"\s+", " ", sent.lower())[:160]
        if _is_dedication_sentence(sent) or key in seen:
            seen.add(key)
            start = i + 1
            continue
        break
    if start <= 0:
        return text
    return " ".join(sentences[start:]).strip()


def _strip_chapter_dedication_body(text: str) -> StripResult | None:
    if not _CHAPTER_DEDICATION_OPENING_RE.match(text):
        return None
    cleaned = _strip_wiki_blocks(text)
    anchored = _cut_at_narrative_anchor(cleaned)
    if anchored is not None:
        cleaned = anchored
    else:
        cleaned = _skip_embedded_duplicate(cleaned)
        cleaned = _strip_leading_dedication_sentences(cleaned)
    if cleaned == text or _word_count(cleaned) < 30:
        return None
    if _CHAPTER_DEDICATION_OPENING_RE.match(cleaned):
        return None
    return StripResult(cleaned, True, reason="chapter_dedication")


def _strip_book_dedication_body(text: str) -> StripResult | None:
    if not _BOOK_DEDICATION_OPENING_RE.match(text):
        return None
    cleaned = _strip_leading_dedication_sentences(_strip_wiki_blocks(text))
    if cleaned == text or _word_count(cleaned) < 30:
        return None
    return StripResult(cleaned, True, reason="book_dedication")


def _strip_ebook_boilerplate_block(text: str) -> StripResult | None:
    """Remove free-ebooks.net / BookRix distribution blurbs at chunk start."""
    stripped = text.lstrip()
    at_start = (
        _EBOOK_BOILERPLATE_TRAILER_START_RE.match(stripped) is not None
        or re.match(
            r"(?:m\.b\.|bookrix gmbh|for your own unlimited reading)",
            stripped,
            re.IGNORECASE,
        )
        is not None
    )
    if not at_start:
        return None

    cleaned = text
    tos = _EBOOK_TOS_END_RE.search(cleaned)
    if tos:
        cleaned = cleaned[tos.end() :].strip()
    else:
        cleaned = _strip_leading_dedication_sentences(_strip_wiki_blocks(cleaned))

    cleaned = _COMPOSITION_LABEL_RE.sub("", cleaned).strip()
    if cleaned == text:
        return None
    return StripResult(cleaned, True, reason="ebook_boilerplate")


def _strip_back_matter_toc(text: str) -> StripResult | None:
    """Drop appendix rows: inline back-matter TOC, not narrative ``epilogue …`` prose."""
    if not _BACK_MATTER_TOC_RE.match(text):
        return None
    return StripResult("", True, reason="back_matter_toc")


def _strip_colon_chapter_title(text: str) -> StripResult | None:
    """BookRix-style ``: chapter title narrative…`` openings."""
    match = _COLON_CHAPTER_TITLE_RE.match(text)
    if not match:
        return None
    title = match.group(1).strip()
    if not title or len(title.split()) > 12:
        return None
    body = text[match.end() :].strip()
    if body == text:
        return None
    return StripResult(body, True, reason="colon_chapter_title")


def _strip_bookrix_publication_trailer(text: str) -> StripResult | None:
    """Trailing ``publication date: … https://www.bookrix.com/…`` metadata."""
    match = _BOOKRIX_PUBLICATION_TRAILER_RE.search(text)
    if not match:
        return None
    cleaned = text[: match.start()].strip()
    if cleaned == text:
        return None
    return StripResult(cleaned, True, reason="bookrix_publication_trailer")


def _strip_ebook_boilerplate_trailer(text: str) -> StripResult | None:
    """Trailing free-ebooks.net / BookRix distribution and TOS blocks after narrative."""
    match = _EBOOK_BOILERPLATE_TRAILER_START_RE.search(text)
    if match is None or match.start() <= 0:
        return None
    if _word_count(text[: match.start()]) < 20:
        return None
    tail = text[match.start() :]
    if not _BOOKRIX_BOILERPLATE_RE.search(tail):
        return None
    cleaned = text[: match.start()].strip()
    if cleaned == text:
        return None
    return StripResult(cleaned, True, reason="ebook_boilerplate_trailer")


def strip_dedication_and_boilerplate(text: str) -> StripResult:
    """Remove chapter dedications, wiki link blocks, and common ebook boilerplate."""
    text = text.strip()
    if not text:
        return StripResult("", False)

    stripped = False
    reason: str | None = None

    for fn in (
        _strip_back_matter_toc,
        _strip_chapter_dedication_body,
        _strip_book_dedication_body,
        _strip_ebook_boilerplate_block,
        _strip_colon_chapter_title,
    ):
        result = fn(text)
        if result is not None:
            text = result.text
            stripped = True
            reason = result.reason

    if not stripped and _WIKI_BLOCK_RE.search(text):
        cleaned = _strip_wiki_blocks(text)
        if cleaned != text and _word_count(cleaned) >= 30:
            text = cleaned
            stripped = True
            reason = "wiki_blocks"

    trailer = _strip_ebook_boilerplate_trailer(text)
    if trailer is not None:
        text = trailer.text
        stripped = True
        if reason is None:
            reason = trailer.reason

    trailer = _strip_bookrix_publication_trailer(text)
    if trailer is not None:
        text = trailer.text
        stripped = True
        if reason is None:
            reason = trailer.reason

    return StripResult(text, stripped, reason=reason)


def strip_record(record: dict, *, min_words: int = 30) -> dict:
    out = dict(record)
    result = strip_dedication_and_boilerplate(record.get("text") or "")
    out["text"] = result.text
    metadata = dict(out.get("metadata") or {})
    metadata["word_count"] = _word_count(result.text)
    if result.stripped:
        metadata["dedication_stripped"] = True
        if result.reason:
            metadata["dedication_strip_reason"] = result.reason
    out["metadata"] = metadata
    if metadata["word_count"] < min_words:
        return out
    return out
