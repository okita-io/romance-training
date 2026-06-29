"""Strip Victorian serial-fiction section titles prefixed with ``. ``."""

from __future__ import annotations

import re
from dataclasses import dataclass

# ``. "ring out your bells! …" three days after…``
_DOT_TITLE_QUOTE_END_RE = re.compile(
    r'^\. (?P<title>.+[!?]")\s+(?P<body>.+)$',
    re.DOTALL,
)

_NARRATIVE_START_RE = re.compile(
    r"^(?:"
    r"as |when |while |after |before |upon |with |but |meantime |meanwhile |"
    r"in |on |at |one |two |three |four |five |six |seven |eight |nine |ten |"
    r"thirty |twenty |fifteen |twelve |half |early |to-morrow |"
    r"it |he |she |they |there |here |"
    r"miss |mr |mrs |sir |lady |lord |captain |jules |heart |"
    r"the (?:monday|morning|afternoon|evening|night|mention|time|baby|debilitated|"
    r"circumlocution|father|last|sun|day|awaking|manager|ebbing)"
    r")",
    re.IGNORECASE,
)

# Gutenberg edition errata / smashed front-matter TOC (Scarlet Letter and similar).
_EDITORIAL_ERRATA_OPENING_RE = re.compile(
    r"^\. conclusion addendum\b",
    re.IGNORECASE,
)

_CUSTOM_HOUSE_NARRATIVE_RE = re.compile(
    r"\bthe custom house in my native town\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StripResult:
    text: str
    stripped: bool
    title: str | None = None


def _word_count(text: str) -> int:
    return len(text.split())


def _looks_like_section_title(title: str) -> bool:
    title = title.strip()
    if not title:
        return False
    words = title.split()
    if len(words) > 12:
        return False
    if title.count(",") > 2:
        return False
    # Long quoted dialogue is usually a section epigraph, not narrative.
    if title.startswith('"') and len(words) <= 14:
        return True
    return True


def _looks_like_narrative_start(body: str) -> bool:
    body = body.strip()
    if _word_count(body) < 20:
        return False
    if body.startswith(('"', "'")) and _word_count(body) >= 20:
        return True
    if _NARRATIVE_START_RE.match(body):
        return True
    if body and body[0].isupper():
        first = body.split()[0].lower()
        if first in {"thirty", "twenty", "fifteen", "twelve", "eight", "seven"}:
            return True
    return False


def _strip_dot_title_period(text: str) -> StripResult | None:
    if not text.startswith(". "):
        return None
    rest = text[2:]
    # Section titles are short; do not scan megabyte chapters sentence by sentence.
    search_region = rest[:280]
    for match in re.finditer(r"\. ", search_region):
        title = rest[: match.start()].strip()
        body = rest[match.end() :].strip()
        if not title or not body:
            continue
        if not _looks_like_section_title(title):
            continue
        if not _looks_like_narrative_start(body):
            continue
        return StripResult(body, True, title=title)
    return None


def _strip_dot_title_quote_end(text: str) -> StripResult | None:
    match = _DOT_TITLE_QUOTE_END_RE.match(text)
    if not match:
        return None
    title = match.group("title").strip()
    body = match.group("body").strip()
    if not _looks_like_section_title(title):
        return None
    if not _looks_like_narrative_start(body):
        return None
    return StripResult(body, True, title=title)


def _strip_dot_title_words(text: str) -> StripResult | None:
    """Dickens-style ``. family affairs as the city clocks…`` (no period after title)."""
    body = text[2:].strip()
    words = body.split()
    if len(words) < 24:
        return None

    for n in range(2, min(13, len(words) - 20)):
        title = " ".join(words[:n])
        rest = " ".join(words[n:])
        if not _looks_like_section_title(title):
            continue
        if not _looks_like_narrative_start(rest):
            continue
        return StripResult(rest, True, title=title)
    return None


def _strip_editorial_errata_prefix(text: str) -> StripResult | None:
    """Drop revision errata / plot-summary blobs before the narrative Custom House."""
    if not _EDITORIAL_ERRATA_OPENING_RE.match(text):
        return None
    match = _CUSTOM_HOUSE_NARRATIVE_RE.search(text)
    if not match:
        return StripResult("", True, title="editorial errata")
    return StripResult(text[match.start() :].strip(), True, title="editorial errata")


def strip_victorian_section_title(text: str) -> StripResult:
    """Remove a leading ``. section title`` when it precedes chapter narrative."""
    text = text.strip()
    if not text.startswith(". "):
        return StripResult(text, False)

    errata = _strip_editorial_errata_prefix(text)
    if errata is not None:
        return errata

    for fn in (_strip_dot_title_period, _strip_dot_title_quote_end, _strip_dot_title_words):
        result = fn(text)
        if result is not None:
            return result
    return StripResult(text, False)


def strip_record(record: dict) -> dict:
    out = dict(record)
    result = strip_victorian_section_title(record.get("text") or "")
    out["text"] = result.text
    metadata = dict(out.get("metadata") or {})
    metadata["word_count"] = _word_count(result.text)
    if result.stripped:
        metadata["victorian_section_title_stripped"] = True
        if result.title:
            metadata["victorian_section_title"] = result.title
    out["metadata"] = metadata
    return out
