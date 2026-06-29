"""Detect non-narrative Gutenberg chunks (catalogs, credits, TOC, etc.)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ProseVerdict = Literal["prose", "non_prose"]

# Reuse patterns aligned with gutenberg_corpus.py where possible.
_TRANSCRIBER_RE = re.compile(
    r"(?:^|\n)(?:Produced by|E-text prepared by|Prepared by)[^\n]{0,200}"
    r"(?:Distributed Proofreading|Online Distributed Proofreading|pgdp\.net|"
    r"Internet Archive \(http?://www\.archive\.org\))",
    re.IGNORECASE,
)
_GUTENBERG_LICENSE_RE = re.compile(
    r"Project Gutenberg(?: EBook| License)?|www\.gutenberg\.org(?:/license)?",
    re.IGNORECASE,
)
_METADATA_LINE_RE = re.compile(
    r"^(?:Title|Author|Translator|Editor|Release Date|Posting Date|Language|"
    r"Character set encoding|EBook #|EBook No\.|Credits|Updated)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
_TOC_HEADER_RE = re.compile(
    r"^(?:Table of Contents|TABLE OF CONTENTS|Contents|CONTENTS(?:\s+OF\s+VOL[^\n]*)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ILLUSTRATION_LIST_RE = re.compile(
    r"^(?:LIST OF ILLUSTRATIONS|List of Illustrations|ILLUSTRATIONS)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ASTERISK_DIVIDER_RE = re.compile(r"\*\s+\*\s+\*\s+\*")
_WIKI_EMPH_RE = re.compile(r"=.+?=")
_PRICE_FORMAT_RE = re.compile(
    r"\b(?:post|crown|square|small)?\s*8vo\b|"
    r"\bcloth(?:\s+(?:extra|limp|boards))?\b|"
    r"\billustrated boards\b|"
    r"\b\d+s\.(?:\s*\d+d\.)?",
    re.IGNORECASE,
)
_WORKS_BY_RE = re.compile(r"\bWorks by\b", re.IGNORECASE)
_TOC_ENTRY_RE = re.compile(
    r"^[ \t]*(?:[IVXLC]+\.|CHAPTER\s+[IVXLC\d]+|[A-Z][A-Z \.'\-]{3,60})\s+\d+\s*$",
    re.MULTILINE,
)
_TOC_DOTTED_RE = re.compile(r"\.{3,}\s*\d+\s*$", re.MULTILINE)
_NARRATIVE_LINE_RE = re.compile(
    r"^[a-z].*[a-z][.!?][\"'\)]?\s*$|"
    r"^[A-Z][a-z]+(?:\s+[a-z]+){4,}.*[.!?][\"'\)]?\s*$"
)
_CATALOG_LINE_RE = re.compile(
    r"\b8vo\b|=\s*.+?=\s*|Works by|\*\s+\*\s+\*|\b\d+s\.\s*\d*d\.|"
    r"^[\w\s,\.'\-]{3,50}\s+\d+\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CATALOG_SENTENCE_RE = re.compile(
    r"\b8vo\b|=\s*.+?=\s*|Works by|\*\s+\*\s+\*|\b\d+s\.\s*\d*d\.|"
    r"cloth extra|illustrated boards",
    re.IGNORECASE,
)
# Back-matter: errata tables, underscore-stanza verse appendices
_ERRATA_HEADER_RE = re.compile(
    r"^CORRECTIONS\s*$|page\s+original\s+text\s+correction",
    re.IGNORECASE | re.MULTILINE,
)
_ERRATA_ROW_RE = re.compile(
    r"^\s*\d{1,4}\s+\S.{5,}\s+\S",
    re.MULTILINE,
)
_VERSE_STANZA_RE = re.compile(
    r"_(?:I{1,3}|IV|VI{0,3}|IX|X{0,3}|XI{0,3}|XII{0,3}|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|"
    r"XXI|XXII|XXIII|XXIV|XXV|XXX|XL|L)\._",
    re.IGNORECASE,
)
_THE_END_RE = re.compile(r"\bTHE END\.?\b", re.IGNORECASE)


@dataclass(frozen=True)
class ProseQuality:
    """Classification result for a text chunk."""

    verdict: ProseVerdict
    reason: str | None = None
    narrative_ratio: float = 0.0
    narrative_word_ratio: float = 0.0
    signals: tuple[str, ...] = ()


def _non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _narrative_line_ratio(lines: list[str]) -> float:
    if not lines:
        return 0.0
    narrative = sum(1 for line in lines if _NARRATIVE_LINE_RE.match(line))
    return narrative / len(lines)


def _catalog_line_ratio(lines: list[str]) -> float:
    if not lines:
        return 0.0
    catalog = sum(1 for line in lines if _CATALOG_LINE_RE.search(line))
    return catalog / len(lines)


def _narrative_word_ratio(text: str) -> float:
    """Share of words in sentences that read like narrative rather than catalog."""
    words = text.split()
    if not words:
        return 0.0
    narrative_words = 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        sent_words = sentence.split()
        if len(sent_words) < 8:
            continue
        if _CATALOG_SENTENCE_RE.search(sentence):
            continue
        if re.search(r"[a-z]", sentence):
            narrative_words += len(sent_words)
    return narrative_words / len(words)


def _errata_row_count(text: str) -> int:
    return len(_ERRATA_ROW_RE.findall(text))


def classify_chunk_prose(text: str, *, min_words: int = 30) -> ProseQuality:
    """
    Return whether a chunk is narrative prose suitable for style training.

    Mixed chunks with substantial narrative body are kept even when they
    contain some front-matter markers at the edges.
    """
    text = text.strip()
    words = text.split()
    if len(words) < min_words:
        return ProseQuality("non_prose", "too_short", 0.0, 0.0, ("too_short",))

    lines = _non_empty_lines(text)
    narrative_ratio = _narrative_line_ratio(lines)
    narrative_words = _narrative_word_ratio(text)
    catalog_ratio = _catalog_line_ratio(lines)

    # Structural back matter — reject even when narrative_word_ratio is high.
    errata_rows = _errata_row_count(text)
    if _ERRATA_HEADER_RE.search(text) or errata_rows >= 4:
        return ProseQuality(
            "non_prose",
            "errata_corrections",
            narrative_ratio,
            narrative_words,
            ("errata_table",),
        )

    verse_stanzas = len(_VERSE_STANZA_RE.findall(text))
    if verse_stanzas >= 3:
        return ProseQuality(
            "non_prose",
            "verse_appendix",
            narrative_ratio,
            narrative_words,
            ("verse_stanzas",),
        )

    if _THE_END_RE.search(text) and (errata_rows >= 2 or _ERRATA_HEADER_RE.search(text)):
        return ProseQuality(
            "non_prose",
            "errata_corrections",
            narrative_ratio,
            narrative_words,
            ("the_end_errata",),
        )

    # Mixed chunks with a substantial narrative body are kept for training.
    if narrative_words >= 0.45:
        return ProseQuality("prose", None, narrative_ratio, narrative_words, ())
    signals: list[str] = []

    price_hits = len(_PRICE_FORMAT_RE.findall(text))
    wiki_hits = len(_WIKI_EMPH_RE.findall(text))
    asterisk_divider = bool(_ASTERISK_DIVIDER_RE.search(text))
    works_by = bool(_WORKS_BY_RE.search(text))

    if _TRANSCRIBER_RE.search(text):
        signals.append("transcriber_credits")
    if _GUTENBERG_LICENSE_RE.search(text):
        signals.append("gutenberg_license")
    if len(_METADATA_LINE_RE.findall(text)) >= 3:
        signals.append("metadata_block")
    if _TOC_HEADER_RE.search(text):
        signals.append("table_of_contents")
    if _ILLUSTRATION_LIST_RE.search(text):
        signals.append("illustration_index")
    if asterisk_divider:
        signals.append("asterisk_divider")
    if price_hits:
        signals.append("publisher_pricing")
    if wiki_hits:
        signals.append("wiki_markup")

    toc_entries = len(_TOC_ENTRY_RE.findall(text)) + len(_TOC_DOTTED_RE.findall(text))

    # Publisher catalog — strong, specific to back-matter ads.
    if (
        narrative_words < 0.35
        and (
            (price_hits >= 2 and (wiki_hits >= 1 or works_by or asterisk_divider))
            or (price_hits >= 1 and asterisk_divider and wiki_hits >= 1)
            or (works_by and price_hits >= 1)
            or (catalog_ratio >= 0.45 and price_hits >= 1)
        )
    ):
        return ProseQuality(
            "non_prose",
            "publisher_catalog",
            narrative_ratio,
            narrative_words,
            tuple(signals),
        )

    # Pure transcriber / license front matter with little narrative body.
    if (
        ("transcriber_credits" in signals or "gutenberg_license" in signals)
        and narrative_words < 0.15
        and len(words) < 250
    ):
        reason = "transcriber_credits" if "transcriber_credits" in signals else "gutenberg_license"
        return ProseQuality("non_prose", reason, narrative_ratio, narrative_words, tuple(signals))

    # TOC / illustration lists — mostly structural, not prose.
    if _TOC_HEADER_RE.search(text) and toc_entries >= 4 and narrative_words < 0.2:
        return ProseQuality("non_prose", "table_of_contents", narrative_ratio, narrative_words, tuple(signals))

    if _ILLUSTRATION_LIST_RE.search(text) and narrative_words < 0.2:
        return ProseQuality("non_prose", "illustration_index", narrative_ratio, narrative_words, tuple(signals))

    if toc_entries >= 8 and narrative_words < 0.15:
        return ProseQuality("non_prose", "table_of_contents", narrative_ratio, narrative_words, tuple(signals))

    # All-caps chapter index blocks (no narrative sentences).
    caps_lines = sum(1 for line in lines if line.isupper() and len(line) > 20)
    if caps_lines >= 6 and narrative_words < 0.1 and len(words) < 400:
        return ProseQuality("non_prose", "structural_index", narrative_ratio, narrative_words, tuple(signals))

    # Metadata-only blocks.
    if "metadata_block" in signals and narrative_words < 0.12 and len(words) < 200:
        return ProseQuality("non_prose", "metadata_block", narrative_ratio, narrative_words, tuple(signals))

    return ProseQuality("prose", None, narrative_ratio, narrative_words, tuple(signals))


def is_narrative_prose(text: str, *, min_words: int = 30) -> bool:
    return classify_chunk_prose(text, min_words=min_words).verdict == "prose"


def filter_chunk_record(record: dict, *, min_words: int = 30) -> ProseQuality:
    return classify_chunk_prose(record.get("text") or "", min_words=min_words)


def filter_records(
    records: list[dict],
    *,
    min_words: int = 30,
) -> tuple[list[dict], dict[str, int]]:
    """Drop non-prose chunk records; return kept list and drop-reason counts."""
    from collections import Counter

    kept: list[dict] = []
    dropped: Counter[str] = Counter()
    for record in records:
        quality = filter_chunk_record(record, min_words=min_words)
        if quality.verdict == "prose":
            kept.append(record)
        elif quality.reason:
            dropped[quality.reason] += 1
        else:
            dropped["non_prose"] += 1
    return kept, dict(dropped)

