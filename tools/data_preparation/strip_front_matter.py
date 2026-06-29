"""Strip Gutenberg front matter from chunk text (per-chunk pass)."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Lone Roman numeral as its own paragraph: ``\n\nI\n\n``, ``\n\nIX\n\n``, etc.
_LONE_ROMAN_PARA_RE = re.compile(r"^[IVXLC]{1,6}\.?$")
_ROMAN_BETWEEN_PARAS_RE = re.compile(
    r"(?:^|\n\n)([IVXLC]{1,6})\.?\s*\n\n(?=[A-Za-z])",
    re.MULTILINE,
)

_REVIEW_BLURB_RE = re.compile(
    r"\b(?:POST|GAZETTE|HERALD|TIMES|CHRONICLE|OBSERVER)\s+says\s*:",
    re.IGNORECASE,
)
_PUBLISHER_CATALOG_RE = re.compile(
    r"\b(?:Crown|Post|Small)\s+8vo\b|Mills .{0,3} Boon|illustrated boards|New Novels",
    re.IGNORECASE,
)
_TITLE_PAGE_RE = re.compile(
    r"\b(?:SECOND|THIRD|FIRST)\s+EDITION\b|"
    r"\bCopyright\b|_Published\b|ALL RIGHTS RESERVED|"
    r"\bUNITED STATES OF AMERICA\b|"
    r"\bAUTHOR OF\b|^\s*BY\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SECTION_HEADER_RE = re.compile(
    r"^(?:PREFACE|INTRODUCTION|FOREWORD|ADVERTISEMENT|"
    r"LIST OF ILLUSTRATIONS|ILLUSTRATIONS|CONTENTS|"
    r"TABLE OF CONTENTS)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CONTENTS_INDEX_RE = re.compile(
    r"^\s*PAGE\s*$|"
    r"^[A-Z][A-Z \.'\-,&]{2,60}\s+\d{1,4}\s*$|"
    r"\.{2,}\s*\d{1,4}\s*$",
    re.MULTILINE,
)
_CHAPTER_HEADING_RE = re.compile(
    r"^CHAPTER\s+(?:"
    r"I{1,3}|IV|VI{0,3}|IX|X{0,3}|XI{0,3}|XII{0,3}|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|"
    r"\d{1,3}|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE"
    r")(?:\b|[\.\s:\-—])",
    re.IGNORECASE | re.MULTILINE,
)
_PART_BOOK_RE = re.compile(
    r"^(?:PART|BOOK|SECTION|PROLOGUE|EPILOGUE)\s+"
    r"(?:I{1,3}|IV|VI{0,3}|IX|X{0,3}|XI{0,3}|XII{0,3}|\d+|[A-Z][A-Z \-]{0,30})\s*$",
    re.IGNORECASE,
)
_ROMAN_NUMERALS = frozenset({
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC", "C",
})
_NARRATIVE_SENTENCE_RE = re.compile(
    r"^[A-Za-z].*[a-z].*[.!?][\"'\)]?\s*$|"
    r"^[A-Za-z][a-z]+(?:\s+[a-z]+){3,}.*[a-z]"
)


@dataclass(frozen=True)
class StripResult:
    text: str
    stripped: bool
    start_paragraph: int = 0
    reason: str | None = None


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _is_valid_roman(token: str) -> bool:
    return token.upper() in _ROMAN_NUMERALS


def _is_lone_roman_chapter(para: str) -> bool:
    """Paragraph that is only a Roman numeral chapter marker (``I``, ``IX.``, etc.)."""
    line = para.strip()
    if not line or "\n" in line:
        # Multi-line paragraphs are never lone Roman markers.
        if "\n" in line:
            lines = [ln.strip() for ln in line.split("\n") if ln.strip()]
            if len(lines) == 1:
                line = lines[0]
            else:
                return False
    if not _LONE_ROMAN_PARA_RE.match(line):
        return False
    return _is_valid_roman(line.rstrip("."))


def find_lone_roman_chapter_splits(text: str) -> list[str]:
    """Return Roman tokens found in ``\\n\\nROMAN\\n\\n`` chapter-boundary positions."""
    return [m.group(1).upper() for m in _ROMAN_BETWEEN_PARAS_RE.finditer(text) if _is_valid_roman(m.group(1))]


def _is_title_heading(para: str) -> bool:
    line = para.strip()
    if not line or len(line.split()) > 12:
        return False
    if _NARRATIVE_SENTENCE_RE.match(line):
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    return upper_ratio >= 0.85


def _is_front_matter_paragraph(para: str) -> bool:
    if _REVIEW_BLURB_RE.search(para):
        return True
    if _PUBLISHER_CATALOG_RE.search(para):
        return True
    if _TITLE_PAGE_RE.search(para):
        return True
    if _SECTION_HEADER_RE.match(para.strip()):
        return True
    if _CONTENTS_INDEX_RE.search(para):
        return True
    if re.search(r"Produced by|Distributed Proofreading|Project Gutenberg", para, re.I):
        return True
    return False


def _is_contents_block_paragraph(para: str) -> bool:
    lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
    if not lines:
        return False
    if lines[0].upper() in ("CONTENTS", "TABLE OF CONTENTS"):
        return True
    if any(ln.upper() == "PAGE" for ln in lines[:3]):
        return True
    index_lines = sum(
        1 for ln in lines
        if re.search(r"\s+\d{1,4}\s*$", ln) or re.search(r"\.{2,}\s*\d{1,4}\s*$", ln)
    )
    return index_lines >= 3 and index_lines / len(lines) >= 0.5


def _is_narrative_paragraph(para: str) -> bool:
    words = para.split()
    if len(words) < 8:
        return False
    if _is_front_matter_paragraph(para):
        return False
    if _PUBLISHER_CATALOG_RE.search(para):
        return False
    lower = sum(1 for c in para if c.islower())
    if lower < 12:
        return False
    if _NARRATIVE_SENTENCE_RE.search(para):
        return True
    return bool(re.search(r"[a-z]{4,}", para))


def _is_body_start_marker(para: str, *, next_para: str | None) -> bool:
    if _CHAPTER_HEADING_RE.match(para.strip()):
        return True
    if _PART_BOOK_RE.match(para.strip()):
        return True
    if _is_lone_roman_chapter(para):
        return next_para is not None and _is_narrative_paragraph(next_para)
    return False


def _find_roman_chapter_body_start(paragraphs: list[str]) -> int | None:
    """
  Find narrative start after ``\\n\\nROMAN\\n\\n`` chapter markers.

  Returns paragraph index of first narrative paragraph following a lone
  Roman numeral, or None if not found.
  """
    for i, para in enumerate(paragraphs):
        if not _is_lone_roman_chapter(para):
            continue
        if i + 1 < len(paragraphs) and _is_narrative_paragraph(paragraphs[i + 1]):
            return i + 1
    return None


def _is_contents_header(para: str) -> bool:
    stripped = para.strip()
    upper = stripped.upper()
    if upper in ("CONTENTS", "TABLE OF CONTENTS"):
        return True
    if re.search(r"\bCONTENTS\b", upper) and len(stripped.split()) <= 8:
        return True
    return False


def find_body_start(paragraphs: list[str]) -> int:
    """Return paragraph index where narrative body should begin."""
    if not paragraphs:
        return 0

    if (
        _is_narrative_paragraph(paragraphs[0])
        and not _is_front_matter_paragraph(paragraphs[0])
        and not _is_title_heading(paragraphs[0])
    ):
        return 0

    roman_start = _find_roman_chapter_body_start(paragraphs)
    if roman_start is not None and roman_start > 0:
        return roman_start

    mode = "scan"
    front_seen = False

    for i, para in enumerate(paragraphs):
        stripped = para.strip()
        upper = stripped.upper()

        if mode == "preface":
            if _is_contents_header(para):
                mode = "contents"
            elif _is_lone_roman_chapter(para) or _CHAPTER_HEADING_RE.match(stripped):
                mode = "scan"
            else:
                front_seen = True
                continue

        if mode == "contents":
            front_seen = True
            next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else None
            if _is_title_heading(para) and next_para and _is_narrative_paragraph(next_para):
                return i + 1
            if _is_narrative_paragraph(para):
                return i
            continue

        if re.match(r"^PREFACE\b", stripped, re.I):
            mode = "preface"
            front_seen = True
            continue
        if _is_contents_header(para):
            mode = "contents"
            front_seen = True
            continue

        if _is_title_heading(para):
            front_seen = True
            next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else None
            if (
                next_para
                and _is_narrative_paragraph(next_para)
                and mode == "scan"
                and not any(_is_front_matter_paragraph(p) for p in paragraphs[:i])
            ):
                return i + 1
            continue

        if _is_front_matter_paragraph(para) or _is_contents_block_paragraph(para):
            front_seen = True
            continue

        next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else None

        if _is_body_start_marker(para, next_para=next_para):
            if next_para and _is_narrative_paragraph(next_para):
                return i + 1
            if _is_narrative_paragraph(para):
                return i
            continue

        if front_seen and _is_title_heading(para) and next_para and _is_narrative_paragraph(next_para):
            return i + 1

        if _is_narrative_paragraph(para) and (front_seen or i > 0):
            return i

    if roman_start is not None:
        return roman_start

    return 0


def strip_front_matter(text: str, *, min_words: int = 30) -> StripResult:
    """
    Remove leading front matter from a chunk; keep paragraph breaks.

    Detects ``\\n\\nI\\n\\n``, ``\\n\\nIX\\n\\n``, and similar lone-Roman chapter
    markers, plus publisher blurbs, prefaces, and contents indexes.
    """
    text = text.strip()
    if not text:
        return StripResult("", False)

    paragraphs = _paragraphs(text)
    if len(paragraphs) <= 1:
        return StripResult(text, False)

    start = find_body_start(paragraphs)
    if start <= 0:
        return StripResult(text, False)

    stripped = "\n\n".join(paragraphs[start:]).strip()
    if len(stripped.split()) < min_words:
        return StripResult(text, False, reason="too_short_after_strip")

    reason = "front_matter"
    if start > 0 and _is_lone_roman_chapter(paragraphs[start - 1]):
        reason = "roman_chapter_marker"

    return StripResult(stripped, True, start_paragraph=start, reason=reason)


def strip_record(record: dict, *, min_words: int = 30) -> dict:
    out = dict(record)
    result = strip_front_matter(record.get("text") or "", min_words=min_words)
    out["text"] = result.text
    metadata = dict(out.get("metadata") or {})
    metadata["word_count"] = len(result.text.split())
    if result.stripped:
        metadata["front_matter_stripped"] = True
        metadata["front_matter_start_paragraph"] = result.start_paragraph
        if result.reason:
            metadata["front_matter_strip_reason"] = result.reason
    out["metadata"] = metadata
    return out
