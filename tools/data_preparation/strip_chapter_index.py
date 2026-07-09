"""Strip inline Gutenberg table-of-contents chapter index runs from prose."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Longest-first so ``CXI`` does not match as ``C`` + ``XI``.
_ROMAN_NUMERALS = (
    "CXXXVI", "CXXXV", "CXXXIV", "CXXXIII", "CXXXII", "CXXXI", "CXXX",
    "CXXIX", "CXXVIII", "CXXVII", "CXXVI", "CXXV", "CXXIV", "CXXIII",
    "CXXII", "CXXI", "CXX", "CXIX", "CXVIII", "CXVII", "CXVI", "CXV",
    "CXIV", "CXIII", "CXII", "CXI", "CX", "CIX", "CVIII", "CVII", "CVI",
    "CV", "CIV", "CIII", "CII", "CI", "C", "XCIX", "XCVIII", "XCVII",
    "XCVI", "XCV", "XCIV", "XCIII", "XCII", "XCI", "XC", "LXXXIX",
    "LXXXVIII", "LXXXVII", "LXXXVI", "LXXXV", "LXXXIV", "LXXXIII", "LXXXII",
    "LXXXI", "LXXX", "LXXIX", "LXXVIII", "LXXVII", "LXXVI", "LXXV", "LXXIV",
    "LXXIII", "LXXII", "LXXI", "LXX", "LXIX", "LXVIII", "LXVII", "LXVI",
    "LXV", "LXIV", "LXIII", "LXII", "LXI", "LX", "LIX", "LVIII", "LVII",
    "LVI", "LV", "LIV", "LIII", "LII", "LI", "L", "XLIX", "XLVIII", "XLVII",
    "XLVI", "XLV", "XLIV", "XLIII", "XLII", "XLI", "XL", "XXXIX", "XXXVIII",
    "XXXVII", "XXXVI", "XXXV", "XXXIV", "XXXIII", "XXXII", "XXXI", "XXX",
    "XXIX", "XXVIII", "XXVII", "XXVI", "XXV", "XXIV", "XXIII", "XXII",
    "XXI", "XX", "XIX", "XVIII", "XVII", "XVI", "XV", "XIV", "XIII",
    "XII", "XI", "X", "IX", "VIII", "VII", "VI", "V", "IV", "III", "II", "I",
)

_CHAPTER_NUMERAL_RE = rf"(?:{'|'.join(_ROMAN_NUMERALS)}|\d{{1,3}})"
_CHAPTER_MARKER_RE = re.compile(rf"CHAPTER\s+{_CHAPTER_NUMERAL_RE}\b", re.IGNORECASE)
_CHAPTER_INDEX_ENTRY_RE = re.compile(
    rf"CHAPTER\s+{_CHAPTER_NUMERAL_RE}\b"
    rf"\s*(?:[\.—–—\-:]+\s*|\s+)"
    rf"(.+?)"
    rf"(?=\s+CHAPTER\s+{_CHAPTER_NUMERAL_RE}\b|\s+EPILOGUE\b|\s+ETYMOLOGY\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_TOC_TAIL_RE = re.compile(r"\s*(?:EPILOGUE\.?\s*ETYMOLOGY\.?|ETYMOLOGY\.?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class StripResult:
    text: str
    stripped: bool


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _clean_toc_paragraph(para: str) -> str:
    first = _CHAPTER_MARKER_RE.search(para)
    if first is None:
        return para

    body = para[first.start():]
    cleaned = _CHAPTER_INDEX_ENTRY_RE.sub("", body)
    cleaned = _TOC_TAIL_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned


def strip_inline_chapter_index(text: str) -> StripResult:
    """
    Remove inline TOC runs like ``CHAPTER I.—Loomings CHAPTER II.—The Carpet Bag``.

    Standalone chapter headings before narrative (``CHAPTER X. Title\\n\\nShe took…``)
    are kept.
    """
    text = text.strip()
    if not text:
        return StripResult("", False)

    paragraphs = _paragraphs(text)
    if not paragraphs:
        return StripResult(text, False)

    changed = False
    cleaned_paragraphs: list[str] = []
    for para in paragraphs:
        markers = list(_CHAPTER_MARKER_RE.finditer(para))
        if len(markers) >= 2:
            new_para = _clean_toc_paragraph(para)
            changed = True
            if new_para:
                cleaned_paragraphs.append(new_para)
            continue
        cleaned_paragraphs.append(para)

    if not changed:
        return StripResult(text, False)

    return StripResult("\n\n".join(cleaned_paragraphs).strip(), True)
