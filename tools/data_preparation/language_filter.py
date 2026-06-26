"""Language detection helpers for corpus cleaning."""

from __future__ import annotations

import re
from typing import Literal

# Scripts outside Latin extended range used in English prose.
_NON_LATIN_SCRIPT = re.compile(
    r"[\u0600-\u06FF"   # Arabic
    r"\u0900-\u097F"    # Devanagari
    r"\u0400-\u04FF"    # Cyrillic
    r"\u4E00-\u9FFF"    # CJK
    r"\u0E00-\u0E7F"    # Thai
    r"\u0590-\u05FF"    # Hebrew
    r"]"
)

_COMMON_EN = re.compile(
    r"\b(the|and|to|of|a|in|that|it|was|for|on|is|with|as|he|she|you|her|his|not|but|"
    r"they|had|at|be|my|this|have|from|or|one|by|love|said|what|when|we|were|been|would)\b",
    re.IGNORECASE,
)


def has_non_latin_script(text: str, *, sample_chars: int = 3000) -> bool:
    return bool(_NON_LATIN_SCRIPT.search(text[:sample_chars]))


def english_word_ratio(text: str, *, max_words: int = 500) -> float:
    words = text.lower().split()[:max_words]
    if not words:
        return 0.0
    hits = sum(1 for w in words if _COMMON_EN.match(w))
    return hits / len(words)


def detect_language(text: str, *, sample_chars: int = 4000) -> str | None:
    """Return ISO 639-1 code, or None if detection fails."""
    sample = text[:sample_chars].strip()
    if len(sample.split()) < 20:
        return None
    try:
        from langdetect import LangDetectException, detect
    except ImportError:
        return None
    try:
        return detect(sample)
    except LangDetectException:
        return None


def classify_language(text: str, *, sample_chars: int = 4000) -> Literal["en", "non_en", "unknown"]:
    if has_non_latin_script(text, sample_chars=sample_chars):
        return "non_en"
    lang = detect_language(text, sample_chars=sample_chars)
    if lang == "en":
        return "en"
    if lang is not None:
        return "non_en"
    # Heuristic fallback when langdetect unavailable or inconclusive
    if english_word_ratio(text) >= 0.04:
        return "en"
    return "unknown"


def is_english_text(text: str, *, sample_chars: int = 4000) -> bool:
    """True when text appears to be English prose suitable for the style pipeline."""
    return classify_language(text, sample_chars=sample_chars) == "en"
