"""Heuristic quality scoring for web-fiction / Literotica-style story rows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from tools.data_preparation.language_filter import (
    classify_language,
    english_word_ratio,
    has_non_latin_script,
)
from tools.data_preparation.prose_filter import classify_chunk_prose

FictionTier = Literal["keep", "review", "drop"]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&(?:nbsp|amp|lt|gt|quot|#39);")
_DISCLAIMER_RE = re.compile(
    r"\b(?:all characters (?:are|were) (?:18|of legal age)|"
    r"this story contains fictional depictions|"
    r"do not read if (?:offended|underage)|"
    r"standard disclaimer)\b",
    re.IGNORECASE,
)
_FORUM_BOILERPLATE_RE = re.compile(
    r"\b(?:please rate and review|comments appreciated|"
    r"follow me on|patreon\.com|onlyfans)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FictionQuality:
    tier: FictionTier
    score: float
    reasons: tuple[str, ...]
    word_count: int
    unique_word_ratio: float
    top_bigram_ratio: float
    avg_sentence_words: float
    html_char_ratio: float
    language: str


def strip_html(text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HTML_ENTITY_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _html_char_ratio(raw: str) -> float:
    if not raw:
        return 0.0
    html_chars = sum(len(m.group(0)) for m in _HTML_TAG_RE.finditer(raw))
    html_chars += sum(len(m.group(0)) for m in _HTML_ENTITY_RE.finditer(raw))
    return html_chars / len(raw)


def _top_bigram_ratio(words: list[str]) -> float:
    if len(words) < 50:
        return 0.0
    counts: dict[str, int] = {}
    for i in range(len(words) - 1):
        bg = f"{words[i].lower()} {words[i+1].lower()}"
        counts[bg] = counts.get(bg, 0) + 1
    return max(counts.values()) / (len(words) - 1)


def _avg_sentence_words(text: str) -> float:
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.split()) >= 3]
    if not sents:
        return 0.0
    return sum(len(s.split()) for s in sents) / len(sents)


def _detect_language(cleaned: str, *, fast: bool) -> str:
    if fast and not has_non_latin_script(cleaned) and english_word_ratio(cleaned) >= 0.06:
        return "en"
    return classify_language(cleaned)


def classify_fiction_quality(
    text: str,
    *,
    min_words_keep: int = 500,
    min_words_drop: int = 200,
    min_unique_ratio: float = 0.30,
    max_bigram_ratio: float = 0.07,
    min_alpha_ratio: float = 0.78,
    fast_language: bool = False,
) -> FictionQuality:
    """
    Score a full story row for style-training suitability.

    Tiers:
      keep   — English narrative with enough length and lexical variety
      review — borderline repetition/vocabulary; human spot-check candidate
      drop   — too short, non-English, garbled, or hard non-prose
    """
    raw = text.strip()
    cleaned = strip_html(raw)
    words = cleaned.split()
    word_count = len(words)
    unique_ratio = len({w.lower() for w in words}) / max(word_count, 1)
    bigram_ratio = _top_bigram_ratio(words)
    avg_sent = _avg_sentence_words(cleaned)
    html_ratio = _html_char_ratio(raw)
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in cleaned) / max(len(cleaned), 1)

    lang = _detect_language(cleaned, fast=fast_language)
    prose = classify_chunk_prose(cleaned, min_words=min_words_drop)

    reasons: list[str] = []
    score = 1.0

    if lang != "en":
        reasons.append("non_english")
        score -= 0.5
    if prose.verdict == "non_prose" and prose.reason:
        reasons.append(prose.reason)
        score -= 0.4
    if word_count < min_words_drop:
        reasons.append("too_short")
        score -= 0.5
    elif word_count < min_words_keep:
        reasons.append("short")
        score -= 0.15
    if alpha_ratio < min_alpha_ratio:
        reasons.append("garbled")
        score -= 0.35
    if unique_ratio < min_unique_ratio:
        reasons.append("low_vocab")
        score -= 0.25
    if bigram_ratio > max_bigram_ratio:
        reasons.append("repetitive")
        score -= 0.25
    if html_ratio > 0.12:
        reasons.append("html_heavy")
        score -= 0.1
    if _DISCLAIMER_RE.search(raw[:500]) and word_count < min_words_keep:
        reasons.append("disclaimer_only")
        score -= 0.2
    if _FORUM_BOILERPLATE_RE.search(raw) and word_count < 800:
        reasons.append("forum_boilerplate")
        score -= 0.15
    if avg_sent < 4.0 and word_count > 300:
        reasons.append("fragmentary")
        score -= 0.15

    hard_drop = {
        "non_english",
        "too_short",
        "garbled",
        "errata_corrections",
        "verse_appendix",
        "publisher_catalog",
        "disclaimer_only",
    }
    if any(r in hard_drop for r in reasons):
        tier: FictionTier = "drop"
    elif any(r in ("low_vocab", "repetitive", "fragmentary", "html_heavy", "short") for r in reasons):
        tier = "review" if score >= 0.40 else "drop"
    elif score >= 0.75:
        tier = "keep"
    elif score >= 0.45:
        tier = "review"
    else:
        tier = "drop"

    return FictionQuality(
        tier=tier,
        score=round(max(0.0, min(1.0, score)), 3),
        reasons=tuple(reasons),
        word_count=word_count,
        unique_word_ratio=round(unique_ratio, 4),
        top_bigram_ratio=round(bigram_ratio, 4),
        avg_sentence_words=round(avg_sent, 2),
        html_char_ratio=round(html_ratio, 4),
        language=lang,
    )
