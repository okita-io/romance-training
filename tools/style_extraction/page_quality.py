"""Quality checks for vision-transcribed PDF page markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Strip model chain-of-thought wrappers (angle-bracket think blocks).
_THINK_BLOCK_RE = re.compile(
    r"<" + "think" + r">[\s\S]*?</" + "think" + r">",
    re.IGNORECASE,
)
_REASONING_HEADER_RE = re.compile(
    r"^(?:#+\s*)?(?:reasoning|analysis|thought process)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_META_PHRASES = (
    "i need to transcribe",
    "let's start by",
    "lets start by",
    "so we need to",
    "in markdown",
    "the instruction says",
    "wait, but",
    "let me think",
    "let's check",
    "lets check",
    "let's verify",
    "following the given rules",
    "output only markdown",
    "return only",
)

# Lines that look like planning steps, not book prose.
_PLANNING_LINE_RE = re.compile(
    r"^(?:first|next|then|now|so|also|wait|okay|ok)\s*,?\s+(?:the|we|i|let's|lets)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PageQualityReport:
    ok: bool
    issues: tuple[str, ...]
    word_count: int
    unique_word_ratio: float

    def summary(self) -> str:
        if self.ok:
            return f"ok ({self.word_count} words, unique ratio {self.unique_word_ratio:.2f})"
        return "; ".join(self.issues)


def clean_model_output(text: str) -> str:
    """Remove thinking/reasoning wrappers; keep transcribed markdown."""
    text = text.strip()
    text = _THINK_BLOCK_RE.sub("", text).strip()

    # Drop leading reasoning preamble before first markdown heading or blockquote.
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("#", ">", "|", "-", "*", "```")):
            start = i
            break
        if _REASONING_HEADER_RE.match(stripped):
            start = i + 1
            continue
        lower = stripped.lower()
        if any(p in lower for p in _META_PHRASES):
            continue
        if _PLANNING_LINE_RE.match(stripped):
            continue
        # First substantive non-meta line — keep from here if it looks like content.
        if len(stripped.split()) >= 4 or stripped.startswith(("#", ">")):
            start = i
            break

    cleaned = "\n".join(lines[start:]).strip()
    return cleaned or text


def _word_stats(text: str) -> tuple[list[str], int, float]:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return words, 0, 0.0
    return words, len(words), len(set(words)) / len(words)


def _dominant_token_ratio(words: list[str]) -> tuple[str, float]:
    if not words:
        return "", 0.0
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    token, count = max(counts.items(), key=lambda kv: kv[1])
    return token, count / len(words)


def _max_repeated_phrase_ratio(text: str, n: int = 3) -> float:
    """Share of text covered by the most repeated n-word phrase (rough heuristic)."""
    words = re.findall(r"\S+", text)
    if len(words) < n * 4:
        return 0.0
    counts: dict[str, int] = {}
    for i in range(len(words) - n + 1):
        phrase = " ".join(words[i : i + n])
        counts[phrase] = counts.get(phrase, 0) + 1
    if not counts:
        return 0.0
    phrase, count = max(counts.items(), key=lambda kv: kv[1])
    if count < 4:
        return 0.0
    return (len(phrase) * count) / max(1, len(text))


def validate_page_markdown(
    text: str,
    *,
    min_words: int = 8,
    min_unique_ratio: float = 0.12,
    max_dominant_token_ratio: float = 0.35,
    max_meta_hits: int = 2,
) -> PageQualityReport:
    """
    Return whether transcribed page markdown looks like usable book content.

    Fails pages that are too short, highly repetitive, or mostly model reasoning.
    """
    cleaned = clean_model_output(text)
    issues: list[str] = []

    words, word_count, unique_ratio = _word_stats(cleaned)
    if word_count < min_words:
        issues.append(f"too short ({word_count} words, need ≥{min_words})")

    if word_count >= min_words and unique_ratio < min_unique_ratio:
        issues.append(f"low vocabulary variety (unique ratio {unique_ratio:.2f})")

    dom_token, dom_ratio = _dominant_token_ratio(words)
    if word_count >= 10 and dom_ratio > max_dominant_token_ratio:
        issues.append(f"repetitive token '{dom_token}' ({dom_ratio:.0%} of words)")

    phrase_ratio = _max_repeated_phrase_ratio(cleaned)
    if phrase_ratio > 0.25:
        issues.append(f"repeated phrase spam ({phrase_ratio:.0%} of text)")

    lower = cleaned.lower()
    meta_hits = sum(1 for p in _META_PHRASES if p in lower)
    planning_lines = sum(1 for line in cleaned.splitlines() if _PLANNING_LINE_RE.match(line.strip()))
    if meta_hits > max_meta_hits or planning_lines >= 4:
        issues.append("looks like model reasoning, not transcription")

    # Numeric spam: "19, 19, 19" style degeneration
    if re.search(r"(?:\b\d+\b[\s,]){12,}", cleaned):
        issues.append("numeric repetition spam")

    return PageQualityReport(
        ok=not issues,
        issues=tuple(issues),
        word_count=word_count,
        unique_word_ratio=round(unique_ratio, 4),
    )


def audit_pages_dir(pages_dir: Path) -> list[dict[str, Any]]:
    """Audit all page_*.md files; return failing page records sorted by page number."""
    failures: list[dict[str, Any]] = []
    for md_path in sorted(pages_dir.glob("page_*.md")):
        page_num = int(md_path.stem.split("_")[1])
        body = md_path.read_text(encoding="utf-8", errors="replace")
        report = validate_page_markdown(body)
        if not report.ok:
            failures.append({
                "page": page_num,
                "path": str(md_path),
                "issues": list(report.issues),
                "word_count": report.word_count,
            })
    return failures
