"""Strip license text, publisher trailers, and nonfiction appendix from fiction chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

_LEADING_SECTION_DIVIDER_RE = re.compile(r"^={3,}\s+")

_NOTE_ABOUT_THIS_BOOK_RE = re.compile(
    r"(?:={3,}\s*)?"
    r"a note about this book,\s+"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+"
    r"\d{1,2},\s+\d{4}\s*:?\s*"
    r"(?:={3,}\s*)?",
    re.IGNORECASE,
)

_CC_LEGAL_START_RE = re.compile(
    r"(?:"
    r"creative\s+commons\s+legal\s+code\b|"
    r"attribution-noncommercial-sharealike\s+3\.0\s+unported\b"
    r")",
    re.IGNORECASE,
)

_CC_LEGAL_END_RE = re.compile(
    r"http://creativecommons\.org/\.\s*",
    re.IGNORECASE,
)

_BACK_MATTER_SECTION_RE = re.compile(
    r"\s*={3,}\s*"
    r"(?:acknowledgements?|about the author|other books by)\s*:?\s*"
    r"={3,}"
    r".*$",
    re.IGNORECASE | re.DOTALL,
)

_PUBLISHER_COPYRIGHT_FOOTER_RE = re.compile(
    r"(?:"
    r"\s*text:\s*copyright\s+\d{4}\b.*$"
    r"|\s*tor books,?\s+"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+"
    r"\d{4}\s+isbn:.*$"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_FICTION_DISCLAIMER_RE = re.compile(
    r"this (?:novel|book|story) is (?:entirely )?a work of fiction\b",
    re.IGNORECASE,
)

_NARRATIVE_AFTER_DISCLAIMER_RE = re.compile(
    r"\b(?:chapter (?:one|1|i)\b|prologue\b|part (?:one|1|i)\b)",
    re.IGNORECASE,
)

# Nonfiction appendix blocks appended after the novel (afterwords, reading lists, acks).
_NONFICTION_APPENDIX_SUFFIX_RES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.\s+afterword by\b", re.IGNORECASE), "afterword"),
    (re.compile(r"\.\s+foreword by\b", re.IGNORECASE), "foreword"),
    (re.compile(r"\bbibliography\s+no writer creates\b", re.IGNORECASE), "bibliography"),
    (
        re.compile(
            r"the end\.\s*(?:\*\s*)+biography and bibliography\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "biography_bibliography",
    ),
    (re.compile(r"\backnowledgments\s+this book owes\b", re.IGNORECASE), "acknowledgments"),
]


@dataclass(frozen=True)
class StripResult:
    text: str
    stripped: bool
    reason: str | None = None


def _normalize_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r" +", " ", text).strip()


def _word_count(text: str) -> int:
    return len(text.split())


def _strip_fiction_disclaimer_block(text: str) -> tuple[str, bool, str | None]:
    """Remove legal disclaimers and author bios when no chapter prose follows."""
    match = _FICTION_DISCLAIMER_RE.search(text)
    if match is None:
        return text, False, None
    narrative = _NARRATIVE_AFTER_DISCLAIMER_RE.search(text, match.end())
    if narrative is not None:
        return text[narrative.start() :].strip(), True, "fiction_disclaimer_block"
    prefix = text[: match.start()].strip()
    if _word_count(prefix) >= 100:
        return prefix, True, "fiction_disclaimer_block"
    return "", True, "fiction_disclaimer_only"


def _strip_nonfiction_appendix_suffix(text: str) -> tuple[str, bool, str | None]:
    """Remove afterwords, bibliographies, and similar appendix material at chunk tail."""
    best: tuple[int, str] | None = None
    for pattern, reason in _NONFICTION_APPENDIX_SUFFIX_RES:
        match = pattern.search(text)
        if match is None:
            continue
        if best is None or match.start() < best[0]:
            best = (match.start(), reason)
    if best is None:
        return text, False, None
    cut, reason = best
    return text[:cut].strip(), True, reason


def _strip_cc_legal_blocks(text: str) -> tuple[str, bool]:
    """Remove embedded Creative Commons legal-code sections."""
    stripped = False
    while True:
        match = _CC_LEGAL_START_RE.search(text)
        if not match:
            break
        start = match.start()
        # Include a short "creative commons" label immediately before legal code.
        prefix = text[max(0, start - 40) : start].lower()
        if "creative commons" in prefix:
            label = prefix.rfind("creative commons")
            if label >= 0:
                start = max(0, start - 40) + label
        end_match = _CC_LEGAL_END_RE.search(text, match.end())
        if end_match:
            end = end_match.end()
        else:
            end = len(text)
        text = (text[:start] + text[end:]).strip()
        stripped = True
    return text, stripped


def strip_license_agreement(text: str) -> StripResult:
    """Remove license legalese, publisher trailers, and nonfiction appendix blocks."""
    text = text.strip()
    if not text:
        return StripResult("", False)

    stripped = False
    reason: str | None = None

    text, disclaimer_stripped, disclaimer_reason = _strip_fiction_disclaimer_block(text)
    if disclaimer_stripped:
        stripped = True
        reason = disclaimer_reason
    if not text:
        return StripResult("", stripped, reason)

    cleaned = _LEADING_SECTION_DIVIDER_RE.sub("", text, count=1)
    if cleaned != text:
        text = cleaned
        stripped = True
        reason = "section_divider"

    text, appendix_stripped, appendix_reason = _strip_nonfiction_appendix_suffix(text)
    if appendix_stripped:
        stripped = True
        reason = appendix_reason

    note = _NOTE_ABOUT_THIS_BOOK_RE.search(text)
    if note:
        text = text[: note.start()].strip()
        stripped = True
        reason = "note_about_this_book"

    text, cc_stripped = _strip_cc_legal_blocks(text)
    if cc_stripped:
        stripped = True
        reason = "cc_legal_code"

    back_matter = _BACK_MATTER_SECTION_RE.search(text)
    if back_matter:
        text = text[: back_matter.start()].strip()
        stripped = True
        reason = "back_matter_section"

    footer = _PUBLISHER_COPYRIGHT_FOOTER_RE.search(text)
    if footer:
        text = text[: footer.start()].strip()
        stripped = True
        reason = "publisher_copyright_footer"

    text = _normalize_ws(text)
    return StripResult(text, stripped, reason)
