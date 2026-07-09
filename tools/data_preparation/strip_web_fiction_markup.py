"""Strip Literotica-style HTML markup and leading author notes from web fiction."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

_HTML_TAG_RE = re.compile(r"<[^>]*>", re.DOTALL)
_HTML_ENTITY_RE = re.compile(r"&(?:#x?[0-9a-f]+|[a-z]+);", re.IGNORECASE)
_LINE_BREAK_TAGS = frozenset({"br", "br/", "p", "/p", "hr", "/hr"})

_AUTHOR_NOTE_MARKER_RE = re.compile(
    r"(?i)(?:author'?s?\s+note|a/?n)\s*:?"
)

_META_BRACKET_PHRASES_RE = re.compile(
    r"(?i)\b(?:"
    r"next installment|"
    r"this story is fiction|"
    r"inspired by (?:an )?actual|"
    r"far from what I usually write|"
    r"trigger(?:ing)?|"
    r"blackmail(?:ing|er)?|"
    r"standard disclaimer|"
    r"all characters (?:are|were)|"
    r"no original work was stolen|"
    r"story was inspired"
    r")\b"
)

_LEADING_META_PREAMBLE_RE = re.compile(
    r"(?is)^\s*"
    r"(?:"
    r"story was inspired\b|"
    r"this (?:story|tale) (?:is|was)\b|"
    r"inspired by (?:already )?existing\b|"
    r"no original work was stolen\b"
    r")"
    r".*?"
    r"\n\s*"
    r"(?:_{5,}|\*{3,}|={5,})"
    r"\s*\n+"
)

_SECTION_DIVIDER_LINE_RE = re.compile(r"^\s*(?:\*{3,}|_{5,}|={5,})\s*$", re.MULTILINE)


@dataclass(frozen=True)
class StripResult:
    text: str
    stripped: bool
    reason: str | None = None


def _normalize_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r" +", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _tag_name(raw: str) -> str:
    inner = raw.strip("<>").strip()
    if not inner:
        return ""
    return inner.split(None, 1)[0].lower().rstrip("/")


def _decode_html_entities(text: str) -> str:
    def replace_entity(match: re.Match[str]) -> str:
        token = match.group(0)
        lowered = token.lower()
        if lowered in {"&nobreak;", "&hairsp;", "&nbsp;"}:
            return " "
        if lowered == "&hellip;":
            return "..."
        return html.unescape(token)

    return _HTML_ENTITY_RE.sub(replace_entity, text)


def strip_html_markup(text: str) -> str:
    """Remove HTML tags and entities while preserving paragraph breaks."""

    def replace_tag(match: re.Match[str]) -> str:
        name = _tag_name(match.group(0))
        if name in _LINE_BREAK_TAGS:
            return "\n"
        return " "

    text = _HTML_TAG_RE.sub(replace_tag, text)
    text = _decode_html_entities(text)
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    text = re.sub(r'(["\'(\[])\s+', r"\1", text)
    lines = [re.sub(r" +", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _strip_trailing_section_dividers(text: str) -> str:
    text = text.lstrip()
    while True:
        match = _SECTION_DIVIDER_LINE_RE.match(text)
        if not match:
            break
        text = text[match.end() :].lstrip("\n")
    return text


def _strip_html_wrapped_leading_block(text: str) -> tuple[str, bool]:
    """Remove a leading HTML block when it looks like an author/meta note."""
    stripped = text.lstrip()
    if not stripped.startswith("<"):
        return text, False

    open_match = re.match(r"(?is)^<([^>]+)>(.*)$", stripped)
    if not open_match:
        return text, False

    tag_name = open_match.group(1).split()[0]
    body = open_match.group(2)
    close = f"</{tag_name}>"
    close_at = body.lower().find(close.lower())
    if close_at < 0:
        return text, False

    block = body[:close_at]
    remainder = body[close_at + len(close) :].lstrip()
    block_stripped = block.strip()
    is_author_note = bool(_AUTHOR_NOTE_MARKER_RE.search(block_stripped))
    is_meta_bracket = block_stripped.startswith("[") and bool(
        _META_BRACKET_PHRASES_RE.search(block_stripped)
    )
    if not (is_author_note or is_meta_bracket):
        return text, False

    remainder = _strip_trailing_section_dividers(remainder)
    return remainder.lstrip("\n"), True


def _strip_plain_author_note(text: str) -> tuple[str, bool]:
    """Remove a leading author-note paragraph, with or without square brackets."""
    stripped = text.lstrip()
    match = re.match(
        r"(?is)^"
        r"(?:\[?\s*)?"
        r"(?:author'?s?\s+note|a/?n)\s*:?"
        r".*?"
        r"(?:\]|(?=\n\n))"
        r"\s*"
        r"(?:\n\s*\*{3,}\s*)?"
        r"\n+",
        stripped,
    )
    if not match:
        return text, False
    remainder = stripped[match.end() :]
    remainder = _strip_trailing_section_dividers(remainder)
    return remainder.lstrip("\n"), True


def _strip_bracketed_leading_meta(text: str) -> tuple[str, bool]:
    """Remove a leading [meta commentary] block without an explicit author-note label."""
    stripped = text.lstrip()
    match = re.match(r"(?is)^\[(.+?)\]\s*(?:\n\s*\*{3,}\s*)?\n+", stripped)
    if not match:
        return text, False
    if not _META_BRACKET_PHRASES_RE.search(match.group(1)):
        return text, False
    remainder = stripped[match.end() :]
    remainder = _strip_trailing_section_dividers(remainder)
    return remainder.lstrip("\n"), True


def strip_leading_author_notes(text: str) -> StripResult:
    """Remove leading author notes and similar web-fiction meta preambles."""
    changed = False
    reason: str | None = None
    current = text
    strippers = (
        _strip_html_wrapped_leading_block,
        _strip_plain_author_note,
        _strip_bracketed_leading_meta,
    )
    while True:
        step_changed = False
        for stripper in strippers:
            new_text, one = stripper(current)
            if one:
                current = new_text
                changed = True
                step_changed = True
                reason = "author_note"
                break
        if not step_changed:
            break

    meta_match = _LEADING_META_PREAMBLE_RE.match(current)
    if meta_match:
        current = current[meta_match.end() :].lstrip("\n")
        changed = True
        reason = "meta_preamble"

    if not changed:
        return StripResult(text, False)
    return StripResult(_normalize_ws(current), True, reason)


def strip_web_fiction_markup(text: str) -> StripResult:
    """Strip leading author notes, then HTML tags/entities."""
    text = text.strip()
    if not text:
        return StripResult("", False)

    note_result = strip_leading_author_notes(text)
    cleaned = strip_html_markup(note_result.text)
    cleaned = _strip_trailing_section_dividers(cleaned)
    cleaned = _normalize_ws(cleaned)
    stripped = note_result.stripped or cleaned != text
    reason = note_result.reason
    if cleaned != note_result.text and reason is None:
        reason = "html_markup"
    return StripResult(cleaned, stripped, reason)
