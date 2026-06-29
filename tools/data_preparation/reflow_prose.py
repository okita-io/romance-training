"""Reflow OCR / Gutenberg line wraps and repair drop-cap paragraph splits."""

from __future__ import annotations

import re

_DROP_CAP_LETTER = re.compile(r"^[A-Z]$")
_DROP_CAP_WORD = re.compile(r"^[A-Z][a-z']{1,11}$")


def _is_drop_cap_fragment(text: str) -> bool:
    """True when a block is an isolated drop-cap letter or short first word."""
    line = text.strip()
    if not line or "\n" in line:
        return False
    return bool(_DROP_CAP_LETTER.fullmatch(line) or _DROP_CAP_WORD.fullmatch(line))


def _merge_drop_cap_paragraphs(paragraphs: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(paragraphs):
        current = paragraphs[index].strip()
        if (
            index + 1 < len(paragraphs)
            and _is_drop_cap_fragment(current)
        ):
            nxt = paragraphs[index + 1].strip()
            if nxt and nxt[0].islower():
                if _DROP_CAP_LETTER.fullmatch(current):
                    current = current + nxt
                else:
                    current = current + " " + nxt
                index += 2
                merged.append(current)
                continue
        merged.append(current)
        index += 1
    return merged


def _reflow_paragraph(para: str) -> str:
    lines = [line.strip() for line in para.split("\n") if line.strip()]
    if not lines:
        return ""

    # Drop cap split with a single newline inside the paragraph block.
    if len(lines) >= 2 and _DROP_CAP_LETTER.fullmatch(lines[0]) and lines[1][0].islower():
        lines[0] = lines[0] + lines[1]
        lines.pop(1)
    elif (
        len(lines) >= 2
        and _DROP_CAP_WORD.fullmatch(lines[0])
        and lines[1][0].islower()
    ):
        lines[0] = lines[0] + " " + lines[1]
        lines.pop(1)

    out = lines[0]
    for line in lines[1:]:
        if out.endswith("-"):
            out = out[:-1] + line
        else:
            out = f"{out} {line}"
    return out


def reflow_ocr_prose(text: str) -> str:
    """
    Join hard-wrapped lines within paragraphs; keep blank-line paragraph breaks.

    Repairs common Gutenberg OCR artifacts:
    - mid-sentence line breaks from fixed page width
    - hyphenation across line breaks (``sum-\\nmer`` -> ``summer``)
    - drop-cap splits: ``I\\n\\nt was...`` / ``It\\n\\nwas...`` / ``I\\nt was...``
    """
    if not text:
        return text

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [block for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not paragraphs:
        return text.strip()

    paragraphs = _merge_drop_cap_paragraphs(paragraphs)
    reflowed = [_reflow_paragraph(para) for para in paragraphs]
    return "\n\n".join(reflowed).strip()


def reflow_record(record: dict) -> dict:
    """Return a copy of a JSONL record with reflowed ``text`` and updated word_count."""
    out = dict(record)
    text = reflow_ocr_prose(record.get("text") or "")
    out["text"] = text
    metadata = dict(out.get("metadata") or {})
    metadata["word_count"] = len(text.split())
    metadata["text_reflowed"] = True
    out["metadata"] = metadata
    return out
