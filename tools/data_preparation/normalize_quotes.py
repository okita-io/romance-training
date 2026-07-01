"""Normalize typographic quotation marks to ASCII for consistent model generation."""

from __future__ import annotations

from dataclasses import dataclass

# Use chr() so source files stay ASCII-only (same convention as metrics_computable).
_QUOTE_TRANSLATION = str.maketrans(
    {
        chr(0x201C): '"',  # left double quotation mark
        chr(0x201D): '"',  # right double quotation mark
        chr(0x201E): '"',  # double low-9 quotation mark
        chr(0x2018): "'",  # left single quotation mark
        chr(0x2019): "'",  # right single quotation mark / apostrophe
        chr(0x201A): "'",  # single low-9 quotation mark
        chr(0x00AB): '"',  # left-pointing double angle (guillemet)
        chr(0x00BB): '"',  # right-pointing double angle
        chr(0x2039): "'",  # single left-pointing angle quotation mark
        chr(0x203A): "'",  # single right-pointing angle quotation mark
    }
)


@dataclass(frozen=True)
class NormalizeResult:
    text: str
    normalized: bool


def normalize_quotes(text: str) -> NormalizeResult:
    """Map curly and guillemet quotes to ASCII ``"`` and ``'``."""
    if not text:
        return NormalizeResult("", False)
    translated = text.translate(_QUOTE_TRANSLATION)
    return NormalizeResult(translated, translated != text)
