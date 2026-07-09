"""Remove invisible Unicode controls and normalize unusual line breaks."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Invisible / formatting characters that should not appear in training prose.
_INVISIBLE_CHARS = "".join(
    chr(code)
    for code in (
        0x00AD,  # soft hyphen
        0x034F,  # combining grapheme joiner
        0x061C,  # arabic letter mark
        0x115F,  # hangul choseong filler
        0x1160,  # hangul jungseong filler
        0x17B4,  # khmer vowel inherent aq
        0x17B5,  # khmer vowel inherent aa
        0x180E,  # mongolian vowel separator
        0x200B,  # zero width space
        0x200C,  # zero width non-joiner
        0x200D,  # zero width joiner
        0x200E,  # left-to-right mark
        0x200F,  # right-to-left mark
        0x202A,  # left-to-right embedding
        0x202B,  # right-to-left embedding
        0x202C,  # pop directional formatting
        0x202D,  # left-to-right override
        0x202E,  # right-to-left override
        0x2060,  # word joiner
        0x2061,  # function application
        0x2062,  # invisible times
        0x2063,  # invisible separator
        0x2064,  # invisible plus
        0x2066,  # left-to-right isolate
        0x2067,  # right-to-left isolate
        0x2068,  # first strong isolate
        0x2069,  # pop directional isolate
        0x206A,  # inhibit symmetric swapping
        0x206B,  # activate symmetric swapping
        0x206C,  # inhibit arabic form shaping
        0x206D,  # activate arabic form shaping
        0x206E,  # national digit shapes
        0x206F,  # nominal digit shapes
        0xFEFF,  # byte order mark / zero width no-break space
        0xFFA0,  # halfwidth hangul filler
    )
)

_LINE_BREAK_TRANSLATION = str.maketrans(
    {
        "\u000b": "\n",  # vertical tab
        "\u000c": "\n",  # form feed
        "\u0085": "\n",  # next line
        "\u2028": "\n",  # line separator
        "\u2029": "\n\n",  # paragraph separator
    }
)


@dataclass(frozen=True)
class NormalizeResult:
    text: str
    normalized: bool


def normalize_unicode_text(text: str) -> NormalizeResult:
    """Strip invisible controls and map Unicode line breaks to newlines."""
    if not text:
        return NormalizeResult("", False)

    original = text
    text = text.translate(_LINE_BREAK_TRANSLATION)
    if _INVISIBLE_CHARS:
        text = text.translate({ord(ch): None for ch in _INVISIBLE_CHARS})
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return NormalizeResult(text.strip(), text != original)
