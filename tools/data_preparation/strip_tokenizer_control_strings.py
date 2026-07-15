"""Remove Tekken/Mistral control-token *strings* that HF encodes as special IDs.

Mistral-NeMo's Tekken tokenizer maps the literal text ``<SPECIAL_23>`` to token
id 23 (a reserved filler control token). If that string — or other visible
control markers like ``[INST]`` — appears in training prose, Supervised Fine-
Tuning teaches the model to emit those IDs mid-generation.

These markers should never appear in ordinary prose; they exist only as token
IDs injected by the chat template / tool protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Filler ranks decode as <SPECIAL_N>; named controls decode as these strings.
_TEKKEN_NAMED_CONTROLS = (
    "<unk>",
    "<s>",
    "</s>",
    "[INST]",
    "[/INST]",
    "[AVAILABLE_TOOLS]",
    "[/AVAILABLE_TOOLS]",
    "[TOOL_RESULTS]",
    "[/TOOL_RESULTS]",
    "[TOOL_CALLS]",
    "<pad>",
    "[PREFIX]",
    "[MIDDLE]",
    "[SUFFIX]",
)

_SPECIAL_FILLER_RE = re.compile(r"<SPECIAL_\d+>", re.IGNORECASE)
_NAMED_CONTROL_RE = re.compile(
    "|".join(re.escape(tok) for tok in sorted(_TEKKEN_NAMED_CONTROLS, key=len, reverse=True))
)


@dataclass(frozen=True)
class StripResult:
    text: str
    stripped: bool
    removed: tuple[str, ...] = ()


def strip_tokenizer_control_strings(text: str) -> StripResult:
    """Delete visible Tekken/Mistral special-token spellings from text."""
    if not text:
        return StripResult(text=text or "", stripped=False)

    removed: list[str] = []

    def _keep_special(match: re.Match[str]) -> str:
        removed.append(match.group(0))
        return ""

    def _keep_named(match: re.Match[str]) -> str:
        removed.append(match.group(0))
        return ""

    out = _SPECIAL_FILLER_RE.sub(_keep_special, text)
    out = _NAMED_CONTROL_RE.sub(_keep_named, out)
    if not removed:
        return StripResult(text=text, stripped=False)

    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return StripResult(text=out.strip(), stripped=True, removed=tuple(removed))
