"""Tests for typographic quote normalization."""

from __future__ import annotations

from tools.data_preparation.normalize_quotes import normalize_quotes
from tools.data_preparation.unified_corpus import normalize_prose_text

_LDQ = chr(0x201C)
_RDQ = chr(0x201D)
_LSQ = chr(0x2018)
_RSQ = chr(0x2019)


def test_normalize_curly_double_quotes() -> None:
    raw = f"He said {_LDQ}hello{_RDQ} and walked away."
    result = normalize_quotes(raw)
    assert result.normalized
    assert result.text == 'He said "hello" and walked away.'


def test_normalize_curly_apostrophe() -> None:
    raw = f"It{_RSQ}s a fine day."
    result = normalize_quotes(raw)
    assert result.normalized
    assert result.text == "It's a fine day."


def test_keep_ascii_quotes() -> None:
    raw = '"Stay," she said. "I won\'t."'
    result = normalize_quotes(raw)
    assert not result.normalized
    assert result.text == raw


def test_normalize_mixed_quotes_in_dialogue() -> None:
    raw = (
        f"{_LDQ}I{_RSQ}m thinking,{_RDQ} darryl said, "
        f'{_LDQ}that we should leave.{_RDQ}'
    )
    result = normalize_quotes(raw)
    assert result.normalized
    assert '"' in result.text
    assert result.text.count('"') == 4
    assert _LDQ not in result.text
    assert _RDQ not in result.text
    assert _RSQ not in result.text


def test_normalize_prose_text_applies_quotes() -> None:
    raw = f"{_LDQ}The morning was cold.{_RDQ}"
    out = normalize_prose_text(raw, reflow_ocr=False)
    assert out == '"The morning was cold."'
