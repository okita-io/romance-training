"""Tests for Tekken/Mistral control-token string stripping."""

from __future__ import annotations

from tools.data_preparation.strip_tokenizer_control_strings import (
    strip_tokenizer_control_strings,
)
from tools.data_preparation.unified_corpus import normalize_prose_text


def test_strip_special_filler_token() -> None:
    raw = "She turned away. <SPECIAL_23> Then he spoke."
    result = strip_tokenizer_control_strings(raw)
    assert result.stripped
    assert "<SPECIAL_23>" not in result.text
    assert "She turned away." in result.text
    assert "Then he spoke." in result.text
    assert "<SPECIAL_23>" in result.removed


def test_strip_named_control_tokens() -> None:
    raw = "Hello [INST]world[/INST] and <s>gone</s>."
    result = strip_tokenizer_control_strings(raw)
    assert result.stripped
    assert "[INST]" not in result.text
    assert "[/INST]" not in result.text
    assert "<s>" not in result.text


def test_normalize_prose_strips_specials() -> None:
    raw = "Quiet night. <SPECIAL_14> Moonlight on the water."
    out = normalize_prose_text(raw, reflow_ocr=False)
    assert "<SPECIAL_14>" not in out
    assert "Moonlight" in out


def test_unchanged_when_clean() -> None:
    raw = "Ordinary prose with no control markers."
    result = strip_tokenizer_control_strings(raw)
    assert not result.stripped
    assert result.text == raw
