"""Tests for invisible Unicode control stripping."""

from __future__ import annotations

from tools.data_preparation.normalize_unicode import normalize_unicode_text

LRE = "\u202a"
PDF = "\u202c"
ZWSP = "\u200b"
LINE_SEP = "\u2028"
PARA_SEP = "\u2029"


def test_strip_bidi_embedding_controls() -> None:
    raw = f"It is {LRE}11.30pm{PDF} and I am finally on the train."
    result = normalize_unicode_text(raw)
    assert result.normalized
    assert result.text == "It is 11.30pm and I am finally on the train."
    assert LRE not in result.text
    assert PDF not in result.text


def test_strip_zero_width_space() -> None:
    raw = f"hello{ZWSP}world"
    result = normalize_unicode_text(raw)
    assert result.normalized
    assert result.text == "helloworld"


def test_normalize_unicode_line_separators() -> None:
    raw = f"first line{LINE_SEP}second line{PARA_SEP}third line"
    result = normalize_unicode_text(raw)
    assert result.normalized
    assert result.text == "first line\nsecond line\n\nthird line"
