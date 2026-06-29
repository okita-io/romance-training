"""Tests for OCR / Gutenberg prose reflow."""

from __future__ import annotations

from tools.data_preparation.reflow_prose import reflow_ocr_prose


def test_reflow_mid_sentence_line_breaks() -> None:
    raw = (
        "Here and there, a fire was lighted in the streets, round which ragged\n"
        "urchins and mendicants were collected."
    )
    assert reflow_ocr_prose(raw) == (
        "Here and there, a fire was lighted in the streets, round which ragged "
        "urchins and mendicants were collected."
    )


def test_reflow_keeps_paragraph_breaks() -> None:
    raw = "First paragraph line\nstill first.\n\nSecond paragraph here."
    assert reflow_ocr_prose(raw) == (
        "First paragraph line still first.\n\nSecond paragraph here."
    )


def test_reflow_hyphenation() -> None:
    assert reflow_ocr_prose("The elixir of long life was sum-\nmer's goal.") == (
        "The elixir of long life was summer's goal."
    )


def test_drop_cap_letter_paragraph_split() -> None:
    assert reflow_ocr_prose("I\n\nt was the best of times.") == "It was the best of times."


def test_drop_cap_letter_single_newline() -> None:
    assert reflow_ocr_prose("I\nt was the best of times.") == "It was the best of times."


def test_drop_cap_word_paragraph_split() -> None:
    assert reflow_ocr_prose("It\n\nwas the best of times.") == "It was the best of times."


def test_drop_cap_does_not_merge_section_headers() -> None:
    raw = "CHAPTER I\n\nThe morning sun shone brightly."
    assert reflow_ocr_prose(raw) == raw


def test_reflow_record_shape() -> None:
    from tools.data_preparation.reflow_prose import reflow_record

    record = {"text": "One\nline.", "metadata": {"source": "test"}}
    out = reflow_record(record)
    assert out["text"] == "One line."
    assert out["metadata"]["text_reflowed"] is True
    assert out["metadata"]["word_count"] == 2
