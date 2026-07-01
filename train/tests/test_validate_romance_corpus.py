"""Tests for romance_corpus validation."""

from __future__ import annotations

from pathlib import Path

from tools.data_preparation.validate_romance_corpus import _is_allowed_file


def test_allowed_styled_names() -> None:
    assert _is_allowed_file("horror_styled.jsonl")
    assert _is_allowed_file("fiction_books_styled_seg_000.jsonl")
    assert _is_allowed_file("fiction_books_deep_seg_000.jsonl")


def test_disallowed_names() -> None:
    assert not _is_allowed_file("fiction_books_pipeline_chunks.jsonl")
    assert not _is_allowed_file("horror_styled.jsonl.pre_strip.bak")
    assert not _is_allowed_file("temp.jsonl")
