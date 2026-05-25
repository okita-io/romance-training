"""Unit tests for V2 establishment band helpers."""

from __future__ import annotations

from romance_factory.generate.pacing_bands import (
    chapter_in_establishment_band,
    establishment_band_bounds,
    generic_establishment_pacing_block,
)


def test_establishment_band_starts_at_chapter_one() -> None:
    low, high = establishment_band_bounds(10)
    assert low == 1
    assert high == 5


def test_chapter_in_band_n10() -> None:
    for ch in (1, 2, 3, 4, 5):
        assert chapter_in_establishment_band(ch, 10) is True
    assert chapter_in_establishment_band(6, 10) is False


def test_chapter_in_band_n20() -> None:
    _, high = establishment_band_bounds(20)
    assert high == 10
    assert chapter_in_establishment_band(1, 20) is True
    assert chapter_in_establishment_band(10, 20) is True
    assert chapter_in_establishment_band(11, 20) is False


def test_generic_block_mentions_band() -> None:
    txt = generic_establishment_pacing_block(10)
    assert "STORY ESTABLISHMENT" in txt
    assert "chapters 1" in txt and "of 10" in txt
