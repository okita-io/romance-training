"""Tests for mechanical anti-slop checks."""

from __future__ import annotations

from romance_factory.story_core.anti_slop import (
    format_slop_mechanical_excerpts,
    slop_score,
)


def test_non_english_script_is_flagged():
    snap = slop_score("He looked at her and whispered 你好 in the dark.")
    assert snap["non_english_script_count"] > 0
    assert snap["slop_penalty"] > 0


def test_hyphen_overuse_is_measured():
    text = (" - " * 180) + (" -- " * 60) + "They kissed."
    snap = slop_score(text)
    assert snap["standalone_hyphen_count"] > 0
    assert snap["double_hyphen_count"] > 0
    assert snap["hyphen_overuse_density"] > 25


def test_format_slop_mechanical_excerpts_quotes_targets():
    text = (
        "She looked away. Her eyes widened at the door. "
        "We should leverage every moment, he said."
    )
    snap = slop_score(text)
    block = format_slop_mechanical_excerpts(text, snap)
    assert "TARGET EXCERPTS" in block
    assert "eyes widened" in block.lower() or "Eyes widened" in block
    assert "leverage" in block.lower()

