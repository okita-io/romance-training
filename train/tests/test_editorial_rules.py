"""Business-logic tests for editorial rule helpers."""

from __future__ import annotations

import romance_factory.story_core.editorial_rules as er


def test_cliffhanger_weight_none_profile():
    base = er.CHAPTER_CLIFFHANGERS_AND_PAYOFFS.weight
    assert er.cliffhanger_editorial_weight_for_profile(None) == base


def test_cliffhanger_weight_clamped():
    base = er.CHAPTER_CLIFFHANGERS_AND_PAYOFFS.weight
    assert er.cliffhanger_editorial_weight_for_profile({"cliffhanger_editorial_weight": 0.01}) == 0.05
    assert er.cliffhanger_editorial_weight_for_profile({"cliffhanger_editorial_weight": 0.99}) == 0.45
    assert er.cliffhanger_editorial_weight_for_profile({"cliffhanger_editorial_weight": 0.2}) == 0.2


def test_cliffhanger_weight_invalid_falls_back():
    base = er.CHAPTER_CLIFFHANGERS_AND_PAYOFFS.weight
    assert er.cliffhanger_editorial_weight_for_profile({"cliffhanger_editorial_weight": "nope"}) == base
