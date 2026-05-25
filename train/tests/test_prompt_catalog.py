"""Smoke tests for generate/prompts/prompt_catalog.yaml and core/prompt_catalog.py."""

from __future__ import annotations

from romance_factory.core import config
from romance_factory.core import prompt_catalog as pc
from romance_factory.story_core.anti_slop import slop_score


def test_prompt_catalog_document_loads() -> None:
    doc = pc.load_prompt_catalog_document()
    assert isinstance(doc, dict)
    assert doc.get("alignment")
    assert str(doc["alignment"].get("romance_fiction_identity", "")).strip()
    anti = doc.get("anti_slop") or {}
    assert isinstance(anti, dict)
    assert anti.get("tier1_banned")


def test_twist_catalog_merge_includes_yaml_plot_twists() -> None:
    tc = config.TWIST_CATALOG
    assert tc.get("default")
    flat = " ".join(tc["default"])
    assert "Red Herring" in flat or "red herring" in flat.lower()


def test_gothic_pool_has_genre_and_merged_categories() -> None:
    g = config.TWIST_CATALOG.get("gothic", [])
    assert len(g) >= 5
    joined = " ".join(g).lower()
    assert "manor" in joined or "diary" in joined


def test_slop_score_hits_tier1_from_catalog() -> None:
    r = slop_score("The board asked them to delve into the Q4 synergy narrative.")
    assert r["tier1_hits"]
    words = {w for w, _ in r["tier1_hits"]}
    assert "delve" in words or "synergy" in words
