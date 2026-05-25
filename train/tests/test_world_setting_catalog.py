"""Tests for world setting catalog in romance_factory.generate (THE-151 / Req. 11.1–11.3)."""

from __future__ import annotations

import re

from romance_factory.generate.config_v2 import V2Config, load_v2_config
from romance_factory.generate.world_setting_catalog import (
    build_world_context_for_character_web_prompt,
    build_world_context_for_conflict_prompt,
    build_world_context_for_story_outline_prompt,
    default_world_setting_catalog,
    resolve_world_setting_catalog,
)

# Requirement 11.2: exact top-level world type keys.
_EXPECTED_KEYS = frozenset(
    {
        "contemporary",
        "historical",
        "fantasy",
        "paranormal",
        "urban_fantasy",
        "sci_fi",
        "steampunk",
        "mythological",
        "post_apocalyptic",
        "hybrid",
    }
)

_SNAKE_CASE_ID = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*$")


def _assert_catalog_entry_shape(wid: str, entry: dict) -> None:
    """Requirement 11.3: name plus non-empty string lists (incl. supporting archetypes).

    Note: ``module_ids`` is enforced on the packaged default in a dedicated test, not
    here, so user settings.yaml overrides are free to omit it.
    """
    assert isinstance(entry["name"], str) and entry["name"].strip()
    for k in (
        "setting_subtypes",
        "world_pressure_types",
        "example_environments",
        "supporting_character_archetypes",
    ):
        assert isinstance(entry[k], list) and len(entry[k]) > 0
        assert all(isinstance(x, str) and x.strip() for x in entry[k])


def test_world_setting_catalog_has_ten_required_keys() -> None:
    """Req. 11.2: exactly ten world types with the specified identifiers."""
    catalog = default_world_setting_catalog()
    assert len(catalog) == 10
    assert frozenset(catalog.keys()) == _EXPECTED_KEYS
    for wid in catalog:
        assert _SNAKE_CASE_ID.match(wid), f"world type id must be snake_case: {wid!r}"


def test_world_setting_catalog_each_entry_required_fields_and_lists() -> None:
    """Req. 11.3: each entry has name and non-empty list fields."""
    catalog = default_world_setting_catalog()
    for wid, entry in catalog.items():
        _assert_catalog_entry_shape(wid, entry)


def test_packaged_catalog_every_entry_declares_module_ids() -> None:
    """Genre-driven world modules: every packaged entry carries a module_ids list."""
    catalog = default_world_setting_catalog()
    for wid, entry in catalog.items():
        assert "module_ids" in entry, f"{wid}: packaged catalog entry missing module_ids"
        mids = entry["module_ids"]
        assert isinstance(mids, list)
        assert all(isinstance(m, str) and m.strip() for m in mids)


def test_default_world_setting_catalog_is_deep_copy() -> None:
    a = default_world_setting_catalog()
    b = default_world_setting_catalog()
    assert a == b
    assert a is not b
    a["contemporary"]["name"] = "mutated"
    assert b["contemporary"]["name"] != "mutated"


def test_resolve_world_setting_catalog() -> None:
    cfg = load_v2_config()
    resolved = resolve_world_setting_catalog(cfg)
    assert len(resolved) == 10
    assert frozenset(resolved.keys()) == _EXPECTED_KEYS
    for wid, entry in resolved.items():
        _assert_catalog_entry_shape(wid, entry)

    empty = V2Config()
    assert resolve_world_setting_catalog(empty) == default_world_setting_catalog()


def test_build_world_context_for_conflict_prompt_includes_core_fields() -> None:
    world = {
        "lore": "Ancient treaty fraying.",
        "culture": "Public honor, private bargains.",
        "world_state": "stable",
        "world_pressure_types": ["political_tension"],
        "world_type": "fantasy",
    }
    text = build_world_context_for_conflict_prompt(world, catalog=default_world_setting_catalog())
    assert "STORY WORLD" in text
    assert "Ancient treaty" in text
    assert "Public honor" in text
    assert "political_tension" in text
    assert "ESCALATION DRIVER" not in text


def test_build_world_context_threatened_adds_escalation_driver() -> None:
    world = {
        "lore": "Lore",
        "culture": "Culture",
        "world_state": "threatened",
        "world_pressure_types": ["x"],
        "world_type": "contemporary",
    }
    text = build_world_context_for_conflict_prompt(world)
    assert "ESCALATION DRIVER" in text
    assert "threatened" in text.lower()


def test_build_world_context_hybrid_layers_catalog_pressures() -> None:
    catalog = default_world_setting_catalog()
    world = {
        "lore": "L",
        "culture": "C",
        "world_state": "stable",
        "world_pressure_types": ["from_artifact"],
        "world_type": "hybrid:sci_fi+paranormal",
    }
    text = build_world_context_for_conflict_prompt(world, catalog=catalog)
    assert "HYBRID WORLD" in text
    sf = catalog["sci_fi"]["world_pressure_types"]
    pa = catalog["paranormal"]["world_pressure_types"]
    for p in sf[:2]:
        assert p in text
    for p in pa[:2]:
        assert p in text
    assert "from_artifact" in text


def test_build_world_context_for_character_web_includes_setting_culture_cast() -> None:
    world = {
        "setting_design": "Coastal arcology with tide-locked docks.",
        "culture": "Merit guilds and public oath courts.",
        "supporting_cast": [
            {
                "name": "Sam Rivera",
                "role_in_world": "Harbor warden",
                "brief_description": "Keeps smugglers honest.",
            },
        ],
        "world_type": "sci_fi",
    }
    text = build_world_context_for_character_web_prompt(world, catalog=default_world_setting_catalog())
    assert "STORY WORLD" in text
    assert "Coastal arcology" in text
    assert "Merit guilds" in text
    assert "Sam Rivera" in text
    assert "Harbor warden" in text


def test_build_world_context_for_character_web_hybrid_adds_notes() -> None:
    catalog = default_world_setting_catalog()
    world = {
        "setting_design": "S",
        "culture": "C",
        "supporting_cast": [],
        "world_type": "hybrid:sci_fi+paranormal",
    }
    text = build_world_context_for_character_web_prompt(world, catalog=catalog)
    assert "HYBRID WORLD NOTES" in text
    assert "Science Fiction" in text or "sci" in text.lower()


def test_build_world_context_for_story_outline_includes_locations_and_cast() -> None:
    world = {
        "setting_design": "The Iron Stair ward and the river markets below.",
        "environmental_logic": "Perpetual drizzle; brass lamps at every crossing.",
        "lore": "Guild charters date to the flood year.",
        "culture": "Debt-bond clerks run the docks.",
        "supporting_cast": [
            {
                "name": "Mara Venn",
                "role_in_world": "Clerk",
                "brief_description": "Knows every lien.",
            },
        ],
        "world_type": "steampunk",
    }
    text = build_world_context_for_story_outline_prompt(world, catalog=default_world_setting_catalog())
    assert "STORY WORLD" in text
    assert "Iron Stair" in text
    assert "Perpetual drizzle" in text
    assert "Guild charters" in text
    assert "Mara Venn" in text
    assert "scene interaction" in text.lower() or "dialogue" in text.lower()


def test_build_world_context_for_story_outline_can_omit_supporting_cast() -> None:
    world = {
        "setting_design": "Neo-Tokyo arcologies over the undercity.",
        "environmental_logic": "Humid neon nights.",
        "lore": "Post-Collapse rebuild.",
        "culture": "Council of humans and elders.",
        "supporting_cast": [
            {
                "name": "Hana Nakamura",
                "role_in_world": "Hacker",
                "brief_description": "Moral compass.",
            },
        ],
        "world_type": "steampunk",
    }
    text = build_world_context_for_story_outline_prompt(
        world,
        catalog=default_world_setting_catalog(),
        include_supporting_cast=False,
    )
    assert "STORY WORLD" in text
    assert "Neo-Tokyo" in text
    assert "Hana Nakamura" not in text
    assert "supporting cast" not in text.lower()
    assert "locations and atmosphere" in text


def test_build_world_context_for_story_outline_hybrid_adds_environment_notes() -> None:
    catalog = default_world_setting_catalog()
    world = {
        "setting_design": "S",
        "supporting_cast": [],
        "world_type": "hybrid:fantasy+contemporary",
    }
    text = build_world_context_for_story_outline_prompt(world, catalog=catalog)
    assert "HYBRID WORLD" in text
    assert "example environments" in text.lower()


def test_build_world_context_for_story_outline_empty_returns_empty() -> None:
    assert build_world_context_for_story_outline_prompt({}) == ""
    assert build_world_context_for_story_outline_prompt(None) == ""


def test_story_outline_world_pacing_block_matches_chapter_count() -> None:
    from romance_factory.generate.pipeline_v2 import PipelineV2

    t = PipelineV2._story_outline_world_pacing_block(10)
    assert "chapters 1–5 of 10" in t or "chapters 1-5 of 10" in t
    assert "establishment band" in t.lower()
    assert "WORLD ARTIFACT" in t
    assert "setting_design" in t
    assert "scene interaction" in t.lower()


def test_conflict_prompt_folds_genre_modules() -> None:
    """Active genre modules should be rendered into the external-conflict prompt."""
    world = {
        "lore": "Colonial charters splintering.",
        "culture": "Frontier guild courts.",
        "world_state": "stable",
        "world_pressure_types": ["scarcity"],
        "world_type": "sci_fi",
        "genre_modules": {
            "tech_stack": {
                "tech_level": "late interplanetary",
                "power_sources": "fusion cores",
                "transport": "torch freighters",
                "daily_tech_examples": "wrist comms",
                "failure_modes": "blackouts on the ring",
            },
        },
    }
    text = build_world_context_for_conflict_prompt(world, catalog=default_world_setting_catalog())
    assert "Tech stack" in text
    assert "fusion cores" in text


def test_character_web_prompt_folds_role_filtered_modules() -> None:
    """Modules without ``character_web`` in downstream_roles must be skipped there."""
    world = {
        "setting_design": "A ringed habitat over Europa.",
        "culture": "Cartel-brokered permits.",
        "supporting_cast": [],
        "world_type": "sci_fi",
        "genre_modules": {
            "tech_stack": {
                "tech_level": "late interplanetary",
                "power_sources": "fusion cores",
                "transport": "torch freighters",
                "daily_tech_examples": "wrist comms",
                "failure_modes": "blackouts",
            },
            "ftl_travel": {
                "ftl_method": "folded jump corridors",
                "travel_time_model": "subjective hours for chart weeks",
                "limitations": "licensed pilots only",
                "infrastructure": "gate towers",
            },
        },
    }
    text = build_world_context_for_character_web_prompt(
        world, catalog=default_world_setting_catalog(),
    )
    assert "Tech stack" in text
    # ftl_travel default roles exclude character_web.
    assert "FTL" not in text


def test_story_outline_prompt_folds_small_town_module() -> None:
    world = {
        "setting_design": "Oak Hollow on the cape.",
        "supporting_cast": [],
        "world_type": "contemporary",
        "genre_modules": {
            "small_town_social": {
                "local_shops_and_hangouts": "Mabel's Diner; Kline's Hardware",
                "annual_events": "Founders' Day parade",
                "gossip_network": "runs through the stylist",
                "institutional_cornerstones": "First Methodist; town hall",
                "outsider_dynamics": "newcomers watched for a full season",
            },
        },
    }
    text = build_world_context_for_story_outline_prompt(
        world, catalog=default_world_setting_catalog(),
    )
    assert "Mabel's Diner" in text
