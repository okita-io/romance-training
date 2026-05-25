"""Unit tests for world generation seed construction (THE-154 / Req 3.x, 10.x)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from romance_factory.generate.config_v2 import V2Config
from romance_factory.generate.pipeline_v2 import PipelineV2
from romance_factory.generate.world_setting_catalog import default_world_setting_catalog


@contextmanager
def _pipeline_v2_with_mocks(story_path: str, tmp_path, **config_overrides):
    """Minimal PipelineV2 instance without loading embedding models or LanceDB."""
    cfg = V2Config(
        story_path=story_path,
        db_path=str(tmp_path / "lancedb"),
        embedding_model="mock",
        **config_overrides,
    )
    patches = [
        patch("romance_factory.generate.pipeline_v2.EmbeddingProvider"),
        patch("romance_factory.generate.pipeline_v2.LanceDBEngine"),
        patch("romance_factory.generate.pipeline_v2.PromptBuilder"),
        patch("romance_factory.generate.pipeline_v2.ActGenerationAgent"),
        patch("romance_factory.generate.pipeline_v2.ActIntroPlanningAgent"),
        patch("romance_factory.generate.pipeline_v2.EditorialAgent"),
        patch("romance_factory.generate.pipeline_v2.RewriteAgent"),
        patch("romance_factory.generate.pipeline_v2.CleanupAgent"),
    ]
    started = [p.start() for p in patches]
    engine_mock = started[1].return_value
    engine_mock.validate_collections.return_value = True
    try:
        yield PipelineV2(story_path, cfg)
    finally:
        for p in reversed(patches):
            p.stop()


@pytest.fixture
def minimal_catalog() -> dict:
    return {
        "fantasy": {
            "name": "Fantasy",
            "setting_subtypes": ["high_fantasy_kingdoms"],
            "world_pressure_types": ["magic_systems"],
            "example_environments": ["Stone citadel"],
            "supporting_character_archetypes": ["royalty", "knights"],
            "module_ids": ["magic_system", "religion_pantheon"],
        },
        "sci_fi": {
            "name": "Science Fiction",
            "setting_subtypes": ["space_stations"],
            "world_pressure_types": ["hierarchy"],
            "example_environments": ["Orbital ring"],
            "supporting_character_archetypes": ["engineers", "ship_officers"],
            "module_ids": ["tech_stack", "ftl_travel"],
        },
        "hybrid": {
            "name": "Hybrid",
            "setting_subtypes": ["fusion"],
            "world_pressure_types": ["stacked_pressures"],
            "example_environments": ["Cross-genre sprawl"],
            "supporting_character_archetypes": ["cross_genre_brokers"],
            "module_ids": [],
        },
    }


def test_require_genre_and_world_for_seed(tmp_path, minimal_catalog):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        world="fantasy",
        world_setting_catalog=minimal_catalog,
    ) as p:
        with pytest.raises(RuntimeError, match="both genre and world"):
            p._build_world_generation_seed()


def test_genre_and_world_merges_genre_brief_with_catalog(tmp_path, minimal_catalog):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        genre="contemporary",
        world="fantasy",
        world_setting_catalog=minimal_catalog,
    ) as p:
        seed, resolved, pressures = p._build_world_generation_seed()
    assert "STORY GENRE:" in seed or "contemporary" in seed.lower()
    assert "Setting subtypes:" in seed
    assert "high_fantasy_kingdoms" in seed
    assert resolved == "fantasy"
    assert "magic_systems" in pressures
    assert "ALIGNMENT:" in seed


def test_world_with_genre_includes_genre_brief_not_world_only_defaults(
    tmp_path, minimal_catalog,
):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        genre="historical",
        world="fantasy",
        world_setting_catalog=minimal_catalog,
    ) as p:
        seed, resolved, pressures = p._build_world_generation_seed()
    assert "STORY GENRE:" in seed or "historical" in seed.lower()
    assert "high_fantasy_kingdoms" in seed
    assert resolved == "fantasy"
    assert pressures == ["magic_systems"]
    assert "ALIGNMENT:" in seed


def test_free_form_world_verbatim(tmp_path, minimal_catalog):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        genre="contemporary",
        world="underwater casino on Europa",
        world_setting_catalog=minimal_catalog,
    ) as p:
        seed, resolved, _p = p._build_world_generation_seed()
    assert "free-form user description" in seed
    assert "underwater casino on Europa" in seed
    assert "free_form:" in resolved


def test_hybrid_merges_constituent_pressures(tmp_path, minimal_catalog):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        genre="contemporary",
        world="hybrid:fantasy+sci_fi",
        world_setting_catalog=minimal_catalog,
    ) as p:
        seed, resolved, pressures = p._build_world_generation_seed()
    assert "fantasy" in seed and "sci_fi" in seed
    assert resolved == "hybrid:fantasy+sci_fi"
    assert "magic_systems" in pressures and "hierarchy" in pressures


def test_hybrid_seed_includes_archetypes_from_both_constituents(tmp_path, minimal_catalog):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        genre="contemporary",
        world="hybrid:fantasy+sci_fi",
        world_setting_catalog=minimal_catalog,
    ) as p:
        seed, resolved, _ = p._build_world_generation_seed()
    assert resolved == "hybrid:fantasy+sci_fi"
    assert "Supporting character archetypes" in seed
    assert "royalty" in seed and "knights" in seed
    assert "engineers" in seed and "ship_officers" in seed


def test_normalize_world_parsed_data_coerces_without_placeholder_filler(tmp_path):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(sp, tmp_path) as p:
        raw = {"setting_design": "A port city"}
        out = p._normalize_world_parsed_data(
            raw,
            resolved_world_type="fantasy",
            fallback_pressure_types=["p1", "p2"],
        )
    assert out["setting_design"] == "A port city"
    assert out["lore"] == ""
    assert out["world_type"] == "fantasy"
    assert out["world_state"] == "stable"
    assert out["world_pressure_types"] == ["p1", "p2"]
    assert out["supporting_cast"] == []
    assert not p._validate_world_parsed_data(out)


def test_validate_world_bible_without_supporting_cast(tmp_path):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(sp, tmp_path) as p:
        bible = {
            "setting_design": "Coastal town",
            "lore": "l",
            "cosmology": "c",
            "culture": "c",
            "technology_or_era": "t",
            "environmental_logic": "e",
            "world_state": "stable",
            "world_pressure_types": ["tension"],
            "world_type": "contemporary",
        }
        assert p._validate_world_parsed_data(bible, require_supporting_cast=False)
        assert not p._validate_world_parsed_data(bible)


def test_validate_world_rejects_placeholder_derived_context(tmp_path):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(sp, tmp_path) as p:
        base = {
            "setting_design": "x",
            "lore": "(Placeholder — refine later.) Derived context for 'sci_fi'.",
            "cosmology": "x",
            "culture": "x",
            "technology_or_era": "x",
            "environmental_logic": "x",
            "supporting_cast": [
                {"name": "a", "role_in_world": "r", "brief_description": "b"},
                {"name": "b", "role_in_world": "r", "brief_description": "b"},
                {"name": "c", "role_in_world": "r", "brief_description": "b"},
            ],
            "world_state": "stable",
            "world_pressure_types": ["tension"],
            "world_type": "sci_fi",
        }
        assert not p._validate_world_parsed_data(base)


def test_validate_world_rejects_supporting_figure_padding(tmp_path):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(sp, tmp_path) as p:
        base = {
            "setting_design": "A ring habitat",
            "lore": "l",
            "cosmology": "c",
            "culture": "c",
            "technology_or_era": "t",
            "environmental_logic": "e",
            "supporting_cast": [
                {
                    "name": "Supporting figure 1",
                    "role_in_world": "Local contact or institutional anchor",
                    "brief_description": (
                        "Placeholder supporting character — replace when expanding the cast."
                    ),
                },
                {
                    "name": "Supporting figure 2",
                    "role_in_world": "Local contact or institutional anchor",
                    "brief_description": (
                        "Placeholder supporting character — replace when expanding the cast."
                    ),
                },
                {
                    "name": "Supporting figure 3",
                    "role_in_world": "Local contact or institutional anchor",
                    "brief_description": (
                        "Placeholder supporting character — replace when expanding the cast."
                    ),
                },
            ],
            "world_state": "stable",
            "world_pressure_types": ["scarcity"],
            "world_type": "sci_fi",
        }
        assert not p._validate_world_parsed_data(base)


def test_validate_world_rejects_missing_top_level_key(tmp_path):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(sp, tmp_path) as p:
        base = {
            "setting_design": "x",
            "lore": "x",
            "cosmology": "x",
            "culture": "x",
            "technology_or_era": "x",
            "environmental_logic": "x",
            "supporting_cast": [
                {"name": "a", "role_in_world": "r", "brief_description": "b"},
                {"name": "b", "role_in_world": "r", "brief_description": "b"},
                {"name": "c", "role_in_world": "r", "brief_description": "b"},
            ],
            "world_state": "stable",
            "world_pressure_types": ["tension"],
            "world_type": "fantasy",
        }
        assert p._validate_world_parsed_data(base)
        missing_lore = {k: v for k, v in base.items() if k != "lore"}
        assert not p._validate_world_parsed_data(missing_lore)


def test_packaged_catalog_has_ten_keys():
    cat = default_world_setting_catalog()
    expected = {
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
    assert set(cat.keys()) == expected


def test_pipeline_resolves_active_modules_as_union(tmp_path, minimal_catalog):
    """Pipeline helper unions --genre and --world module sets via catalog."""
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        genre="fantasy",
        world="sci_fi",
        world_setting_catalog=minimal_catalog,
    ) as p:
        ids = p._resolve_active_world_modules()
    assert set(ids) >= {"magic_system", "religion_pantheon", "tech_stack", "ftl_travel"}


def test_pipeline_resolves_no_modules_for_free_form_world(tmp_path, minimal_catalog):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        world="underwater casino on Europa",
        world_setting_catalog=minimal_catalog,
    ) as p:
        ids = p._resolve_active_world_modules()
    assert ids == []


def test_validate_world_requires_genre_modules_when_active(tmp_path):
    """When active module ids are supplied, validator demands genre_modules."""
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(sp, tmp_path) as p:
        base = {
            "setting_design": "A ring habitat",
            "lore": "l",
            "cosmology": "c",
            "culture": "c",
            "technology_or_era": "t",
            "environmental_logic": "e",
            "supporting_cast": [
                {"name": "a", "role_in_world": "r", "brief_description": "b"},
                {"name": "b", "role_in_world": "r", "brief_description": "b"},
                {"name": "c", "role_in_world": "r", "brief_description": "b"},
            ],
            "world_state": "stable",
            "world_pressure_types": ["scarcity"],
            "world_type": "sci_fi",
        }
        # Without active module ids the base passes.
        assert p._validate_world_parsed_data(base)
        # With an active module id and no genre_modules, validation fails.
        assert not p._validate_world_parsed_data(
            base, active_module_ids=["tech_stack"],
        )
        # Add a fully-populated genre_modules block and it passes again.
        base["genre_modules"] = {
            "tech_stack": {
                "tech_level": "late interplanetary",
                "power_sources": "fusion cores",
                "transport": "torch freighters",
                "daily_tech_examples": "wrist comms",
                "failure_modes": "blackouts on the ring",
            },
        }
        assert p._validate_world_parsed_data(
            base, active_module_ids=["tech_stack"],
        )


def test_normalize_world_populates_genre_modules_when_active(tmp_path):
    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(sp, tmp_path) as p:
        raw = {"setting_design": "A port city"}
        out = p._normalize_world_parsed_data(
            raw,
            resolved_world_type="fantasy",
            fallback_pressure_types=["p1"],
            active_module_ids=["magic_system"],
        )
    assert "genre_modules" in out
    assert "magic_system" in out["genre_modules"]
