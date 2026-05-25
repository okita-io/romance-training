"""Tests for genre-driven world modules (plan: genre-driven world modules)."""

from __future__ import annotations

import pytest

from romance_factory.generate.world_modules import (
    default_module_definitions,
    format_modules_for_character_web,
    format_modules_for_external_conflict,
    format_modules_for_story_outline,
    module_prompt_instructions,
    module_skeleton_block,
    normalize_genre_modules,
    resolve_active_modules,
    resolve_module_definitions,
    validate_modules,
)
from romance_factory.generate.world_setting_catalog import (
    default_world_setting_catalog,
)


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


def test_packaged_modules_cover_all_catalog_module_ids() -> None:
    """Every catalog-declared module id resolves to a registered module."""
    mods = default_module_definitions()
    catalog = default_world_setting_catalog()
    declared: set[str] = set()
    for entry in catalog.values():
        for mid in entry.get("module_ids") or []:
            declared.add(str(mid))
    missing = declared - set(mods.keys())
    assert not missing, f"catalog declares unregistered modules: {sorted(missing)}"


def test_every_catalog_entry_has_module_ids_field() -> None:
    """Every world catalog entry exposes a module_ids list (may be empty in principle)."""
    catalog = default_world_setting_catalog()
    for key, entry in catalog.items():
        assert "module_ids" in entry, f"catalog entry {key!r} missing module_ids"
        mids = entry["module_ids"]
        assert isinstance(mids, list)
        assert all(isinstance(m, str) and m for m in mids)


def test_default_module_definitions_is_deep_copy() -> None:
    a = default_module_definitions()
    b = default_module_definitions()
    assert set(a.keys()) == set(b.keys())
    assert a is not b


def test_every_module_has_required_keys_in_skeleton() -> None:
    """required_keys must be a subset of json_skeleton keys."""
    mods = default_module_definitions()
    for mid, m in mods.items():
        skel_keys = set(m.json_skeleton.keys())
        for req in m.required_keys:
            assert req in skel_keys, (
                f"module {mid}: required_keys field {req!r} "
                f"not in json_skeleton"
            )


# ---------------------------------------------------------------------------
# Active module resolution (union, hybrid, subtype)
# ---------------------------------------------------------------------------


def test_resolve_active_modules_no_inputs_returns_empty() -> None:
    catalog = default_world_setting_catalog()
    assert resolve_active_modules(None, None, catalog) == []
    assert resolve_active_modules("", "", catalog) == []


def test_resolve_active_modules_world_only() -> None:
    catalog = default_world_setting_catalog()
    ids = resolve_active_modules(None, "fantasy", catalog)
    assert "magic_system" in ids
    assert "religion_pantheon" in ids


def test_resolve_active_modules_genre_only_with_prefix_fallback() -> None:
    """paranormal_vampire resolves to paranormal modules via prefix fallback."""
    catalog = default_world_setting_catalog()
    ids = resolve_active_modules("paranormal_vampire", None, catalog)
    assert "species_lore" in ids
    assert "supernatural_law" in ids


def test_resolve_active_modules_union_of_genre_and_world() -> None:
    """--genre paranormal_vampire --world sci_fi unions both module sets."""
    catalog = default_world_setting_catalog()
    ids = resolve_active_modules("paranormal_vampire", "sci_fi", catalog)
    assert "species_lore" in ids
    assert "supernatural_law" in ids
    assert "tech_stack" in ids
    assert "ftl_travel" in ids


def test_resolve_active_modules_union_is_idempotent() -> None:
    catalog = default_world_setting_catalog()
    once = resolve_active_modules("paranormal", "sci_fi", catalog)
    assert len(once) == len(set(once)), "union must deduplicate"


def test_resolve_active_modules_hybrid_syntax() -> None:
    catalog = default_world_setting_catalog()
    ids = resolve_active_modules(None, "hybrid:sci_fi+paranormal", catalog)
    for required in ("tech_stack", "ftl_travel", "supernatural_law", "species_lore"):
        assert required in ids, f"hybrid missing {required}"


def test_subtype_filter_small_town_drops_urban_geography() -> None:
    catalog = default_world_setting_catalog()
    ids = resolve_active_modules("contemporary_small_town", "contemporary", catalog)
    assert "small_town_social" in ids
    assert "urban_geography" not in ids


def test_subtype_filter_big_city_drops_small_town_social() -> None:
    catalog = default_world_setting_catalog()
    ids = resolve_active_modules("contemporary_big_city", "contemporary", catalog)
    assert "urban_geography" in ids
    assert "small_town_social" not in ids


def test_subtype_filter_no_signal_keeps_both() -> None:
    catalog = default_world_setting_catalog()
    ids = resolve_active_modules(None, "contemporary", catalog)
    assert "urban_geography" in ids
    assert "small_town_social" in ids


def test_urban_fantasy_does_not_trigger_big_city_filter() -> None:
    """'urban' inside 'urban_fantasy' must not be treated as a big_city signal."""
    catalog = default_world_setting_catalog()
    ids = resolve_active_modules(None, "urban_fantasy", catalog)
    assert "small_town_social" not in ids or "urban_geography" in ids


def test_free_form_world_resolves_no_modules() -> None:
    catalog = default_world_setting_catalog()
    ids = resolve_active_modules(None, "underwater casino on Europa", catalog)
    assert ids == []


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def test_module_skeleton_block_is_valid_json_fragment() -> None:
    """Emitted block should parse when placed inside a top-level JSON object."""
    import json

    block = module_skeleton_block(["tech_stack", "magic_system"])
    wrapped = "{\n" + block + "\n}"
    parsed = json.loads(wrapped)
    assert "genre_modules" in parsed
    assert "tech_stack" in parsed["genre_modules"]
    assert "magic_system" in parsed["genre_modules"]
    assert "tech_level" in parsed["genre_modules"]["tech_stack"]


def test_module_skeleton_block_empty_when_no_ids() -> None:
    assert module_skeleton_block([]) == ""


def test_module_prompt_instructions_lists_each_module() -> None:
    text = module_prompt_instructions(["tech_stack", "small_town_social"])
    assert "TECH STACK" in text
    assert "SMALL TOWN" in text


# ---------------------------------------------------------------------------
# Normalization + validation
# ---------------------------------------------------------------------------


def _valid_tech_stack_payload() -> dict:
    return {
        "tech_level": "late interplanetary",
        "power_sources": "fusion cores and orbital microwave grids",
        "transport": "commuter shuttles, torch-drive freighters, orbital elevators",
        "daily_tech_examples": "wrist-comms, ration synths, atmosphere scrubbers",
        "failure_modes": "brownouts on the station ring, smuggled bootleg mods",
    }


def test_validate_modules_accepts_well_formed_payload() -> None:
    gm = {"tech_stack": _valid_tech_stack_payload()}
    issues = validate_modules(gm, ["tech_stack"])
    assert issues == []


def test_validate_modules_rejects_missing_required_field() -> None:
    payload = _valid_tech_stack_payload()
    payload.pop("tech_level")
    gm = {"tech_stack": payload}
    issues = validate_modules(gm, ["tech_stack"])
    assert any("tech_level" in msg for msg in issues)


def test_validate_modules_rejects_placeholder_text() -> None:
    payload = _valid_tech_stack_payload()
    payload["tech_level"] = "TBD"
    gm = {"tech_stack": payload}
    issues = validate_modules(gm, ["tech_stack"])
    assert any("tech_level" in msg for msg in issues)


def test_validate_modules_rejects_missing_module() -> None:
    issues = validate_modules({}, ["tech_stack"])
    assert any("tech_stack" in msg for msg in issues)


def test_validate_modules_no_active_ids_is_noop() -> None:
    """Validator is permissive when no active modules were requested."""
    assert validate_modules({}, []) == []
    assert validate_modules(None, []) == []


def test_normalize_genre_modules_strips_placeholder_text() -> None:
    out = normalize_genre_modules(
        {
            "tech_stack": {
                "tech_level": "(Placeholder — refine later.)",
                "power_sources": "fusion cores",
                "transport": "shuttles",
                "daily_tech_examples": "wrist comms",
                "failure_modes": "brownouts",
            },
        },
        ["tech_stack"],
    )
    assert out["tech_stack"]["tech_level"] == ""
    assert out["tech_stack"]["power_sources"] == "fusion cores"


def test_normalize_genre_modules_adds_empty_entry_for_missing_module() -> None:
    out = normalize_genre_modules({}, ["tech_stack"])
    assert "tech_stack" in out
    assert all(v == "" for v in out["tech_stack"].values())


# ---------------------------------------------------------------------------
# Downstream formatters
# ---------------------------------------------------------------------------


def test_format_modules_for_character_web_emits_configured_modules_only() -> None:
    gm = {
        "tech_stack": _valid_tech_stack_payload(),
        "ftl_travel": {
            "ftl_method": "folded jump corridors",
            "travel_time_model": "subjective hours for weeks of chart time",
            "limitations": "licensed pilots only; shard instability",
            "infrastructure": "gate towers at every Lagrange point",
        },
    }
    # ftl_travel is NOT in the character_web downstream_roles set by default.
    text = format_modules_for_character_web(gm)
    assert "Tech stack" in text
    assert "FTL" not in text


def test_format_modules_for_external_conflict_emits_tech_stack() -> None:
    gm = {"tech_stack": _valid_tech_stack_payload()}
    text = format_modules_for_external_conflict(gm)
    assert "Tech stack" in text
    assert "fusion cores" in text


def test_format_modules_for_story_outline_emits_small_town() -> None:
    gm = {
        "small_town_social": {
            "local_shops_and_hangouts": "Mabel's Diner; Harbor Ice Cream; Kline's Hardware",
            "annual_events": "Founders' Day parade; Harvest Dance",
            "gossip_network": "runs through the stylist at Ruth's Cut",
            "institutional_cornerstones": "First Methodist, the VFW hall",
            "outsider_dynamics": "newcomers watched for a full season",
        },
    }
    text = format_modules_for_story_outline(gm)
    assert "Small-town social fabric" in text or "Small Town" in text.title()
    assert "Mabel's Diner" in text


def test_format_modules_empty_when_genre_modules_absent() -> None:
    assert format_modules_for_character_web(None) == ""
    assert format_modules_for_external_conflict({}) == ""
    assert format_modules_for_story_outline({"unknown_module": {"x": "y"}}) == ""


# ---------------------------------------------------------------------------
# YAML override surface
# ---------------------------------------------------------------------------


def test_resolve_module_definitions_patches_existing_module() -> None:
    mods = resolve_module_definitions(
        {
            "tech_stack": {
                "label": "Custom tech",
                "prompt_instructions": "Only describe transport.",
            },
        },
    )
    assert mods["tech_stack"].label == "Custom tech"
    assert mods["tech_stack"].prompt_instructions == "Only describe transport."
    # Non-overridden fields should survive.
    assert "tech_level" in mods["tech_stack"].json_skeleton


def test_resolve_module_definitions_adds_new_module() -> None:
    mods = resolve_module_definitions(
        {
            "cyberdeck_protocols": {
                "label": "Cyberdeck protocols",
                "json_skeleton": {
                    "deck_types": "The hardware tiers",
                    "ice_rules": "What happens when ICE bites back",
                },
                "required_keys": ["deck_types", "ice_rules"],
                "prompt_instructions": "Describe deck tiers and ICE rules.",
            },
        },
    )
    assert "cyberdeck_protocols" in mods
    assert mods["cyberdeck_protocols"].label == "Cyberdeck protocols"
    assert mods["cyberdeck_protocols"].required_keys == ("deck_types", "ice_rules")
