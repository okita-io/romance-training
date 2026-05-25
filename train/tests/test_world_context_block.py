"""Tests for World Context Block compact schema helpers."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from romance_factory.generate.world_context_block import (
    WCB_LAYER_KEYS,
    build_world_context_block_prompt,
    normalize_world_context_block,
    validate_world_context_block,
    wcb_is_compact_schema,
    world_context_block_embedding_text,
)

_MIN_WORLD = {
    "setting_design": "Coastal inn",
    "environmental_logic": "Salt air, damp wood",
    "culture": "Small-town politeness",
    "world_state": "stable",
}


def test_validate_accepts_full_wcb() -> None:
    wcb = normalize_world_context_block(
        None,
        world_artifact=_MIN_WORLD,
        scene_summary="They argue by the pier",
        chapter_number=2,
        act_number=1,
    )
    ok, reasons = validate_world_context_block(wcb)
    assert ok, reasons
    for layer in WCB_LAYER_KEYS:
        assert layer in wcb
    assert wcb_is_compact_schema(wcb)
    assert "location_ids" in wcb["macro_world"]
    assert wcb["meso_world"]["venue_ids"]
    assert wcb["micro_world"]["stages"]


def test_validate_rejects_missing_layer() -> None:
    wcb = normalize_world_context_block(
        None, world_artifact=_MIN_WORLD, scene_summary="x", chapter_number=1, act_number=1,
    )
    del wcb["micro_characters"]
    ok, reasons = validate_world_context_block(wcb)
    assert not ok
    assert any("micro_characters" in r for r in reasons)


def test_normalize_fills_from_world_artifact() -> None:
    wcb = normalize_world_context_block(
        {"macro_world": {"world_change": ""}},
        world_artifact=_MIN_WORLD,
        scene_summary="",
        chapter_number=1,
        act_number=3,
        characters_involved=["Morgan"],
    )
    assert isinstance(wcb["macro_world"]["location_ids"], list)
    assert wcb["macro_world"]["location_ids"]
    assert "Morgan" in wcb["macro_world"]["involved"]
    assert wcb["meso_story"]["arc_plot_changes"]
    assert wcb["micro_characters"]["character_state"]


def test_normalize_migrates_legacy_six_layer() -> None:
    legacy = {
        "macro_world": {
            "location": "Pier",
            "environmental_logic": "Wind",
            "cultural_operational_norms": "",
            "world_state": "stable",
        },
        "meso_environment": {
            "spatial_layout": "Open deck",
            "key_objects": "",
            "ambient_sensory_field": {"sound": "gulls", "scent": "", "temperature": "", "light": "", "texture": ""},
        },
        "micro_interaction": {"physical_constraints": "", "environmental_touchpoints": "", "proximity_architecture": ""},
        "diegetic_information": {"readouts_displays": "", "environmental_signals": "", "diegetic_media": ""},
        "emotional_atmosphere": {"atmospheric_tone": "tense", "thematic_echo": "", "emotional_topography": ""},
        "narrative_function": {
            "scene_purpose": "Fight",
            "world_stakes": "",
            "state_change": "",
            "character_involvement": "A and B",
        },
    }
    wcb = normalize_world_context_block(
        legacy,
        world_artifact=_MIN_WORLD,
        scene_summary="Argue",
        chapter_number=1,
        act_number=1,
    )
    assert wcb_is_compact_schema(wcb)
    assert wcb["macro_world"]["location_ids"]
    assert "A and B" in wcb["macro_world"]["involved"]
    assert wcb["micro_world"]["stages"][0]["stage_slug"]


def test_normalize_uses_chapter_geo_defaults() -> None:
    ch = {
        "chapter_number": 1,
        "macro_world": {
            "locations": [
                {
                    "location_id": "loc-testville",
                    "label": "Testville",
                    "role_in_chapter": "r",
                    "venues": ["ven-test-hall"],
                },
            ],
        },
    }
    wcb = normalize_world_context_block(
        {
            "macro_world": {"involved": "Pat", "world_change": "c", "why_involved": "w"},
            "meso_story": {"arc_plot_changes": "None."},
            "micro_characters": {"character_state": "Unchanged."},
        },
        world_artifact=_MIN_WORLD,
        scene_summary="s",
        chapter_number=1,
        act_number=2,
        characters_involved=["Pat"],
        chapter_data=ch,
    )
    assert wcb["macro_world"]["location_ids"] == ["loc-testville"]
    assert wcb["meso_world"]["venue_ids"] == ["ven-test-hall"]
    assert wcb["micro_world"]["stages"][0]["venue_id"] == "ven-test-hall"


def test_build_prompt_contains_scene_ids() -> None:
    user, system = build_world_context_block_prompt(
        world_json_text="{}",
        story_arc_text="{}",
        chapter_number=4,
        chapter_title="Storm",
        chapter_summary="Rain",
        act_number=2,
        act_summary="First kiss",
        characters_involved=["A", "B"],
        emotional_tone="longing",
        plot_function="bonding",
    )
    assert "Chapter 4" in user
    assert "Act 2" in user
    assert "First kiss" in user
    assert "meso_story" in user
    assert "location_ids" in user
    assert "meso_world" in user
    assert "valid JSON" in system


def test_embedding_text_is_sorted_json() -> None:
    wcb = {"z": 1, "a": {"b": 2}}
    s = world_context_block_embedding_text(wcb)
    parsed = json.loads(s)
    assert list(parsed.keys()) == sorted(parsed.keys())


@given(
    inv=st.text(min_size=1, max_size=80),
    arc=st.text(min_size=1, max_size=120),
    micro=st.text(min_size=1, max_size=120),
)
def test_property_normalized_wcb_always_valid(
    inv: str,
    arc: str,
    micro: str,
) -> None:
    world = {**_MIN_WORLD, "setting_design": "Somewhere coastal"}
    wcb = normalize_world_context_block(
        {
            "macro_world": {
                "involved": inv,
                "world_change": "x",
                "why_involved": "y",
            },
            "meso_story": {"arc_plot_changes": arc},
            "micro_characters": {"character_state": micro},
        },
        world_artifact=world,
        scene_summary=arc,
        chapter_number=1,
        act_number=1,
    )
    ok, reasons = validate_world_context_block(wcb)
    assert ok, reasons


_non_empty_str = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())


@st.composite
def random_valid_wcb_dicts(draw: st.DrawFn) -> dict:
    """Hypothesis-built dicts that satisfy validate_world_context_block."""
    vid = "ven-" + draw(_non_empty_str)[:24].replace(" ", "-")
    lid = "loc-" + draw(_non_empty_str)[:24].replace(" ", "-")
    ch_name = draw(_non_empty_str)[:40]
    return {
        "macro_world": {
            "involved": draw(_non_empty_str),
            "location_ids": [lid],
            "world_change": draw(_non_empty_str),
            "why_involved": draw(_non_empty_str),
        },
        "meso_world": {
            "venue_ids": [vid],
        },
        "micro_world": {
            "stages": [
                {
                    "stage_id": "stg-ch1-act1-001",
                    "venue_id": vid,
                    "stage_slug": "room-a",
                    "characters": [ch_name],
                    "contains": [],
                },
            ],
        },
        "meso_story": {
            "arc_plot_changes": draw(_non_empty_str),
        },
        "micro_characters": {
            "character_state": draw(_non_empty_str),
        },
    }


@settings(max_examples=60)
@given(wcb=random_valid_wcb_dicts())
def test_property_random_wcb_passes_schema_validation(wcb: dict) -> None:
    ok, reasons = validate_world_context_block(wcb)
    assert ok, reasons

    for layer in WCB_LAYER_KEYS:
        assert layer in wcb
        assert isinstance(wcb[layer], dict)


@contextmanager
def _pipeline_v2_light(story_path: str, tmp_path):
    from romance_factory.generate.config_v2 import V2Config
    from romance_factory.generate.pipeline_v2 import PipelineV2

    cfg = V2Config(
        story_path=story_path,
        db_path=str(tmp_path / "lancedb"),
        embedding_model="mock",
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


def test_sync_outline_preserves_world_context_block_on_beat_replace(tmp_path) -> None:
    from romance_factory.generate.models import DocumentMetadata, RetrievalResult

    sp = str(tmp_path / "story")
    tmp_path.joinpath("story").mkdir(parents=True)

    wcb = normalize_world_context_block(
        None,
        world_artifact={**_MIN_WORLD, "setting_design": "Hall"},
        scene_summary="Meet",
        chapter_number=1,
        act_number=1,
    )
    outline = {
        "story_arc": {"premise": "p"},
        "chapters": [
            {
                "chapter_number": 1,
                "title": "T",
                "acts": [
                    {
                        "act_number": 1,
                        "summary": "old",
                        "world_context_block": wcb,
                    },
                ],
            },
        ],
    }
    outline_path = tmp_path / "story" / "story_outline.json"
    outline_path.write_text(
        json.dumps(
            {
                "artifact_type": "story_outline",
                "text": json.dumps(outline),
                "metadata": {"type": "outline", "summary": "x"},
                "created_at": "2020-01-01T00:00:00+00:00",
                "file_path": str(outline_path),
                "parsed_data": outline,
            },
        ),
        encoding="utf-8",
    )

    replacement_text = json.dumps(
        {
            "act_number": 1,
            "summary": "new summary from editorial",
            "characters_involved": ["A"],
            "emotional_tone": "warm",
            "plot_function": "bonding",
            "foreshadowing": [],
            "is_plot_twist": False,
        },
    )
    beat_row = RetrievalResult(
        text=replacement_text,
        metadata=DocumentMetadata(
            type="beat",
            chapter=1,
            act=1,
            characters_involved=["A"],
            emotional_tone="warm",
            plot_function="bonding",
            summary="new",
        ),
        similarity_score=1.0,
    )

    with _pipeline_v2_light(sp, tmp_path) as pipeline:
        pipeline.engine.query.return_value = [beat_row]
        pipeline._sync_story_outline_json_from_lancedb_beats()

    saved = json.loads(outline_path.read_text(encoding="utf-8"))
    acts = saved["parsed_data"]["chapters"][0]["acts"]
    assert acts[0]["summary"] == "new summary from editorial"
    assert acts[0]["world_context_block"] == wcb
