"""Property tests for world context in downstream prompts (THE-167 / Req 4.1, 5.1, 7.1, 8.4).

Property 7: World Context Propagation — random world artifacts must surface a stable
unique token in conflict, character web, story outline, world-context-block generation,
and act-generation prompt assembly.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from romance_factory.generate.models import (
    DocumentMetadata,
    RetrievalResult,
    RetrievedContext,
)
from romance_factory.generate.prompt_builder import PromptBuilder
from romance_factory.generate.world_context_block import build_world_context_block_prompt
from romance_factory.generate.world_setting_catalog import (
    build_world_context_for_character_web_prompt,
    build_world_context_for_conflict_prompt,
    build_world_context_for_story_outline_prompt,
    default_world_setting_catalog,
)

pytest.importorskip("hypothesis")

# Alphanumeric markers avoid JSON / prompt edge cases and stay unique vs boilerplate.
_marker = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=12,
    max_size=24,
).filter(str.isalnum)


def _world_artifact(marker: str) -> dict:
    """Minimal world dict with *marker* woven into every builder-relevant field."""
    return {
        "setting_design": f"Central district known as {marker} Row",
        "lore": f"Legends of the {marker} founders",
        "culture": f"{marker} kinship rules and taboos",
        "cosmology": "Mundane physics",
        "technology_or_era": "Present day",
        "environmental_logic": f"{marker} sea winds and fog",
        "world_state": "stable",
        "world_type": "fantasy",
        "world_pressure_types": [f"{marker}_institutional"],
        "supporting_cast": [
            {
                "name": f"{marker}_guide",
                "role_in_world": "local fixer",
                "brief_description": f"Knows every alley in {marker} Row",
            },
        ],
    }


def _minimal_wcb_json(marker: str) -> str:
    """Valid compact WCB JSON with *marker* in multiple fields (PromptBuilder path)."""
    wcb = {
        "macro_world": {
            "involved": f"Leads at {marker} Hall",
            "location_ids": [f"loc-{marker}-hall"],
            "world_change": f"Stone and damp air; {marker} district rules apply",
            "why_involved": f"Establish stakes near {marker}",
        },
        "meso_world": {
            "venue_ids": [f"ven-{marker}-hall"],
        },
        "micro_world": {
            "stages": [
                {
                    "stage_id": "stg-ch1-act1-001",
                    "venue_id": f"ven-{marker}-hall",
                    "stage_slug": "main",
                    "characters": ["Lead A", "Lead B"],
                    "contains": [],
                },
            ],
        },
        "meso_story": {
            "arc_plot_changes": f"Central conflict touches {marker} reputation",
        },
        "micro_characters": {
            "character_state": f"Watchful mood; {marker} echo in posture",
        },
    }
    return json.dumps(wcb, ensure_ascii=False)


@settings(max_examples=30, deadline=None)
@given(marker=_marker)
def test_property_world_marker_propagates_to_all_downstream_prompts(marker: str) -> None:
    """Property 7: a unique world token appears in every downstream prompt surface."""
    world = _world_artifact(marker)
    catalog = default_world_setting_catalog()

    conflict_block = build_world_context_for_conflict_prompt(world, catalog=catalog)
    char_web_block = build_world_context_for_character_web_prompt(world, catalog=catalog)
    outline_block = build_world_context_for_story_outline_prompt(world, catalog=catalog)

    assert marker in conflict_block
    assert marker in char_web_block
    assert marker in outline_block

    world_json = json.dumps(world, ensure_ascii=False)
    user_wcb, _sys = build_world_context_block_prompt(
        world_json_text=world_json,
        story_arc_text="{}",
        chapter_number=1,
        chapter_title="Arrival",
        chapter_summary="Leads enter the district.",
        act_number=1,
        act_summary="First look at the hall.",
        characters_involved=["A", "B"],
        emotional_tone="curious",
        plot_function="setup",
    )
    assert marker in user_wcb
    assert "STORY WORLD (JSON):" in user_wcb

    wcb_json = _minimal_wcb_json(marker)
    builder = PromptBuilder(max_context_chars=32000)
    ctx = RetrievedContext(
        author_profile=[
            RetrievalResult(
                text="Author voice sample.",
                metadata=DocumentMetadata(type="author_profile"),
                similarity_score=0.9,
            ),
        ],
        character_web=[
            RetrievalResult(
                text="Lead A and Lead B.",
                metadata=DocumentMetadata(type="character_web"),
                similarity_score=0.9,
            ),
        ],
        world=[
            RetrievalResult(
                text=world_json,
                metadata=DocumentMetadata(type="world"),
                similarity_score=0.95,
            ),
        ],
        world_outline=[
            RetrievalResult(
                text=wcb_json,
                metadata=DocumentMetadata(type="world_context_block"),
                similarity_score=0.95,
            ),
        ],
        story_outline=[
            RetrievalResult(
                text="Chapter beats placeholder.",
                metadata=DocumentMetadata(type="story_outline"),
                similarity_score=0.9,
            ),
        ],
        chapter_outline=[
            RetrievalResult(
                text="Scene list placeholder.",
                metadata=DocumentMetadata(type="chapter_outline"),
                similarity_score=0.9,
            ),
        ],
        act_outline=[
            RetrievalResult(
                text="Act goal placeholder.",
                metadata=DocumentMetadata(type="act_outline"),
                similarity_score=0.9,
            ),
        ],
    )
    act_prompt, _ = builder.build_act_generation_prompt(
        chapter=1,
        act=1,
        context=ctx,
    )
    assert marker in act_prompt
    assert "## World Lore (retrieved)" in act_prompt
    assert "## World Context Block (continuity capsule)" in act_prompt
