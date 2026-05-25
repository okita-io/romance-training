"""Property tests for world-driven external conflict prompt context (THE-158 / Req 4.3–4.4).

Property 8: Non-stable world_state drives escalation language in the conflict prompt block.
Property 9: Hybrid world_type layers pressures from both constituent catalog types.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from romance_factory.generate.world_setting_catalog import (
    build_world_context_for_conflict_prompt,
)

pytest.importorskip("hypothesis")

# Printable slug tokens (aligned with other world property tests).
_slug = st.text(
    alphabet=st.characters(min_codepoint=0x61, max_codepoint=0x7A),
    min_size=3,
    max_size=12,
).filter(lambda s: s.isalpha() and s == s.lower())

_non_stable_world_state = st.sampled_from(("threatened", "anomalous", "decaying"))


def _minimal_world_artifact(
    *,
    world_state: str = "stable",
    world_type: str = "fantasy",
    world_pressure_types: list[str] | None = None,
    lore: str = "Synthetic lore for tests.",
    culture: str = "Synthetic culture for tests.",
) -> dict:
    return {
        "lore": lore,
        "culture": culture,
        "world_state": world_state,
        "world_type": world_type,
        "world_pressure_types": list(world_pressure_types or []),
    }


@settings(max_examples=40)
@given(world_state=_non_stable_world_state, lore=_slug, culture=_slug)
def test_property_non_stable_world_state_includes_escalation_driver(
    world_state,
    lore,
    culture,
):
    """Property 8: threatened / anomalous / decaying world_state adds escalation driver text."""
    world = _minimal_world_artifact(
        world_state=world_state,
        world_type="fantasy",
        lore=lore,
        culture=culture,
    )
    block = build_world_context_for_conflict_prompt(world, catalog={})
    assert "ESCALATION DRIVER" in block
    assert "escalation_beats" in block
    assert world_state in block


@settings(max_examples=40)
@given(key_a=_slug, key_b=_slug, p_a=_slug, p_b=_slug)
def test_property_hybrid_world_layers_pressures_from_both_constituents(
    key_a,
    key_b,
    p_a,
    p_b,
):
    """Property 9: hybrid:a+b includes catalog pressures from both types and hybrid guidance."""
    assume(key_a != key_b)
    assume(p_a != p_b)

    catalog = {
        key_a: {
            "name": "A",
            "setting_subtypes": [],
            "world_pressure_types": [p_a],
            "example_environments": [],
        },
        key_b: {
            "name": "B",
            "setting_subtypes": [],
            "world_pressure_types": [p_b],
            "example_environments": [],
        },
    }
    world = _minimal_world_artifact(
        world_state="stable",
        world_type=f"hybrid:{key_a}+{key_b}",
        world_pressure_types=[],
    )
    block = build_world_context_for_conflict_prompt(world, catalog=catalog)

    assert "HYBRID WORLD" in block
    assert "BOTH" in block
    assert f"  - {p_a}" in block
    assert f"  - {p_b}" in block
