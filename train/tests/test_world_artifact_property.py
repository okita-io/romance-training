"""Property tests for World Artifact ``parsed_data`` (world.json schema).

Property 1 (schema): Hypothesis-generated dicts match
``PipelineV2._validate_world_parsed_data`` — required keys, supporting_cast
shape, world_state enum, world_pressure_types, non-empty world_type.

Validates Requirements 3.4, 6.1, 12.1–12.5.

Property 2 (round-trip): JSON serialize → deserialize preserves the object.

Validates Requirement 12.7.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from romance_factory.generate.pipeline_v2 import PipelineV2

_WORLD_STATES = ("stable", "threatened", "anomalous", "decaying")

# Text safe for UTF-8 JSON (no lone surrogates). Pipeline validation uses
# ``str(...).strip()`` — exclude whitespace-only strings Hypothesis can draw.
_safe_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=256,
).filter(lambda s: s.strip())

# Pipeline rejects substrings matching ``_WORLD_PLACEHOLDER_MARKERS`` on these fields.
_no_world_placeholder = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=256,
).filter(
    lambda s: s.strip() and not PipelineV2._world_field_looks_like_placeholder(s),
)

_supporting_cast_member = st.fixed_dictionaries(
    {
        "name": _safe_text,
        "role_in_world": _no_world_placeholder,
        "brief_description": _no_world_placeholder,
    }
)

_world_artifact_parsed_data = st.fixed_dictionaries(
    {
        "setting_design": _no_world_placeholder,
        "lore": _no_world_placeholder,
        "cosmology": _no_world_placeholder,
        "culture": _no_world_placeholder,
        "technology_or_era": _no_world_placeholder,
        "environmental_logic": _no_world_placeholder,
        "supporting_cast": st.lists(_supporting_cast_member, min_size=3, max_size=8),
        "world_state": st.sampled_from(_WORLD_STATES),
        "world_pressure_types": st.lists(_safe_text, min_size=1, max_size=8),
        "world_type": _no_world_placeholder,
    }
)


@given(d=_world_artifact_parsed_data)
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_world_artifact_parsed_data_schema_validation(d: dict) -> None:
    """Every generated artifact satisfies pipeline world.json schema checks."""
    assert PipelineV2._validate_world_parsed_data(d) is True


@given(d=_world_artifact_parsed_data)
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_world_artifact_parsed_data_json_round_trip(d: dict) -> None:
    """Serialize valid world artifact dicts to JSON and back; value must match."""
    dumped = json.dumps(d, ensure_ascii=False)
    assert json.loads(dumped) == d
