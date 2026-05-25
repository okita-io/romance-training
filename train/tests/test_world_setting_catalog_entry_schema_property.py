"""Property test: World_Setting_Catalog entry schema (Req. 11.3 / THE-150).

*For any* randomly generated catalog-shaped entry, required keys exist and have
the correct Python types (``name`` str; three list-of-str fields).
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_REQUIRED_ENTRY_KEYS = frozenset(
    {
        "name",
        "setting_subtypes",
        "world_pressure_types",
        "example_environments",
    }
)

# UTF-8 JSON–safe text; exclude lone surrogates. Require non-whitespace content
# so ``.strip()`` checks match real catalog entries.
_non_empty_safe_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=128,
).filter(lambda s: bool(s.strip()))

_world_setting_catalog_entry = st.fixed_dictionaries(
    {
        "name": _non_empty_safe_text,
        "setting_subtypes": st.lists(_non_empty_safe_text, min_size=1, max_size=16),
        "world_pressure_types": st.lists(_non_empty_safe_text, min_size=1, max_size=16),
        "example_environments": st.lists(_non_empty_safe_text, min_size=1, max_size=16),
    }
)


def _assert_entry_matches_world_setting_catalog_schema(entry: dict) -> None:
    assert set(entry.keys()) == _REQUIRED_ENTRY_KEYS
    assert isinstance(entry["name"], str) and entry["name"].strip()
    for key in (
        "setting_subtypes",
        "world_pressure_types",
        "example_environments",
    ):
        val = entry[key]
        assert isinstance(val, list) and val
        assert all(isinstance(x, str) and x.strip() for x in val)


@given(entry=_world_setting_catalog_entry)
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_random_catalog_entry_validates_schema(entry: dict) -> None:
    _assert_entry_matches_world_setting_catalog_schema(entry)
