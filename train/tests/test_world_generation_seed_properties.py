"""Property tests for world generation seed construction (THE-156 / Req 3.1–3.3, 10.4–10.5).

Uses Hypothesis to validate merge and seed-building invariants across random catalog data.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from romance_factory.generate.config_v2 import V2Config
from romance_factory.generate.pipeline_v2 import PipelineV2


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


# Printable tokens without newlines (stable for prompt substring checks).
_token = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E, blacklist_characters="\\"),
    min_size=4,
    max_size=24,
).filter(lambda s: s.strip() == s and "," not in s)

_non_empty_list = st.lists(_token, min_size=1, max_size=5, unique=True)


def _world_catalog_entry(
    subtype_tokens: list[str],
    pressure_tokens: list[str],
    env_tokens: list[str],
    archetype_tokens: list[str],
) -> dict:
    return {
        "name": "Synthetic World",
        "setting_subtypes": list(subtype_tokens),
        "world_pressure_types": list(pressure_tokens),
        "example_environments": list(env_tokens),
        "supporting_character_archetypes": list(archetype_tokens),
    }


@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    genre_key=_token,
    world_key=_token,
    themes=_non_empty_list,
    tropes=_non_empty_list,
    subtypes=_non_empty_list,
    pressures=_non_empty_list,
    environments=_non_empty_list,
    archetypes=_non_empty_list,
)
def test_property_genre_and_world_merge_contains_both(
    tmp_path,
    genre_key,
    world_key,
    themes,
    tropes,
    subtypes,
    pressures,
    environments,
    archetypes,
):
    """Property 3: merged seed includes genre catalog content and world catalog content."""
    gk = genre_key.lower().replace("-", "_")
    wk = world_key.lower().replace("-", "_")
    assume(gk != wk)

    genre_catalog = {
        gk: {
            "name": f"Genre {gk}",
            "themes": themes,
            "trope_pool": tropes,
            "world": ["ignored_when_world_flag_set"],
            "atmosphere": "test atmosphere",
        },
    }
    world_catalog = {wk: _world_catalog_entry(subtypes, pressures, environments, archetypes)}

    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        genre=gk,
        world=wk,
        genre_catalog=genre_catalog,
        world_setting_catalog=world_catalog,
    ) as p:
        seed, resolved, merged_pressures = p._build_world_generation_seed()

    assert resolved == wk
    for t in themes:
        assert t in seed
    for tr in tropes:
        assert tr in seed
    for s in subtypes:
        assert s in seed
    for e in environments[:8]:
        assert e in seed
    for pr in pressures:
        assert pr in merged_pressures


@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    genre_key=_token,
    motifs=_non_empty_list,
    themes=_non_empty_list,
    tropes=_non_empty_list,
    world_key=_token,
    subtypes=_non_empty_list,
    pressures=_non_empty_list,
    environments=_non_empty_list,
    archetypes=_non_empty_list,
)
def test_property_genre_motifs_surface_when_genre_and_world_set(
    tmp_path,
    genre_key,
    motifs,
    themes,
    tropes,
    world_key,
    subtypes,
    pressures,
    environments,
    archetypes,
):
    """Genre catalog `world` motifs remain visible when --world also pins a catalog type."""
    gk = genre_key.lower().replace("-", "_")
    wk = world_key.lower().replace("-", "_")
    assume(gk != wk)

    genre_catalog = {
        gk: {
            "name": f"Genre {gk}",
            "themes": themes,
            "trope_pool": tropes,
            "world": motifs,
        },
    }
    world_catalog = {wk: _world_catalog_entry(subtypes, pressures, environments, archetypes)}

    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        genre=gk,
        world=wk,
        genre_catalog=genre_catalog,
        world_setting_catalog=world_catalog,
    ) as p:
        seed, resolved, _ = p._build_world_generation_seed()

    assert resolved == wk
    assert "World / motifs:" in seed or "themes" in seed.lower()
    for m in motifs:
        assert m in seed


@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    world_key=_token,
    subtypes=_non_empty_list,
    pressures=_non_empty_list,
    environments=_non_empty_list,
    archetypes=_non_empty_list,
)
def test_property_world_with_genre_includes_catalog_lines(
    tmp_path,
    world_key,
    subtypes,
    pressures,
    environments,
    archetypes,
):
    """With both genre and world set, seed includes genre header and world catalog lines."""
    wk = world_key.lower().replace("-", "_")
    world_catalog = {wk: _world_catalog_entry(subtypes, pressures, environments, archetypes)}

    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        genre="synth_genre",
        world=wk,
        genre_catalog={
            "synth_genre": {
                "name": "Synthetic",
                "themes": ["romance"],
                "trope_pool": ["enemies_to_lovers"],
            },
        },
        world_setting_catalog=world_catalog,
    ) as p:
        seed, resolved, merged_pressures = p._build_world_generation_seed()

    assert resolved == wk
    assert "STORY GENRE:" in seed
    for s in subtypes:
        assert s in seed
    for pr in pressures:
        assert pr in merged_pressures


@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    catalog_key=_token,
    freeform_body=_token,
    subtypes=_non_empty_list,
    pressures=_non_empty_list,
    environments=_non_empty_list,
    archetypes=_non_empty_list,
)
def test_property_free_form_world_verbatim_with_genre(
    tmp_path,
    catalog_key,
    freeform_body,
    subtypes,
    pressures,
    environments,
    archetypes,
):
    """Property 6 (with genre): unknown --world value appears verbatim in the seed."""
    ck = catalog_key.lower().replace("-", "_")
    # Ensure the free-form flag cannot normalize to an existing catalog key.
    raw_world = f"__freeform__{freeform_body}"
    assume(PipelineV2._normalize_world_setting_key(raw_world) != ck)

    genre_catalog = {
        "synth_genre": {
            "name": "Synthetic",
            "themes": ["t1"],
            "trope_pool": ["tp1"],
        },
    }
    world_catalog = {ck: _world_catalog_entry(subtypes, pressures, environments, archetypes)}

    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        genre="synth_genre",
        world=raw_world,
        genre_catalog=genre_catalog,
        world_setting_catalog=world_catalog,
    ) as p:
        seed, resolved, _ = p._build_world_generation_seed()

    assert raw_world in seed
    assert "free-form user description" in seed
    assert "free_form:" in resolved


@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    catalog_key=_token,
    freeform_body=_token,
    subtypes=_non_empty_list,
    pressures=_non_empty_list,
    environments=_non_empty_list,
    archetypes=_non_empty_list,
)
def test_property_free_form_world_verbatim_world_only(
    tmp_path,
    catalog_key,
    freeform_body,
    subtypes,
    pressures,
    environments,
    archetypes,
):
    """Unknown --world value appears verbatim when paired with a genre."""
    ck = catalog_key.lower().replace("-", "_")
    raw_world = f"__freeform__{freeform_body}"
    assume(PipelineV2._normalize_world_setting_key(raw_world) != ck)

    world_catalog = {ck: _world_catalog_entry(subtypes, pressures, environments, archetypes)}

    sp = str(tmp_path / "story")
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        genre="synth_slot",
        world=raw_world,
        genre_catalog={
            "synth_slot": {
                "name": "Synthetic",
                "themes": ["t1"],
                "trope_pool": ["tp1"],
            },
        },
        world_setting_catalog=world_catalog,
    ) as p:
        seed, resolved, _ = p._build_world_generation_seed()

    assert raw_world in seed
    assert "free-form user description" in seed
    assert "free_form:" in resolved


@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    key_a=_token,
    key_b=_token,
    themes=_non_empty_list,
    tropes=_non_empty_list,
    sub_a=_non_empty_list,
    pr_a=_non_empty_list,
    env_a=_non_empty_list,
    arch_a=_non_empty_list,
    sub_b=_non_empty_list,
    pr_b=_non_empty_list,
    env_b=_non_empty_list,
    arch_b=_non_empty_list,
)
def test_property_hybrid_world_includes_archetypes_from_both_constituents(
    tmp_path,
    key_a,
    key_b,
    themes,
    tropes,
    sub_a,
    pr_a,
    env_a,
    arch_a,
    sub_b,
    pr_b,
    env_b,
    arch_b,
):
    """Property 10 (THE-160 / Req 6.3): hybrid seed lists archetypes from both catalog types."""
    ka = key_a.lower().replace("-", "_")
    kb = key_b.lower().replace("-", "_")
    assume(ka != kb)
    assume(ka != "hybrid" and kb != "hybrid")
    # Hybrid parsing splits on ``+`` and ``,`` — those chars must not appear in keys.
    assume("+" not in ka and "+" not in kb)
    assume("," not in ka and "," not in kb)

    genre_catalog = {
        "synth_genre": {
            "name": "Synthetic",
            "themes": themes,
            "trope_pool": tropes,
        },
    }
    world_catalog = {
        ka: _world_catalog_entry(sub_a, pr_a, env_a, arch_a),
        kb: _world_catalog_entry(sub_b, pr_b, env_b, arch_b),
        "hybrid": _world_catalog_entry(["fusion"], ["stacked"], ["sprawl"], ["hybrid_broker"]),
    }

    sp = str(tmp_path / "story")
    hybrid_flag = f"hybrid:{ka}+{kb}"
    with _pipeline_v2_with_mocks(
        sp,
        tmp_path,
        genre="synth_genre",
        world=hybrid_flag,
        genre_catalog=genre_catalog,
        world_setting_catalog=world_catalog,
    ) as p:
        seed, resolved, _ = p._build_world_generation_seed()

    assert resolved == hybrid_flag
    assert "Supporting character archetypes" in seed
    for a in arch_a:
        assert a in seed
    for b in arch_b:
        assert b in seed
