"""
Field sets for two-pass LLM classification (see source/multi-pass.md).
"""

from __future__ import annotations

from typing import Literal

PassMode = Literal["full", "fast", "deep"]

# Pass 1 — small model: lexical, syntax (semantic), discourse, textual principles
PASS1_LLM_FIELDS: frozenset[str] = frozenset({
    "lexical_complexity",
    "register",
    "figurative_density",
    "sentence_complexity",
    "pov",
    "cohesion",
    "segmentation",
    "prose_rhythm",
    "end_focus",
    "subordination_salience",
    "textual_relations",
    "climax",
})

# Pass 2 — large model: tone + viewpoint (climax scored in pass 1)
PASS2_LLM_FIELDS: frozenset[str] = frozenset({
    "tone",
    "narrative_distance",
    "mind_style",
    "free_indirect_discourse",
})

ALL_LLM_FIELDS: frozenset[str] = PASS1_LLM_FIELDS | PASS2_LLM_FIELDS


def fields_for_pass(pass_mode: PassMode) -> frozenset[str] | None:
    if pass_mode == "fast":
        return PASS1_LLM_FIELDS
    if pass_mode == "deep":
        return PASS2_LLM_FIELDS
    return None


def pass_complete(profile: dict, pass_mode: PassMode) -> bool:
    """True when all LLM fields for this pass are present in profile."""
    if pass_mode == "full":
        return bool(profile) and all(profile.get(f) is not None for f in ALL_LLM_FIELDS)
    required = PASS1_LLM_FIELDS if pass_mode == "fast" else PASS2_LLM_FIELDS
    return bool(profile) and all(profile.get(f) is not None for f in required)


def suggested_workers(pass_mode: PassMode) -> int | None:
    if pass_mode == "fast":
        return 4
    if pass_mode == "deep":
        return 2
    return None
