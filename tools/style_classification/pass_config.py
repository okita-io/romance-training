"""
Field sets for two-pass LLM classification (see source/multi-pass.md).
"""

from __future__ import annotations

from typing import Literal

PassMode = Literal["full", "fast", "deep", "both"]

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

# Semantic batches of ≤3 labels (joint mode). Order matters for within-pass prior.
PASS1_FIELD_BATCHES: tuple[tuple[str, ...], ...] = (
    ("lexical_complexity", "register", "figurative_density"),
    ("sentence_complexity", "pov", "cohesion"),
    ("segmentation", "prose_rhythm", "end_focus"),
    ("subordination_salience", "textual_relations", "climax"),
)

PASS2_FIELD_BATCHES: tuple[tuple[str, ...], ...] = (
    ("tone", "narrative_distance", "mind_style"),
    ("free_indirect_discourse",),
)

ALL_FIELD_BATCHES: tuple[tuple[str, ...], ...] = PASS1_FIELD_BATCHES + PASS2_FIELD_BATCHES

# Default: split joint prompts into the batches above. 0 = one call per pass (legacy).
DEFAULT_FIELD_BATCH_SIZE = 3


def fields_for_pass(pass_mode: PassMode) -> frozenset[str] | None:
    if pass_mode == "fast":
        return PASS1_LLM_FIELDS
    if pass_mode == "deep":
        return PASS2_LLM_FIELDS
    return None


def batches_for_pass(
    pass_mode: PassMode,
    *,
    field_batch_size: int = DEFAULT_FIELD_BATCH_SIZE,
) -> list[frozenset[str]]:
    """
    Return ordered field batches for joint LLM calls.

    ``field_batch_size <= 0`` → one batch covering the whole pass (legacy joint).
    ``field_batch_size >= 3`` → use the predefined semantic batches of ≤3.
    ``1`` or ``2`` → chunk the pass fields into groups of that size (rare).
    """
    if pass_mode == "fast":
        batches = PASS1_FIELD_BATCHES
        all_fields = PASS1_LLM_FIELDS
    elif pass_mode == "deep":
        batches = PASS2_FIELD_BATCHES
        all_fields = PASS2_LLM_FIELDS
    else:
        # full / both treated as all fields when called for a single assess
        batches = ALL_FIELD_BATCHES
        all_fields = ALL_LLM_FIELDS

    if field_batch_size <= 0:
        return [frozenset(all_fields)]

    if field_batch_size >= 3:
        return [frozenset(b) for b in batches]

    # Arbitrary chunking for size 1–2
    ordered = [f for batch in batches for f in batch]
    out: list[frozenset[str]] = []
    for i in range(0, len(ordered), field_batch_size):
        out.append(frozenset(ordered[i : i + field_batch_size]))
    return out


def pass_complete(profile: dict, pass_mode: PassMode) -> bool:
    """True when all LLM fields for this pass are present in profile."""
    if pass_mode in ("full", "both"):
        return bool(profile) and all(profile.get(f) is not None for f in ALL_LLM_FIELDS)
    required = PASS1_LLM_FIELDS if pass_mode == "fast" else PASS2_LLM_FIELDS
    return bool(profile) and all(profile.get(f) is not None for f in required)


def suggested_workers(pass_mode: PassMode) -> int | None:
    if pass_mode in ("fast", "both"):
        return 4
    if pass_mode == "deep":
        return 2
    return None
