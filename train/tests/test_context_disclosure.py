"""Tests for progressive context disclosure (phases 0–4).

Covers:
  - Phase computation logic (monotonic transitions, edge cases)
  - Retrieval gating (character web filtering, section suppression)
  - Prompt builder section inclusion/suppression per phase
"""

from __future__ import annotations

import pytest

from romance_factory.generate.context_disclosure import (
    OUTLINE_TIER_BOTH_NO_ROMANCE,
    OUTLINE_TIER_FULL,
    OUTLINE_TIER_PROTAGONIST,
    PHASE_BOTH_LEADS,
    PHASE_FULL,
    PHASE_LOVE_INTEREST,
    PHASE_PROTAGONIST,
    PHASE_WORLD_ONLY,
    clamp_disclosure_phase,
    compute_context_disclosure_phases,
    extract_lead_names_from_character_web,
    format_disclosure_phase_constraint,
    outline_chapter1_progressive_world_first_beat_note,
    outline_disclosure_tier,
    redact_arc_for_outline,
)
from romance_factory.generate.romance_arc_mechanics import (
    build_outline_romance_milestone_instruction_world_first,
)
from romance_factory.generate.models import (
    DocumentMetadata,
    RetrievalResult,
    RetrievedContext,
)
from romance_factory.generate.prompt_builder import PromptBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chapters(
    act_specs: list[tuple[int, int, str, list[str]]],
) -> list[dict]:
    """Build minimal chapters from (ch, act, rom_milestone, characters_involved)."""
    by_ch: dict[int, list[dict]] = {}
    for ch, act, rom, cast in act_specs:
        by_ch.setdefault(ch, []).append({
            "act_number": act,
            "romance_milestone": rom,
            "characters_involved": cast,
        })
    return [
        {"chapter_number": ch, "acts": acts}
        for ch, acts in sorted(by_ch.items())
    ]


def _make_result(text: str = "some context", **meta_kw) -> RetrievalResult:
    return RetrievalResult(
        text=text,
        metadata=DocumentMetadata(type="act", **meta_kw),
        similarity_score=0.9,
    )


def _make_context(**overrides) -> RetrievedContext:
    defaults = {
        "author_profile": [_make_result("author")],
        "character_web": [_make_result("Alice is the protagonist")],
        "world": [_make_result("world lore")],
        "world_outline": [_make_result("world context block")],
        "story_outline": [_make_result("story arc")],
        "chapter_outline": [_make_result("chapter outline")],
        "act_outline": [_make_result('{"act_number":1,"summary":"test"}')],
        "foreshadowing": [_make_result("foreshadow")],
        "relationship_arcs": [_make_result("arcs")],
        "previous_acts": [],
        "planned_act_intro": [],
    }
    defaults.update(overrides)
    return RetrievedContext(**defaults)


# ---------------------------------------------------------------------------
# clamp_disclosure_phase
# ---------------------------------------------------------------------------

class TestClampDisclosurePhase:
    def test_valid_values(self):
        for v in range(5):
            assert clamp_disclosure_phase(v) == v

    def test_clamps_below(self):
        assert clamp_disclosure_phase(-1) == PHASE_WORLD_ONLY

    def test_clamps_above(self):
        assert clamp_disclosure_phase(99) == PHASE_FULL

    def test_none_defaults_full(self):
        assert clamp_disclosure_phase(None) == PHASE_FULL

    def test_string_defaults_full(self):
        assert clamp_disclosure_phase("abc") == PHASE_FULL


# ---------------------------------------------------------------------------
# extract_lead_names_from_character_web
# ---------------------------------------------------------------------------

class TestExtractLeadNames:
    def test_extracts_names(self):
        cw = {
            "characters": [
                {"name": "Alice", "story_role": "main_character"},
                {"name": "Bob", "story_role": "love_interest"},
                {"name": "Charlie", "story_role": "supporting"},
            ]
        }
        prot, li = extract_lead_names_from_character_web(cw)
        assert prot == "Alice"
        assert li == "Bob"

    def test_missing_characters(self):
        assert extract_lead_names_from_character_web({}) == (None, None)

    def test_no_love_interest(self):
        cw = {
            "characters": [
                {"name": "Alice", "story_role": "main_character"},
            ]
        }
        prot, li = extract_lead_names_from_character_web(cw)
        assert prot == "Alice"
        assert li is None


# ---------------------------------------------------------------------------
# compute_context_disclosure_phases
# ---------------------------------------------------------------------------

class TestComputePhases:
    def test_world_only_when_no_leads(self):
        chapters = _make_chapters([
            (1, 1, "ROM-M0", ["townspeople"]),
            (1, 2, "ROM-M0", ["townspeople"]),
        ])
        phases = compute_context_disclosure_phases(chapters, "Alice", "Bob", 5)
        assert phases[(1, 1)] == PHASE_WORLD_ONLY
        assert phases[(1, 2)] == PHASE_WORLD_ONLY

    def test_protagonist_phase_on_entry(self):
        chapters = _make_chapters([
            (1, 1, "ROM-M0", ["townspeople"]),
            (1, 2, "ROM-M0", ["Alice"]),
            (1, 3, "ROM-M0", ["Alice"]),
        ])
        phases = compute_context_disclosure_phases(chapters, "Alice", "Bob", 5)
        assert phases[(1, 1)] == PHASE_WORLD_ONLY
        assert phases[(1, 2)] == PHASE_PROTAGONIST
        assert phases[(1, 3)] == PHASE_PROTAGONIST

    def test_love_interest_solo_phase(self):
        chapters = _make_chapters([
            (1, 1, "ROM-M0", ["Alice"]),
            (1, 2, "ROM-M0", ["Bob"]),
            (1, 3, "ROM-M0", ["Bob"]),
        ])
        phases = compute_context_disclosure_phases(chapters, "Alice", "Bob", 5)
        assert phases[(1, 1)] == PHASE_PROTAGONIST
        assert phases[(1, 2)] == PHASE_LOVE_INTEREST
        assert phases[(1, 3)] == PHASE_LOVE_INTEREST

    def test_both_leads_after_li_establishment(self):
        chapters = _make_chapters([
            (1, 1, "ROM-M0", ["Alice"]),
            (1, 2, "ROM-M0", ["Bob"]),
            (1, 3, "ROM-M0", ["Alice", "Bob"]),
        ])
        phases = compute_context_disclosure_phases(chapters, "Alice", "Bob", 5)
        assert phases[(1, 3)] == PHASE_BOTH_LEADS

    def test_skip_phase2_when_both_appear_together(self):
        """If leads appear together before LI gets solo time, skip to phase 3."""
        chapters = _make_chapters([
            (1, 1, "ROM-M0", ["Alice"]),
            (1, 2, "ROM-M0", ["Alice", "Bob"]),
        ])
        phases = compute_context_disclosure_phases(chapters, "Alice", "Bob", 5)
        assert phases[(1, 1)] == PHASE_PROTAGONIST
        assert phases[(1, 2)] == PHASE_BOTH_LEADS

    def test_rom_m1_triggers_full(self):
        chapters = _make_chapters([
            (1, 1, "ROM-M0", ["Alice"]),
            (1, 2, "ROM-M1", ["Alice", "Bob"]),
        ])
        phases = compute_context_disclosure_phases(chapters, "Alice", "Bob", 5)
        assert phases[(1, 2)] == PHASE_FULL

    def test_past_establishment_band_triggers_full(self):
        """Acts in chapters past establishment band should be PHASE_FULL."""
        chapters = _make_chapters([
            (4, 1, "ROM-M0", ["Alice"]),
        ])
        phases = compute_context_disclosure_phases(chapters, "Alice", "Bob", 5)
        assert phases[(4, 1)] == PHASE_FULL

    def test_monotonic_phase_never_decreases(self):
        chapters = _make_chapters([
            (1, 1, "ROM-M0", ["townspeople"]),
            (1, 2, "ROM-M0", ["Alice"]),
            (1, 3, "ROM-M0", ["Bob"]),
            (1, 4, "ROM-M0", ["Alice", "Bob"]),
            (1, 5, "ROM-M1", ["Alice", "Bob"]),
            (2, 1, "ROM-M0", ["townspeople"]),
        ])
        phases = compute_context_disclosure_phases(chapters, "Alice", "Bob", 5)
        values = [phases[(ch, act)] for ch, act in sorted(phases.keys())]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], (
                f"Phase decreased at index {i}: {values}"
            )

    def test_no_protagonist_name_stays_world(self):
        """When protagonist name is None, phase stays at 0."""
        chapters = _make_chapters([
            (1, 1, "ROM-M0", ["Alice"]),
        ])
        phases = compute_context_disclosure_phases(chapters, None, None, 5)
        assert phases[(1, 1)] == PHASE_WORLD_ONLY

    def test_cross_chapter_progression(self):
        chapters = _make_chapters([
            (1, 1, "ROM-M0", ["Alice"]),
            (1, 2, "ROM-M0", ["Bob"]),
            (2, 1, "ROM-M0", ["Alice", "Bob"]),
        ])
        phases = compute_context_disclosure_phases(chapters, "Alice", "Bob", 5)
        assert phases[(1, 1)] == PHASE_PROTAGONIST
        assert phases[(1, 2)] == PHASE_LOVE_INTEREST
        assert phases[(2, 1)] == PHASE_BOTH_LEADS


# ---------------------------------------------------------------------------
# format_disclosure_phase_constraint
# ---------------------------------------------------------------------------

class TestFormatConstraint:
    def test_phase_full_returns_empty(self):
        assert format_disclosure_phase_constraint(PHASE_FULL) == ""

    def test_phase_0_has_world_text(self):
        text = format_disclosure_phase_constraint(PHASE_WORLD_ONLY)
        assert "world" in text.lower()

    def test_phase_1_has_protagonist_text(self):
        text = format_disclosure_phase_constraint(PHASE_PROTAGONIST)
        assert "protagonist" in text.lower()

    def test_phase_2_has_new_character_text(self):
        text = format_disclosure_phase_constraint(PHASE_LOVE_INTEREST)
        assert "new character" in text.lower()


# ---------------------------------------------------------------------------
# Prompt builder integration: section gating per phase
# ---------------------------------------------------------------------------

class TestPromptBuilderDisclosureGating:
    """Verify that build_act_generation_prompt includes/suppresses sections."""

    def _build(self, phase: int, **ctx_overrides) -> tuple[str, str]:
        builder = PromptBuilder(max_context_chars=8000)
        _ctx_kw = {
            k: v
            for k, v in ctx_overrides.items()
            if k not in ("protagonist_name", "love_interest_name")
        }
        context = _make_context(**_ctx_kw)
        kw = {
            k: v
            for k, v in ctx_overrides.items()
            if k in ("protagonist_name", "love_interest_name")
        }
        return builder.build_act_generation_prompt(
            chapter=1,
            act=1,
            context=context,
            disclosure_phase=phase,
            **kw,
        )

    def test_phase_full_includes_character_web(self):
        prompt, _ = self._build(PHASE_FULL)
        assert "Character Web" in prompt

    def test_phase_full_includes_relationship_arcs(self):
        prompt, _ = self._build(PHASE_FULL)
        assert "Relationship Arcs" in prompt

    def test_phase_world_only_omits_character_web(self):
        prompt, _ = self._build(PHASE_WORLD_ONLY, character_web=[])
        assert "Character Web" not in prompt

    def test_phase_world_only_omits_foreshadowing(self):
        prompt, _ = self._build(PHASE_WORLD_ONLY, foreshadowing=[])
        assert "Foreshadowing" not in prompt

    def test_phase_world_only_omits_relationship_arcs(self):
        prompt, _ = self._build(PHASE_WORLD_ONLY, relationship_arcs=[])
        assert "Relationship Arcs" not in prompt

    def test_phase_world_only_includes_world_lore(self):
        prompt, _ = self._build(
            PHASE_WORLD_ONLY, character_web=[], foreshadowing=[], relationship_arcs=[],
        )
        assert "World Lore" in prompt

    def test_phase_world_only_has_constraint(self):
        prompt, _ = self._build(
            PHASE_WORLD_ONLY, character_web=[], foreshadowing=[], relationship_arcs=[],
        )
        assert "CONTEXT DISCLOSURE" in prompt

    def test_phase_world_only_drops_author_profile_section(self):
        prompt, _ = self._build(
            PHASE_WORLD_ONLY,
            character_web=[],
            foreshadowing=[],
            relationship_arcs=[],
            author_profile=[_make_result("heat: inferno")],
        )
        assert "Author Profile" not in prompt

    def test_phase_world_only_system_prompt_not_romance_only(self):
        _, system = self._build(
            PHASE_WORLD_ONLY,
            character_web=[],
            foreshadowing=[],
            relationship_arcs=[],
        )
        assert "published novel in this genre" in system
        assert "published romance novel" not in system

    def test_phase_protagonist_includes_character_web(self):
        prompt, _ = self._build(PHASE_PROTAGONIST)
        assert "Character Web" in prompt

    def test_phase_protagonist_has_preamble(self):
        prompt, _ = self._build(PHASE_PROTAGONIST)
        assert "Only the protagonist" in prompt
        assert "introduce" in prompt.lower()

    def test_phase_protagonist_lead_introduction_constraint(self):
        prompt, _ = self._build(
            PHASE_PROTAGONIST,
            protagonist_name="Jordan",
        )
        assert "LEAD INTRODUCTION" in prompt
        assert "Jordan" in prompt

    def test_phase_love_interest_has_preamble(self):
        prompt, _ = self._build(PHASE_LOVE_INTEREST)
        assert "new character is entering" in prompt

    def test_phase_love_interest_lead_introduction_constraint(self):
        prompt, _ = self._build(
            PHASE_LOVE_INTEREST,
            love_interest_name="Riley",
        )
        assert "LEAD INTRODUCTION" in prompt
        assert "Riley" in prompt

    def test_phase_both_leads_omits_relationship_arcs(self):
        prompt, _ = self._build(PHASE_BOTH_LEADS, relationship_arcs=[])
        assert "Relationship Arcs" not in prompt

    def test_phase_full_no_disclosure_constraint(self):
        prompt, _ = self._build(PHASE_FULL)
        assert "CONTEXT DISCLOSURE" not in prompt


# ---------------------------------------------------------------------------
# Outline disclosure tier assignment
# ---------------------------------------------------------------------------

class TestOutlineProgressiveWorldFirstCopy:
    def test_world_first_beat_note_mentions_world_only_acts(self):
        t = outline_chapter1_progressive_world_first_beat_note()
        assert "World-only acts" in t
        assert "ROM-M0" in t

    def test_world_first_rom_m_instruction_names_leads(self):
        t = build_outline_romance_milestone_instruction_world_first(
            chapter_num=1,
            num_chapters=10,
            slow_burn=False,
            heat_level_hint="medium",
            prior_max_rom_m_index_before_chapter=-1,
            protagonist_name="Alex",
            love_interest_name="Blake",
        )
        assert "Alex" in t
        assert "Blake" in t
        assert "WORLD-FIRST" in t


class TestOutlineDisclosureTier:
    def test_chapter_1_is_protagonist_tier(self):
        assert outline_disclosure_tier(1, 10) == OUTLINE_TIER_PROTAGONIST

    def test_chapter_2_in_establishment_is_both_no_romance(self):
        assert outline_disclosure_tier(2, 10) == OUTLINE_TIER_BOTH_NO_ROMANCE

    def test_chapter_2_outside_establishment_is_full(self):
        # 2-chapter book: establishment band is ch1 only
        assert outline_disclosure_tier(2, 2) == OUTLINE_TIER_FULL

    def test_chapter_3_is_full(self):
        assert outline_disclosure_tier(3, 10) == OUTLINE_TIER_FULL

    def test_chapter_0_is_full(self):
        assert outline_disclosure_tier(0, 10) == OUTLINE_TIER_FULL

    def test_short_book_ch1_still_protagonist(self):
        assert outline_disclosure_tier(1, 3) == OUTLINE_TIER_PROTAGONIST

    def test_late_chapter_always_full(self):
        assert outline_disclosure_tier(8, 10) == OUTLINE_TIER_FULL


# ---------------------------------------------------------------------------
# Arc redaction
# ---------------------------------------------------------------------------

_SAMPLE_ARC = {
    "title": "Briarwood Hearts",
    "premise": "Alice, a watchmaker, and Bob, a traveling merchant, find their paths crossing in the village of Briarwood.",
    "central_conflict": "Alice must choose between her duty and Bob's offer of a different life.",
    "romantic_arc": "From a chance meeting to a passionate romance, Alice and Bob discover love despite obstacles.",
    "theme": "Duty vs desire",
    "setting": "A quiet village in the countryside",
    "num_chapters": 10,
}


class TestRedactArcForOutline:
    def test_tier2_returns_unmodified(self):
        result = redact_arc_for_outline(_SAMPLE_ARC, 5, 10, "Alice", "Bob")
        assert result is _SAMPLE_ARC

    def test_no_li_name_returns_unmodified(self):
        result = redact_arc_for_outline(_SAMPLE_ARC, 1, 10, "Alice", None)
        assert result is _SAMPLE_ARC

    def test_tier0_redacts_romantic_arc(self):
        result = redact_arc_for_outline(_SAMPLE_ARC, 1, 10, "Alice", "Bob")
        assert result is not _SAMPLE_ARC
        assert "Bob" not in result["romantic_arc"]
        assert "passionate romance" not in result["romantic_arc"]
        assert "emerge" in result["romantic_arc"].lower()

    def test_tier0_redacts_li_from_premise(self):
        result = redact_arc_for_outline(_SAMPLE_ARC, 1, 10, "Alice", "Bob")
        assert "Bob" not in result["premise"]
        assert "another figure" in result["premise"].lower()

    def test_tier0_redacts_li_from_central_conflict(self):
        result = redact_arc_for_outline(_SAMPLE_ARC, 1, 10, "Alice", "Bob")
        assert "Bob" not in result["central_conflict"]

    def test_tier0_preserves_protagonist_name(self):
        result = redact_arc_for_outline(_SAMPLE_ARC, 1, 10, "Alice", "Bob")
        assert "Alice" in result["premise"]

    def test_tier0_preserves_other_fields(self):
        result = redact_arc_for_outline(_SAMPLE_ARC, 1, 10, "Alice", "Bob")
        assert result["title"] == _SAMPLE_ARC["title"]
        assert result["theme"] == _SAMPLE_ARC["theme"]
        assert result["setting"] == _SAMPLE_ARC["setting"]
        assert result["num_chapters"] == _SAMPLE_ARC["num_chapters"]

    def test_tier1_replaces_romantic_arc_vaguely(self):
        result = redact_arc_for_outline(_SAMPLE_ARC, 2, 10, "Alice", "Bob")
        assert result is not _SAMPLE_ARC
        assert "passionate" not in result["romantic_arc"]
        assert "paths crossing" in result["romantic_arc"].lower()

    def test_tier1_keeps_li_name_in_premise(self):
        result = redact_arc_for_outline(_SAMPLE_ARC, 2, 10, "Alice", "Bob")
        assert "Bob" in result["premise"]

    def test_tier1_keeps_central_conflict(self):
        result = redact_arc_for_outline(_SAMPLE_ARC, 2, 10, "Alice", "Bob")
        assert result["central_conflict"] == _SAMPLE_ARC["central_conflict"]

    def test_does_not_mutate_original(self):
        original_arc = dict(_SAMPLE_ARC)
        redact_arc_for_outline(_SAMPLE_ARC, 1, 10, "Alice", "Bob")
        assert _SAMPLE_ARC["romantic_arc"] == original_arc["romantic_arc"]
        assert _SAMPLE_ARC["premise"] == original_arc["premise"]

    def test_case_insensitive_name_replacement(self):
        arc = dict(_SAMPLE_ARC)
        arc["premise"] = "alice and BOB meet in town"
        result = redact_arc_for_outline(arc, 1, 10, "Alice", "Bob")
        assert "bob" not in result["premise"].lower()
        assert "alice" in result["premise"].lower()
