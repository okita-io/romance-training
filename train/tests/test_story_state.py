"""Unit tests for the enhanced StoryState module."""

import json
import pytest

from romance_factory.story_core.story_state import (
    CharacterState,
    PlotThread,
    RomanceMilestone,
    ForeshadowingPlant,
    StoryState,
    prune_stale_threads,
    _is_milestone_chronologically_valid,
    MAX_CHAPTERS_CAP,
    STALE_THREAD_THRESHOLD,
)


# ---------------------------------------------------------------------------
# CharacterState / PlotThread / RomanceMilestone / ForeshadowingPlant basics
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_character_state_defaults(self):
        cs = CharacterState(name="Elena")
        assert cs.name == "Elena"
        assert cs.emotional_state == ""
        assert cs.last_updated_chapter == 0

    def test_plot_thread_auto_id(self):
        t = PlotThread(description="locked door")
        assert t.thread_id  # auto-generated
        assert t.status == "open"

    def test_romance_milestone_fields(self):
        m = RomanceMilestone(milestone="first_kiss", chapter=3, act=2, description="They kissed")
        assert m.milestone == "first_kiss"
        assert m.chapter == 3

    def test_foreshadowing_plant_defaults(self):
        fp = ForeshadowingPlant(element="portrait", planted_chapter=2, planted_act=1)
        assert not fp.resolved
        assert fp.resolved_chapter is None


# ---------------------------------------------------------------------------
# StoryState construction and validation
# ---------------------------------------------------------------------------

class TestStoryStateConstruction:
    def test_default_construction(self):
        state = StoryState()
        assert state.max_chapters == 25
        assert state.current_chapter == 0
        assert state.characters == []

    def test_max_chapters_cap(self):
        """Req 4.6: max_chapters capped at 25."""
        state = StoryState(max_chapters=50)
        assert state.max_chapters == MAX_CHAPTERS_CAP

    def test_max_chapters_negative(self):
        state = StoryState(max_chapters=-5)
        assert state.max_chapters == 0

    def test_current_chapter_clamped(self):
        """Req 12.3: current_chapter between 0 and max_chapters."""
        state = StoryState(current_chapter=30, max_chapters=20)
        assert state.current_chapter == 20

    def test_current_chapter_negative_clamped(self):
        state = StoryState(current_chapter=-1)
        assert state.current_chapter == 0


# ---------------------------------------------------------------------------
# to_context_string
# ---------------------------------------------------------------------------

class TestContextString:
    def test_truncation(self):
        """Req 10.3: to_context_string truncates to max_chars."""
        state = StoryState(
            characters=[CharacterState(name=f"Char{i}", emotional_state="happy") for i in range(20)],
            genre_plants=["gothic"] * 10,
        )
        result = state.to_context_string(max_chars=100)
        assert len(result) <= 100

    def test_empty_state(self):
        state = StoryState()
        result = state.to_context_string()
        assert isinstance(result, str)

    def test_includes_cliffhanger(self):
        state = StoryState(pending_cliffhanger="Elena found the letter")
        result = state.to_context_string()
        assert "Elena found the letter" in result

    def test_max_chars_zero(self):
        state = StoryState(genre_plants=["gothic"])
        result = state.to_context_string(max_chars=0)
        assert result == ""


# ---------------------------------------------------------------------------
# to_continuity_checklist
# ---------------------------------------------------------------------------

class TestContinuityChecklist:
    def test_empty_state_returns_empty(self):
        state = StoryState()
        assert state.to_continuity_checklist() == ""

    def test_stale_thread_flagged(self):
        state = StoryState(
            current_chapter=10,
            plot_threads=[PlotThread(description="locked door", planted_chapter=2, status="open")],
        )
        result = state.to_continuity_checklist()
        assert "STALE THREAD" in result

    def test_pending_cliffhanger_in_checklist(self):
        state = StoryState(pending_cliffhanger="The door opened")
        result = state.to_continuity_checklist()
        assert "PENDING CLIFFHANGER" in result


# ---------------------------------------------------------------------------
# JSON serialization round-trip  (Req 10.1)
# ---------------------------------------------------------------------------

class TestJsonSerialization:
    def _make_populated_state(self) -> StoryState:
        return StoryState(
            characters=[CharacterState(name="Elena", emotional_state="conflicted", location="library")],
            plot_threads=[PlotThread(thread_id="t1", description="locked door", planted_chapter=1, planted_act=2, status="open")],
            romance_milestones=[RomanceMilestone(milestone="first_touch", chapter=3, act=2, description="Hands brush")],
            foreshadowing=[ForeshadowingPlant(element="portrait", planted_chapter=2, planted_act=1, expected_payoff="family secret")],
            genre_plants=["gothic"],
            trope_plants=["forced_proximity"],
            pending_cliffhanger="Elena found the letter",
            current_chapter=5,
            current_act=2,
            max_chapters=20,
        )

    def test_round_trip(self):
        original = self._make_populated_state()
        json_dict = original.to_json_dict()
        restored = StoryState.from_json(json_dict)

        assert restored.current_chapter == original.current_chapter
        assert restored.current_act == original.current_act
        assert restored.max_chapters == original.max_chapters
        assert restored.pending_cliffhanger == original.pending_cliffhanger
        assert len(restored.characters) == len(original.characters)
        assert restored.characters[0].name == "Elena"
        assert len(restored.plot_threads) == len(original.plot_threads)
        assert restored.plot_threads[0].thread_id == "t1"
        assert len(restored.romance_milestones) == len(original.romance_milestones)
        assert restored.romance_milestones[0].milestone == "first_touch"
        assert len(restored.foreshadowing) == len(original.foreshadowing)
        assert restored.foreshadowing[0].element == "portrait"
        assert restored.genre_plants == original.genre_plants
        assert restored.trope_plants == original.trope_plants

    def test_json_is_valid_json(self):
        state = self._make_populated_state()
        json_str = json.dumps(state.to_json_dict())
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_from_json_empty(self):
        state = StoryState.from_json({})
        assert state.current_chapter == 0
        assert state.characters == []


# ---------------------------------------------------------------------------
# series_state_handoff  (Req 4.7)
# ---------------------------------------------------------------------------

class TestSeriesStateHandoff:
    def test_only_carry_over_threads_preserved(self):
        """Req 4.7: Only carry_over threads survive handoff."""
        state = StoryState(
            plot_threads=[
                PlotThread(thread_id="t1", description="local subplot", carry_over=False, status="open"),
                PlotThread(thread_id="t2", description="family conflict", carry_over=True, status="open"),
                PlotThread(thread_id="t3", description="resolved mystery", carry_over=True, status="resolved"),
            ],
        )
        new_state = state.series_state_handoff()
        assert len(new_state.plot_threads) == 1
        assert new_state.plot_threads[0].thread_id == "t2"

    def test_characters_reset_ephemeral_state(self):
        state = StoryState(
            characters=[CharacterState(name="Elena", emotional_state="angry", location="library")],
        )
        new_state = state.series_state_handoff()
        assert len(new_state.characters) == 1
        assert new_state.characters[0].emotional_state == ""
        assert new_state.characters[0].physical_state == "fine"
        assert new_state.characters[0].location == ""

    def test_romance_milestones_preserved(self):
        state = StoryState(
            romance_milestones=[RomanceMilestone(milestone="first_kiss", chapter=5, act=2, description="They kissed")],
        )
        new_state = state.series_state_handoff()
        assert len(new_state.romance_milestones) == 1

    def test_unresolved_foreshadowing_preserved(self):
        state = StoryState(
            foreshadowing=[
                ForeshadowingPlant(element="portrait", planted_chapter=2, planted_act=1, resolved=False),
                ForeshadowingPlant(element="letter", planted_chapter=3, planted_act=1, resolved=True),
            ],
        )
        new_state = state.series_state_handoff()
        assert len(new_state.foreshadowing) == 1
        assert new_state.foreshadowing[0].element == "portrait"

    def test_chapter_tracking_reset(self):
        state = StoryState(current_chapter=15, current_act=3)
        new_state = state.series_state_handoff()
        assert new_state.current_chapter == 0
        assert new_state.current_act == 0
        assert new_state.pending_cliffhanger is None


# ---------------------------------------------------------------------------
# Romance milestone regression  (Req 4.2)
# ---------------------------------------------------------------------------

class TestMilestoneRegression:
    def test_valid_forward_milestone(self):
        existing = [RomanceMilestone(milestone="first_touch", chapter=3, act=2, description="")]
        new = RomanceMilestone(milestone="first_kiss", chapter=5, act=1, description="")
        assert _is_milestone_chronologically_valid(existing, new) is True

    def test_backward_regression_rejected(self):
        existing = [RomanceMilestone(milestone="first_kiss", chapter=5, act=2, description="")]
        new = RomanceMilestone(milestone="first_touch", chapter=3, act=1, description="")
        assert _is_milestone_chronologically_valid(existing, new) is False

    def test_same_chapter_act_allowed(self):
        existing = [RomanceMilestone(milestone="first_touch", chapter=3, act=2, description="")]
        new = RomanceMilestone(milestone="vulnerability", chapter=3, act=2, description="")
        assert _is_milestone_chronologically_valid(existing, new) is True

    def test_empty_existing_always_valid(self):
        assert _is_milestone_chronologically_valid([], RomanceMilestone(chapter=1, act=1)) is True


# ---------------------------------------------------------------------------
# Stale thread detection  (Req 4.3)
# ---------------------------------------------------------------------------

class TestStaleThreadDetection:
    def test_thread_flagged_stale(self):
        state = StoryState(
            plot_threads=[PlotThread(description="old thread", planted_chapter=1, status="open")],
        )
        prune_stale_threads(state, current_chapter=7)
        assert state.plot_threads[0].status == "stale"

    def test_thread_not_stale_within_threshold(self):
        state = StoryState(
            plot_threads=[PlotThread(description="recent thread", planted_chapter=3, status="open")],
        )
        prune_stale_threads(state, current_chapter=7)
        assert state.plot_threads[0].status == "open"

    def test_resolved_thread_not_flagged(self):
        state = StoryState(
            plot_threads=[PlotThread(description="done", planted_chapter=1, status="resolved")],
        )
        prune_stale_threads(state, current_chapter=10)
        assert state.plot_threads[0].status == "resolved"

    def test_exactly_at_threshold_not_stale(self):
        """Thread at exactly 5 chapters gap should NOT be stale (> 5 required)."""
        state = StoryState(
            plot_threads=[PlotThread(description="borderline", planted_chapter=2, status="open")],
        )
        prune_stale_threads(state, current_chapter=7)  # gap = 5, not > 5
        assert state.plot_threads[0].status == "open"
