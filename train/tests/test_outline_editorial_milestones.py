"""Unit tests for outline editorial milestone ratchet (Task 8.1).

Tests track_achieved_milestones, and the milestone extensions to
build_outline_beat_review_prompt and build_outline_beat_rewrite_prompt.
"""

from __future__ import annotations

from romance_factory.generate.outline_editorial import (
    ROMANCE_MILESTONES,
    build_outline_beat_review_prompt,
    build_outline_beat_rewrite_prompt,
    track_achieved_milestones,
)


# ---------------------------------------------------------------------------
# track_achieved_milestones
# ---------------------------------------------------------------------------

class TestTrackAchievedMilestones:
    def test_empty_chapters(self):
        assert track_achieved_milestones([]) == []

    def test_no_milestone_references(self):
        chapters = [{"acts": [{"summary": "They argued about dinner", "emotional_tone": "anger", "plot_function": "conflict"}]}]
        assert track_achieved_milestones(chapters) == []

    def test_single_milestone_in_summary(self):
        chapters = [{"acts": [{"summary": "Their first meeting at the bookshop", "emotional_tone": "curiosity", "plot_function": "setup"}]}]
        assert track_achieved_milestones(chapters) == ["first_meeting"]

    def test_milestone_in_emotional_tone(self):
        chapters = [{"acts": [{"summary": "A quiet evening", "emotional_tone": "vulnerability", "plot_function": "development"}]}]
        assert track_achieved_milestones(chapters) == ["vulnerability"]

    def test_milestone_in_plot_function(self):
        chapters = [{"acts": [{"summary": "The evening continues", "emotional_tone": "warmth", "plot_function": "first kiss scene"}]}]
        assert track_achieved_milestones(chapters) == ["first_kiss"]

    def test_multiple_milestones_in_order(self):
        chapters = [
            {"acts": [
                {"summary": "First meeting at the cafe", "emotional_tone": "curiosity", "plot_function": "intro"},
                {"summary": "Hands touch across the table", "emotional_tone": "tension", "plot_function": "escalation"},
            ]},
            {"acts": [
                {"summary": "Their first kiss", "emotional_tone": "passion", "plot_function": "climax"},
            ]},
        ]
        result = track_achieved_milestones(chapters)
        assert result == ["first_meeting", "first_touch", "first_kiss"]

    def test_duplicate_milestone_only_counted_once(self):
        chapters = [
            {"acts": [
                {"summary": "First meeting at the park", "emotional_tone": "joy", "plot_function": "setup"},
                {"summary": "Another first meeting reference", "emotional_tone": "surprise", "plot_function": "callback"},
            ]},
        ]
        result = track_achieved_milestones(chapters)
        assert result.count("first_meeting") == 1

    def test_all_milestones(self):
        chapters = [{"acts": [
            {"summary": "first meeting", "emotional_tone": "", "plot_function": ""},
            {"summary": "first touch", "emotional_tone": "", "plot_function": ""},
            {"summary": "first kiss", "emotional_tone": "", "plot_function": ""},
            {"summary": "vulnerability moment", "emotional_tone": "", "plot_function": ""},
            {"summary": "love confession", "emotional_tone": "", "plot_function": ""},
            {"summary": "physical intimacy", "emotional_tone": "", "plot_function": ""},
            {"summary": "nudity", "emotional_tone": "", "plot_function": ""},
            {"summary": "graphic sexual pornographic content", "emotional_tone": "", "plot_function": ""},
        ]}]
        assert track_achieved_milestones(chapters) == ROMANCE_MILESTONES

    def test_missing_acts_key(self):
        chapters = [{"chapters": "no acts here"}]
        assert track_achieved_milestones(chapters) == []

    def test_non_dict_acts(self):
        chapters = [{"acts": ["not a dict"]}]
        assert track_achieved_milestones(chapters) == []

    def test_none_field_values(self):
        chapters = [{"acts": [{"summary": None, "emotional_tone": None, "plot_function": None}]}]
        assert track_achieved_milestones(chapters) == []


# ---------------------------------------------------------------------------
# build_outline_beat_review_prompt — milestone extension
# ---------------------------------------------------------------------------

class TestReviewPromptMilestones:
    def test_no_milestones_no_section(self):
        _, prompt = build_outline_beat_review_prompt(1, 1, "test beat")
        assert "Already-achieved romance milestones" not in prompt

    def test_empty_milestones_no_section(self):
        _, prompt = build_outline_beat_review_prompt(1, 1, "test beat", achieved_milestones=[])
        assert "Already-achieved romance milestones" not in prompt

    def test_milestones_section_present(self):
        _, prompt = build_outline_beat_review_prompt(
            1, 1, "test beat",
            achieved_milestones=["first_meeting", "first_touch"],
        )
        assert "### Already-achieved romance milestones (prior beats only)" in prompt
        assert "first_meeting (achieved)" in prompt
        assert "first_touch (achieved)" in prompt

    def test_must_not_revisit_constraint(self):
        _, prompt = build_outline_beat_review_prompt(
            1, 1, "test beat",
            achieved_milestones=["first_kiss"],
        )
        assert "must not" in prompt.lower()
        assert "revisit" in prompt.lower()

    def test_milestone_regression_instruction(self):
        _, prompt = build_outline_beat_review_prompt(
            1, 1, "test beat",
            achieved_milestones=["first_meeting"],
        )
        assert "milestone_regression" in prompt
        assert "BLOCKING" in prompt


# ---------------------------------------------------------------------------
# build_outline_beat_rewrite_prompt — milestone extension
# ---------------------------------------------------------------------------

class TestRewritePromptMilestones:
    def _make_act_dict(self):
        return {"act_number": 1, "summary": "A placeholder beat summary text"}

    def test_no_milestones_no_section(self):
        _, prompt = build_outline_beat_rewrite_prompt(1, 1, self._make_act_dict(), [], "")
        assert "Already-achieved romance milestones" not in prompt

    def test_empty_milestones_no_section(self):
        _, prompt = build_outline_beat_rewrite_prompt(
            1, 1, self._make_act_dict(), [], "",
            achieved_milestones=[],
        )
        assert "Already-achieved romance milestones" not in prompt

    def test_milestones_section_present(self):
        _, prompt = build_outline_beat_rewrite_prompt(
            1, 1, self._make_act_dict(), [], "",
            achieved_milestones=["first_meeting", "first_kiss"],
        )
        assert "### Already-achieved romance milestones (prior beats only)" in prompt
        assert "first_meeting (achieved)" in prompt
        assert "first_kiss (achieved)" in prompt

    def test_must_not_revisit_constraint(self):
        _, prompt = build_outline_beat_rewrite_prompt(
            1, 1, self._make_act_dict(), [], "",
            achieved_milestones=["vulnerability"],
        )
        assert "must not" in prompt.lower()
        assert "revisit" in prompt.lower()
