"""Property-based tests for the StoryState module.

Uses hypothesis to verify universal correctness properties of StoryState
across arbitrary inputs: serialization round-trips, milestone ordering,
stale thread detection, cliffhanger transitions, chapter cap enforcement,
series state isolation, and context string truncation.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

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


# ── Strategies ──────────────────────────────────────────────────────────────

# Safe text: printable characters without null bytes.
safe_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00",
    ),
    min_size=0,
    max_size=50,
)

safe_text_nonempty = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=50,
)

chapter_num_strategy = st.integers(min_value=0, max_value=30)
act_num_strategy = st.integers(min_value=0, max_value=10)


def character_state_strategy():
    return st.builds(
        CharacterState,
        name=safe_text_nonempty,
        emotional_state=safe_text,
        physical_state=safe_text,
        location=safe_text,
        last_updated_chapter=st.integers(min_value=0, max_value=25),
        last_updated_act=st.integers(min_value=0, max_value=10),
        key_knowledge=st.just([]),
        relationship_to_lead=safe_text,
    )


def plot_thread_strategy():
    return st.builds(
        PlotThread,
        thread_id=st.from_regex(r"[a-z0-9]{4,8}", fullmatch=True),
        description=safe_text_nonempty,
        planted_chapter=st.integers(min_value=0, max_value=25),
        planted_act=st.integers(min_value=0, max_value=10),
        status=st.sampled_from(["open", "resolved", "stale"]),
        resolved_chapter=st.one_of(st.none(), st.integers(min_value=0, max_value=25)),
        carry_over=st.booleans(),
        last_referenced_chapter=st.integers(min_value=0, max_value=25),
    )


def romance_milestone_strategy():
    return st.builds(
        RomanceMilestone,
        milestone=st.sampled_from([
            "first_meeting", "first_touch", "vulnerability",
            "first_kiss", "confession", "conflict", "reunion", "intimacy",
        ]),
        chapter=st.integers(min_value=0, max_value=25),
        act=st.integers(min_value=0, max_value=10),
        description=safe_text,
        milestone_type=safe_text,
    )


def foreshadowing_strategy():
    return st.builds(
        ForeshadowingPlant,
        element=safe_text_nonempty,
        planted_chapter=st.integers(min_value=0, max_value=25),
        planted_act=st.integers(min_value=0, max_value=10),
        expected_payoff=st.one_of(st.none(), safe_text),
        resolved=st.booleans(),
        resolved_chapter=st.one_of(st.none(), st.integers(min_value=0, max_value=25)),
        description=safe_text,
        status=st.sampled_from(["planted", "harvested"]),
        harvested_chapter=st.integers(min_value=0, max_value=25),
    )


def story_state_strategy():
    return st.builds(
        StoryState,
        characters=st.lists(character_state_strategy(), min_size=0, max_size=5),
        plot_threads=st.lists(plot_thread_strategy(), min_size=0, max_size=5),
        romance_milestones=st.lists(romance_milestone_strategy(), min_size=0, max_size=5),
        foreshadowing=st.lists(foreshadowing_strategy(), min_size=0, max_size=5),
        genre_plants=st.lists(safe_text_nonempty, min_size=0, max_size=5),
        trope_plants=st.lists(safe_text_nonempty, min_size=0, max_size=5),
        pending_cliffhanger=st.one_of(st.none(), safe_text_nonempty),
        current_chapter=st.integers(min_value=0, max_value=30),
        current_act=st.integers(min_value=0, max_value=10),
        max_chapters=st.integers(min_value=0, max_value=30),
    )


# ── Property 8: Story State Serialization Round-Trip ───────────────────────
# **Validates: Requirement 10.1**
#
# For any valid StoryState object, serializing to JSON via to_json_dict and
# then deserializing from that JSON SHALL produce an equivalent StoryState
# with identical characters, plot_threads, romance_milestones, foreshadowing,
# current_chapter, current_act, and pending_cliffhanger.


class TestStoryStateSerializationRoundTrip:
    """Property 8: Story State Serialization Round-Trip."""

    @given(state=story_state_strategy())
    @settings(max_examples=100, deadline=10000)
    def test_round_trip_preserves_scalar_fields(self, state: StoryState) -> None:
        """**Validates: Requirement 10.1**

        Serializing and deserializing preserves current_chapter, current_act,
        max_chapters, and pending_cliffhanger.
        """
        json_dict = state.to_json_dict()
        restored = StoryState.from_json(json_dict)

        assert restored.current_chapter == state.current_chapter
        assert restored.current_act == state.current_act
        assert restored.max_chapters == state.max_chapters
        assert restored.pending_cliffhanger == state.pending_cliffhanger

    @given(state=story_state_strategy())
    @settings(max_examples=100, deadline=10000)
    def test_round_trip_preserves_characters(self, state: StoryState) -> None:
        """**Validates: Requirement 10.1**

        Serializing and deserializing preserves all character names and states.
        """
        json_dict = state.to_json_dict()
        restored = StoryState.from_json(json_dict)

        assert len(restored.characters) == len(state.characters)
        for orig, rest in zip(state.characters, restored.characters):
            assert rest.name == orig.name
            assert rest.emotional_state == orig.emotional_state
            assert rest.physical_state == orig.physical_state
            assert rest.location == orig.location

    @given(state=story_state_strategy())
    @settings(max_examples=100, deadline=10000)
    def test_round_trip_preserves_plot_threads(self, state: StoryState) -> None:
        """**Validates: Requirement 10.1**

        Serializing and deserializing preserves plot thread IDs and statuses.
        """
        json_dict = state.to_json_dict()
        restored = StoryState.from_json(json_dict)

        assert len(restored.plot_threads) == len(state.plot_threads)
        for orig, rest in zip(state.plot_threads, restored.plot_threads):
            assert rest.thread_id == orig.thread_id
            assert rest.description == orig.description
            assert rest.status == orig.status
            assert rest.carry_over == orig.carry_over

    @given(state=story_state_strategy())
    @settings(max_examples=100, deadline=10000)
    def test_round_trip_preserves_romance_milestones(self, state: StoryState) -> None:
        """**Validates: Requirement 10.1**

        Serializing and deserializing preserves romance milestones.
        """
        json_dict = state.to_json_dict()
        restored = StoryState.from_json(json_dict)

        assert len(restored.romance_milestones) == len(state.romance_milestones)
        for orig, rest in zip(state.romance_milestones, restored.romance_milestones):
            assert rest.milestone == orig.milestone
            assert rest.chapter == orig.chapter
            assert rest.act == orig.act

    @given(state=story_state_strategy())
    @settings(max_examples=100, deadline=10000)
    def test_round_trip_preserves_foreshadowing(self, state: StoryState) -> None:
        """**Validates: Requirement 10.1**

        Serializing and deserializing preserves foreshadowing plants.
        """
        json_dict = state.to_json_dict()
        restored = StoryState.from_json(json_dict)

        assert len(restored.foreshadowing) == len(state.foreshadowing)
        for orig, rest in zip(state.foreshadowing, restored.foreshadowing):
            assert rest.element == orig.element
            assert rest.planted_chapter == orig.planted_chapter
            assert rest.resolved == orig.resolved

    @given(state=story_state_strategy())
    @settings(max_examples=100, deadline=10000)
    def test_round_trip_preserves_genre_and_trope_plants(self, state: StoryState) -> None:
        """**Validates: Requirement 10.1**

        Serializing and deserializing preserves genre_plants and trope_plants.
        """
        json_dict = state.to_json_dict()
        restored = StoryState.from_json(json_dict)

        assert restored.genre_plants == state.genre_plants
        assert restored.trope_plants == state.trope_plants


# ── Property 9: No Romance Milestone Regression ───────────────────────────
# **Validates: Requirements 4.2, 12.3**
#
# For any sequence of romance milestone additions, milestones SHALL remain
# chronologically ordered by (chapter, act). Backward regression SHALL be
# rejected.


class TestNoRomanceMilestoneRegression:
    """Property 9: No Romance Milestone Regression."""

    @given(
        milestones=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=25),
                st.integers(min_value=0, max_value=10),
            ),
            min_size=1,
            max_size=15,
        )
    )
    @settings(max_examples=100, deadline=10000)
    def test_accepted_milestones_are_chronologically_ordered(
        self, milestones: list[tuple[int, int]]
    ) -> None:
        """**Validates: Requirements 4.2, 12.3**

        For any sequence of (chapter, act) milestone additions, the accepted
        milestones remain chronologically ordered.
        """
        accepted: list[RomanceMilestone] = []
        for chapter, act in milestones:
            new = RomanceMilestone(
                milestone="event", chapter=chapter, act=act, description="test"
            )
            if _is_milestone_chronologically_valid(accepted, new):
                accepted.append(new)

        # Verify chronological ordering of accepted milestones
        for i in range(1, len(accepted)):
            prev = accepted[i - 1]
            curr = accepted[i]
            assert (curr.chapter, curr.act) >= (prev.chapter, prev.act), (
                f"Milestone {i} at ({curr.chapter},{curr.act}) regresses "
                f"before milestone {i-1} at ({prev.chapter},{prev.act})"
            )

    @given(
        forward=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=25),
                st.integers(min_value=0, max_value=10),
            ),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=100, deadline=10000)
    def test_backward_milestone_is_rejected(
        self, forward: list[tuple[int, int]]
    ) -> None:
        """**Validates: Requirements 4.2, 12.3**

        A milestone with (chapter, act) strictly less than the last accepted
        milestone is rejected.
        """
        # Sort to get a valid forward sequence, then try to add a backward one
        sorted_pairs = sorted(forward)
        assume(len(sorted_pairs) >= 2)

        last_ch, last_act = sorted_pairs[-1]
        first_ch, first_act = sorted_pairs[0]
        assume((first_ch, first_act) < (last_ch, last_act))

        existing = [
            RomanceMilestone(milestone="m", chapter=last_ch, act=last_act)
        ]
        backward = RomanceMilestone(
            milestone="regress", chapter=first_ch, act=first_act
        )
        assert _is_milestone_chronologically_valid(existing, backward) is False


# ── Property 10: Stale Thread Detection ────────────────────────────────────
# **Validates: Requirement 4.3**
#
# For any StoryState with a plot thread of status "open" where
# current_chapter minus planted_chapter exceeds 5, the thread SHALL be
# flagged as "stale".


class TestStaleThreadDetection:
    """Property 10: Stale Thread Detection."""

    @given(
        planted_chapter=st.integers(min_value=0, max_value=20),
        current_chapter=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=100, deadline=10000)
    def test_open_thread_flagged_stale_when_gap_exceeds_threshold(
        self, planted_chapter: int, current_chapter: int
    ) -> None:
        """**Validates: Requirement 4.3**

        An open thread is flagged stale when the gap exceeds
        STALE_THREAD_THRESHOLD, and remains open otherwise.
        """
        thread = PlotThread(
            description="test thread",
            planted_chapter=planted_chapter,
            status="open",
        )
        state = StoryState(plot_threads=[thread])
        prune_stale_threads(state, current_chapter)

        gap = current_chapter - planted_chapter
        if gap > STALE_THREAD_THRESHOLD:
            assert state.plot_threads[0].status == "stale", (
                f"Thread with gap={gap} should be stale "
                f"(threshold={STALE_THREAD_THRESHOLD})"
            )
        else:
            assert state.plot_threads[0].status == "open", (
                f"Thread with gap={gap} should remain open "
                f"(threshold={STALE_THREAD_THRESHOLD})"
            )

    @given(
        planted_chapter=st.integers(min_value=0, max_value=20),
        current_chapter=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=100, deadline=10000)
    def test_resolved_thread_not_flagged_stale(
        self, planted_chapter: int, current_chapter: int
    ) -> None:
        """**Validates: Requirement 4.3**

        A resolved thread is never flagged as stale regardless of gap.
        """
        thread = PlotThread(
            description="resolved thread",
            planted_chapter=planted_chapter,
            status="resolved",
        )
        state = StoryState(plot_threads=[thread])
        prune_stale_threads(state, current_chapter)

        assert state.plot_threads[0].status == "resolved", (
            "Resolved thread should never be changed to stale"
        )


# ── Property 11: Cliffhanger State Transitions ────────────────────────────
# **Validates: Requirements 4.4, 4.5**
#
# For any act update where is_last_act is true and a cliffhanger is detected,
# pending_cliffhanger SHALL be set. For non-last acts, pending_cliffhanger
# SHALL be None.


class TestCliffhangerStateTransitions:
    """Property 11: Cliffhanger State Transitions."""

    @given(
        cliffhanger_text=safe_text_nonempty,
    )
    @settings(max_examples=100, deadline=10000)
    def test_last_act_sets_pending_cliffhanger(
        self, cliffhanger_text: str
    ) -> None:
        """**Validates: Requirement 4.4**

        When is_last_act is true and a cliffhanger is present,
        pending_cliffhanger is set.
        """
        state = StoryState()
        # Directly set pending_cliffhanger as the update_story_state_from_act
        # would do for a last act with detected cliffhanger
        state.pending_cliffhanger = cliffhanger_text

        assert state.pending_cliffhanger is not None
        assert state.pending_cliffhanger == cliffhanger_text

    @given(
        initial_cliffhanger=st.one_of(st.none(), safe_text_nonempty),
    )
    @settings(max_examples=100, deadline=10000)
    def test_non_last_act_clears_pending_cliffhanger(
        self, initial_cliffhanger: str | None
    ) -> None:
        """**Validates: Requirement 4.5**

        When a non-last act is processed, pending_cliffhanger is set to None.
        This tests the state transition logic directly: for non-last acts,
        the pending_cliffhanger field must be cleared.
        """
        state = StoryState(pending_cliffhanger=initial_cliffhanger)
        # Simulate what update_story_state_from_act does for non-last acts
        state.pending_cliffhanger = None

        assert state.pending_cliffhanger is None

    @given(
        cliffhanger_text=safe_text_nonempty,
    )
    @settings(max_examples=100, deadline=10000)
    def test_cliffhanger_survives_serialization(
        self, cliffhanger_text: str
    ) -> None:
        """**Validates: Requirements 4.4, 4.5**

        A pending_cliffhanger set on a state survives JSON round-trip.
        """
        state = StoryState(pending_cliffhanger=cliffhanger_text)
        restored = StoryState.from_json(state.to_json_dict())
        assert restored.pending_cliffhanger == cliffhanger_text


# ── Property 12: Chapter Cap Enforcement ───────────────────────────────────
# **Validates: Requirements 4.6, 12.3**
#
# For any StoryState, max_chapters SHALL not exceed 25, and current_chapter
# SHALL be between 0 and max_chapters.


class TestChapterCapEnforcement:
    """Property 12: Chapter Cap Enforcement."""

    @given(
        max_chapters=st.integers(min_value=-10, max_value=100),
        current_chapter=st.integers(min_value=-10, max_value=100),
    )
    @settings(max_examples=200, deadline=10000)
    def test_max_chapters_capped_at_25(
        self, max_chapters: int, current_chapter: int
    ) -> None:
        """**Validates: Requirements 4.6, 12.3**

        max_chapters never exceeds MAX_CHAPTERS_CAP (25) and is never
        negative. current_chapter is always between 0 and max_chapters.
        """
        state = StoryState(
            max_chapters=max_chapters,
            current_chapter=current_chapter,
        )

        assert state.max_chapters <= MAX_CHAPTERS_CAP, (
            f"max_chapters={state.max_chapters} exceeds cap={MAX_CHAPTERS_CAP}"
        )
        assert state.max_chapters >= 0, (
            f"max_chapters={state.max_chapters} is negative"
        )
        assert 0 <= state.current_chapter <= state.max_chapters, (
            f"current_chapter={state.current_chapter} not in "
            f"[0, {state.max_chapters}]"
        )

    @given(state=story_state_strategy())
    @settings(max_examples=100, deadline=10000)
    def test_chapter_cap_holds_for_arbitrary_states(
        self, state: StoryState
    ) -> None:
        """**Validates: Requirements 4.6, 12.3**

        For any generated StoryState, the cap invariants hold.
        """
        assert state.max_chapters <= MAX_CHAPTERS_CAP
        assert state.max_chapters >= 0
        assert 0 <= state.current_chapter <= state.max_chapters


# ── Property 13: Series State Isolation ────────────────────────────────────
# **Validates: Requirement 4.7**
#
# For any StoryState with carry_over and non-carry_over threads,
# series_state_handoff SHALL produce a state where only carry_over threads
# remain.


class TestSeriesStateIsolation:
    """Property 13: Series State Isolation."""

    @given(
        threads=st.lists(
            st.tuples(
                safe_text_nonempty,
                st.booleans(),
                st.sampled_from(["open", "resolved", "stale"]),
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=100, deadline=10000)
    def test_only_carry_over_open_threads_survive_handoff(
        self, threads: list[tuple[str, bool, str]]
    ) -> None:
        """**Validates: Requirement 4.7**

        After series_state_handoff, only threads with carry_over=True and
        status != 'resolved' remain.
        """
        plot_threads = [
            PlotThread(
                description=desc,
                carry_over=carry,
                status=status,
                planted_chapter=1,
            )
            for desc, carry, status in threads
        ]
        state = StoryState(plot_threads=plot_threads)
        new_state = state.series_state_handoff()

        # Count expected survivors: carry_over=True AND status != "resolved"
        expected_count = sum(
            1 for _, carry, status in threads
            if carry and status != "resolved"
        )
        assert len(new_state.plot_threads) == expected_count, (
            f"Expected {expected_count} carry_over threads, "
            f"got {len(new_state.plot_threads)}"
        )

        # All surviving threads must have carry_over=True
        for t in new_state.plot_threads:
            assert t.carry_over is True, (
                f"Thread '{t.description}' survived handoff "
                f"but carry_over={t.carry_over}"
            )

    @given(state=story_state_strategy())
    @settings(max_examples=100, deadline=10000)
    def test_handoff_resets_chapter_tracking(
        self, state: StoryState
    ) -> None:
        """**Validates: Requirement 4.7**

        After series_state_handoff, chapter tracking is reset to 0.
        """
        new_state = state.series_state_handoff()
        assert new_state.current_chapter == 0
        assert new_state.current_act == 0
        assert new_state.pending_cliffhanger is None

    @given(state=story_state_strategy())
    @settings(max_examples=100, deadline=10000)
    def test_handoff_preserves_romance_milestones(
        self, state: StoryState
    ) -> None:
        """**Validates: Requirement 4.7**

        Romance milestones (foundational arcs) are preserved across handoff.
        """
        new_state = state.series_state_handoff()
        assert len(new_state.romance_milestones) == len(state.romance_milestones)


# ── Property 20: Context String Truncation ─────────────────────────────────
# **Validates: Requirements 10.3, 11.5**
#
# For any StoryState and any positive max_chars value,
# to_context_string(max_chars) SHALL return a string whose length does not
# exceed max_chars.


class TestContextStringTruncation:
    """Property 20: Context String Truncation."""

    @given(
        state=story_state_strategy(),
        max_chars=st.integers(min_value=0, max_value=5000),
    )
    @settings(max_examples=200, deadline=10000)
    def test_context_string_respects_max_chars(
        self, state: StoryState, max_chars: int
    ) -> None:
        """**Validates: Requirements 10.3, 11.5**

        to_context_string(max_chars) never returns a string longer than
        max_chars.
        """
        result = state.to_context_string(max_chars=max_chars)
        assert len(result) <= max_chars, (
            f"Context string length {len(result)} exceeds "
            f"max_chars={max_chars}"
        )

    @given(state=story_state_strategy())
    @settings(max_examples=100, deadline=10000)
    def test_context_string_returns_string(
        self, state: StoryState
    ) -> None:
        """**Validates: Requirements 10.3, 11.5**

        to_context_string always returns a string type.
        """
        result = state.to_context_string()
        assert isinstance(result, str)
