"""Unit tests for to_context_string() progression sections (Task 2.5)."""

from romance_factory.story_core.story_state import (
    StoryState,
    CommittedDecision,
    IrreversibleOutcome,
)


def test_context_string_includes_committed_decisions():
    """Req 8.1: COMMITTED DECISIONS section with last 10 entries."""
    state = StoryState(
        committed_decisions=[
            CommittedDecision(
                description=f"Decision {i}",
                character="Alice",
                chapter=i,
                act=1,
                category="contract",
            )
            for i in range(1, 4)
        ],
    )
    ctx = state.to_context_string(max_chars=5000)
    assert "COMMITTED DECISIONS:" in ctx
    assert "Ch1/Act1: Decision 1 (contract)" in ctx
    assert "Ch2/Act1: Decision 2 (contract)" in ctx
    assert "Ch3/Act1: Decision 3 (contract)" in ctx


def test_context_string_includes_irreversible_outcomes():
    """Req 8.2: IRREVERSIBLE OUTCOMES section with last 10 entries."""
    state = StoryState(
        irreversible_outcomes=[
            IrreversibleOutcome(
                description="Lease expired",
                chapter=4,
                act=3,
                outcome_type="deadline_passed",
            ),
        ],
    )
    ctx = state.to_context_string(max_chars=5000)
    assert "IRREVERSIBLE OUTCOMES:" in ctx
    assert "Ch4/Act3: Lease expired (deadline_passed)" in ctx


def test_context_string_includes_timeline_position():
    """Req 8.3: timeline_position value in context."""
    state = StoryState(_timeline_position=14)
    ctx = state.to_context_string(max_chars=5000)
    assert "Timeline Position: 14" in ctx


def test_context_string_timeline_position_zero():
    """Timeline position shown even when 0."""
    state = StoryState()
    ctx = state.to_context_string(max_chars=5000)
    assert "Timeline Position: 0" in ctx


def test_context_string_no_empty_committed_decisions_section():
    """Don't add COMMITTED DECISIONS header when list is empty."""
    state = StoryState()
    ctx = state.to_context_string(max_chars=5000)
    assert "COMMITTED DECISIONS:" not in ctx


def test_context_string_no_empty_irreversible_outcomes_section():
    """Don't add IRREVERSIBLE OUTCOMES header when list is empty."""
    state = StoryState()
    ctx = state.to_context_string(max_chars=5000)
    assert "IRREVERSIBLE OUTCOMES:" not in ctx


def test_context_string_limits_to_last_10_decisions():
    """Only the last 10 committed decisions should appear."""
    state = StoryState(
        committed_decisions=[
            CommittedDecision(
                description=f"Decision-{i:03d}",
                character="Alice",
                chapter=i,
                act=1,
                category="contract",
            )
            for i in range(1, 16)  # 15 decisions
        ],
    )
    ctx = state.to_context_string(max_chars=10000)
    # First 5 should be excluded (only last 10 shown)
    for i in range(1, 6):
        assert f"Decision-{i:03d}" not in ctx
    # Last 10 should be present
    for i in range(6, 16):
        assert f"Decision-{i:03d}" in ctx


def test_context_string_limits_to_last_10_outcomes():
    """Only the last 10 irreversible outcomes should appear."""
    state = StoryState(
        irreversible_outcomes=[
            IrreversibleOutcome(
                description=f"Outcome-{i:03d}",
                chapter=i,
                act=1,
                outcome_type="action_taken",
            )
            for i in range(1, 13)  # 12 outcomes
        ],
    )
    ctx = state.to_context_string(max_chars=10000)
    # First 2 should be excluded
    assert "Outcome-001" not in ctx
    assert "Outcome-002" not in ctx
    # Last 10 should be present
    for i in range(3, 13):
        assert f"Outcome-{i:03d}" in ctx
