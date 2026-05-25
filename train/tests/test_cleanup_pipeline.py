"""Property-based tests for the cleanup pipeline.

Uses hypothesis to verify universal correctness properties of
run_cleanup_pipeline across arbitrary text inputs and artifact types.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from romance_factory.story_core.cleanup_pipeline import (
    CleanupResult,
    run_cleanup_pipeline,
)

# ── Strategies ──────────────────────────────────────────────────────────────

VALID_ARTIFACT_TYPES = ["author_profile", "character_web", "outline", "act", "chapter"]

STANDARD_PASSES = [
    "mojibake",
    "glued_words",
    "repeated_passage",
    "anti_patterns",
    "anti_slop",
]

artifact_type_strategy = st.sampled_from(VALID_ARTIFACT_TYPES)

# Generate non-empty text: printable strings that represent plausible prose.
# We use text() with a reasonable alphabet to avoid pathological unicode that
# would crash upstream modules unrelated to the property under test.
text_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z", "S"),
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=500,
)


# ── Property 1: Cleanup Pipeline Completeness ──────────────────────────────
# **Validates: Requirements 1.1, 1.2, 1.5**
#
# For any non-empty text artifact and any artifact type in
# {author_profile, character_web, outline, act, chapter}, running the cleanup
# pipeline SHALL produce a CleanupResult where:
#   (a) passes_applied contains all five standard passes
#       (mojibake, glued_words, repeated_passage, anti_patterns, anti_slop) in that order
#   (b) issues_found equals issues_fixed for all auto-fixable categories


class TestCleanupPipelineCompleteness:
    """Property 1: Cleanup Pipeline Completeness."""

    @given(text=text_strategy, artifact_type=artifact_type_strategy)
    @settings(max_examples=100, deadline=30000)
    def test_all_standard_passes_present_in_order(
        self, text: str, artifact_type: str
    ) -> None:
        """**Validates: Requirements 1.1, 1.2**

        For any non-empty text and valid artifact_type, the result's
        passes_applied must contain the five standard passes in order.
        """
        result = run_cleanup_pipeline(text, artifact_type=artifact_type)

        assert isinstance(result, CleanupResult)

        # Extract only the standard passes from passes_applied (outline adds extras).
        standard_in_result = [p for p in result.passes_applied if p in STANDARD_PASSES]

        assert standard_in_result == STANDARD_PASSES, (
            f"Expected standard passes {STANDARD_PASSES} in order, "
            f"got {standard_in_result} (full: {result.passes_applied})"
        )

    @given(text=text_strategy, artifact_type=artifact_type_strategy)
    @settings(max_examples=100, deadline=30000)
    def test_issues_found_equals_issues_fixed(
        self, text: str, artifact_type: str
    ) -> None:
        """**Validates: Requirement 1.5**

        For any non-empty text and valid artifact_type, the result must
        have issues_found == issues_fixed (all auto-fixable issues are fixed).
        """
        result = run_cleanup_pipeline(text, artifact_type=artifact_type)

        assert result.issues_found == result.issues_fixed, (
            f"issues_found ({result.issues_found}) != "
            f"issues_fixed ({result.issues_fixed})"
        )


# ── Property 2: Outline Cleanup Includes Romance and Foreshadowing Passes ──
# **Validates: Requirement 1.3**
#
# For any text artifact processed with artifact_type "outline", the cleanup
# pipeline SHALL produce a CleanupResult where passes_applied contains
# romance_alignment and foreshadowing in addition to the five standard passes.

OUTLINE_EXTRA_PASSES = ["romance_alignment", "foreshadowing"]


class TestOutlineCleanupIncludesRomanceAndForeshadowing:
    """Property 2: Outline Cleanup Includes Romance and Foreshadowing Passes."""

    @given(text=text_strategy)
    @settings(max_examples=100, deadline=30000)
    def test_outline_includes_romance_alignment_and_foreshadowing(
        self, text: str
    ) -> None:
        """**Validates: Requirement 1.3**

        For any non-empty text with artifact_type "outline", passes_applied
        must contain romance_alignment and foreshadowing in addition to the
        five standard passes.
        """
        result = run_cleanup_pipeline(text, artifact_type="outline")

        assert isinstance(result, CleanupResult)

        # All standard passes must still be present.
        standard_in_result = [p for p in result.passes_applied if p in STANDARD_PASSES]
        assert standard_in_result == STANDARD_PASSES, (
            f"Expected standard passes {STANDARD_PASSES} in order, "
            f"got {standard_in_result}"
        )

        # Outline-specific passes must also be present.
        for extra_pass in OUTLINE_EXTRA_PASSES:
            assert extra_pass in result.passes_applied, (
                f"Expected '{extra_pass}' in passes_applied for outline, "
                f"got {result.passes_applied}"
            )


# ── Property 3: Cleanup Pipeline Termination ───────────────────────────────
# **Validates: Requirement 1.4**
#
# For any input text, the cleanup pipeline SHALL terminate within a bounded
# maximum number of full passes, even when fixes introduce new issues.


class TestCleanupPipelineTermination:
    """Property 3: Cleanup Pipeline Termination."""

    @given(text=text_strategy, artifact_type=artifact_type_strategy)
    @settings(max_examples=100, deadline=30000)
    def test_pipeline_always_terminates_and_returns_result(
        self, text: str, artifact_type: str
    ) -> None:
        """**Validates: Requirement 1.4**

        For any non-empty text and valid artifact_type, the pipeline must
        terminate (return a CleanupResult) within a bounded number of fix
        iterations. If residual issues remain after the maximum passes,
        the warning field must be set.
        """
        result = run_cleanup_pipeline(text, artifact_type=artifact_type)

        # Pipeline must always terminate and return a valid result.
        assert isinstance(result, CleanupResult)
        assert isinstance(result.text, str)
        assert isinstance(result.original_text, str)

        # If there are residual issues (issues that couldn't be fully resolved
        # within the max passes), the warning field must be set.
        # Re-run a single pass on the output to check for residual issues.
        from romance_factory.story_core.cleanup_pipeline import _run_single_pass

        verify = _run_single_pass(result.text, artifact_type)
        if verify.issues_found > 0:
            assert result.warning is not None, (
                f"Pipeline output still has {verify.issues_found} issues "
                f"but warning is None — max-pass cap was not signalled."
            )
