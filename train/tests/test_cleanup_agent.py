"""Property-based tests for the Cleanup Agent.

Feature: lancedb-rag-pipeline-v2, Property 9: Cleanup Mode Correctness

Tests:
  - Property 9: Cleanup Mode Correctness
    Validates: Requirements 7.3, 7.4
"""

from __future__ import annotations

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from romance_factory.story_core.cleanup_pipeline import CleanupResult
from romance_factory.generate.agents.cleanup import CleanupAgent


# ---------------------------------------------------------------------------
# Property 9: Cleanup Mode Correctness
# ---------------------------------------------------------------------------


class TestCleanupModeCorrectness:
    """Feature: lancedb-rag-pipeline-v2, Property 9: Cleanup Mode Correctness

    **Validates: Requirements 7.3, 7.4**

    For any input text, the Cleanup Agent in Lightweight_Cleanup mode SHALL
    run mojibake repair, optional glued-word repair, then a repeated-passage
    scan (no anti-pattern or anti-slop). In Full_Cleanup mode, the Cleanup
    Agent SHALL execute the full cleanup pipeline (including glued_words).
    """

    @given(text=st.text(min_size=1, max_size=500))
    @settings(max_examples=100)
    def test_lightweight_calls_only_bakemoji_and_repeated(self, text: str) -> None:
        """Lightweight mode runs fix_prose_mojibake, repair_prose_glued_words, and
        scan_repeated_passages_for_acts — never run_cleanup_pipeline."""
        agent = CleanupAgent()

        with (
            patch(
                "romance_factory.generate.agents.cleanup.fix_prose_mojibake",
                return_value=(text, 0, 0),
            ) as mock_mojibake,
            patch(
                "romance_factory.generate.agents.cleanup.strip_llm_trailing_annotations",
                side_effect=lambda t: t,
            ) as _mock_strip,
            patch(
                "romance_factory.generate.agents.cleanup.fix_tight_punctuation_spacing",
                return_value=(text, 0),
            ) as mock_tight,
            patch(
                "romance_factory.generate.agents.cleanup.repair_prose_glued_words",
                return_value=(text, 0),
            ) as mock_glued,
            patch(
                "romance_factory.generate.agents.cleanup.scan_repeated_passages_for_acts",
                return_value={"found": False, "clusters": []},
            ) as mock_repeated,
            patch(
                "romance_factory.generate.agents.cleanup.run_cleanup_pipeline",
            ) as mock_full_pipeline,
        ):
            result = agent.lightweight_cleanup(text)

            mock_mojibake.assert_called_once_with(text)
            mock_tight.assert_called_once_with(text)
            mock_glued.assert_called_once_with(
                text,
                extra_names=None,
                story_path=None,
            )
            mock_repeated.assert_called_once_with([text])

            mock_full_pipeline.assert_not_called()

            assert result == text

    @given(text=st.text(min_size=1, max_size=500))
    @settings(max_examples=100)
    def test_full_calls_run_cleanup_pipeline(self, text: str) -> None:
        """Full mode delegates entirely to run_cleanup_pipeline — it does
        NOT call fix_prose_mojibake or scan_repeated_passages_for_acts
        directly (those are called internally by the pipeline)."""
        agent = CleanupAgent()

        mock_result = CleanupResult(text=text, original_text=text)

        with (
            patch(
                "romance_factory.generate.agents.cleanup.run_cleanup_pipeline",
                return_value=mock_result,
            ) as mock_full_pipeline,
            patch(
                "romance_factory.generate.agents.cleanup.fix_prose_mojibake",
            ) as mock_mojibake,
            patch(
                "romance_factory.generate.agents.cleanup.scan_repeated_passages_for_acts",
            ) as mock_repeated,
        ):
            result = agent.full_cleanup(text)

            # full pipeline MUST be called
            mock_full_pipeline.assert_called_once_with(
                text=text,
                artifact_type="act",
                author_profile=None,
                glued_word_custom_names=None,
                story_path=None,
            )

            # bakemoji and repeated passage scan MUST NOT be called directly
            mock_mojibake.assert_not_called()
            mock_repeated.assert_not_called()

            # returns the CleanupResult from the pipeline
            assert result is mock_result
