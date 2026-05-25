"""Bug condition exploration tests for the repetitive acts fix.

**Property 1: Bug Condition** — Unfiltered Previous Acts Retrieval and Unlabelled Formatting

These tests encode the EXPECTED (fixed) behavior. They MUST FAIL on the
current unfixed code, confirming the bug exists:
  - _retrieve_context() queries "acts" with NO metadata_filters and NO raw_where
  - _format_context_section() concatenates text without chapter/act labels
  - build_act_generation_prompt() has no anti-repetition constraint

**Validates: Requirements 1.1, 1.2, 1.3**
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from romance_factory.generate.agents.act_generation import ActGenerationAgent
from romance_factory.generate.config_v2 import V2Config
from romance_factory.generate.models import (
    DocumentMetadata,
    RetrievalResult,
    RetrievedContext,
)
from romance_factory.generate.prompt_builder import PromptBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_act_result(chapter: int, act: int, text: str | None = None) -> RetrievalResult:
    """Build a RetrievalResult representing a stored act."""
    return RetrievalResult(
        text=text or f"Prose for chapter {chapter}, act {act}. The morning sun rose.",
        metadata=DocumentMetadata(
            type="act",
            chapter=chapter,
            act=act,
            characters_involved=["Alice", "Bob"],
            emotional_tone="tension",
            plot_function="rising_action",
            summary=f"Summary of ch{chapter} act{act}",
        ),
        similarity_score=0.85,
    )


def _build_mock_engine_returning_acts(previous_act_results: list[RetrievalResult]):
    """Create a mock LanceDBEngine whose query() returns *previous_act_results*
    for the "acts" collection and empty lists for everything else."""
    mock_engine = MagicMock()

    def _query_side_effect(collection, query_text, **kwargs):
        if collection == "acts":
            return previous_act_results
        return [
            RetrievalResult(
                text=f"Context for {collection}",
                metadata=DocumentMetadata(type=collection),
                similarity_score=0.9,
            )
        ]

    mock_engine.query.side_effect = _query_side_effect
    return mock_engine


# ---------------------------------------------------------------------------
# Test 1 — Filtered Retrieval
# ---------------------------------------------------------------------------


class TestFilteredRetrieval:
    """For any (chapter, act) with act > 1, _retrieve_context() should query
    the "acts" collection with metadata_filters containing {"chapter": chapter}
    AND a raw_where parameter containing f"act < {act}".

    **Validates: Requirements 1.1**

    On UNFIXED code this MUST FAIL because the query uses no metadata_filters
    and no raw_where parameter.
    """

    @given(
        chapter=st.integers(min_value=1, max_value=20),
        act=st.integers(min_value=2, max_value=7),
    )
    @settings(max_examples=50)
    def test_acts_query_uses_chapter_filter_and_act_inequality(
        self, chapter: int, act: int
    ) -> None:
        """_retrieve_context(chapter, act) must pass metadata_filters and
        raw_where to engine.query('acts', ...)."""
        # Build mock engine that returns realistic previous act results
        prev_acts = [_make_act_result(chapter, a) for a in range(1, act)]
        mock_engine = _build_mock_engine_returning_acts(prev_acts)

        builder = PromptBuilder(max_context_chars=8000)
        config = V2Config()
        agent = ActGenerationAgent(
            lancedb_engine=mock_engine,
            prompt_builder=builder,
            config=config,
        )

        agent._retrieve_context(chapter, act)

        # Find the call to engine.query("acts", ...)
        acts_calls = [
            call for call in mock_engine.query.call_args_list
            if call[0][0] == "acts"
        ]
        filtered = [
            c for c in acts_calls
            if c.kwargs.get("raw_where") and "act <" in str(c.kwargs.get("raw_where", ""))
        ]
        assert len(filtered) == 1, (
            f"Expected exactly 1 filtered 'acts' query (previous acts), got {len(filtered)} "
            f"among {len(acts_calls)} total acts queries"
        )

        call_args = filtered[0]
        _, kwargs = call_args[0], call_args[1]

        # Assert metadata_filters contains chapter filter
        metadata_filters = kwargs.get("metadata_filters", None)
        assert metadata_filters is not None, (
            "Expected metadata_filters to be passed to engine.query('acts', ...) "
            f"but got None. kwargs={kwargs}"
        )
        assert metadata_filters.get("chapter") == chapter, (
            f"Expected metadata_filters['chapter'] == {chapter}, "
            f"got {metadata_filters}"
        )

        # Assert raw_where contains act inequality
        raw_where = kwargs.get("raw_where", None)
        assert raw_where is not None, (
            "Expected raw_where parameter to be passed to engine.query('acts', ...) "
            f"but got None. kwargs={kwargs}"
        )
        assert f"act < {act}" in raw_where, (
            f"Expected raw_where to contain 'act < {act}', got '{raw_where}'"
        )


# ---------------------------------------------------------------------------
# Test 2 — Labelled Formatting
# ---------------------------------------------------------------------------


class TestLabelledFormatting:
    """build_act_generation_prompt() should produce a "Previous Acts" section
    where each act is labelled with "Chapter X, Act Y".

    **Validates: Requirements 1.2**

    On UNFIXED code this MUST FAIL because _format_context_section()
    concatenates raw text without chapter/act labels.
    """

    @given(
        chapter=st.integers(min_value=1, max_value=20),
        act=st.integers(min_value=2, max_value=7),
    )
    @settings(max_examples=50)
    def test_previous_acts_section_contains_chapter_act_labels(
        self, chapter: int, act: int
    ) -> None:
        """The prompt's Previous Acts section must contain 'Chapter X, Act Y'
        labels for each previous act."""
        # Build context with previous acts from the same chapter
        prev_acts = [_make_act_result(chapter, a) for a in range(1, act)]

        context = RetrievedContext(
            author_profile=[
                RetrievalResult(
                    text="Author writes romance.",
                    metadata=DocumentMetadata(type="author_profile"),
                    similarity_score=0.9,
                )
            ],
            previous_acts=prev_acts,
        )

        builder = PromptBuilder(max_context_chars=8000)
        prompt, _ = builder.build_act_generation_prompt(
            chapter=chapter, act=act, context=context,
        )

        # Assert each previous act has a label in the prompt
        for a in range(1, act):
            label = f"Chapter {chapter}, Act {a}"
            assert label in prompt, (
                f"Expected label '{label}' in prompt's Previous Acts section, "
                f"but it was not found."
            )


# ---------------------------------------------------------------------------
# Test 3 — Anti-Repetition Constraint
# ---------------------------------------------------------------------------


class TestAntiRepetitionConstraint:
    """build_act_generation_prompt() should include anti-repetition language
    in the constraints section.

    **Validates: Requirements 1.3**

    On UNFIXED code this MUST FAIL because the constraints list has no
    anti-repetition instruction.
    """

    @given(
        chapter=st.integers(min_value=1, max_value=20),
        act=st.integers(min_value=2, max_value=7),
    )
    @settings(max_examples=50)
    def test_prompt_contains_anti_repetition_constraint(
        self, chapter: int, act: int
    ) -> None:
        """The prompt must contain anti-repetition language such as
        'distinct opening' or 'Do not reuse'."""
        prev_acts = [_make_act_result(chapter, a) for a in range(1, act)]

        context = RetrievedContext(
            author_profile=[
                RetrievalResult(
                    text="Author writes romance.",
                    metadata=DocumentMetadata(type="author_profile"),
                    similarity_score=0.9,
                )
            ],
            previous_acts=prev_acts,
        )

        builder = PromptBuilder(max_context_chars=8000)
        prompt, _ = builder.build_act_generation_prompt(
            chapter=chapter, act=act, context=context,
        )

        prompt_lower = prompt.lower()
        has_anti_repetition = (
            "distinct opening" in prompt_lower
            or "do not reuse" in prompt_lower
            or "vary your narrative entry point" in prompt_lower
            or "do not repeat" in prompt_lower
            or "avoid reusing" in prompt_lower
        )
        assert has_anti_repetition, (
            "Expected anti-repetition language in the prompt constraints "
            "(e.g., 'distinct opening', 'Do not reuse', 'vary your narrative "
            "entry point') but none was found."
        )
