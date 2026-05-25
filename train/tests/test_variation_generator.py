"""Property and unit tests for VariationGenerator.

Feature: repeated-phrase-detection, Property 11: Variation Count Correctness
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from romance_factory.generate.phrase_detection.models import (
    ChapterSegment,
    PhraseOccurrence,
    RepetitionCluster,
)
from romance_factory.generate.phrase_detection.variation_generator import (
    VariationGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_occurrence(
    text: str,
    chapter: int,
    paragraph: int,
    char_start: int,
    char_end: int,
) -> PhraseOccurrence:
    return PhraseOccurrence(
        original_text=text,
        normalized_text=text.lower(),
        chapter_index=chapter,
        paragraph_index=paragraph,
        char_start=char_start,
        char_end=char_end,
        word_count=len(text.split()),
    )


def _mock_llm_response(num_variations: int) -> str:
    """Build a numbered-list LLM response with the requested count."""
    return "\n".join(
        f"{i + 1}. variation number {i + 1} text here"
        for i in range(num_variations)
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_word_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz"),
    min_size=3,
    max_size=8,
)

_phrase_st = st.lists(_word_st, min_size=4, max_size=8).map(
    lambda words: " ".join(words)
)


@st.composite
def cluster_and_chapters(draw):
    """Generate a RepetitionCluster with N occurrences (2..6) and matching chapters.

    Each occurrence is placed in its own chapter with text that contains the phrase.
    Returns (cluster, chapters, N).
    """
    n = draw(st.integers(min_value=2, max_value=6))
    phrase = draw(_phrase_st)

    occurrences: list[PhraseOccurrence] = []
    chapters: list[ChapterSegment] = []

    for i in range(n):
        # Build chapter text with the phrase embedded in a sentence
        prefix = f"Chapter {i} begins here. "
        suffix = " And the story continues onward."
        chapter_text = prefix + phrase + suffix

        char_start = len(prefix)
        char_end = char_start + len(phrase)

        occurrences.append(
            _make_occurrence(
                text=phrase,
                chapter=i,
                paragraph=0,
                char_start=char_start,
                char_end=char_end,
            )
        )
        chapters.append(
            ChapterSegment(
                chapter_index=i,
                title=f"Chapter {i}",
                text=chapter_text,
                global_char_offset=0,
            )
        )

    cluster = RepetitionCluster(
        cluster_id=1,
        canonical_phrase=phrase,
        occurrences=occurrences,
        similarity_scores=[0.95] * max(1, n - 1),
        avg_similarity=0.95,
    )

    return cluster, chapters, n


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

class TestProperty11VariationCountCorrectness:
    """Feature: repeated-phrase-detection, Property 11: Variation Count Correctness

    For any RepetitionCluster with N occurrences, the VariationGenerator SHALL
    produce a VariationResult containing exactly N-1 VariationMapping entries
    (one per duplicate occurrence, excluding the kept original). Each mapping
    SHALL reference a valid PhraseOccurrence from the cluster and contain a
    non-empty variation_text.

    **Validates: Requirements 5.2, 5.6**
    """

    @given(data=cluster_and_chapters())
    @settings(max_examples=100)
    @patch("romance_factory.generate.phrase_detection.variation_generator.llm_generate")
    def test_variation_count_equals_n_minus_one(self, mock_llm, data):
        """VariationResult SHALL contain exactly N-1 VariationMapping entries
        for a cluster with N occurrences."""
        cluster, chapters, n = data

        # Mock LLM to return the right number of numbered variations
        num_needed = n - 1
        mock_llm.return_value = _mock_llm_response(num_needed)

        generator = VariationGenerator(context_sentences=2)
        result = generator.generate(cluster, chapters)

        assert len(result.variations) == n - 1

    @given(data=cluster_and_chapters())
    @settings(max_examples=100)
    @patch("romance_factory.generate.phrase_detection.variation_generator.llm_generate")
    def test_each_mapping_references_valid_occurrence(self, mock_llm, data):
        """Each VariationMapping.occurrence SHALL be a valid PhraseOccurrence
        from the cluster."""
        cluster, chapters, n = data

        num_needed = n - 1
        mock_llm.return_value = _mock_llm_response(num_needed)

        generator = VariationGenerator(context_sentences=2)
        result = generator.generate(cluster, chapters)

        cluster_occurrences = set(
            (o.chapter_index, o.paragraph_index, o.char_start, o.char_end)
            for o in cluster.occurrences
        )

        for mapping in result.variations:
            occ_key = (
                mapping.occurrence.chapter_index,
                mapping.occurrence.paragraph_index,
                mapping.occurrence.char_start,
                mapping.occurrence.char_end,
            )
            assert occ_key in cluster_occurrences, (
                f"Mapping references occurrence {occ_key} not in cluster"
            )

    @given(data=cluster_and_chapters())
    @settings(max_examples=100)
    @patch("romance_factory.generate.phrase_detection.variation_generator.llm_generate")
    def test_each_mapping_has_non_empty_variation_text(self, mock_llm, data):
        """Each VariationMapping.variation_text SHALL be non-empty."""
        cluster, chapters, n = data

        num_needed = n - 1
        mock_llm.return_value = _mock_llm_response(num_needed)

        generator = VariationGenerator(context_sentences=2)
        result = generator.generate(cluster, chapters)

        for mapping in result.variations:
            assert mapping.variation_text, (
                "variation_text must be non-empty"
            )

    @given(data=cluster_and_chapters())
    @settings(max_examples=100)
    @patch("romance_factory.generate.phrase_detection.variation_generator.llm_generate")
    def test_kept_occurrence_is_first_in_document_order(self, mock_llm, data):
        """result.kept_occurrence SHALL be the first occurrence in document
        order (earliest chapter_index, then earliest char_start)."""
        cluster, chapters, n = data

        num_needed = n - 1
        mock_llm.return_value = _mock_llm_response(num_needed)

        generator = VariationGenerator(context_sentences=2)
        result = generator.generate(cluster, chapters)

        sorted_occs = sorted(
            cluster.occurrences,
            key=lambda o: (o.chapter_index, o.char_start),
        )
        first = sorted_occs[0]

        assert result.kept_occurrence.chapter_index == first.chapter_index
        assert result.kept_occurrence.char_start == first.char_start
        assert result.kept_occurrence.char_end == first.char_end


# ---------------------------------------------------------------------------
# Example-based tests
# ---------------------------------------------------------------------------

class TestLLMFailureFallback:
    """When the LLM raises an exception, variations should fall back to
    original text for each occurrence."""

    @patch("romance_factory.generate.phrase_detection.variation_generator.llm_generate")
    def test_llm_exception_uses_original_text(self, mock_llm):
        mock_llm.side_effect = RuntimeError("LLM connection refused")

        phrase = "the warm summer breeze"
        occurrences = [
            _make_occurrence(phrase, chapter=0, paragraph=0, char_start=10, char_end=31),
            _make_occurrence(phrase, chapter=1, paragraph=0, char_start=10, char_end=31),
            _make_occurrence(phrase, chapter=2, paragraph=0, char_start=10, char_end=31),
        ]
        chapters = [
            ChapterSegment(
                chapter_index=i,
                title=f"Chapter {i}",
                text=f"Prefix.   {phrase} and more text follows.",
                global_char_offset=0,
            )
            for i in range(3)
        ]
        cluster = RepetitionCluster(
            cluster_id=1,
            canonical_phrase=phrase,
            occurrences=occurrences,
            similarity_scores=[0.9, 0.9],
            avg_similarity=0.9,
        )

        generator = VariationGenerator(context_sentences=2)
        result = generator.generate(cluster, chapters)

        # Should still produce N-1 = 2 variation mappings
        assert len(result.variations) == 2
        # Each variation_text should fall back to the original text
        for mapping in result.variations:
            assert mapping.variation_text == phrase


class TestSingleOccurrenceCluster:
    """A cluster with a single occurrence needs 0 variations."""

    @patch("romance_factory.generate.phrase_detection.variation_generator.llm_generate")
    def test_single_occurrence_produces_no_variations(self, mock_llm):
        phrase = "the warm summer breeze"
        occurrences = [
            _make_occurrence(phrase, chapter=0, paragraph=0, char_start=10, char_end=31),
        ]
        chapters = [
            ChapterSegment(
                chapter_index=0,
                title="Chapter 0",
                text=f"Prefix.   {phrase} and more text follows.",
                global_char_offset=0,
            )
        ]
        cluster = RepetitionCluster(
            cluster_id=1,
            canonical_phrase=phrase,
            occurrences=occurrences,
            similarity_scores=[],
            avg_similarity=0.0,
        )

        generator = VariationGenerator(context_sentences=2)
        result = generator.generate(cluster, chapters)

        assert len(result.variations) == 0
        assert result.kept_occurrence == occurrences[0]
        # LLM should not be called at all
        mock_llm.assert_not_called()
