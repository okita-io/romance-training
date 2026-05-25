"""Tests for the PhraseReplacer module."""

from __future__ import annotations

import os
import tempfile

import pytest

from romance_factory.generate.phrase_detection.models import (
    ChapterSegment,
    ClusterReportEntry,
    PhraseOccurrence,
    ReplacementEntry,
    ReplacementReport,
    VariationMapping,
    VariationResult,
)
from romance_factory.generate.phrase_detection.phrase_replacer import (
    PhraseReplacer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _occ(
    text: str,
    chapter: int,
    para: int,
    start: int,
    *,
    normalized: str | None = None,
) -> PhraseOccurrence:
    """Shorthand to build a PhraseOccurrence."""
    return PhraseOccurrence(
        original_text=text,
        normalized_text=normalized or text.lower(),
        chapter_index=chapter,
        paragraph_index=para,
        char_start=start,
        char_end=start + len(text),
        word_count=len(text.split()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhraseReplacerNoClusters:
    """When no variations are provided, the original is copied unchanged."""

    def test_empty_variations_copies_original(self, tmp_path):
        chapters = [
            ChapterSegment(chapter_index=0, title="", text="Hello world.", global_char_offset=0),
        ]
        out = str(tmp_path / "out.txt")
        report = PhraseReplacer().replace(chapters, [], out)

        assert os.path.exists(out)
        with open(out) as f:
            assert f.read() == "Hello world."
        assert report.clusters_processed == 0
        assert report.total_replacements == 0
        assert report.cluster_details == []

    def test_empty_chapters_empty_variations(self, tmp_path):
        out = str(tmp_path / "out.txt")
        report = PhraseReplacer().replace([], [], out)
        with open(out) as f:
            assert f.read() == ""
        assert report.total_replacements == 0


class TestPhraseReplacerSingleCluster:
    """Basic replacement with a single cluster."""

    def test_replaces_second_occurrence(self, tmp_path):
        text = "She felt a warm glow inside. Later she felt a warm glow inside again."
        chapters = [
            ChapterSegment(chapter_index=0, title="", text=text, global_char_offset=0),
        ]
        # First occurrence at 8..30, second at 40..62
        kept = _occ("a warm glow inside", chapter=0, para=0, start=8)
        replaced_occ = _occ("a warm glow inside", chapter=0, para=1, start=40)

        variations = [
            VariationResult(
                cluster_id=1,
                original_phrase="a warm glow inside",
                kept_occurrence=kept,
                variations=[
                    VariationMapping(occurrence=replaced_occ, variation_text="a gentle warmth within"),
                ],
            )
        ]

        out = str(tmp_path / "out.txt")
        report = PhraseReplacer().replace(chapters, variations, out)

        with open(out) as f:
            result = f.read()

        # First occurrence preserved
        assert "She felt a warm glow inside." in result
        # Second occurrence replaced
        assert "a gentle warmth within" in result
        assert report.clusters_processed == 1
        assert report.total_replacements == 1

    def test_replacement_with_different_length(self, tmp_path):
        text = "AAAA BBBB AAAA"
        chapters = [
            ChapterSegment(chapter_index=0, title="", text=text, global_char_offset=0),
        ]
        kept = _occ("AAAA", chapter=0, para=0, start=0)
        replaced_occ = _occ("AAAA", chapter=0, para=1, start=10)

        variations = [
            VariationResult(
                cluster_id=1,
                original_phrase="AAAA",
                kept_occurrence=kept,
                variations=[
                    VariationMapping(occurrence=replaced_occ, variation_text="CCCCCCCC"),
                ],
            )
        ]

        out = str(tmp_path / "out.txt")
        PhraseReplacer().replace(chapters, variations, out)

        with open(out) as f:
            assert f.read() == "AAAA BBBB CCCCCCCC"


class TestPhraseReplacerMultipleChapters:
    """Replacements across multiple chapters."""

    def test_replacements_in_different_chapters(self, tmp_path):
        chapters = [
            ChapterSegment(chapter_index=0, title="Chapter 1\n", text="She smiled warmly. End.", global_char_offset=0),
            ChapterSegment(chapter_index=1, title="Chapter 2\n", text="She smiled warmly. Done.", global_char_offset=32),
        ]
        kept = _occ("smiled warmly", chapter=0, para=0, start=4)
        replaced_occ = _occ("smiled warmly", chapter=1, para=1, start=4)

        variations = [
            VariationResult(
                cluster_id=1,
                original_phrase="smiled warmly",
                kept_occurrence=kept,
                variations=[
                    VariationMapping(occurrence=replaced_occ, variation_text="grinned softly"),
                ],
            )
        ]

        out = str(tmp_path / "out.txt")
        report = PhraseReplacer().replace(chapters, variations, out)

        with open(out) as f:
            result = f.read()

        assert "Chapter 1\nShe smiled warmly." in result
        assert "Chapter 2\nShe grinned softly." in result
        assert report.clusters_processed == 1
        assert report.total_replacements == 1


class TestPhraseReplacerReverseOrder:
    """Multiple replacements in the same chapter use reverse order."""

    def test_two_replacements_same_chapter(self, tmp_path):
        text = "XX YY XX YY XX"
        chapters = [
            ChapterSegment(chapter_index=0, title="", text=text, global_char_offset=0),
        ]
        kept = _occ("XX", chapter=0, para=0, start=0)
        occ2 = _occ("XX", chapter=0, para=1, start=6)
        occ3 = _occ("XX", chapter=0, para=2, start=12)

        variations = [
            VariationResult(
                cluster_id=1,
                original_phrase="XX",
                kept_occurrence=kept,
                variations=[
                    VariationMapping(occurrence=occ2, variation_text="AA"),
                    VariationMapping(occurrence=occ3, variation_text="BB"),
                ],
            )
        ]

        out = str(tmp_path / "out.txt")
        PhraseReplacer().replace(chapters, variations, out)

        with open(out) as f:
            assert f.read() == "XX YY AA YY BB"


class TestPhraseReplacerOriginalUnmodified:
    """The original chapter objects must not be mutated."""

    def test_original_chapters_unchanged(self, tmp_path):
        original_text = "Hello repeated phrase. And repeated phrase again."
        chapters = [
            ChapterSegment(chapter_index=0, title="", text=original_text, global_char_offset=0),
        ]
        kept = _occ("repeated phrase", chapter=0, para=0, start=6)
        replaced_occ = _occ("repeated phrase", chapter=0, para=1, start=26)

        variations = [
            VariationResult(
                cluster_id=1,
                original_phrase="repeated phrase",
                kept_occurrence=kept,
                variations=[
                    VariationMapping(occurrence=replaced_occ, variation_text="echoed words"),
                ],
            )
        ]

        out = str(tmp_path / "out.txt")
        PhraseReplacer().replace(chapters, variations, out)

        # Original chapter text must be unchanged
        assert chapters[0].text == original_text


class TestReplacementReport:
    """Verify the report structure."""

    def test_report_has_correct_structure(self, tmp_path):
        text = "AA BB AA"
        chapters = [
            ChapterSegment(chapter_index=0, title="", text=text, global_char_offset=0),
        ]
        kept = _occ("AA", chapter=0, para=0, start=0)
        replaced_occ = _occ("AA", chapter=0, para=1, start=6)

        variations = [
            VariationResult(
                cluster_id=42,
                original_phrase="AA",
                kept_occurrence=kept,
                variations=[
                    VariationMapping(occurrence=replaced_occ, variation_text="CC"),
                ],
            )
        ]

        out = str(tmp_path / "out.txt")
        report = PhraseReplacer().replace(chapters, variations, out)

        assert report.clusters_processed == 1
        assert report.total_replacements == 1
        assert report.output_path == out
        assert len(report.cluster_details) == 1

        detail = report.cluster_details[0]
        assert detail.cluster_id == 42
        assert detail.original_phrase == "AA"
        assert detail.kept_location == (0, 0, 0)
        assert len(detail.replacements) == 1
        assert detail.replacements[0].original_text == "AA"
        assert detail.replacements[0].variation_text == "CC"
        assert detail.replacements[0].location == (0, 1, 6)


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings, assume
from hypothesis import strategies as st


def _phrase_in_chapter(phrase: str, chapter_text: str, positions: list[int]) -> bool:
    """Check that phrase appears at all given positions in chapter_text."""
    for pos in positions:
        if chapter_text[pos : pos + len(phrase)] != phrase:
            return False
    return True


@st.composite
def phrase_replacer_scenario(draw):
    """Generate a chapter text with a known phrase embedded at multiple
    non-overlapping positions, plus VariationResult data for testing.

    Returns (chapters, variations, phrase, positions, variation_texts).
    """
    # Generate a phrase of 2-5 words
    word = st.text(
        alphabet=st.characters(whitelist_categories=("L",), min_codepoint=65, max_codepoint=122),
        min_size=2,
        max_size=8,
    )
    phrase_words = draw(st.lists(word, min_size=2, max_size=5))
    phrase = " ".join(phrase_words)
    assume(len(phrase) >= 3)

    # Generate filler segments (text between phrase occurrences)
    filler = st.text(
        alphabet=st.characters(whitelist_categories=("L",), min_codepoint=65, max_codepoint=122),
        min_size=3,
        max_size=30,
    )
    # We need at least 2 occurrences (1 kept + at least 1 replaced)
    num_occurrences = draw(st.integers(min_value=2, max_value=5))
    fillers = draw(st.lists(filler, min_size=num_occurrences + 1, max_size=num_occurrences + 1))

    # Build chapter text: filler0 + phrase + filler1 + phrase + filler2 + ...
    parts = []
    positions = []
    for i in range(num_occurrences):
        parts.append(fillers[i])
        positions.append(sum(len(p) for p in parts) + len(phrase) * i + len(" ") * (2 * i))
        parts.append(phrase)
    parts.append(fillers[num_occurrences])

    # Recalculate positions precisely by building the text
    chapter_text = " ".join(parts)

    # Find actual positions of the phrase in the built text
    actual_positions = []
    search_start = 0
    for _ in range(num_occurrences):
        idx = chapter_text.find(phrase, search_start)
        if idx == -1:
            break
        actual_positions.append(idx)
        search_start = idx + len(phrase)

    assume(len(actual_positions) >= 2)

    # First occurrence is kept, rest are replaced
    kept_pos = actual_positions[0]
    replaced_positions = actual_positions[1:]

    # Generate variation texts (one per replaced occurrence)
    variation_texts = draw(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L",), min_codepoint=65, max_codepoint=122),
                min_size=1,
                max_size=30,
            ),
            min_size=len(replaced_positions),
            max_size=len(replaced_positions),
        )
    )

    chapters = [
        ChapterSegment(chapter_index=0, title="", text=chapter_text, global_char_offset=0),
    ]

    kept_occ = PhraseOccurrence(
        original_text=phrase,
        normalized_text=phrase.lower(),
        chapter_index=0,
        paragraph_index=0,
        char_start=kept_pos,
        char_end=kept_pos + len(phrase),
        word_count=len(phrase.split()),
    )

    variation_mappings = []
    for i, rpos in enumerate(replaced_positions):
        occ = PhraseOccurrence(
            original_text=phrase,
            normalized_text=phrase.lower(),
            chapter_index=0,
            paragraph_index=i + 1,
            char_start=rpos,
            char_end=rpos + len(phrase),
            word_count=len(phrase.split()),
        )
        variation_mappings.append(
            VariationMapping(occurrence=occ, variation_text=variation_texts[i])
        )

    variations = [
        VariationResult(
            cluster_id=1,
            original_phrase=phrase,
            kept_occurrence=kept_occ,
            variations=variation_mappings,
        )
    ]

    return chapters, variations, phrase, actual_positions, variation_texts


class TestProperty12FirstOccurrencePreservation:
    """Property 12: First Occurrence Preservation.

    Feature: repeated-phrase-detection, Property 12: First Occurrence Preservation

    The first occurrence in document order SHALL remain unchanged in output text.
    Only subsequent occurrences SHALL be replaced.

    **Validates: Requirements 6.1**
    """

    @given(data=phrase_replacer_scenario())
    @settings(max_examples=100)
    def test_first_occurrence_preserved(self, data):
        chapters, variations, phrase, positions, variation_texts = data
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.txt")
            PhraseReplacer().replace(chapters, variations, out)

            with open(out) as f:
                result = f.read()

        # The first occurrence position in the original text
        first_pos = positions[0]
        original_text = chapters[0].text

        # The text before the first occurrence should be unchanged
        assert result[:first_pos] == original_text[:first_pos]

        # The first occurrence itself should be preserved at its original position
        assert result[first_pos : first_pos + len(phrase)] == phrase


class TestProperty13ReplacementOffsetCorrectness:
    """Property 13: Replacement Offset Correctness.

    Feature: repeated-phrase-detection, Property 13: Replacement Offset Correctness

    For non-overlapping replacements, all text outside replacement spans
    SHALL be preserved exactly.

    **Validates: Requirements 6.2, 6.3, 6.5**
    """

    @given(data=phrase_replacer_scenario())
    @settings(max_examples=100)
    def test_text_outside_replacements_preserved(self, data):
        chapters, variations, phrase, positions, variation_texts = data
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.txt")
            PhraseReplacer().replace(chapters, variations, out)

            with open(out) as f:
                result = f.read()

        original_text = chapters[0].text

        # Build the set of replacement spans (positions[1:] are replaced)
        replaced_spans = []
        for pos in positions[1:]:
            replaced_spans.append((pos, pos + len(phrase)))

        # Sort spans by start position
        replaced_spans.sort()

        # Extract text segments outside replacement spans from original
        outside_original = []
        prev_end = 0
        for span_start, span_end in replaced_spans:
            outside_original.append(original_text[prev_end:span_start])
            prev_end = span_end
        outside_original.append(original_text[prev_end:])

        # Extract corresponding segments from result
        # We need to account for length changes from replacements
        outside_result = []
        offset_adjustment = 0
        prev_end = 0
        for i, (span_start, span_end) in enumerate(replaced_spans):
            # In the result, the segment before this replacement starts at
            # prev_end + offset_adjustment and goes to span_start + offset_adjustment
            result_seg_start = prev_end + offset_adjustment
            result_seg_end = span_start + offset_adjustment
            outside_result.append(result[result_seg_start:result_seg_end])

            # Update offset for the length change of this replacement
            replacement_len = len(variation_texts[i])
            original_len = len(phrase)
            offset_adjustment += replacement_len - original_len
            prev_end = span_end

        # Final segment after last replacement
        result_seg_start = prev_end + offset_adjustment
        outside_result.append(result[result_seg_start:])

        # All segments outside replacement spans must match exactly
        for orig_seg, result_seg in zip(outside_original, outside_result):
            assert orig_seg == result_seg, (
                f"Text outside replacement spans differs.\n"
                f"Original segment: {orig_seg!r}\n"
                f"Result segment:   {result_seg!r}"
            )


class TestProperty14ReplacementReportCompleteness:
    """Property 14: Replacement Report Completeness.

    Feature: repeated-phrase-detection, Property 14: Replacement Report Completeness

    ReplacementReport SHALL contain one ClusterReportEntry per processed cluster.
    Each entry SHALL list original phrase, kept location, and one ReplacementEntry
    per replaced occurrence.

    **Validates: Requirements 6.6**
    """

    @given(data=phrase_replacer_scenario())
    @settings(max_examples=100)
    def test_report_completeness(self, data):
        chapters, variations, phrase, positions, variation_texts = data
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.txt")
            report = PhraseReplacer().replace(chapters, variations, out)

        # One ClusterReportEntry per processed cluster
        assert report.clusters_processed == len(variations)
        assert len(report.cluster_details) == len(variations)

        for vr, detail in zip(variations, report.cluster_details):
            # Original phrase matches
            assert detail.original_phrase == vr.original_phrase

            # Kept location matches the kept occurrence
            kept = vr.kept_occurrence
            assert detail.kept_location == (
                kept.chapter_index,
                kept.paragraph_index,
                kept.char_start,
            )

            # One ReplacementEntry per replaced occurrence
            assert len(detail.replacements) == len(vr.variations)

            for vm, entry in zip(vr.variations, detail.replacements):
                occ = vm.occurrence
                assert entry.location == (
                    occ.chapter_index,
                    occ.paragraph_index,
                    occ.char_start,
                )
                assert entry.original_text == occ.original_text
                assert entry.variation_text == vm.variation_text

        # Total replacements count
        expected_total = sum(len(vr.variations) for vr in variations)
        assert report.total_replacements == expected_total

        # Output path matches
        assert report.output_path == out
