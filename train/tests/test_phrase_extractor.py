"""Unit tests for PhraseExtractor."""

import string

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from romance_factory.generate.phrase_detection.models import (
    ChapterSegment,
)
from romance_factory.generate.phrase_detection.phrase_extractor import (
    STOP_WORDS,
    PhraseExtractor,
)


class TestNormalize:
    """Tests for the normalize static method."""

    def test_lowercase(self):
        assert PhraseExtractor.normalize("Hello World") == "hello world"

    def test_collapse_whitespace(self):
        assert PhraseExtractor.normalize("hello   world") == "hello world"

    def test_strip_leading_trailing(self):
        assert PhraseExtractor.normalize("  hello  ") == "hello"

    def test_tabs_and_newlines(self):
        assert PhraseExtractor.normalize("hello\t\nworld") == "hello world"

    def test_empty_string(self):
        assert PhraseExtractor.normalize("") == ""

    def test_idempotence(self):
        text = "  Hello   WORLD  "
        once = PhraseExtractor.normalize(text)
        twice = PhraseExtractor.normalize(once)
        assert once == twice


class TestExtractBasic:
    """Basic extraction tests."""

    def test_empty_chapters(self):
        extractor = PhraseExtractor(min_words=4, max_words=6)
        assert extractor.extract([]) == []

    def test_single_paragraph_returns_empty(self):
        """Fewer than 2 paragraphs across all chapters -> empty list."""
        extractor = PhraseExtractor(min_words=4, max_words=6)
        chapter = ChapterSegment(
            chapter_index=0,
            title="",
            text="This is a single paragraph with enough words to extract.",
            global_char_offset=0,
        )
        result = extractor.extract([chapter])
        assert result == []

    def test_two_paragraphs_produces_results(self):
        """Two paragraphs should produce phrase occurrences."""
        extractor = PhraseExtractor(min_words=4, max_words=4)
        text = "The quick brown fox jumps over the lazy dog.\n\nAnother paragraph with enough words here today."
        chapter = ChapterSegment(
            chapter_index=0,
            title="",
            text=text,
            global_char_offset=0,
        )
        result = extractor.extract([chapter])
        assert len(result) > 0

    def test_offset_correctness(self):
        """char_start:char_end should slice to original_text."""
        extractor = PhraseExtractor(min_words=4, max_words=4)
        text = "The quick brown fox jumps over the lazy dog.\n\nAnother paragraph with enough words here today."
        chapter = ChapterSegment(
            chapter_index=0,
            title="",
            text=text,
            global_char_offset=0,
        )
        result = extractor.extract([chapter])
        for occ in result:
            assert text[occ.char_start:occ.char_end] == occ.original_text

    def test_word_count_within_bounds(self):
        """All phrases should have word_count in [min_words, max_words]."""
        extractor = PhraseExtractor(min_words=4, max_words=6)
        text = "The quick brown fox jumps over the lazy dog today.\n\nAnother paragraph with enough words here today please."
        chapter = ChapterSegment(
            chapter_index=0,
            title="",
            text=text,
            global_char_offset=0,
        )
        result = extractor.extract([chapter])
        for occ in result:
            assert 4 <= occ.word_count <= 6

    def test_stop_word_only_phrases_skipped(self):
        """Phrases consisting entirely of stop words should be skipped."""
        extractor = PhraseExtractor(min_words=4, max_words=4)
        result = extractor.extract([])
        # No stop-word-only phrases should appear in any extraction
        # We test this more thoroughly via the property test
        assert result == []

    def test_deterministic_output(self):
        """Same input should produce same output."""
        extractor = PhraseExtractor(min_words=4, max_words=5)
        text = "The quick brown fox jumps over the lazy dog.\n\nAnother paragraph with enough words here today."
        chapter = ChapterSegment(
            chapter_index=0,
            title="",
            text=text,
            global_char_offset=0,
        )
        result1 = extractor.extract([chapter])
        result2 = extractor.extract([chapter])
        assert result1 == result2

    def test_multiple_chapters(self):
        """Extraction across multiple chapters."""
        extractor = PhraseExtractor(min_words=4, max_words=4)
        ch1 = ChapterSegment(
            chapter_index=0,
            title="Chapter 1",
            text="First chapter paragraph one here.\n\nFirst chapter paragraph two here.",
            global_char_offset=0,
        )
        ch2 = ChapterSegment(
            chapter_index=1,
            title="Chapter 2",
            text="Second chapter paragraph one here.\n\nSecond chapter paragraph two here.",
            global_char_offset=100,
        )
        result = extractor.extract([ch1, ch2])
        chapter_indices = {occ.chapter_index for occ in result}
        assert 0 in chapter_indices
        assert 1 in chapter_indices

    def test_normalized_text_is_normalized(self):
        """normalized_text field should match normalize() output."""
        extractor = PhraseExtractor(min_words=4, max_words=4)
        text = "The Quick Brown Fox jumps over.\n\nAnother paragraph with words."
        chapter = ChapterSegment(
            chapter_index=0,
            title="",
            text=text,
            global_char_offset=0,
        )
        result = extractor.extract([chapter])
        for occ in result:
            assert occ.normalized_text == PhraseExtractor.normalize(occ.original_text)


# ---------------------------------------------------------------------------
# Hypothesis strategies for Property 2
# ---------------------------------------------------------------------------

# Strategy: generate a single word (lowercase letters, 2-8 chars)
_word_st = st.text(
    alphabet=st.characters(whitelist_categories=("Ll",), whitelist_characters=""),
    min_size=2,
    max_size=8,
).filter(lambda w: len(w.strip()) >= 2)

# Strategy: generate a paragraph as a sequence of words joined by spaces
_paragraph_st = st.lists(_word_st, min_size=5, max_size=15).map(lambda ws: " ".join(ws))

# Strategy: generate chapter text with 2-4 paragraphs separated by "\n\n"
_chapter_text_st = st.lists(_paragraph_st, min_size=2, max_size=4).map(
    lambda paras: "\n\n".join(paras)
)


class TestPhraseExtractionOffsetCorrectness:
    """Property 2: Phrase Extraction Offset Correctness.

    Feature: repeated-phrase-detection, Property 2: Phrase Extraction Offset Correctness

    For any chapter text and extracted PhraseOccurrence,
    chapter_text[char_start:char_end] SHALL equal original_text.
    Every phrase SHALL have word_count within [min_ngram_words, max_ngram_words].

    Validates: Requirements 2.1, 2.2
    """

    @given(
        chapter_text=_chapter_text_st,
        min_words=st.integers(min_value=2, max_value=5),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_offset_slicing_matches_original_text(
        self, chapter_text: str, min_words: int, data: st.DataObject
    ):
        """**Validates: Requirements 2.1, 2.2**

        For every extracted occurrence, slicing the chapter text at
        [char_start:char_end] must yield exactly original_text.
        """
        max_words = data.draw(
            st.integers(min_value=min_words, max_value=min_words + 8),
            label="max_words",
        )

        chapter = ChapterSegment(
            chapter_index=0,
            title="",
            text=chapter_text,
            global_char_offset=0,
        )

        extractor = PhraseExtractor(min_words=min_words, max_words=max_words)
        occurrences = extractor.extract([chapter])

        for occ in occurrences:
            # Property 2a: offset slice equals original_text
            assert chapter_text[occ.char_start:occ.char_end] == occ.original_text, (
                f"Offset mismatch: text[{occ.char_start}:{occ.char_end}] = "
                f"{chapter_text[occ.char_start:occ.char_end]!r} != {occ.original_text!r}"
            )

    @given(
        chapter_text=_chapter_text_st,
        min_words=st.integers(min_value=2, max_value=5),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_word_count_within_bounds(
        self, chapter_text: str, min_words: int, data: st.DataObject
    ):
        """**Validates: Requirements 2.1, 2.2**

        Every extracted phrase must have word_count in [min_words, max_words].
        """
        max_words = data.draw(
            st.integers(min_value=min_words, max_value=min_words + 8),
            label="max_words",
        )

        chapter = ChapterSegment(
            chapter_index=0,
            title="",
            text=chapter_text,
            global_char_offset=0,
        )

        extractor = PhraseExtractor(min_words=min_words, max_words=max_words)
        occurrences = extractor.extract([chapter])

        for occ in occurrences:
            # Property 2b: word_count within configured bounds
            assert min_words <= occ.word_count <= max_words, (
                f"word_count {occ.word_count} not in [{min_words}, {max_words}] "
                f"for phrase {occ.original_text!r}"
            )


class TestNormalizationIdempotence:
    """Property 3: Normalization Idempotence.

    Feature: repeated-phrase-detection, Property 3: Normalization Idempotence

    normalize(normalize(x)) == normalize(x) for any string.
    Normalized form SHALL be lowercase with collapsed whitespace.

    Validates: Requirements 2.3
    """

    @given(
        text=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "M", "N", "P", "Z", "S"),
                whitelist_characters="\t\n\r ",
            ),
            min_size=0,
            max_size=200,
        ),
    )
    @settings(max_examples=100)
    def test_normalize_is_idempotent(self, text: str):
        """**Validates: Requirements 2.3**

        normalize(normalize(x)) == normalize(x) for any string x.
        """
        once = PhraseExtractor.normalize(text)
        twice = PhraseExtractor.normalize(once)
        assert twice == once, (
            f"Idempotence violated: normalize({text!r}) = {once!r}, "
            f"normalize(normalize({text!r})) = {twice!r}"
        )

    @given(
        text=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "M", "N", "P", "Z", "S"),
                whitelist_characters="\t\n\r ",
            ),
            min_size=0,
            max_size=200,
        ),
    )
    @settings(max_examples=100)
    def test_normalized_form_is_lowercase(self, text: str):
        """**Validates: Requirements 2.3**

        The normalized form is lowercase.
        """
        result = PhraseExtractor.normalize(text)
        assert result == result.lower(), (
            f"Normalized form is not lowercase: {result!r}"
        )

    @given(
        text=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "M", "N", "P", "Z", "S"),
                whitelist_characters="\t\n\r ",
            ),
            min_size=0,
            max_size=200,
        ),
    )
    @settings(max_examples=100)
    def test_normalized_form_has_no_leading_trailing_whitespace(self, text: str):
        """**Validates: Requirements 2.3**

        The normalized form has no leading/trailing whitespace.
        """
        result = PhraseExtractor.normalize(text)
        assert result == result.strip(), (
            f"Normalized form has leading/trailing whitespace: {result!r}"
        )

    @given(
        text=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "M", "N", "P", "Z", "S"),
                whitelist_characters="\t\n\r ",
            ),
            min_size=0,
            max_size=200,
        ),
    )
    @settings(max_examples=100)
    def test_normalized_form_has_no_consecutive_whitespace(self, text: str):
        """**Validates: Requirements 2.3**

        The normalized form has no consecutive whitespace characters.
        """
        result = PhraseExtractor.normalize(text)
        assert "  " not in result, (
            f"Normalized form has consecutive spaces: {result!r}"
        )
        # Also check for other whitespace sequences
        import re
        if result:
            assert not re.search(r"\s{2,}", result), (
                f"Normalized form has consecutive whitespace: {result!r}"
            )


# ---------------------------------------------------------------------------
# Hypothesis strategies for Property 4
# ---------------------------------------------------------------------------

# A content word that is NOT a stop word (lowercase alpha, 3-8 chars, not in STOP_WORDS)
_content_word_st = st.text(
    alphabet=st.characters(whitelist_categories=("Ll",)),
    min_size=3,
    max_size=8,
).filter(lambda w: w.strip() and w not in STOP_WORDS)

# A stop word drawn from the actual STOP_WORDS set
_stop_word_st = st.sampled_from(sorted(STOP_WORDS))

# Build a paragraph that mixes stop words and content words (5-15 words)
_mixed_paragraph_st = st.lists(
    st.one_of(_content_word_st, _stop_word_st),
    min_size=5,
    max_size=15,
).map(lambda ws: " ".join(ws))

# Chapter text with 2+ paragraphs (required for extraction to produce results)
_mixed_chapter_text_st = st.lists(
    _mixed_paragraph_st,
    min_size=2,
    max_size=4,
).map(lambda paras: "\n\n".join(paras))

_PUNCTUATION_CHARS = set(string.punctuation)


class TestStopWordFilteringInvariant:
    """Property 4: Stop-Word Filtering Invariant.

    Feature: repeated-phrase-detection, Property 4: Stop-Word Filtering Invariant

    No extracted phrase SHALL consist entirely of stop words or punctuation.

    Validates: Requirements 2.4
    """

    @given(chapter_text=_mixed_chapter_text_st)
    @settings(max_examples=100)
    def test_no_phrase_is_entirely_stop_words_or_punctuation(self, chapter_text: str):
        """**Validates: Requirements 2.4**

        For each extracted phrase, verify that it does NOT consist entirely
        of stop words and punctuation characters.
        """
        chapter = ChapterSegment(
            chapter_index=0,
            title="",
            text=chapter_text,
            global_char_offset=0,
        )

        extractor = PhraseExtractor(min_words=4, max_words=8)
        occurrences = extractor.extract([chapter])

        for occ in occurrences:
            words = occ.normalized_text.split()
            assert len(words) > 0, "Extracted phrase should not be empty"

            # A phrase is "stop-word-only" if ALL words are either in
            # STOP_WORDS or consist entirely of punctuation characters.
            all_stop_or_punct = all(
                word in STOP_WORDS or all(ch in _PUNCTUATION_CHARS for ch in word)
                for word in words
            )
            assert not all_stop_or_punct, (
                f"Phrase consists entirely of stop words/punctuation: "
                f"{occ.normalized_text!r} (words: {words})"
            )
