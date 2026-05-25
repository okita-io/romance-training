"""Unit tests for PhraseEmbedder."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from romance_factory.generate.embedding_provider import EmbeddingError
from romance_factory.generate.phrase_detection.models import (
    PhraseOccurrence,
)
from romance_factory.generate.phrase_detection.phrase_embedder import (
    PhraseEmbedder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phrase(
    original: str,
    normalized: str,
    chapter: int = 0,
    paragraph: int = 0,
    char_start: int = 0,
    char_end: int = 10,
    word_count: int = 4,
) -> PhraseOccurrence:
    return PhraseOccurrence(
        original_text=original,
        normalized_text=normalized,
        chapter_index=chapter,
        paragraph_index=paragraph,
        char_start=char_start,
        char_end=char_end,
        word_count=word_count,
    )


def _mock_lancedb_engine():
    """Create a mock LanceDBEngine with an in-memory mock _db."""
    engine = MagicMock()
    db = MagicMock()
    db.table_names.return_value = []
    engine._db = db
    return engine, db


def _mock_embedding_provider(dim: int = 4):
    """Create a mock EmbeddingProvider that returns fixed-dim vectors."""
    provider = MagicMock()
    call_count = [0]

    def embed_side_effect(text: str) -> list[float]:
        call_count[0] += 1
        return [float(call_count[0])] * dim

    provider.embed.side_effect = embed_side_effect
    provider.dimensionality = dim
    provider._call_count = call_count
    return provider


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestEmbedAndStore:
    """Tests for embed_and_store method."""

    def test_empty_phrases_returns_zero(self):
        engine, db = _mock_lancedb_engine()
        provider = _mock_embedding_provider()
        embedder = PhraseEmbedder(engine)

        result = embedder.embed_and_store([], provider)

        assert result == 0
        provider.embed.assert_not_called()

    def test_single_phrase_stored(self):
        engine, db = _mock_lancedb_engine()
        provider = _mock_embedding_provider()
        embedder = PhraseEmbedder(engine)

        phrases = [_make_phrase("Hello World Foo Bar", "hello world foo bar")]
        result = embedder.embed_and_store(phrases, provider)

        assert result == 1
        provider.embed.assert_called_once_with("Hello World Foo Bar")
        db.create_table.assert_called_once()
        call_args = db.create_table.call_args
        assert call_args[0][0] == "phrase_ngrams"
        rows = call_args[0][1]
        assert len(rows) == 1
        assert rows[0]["text"] == "Hello World Foo Bar"
        assert rows[0]["normalized_text"] == "hello world foo bar"

    def test_deduplication_by_normalized_text(self):
        engine, db = _mock_lancedb_engine()
        provider = _mock_embedding_provider()
        embedder = PhraseEmbedder(engine)

        phrases = [
            _make_phrase("Hello World Foo Bar", "hello world foo bar", chapter=0),
            _make_phrase("hello world foo bar", "hello world foo bar", chapter=1),
            _make_phrase("HELLO WORLD FOO BAR", "hello world foo bar", chapter=2),
        ]
        result = embedder.embed_and_store(phrases, provider)

        assert result == 1
        # Only one embedding call for the deduplicated phrase
        assert provider.embed.call_count == 1

    def test_multiple_unique_phrases(self):
        engine, db = _mock_lancedb_engine()
        provider = _mock_embedding_provider()
        embedder = PhraseEmbedder(engine)

        phrases = [
            _make_phrase("Alpha Beta Gamma Delta", "alpha beta gamma delta"),
            _make_phrase("One Two Three Four", "one two three four"),
            _make_phrase("Red Green Blue Yellow", "red green blue yellow"),
        ]
        result = embedder.embed_and_store(phrases, provider)

        assert result == 3
        assert provider.embed.call_count == 3

    def test_drops_existing_collection(self):
        engine, db = _mock_lancedb_engine()
        db.table_names.return_value = ["phrase_ngrams"]
        provider = _mock_embedding_provider()
        embedder = PhraseEmbedder(engine)

        phrases = [_make_phrase("Hello World Foo Bar", "hello world foo bar")]
        embedder.embed_and_store(phrases, provider)

        db.drop_table.assert_called_once_with("phrase_ngrams")

    def test_no_drop_when_collection_absent(self):
        engine, db = _mock_lancedb_engine()
        db.table_names.return_value = []
        provider = _mock_embedding_provider()
        embedder = PhraseEmbedder(engine)

        phrases = [_make_phrase("Hello World Foo Bar", "hello world foo bar")]
        embedder.embed_and_store(phrases, provider)

        db.drop_table.assert_not_called()

    def test_skip_phrase_on_embedding_error(self, caplog):
        engine, db = _mock_lancedb_engine()
        provider = MagicMock()
        call_count = [0]

        def embed_side_effect(text: str) -> list[float]:
            call_count[0] += 1
            if call_count[0] == 2:
                raise EmbeddingError("Model OOM")
            return [1.0, 2.0, 3.0, 4.0]

        provider.embed.side_effect = embed_side_effect
        embedder = PhraseEmbedder(engine)

        phrases = [
            _make_phrase("Good Phrase One Here", "good phrase one here"),
            _make_phrase("Bad Phrase Two Here", "bad phrase two here"),
            _make_phrase("Good Phrase Three Here", "good phrase three here"),
        ]

        with caplog.at_level(logging.ERROR):
            result = embedder.embed_and_store(phrases, provider)

        assert result == 2
        assert "Failed to embed phrase" in caplog.text

    def test_all_embeddings_fail_returns_zero(self, caplog):
        engine, db = _mock_lancedb_engine()
        provider = MagicMock()
        provider.embed.side_effect = EmbeddingError("Always fails")
        embedder = PhraseEmbedder(engine)

        phrases = [
            _make_phrase("Phrase One Here Now", "phrase one here now"),
            _make_phrase("Phrase Two Here Now", "phrase two here now"),
        ]

        with caplog.at_level(logging.ERROR):
            result = embedder.embed_and_store(phrases, provider)

        assert result == 0
        db.create_table.assert_not_called()

    def test_occurrence_count_in_stored_row(self):
        engine, db = _mock_lancedb_engine()
        provider = _mock_embedding_provider()
        embedder = PhraseEmbedder(engine)

        phrases = [
            _make_phrase("Hello World Foo Bar", "hello world foo bar", chapter=0),
            _make_phrase("hello world foo bar", "hello world foo bar", chapter=1),
            _make_phrase("HELLO WORLD FOO BAR", "hello world foo bar", chapter=2),
        ]
        embedder.embed_and_store(phrases, provider)

        rows = db.create_table.call_args[0][1]
        assert rows[0]["occurrence_count"] == 3

    def test_table_schema_fields(self):
        engine, db = _mock_lancedb_engine()
        provider = _mock_embedding_provider(dim=4)
        embedder = PhraseEmbedder(engine)

        phrases = [_make_phrase("Hello World Foo Bar", "hello world foo bar",
                                chapter=2, paragraph=3, char_start=10, char_end=29)]
        embedder.embed_and_store(phrases, provider)

        rows = db.create_table.call_args[0][1]
        row = rows[0]
        assert "text" in row
        assert "vector" in row
        assert "normalized_text" in row
        assert "chapter_index" in row
        assert "paragraph_index" in row
        assert "char_start" in row
        assert "char_end" in row
        assert "occurrence_count" in row
        assert row["chapter_index"] == 2
        assert row["paragraph_index"] == 3
        assert row["char_start"] == 10
        assert row["char_end"] == 29

    def test_unique_entries_populated(self):
        engine, db = _mock_lancedb_engine()
        provider = _mock_embedding_provider()
        embedder = PhraseEmbedder(engine)

        phrases = [
            _make_phrase("Alpha Beta Gamma Delta", "alpha beta gamma delta", chapter=0),
            _make_phrase("alpha beta gamma delta", "alpha beta gamma delta", chapter=1),
            _make_phrase("One Two Three Four", "one two three four"),
        ]
        embedder.embed_and_store(phrases, provider)

        entries = embedder.unique_entries
        assert len(entries) == 2
        # First entry should have 2 occurrences
        alpha_entry = next(e for e in entries if e.normalized_text == "alpha beta gamma delta")
        assert len(alpha_entry.occurrences) == 2
        assert alpha_entry.representative_text == "Alpha Beta Gamma Delta"
        assert alpha_entry.embedding_vector is not None

    def test_first_occurrence_kept_as_representative(self):
        engine, db = _mock_lancedb_engine()
        provider = _mock_embedding_provider()
        embedder = PhraseEmbedder(engine)

        phrases = [
            _make_phrase("First Version Here Now", "first version here now", chapter=0),
            _make_phrase("FIRST VERSION HERE NOW", "first version here now", chapter=1),
        ]
        embedder.embed_and_store(phrases, provider)

        rows = db.create_table.call_args[0][1]
        assert rows[0]["text"] == "First Version Here Now"


class TestCleanup:
    """Tests for cleanup method."""

    def test_cleanup_drops_collection(self):
        engine, db = _mock_lancedb_engine()
        db.table_names.return_value = ["phrase_ngrams"]
        embedder = PhraseEmbedder(engine)

        embedder.cleanup()

        db.drop_table.assert_called_once_with("phrase_ngrams")

    def test_cleanup_no_op_when_absent(self):
        engine, db = _mock_lancedb_engine()
        db.table_names.return_value = []
        embedder = PhraseEmbedder(engine)

        embedder.cleanup()

        db.drop_table.assert_not_called()

    def test_cleanup_logs_warning_on_error(self, caplog):
        engine, db = _mock_lancedb_engine()
        db.table_names.return_value = ["phrase_ngrams"]
        db.drop_table.side_effect = Exception("Permission denied")
        embedder = PhraseEmbedder(engine)

        with caplog.at_level(logging.WARNING):
            embedder.cleanup()  # Should not raise

        assert "Failed to clean up" in caplog.text


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for generating normalized text values (lowercase, single-spaced words)
_word_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz"),
    min_size=2,
    max_size=8,
)

_normalized_phrase_st = st.lists(
    _word_st, min_size=4, max_size=8
).map(lambda words: " ".join(words))


@st.composite
def phrase_occurrences_with_duplicates(draw):
    """Generate a list of PhraseOccurrence objects where some share normalized_text.

    Returns (phrases, expected_distinct_count).
    """
    # Draw a pool of distinct normalized texts (1..6)
    distinct_texts = draw(
        st.lists(_normalized_phrase_st, min_size=1, max_size=6, unique=True)
    )

    # For each distinct text, draw how many occurrences (1..4)
    phrases: list[PhraseOccurrence] = []
    for norm_text in distinct_texts:
        count = draw(st.integers(min_value=1, max_value=4))
        for i in range(count):
            # Vary original_text casing per occurrence
            original = norm_text.upper() if i % 2 == 0 else norm_text.title()
            phrases.append(
                PhraseOccurrence(
                    original_text=original,
                    normalized_text=norm_text,
                    chapter_index=i,
                    paragraph_index=i,
                    char_start=0,
                    char_end=len(original),
                    word_count=len(norm_text.split()),
                )
            )

    # Shuffle to avoid ordering bias
    shuffled = draw(st.permutations(phrases))
    return list(shuffled), len(distinct_texts)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

class TestProperty5DeduplicationBeforeEmbedding:
    """Feature: repeated-phrase-detection, Property 5: Phrase Deduplication Before Embedding

    Number of embedding computations SHALL equal number of distinct normalized_text values.
    Each unique normalized phrase SHALL be embedded exactly once.

    **Validates: Requirements 3.1, 3.3**
    """

    @given(data=phrase_occurrences_with_duplicates())
    @settings(max_examples=100)
    def test_embed_call_count_equals_distinct_normalized_texts(self, data):
        """The number of embed() calls SHALL equal the number of distinct
        normalized_text values in the input phrase list."""
        phrases, expected_distinct = data

        engine, db = _mock_lancedb_engine()
        provider = _mock_embedding_provider()
        embedder = PhraseEmbedder(engine)

        result = embedder.embed_and_store(phrases, provider)

        # embed() call count must equal distinct normalized_text count
        assert provider.embed.call_count == expected_distinct
        # stored count must also equal distinct count
        assert result == expected_distinct

    @given(data=phrase_occurrences_with_duplicates())
    @settings(max_examples=100)
    def test_each_unique_phrase_embedded_exactly_once(self, data):
        """Each unique normalized phrase SHALL be embedded exactly once —
        verified by inspecting the texts passed to embed()."""
        phrases, expected_distinct = data

        engine, db = _mock_lancedb_engine()
        # Track which texts were passed to embed()
        embed_calls: list[str] = []
        provider = MagicMock()

        def tracking_embed(text: str) -> list[float]:
            embed_calls.append(text)
            return [1.0, 2.0, 3.0, 4.0]

        provider.embed.side_effect = tracking_embed
        embedder = PhraseEmbedder(engine)

        embedder.embed_and_store(phrases, provider)

        # Number of embed calls equals distinct normalized texts
        assert len(embed_calls) == expected_distinct

        # Each call should correspond to a unique normalized_text
        # (the embedder embeds the representative original_text for each unique normalized_text)
        distinct_normalized = {p.normalized_text for p in phrases}
        assert len(embed_calls) == len(distinct_normalized)

        # No two embed calls should be for the same normalized phrase
        # Build a map from original_text -> normalized_text for the calls
        norm_of_calls = set()
        for call_text in embed_calls:
            # Find the normalized_text for this original_text
            matching = [p for p in phrases if p.original_text == call_text]
            assert len(matching) > 0, f"embed() called with unknown text: {call_text}"
            norm_of_calls.add(matching[0].normalized_text)

        assert len(norm_of_calls) == expected_distinct


class TestProperty6PhraseStorageRoundTrip:
    """Feature: repeated-phrase-detection, Property 6: Phrase Storage Round-Trip

    For any phrase stored in the phrase_ngrams LanceDB collection and then
    retrieved by exact normalized text match, the returned original_text
    (representative text) SHALL be identical to the text that was stored.

    **Validates: Requirements 3.6**
    """

    @given(data=phrase_occurrences_with_duplicates())
    @settings(max_examples=100)
    def test_unique_entries_representative_text_matches_first_occurrence(self, data):
        """For each unique phrase stored, the unique_entries property SHALL
        contain an entry whose representative_text matches the first
        occurrence's original_text for that normalized_text."""
        phrases, expected_distinct = data

        engine, db = _mock_lancedb_engine()
        provider = _mock_embedding_provider()
        embedder = PhraseEmbedder(engine)

        embedder.embed_and_store(phrases, provider)

        entries = embedder.unique_entries
        assert len(entries) == expected_distinct

        # Build expected: first occurrence per normalized_text (in input order)
        first_occurrence: dict[str, str] = {}
        for p in phrases:
            if p.normalized_text not in first_occurrence:
                first_occurrence[p.normalized_text] = p.original_text

        for entry in entries:
            expected_rep = first_occurrence[entry.normalized_text]
            assert entry.representative_text == expected_rep, (
                f"representative_text mismatch for '{entry.normalized_text}': "
                f"got '{entry.representative_text}', expected '{expected_rep}'"
            )

    @given(data=phrase_occurrences_with_duplicates())
    @settings(max_examples=100)
    def test_stored_row_text_matches_representative_text(self, data):
        """The stored row's 'text' field in LanceDB SHALL match the
        representative_text from unique_entries for each normalized_text."""
        phrases, expected_distinct = data

        engine, db = _mock_lancedb_engine()
        provider = _mock_embedding_provider()
        embedder = PhraseEmbedder(engine)

        embedder.embed_and_store(phrases, provider)

        # Inspect the rows passed to db.create_table
        assert db.create_table.called
        rows = db.create_table.call_args[0][1]
        assert len(rows) == expected_distinct

        # Build a lookup from normalized_text -> stored row text
        stored_text_by_norm: dict[str, str] = {}
        for row in rows:
            stored_text_by_norm[row["normalized_text"]] = row["text"]

        # Each unique_entry's representative_text must match the stored row text
        for entry in embedder.unique_entries:
            stored_text = stored_text_by_norm[entry.normalized_text]
            assert stored_text == entry.representative_text, (
                f"Stored row text mismatch for '{entry.normalized_text}': "
                f"row text='{stored_text}', representative='{entry.representative_text}'"
            )
