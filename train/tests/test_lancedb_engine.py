"""Property-based tests for LanceDB Engine.

Feature: lancedb-rag-pipeline-v2, Property 1: Metadata Schema Validation

Validates: Requirements 1.2, 17.1, 17.2, 17.3, 17.5
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from romance_factory.generate.lancedb_engine import (
    LanceDBEngine,
    MetadataValidationError,
)
from romance_factory.generate.models import DocumentMetadata


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

valid_metadata_strategy = st.builds(
    DocumentMetadata,
    type=st.text(min_size=1),
    chapter=st.integers(min_value=0),
    act=st.integers(min_value=0),
    characters_involved=st.lists(st.text()),
    emotional_tone=st.text(),
    plot_function=st.text(),
    summary=st.text(),
)


# ---------------------------------------------------------------------------
# Property 1: Metadata Schema Validation — valid metadata accepted
# ---------------------------------------------------------------------------


class TestMetadataSchemaValidation:
    """Feature: lancedb-rag-pipeline-v2, Property 1: Metadata Schema Validation

    Validates: Requirements 1.2, 17.1, 17.2, 17.3, 17.5
    """

    @given(metadata=valid_metadata_strategy)
    @settings(max_examples=100)
    def test_valid_metadata_accepted(self, metadata: DocumentMetadata) -> None:
        """Valid DocumentMetadata instances must NOT raise MetadataValidationError."""
        # Should not raise
        LanceDBEngine._validate_metadata(metadata)

    @given(
        chapter=st.integers(min_value=0),
        act=st.integers(min_value=0),
        characters_involved=st.lists(st.text()),
        summary=st.text(),
    )
    @settings(max_examples=100)
    def test_empty_type_rejected(
        self,
        chapter: int,
        act: int,
        characters_involved: list[str],
        summary: str,
    ) -> None:
        """Empty type string must raise MetadataValidationError."""
        metadata = DocumentMetadata(
            type="",
            chapter=chapter,
            act=act,
            characters_involved=characters_involved,
            summary=summary,
        )
        with pytest.raises(MetadataValidationError):
            LanceDBEngine._validate_metadata(metadata)

    @given(
        type_=st.text(min_size=1),
        act=st.integers(min_value=0),
        characters_involved=st.lists(st.text()),
        summary=st.text(),
        chapter=st.integers(max_value=-1),
    )
    @settings(max_examples=100)
    def test_negative_chapter_rejected(
        self,
        type_: str,
        act: int,
        characters_involved: list[str],
        summary: str,
        chapter: int,
    ) -> None:
        """Negative chapter must raise MetadataValidationError."""
        metadata = DocumentMetadata(
            type=type_,
            chapter=chapter,
            act=act,
            characters_involved=characters_involved,
            summary=summary,
        )
        with pytest.raises(MetadataValidationError):
            LanceDBEngine._validate_metadata(metadata)

    @given(
        type_=st.text(min_size=1),
        chapter=st.integers(min_value=0),
        characters_involved=st.lists(st.text()),
        summary=st.text(),
        act=st.integers(max_value=-1),
    )
    @settings(max_examples=100)
    def test_negative_act_rejected(
        self,
        type_: str,
        chapter: int,
        characters_involved: list[str],
        summary: str,
        act: int,
    ) -> None:
        """Negative act must raise MetadataValidationError."""
        metadata = DocumentMetadata(
            type=type_,
            chapter=chapter,
            act=act,
            characters_involved=characters_involved,
            summary=summary,
        )
        with pytest.raises(MetadataValidationError):
            LanceDBEngine._validate_metadata(metadata)

    @given(
        type_=st.text(min_size=1),
        chapter=st.integers(min_value=0),
        act=st.integers(min_value=0),
        summary=st.text(),
        characters_involved=st.text(),  # wrong type: str instead of list
    )
    @settings(max_examples=100)
    def test_non_list_characters_involved_rejected(
        self,
        type_: str,
        chapter: int,
        act: int,
        summary: str,
        characters_involved: str,
    ) -> None:
        """Non-list characters_involved must raise MetadataValidationError."""
        metadata = DocumentMetadata(
            type=type_,
            chapter=chapter,
            act=act,
            characters_involved=characters_involved,  # type: ignore[arg-type]
            summary=summary,
        )
        with pytest.raises(MetadataValidationError):
            LanceDBEngine._validate_metadata(metadata)

    @given(
        type_=st.text(min_size=1),
        chapter=st.integers(min_value=0),
        act=st.integers(min_value=0),
        characters_involved=st.lists(st.text()),
        summary=st.integers(),  # wrong type: int instead of str
    )
    @settings(max_examples=100)
    def test_non_string_summary_rejected(
        self,
        type_: str,
        chapter: int,
        act: int,
        characters_involved: list[str],
        summary: int,
    ) -> None:
        """Non-string summary must raise MetadataValidationError."""
        metadata = DocumentMetadata(
            type=type_,
            chapter=chapter,
            act=act,
            characters_involved=characters_involved,
            summary=summary,  # type: ignore[arg-type]
        )
        with pytest.raises(MetadataValidationError):
            LanceDBEngine._validate_metadata(metadata)


# ---------------------------------------------------------------------------
# Mock EmbeddingProvider for Property 2 (deterministic hash-based)
# ---------------------------------------------------------------------------

import hashlib
import struct


class HashEmbeddingProvider:
    """Deterministic hash-based embedding provider for testing.

    Produces a fixed-dimension vector from text using SHA-256.
    """

    DIMENSIONALITY = 16

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Unpack first 16 bytes as 16 unsigned chars, normalise to [0, 1].
        values = struct.unpack(f"{self.DIMENSIONALITY}B", digest[: self.DIMENSIONALITY])
        return [v / 255.0 for v in values]

    @property
    def dimensionality(self) -> int:
        return self.DIMENSIONALITY


# ---------------------------------------------------------------------------
# Fixture: real LanceDB engine backed by tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture()
def lancedb_engine(tmp_path):
    """Create a LanceDBEngine with a tmp_path-based db and initialised collections."""
    db_path = str(tmp_path / "test_lancedb")
    provider = HashEmbeddingProvider()
    engine = LanceDBEngine(db_path=db_path, embedding_provider=provider)
    engine.initialize_collections()
    return engine


# ---------------------------------------------------------------------------
# Property 2: Document Storage Round-Trip Integrity
# ---------------------------------------------------------------------------

# Use a shared counter to generate unique chapter+act combos per iteration.
import itertools

_unique_counter = itertools.count(1)


class TestDocumentStorageRoundTrip:
    """Feature: lancedb-rag-pipeline-v2, Property 2: Document Storage Round-Trip Integrity

    **Validates: Requirements 2.6, 17.4**

    For any document stored in LanceDB with valid metadata, retrieving that
    document by exact metadata match (type + chapter + act) SHALL return the
    original document text and metadata without modification.
    """

    @given(
        text=st.text(min_size=1, max_size=500),
        emotional_tone=st.text(max_size=50),
        plot_function=st.text(max_size=50),
        summary=st.text(max_size=100),
        characters_involved=st.lists(st.text(min_size=1, max_size=20), max_size=5),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_store_and_retrieve_round_trip(
        self,
        lancedb_engine: LanceDBEngine,
        text: str,
        emotional_tone: str,
        plot_function: str,
        summary: str,
        characters_involved: list[str],
    ) -> None:
        """Stored documents are retrievable with identical text and metadata."""
        # Use a unique chapter+act pair per iteration to avoid collisions.
        idx = next(_unique_counter)
        chapter = idx
        act = idx

        metadata = DocumentMetadata(
            type="act",
            chapter=chapter,
            act=act,
            characters_involved=characters_involved,
            emotional_tone=emotional_tone,
            plot_function=plot_function,
            summary=summary,
        )

        # Store
        lancedb_engine.store_document("acts", text, metadata)

        # Retrieve by exact metadata match
        results = lancedb_engine.query(
            collection="acts",
            query_text=text,
            metadata_filters={"type": "act", "chapter": chapter, "act": act},
            top_k=1,
        )

        # Must find at least one result
        assert len(results) >= 1, (
            f"Expected at least 1 result for chapter={chapter}, act={act}, "
            f"got {len(results)}"
        )

        result = results[0]

        # Text round-trip integrity
        assert result.text == text, (
            f"Text mismatch: stored {text!r}, got {result.text!r}"
        )

        # Metadata round-trip integrity
        assert result.metadata.type == metadata.type
        assert result.metadata.chapter == metadata.chapter
        assert result.metadata.act == metadata.act
        assert result.metadata.emotional_tone == metadata.emotional_tone
        assert result.metadata.plot_function == metadata.plot_function
        assert result.metadata.summary == metadata.summary
        assert result.metadata.characters_involved == metadata.characters_involved


# ---------------------------------------------------------------------------
# Property 3: Retrieval Query Filter and Ordering Correctness
# ---------------------------------------------------------------------------


class TestRetrievalQueryFilterAndOrdering:
    """Feature: lancedb-rag-pipeline-v2, Property 3: Retrieval Query Filter and Ordering Correctness

    **Validates: Requirements 2.1, 2.2, 2.3**

    For any set of stored documents and any retrieval query with metadata
    filters, all returned results SHALL (a) satisfy every specified filter
    criterion, (b) be ranked in descending order of cosine similarity score,
    and (c) contain at most top-k results where k is the configured limit.
    """

    @given(
        doc_texts=st.lists(
            st.text(min_size=1, max_size=200),
            min_size=2,
            max_size=5,
        ),
        query_text=st.text(min_size=1, max_size=100),
        top_k=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_filter_ordering_and_top_k(
        self,
        lancedb_engine: LanceDBEngine,
        doc_texts: list[str],
        query_text: str,
        top_k: int,
    ) -> None:
        """Stored documents queried with a chapter filter return only matching
        results, ranked by descending similarity, capped at top_k."""
        # Use unique chapter numbers to avoid cross-iteration collisions.
        base = next(_unique_counter) * 1000

        # Assign the first document a "target" chapter; the rest get different chapters.
        target_chapter = base
        chapters = [target_chapter] + [base + i + 1 for i in range(len(doc_texts) - 1)]

        for i, text in enumerate(doc_texts):
            metadata = DocumentMetadata(
                type="act",
                chapter=chapters[i],
                act=i + 1,
                characters_involved=[],
                emotional_tone="neutral",
                plot_function="scene",
                summary=f"doc {i}",
            )
            lancedb_engine.store_document("acts", text, metadata)

        # Query with a filter that matches only the target chapter.
        results = lancedb_engine.query(
            collection="acts",
            query_text=query_text,
            metadata_filters={"chapter": target_chapter},
            top_k=top_k,
        )

        # (a) All returned results satisfy the filter criterion.
        for r in results:
            assert r.metadata.chapter == target_chapter, (
                f"Expected chapter={target_chapter}, got {r.metadata.chapter}"
            )

        # (b) Results are ranked by descending similarity_score.
        for i in range(len(results) - 1):
            assert results[i].similarity_score >= results[i + 1].similarity_score, (
                f"Results not in descending similarity order: "
                f"score[{i}]={results[i].similarity_score} < "
                f"score[{i + 1}]={results[i + 1].similarity_score}"
            )

        # (c) Number of results <= top_k.
        assert len(results) <= top_k, (
            f"Expected at most {top_k} results, got {len(results)}"
        )
