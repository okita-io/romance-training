"""Shared test fixtures and hypothesis strategies for the v2 RAG pipeline.

Provides:
- Mock LLM backend (patches ``ollama_client.generate()``)
- HashEmbeddingProvider (deterministic hash-based, 16 dimensions)
- In-memory LanceDB instance fixture
- ``tmp_path``-based story directory fixture
- Shared hypothesis strategies for DocumentMetadata, RetrievedContext,
  EditorialIssue generation
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import strategies as st

from romance_factory.generate.lancedb_engine import LanceDBEngine
from romance_factory.generate.models import (
    DocumentMetadata,
    EditorialIssue,
    RetrievalResult,
    RetrievedContext,
)


# ---------------------------------------------------------------------------
# HashEmbeddingProvider — deterministic, no model download required
# ---------------------------------------------------------------------------


class HashEmbeddingProvider:
    """Deterministic hash-based embedding provider for testing.

    Produces a fixed-dimension vector from text using SHA-256.
    """

    DIMENSIONALITY = 16

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = struct.unpack(f"{self.DIMENSIONALITY}B", digest[: self.DIMENSIONALITY])
        return [v / 255.0 for v in values]

    @property
    def dimensionality(self) -> int:
        return self.DIMENSIONALITY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hash_embedding_provider():
    """Return a HashEmbeddingProvider instance."""
    return HashEmbeddingProvider()


@pytest.fixture()
def lancedb_engine(tmp_path):
    """Create a LanceDBEngine backed by tmp_path with initialised collections."""
    db_path = str(tmp_path / "test_lancedb")
    provider = HashEmbeddingProvider()
    engine = LanceDBEngine(db_path=db_path, embedding_provider=provider)
    engine.initialize_collections()
    return engine


@pytest.fixture()
def story_dir(tmp_path):
    """Create a temporary story directory with standard sub-directories."""
    story_path = str(tmp_path / "test_story")
    os.makedirs(story_path, exist_ok=True)
    os.makedirs(os.path.join(story_path, "drafts"), exist_ok=True)
    os.makedirs(os.path.join(story_path, "revisions"), exist_ok=True)
    return story_path


@pytest.fixture()
def mock_llm():
    """Patch ``ollama_client.generate()`` to return deterministic responses.

    The mock returns different canned responses depending on the prompt
    content (act prose, editorial JSON, rewrite prose).
    """
    def _deterministic_generate(prompt: str, **kwargs) -> str:
        lower = prompt.lower() if isinstance(prompt, str) else ""
        if "editorial" in lower or "evaluate" in lower or "score" in lower:
            return json.dumps({
                "score": 7.5,
                "issues": [],
                "rewrite_plan": "",
            })
        if "rewrite" in lower:
            return "Rewritten prose for the act. The characters moved forward."
        # Default: act generation prose
        return (
            "The morning sun cast golden light across the garden. "
            "She turned to face him, her heart racing."
        )

    with patch(
        "romance_factory.core.ollama_client.generate",
        side_effect=_deterministic_generate,
    ) as mock_gen:
        yield mock_gen


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for valid DocumentMetadata
document_metadata_strategy = st.builds(
    DocumentMetadata,
    type=st.sampled_from([
        "author_profile", "character", "outline", "beat",
        "act", "chapter", "editorial", "character_canon",
    ]),
    chapter=st.integers(min_value=0, max_value=50),
    act=st.integers(min_value=0, max_value=10),
    characters_involved=st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
        min_size=0,
        max_size=5,
    ),
    emotional_tone=st.sampled_from([
        "tension", "tenderness", "conflict", "joy", "sorrow", "passion", "",
    ]),
    plot_function=st.sampled_from([
        "inciting_incident", "climax", "resolution", "rising_action",
        "falling_action", "scene", "",
    ]),
    summary=st.text(min_size=0, max_size=100),
    revision_number=st.integers(min_value=0, max_value=10),
    editorial_score=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    foreshadowing_created=st.lists(st.text(min_size=1, max_size=30), max_size=3),
    relationship_changes=st.lists(st.text(min_size=1, max_size=30), max_size=3),
)

# Strategy for act-specific metadata (type="act", chapter>=1, act>=1)
act_metadata_strategy = st.builds(
    DocumentMetadata,
    type=st.just("act"),
    chapter=st.integers(min_value=1, max_value=20),
    act=st.integers(min_value=1, max_value=7),
    characters_involved=st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
        min_size=1,
        max_size=4,
    ),
    emotional_tone=st.sampled_from(["tension", "tenderness", "conflict", "joy"]),
    plot_function=st.sampled_from(["rising_action", "climax", "resolution"]),
    summary=st.text(min_size=5, max_size=80),
    revision_number=st.integers(min_value=0, max_value=5),
    editorial_score=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
)

# Strategy for EditorialIssue
editorial_issue_strategy = st.builds(
    EditorialIssue,
    type=st.sampled_from([
        "continuity", "pacing", "motivation", "slop", "anti_pattern",
        "repetition", "genre_drift", "outline_deviation",
        "missing_cliffhanger", "weak_plot_twist",
    ]),
    severity=st.sampled_from(["BLOCKING", "MAJOR", "MINOR", "INFO"]),
    location=st.text(min_size=1, max_size=30),
    explanation=st.text(min_size=5, max_size=100),
    suggested_fix=st.text(min_size=5, max_size=100),
)

# Strategy for RetrievalResult
retrieval_result_strategy = st.builds(
    RetrievalResult,
    text=st.text(min_size=1, max_size=200),
    metadata=document_metadata_strategy,
    similarity_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)

# Strategy for RetrievedContext (at least one non-empty section)
retrieved_context_strategy = st.builds(
    RetrievedContext,
    author_profile=st.lists(retrieval_result_strategy, min_size=0, max_size=2),
    character_web=st.lists(retrieval_result_strategy, min_size=0, max_size=2),
    world=st.lists(retrieval_result_strategy, min_size=0, max_size=2),
    world_outline=st.lists(retrieval_result_strategy, min_size=0, max_size=2),
    story_outline=st.lists(retrieval_result_strategy, min_size=0, max_size=2),
    chapter_outline=st.lists(retrieval_result_strategy, min_size=0, max_size=2),
    act_outline=st.lists(retrieval_result_strategy, min_size=0, max_size=2),
    foreshadowing=st.lists(retrieval_result_strategy, min_size=0, max_size=2),
    relationship_arcs=st.lists(retrieval_result_strategy, min_size=0, max_size=2),
    previous_acts=st.lists(retrieval_result_strategy, min_size=0, max_size=2),
    editorial_issues=st.lists(retrieval_result_strategy, min_size=0, max_size=2),
    rewrite_plan=st.lists(retrieval_result_strategy, min_size=0, max_size=2),
)


# ---------------------------------------------------------------------------
# Phrase Detection — imports, strategies, and fixtures
# ---------------------------------------------------------------------------

from romance_factory.generate.phrase_detection.models import (
    ChapterSegment,
    PhraseOccurrence,
    RepetitionCluster,
    UniquePhraseEntry,
)
from romance_factory.generate.phrase_detection.config import PhraseDetectionConfig


# -- Reusable atomic strategies for phrase detection -------------------------

# A single lowercase word (2-8 chars, letters only)
_pd_word_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz"),
    min_size=2,
    max_size=8,
)

# A sentence built from 5-15 words
_pd_sentence_st = st.lists(_pd_word_st, min_size=5, max_size=15).map(
    lambda ws: " ".join(ws) + "."
)

# A paragraph built from 1-4 sentences
_pd_paragraph_st = st.lists(_pd_sentence_st, min_size=1, max_size=4).map(
    lambda ss: " ".join(ss)
)


# ---------------------------------------------------------------------------
# Strategy: manuscript_text()
# ---------------------------------------------------------------------------

@st.composite
def manuscript_text(draw, min_chapters=0, max_chapters=4):
    """Generate random prose text with optional chapter headings.

    Returns a string that may contain ``Chapter N`` headings interspersed
    with paragraph text.  When *min_chapters* is 0 the text may have no
    headings at all (single-chapter fallback scenario).
    """
    n_chapters = draw(st.integers(min_value=min_chapters, max_value=max_chapters))

    parts: list[str] = []
    for i in range(max(n_chapters, 1)):
        if n_chapters > 0:
            parts.append(f"Chapter {i + 1}")
        # 2-5 paragraphs per chapter, separated by blank lines
        n_paras = draw(st.integers(min_value=2, max_value=5))
        paras = [draw(_pd_paragraph_st) for _ in range(n_paras)]
        parts.append("\n\n".join(paras))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Strategy: chapter_segments()
# ---------------------------------------------------------------------------

@st.composite
def chapter_segments(draw, min_chapters=1, max_chapters=4):
    """Generate a list of :class:`ChapterSegment` with valid offsets.

    Each segment has monotonically increasing ``global_char_offset`` and
    non-empty ``text`` containing 2+ paragraphs.
    """
    n = draw(st.integers(min_value=min_chapters, max_value=max_chapters))
    segments: list[ChapterSegment] = []
    offset = 0

    for i in range(n):
        title = f"Chapter {i + 1}"
        n_paras = draw(st.integers(min_value=2, max_value=4))
        paras = [draw(_pd_paragraph_st) for _ in range(n_paras)]
        text = "\n\n".join(paras)
        segments.append(
            ChapterSegment(
                chapter_index=i,
                title=title,
                text=text,
                global_char_offset=offset,
            )
        )
        # Advance offset by title + newline + text length
        offset += len(title) + 1 + len(text)

    return segments


# ---------------------------------------------------------------------------
# Strategy: phrase_occurrences()
# ---------------------------------------------------------------------------

@st.composite
def phrase_occurrences(draw, min_count=1, max_count=8):
    """Generate :class:`PhraseOccurrence` objects with consistent offsets.

    Each occurrence has ``char_end == char_start + len(original_text)`` and
    ``word_count`` matching the actual word count of the text.
    """
    n = draw(st.integers(min_value=min_count, max_value=max_count))
    results: list[PhraseOccurrence] = []

    for _ in range(n):
        words = draw(st.lists(_pd_word_st, min_size=4, max_size=10))
        original = " ".join(words)
        normalized = original.lower()
        chapter = draw(st.integers(min_value=0, max_value=5))
        paragraph = draw(st.integers(min_value=0, max_value=10))
        char_start = draw(st.integers(min_value=0, max_value=500))
        results.append(
            PhraseOccurrence(
                original_text=original,
                normalized_text=normalized,
                chapter_index=chapter,
                paragraph_index=paragraph,
                char_start=char_start,
                char_end=char_start + len(original),
                word_count=len(words),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Strategy: repetition_clusters()
# ---------------------------------------------------------------------------

@st.composite
def repetition_clusters(draw, min_clusters=1, max_clusters=4, threshold=0.85):
    """Generate :class:`RepetitionCluster` objects with valid similarity scores.

    Every similarity score is above *threshold*, each cluster has ≥ 2
    occurrences spanning ≥ 2 distinct paragraphs, and ``avg_similarity``
    equals the mean of the scores.
    """
    n = draw(st.integers(min_value=min_clusters, max_value=max_clusters))
    clusters: list[RepetitionCluster] = []

    for cid in range(n):
        n_occ = draw(st.integers(min_value=2, max_value=5))
        phrase_words = draw(st.lists(_pd_word_st, min_size=4, max_size=8))
        canonical = " ".join(phrase_words)

        # Ensure at least 2 distinct (chapter, paragraph) tuples
        occs: list[PhraseOccurrence] = []
        used_paras: set[tuple[int, int]] = set()
        for j in range(n_occ):
            chapter = draw(st.integers(min_value=0, max_value=5))
            paragraph = draw(st.integers(min_value=0, max_value=10))
            # Force diversity for the first two occurrences
            if j == 1 and len(used_paras) < 2:
                for _attempt in range(20):
                    chapter = draw(st.integers(min_value=0, max_value=5))
                    paragraph = draw(st.integers(min_value=0, max_value=10))
                    if (chapter, paragraph) not in used_paras:
                        break
            used_paras.add((chapter, paragraph))
            char_start = draw(st.integers(min_value=0, max_value=500))
            occs.append(
                PhraseOccurrence(
                    original_text=canonical.title(),
                    normalized_text=canonical,
                    chapter_index=chapter,
                    paragraph_index=paragraph,
                    char_start=char_start,
                    char_end=char_start + len(canonical),
                    word_count=len(phrase_words),
                )
            )

        # Generate similarity scores all above threshold
        n_scores = max(1, n_occ - 1)
        scores = [
            draw(st.floats(min_value=threshold + 0.001, max_value=1.0))
            for _ in range(n_scores)
        ]
        avg_sim = sum(scores) / len(scores)

        clusters.append(
            RepetitionCluster(
                cluster_id=cid,
                canonical_phrase=canonical,
                occurrences=occs,
                similarity_scores=scores,
                avg_similarity=avg_sim,
            )
        )

    return clusters


# ---------------------------------------------------------------------------
# Strategy: phrase_detection_configs()
# ---------------------------------------------------------------------------

@st.composite
def phrase_detection_configs(draw, valid_only=False):
    """Generate :class:`PhraseDetectionConfig` instances.

    When *valid_only* is ``True``, all generated configs pass ``validate()``.
    When ``False``, configs may have invalid parameter combinations.
    """
    if valid_only:
        min_n = draw(st.integers(min_value=2, max_value=20))
        max_n = min_n + draw(st.integers(min_value=0, max_value=30))
        sim = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        top_k = draw(st.integers(min_value=1, max_value=500))
        max_cl = draw(st.integers(min_value=1, max_value=200))
    else:
        min_n = draw(st.integers(min_value=-5, max_value=20))
        max_n = draw(st.integers(min_value=-5, max_value=50))
        sim = draw(st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False))
        top_k = draw(st.integers(min_value=-5, max_value=500))
        max_cl = draw(st.integers(min_value=-5, max_value=200))

    return PhraseDetectionConfig(
        min_ngram_words=min_n,
        max_ngram_words=max_n,
        similarity_threshold=sim,
        top_k_retrieval=top_k,
        max_clusters=max_cl,
    )


# ---------------------------------------------------------------------------
# Strategy: replacement_sets()
# ---------------------------------------------------------------------------

@st.composite
def replacement_sets(draw, min_replacements=1, max_replacements=5):
    """Generate non-overlapping replacement spans within generated text.

    Returns ``(text, replacements)`` where each replacement is a tuple
    ``(char_start, char_end, replacement_text)`` and no spans overlap.
    """
    # Build a base text from paragraphs
    n_paras = draw(st.integers(min_value=2, max_value=4))
    paras = [draw(_pd_paragraph_st) for _ in range(n_paras)]
    text = "\n\n".join(paras)

    n = draw(st.integers(min_value=min_replacements, max_value=max_replacements))
    text_len = len(text)

    if text_len < 10:
        return text, []

    # Generate non-overlapping spans
    spans: list[tuple[int, int]] = []
    for _ in range(n):
        if not spans:
            start = draw(st.integers(min_value=0, max_value=max(0, text_len - 5)))
            length = draw(st.integers(min_value=1, max_value=min(10, text_len - start)))
            spans.append((start, start + length))
        else:
            last_end = max(e for _, e in spans)
            if last_end + 2 >= text_len:
                break
            start = draw(st.integers(min_value=last_end + 1, max_value=max(last_end + 1, text_len - 2)))
            length = draw(st.integers(min_value=1, max_value=min(10, text_len - start)))
            spans.append((start, start + length))

    # Generate replacement text for each span
    replacements = []
    for s, e in spans:
        rep_text = draw(st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz "),
            min_size=1,
            max_size=20,
        ))
        replacements.append((s, e, rep_text))

    return text, replacements


# ---------------------------------------------------------------------------
# Phrase Detection Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_phrase_lancedb():
    """Mock LanceDB engine for phrase detection tests.

    Provides a ``MagicMock`` with an in-memory ``_db`` attribute whose
    ``table_names()`` returns an empty list by default.
    """
    engine = MagicMock()
    db = MagicMock()
    db.table_names.return_value = []
    engine._db = db
    return engine


@pytest.fixture()
def mock_phrase_embedding_provider():
    """Mock embedding provider that counts calls.

    Returns a ``MagicMock`` whose ``.embed(text)`` produces a 4-dimensional
    vector and tracks the total number of invocations via the
    ``call_count`` attribute.
    """
    provider = MagicMock()
    counter = {"n": 0}

    def _embed(text: str) -> list[float]:
        counter["n"] += 1
        # Simple deterministic vector from text hash
        h = hashlib.md5(text.encode()).digest()
        return [b / 255.0 for b in h[:4]]

    provider.embed.side_effect = _embed
    provider.dimensionality = 4
    provider.call_count = counter
    return provider


@pytest.fixture()
def mock_phrase_llm():
    """Mock LLM client for variation generation in phrase detection tests.

    Returns a callable mock that produces numbered variation lines.
    The mock tracks calls via ``.call_count`` and ``.call_args_list``.
    """
    def _generate(prompt: str, **kwargs) -> str:
        # Parse how many variations are requested from the prompt
        # Default to 3 if we can't determine
        import re
        match = re.search(r"(\d+)\s+variation", prompt.lower())
        n = int(match.group(1)) if match else 3
        return "\n".join(f"{i + 1}. variation {i + 1} of the phrase" for i in range(n))

    mock = MagicMock(side_effect=_generate)
    return mock


@pytest.fixture(autouse=True)
def _stub_sentence_transformers_for_train_tests() -> None:
    """Avoid Hugging Face downloads when tests construct ``EmbeddingProvider``."""
    try:
        import numpy as np
    except ImportError:
        yield
        return
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = np.zeros(384, dtype=np.float32)
    try:
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            yield
    except ImportError:
        yield
