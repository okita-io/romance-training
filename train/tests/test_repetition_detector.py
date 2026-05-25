"""Property-based and unit tests for RepetitionDetector.

Tests Properties 7-10 from the design document plus example-based unit tests
for edge cases (empty input, no matches, zero clusters above threshold).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from romance_factory.generate.phrase_detection.models import (
    PhraseOccurrence,
    UniquePhraseEntry,
)
from romance_factory.generate.phrase_detection.repetition_detector import (
    RepetitionDetector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _occ(
    text: str = "some phrase here now",
    normalized: str = "some phrase here now",
    chapter: int = 0,
    paragraph: int = 0,
    char_start: int = 0,
    char_end: int = 20,
    word_count: int = 4,
) -> PhraseOccurrence:
    return PhraseOccurrence(
        original_text=text,
        normalized_text=normalized,
        chapter_index=chapter,
        paragraph_index=paragraph,
        char_start=char_start,
        char_end=char_end,
        word_count=word_count,
    )


def _entry(
    normalized: str,
    representative: str | None = None,
    occurrences: list[PhraseOccurrence] | None = None,
    vector: list[float] | None = None,
) -> UniquePhraseEntry:
    return UniquePhraseEntry(
        normalized_text=normalized,
        representative_text=representative or normalized.title(),
        occurrences=occurrences or [],
        embedding_vector=vector or [1.0, 0.0, 0.0, 0.0],
    )


def _mock_engine_with_search(search_results_by_vector=None):
    """Create a mock LanceDBEngine whose _db.open_table().search() returns
    controlled results.

    ``search_results_by_vector`` maps tuple(vector) -> list[dict] of rows.
    Each row dict should have: normalized_text, _distance, text,
    chapter_index, paragraph_index, char_start, char_end, occurrence_count.
    """
    engine = MagicMock()
    db = MagicMock()
    engine._db = db

    table = MagicMock()
    db.open_table.return_value = table

    def _search(vector):
        query = MagicMock()
        key = tuple(vector) if vector is not None else ()
        rows = (search_results_by_vector or {}).get(key, [])

        def _limit(n):
            limited = MagicMock()
            limited.to_list.return_value = rows[:n]
            return limited

        query.limit = _limit
        return query

    table.search.side_effect = _search
    return engine


# ---------------------------------------------------------------------------
# Example-based unit tests
# ---------------------------------------------------------------------------


class TestDetectEdgeCases:
    """Example-based unit tests for RepetitionDetector edge cases."""

    def test_empty_input_returns_empty_list(self):
        """Empty unique_phrases input returns empty cluster list."""
        engine = _mock_engine_with_search({})
        detector = RepetitionDetector(engine, similarity_threshold=0.85)
        result = detector.detect([])
        assert result == []

    def test_single_phrase_no_similar_matches_returns_empty(self):
        """A single phrase with no similar matches in LanceDB returns empty."""
        vec = (1.0, 0.0, 0.0, 0.0)
        search_results = {
            vec: [
                {
                    "normalized_text": "the quick brown fox",
                    "_distance": 0.0,
                    "text": "The Quick Brown Fox",
                    "chapter_index": 0,
                    "paragraph_index": 0,
                    "char_start": 0,
                    "char_end": 19,
                    "occurrence_count": 1,
                },
            ]
        }
        engine = _mock_engine_with_search(search_results)
        detector = RepetitionDetector(engine, similarity_threshold=0.85)

        entry = _entry(
            "the quick brown fox",
            "The Quick Brown Fox",
            [_occ("The Quick Brown Fox", "the quick brown fox",
                  chapter=0, paragraph=0)],
            list(vec),
        )
        result = detector.detect([entry])
        assert result == []

    def test_zero_clusters_above_threshold_returns_empty(self):
        """When all similarities are below threshold, returns empty list."""
        vec_a = (1.0, 0.0, 0.0, 0.0)
        vec_b = (0.0, 1.0, 0.0, 0.0)

        search_results = {
            vec_a: [
                {
                    "normalized_text": "phrase beta gamma delta",
                    "_distance": 0.90,
                    "text": "Phrase Beta Gamma Delta",
                    "chapter_index": 1,
                    "paragraph_index": 1,
                    "char_start": 0,
                    "char_end": 23,
                    "occurrence_count": 1,
                },
            ],
            vec_b: [
                {
                    "normalized_text": "phrase alpha gamma delta",
                    "_distance": 0.90,
                    "text": "Phrase Alpha Gamma Delta",
                    "chapter_index": 0,
                    "paragraph_index": 0,
                    "char_start": 0,
                    "char_end": 24,
                    "occurrence_count": 1,
                },
            ],
        }
        engine = _mock_engine_with_search(search_results)
        detector = RepetitionDetector(engine, similarity_threshold=0.85)

        entries = [
            _entry(
                "phrase alpha gamma delta",
                "Phrase Alpha Gamma Delta",
                [_occ("Phrase Alpha Gamma Delta", "phrase alpha gamma delta",
                      chapter=0, paragraph=0, char_start=0, char_end=24)],
                list(vec_a),
            ),
            _entry(
                "phrase beta gamma delta",
                "Phrase Beta Gamma Delta",
                [_occ("Phrase Beta Gamma Delta", "phrase beta gamma delta",
                      chapter=1, paragraph=1, char_start=0, char_end=23)],
                list(vec_b),
            ),
        ]
        result = detector.detect(entries)
        assert result == []


# ---------------------------------------------------------------------------
# Hypothesis strategies for property tests
# ---------------------------------------------------------------------------

_word_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz"),
    min_size=2,
    max_size=6,
)

_norm_phrase_st = st.lists(_word_st, min_size=4, max_size=6).map(
    lambda ws: " ".join(ws)
)


@st.composite
def _cluster_scenario(draw):
    """Generate a scenario with 2-4 unique phrases that are 'similar' to each
    other, each having occurrences across at least 2 distinct paragraphs
    collectively.

    Returns (entries, threshold, search_results_map, max_clusters, pair_sims).
    """
    threshold = draw(st.floats(min_value=0.50, max_value=0.95))

    n_phrases = draw(st.integers(min_value=2, max_value=4))

    norm_texts = draw(
        st.lists(_norm_phrase_st, min_size=n_phrases, max_size=n_phrases, unique=True)
    )

    # Generate pairwise similarities all above threshold
    pair_sims: dict[tuple[str, str], float] = {}
    for i in range(len(norm_texts)):
        for j in range(i + 1, len(norm_texts)):
            sim = draw(st.floats(min_value=threshold + 0.001, max_value=1.0))
            key = tuple(sorted([norm_texts[i], norm_texts[j]]))
            pair_sims[key] = sim

    # Build entries — ensure at least 2 distinct (chapter, paragraph) tuples
    # across all occurrences so the cluster passes the cross-paragraph filter
    entries: list[UniquePhraseEntry] = []
    all_para_slots: set[tuple[int, int]] = set()

    for idx, norm in enumerate(norm_texts):
        n_occ = draw(st.integers(min_value=1, max_value=3))
        occs: list[PhraseOccurrence] = []
        for occ_i in range(n_occ):
            chap = draw(st.integers(min_value=0, max_value=3))
            para = draw(st.integers(min_value=0, max_value=5))
            start = draw(st.integers(min_value=0, max_value=500))
            end = start + len(norm)
            occs.append(_occ(
                text=norm.title(),
                normalized=norm,
                chapter=chap,
                paragraph=para,
                char_start=start,
                char_end=end,
            ))
            all_para_slots.add((chap, para))
        vec = [0.0] * 4
        vec[idx % 4] = 1.0
        entries.append(_entry(norm, norm.title(), occs, vec))

    # Build search results: for each entry, return the other entries
    search_map: dict[tuple, list[dict]] = {}
    for entry in entries:
        vec_key = tuple(entry.embedding_vector)
        rows: list[dict] = []
        for other in entries:
            if other.normalized_text == entry.normalized_text:
                continue
            key = tuple(sorted([entry.normalized_text, other.normalized_text]))
            sim = pair_sims.get(key, 0.0)
            distance = 1.0 - sim
            rows.append({
                "normalized_text": other.normalized_text,
                "_distance": distance,
                "text": other.representative_text,
                "chapter_index": other.occurrences[0].chapter_index if other.occurrences else 0,
                "paragraph_index": other.occurrences[0].paragraph_index if other.occurrences else 0,
                "char_start": other.occurrences[0].char_start if other.occurrences else 0,
                "char_end": other.occurrences[0].char_end if other.occurrences else 0,
                "occurrence_count": len(other.occurrences),
            })
        search_map[vec_key] = rows

    max_clusters = draw(st.integers(min_value=1, max_value=10))

    return entries, threshold, search_map, max_clusters, pair_sims


@st.composite
def _multi_cluster_scenario(draw):
    """Generate a scenario with 2-3 independent clusters that do NOT share
    occurrences, ensuring non-overlap and ordering properties can be tested.

    Returns (all_entries, threshold, search_map, max_clusters).
    """
    threshold = 0.80
    n_clusters = draw(st.integers(min_value=2, max_value=3))

    all_entries: list[UniquePhraseEntry] = []
    search_map: dict[tuple, list[dict]] = {}
    used_slots: set[tuple[int, int, int, int]] = set()

    for cluster_idx in range(n_clusters):
        n_phrases = draw(st.integers(min_value=2, max_value=3))
        norm_texts = draw(
            st.lists(_norm_phrase_st, min_size=n_phrases, max_size=n_phrases, unique=True)
        )

        existing_norms = {e.normalized_text for e in all_entries}
        assume(all(nt not in existing_norms for nt in norm_texts))

        pair_sims: dict[tuple[str, str], float] = {}
        for i in range(len(norm_texts)):
            for j in range(i + 1, len(norm_texts)):
                sim = draw(st.floats(min_value=threshold + 0.01, max_value=1.0))
                key = tuple(sorted([norm_texts[i], norm_texts[j]]))
                pair_sims[key] = sim

        cluster_entries: list[UniquePhraseEntry] = []
        for idx, norm in enumerate(norm_texts):
            n_occ = draw(st.integers(min_value=1, max_value=3))
            occs: list[PhraseOccurrence] = []
            for _ in range(n_occ):
                for _attempt in range(20):
                    chap = draw(st.integers(min_value=0, max_value=5))
                    para = draw(st.integers(min_value=0, max_value=10))
                    start = draw(st.integers(min_value=0, max_value=500))
                    end = start + len(norm)
                    slot = (chap, para, start, end)
                    if slot not in used_slots:
                        used_slots.add(slot)
                        break
                occs.append(_occ(
                    text=norm.title(),
                    normalized=norm,
                    chapter=chap,
                    paragraph=para,
                    char_start=start,
                    char_end=end,
                ))

            # Use distinct vector dimensions per cluster
            vec = [0.0] * 4
            vec_idx = (cluster_idx * 2 + idx) % 4
            vec[vec_idx] = float(cluster_idx + 1)
            cluster_entries.append(_entry(norm, norm.title(), occs, vec))

        # Build search results within this cluster only
        for entry in cluster_entries:
            vec_key = tuple(entry.embedding_vector)
            rows: list[dict] = []
            for other in cluster_entries:
                if other.normalized_text == entry.normalized_text:
                    continue
                key = tuple(sorted([entry.normalized_text, other.normalized_text]))
                sim = pair_sims.get(key, 0.0)
                distance = 1.0 - sim
                rows.append({
                    "normalized_text": other.normalized_text,
                    "_distance": distance,
                    "text": other.representative_text,
                    "chapter_index": other.occurrences[0].chapter_index if other.occurrences else 0,
                    "paragraph_index": other.occurrences[0].paragraph_index if other.occurrences else 0,
                    "char_start": other.occurrences[0].char_start if other.occurrences else 0,
                    "char_end": other.occurrences[0].char_end if other.occurrences else 0,
                    "occurrence_count": len(other.occurrences),
                })
            search_map[vec_key] = rows

        all_entries.extend(cluster_entries)

    max_clusters = draw(st.integers(min_value=1, max_value=10))
    return all_entries, threshold, search_map, max_clusters


# ---------------------------------------------------------------------------
# Property 7: Cluster Similarity Threshold Invariant
# ---------------------------------------------------------------------------


class TestProperty7ClusterSimilarityThreshold:
    """Feature: repeated-phrase-detection, Property 7: Cluster Similarity Threshold Invariant

    All pairwise similarity scores within a cluster SHALL exceed the configured
    similarity_threshold. Each cluster SHALL have non-empty canonical_phrase,
    at least 2 occurrences, and non-empty similarity_scores.

    **Validates: Requirements 4.2, 4.7**
    """

    @given(scenario=_cluster_scenario())
    @settings(max_examples=100)
    def test_all_similarity_scores_exceed_threshold(self, scenario):
        entries, threshold, search_map, max_clusters, _pair_sims = scenario

        engine = _mock_engine_with_search(search_map)
        detector = RepetitionDetector(
            engine,
            similarity_threshold=threshold,
            top_k=20,
            max_clusters=max_clusters,
        )
        clusters = detector.detect(entries)

        for cluster in clusters:
            for score in cluster.similarity_scores:
                assert score >= threshold, (
                    f"Cluster '{cluster.canonical_phrase}' has similarity score "
                    f"{score} below threshold {threshold}"
                )

    @given(scenario=_cluster_scenario())
    @settings(max_examples=100)
    def test_cluster_structural_invariants(self, scenario):
        entries, threshold, search_map, max_clusters, _pair_sims = scenario

        engine = _mock_engine_with_search(search_map)
        detector = RepetitionDetector(
            engine,
            similarity_threshold=threshold,
            top_k=20,
            max_clusters=max_clusters,
        )
        clusters = detector.detect(entries)

        for cluster in clusters:
            assert cluster.canonical_phrase, (
                f"Cluster {cluster.cluster_id} has empty canonical_phrase"
            )
            assert len(cluster.occurrences) >= 2, (
                f"Cluster '{cluster.canonical_phrase}' has "
                f"{len(cluster.occurrences)} occurrences, expected >= 2"
            )
            assert len(cluster.similarity_scores) > 0, (
                f"Cluster '{cluster.canonical_phrase}' has empty similarity_scores"
            )


# ---------------------------------------------------------------------------
# Property 8: Cluster Non-Overlap Invariant
# ---------------------------------------------------------------------------


class TestProperty8ClusterNonOverlap:
    """Feature: repeated-phrase-detection, Property 8: Cluster Non-Overlap Invariant

    No PhraseOccurrence (by chapter_index, paragraph_index, char_start,
    char_end) SHALL appear in more than one cluster.

    **Validates: Requirements 4.3**
    """

    @given(scenario=_multi_cluster_scenario())
    @settings(max_examples=100)
    def test_no_occurrence_in_multiple_clusters(self, scenario):
        all_entries, threshold, search_map, max_clusters = scenario

        engine = _mock_engine_with_search(search_map)
        detector = RepetitionDetector(
            engine,
            similarity_threshold=threshold,
            top_k=20,
            max_clusters=max_clusters,
        )
        clusters = detector.detect(all_entries)

        seen: set[tuple[int, int, int, int]] = set()
        for cluster in clusters:
            for occ in cluster.occurrences:
                key = (occ.chapter_index, occ.paragraph_index,
                       occ.char_start, occ.char_end)
                assert key not in seen, (
                    f"Occurrence {key} appears in multiple clusters"
                )
                seen.add(key)


# ---------------------------------------------------------------------------
# Property 9: Cluster Cross-Paragraph Invariant
# ---------------------------------------------------------------------------


class TestProperty9ClusterCrossParagraph:
    """Feature: repeated-phrase-detection, Property 9: Cluster Cross-Paragraph Invariant

    Occurrences within each cluster SHALL span at least 2 distinct paragraphs
    (identified by the tuple of chapter_index and paragraph_index).

    **Validates: Requirements 4.4**
    """

    @given(scenario=_cluster_scenario())
    @settings(max_examples=100)
    def test_each_cluster_spans_at_least_two_paragraphs(self, scenario):
        entries, threshold, search_map, max_clusters, _pair_sims = scenario

        engine = _mock_engine_with_search(search_map)
        detector = RepetitionDetector(
            engine,
            similarity_threshold=threshold,
            top_k=20,
            max_clusters=max_clusters,
        )
        clusters = detector.detect(entries)

        for cluster in clusters:
            distinct_paragraphs = {
                (occ.chapter_index, occ.paragraph_index)
                for occ in cluster.occurrences
            }
            assert len(distinct_paragraphs) >= 2, (
                f"Cluster '{cluster.canonical_phrase}' spans only "
                f"{len(distinct_paragraphs)} distinct paragraph(s): "
                f"{distinct_paragraphs}"
            )


# ---------------------------------------------------------------------------
# Property 10: Cluster Ordering
# ---------------------------------------------------------------------------


class TestProperty10ClusterOrdering:
    """Feature: repeated-phrase-detection, Property 10: Cluster Ordering

    Clusters SHALL be sorted descending by occurrence count, ties broken by
    descending avg similarity. List SHALL contain at most max_clusters entries.

    **Validates: Requirements 4.5**
    """

    @given(scenario=_multi_cluster_scenario())
    @settings(max_examples=100)
    def test_clusters_sorted_by_occurrence_count_then_avg_similarity(self, scenario):
        all_entries, threshold, search_map, max_clusters = scenario

        engine = _mock_engine_with_search(search_map)
        detector = RepetitionDetector(
            engine,
            similarity_threshold=threshold,
            top_k=20,
            max_clusters=max_clusters,
        )
        clusters = detector.detect(all_entries)

        for i in range(len(clusters) - 1):
            curr_count = len(clusters[i].occurrences)
            next_count = len(clusters[i + 1].occurrences)
            assert curr_count >= next_count, (
                f"Cluster ordering violated: cluster {i} has {curr_count} "
                f"occurrences but cluster {i+1} has {next_count}"
            )
            if curr_count == next_count:
                assert clusters[i].avg_similarity >= clusters[i + 1].avg_similarity, (
                    f"Tie-breaking violated: cluster {i} avg_sim="
                    f"{clusters[i].avg_similarity} < cluster {i+1} avg_sim="
                    f"{clusters[i + 1].avg_similarity}"
                )

    @given(scenario=_multi_cluster_scenario())
    @settings(max_examples=100)
    def test_cluster_count_respects_max_clusters(self, scenario):
        all_entries, threshold, search_map, max_clusters = scenario

        engine = _mock_engine_with_search(search_map)
        detector = RepetitionDetector(
            engine,
            similarity_threshold=threshold,
            top_k=20,
            max_clusters=max_clusters,
        )
        clusters = detector.detect(all_entries)

        assert len(clusters) <= max_clusters, (
            f"Got {len(clusters)} clusters but max_clusters={max_clusters}"
        )
