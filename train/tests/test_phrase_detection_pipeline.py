"""Integration tests for the PhraseDetectionPipeline orchestrator.

Validates:
- Phase ordering (phases called in sequence) — Requirement 7.1
- Error stops execution (no output on phase failure) — Requirement 7.4
- Collection cleanup on failure and success — Requirement 7.6
- Config defaults match spec — Requirement 8.2
- Output path computation helper
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from romance_factory.generate.phrase_detection.config import PhraseDetectionConfig
from romance_factory.generate.phrase_detection.models import (
    ChapterSegment,
    ClusterReportEntry,
    PhraseOccurrence,
    ReplacementReport,
    RepetitionCluster,
    VariationMapping,
    VariationResult,
)
from romance_factory.generate.phrase_detection.pipeline import (
    PhraseDetectionPipeline,
    _compute_output_path,
)


# ---------------------------------------------------------------------------
# Shared patch target prefix
# ---------------------------------------------------------------------------
_MOD = "romance_factory.generate.phrase_detection.pipeline"


def _make_chapter() -> ChapterSegment:
    return ChapterSegment(chapter_index=0, title="Chapter 1", text="Hello world.", global_char_offset=0)


def _make_occurrence() -> PhraseOccurrence:
    return PhraseOccurrence(
        original_text="hello world",
        normalized_text="hello world",
        chapter_index=0,
        paragraph_index=0,
        char_start=0,
        char_end=11,
        word_count=2,
    )


def _make_cluster() -> RepetitionCluster:
    occ = _make_occurrence()
    return RepetitionCluster(
        cluster_id=0,
        canonical_phrase="hello world",
        occurrences=[occ, occ],
        similarity_scores=[0.9],
        avg_similarity=0.9,
    )


def _make_variation(cluster: RepetitionCluster) -> VariationResult:
    return VariationResult(
        cluster_id=cluster.cluster_id,
        original_phrase=cluster.canonical_phrase,
        kept_occurrence=cluster.occurrences[0],
        variations=[
            VariationMapping(occurrence=cluster.occurrences[1], variation_text="hi earth")
        ],
    )


def _make_replacement_report() -> ReplacementReport:
    return ReplacementReport(
        clusters_processed=1,
        total_replacements=1,
        output_path="/tmp/out_deduped.txt",
        cluster_details=[],
    )


# ---------------------------------------------------------------------------
# Helpers to build a fully-mocked pipeline
# ---------------------------------------------------------------------------

def _build_mocks():
    """Return a dict of mock objects for all pipeline dependencies."""
    chapters = [_make_chapter()]
    phrases = [_make_occurrence()]
    cluster = _make_cluster()
    variation = _make_variation(cluster)
    report = _make_replacement_report()

    mock_loader = MagicMock()
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = chapters
    mock_loader.return_value = mock_loader_instance

    mock_extractor = MagicMock()
    mock_extractor_instance = MagicMock()
    mock_extractor_instance.extract.return_value = phrases
    mock_extractor.return_value = mock_extractor_instance

    mock_embedder = MagicMock()
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.embed_and_store.return_value = 1
    mock_embedder_instance.unique_entries = []
    mock_embedder.return_value = mock_embedder_instance

    mock_detector = MagicMock()
    mock_detector_instance = MagicMock()
    mock_detector_instance.detect.return_value = [cluster]
    mock_detector.return_value = mock_detector_instance

    mock_generator = MagicMock()
    mock_generator_instance = MagicMock()
    mock_generator_instance.generate.return_value = variation
    mock_generator.return_value = mock_generator_instance

    mock_replacer = MagicMock()
    mock_replacer_instance = MagicMock()
    mock_replacer_instance.replace.return_value = report
    mock_replacer.return_value = mock_replacer_instance

    mock_embedding_provider = MagicMock()
    mock_lancedb_engine = MagicMock()

    return {
        "loader": mock_loader,
        "loader_inst": mock_loader_instance,
        "extractor": mock_extractor,
        "extractor_inst": mock_extractor_instance,
        "embedder": mock_embedder,
        "embedder_inst": mock_embedder_instance,
        "detector": mock_detector,
        "detector_inst": mock_detector_instance,
        "generator": mock_generator,
        "generator_inst": mock_generator_instance,
        "replacer": mock_replacer,
        "replacer_inst": mock_replacer_instance,
        "embedding_provider": mock_embedding_provider,
        "lancedb_engine": mock_lancedb_engine,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhasesCalledInSequence:
    """Validates: Requirements 7.1 — phases execute in order."""

    @patch(f"{_MOD}.PhraseReplacer")
    @patch(f"{_MOD}.VariationGenerator")
    @patch(f"{_MOD}.RepetitionDetector")
    @patch(f"{_MOD}.PhraseEmbedder")
    @patch(f"{_MOD}.PhraseExtractor")
    @patch(f"{_MOD}.ManuscriptLoader")
    @patch(f"{_MOD}.LanceDBEngine")
    @patch(f"{_MOD}.EmbeddingProvider")
    def test_phases_called_in_sequence(
        self,
        mock_ep_cls,
        mock_ldb_cls,
        mock_loader_cls,
        mock_extractor_cls,
        mock_embedder_cls,
        mock_detector_cls,
        mock_generator_cls,
        mock_replacer_cls,
    ):
        mocks = _build_mocks()
        mock_loader_cls.return_value = mocks["loader_inst"]
        mock_extractor_cls.return_value = mocks["extractor_inst"]
        mock_embedder_cls.return_value = mocks["embedder_inst"]
        mock_detector_cls.return_value = mocks["detector_inst"]
        mock_generator_cls.return_value = mocks["generator_inst"]
        mock_replacer_cls.return_value = mocks["replacer_inst"]

        cfg = PhraseDetectionConfig()
        pipeline = PhraseDetectionPipeline(cfg)
        result = pipeline.run("/tmp/manuscript.txt")

        # Verify each phase method was called
        mocks["loader_inst"].load.assert_called_once_with("/tmp/manuscript.txt")
        mocks["extractor_inst"].extract.assert_called_once()
        mocks["embedder_inst"].embed_and_store.assert_called_once()
        mocks["detector_inst"].detect.assert_called_once()
        mocks["generator_inst"].generate.assert_called_once()
        mocks["replacer_inst"].replace.assert_called_once()

        # Verify phases_completed in the report reflects all 6 phases
        assert result.phases_completed == [
            "load_manuscript",
            "extract_phrases",
            "embed_phrases",
            "detect_repetitions",
            "generate_variations",
            "replace_phrases",
        ]


class TestErrorStopsExecution:
    """Validates: Requirements 7.4 — error stops pipeline, no output."""

    @patch(f"{_MOD}.PhraseReplacer")
    @patch(f"{_MOD}.VariationGenerator")
    @patch(f"{_MOD}.RepetitionDetector")
    @patch(f"{_MOD}.PhraseEmbedder")
    @patch(f"{_MOD}.PhraseExtractor")
    @patch(f"{_MOD}.ManuscriptLoader")
    @patch(f"{_MOD}.LanceDBEngine")
    @patch(f"{_MOD}.EmbeddingProvider")
    def test_error_in_phase_stops_execution(
        self,
        mock_ep_cls,
        mock_ldb_cls,
        mock_loader_cls,
        mock_extractor_cls,
        mock_embedder_cls,
        mock_detector_cls,
        mock_generator_cls,
        mock_replacer_cls,
    ):
        mocks = _build_mocks()
        # Make loader.load raise FileNotFoundError
        mock_loader_inst = MagicMock()
        mock_loader_inst.load.side_effect = FileNotFoundError("manuscript.txt not found")
        mock_loader_cls.return_value = mock_loader_inst

        mock_embedder_cls.return_value = mocks["embedder_inst"]
        mock_extractor_cls.return_value = mocks["extractor_inst"]
        mock_detector_cls.return_value = mocks["detector_inst"]
        mock_generator_cls.return_value = mocks["generator_inst"]
        mock_replacer_cls.return_value = mocks["replacer_inst"]

        cfg = PhraseDetectionConfig()
        pipeline = PhraseDetectionPipeline(cfg)

        with pytest.raises(FileNotFoundError):
            pipeline.run("/tmp/manuscript.txt")

        # Later phases should NOT have been called
        mocks["extractor_inst"].extract.assert_not_called()
        mocks["embedder_inst"].embed_and_store.assert_not_called()
        mocks["detector_inst"].detect.assert_not_called()
        mocks["generator_inst"].generate.assert_not_called()
        mocks["replacer_inst"].replace.assert_not_called()


class TestCollectionCleanup:
    """Validates: Requirements 7.6 — cleanup on success and failure."""

    @patch(f"{_MOD}.PhraseReplacer")
    @patch(f"{_MOD}.VariationGenerator")
    @patch(f"{_MOD}.RepetitionDetector")
    @patch(f"{_MOD}.PhraseEmbedder")
    @patch(f"{_MOD}.PhraseExtractor")
    @patch(f"{_MOD}.ManuscriptLoader")
    @patch(f"{_MOD}.LanceDBEngine")
    @patch(f"{_MOD}.EmbeddingProvider")
    def test_collection_cleanup_on_failure(
        self,
        mock_ep_cls,
        mock_ldb_cls,
        mock_loader_cls,
        mock_extractor_cls,
        mock_embedder_cls,
        mock_detector_cls,
        mock_generator_cls,
        mock_replacer_cls,
    ):
        mocks = _build_mocks()
        mock_loader_cls.return_value = mocks["loader_inst"]

        # Make extractor.extract raise RuntimeError
        mock_extractor_inst = MagicMock()
        mock_extractor_inst.extract.side_effect = RuntimeError("extraction failed")
        mock_extractor_cls.return_value = mock_extractor_inst

        mock_embedder_cls.return_value = mocks["embedder_inst"]
        mock_detector_cls.return_value = mocks["detector_inst"]
        mock_generator_cls.return_value = mocks["generator_inst"]
        mock_replacer_cls.return_value = mocks["replacer_inst"]

        cfg = PhraseDetectionConfig()
        pipeline = PhraseDetectionPipeline(cfg)

        with pytest.raises(RuntimeError):
            pipeline.run("/tmp/manuscript.txt")

        # cleanup() must still be called even after failure
        mocks["embedder_inst"].cleanup.assert_called_once()

    @patch(f"{_MOD}.PhraseReplacer")
    @patch(f"{_MOD}.VariationGenerator")
    @patch(f"{_MOD}.RepetitionDetector")
    @patch(f"{_MOD}.PhraseEmbedder")
    @patch(f"{_MOD}.PhraseExtractor")
    @patch(f"{_MOD}.ManuscriptLoader")
    @patch(f"{_MOD}.LanceDBEngine")
    @patch(f"{_MOD}.EmbeddingProvider")
    def test_collection_cleanup_on_success(
        self,
        mock_ep_cls,
        mock_ldb_cls,
        mock_loader_cls,
        mock_extractor_cls,
        mock_embedder_cls,
        mock_detector_cls,
        mock_generator_cls,
        mock_replacer_cls,
    ):
        mocks = _build_mocks()
        mock_loader_cls.return_value = mocks["loader_inst"]
        mock_extractor_cls.return_value = mocks["extractor_inst"]
        mock_embedder_cls.return_value = mocks["embedder_inst"]
        mock_detector_cls.return_value = mocks["detector_inst"]
        mock_generator_cls.return_value = mocks["generator_inst"]
        mock_replacer_cls.return_value = mocks["replacer_inst"]

        cfg = PhraseDetectionConfig()
        pipeline = PhraseDetectionPipeline(cfg)
        pipeline.run("/tmp/manuscript.txt")

        # cleanup() must be called on success too
        mocks["embedder_inst"].cleanup.assert_called_once()


class TestConfigDefaultsMatchSpec:
    """Validates: Requirements 8.2 — default values match specification."""

    def test_config_defaults_match_spec(self):
        cfg = PhraseDetectionConfig()
        assert cfg.min_ngram_words == 4
        assert cfg.max_ngram_words == 12
        assert cfg.similarity_threshold == 0.85
        assert cfg.top_k_retrieval == 20
        assert cfg.max_clusters == 50
        assert cfg.context_sentences == 2
        assert cfg.output_suffix == "_deduped"


class TestComputeOutputPath:
    """Test the _compute_output_path helper function."""

    def test_compute_output_path(self):
        assert _compute_output_path("/path/to/manuscript.txt", "_deduped") == "/path/to/manuscript_deduped.txt"

    def test_compute_output_path_no_extension(self):
        assert _compute_output_path("/path/to/manuscript", "_deduped") == "/path/to/manuscript_deduped"

    def test_compute_output_path_multiple_dots(self):
        assert _compute_output_path("/path/to/my.manuscript.txt", "_deduped") == "/path/to/my.manuscript_deduped.txt"
