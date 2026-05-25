"""Property-based tests for Crash Recovery and LanceDB Rehydration.

Feature: lancedb-rag-pipeline-v2, Property 11: Crash Recovery and LanceDB Rehydration from JSON

**Validates: Requirements 11.7, 16.4, 19.6, 19.8**
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from romance_factory.generate.config_v2 import V2Config
from romance_factory.generate.lancedb_engine import LanceDBEngine
from romance_factory.generate.models import DocumentMetadata, JSONArtifact


# ---------------------------------------------------------------------------
# Deterministic embedding provider for testing
# ---------------------------------------------------------------------------

import hashlib
import struct

import itertools

_unique_counter = itertools.count(1)


class HashEmbeddingProvider:
    """Deterministic hash-based embedding provider for testing."""

    DIMENSIONALITY = 16

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = struct.unpack(f"{self.DIMENSIONALITY}B", digest[: self.DIMENSIONALITY])
        return [v / 255.0 for v in values]

    @property
    def dimensionality(self) -> int:
        return self.DIMENSIONALITY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_artifact(story_path: str, artifact_type: str, text: str,
                   metadata: DocumentMetadata, filename: str,
                   subdir: str | None = None) -> str:
    """Save a JSONArtifact to disk and return the file path."""
    if subdir:
        dir_path = os.path.join(story_path, subdir)
    else:
        dir_path = story_path
    file_path = os.path.join(dir_path, filename)
    artifact = JSONArtifact(
        artifact_type=artifact_type,
        text=text,
        metadata=metadata,
        created_at=datetime.now(timezone.utc).isoformat(),
        file_path=file_path,
    )
    artifact.save()
    return file_path


# ---------------------------------------------------------------------------
# Property 11: Crash Recovery and LanceDB Rehydration from JSON
# ---------------------------------------------------------------------------


class TestCrashRecoveryRehydration:
    """Feature: lancedb-rag-pipeline-v2, Property 11: Crash Recovery and LanceDB Rehydration

    **Validates: Requirements 11.7, 16.4, 19.6, 19.8**

    For any set of JSON_Artifact files on disk representing a partially
    completed pipeline run, the Pipeline_Orchestrator SHALL correctly detect
    the last completed phase, rehydrate LanceDB by computing embeddings and
    inserting all documents from JSON files, and resume from the next
    incomplete step without regenerating existing content.
    """

    @given(
        num_acts=st.integers(min_value=1, max_value=5),
        act_texts=st.lists(
            st.text(min_size=5, max_size=100),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_rehydrate_from_json_restores_documents(
        self,
        tmp_path,
        num_acts: int,
        act_texts: list[str],
    ) -> None:
        """LanceDB rehydration from JSON files restores all documents."""
        # Use unique subdirectory per hypothesis iteration
        run_id = next(_unique_counter)
        base = str(tmp_path / f"run_{run_id}")
        drafts_dir = os.path.join(base, "drafts")
        os.makedirs(drafts_dir, exist_ok=True)

        # Limit to actual number of texts available
        actual_count = min(num_acts, len(act_texts))

        # Save act artifacts to disk (only in drafts/ subdirectory)
        for i in range(actual_count):
            metadata = DocumentMetadata(
                type="act",
                chapter=1,
                act=i + 1,
                summary=f"Act {i + 1}",
            )
            _save_artifact(
                base, "act", act_texts[i], metadata,
                f"chapter_01_act_{i + 1:02d}.json", subdir="drafts",
            )

        # Create a fresh LanceDB engine (simulating crash recovery)
        db_path = os.path.join(base, "lancedb_recovery")
        provider = HashEmbeddingProvider()
        engine = LanceDBEngine(db_path=db_path, embedding_provider=provider)
        engine.initialize_collections()

        # Rehydrate only from the drafts directory
        rehydrated = engine.rehydrate_from_json(drafts_dir)
        assert rehydrated == actual_count, (
            f"Expected {actual_count} rehydrated, got {rehydrated}"
        )

        # Verify documents are queryable
        for i in range(actual_count):
            results = engine.query(
                "acts",
                act_texts[i],
                metadata_filters={"chapter": 1, "act": i + 1},
                top_k=1,
            )
            assert len(results) >= 1, (
                f"Act {i + 1} not found after rehydration"
            )
            assert results[0].text == act_texts[i]

    def test_rehydrate_from_json_restores_outline_beat_revisions(self, tmp_path) -> None:
        """Outline beat revisions persisted on disk rehydrate into act_revisions."""
        run_id = next(_unique_counter)
        base = str(tmp_path / f"run_outline_rev_{run_id}")
        drafts_dir = os.path.join(base, "drafts")
        rev_dir = os.path.join(drafts_dir, "outline_beat_revisions")
        os.makedirs(rev_dir, exist_ok=True)

        # Two revisions for the same beat slot.
        for rev in (1, 2):
            meta = DocumentMetadata(
                type="outline_beat",
                chapter=1,
                act=3,
                summary=f"Outline beat rev{rev}",
                revision_number=rev,
            )
            _save_artifact(
                base,
                "outline_beat_revision",
                f"{{\"act_number\": 3, \"summary\": \"rev {rev}\"}}",
                meta,
                f"chapter_01_act_03_rev{rev:02d}.json",
                subdir=os.path.join("drafts", "outline_beat_revisions"),
            )

        db_path = os.path.join(base, "lancedb_recovery")
        provider = HashEmbeddingProvider()
        engine = LanceDBEngine(db_path=db_path, embedding_provider=provider)
        engine.initialize_collections()

        rehydrated = engine.rehydrate_from_json(drafts_dir)
        assert rehydrated == 2

        rows = engine.query(
            "act_revisions",
            "outline beat",
            metadata_filters={"chapter": 1, "act": 3, "type": "outline_beat"},
            top_k=10,
        )
        assert len(rows) == 2

    @given(
        has_author_profile=st.booleans(),
        has_character_web=st.booleans(),
        has_outline=st.booleans(),
        num_draft_acts=st.integers(min_value=0, max_value=3),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_detect_resume_phase_from_artifacts(
        self,
        tmp_path,
        has_author_profile: bool,
        has_character_web: bool,
        has_outline: bool,
        num_draft_acts: int,
    ) -> None:
        """Pipeline detects last completed phase from JSON artifacts on disk."""
        run_id = next(_unique_counter)
        story_path = str(tmp_path / f"story_{run_id}")
        os.makedirs(os.path.join(story_path, "drafts"), exist_ok=True)

        # Save pre-production artifacts based on flags
        if has_author_profile:
            _save_artifact(
                story_path, "author_profile", "Author profile text.",
                DocumentMetadata(type="author_profile", summary="profile"),
                "author_profile.json",
            )
        if has_character_web:
            _save_artifact(
                story_path, "world", "World text.",
                DocumentMetadata(type="world", summary="world"),
                "world.json",
            )
            _save_artifact(
                story_path, "character_web", "Character web text.",
                DocumentMetadata(type="character_web", summary="chars"),
                "character_web.json",
            )
        if has_outline:
            _save_artifact(
                story_path, "story_outline", json.dumps({"chapters": []}),
                DocumentMetadata(type="outline", summary="outline"),
                "story_outline.json",
            )

        # Save draft acts
        for i in range(num_draft_acts):
            _save_artifact(
                story_path, "act", f"Act {i + 1} prose.",
                DocumentMetadata(type="act", chapter=1, act=i + 1, summary=f"act{i+1}"),
                f"chapter_01_act_{i + 1:02d}.json", subdir="drafts",
            )

        # Determine expected last completed phase (12-phase pipeline with world gen).
        # Phase 2 = author_profile, Phase 3 = world.json, Phase 4 = character_web,
        # Phase 5 = story_outline, Phase 6 = outline editorial (same file as 5).
        expected_last = 1  # Phase 1 always "done" if collections valid
        if has_author_profile:
            expected_last = 2
        if has_author_profile and has_character_web:
            # world.json is written whenever character_web is (pipeline order).
            expected_last = 4
        if has_author_profile and has_character_web and has_outline:
            # story_outline.json exists but stub text has no parsed_data / full chapters,
            # so phase 5 completeness check fails — resume at outline generation.
            expected_last = 4

        # Use a mock pipeline to test _detect_resume_phase
        config = V2Config(
            story_path=story_path,
            db_path=str(tmp_path / f"lancedb_{run_id}"),
        )

        with (
            patch("romance_factory.generate.pipeline_v2.EmbeddingProvider"),
            patch("romance_factory.generate.pipeline_v2.LanceDBEngine") as MockEngine,
            patch("romance_factory.generate.pipeline_v2.PromptBuilder"),
            patch("romance_factory.generate.pipeline_v2.ActGenerationAgent"),
            patch("romance_factory.generate.pipeline_v2.ActIntroPlanningAgent"),
            patch("romance_factory.generate.pipeline_v2.EditorialAgent"),
            patch("romance_factory.generate.pipeline_v2.RewriteAgent"),
        ):
            engine_instance = MockEngine.return_value
            engine_instance.validate_collections.return_value = True

            from romance_factory.generate.pipeline_v2 import PipelineV2
            pipeline = PipelineV2(story_path, config)

            resume_phase = pipeline._detect_resume_phase()

            # Resume phase should be the one AFTER the last completed
            expected_resume = min(expected_last + 1, 12)
            assert resume_phase == expected_resume, (
                f"Expected resume from phase {expected_resume}, "
                f"got {resume_phase} "
                f"(profile={has_author_profile}, web={has_character_web}, "
                f"outline={has_outline})"
            )

    @given(
        num_artifacts=st.integers(min_value=1, max_value=4),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_rehydration_does_not_regenerate_existing(
        self,
        tmp_path,
        num_artifacts: int,
    ) -> None:
        """Rehydration inserts documents without regenerating content."""
        run_id = next(_unique_counter)
        base = str(tmp_path / f"story_{run_id}")
        drafts_dir = os.path.join(base, "drafts")
        os.makedirs(drafts_dir, exist_ok=True)

        original_texts: list[str] = []
        for i in range(num_artifacts):
            text = f"Original act {i + 1} content that must not change."
            original_texts.append(text)
            _save_artifact(
                base, "act", text,
                DocumentMetadata(type="act", chapter=1, act=i + 1, summary=f"act{i+1}"),
                f"chapter_01_act_{i + 1:02d}.json", subdir="drafts",
            )

        # Create fresh engine and rehydrate from drafts only
        db_path = os.path.join(base, "lancedb_fresh")
        provider = HashEmbeddingProvider()
        engine = LanceDBEngine(db_path=db_path, embedding_provider=provider)
        engine.initialize_collections()
        engine.rehydrate_from_json(drafts_dir)

        # Verify original text is preserved exactly
        for i, original_text in enumerate(original_texts):
            results = engine.query(
                "acts", original_text,
                metadata_filters={"chapter": 1, "act": i + 1},
                top_k=1,
            )
            assert len(results) >= 1
            assert results[0].text == original_text, (
                f"Text was modified during rehydration: "
                f"expected {original_text!r}, got {results[0].text!r}"
            )
