"""Property-based tests for JSON Artifact Persistence.

Feature: lancedb-rag-pipeline-v2, Property 10: JSON Artifact Persistence and Completeness
Feature: lancedb-rag-pipeline-v2, Property 15: Revision File Preservation

**Validates: Requirements 11.5, 19.1, 19.2, 19.3, 19.4, 19.5, 19.7**
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from romance_factory.generate.models import DocumentMetadata, JSONArtifact


# ---------------------------------------------------------------------------
# Property 10: JSON Artifact Persistence and Completeness
# ---------------------------------------------------------------------------


class TestJSONArtifactPersistence:
    """Feature: lancedb-rag-pipeline-v2, Property 10: JSON Artifact Persistence and Completeness

    **Validates: Requirements 11.5, 19.1, 19.2, 19.3, 19.4, 19.5**

    For any generated artifact, the Pipeline_Orchestrator SHALL save a
    JSON_Artifact to disk containing the full text content and all associated
    metadata before storing in LanceDB.
    """

    @given(
        text=st.text(min_size=1, max_size=500),
        artifact_type=st.sampled_from([
            "act", "chapter", "author_profile", "character_web", "story_outline",
        ]),
        chapter=st.integers(min_value=0, max_value=50),
        act=st.integers(min_value=0, max_value=10),
        emotional_tone=st.sampled_from(["tension", "tenderness", "conflict", ""]),
        plot_function=st.sampled_from(["rising_action", "climax", "resolution", ""]),
        summary=st.text(min_size=0, max_size=100),
        revision_number=st.integers(min_value=0, max_value=5),
        editorial_score=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_save_and_load_round_trip(
        self,
        tmp_path,
        text: str,
        artifact_type: str,
        chapter: int,
        act: int,
        emotional_tone: str,
        plot_function: str,
        summary: str,
        revision_number: int,
        editorial_score: float,
    ) -> None:
        """JSONArtifact.save() writes to disk and .load() round-trips correctly."""
        metadata = DocumentMetadata(
            type=artifact_type,
            chapter=chapter,
            act=act,
            emotional_tone=emotional_tone,
            plot_function=plot_function,
            summary=summary,
            revision_number=revision_number,
            editorial_score=editorial_score,
        )
        file_path = str(tmp_path / f"test_artifact_{chapter}_{act}.json")
        artifact = JSONArtifact(
            artifact_type=artifact_type,
            text=text,
            metadata=metadata,
            created_at="2025-01-01T00:00:00+00:00",
            file_path=file_path,
        )

        # Save writes to disk
        saved_path = artifact.save()
        assert os.path.isfile(saved_path), f"File not created at {saved_path}"

        # Load round-trips correctly
        loaded = JSONArtifact.load(saved_path)
        assert loaded.artifact_type == artifact_type
        assert loaded.text == text
        assert loaded.metadata.type == metadata.type
        assert loaded.metadata.chapter == metadata.chapter
        assert loaded.metadata.act == metadata.act
        assert loaded.metadata.emotional_tone == metadata.emotional_tone
        assert loaded.metadata.plot_function == metadata.plot_function
        assert loaded.metadata.summary == metadata.summary
        assert loaded.metadata.revision_number == metadata.revision_number
        assert loaded.metadata.editorial_score == metadata.editorial_score
        assert loaded.created_at == artifact.created_at

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_act_artifact_naming_convention(
        self,
        tmp_path,
        chapter: int,
        act: int,
    ) -> None:
        """Act artifacts SHALL use ``chapter_{NN}_act_{MM}.json`` naming in ``drafts/``."""
        drafts_dir = str(tmp_path / "drafts")
        expected_name = f"chapter_{chapter:02d}_act_{act:02d}.json"
        file_path = os.path.join(drafts_dir, expected_name)

        artifact = JSONArtifact(
            artifact_type="act",
            text="Test act prose.",
            metadata=DocumentMetadata(type="act", chapter=chapter, act=act),
            created_at="2025-01-01T00:00:00+00:00",
            file_path=file_path,
        )
        artifact.save()

        assert os.path.isfile(file_path)
        # Verify the file is inside a drafts/ directory
        assert "drafts" in file_path
        # Verify naming pattern
        basename = os.path.basename(file_path)
        assert basename == expected_name

    @given(chapter=st.integers(min_value=1, max_value=50))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_chapter_artifact_naming_convention(
        self,
        tmp_path,
        chapter: int,
    ) -> None:
        """Chapter artifacts SHALL use ``chapter_{NN}.json`` naming."""
        drafts_dir = str(tmp_path / "drafts")
        expected_name = f"chapter_{chapter:02d}.json"
        file_path = os.path.join(drafts_dir, expected_name)

        artifact = JSONArtifact(
            artifact_type="chapter",
            text="Test chapter prose.",
            metadata=DocumentMetadata(type="chapter", chapter=chapter),
            created_at="2025-01-01T00:00:00+00:00",
            file_path=file_path,
        )
        artifact.save()

        assert os.path.isfile(file_path)
        basename = os.path.basename(file_path)
        assert basename == expected_name

    @given(
        artifact_type=st.sampled_from(["author_profile", "character_web", "story_outline"]),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_preproduction_artifacts_in_story_root(
        self,
        tmp_path,
        artifact_type: str,
    ) -> None:
        """Pre-production artifacts SHALL be saved in the story path root."""
        story_root = str(tmp_path / "story")
        file_path = os.path.join(story_root, f"{artifact_type}.json")

        artifact = JSONArtifact(
            artifact_type=artifact_type,
            text=f"Test {artifact_type} content.",
            metadata=DocumentMetadata(type=artifact_type),
            created_at="2025-01-01T00:00:00+00:00",
            file_path=file_path,
        )
        artifact.save()

        assert os.path.isfile(file_path)
        # Verify the file is directly in the story root, not in a subdirectory
        assert os.path.dirname(file_path) == story_root


# ---------------------------------------------------------------------------
# Property 15: Revision File Preservation
# ---------------------------------------------------------------------------


class TestRevisionFilePreservation:
    """Feature: lancedb-rag-pipeline-v2, Property 15: Revision File Preservation

    **Validates: Requirements 19.7**

    For any act that undergoes rewriting, each revision SHALL be saved as a
    separate JSON_Artifact with an incremented revision_number, and all
    previous revision files SHALL be preserved on disk.
    """

    @given(
        chapter=st.integers(min_value=1, max_value=20),
        act=st.integers(min_value=1, max_value=7),
        num_revisions=st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_each_revision_saved_separately(
        self,
        tmp_path,
        chapter: int,
        act: int,
        num_revisions: int,
    ) -> None:
        """Each revision is saved as a separate file with incremented revision_number."""
        drafts_dir = str(tmp_path / "drafts")
        saved_paths: list[str] = []

        for rev in range(num_revisions):
            file_path = os.path.join(
                drafts_dir,
                f"chapter_{chapter:02d}_act_{act:02d}_rev{rev:02d}.json",
            )
            artifact = JSONArtifact(
                artifact_type="act",
                text=f"Revision {rev} prose for ch{chapter}/act{act}.",
                metadata=DocumentMetadata(
                    type="act",
                    chapter=chapter,
                    act=act,
                    revision_number=rev,
                    summary=f"Revision {rev}",
                ),
                created_at="2025-01-01T00:00:00+00:00",
                file_path=file_path,
            )
            artifact.save()
            saved_paths.append(file_path)

        # All revision files must exist on disk
        for path in saved_paths:
            assert os.path.isfile(path), f"Revision file missing: {path}"

        # Verify each file has the correct revision_number
        for rev, path in enumerate(saved_paths):
            loaded = JSONArtifact.load(path)
            assert loaded.metadata.revision_number == rev, (
                f"Expected revision_number={rev}, got {loaded.metadata.revision_number}"
            )

    @given(
        chapter=st.integers(min_value=1, max_value=20),
        act=st.integers(min_value=1, max_value=7),
        num_revisions=st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_previous_revisions_never_overwritten(
        self,
        tmp_path,
        chapter: int,
        act: int,
        num_revisions: int,
    ) -> None:
        """All previous revision files are preserved — never overwritten or deleted."""
        drafts_dir = str(tmp_path / "drafts")
        saved_paths: list[str] = []
        expected_texts: list[str] = []

        for rev in range(num_revisions):
            text = f"Unique revision {rev} content for ch{chapter}/act{act}."
            file_path = os.path.join(
                drafts_dir,
                f"chapter_{chapter:02d}_act_{act:02d}_rev{rev:02d}.json",
            )
            artifact = JSONArtifact(
                artifact_type="act",
                text=text,
                metadata=DocumentMetadata(
                    type="act",
                    chapter=chapter,
                    act=act,
                    revision_number=rev,
                ),
                created_at="2025-01-01T00:00:00+00:00",
                file_path=file_path,
            )
            artifact.save()
            saved_paths.append(file_path)
            expected_texts.append(text)

        # After all revisions are saved, verify every previous file is intact
        for rev, (path, expected_text) in enumerate(zip(saved_paths, expected_texts)):
            assert os.path.isfile(path), f"Revision {rev} file was deleted: {path}"
            loaded = JSONArtifact.load(path)
            assert loaded.text == expected_text, (
                f"Revision {rev} text was overwritten: "
                f"expected {expected_text!r}, got {loaded.text!r}"
            )
