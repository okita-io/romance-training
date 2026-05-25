"""Tests for Pipeline V2 Phase 7 (Rough Draft Acts) and Phase 8 (Chapter Assembly).

These tests use mocked agents and LanceDB engine to verify the orchestration
logic without requiring actual LLM calls or embedding models.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from romance_factory.story_core.story_state import StoryState
from romance_factory.generate.config_v2 import V2Config
from romance_factory.generate.models import (
    ActResult,
    DocumentMetadata,
    JSONArtifact,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_outline(num_chapters: int = 2, acts_per_chapter: int = 3) -> dict:
    """Build a minimal story outline dict."""
    chapters = []
    for ch in range(1, num_chapters + 1):
        acts = []
        for a in range(1, acts_per_chapter + 1):
            acts.append({
                "act_number": a,
                "summary": f"Ch{ch} Act{a} summary",
                "characters_involved": ["Alice", "Bob"],
                "emotional_tone": "tension",
                "plot_function": "rising_action",
                "is_plot_twist": a == acts_per_chapter,  # last act is twist
            })
        chapters.append({
            "chapter_number": ch,
            "title": f"Chapter {ch}",
            "summary": f"Chapter {ch} summary",
            "acts": acts,
        })
    return {
        "story_arc": {
            "title": "Test Novel",
            "premise": "Test premise for phase tests.",
            "central_conflict": "Conflict",
            "romantic_arc": "Arc",
            "theme": "Theme",
            "setting": "Setting",
            "num_chapters": num_chapters,
        },
        "chapters": chapters,
    }


def _save_outline_artifact(story_path: str, outline: dict) -> str:
    """Save an outline as a JSONArtifact (mimicking Phase 4 output)."""
    text = json.dumps(outline, indent=2, ensure_ascii=False)
    artifact = JSONArtifact(
        artifact_type="story_outline",
        text=text,
        metadata=DocumentMetadata(type="outline", summary="test outline"),
        created_at=datetime.now(timezone.utc).isoformat(),
        file_path=os.path.join(story_path, "story_outline.json"),
        parsed_data=outline,
    )
    artifact.save()
    return artifact.file_path


def _make_act_result(chapter: int, act: int) -> ActResult:
    """Build a minimal ActResult for mocking."""
    # Phase 7 runs act_draft_tier1_fail_reason which requires enough prose words
    # after reasoning-strip (see V2Config.min_words_per_act floor in draft_quality).
    filler = ("They walked through the evening air and spoke softly about nothing "
              "and everything while the city lights blurred into gold. ") * 12
    text = f"Prose for chapter {chapter} act {act}. {filler}"
    return ActResult(
        text=text,
        metadata=DocumentMetadata(
            type="act",
            chapter=chapter,
            act=act,
            characters_involved=["Alice", "Bob"],
            emotional_tone="tension",
            plot_function="rising_action",
            summary=f"Summary ch{chapter} act{act}",
        ),
        summary=f"Summary ch{chapter} act{act}",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def story_dir(tmp_path):
    """Create a temporary story directory with an outline artifact."""
    story_path = str(tmp_path / "test_story")
    os.makedirs(story_path, exist_ok=True)
    outline = _make_outline(num_chapters=2, acts_per_chapter=3)
    _save_outline_artifact(story_path, outline)
    return story_path


@pytest.fixture()
def mock_pipeline(story_dir):
    """Build a PipelineV2 with all external dependencies mocked."""
    config = V2Config(
        story_path=story_dir,
        db_path=os.path.join(story_dir, "lancedb"),
        embedding_model="mock",
        num_chapters=2,
        min_acts_per_chapter=3,
        max_acts_per_chapter=5,
        # Avoid no_advancement warnings when update_story_state_from_act is stubbed.
        enable_progression_enforcement=False,
        # Resume / phase tests must not call a real LLM for draft sanity checks.
        disable_act_draft_llm_sanity=True,
    )

    with (
        patch("romance_factory.generate.pipeline_v2.EmbeddingProvider"),
        patch("romance_factory.generate.pipeline_v2.LanceDBEngine") as MockEngine,
        patch("romance_factory.generate.pipeline_v2.PromptBuilder"),
        patch("romance_factory.generate.pipeline_v2.ActGenerationAgent") as MockActAgent,
        patch(
            "romance_factory.generate.pipeline_v2.ActIntroPlanningAgent",
        ) as MockIntroAgent,
        patch("romance_factory.generate.pipeline_v2.EditorialAgent"),
        patch("romance_factory.generate.pipeline_v2.RewriteAgent"),
    ):
        # Configure mock engine
        engine_instance = MockEngine.return_value
        engine_instance.validate_collections.return_value = True
        engine_instance.store_document.return_value = None
        engine_instance.replace_document.return_value = None
        engine_instance.query.return_value = []

        # Configure mock act generation agent
        act_agent_instance = MockActAgent.return_value

        def _mock_generate(chapter, act, is_last_act=False, is_plot_twist=False, **kwargs):
            return _make_act_result(chapter, act)

        act_agent_instance.generate.side_effect = _mock_generate

        intro_instance = MockIntroAgent.return_value
        intro_instance.plan_after_act.return_value = ""

        # Patch update_story_state_from_act to avoid LLM calls
        with patch(
            "romance_factory.generate.pipeline_v2.update_story_state_from_act",
            return_value=StoryState(),
        ) as mock_update_state:
            from romance_factory.generate.pipeline_v2 import PipelineV2

            pipeline = PipelineV2(story_dir, config)
            pipeline._mock_update_state = mock_update_state
            # Phase 8 may run a light editorial touch-up after repair; keep score
            # at/above threshold so the rewrite path (mock MagicMock text) never runs.
            pipeline.editorial_agent.evaluate.return_value = MagicMock(
                score=float(config.passing_score_threshold) + 1.0,
                issues=[],
            )
            yield pipeline


# ---------------------------------------------------------------------------
# Phase 6 Tests
# ---------------------------------------------------------------------------


class TestPhase07RoughDraftActs:
    """Tests for _phase_07_rough_draft_acts."""

    def test_generates_all_acts(self, mock_pipeline):
        """All acts from the outline should be generated."""
        mock_pipeline._phase_07_rough_draft_acts()

        # 2 chapters × 3 acts = 6 calls
        assert mock_pipeline.act_generation_agent.generate.call_count == 6

    def test_saves_json_artifacts(self, mock_pipeline):
        """Each act should be saved as a JSON artifact on disk."""
        mock_pipeline._phase_07_rough_draft_acts()

        drafts_dir = os.path.join(mock_pipeline.story_path, "drafts")
        for ch in range(1, 3):
            for act in range(1, 4):
                path = os.path.join(drafts_dir, f"chapter_{ch:02d}_act_{act:02d}.json")
                assert os.path.isfile(path), f"Missing draft: {path}"

    def test_stores_in_lancedb(self, mock_pipeline):
        """Each act should be stored in LanceDB acts collection."""
        mock_pipeline._phase_07_rough_draft_acts()

        # 6 acts stored (replace_document is the canonical upsert for act slots).
        assert mock_pipeline.engine.replace_document.call_count == 6
        for call in mock_pipeline.engine.replace_document.call_args_list:
            assert call[0][0] == "acts"

    def test_updates_story_state(self, mock_pipeline):
        """update_story_state_from_act should be called for each act."""
        mock_pipeline._phase_07_rough_draft_acts()
        assert mock_pipeline._mock_update_state.call_count == 6

    def test_resume_skips_existing_drafts(self, mock_pipeline):
        """Acts with existing draft files should be skipped."""
        # Pre-create one draft file
        drafts_dir = os.path.join(mock_pipeline.story_path, "drafts")
        os.makedirs(drafts_dir, exist_ok=True)
        ar1 = _make_act_result(1, 1)
        existing = JSONArtifact(
            artifact_type="act",
            text=ar1.text,
            metadata=ar1.metadata,
            created_at=datetime.now(timezone.utc).isoformat(),
            file_path=os.path.join(drafts_dir, "chapter_01_act_01.json"),
        )
        existing.save()

        mock_pipeline._phase_07_rough_draft_acts()

        # Should skip ch1/act1, generate 5 instead of 6
        assert mock_pipeline.act_generation_agent.generate.call_count == 5

    def test_is_last_act_flag(self, mock_pipeline):
        """The last act in each chapter should have is_last_act=True."""
        mock_pipeline._phase_07_rough_draft_acts()

        calls = mock_pipeline.act_generation_agent.generate.call_args_list
        for call in calls:
            ch = call.kwargs.get("chapter", call[1].get("chapter") if len(call) > 1 else call[0][0])
            act = call.kwargs.get("act", call[1].get("act") if len(call) > 1 else call[0][1])
            is_last = call.kwargs.get("is_last_act", False)
            # Act 3 is the last in our 3-act chapters
            if act == 3:
                assert is_last is True
            else:
                assert is_last is False

    def test_is_plot_twist_flag(self, mock_pipeline):
        """Acts marked as plot twists in the outline should pass the flag."""
        mock_pipeline._phase_07_rough_draft_acts()

        calls = mock_pipeline.act_generation_agent.generate.call_args_list
        for call in calls:
            act = call.kwargs.get("act", False)
            is_twist = call.kwargs.get("is_plot_twist", False)
            # In our outline, act 3 (last) is the twist
            if act == 3:
                assert is_twist is True
            else:
                assert is_twist is False

    def test_no_editorial_during_phase7(self, mock_pipeline):
        """Editorial and rewrite agents should NOT be called."""
        mock_pipeline._phase_07_rough_draft_acts()
        mock_pipeline.editorial_agent.evaluate.assert_not_called()
        mock_pipeline.rewrite_agent.rewrite.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 7 Tests
# ---------------------------------------------------------------------------


class TestPhase08ChapterAssembly:
    """Tests for _phase_08_chapter_assembly."""

    def _setup_acts_in_lancedb(self, mock_pipeline, num_chapters=2, acts_per_chapter=3):
        """Configure mock LanceDB chapter assembly (uses filter_scan, not query)."""
        def _mock_filter_scan(collection, filters):
            if collection != "acts":
                return []
            ch = filters.get("chapter", 0)
            typ = filters.get("type")
            if typ != "act" or not ch:
                return []
            results = []
            for a in range(1, acts_per_chapter + 1):
                ar = _make_act_result(ch, a)
                results.append(RetrievalResult(
                    text=ar.text,
                    metadata=DocumentMetadata(
                        type="act", chapter=ch, act=a,
                        summary=f"Summary ch{ch} act{a}",
                    ),
                    similarity_score=0.9,
                ))
            return results

        mock_pipeline.engine.filter_scan.side_effect = _mock_filter_scan

    def test_assembles_all_chapters(self, mock_pipeline):
        """Each chapter should be assembled and saved."""
        self._setup_acts_in_lancedb(mock_pipeline)
        mock_pipeline._phase_08_chapter_assembly()

        drafts_dir = os.path.join(mock_pipeline.story_path, "drafts")
        for ch in range(1, 3):
            path = os.path.join(drafts_dir, f"chapter_{ch:02d}.json")
            assert os.path.isfile(path), f"Missing chapter: {path}"

    def test_concatenates_acts_in_order(self, mock_pipeline):
        """Acts should be concatenated in act-number order."""
        self._setup_acts_in_lancedb(mock_pipeline)
        mock_pipeline._phase_08_chapter_assembly()

        drafts_dir = os.path.join(mock_pipeline.story_path, "drafts")
        artifact = JSONArtifact.load(os.path.join(drafts_dir, "chapter_01.json"))
        # Should contain all 3 acts separated by \n\n
        assert "chapter 1 act 1" in artifact.text
        assert "chapter 1 act 2" in artifact.text
        assert "chapter 1 act 3" in artifact.text

    def test_stores_chapters_in_lancedb(self, mock_pipeline):
        """Each chapter should be stored in story_outline collection."""
        self._setup_acts_in_lancedb(mock_pipeline)
        mock_pipeline._phase_08_chapter_assembly()

        # 2 chapters stored (assembled rows use replace_document).
        chap_calls = [
            c for c in mock_pipeline.engine.replace_document.call_args_list
            if c[0][0] == "story_outline"
        ]
        assert len(chap_calls) == 2
        for call in chap_calls:
            assert call[0][2].type == "chapter"

    def test_chapter_metadata_type(self, mock_pipeline):
        """Chapter metadata should have type='chapter'."""
        self._setup_acts_in_lancedb(mock_pipeline)
        mock_pipeline._phase_08_chapter_assembly()

        drafts_dir = os.path.join(mock_pipeline.story_path, "drafts")
        artifact = JSONArtifact.load(os.path.join(drafts_dir, "chapter_01.json"))
        assert artifact.metadata.type == "chapter"
        assert artifact.metadata.chapter == 1

    def test_no_editorial_during_phase8(self, mock_pipeline):
        """Editorial and rewrite agents should NOT be called."""
        self._setup_acts_in_lancedb(mock_pipeline)
        mock_pipeline._phase_08_chapter_assembly()
        mock_pipeline.editorial_agent.evaluate.assert_not_called()
        mock_pipeline.rewrite_agent.rewrite.assert_not_called()

    def test_skips_chapter_with_no_acts(self, mock_pipeline):
        """Outline chapters with no act beats are skipped (nothing to assemble)."""
        outline_path = os.path.join(mock_pipeline.story_path, "story_outline.json")
        outline = _make_outline(num_chapters=1, acts_per_chapter=1)
        outline["chapters"] = [
            {
                "chapter_number": 1,
                "title": "Empty chapter",
                "summary": "No beats",
                "acts": [],
            },
        ]
        JSONArtifact(
            artifact_type="story_outline",
            text=json.dumps(outline, indent=2, ensure_ascii=False),
            metadata=DocumentMetadata(type="outline", summary="test"),
            created_at=datetime.now(timezone.utc).isoformat(),
            file_path=outline_path,
            parsed_data=outline,
        ).save()

        mock_pipeline._phase_08_chapter_assembly()

        chap_calls = [
            c for c in mock_pipeline.engine.replace_document.call_args_list
            if c[0][0] == "story_outline" and getattr(c[0][2], "type", None) == "chapter"
        ]
        assert len(chap_calls) == 0

    def test_paragraph_separator(self, mock_pipeline):
        """Acts should be joined with double newlines."""
        # Align outline act count with mocked LanceDB rows (default outline has 3 acts).
        outline_path = os.path.join(mock_pipeline.story_path, "story_outline.json")
        outline = _make_outline(num_chapters=2, acts_per_chapter=2)
        JSONArtifact(
            artifact_type="story_outline",
            text=json.dumps(outline, indent=2, ensure_ascii=False),
            metadata=DocumentMetadata(type="outline", summary="test"),
            created_at=datetime.now(timezone.utc).isoformat(),
            file_path=outline_path,
            parsed_data=outline,
        ).save()

        self._setup_acts_in_lancedb(mock_pipeline, acts_per_chapter=2)
        mock_pipeline._phase_08_chapter_assembly()

        drafts_dir = os.path.join(mock_pipeline.story_path, "drafts")
        artifact = JSONArtifact.load(os.path.join(drafts_dir, "chapter_01.json"))
        # The text should have \n\n between acts
        parts = artifact.text.split("\n\n")
        assert len(parts) >= 2


# ---------------------------------------------------------------------------
# _load_outline Tests
# ---------------------------------------------------------------------------


class TestLoadOutline:
    """Tests for the _load_outline helper."""

    def test_loads_valid_outline(self, mock_pipeline):
        """Should parse a valid outline from a JSONArtifact."""
        result = mock_pipeline._load_outline(
            os.path.join(mock_pipeline.story_path, "story_outline.json")
        )
        assert "chapters" in result
        assert len(result["chapters"]) == 2

    def test_returns_empty_for_missing_file(self, mock_pipeline):
        """Should return empty dict for missing file."""
        result = mock_pipeline._load_outline("/nonexistent/path.json")
        assert result == {}
