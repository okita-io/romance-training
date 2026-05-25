"""Unit tests for the ActGenerationAgent.

Tests the agent's context retrieval, prompt building, LLM invocation,
and response parsing logic.
"""

from __future__ import annotations

from unittest.mock import ANY, MagicMock, patch

import pytest

from romance_factory.generate.agents.act_generation import (
    ActGenerationAgent,
)
from romance_factory.generate.config_v2 import V2Config
from romance_factory.generate.models import (
    ActResult,
    DocumentMetadata,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retrieval_result(text: str = "some text", **meta_kw) -> RetrievalResult:
    defaults = dict(type="act", chapter=1, act=1)
    defaults.update(meta_kw)
    return RetrievalResult(
        text=text,
        metadata=DocumentMetadata(**defaults),
        similarity_score=0.9,
    )


def _stub_engine() -> MagicMock:
    """Return a mock LanceDBEngine whose query() returns a single result."""
    engine = MagicMock()
    engine.query.return_value = [_make_retrieval_result()]
    return engine


def _stub_builder() -> MagicMock:
    """Return a mock PromptBuilder."""
    builder = MagicMock()
    builder.build_act_generation_prompt.return_value = (
        "Write act 1 of chapter 1.",
        "You are a romance novelist.",
    )
    return builder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestActGenerationAgentGenerate:
    """Core generate() flow."""

    @patch("romance_factory.generate.agents.act_generation.generate")
    def test_returns_act_result_with_plain_prose(self, mock_llm):
        """Plain prose (no JSON block) → ActResult with full text, empty metadata lists."""
        mock_llm.return_value = "She stepped into the rain."

        agent = ActGenerationAgent(_stub_engine(), _stub_builder())
        result = agent.generate(chapter=1, act=2)

        assert isinstance(result, ActResult)
        assert result.text == "She stepped into the rain."
        assert result.metadata.type == "act"
        assert result.metadata.chapter == 1
        assert result.metadata.act == 2
        assert result.metadata.characters_involved == []
        assert result.metadata.emotional_tone == ""
        assert result.summary == ""

    @patch("romance_factory.generate.agents.act_generation.generate")
    def test_parses_json_metadata_block(self, mock_llm):
        """Prose followed by a ```json block → metadata extracted, JSON stripped."""
        mock_llm.return_value = (
            "She stepped into the rain.\n\n"
            '```json\n'
            '{"characters_involved": ["Alice", "Bob"], '
            '"emotional_tone": "tension", '
            '"plot_function": "inciting_incident", '
            '"summary": "Alice meets Bob in the rain.", '
            '"foreshadowing_created": ["umbrella motif"], '
            '"relationship_changes": ["first meeting"]}\n'
            '```'
        )

        agent = ActGenerationAgent(_stub_engine(), _stub_builder())
        result = agent.generate(chapter=2, act=3)

        assert result.text == "She stepped into the rain."
        assert result.metadata.characters_involved == ["Alice", "Bob"]
        assert result.metadata.emotional_tone == "tension"
        assert result.metadata.plot_function == "inciting_incident"
        assert result.summary == "Alice meets Bob in the rain."
        assert result.metadata.foreshadowing_created == ["umbrella motif"]
        assert result.metadata.relationship_changes == ["first meeting"]

    @patch("romance_factory.generate.agents.act_generation.generate")
    def test_malformed_json_falls_back_to_plain_text(self, mock_llm):
        """Malformed JSON block → treated as plain prose, no crash."""
        mock_llm.return_value = (
            "She stepped into the rain.\n\n"
            "```json\n{bad json\n```"
        )

        agent = ActGenerationAgent(_stub_engine(), _stub_builder())
        result = agent.generate(chapter=1, act=1)

        # The whole raw text (minus the json block) is kept as prose
        assert isinstance(result, ActResult)
        assert result.metadata.characters_involved == []

    @patch("romance_factory.generate.agents.act_generation.generate")
    def test_empty_llm_response(self, mock_llm):
        """Empty LLM response → ActResult with empty text."""
        mock_llm.return_value = ""

        agent = ActGenerationAgent(_stub_engine(), _stub_builder())
        result = agent.generate(chapter=1, act=1)

        assert result.text == ""
        assert result.metadata.type == "act"


class TestActGenerationAgentContextRetrieval:
    """Verify the agent queries the right LanceDB collections."""

    @patch("romance_factory.generate.agents.act_generation.generate")
    def test_queries_all_expected_collections(self, mock_llm):
        mock_llm.return_value = "prose"
        engine = _stub_engine()
        builder = _stub_builder()

        agent = ActGenerationAgent(engine, builder)
        agent.generate(chapter=3, act=2)

        # Collect all collection names from query calls
        collections_queried = [
            call.args[0] for call in engine.query.call_args_list
        ]
        assert "author_profile" in collections_queried
        assert "character_web" in collections_queried
        assert "world" in collections_queried
        assert "story_outline" in collections_queried
        assert "story_beats" in collections_queried
        assert "acts" in collections_queried

    def test_world_query_uses_retrieval_top_k_world(self):
        """World collection query SHALL use V2Config.retrieval_top_k_world (req 9.4)."""
        engine = _stub_engine()
        builder = _stub_builder()
        cfg = V2Config(retrieval_top_k_world=7)
        agent = ActGenerationAgent(engine, builder, config=cfg)
        agent._retrieve_context(chapter=2, act=1)

        world_calls = [
            c for c in engine.query.call_args_list if c.args[0] == "world"
        ]
        assert len(world_calls) == 1
        assert world_calls[0].kwargs.get("top_k") == 7

    @patch("romance_factory.generate.agents.act_generation.generate")
    def test_passes_chapter_act_filters(self, mock_llm):
        mock_llm.return_value = "prose"
        engine = _stub_engine()
        builder = _stub_builder()

        agent = ActGenerationAgent(engine, builder)
        agent.generate(chapter=5, act=3)

        # Find the story_beats query that filters by chapter+act
        for call in engine.query.call_args_list:
            if call.args[0] == "story_beats" and call.kwargs.get("metadata_filters"):
                filters = call.kwargs["metadata_filters"]
                if "act" in filters:
                    assert filters["chapter"] == 5
                    assert filters["act"] == 3
                    break
        else:
            pytest.fail("No story_beats query with chapter+act filter found")


class TestActGenerationAgentPromptDelegation:
    """Verify prompt builder is called with correct flags."""

    @patch("romance_factory.generate.agents.act_generation.generate")
    def test_passes_is_last_act_flag(self, mock_llm):
        mock_llm.return_value = "prose"
        engine = _stub_engine()
        builder = _stub_builder()

        agent = ActGenerationAgent(engine, builder)
        agent.generate(chapter=1, act=5, is_last_act=True)

        _, kwargs = builder.build_act_generation_prompt.call_args
        assert kwargs["is_last_act"] is True

    @patch("romance_factory.generate.agents.act_generation.generate")
    def test_passes_is_plot_twist_flag(self, mock_llm):
        mock_llm.return_value = "prose"
        engine = _stub_engine()
        builder = _stub_builder()

        agent = ActGenerationAgent(engine, builder)
        agent.generate(chapter=2, act=3, is_plot_twist=True)

        _, kwargs = builder.build_act_generation_prompt.call_args
        assert kwargs["is_plot_twist"] is True

    @patch("romance_factory.generate.agents.act_generation.generate")
    def test_calls_llm_with_prompt_and_system(self, mock_llm):
        # Prose call returns plain text; metadata extraction must return valid JSON
        # so _fill_act_metadata_gaps does not issue extra generate() retries.
        meta_json = (
            '{"summary": "s", "characters_involved": [], '
            '"emotional_tone": "calm", "plot_function": "setup", '
            '"foreshadowing_created": [], "relationship_changes": []}'
        )
        mock_llm.side_effect = ("prose", meta_json)
        engine = _stub_engine()
        builder = _stub_builder()
        builder.build_act_generation_prompt.return_value = (
            "my prompt",
            "my system",
        )

        agent = ActGenerationAgent(engine, builder)
        agent.generate(chapter=1, act=1)

        assert mock_llm.call_count == 2
        mock_llm.assert_any_call(
            "my prompt", system_prompt="my system", progress_hint=ANY
        )
