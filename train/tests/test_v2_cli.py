"""Unit tests for the v2 CLI entry point (cli.py).

Validates: Requirements 7.1, 7.2, 7.3, 7.4
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from romance_factory.core import config as core_config
from romance_factory.generate.cli import _apply_llm_overrides, _build_parser


class TestBuildParser:
    """Test CLI argument parsing for the v2 entry point."""

    def test_all_args_default_to_none_except_story_path(self):
        """Req 7.4 — unprovided args should be None so precedence chain resolves."""
        parser = _build_parser()
        args = parser.parse_args(["--story-path", "stories/test"])

        assert args.story_path == "stories/test"
        assert args.db_path is None
        assert args.embedding_model is None
        assert args.top_k is None
        assert args.max_context_chars is None
        assert args.threshold is None
        assert args.max_rewrites is None
        assert args.num_chapters is None
        assert args.min_acts is None
        assert args.max_acts is None
        assert args.editorial_max_retries is None

    def test_editorial_max_retries_parsed(self):
        """Req 7.2 — --editorial-max-retries is accepted and parsed as int."""
        parser = _build_parser()
        args = parser.parse_args(["--story-path", "s", "--editorial-max-retries", "5"])
        assert args.editorial_max_retries == 5

    def test_all_existing_args_still_parsed(self):
        """Req 7.1 — all existing CLI args are still accepted."""
        parser = _build_parser()
        args = parser.parse_args([
            "--story-path", "stories/test",
            "--db-path", "/tmp/lance",
            "--embedding-model", "e5-large",
            "--top-k", "20",
            "--max-context-chars", "4000",
            "--threshold", "7.5",
            "--max-rewrites", "2",
            "--num-chapters", "5",
            "--min-acts", "2",
            "--max-acts", "4",
            "--editorial-max-retries", "3",
        ])
        assert args.story_path == "stories/test"
        assert args.db_path == "/tmp/lance"
        assert args.embedding_model == "e5-large"
        assert args.top_k == 20
        assert args.max_context_chars == 4000
        assert args.threshold == 7.5
        assert args.max_rewrites == 2
        assert args.num_chapters == 5
        assert args.min_acts == 2
        assert args.max_acts == 4
        assert args.editorial_max_retries == 3

    def test_llm_backend_args_still_accepted(self):
        """Req 7.3 — LLM backend args are still supported."""
        parser = _build_parser()
        args = parser.parse_args([
            "--story-path", "s",
            "--llm-backend", "openrouter",
            "--model", "anthropic/claude-sonnet-4",
            "--ollama-url", "http://localhost:1234/v1",
        ])
        assert args.llm_backend == "openrouter"
        assert args.model == "anthropic/claude-sonnet-4"
        assert args.ollama_url == "http://localhost:1234/v1"


class TestApplyLlmOverrides:
    """Test _apply_llm_overrides updates ``romance_factory.core.config`` globals."""

    def test_sets_backend_on_config(self, monkeypatch):
        """Req 7.3 — --llm-backend updates config.LLM_BACKEND."""
        prior = core_config.LLM_BACKEND
        try:
            parser = _build_parser()
            args = parser.parse_args(["--story-path", "s", "--llm-backend", "openrouter"])
            _apply_llm_overrides(args)
            assert core_config.LLM_BACKEND == "openrouter"
        finally:
            core_config.LLM_BACKEND = prior

    def test_sets_model_on_config(self, monkeypatch):
        """Req 7.3 — --model updates MODEL_NAME and OPENROUTER_MODEL."""
        pm, por = core_config.MODEL_NAME, core_config.OPENROUTER_MODEL
        try:
            parser = _build_parser()
            args = parser.parse_args(["--story-path", "s", "--model", "my-model"])
            _apply_llm_overrides(args)
            assert core_config.MODEL_NAME == "my-model"
            assert core_config.OPENROUTER_MODEL == "my-model"
        finally:
            core_config.MODEL_NAME = pm
            core_config.OPENROUTER_MODEL = por

    def test_sets_ollama_url_on_config(self, monkeypatch):
        """Req 7.3 — --ollama-url updates OLLAMA_URL."""
        prior = core_config.OLLAMA_URL
        try:
            parser = _build_parser()
            args = parser.parse_args(["--story-path", "s", "--ollama-url", "http://x:1234"])
            _apply_llm_overrides(args)
            assert core_config.OLLAMA_URL == "http://x:1234"
        finally:
            core_config.OLLAMA_URL = prior

    def test_no_change_when_llm_args_not_provided(self, monkeypatch):
        """Req 7.3 — when LLM args are not provided, globals are untouched."""
        bk, m, ou, om = (
            core_config.LLM_BACKEND,
            core_config.MODEL_NAME,
            core_config.OLLAMA_URL,
            core_config.OPENROUTER_MODEL,
        )
        parser = _build_parser()
        args = parser.parse_args(["--story-path", "s"])
        _apply_llm_overrides(args)
        assert core_config.LLM_BACKEND == bk
        assert core_config.MODEL_NAME == m
        assert core_config.OLLAMA_URL == ou
        assert core_config.OPENROUTER_MODEL == om


class TestMainCallsLoadV2Config:
    """Test that main() passes CLI args to load_v2_config correctly."""

    @staticmethod
    def _stub_pipeline_v2_module() -> None:
        """Avoid importing the real pipeline (pulls lancedb) when main() loads PipelineV2."""
        fake = ModuleType("romance_factory.generate.pipeline_v2")
        inst = MagicMock()
        inst.run.return_value = {"phases_completed": []}
        fake.PipelineV2 = MagicMock(return_value=inst)
        sys.modules["romance_factory.generate.pipeline_v2"] = fake

    @patch("romance_factory.generate.config_v2.load_v2_config")
    def test_load_v2_config_called_with_cli_args(self, mock_load):
        """Req 7.1, 7.2 — CLI args are forwarded as keyword overrides."""
        from romance_factory.generate.config_v2 import V2Config

        mock_load.return_value = V2Config(story_path="stories/test")
        self._stub_pipeline_v2_module()
        try:
            from romance_factory.generate.cli import main

            main([
                "--story-path", "stories/test",
                "--threshold", "7.0",
                "--editorial-max-retries", "4",
            ])
        finally:
            sys.modules.pop("romance_factory.generate.pipeline_v2", None)

        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["story_path"] == "stories/test"
        assert call_kwargs["passing_score_threshold"] == 7.0
        assert call_kwargs["editorial_max_retries"] == 4

    @patch("romance_factory.generate.config_v2.load_v2_config")
    def test_unprovided_args_passed_as_none(self, mock_load):
        """Req 7.4 — unprovided args are passed as None."""
        from romance_factory.generate.config_v2 import V2Config

        mock_load.return_value = V2Config(story_path="stories/test")
        self._stub_pipeline_v2_module()
        try:
            from romance_factory.generate.cli import main

            main(["--story-path", "stories/test"])
        finally:
            sys.modules.pop("romance_factory.generate.pipeline_v2", None)

        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["db_path"] is None
        assert call_kwargs["embedding_model"] is None
        assert call_kwargs["default_top_k"] is None
        assert call_kwargs["max_context_chars"] is None
        assert call_kwargs["passing_score_threshold"] is None
        assert call_kwargs["max_rewrite_iterations_per_act"] is None
        assert call_kwargs["num_chapters"] is None
        assert call_kwargs["min_acts_per_chapter"] is None
        assert call_kwargs["max_acts_per_chapter"] is None
        assert call_kwargs["editorial_max_retries"] is None
