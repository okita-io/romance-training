"""Tests for small, deterministic helpers on the config module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import romance_factory.core.config as config


def test_format_compact_local_ts_fixed_instant():
    dt = datetime(2025, 3, 25, 14, 30, 45, tzinfo=timezone.utc)
    assert config.format_compact_local_ts(dt) == "250325143045"


def test_format_elapsed_hhmmss():
    start = datetime(2025, 1, 1, 10, 0, 0)
    end = start + timedelta(hours=1, minutes=2, seconds=3)
    assert config.format_elapsed_hhmmss(start, end) == "010203"


def test_infer_llm_inference_system_label_openrouter(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_URL", "https://openrouter.ai/api/v1/chat/completions")
    monkeypatch.setattr(config, "LLM_BACKEND", "openai_chat")
    assert config.infer_llm_inference_system_label() == "OpenRouter"


def test_get_llm_inference_profile_keys(monkeypatch):
    monkeypatch.setattr(config, "LLM_INFERENCE_SYSTEM", "TestStack")
    monkeypatch.setattr(config, "MODEL_NAME", "test-model")
    monkeypatch.setattr(config, "LLM_BACKEND", "openai_chat")
    monkeypatch.setattr(config, "RUN_SEED", None)
    monkeypatch.setattr(config, "LLM_SEED_MODE", "random")
    monkeypatch.setattr(config, "LLM_SEED_FIXED", None)
    prof = config.get_llm_inference_profile()
    assert prof == {
        "llm_inference_system": "TestStack",
        "llm_model": "test-model",
        "llm_backend": "openai_chat",
        "llm_seed_mode": "random",
    }
