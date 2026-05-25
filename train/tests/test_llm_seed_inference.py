"""Unit tests for per-request LLM seed selection (no network)."""

from __future__ import annotations

from unittest.mock import patch

import romance_factory.core.ollama_client as oc


def test_inference_seed_omit_returns_none():
    with patch.object(oc._cfg, "LLM_SEED_MODE", "omit"):
        assert oc._inference_seed_for_request() is None


def test_inference_seed_fixed_returns_configured_value():
    with (
        patch.object(oc._cfg, "LLM_SEED_MODE", "fixed"),
        patch.object(oc._cfg, "LLM_SEED_FIXED", 123456789),
    ):
        assert oc._inference_seed_for_request() == 123456789


def test_inference_seed_random_mode_produces_variation():
    """Random mode should not return a single constant across repeated draws."""
    with patch.object(oc._cfg, "LLM_SEED_MODE", "random"):
        draws = [oc._inference_seed_for_request() for _ in range(40)]
    assert all(isinstance(x, int) for x in draws)
    assert len(set(draws)) >= 10
