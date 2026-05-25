"""Optional live call to OpenRouter (free router). Requires network + API key."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import romance_factory.core.config as config
import romance_factory.core.ollama_client as client


def _openrouter_live_enabled() -> bool:
    v = (os.environ.get("ROMANCE_FACTORY_OPENROUTER_LIVE") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


@pytest.mark.openrouter_live
def test_openrouter_free_smoke():
    """One short generation via ``openrouter/free`` (override with OPENROUTER_LIVE_MODEL)."""
    if not _openrouter_live_enabled():
        pytest.skip(
            "Set ROMANCE_FACTORY_OPENROUTER_LIVE=1 to run this network test "
            "(requires OPENROUTER_API_KEY and a reachable OpenRouter or gateway URL)."
        )
    if not (config.OPENROUTER_API_KEY or "").strip():
        pytest.skip(
            "OPENROUTER_API_KEY not set — add to repo-root .env or export it "
            "(loaded automatically when config imports)."
        )

    model = (os.environ.get("OPENROUTER_LIVE_MODEL") or "openrouter/free").strip()

    with (
        patch.object(client, "LLM_BACKEND", "openrouter"),
        patch.object(client, "OPENROUTER_MODEL", model),
        patch.object(client, "LLM_STREAM", False),
        patch.object(client, "LLM_SEED_MODE", "omit"),
    ):
        text = client.generate(
            'Reply with a single word only: "pong". No punctuation or explanation.',
            max_tokens=32,
            temperature=0.2,
        )

    assert text.strip(), "OpenRouter returned empty text"
    lowered = text.strip().lower()
    assert "pong" in lowered, f"expected 'pong' in response, got: {text!r}"
