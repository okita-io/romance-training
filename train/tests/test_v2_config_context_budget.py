"""Tests for llm_context_tokens-derived prompt caps."""

from __future__ import annotations

from romance_factory.generate.config_v2 import (
    V2Config,
    apply_llm_context_token_budget,
)


def test_apply_budget_derives_caps_when_not_overridden() -> None:
    cfg = V2Config(
        llm_context_tokens=131072,
        max_tokens=12000,
        max_context_chars=8000,
        max_story_arc_chars=56000,
        max_previous_acts_chars=8000,
    )
    apply_llm_context_token_budget(cfg, {})
    assert cfg.max_context_chars > 8000
    assert cfg.max_story_arc_chars > 56000
    assert cfg.max_previous_acts_chars >= 2000


def test_apply_budget_respects_explicit_resolved_keys() -> None:
    cfg = V2Config(
        llm_context_tokens=100000,
        max_tokens=8000,
        max_context_chars=3000,
        max_story_arc_chars=9000,
        max_previous_acts_chars=4000,
    )
    resolved = {
        "max_context_chars": 3000,
        "max_story_arc_chars": 9000,
        "max_previous_acts_chars": 4000,
    }
    apply_llm_context_token_budget(cfg, resolved)
    assert cfg.max_context_chars == 3000
    assert cfg.max_story_arc_chars == 9000
    assert cfg.max_previous_acts_chars == 4000


def test_apply_budget_noop_without_tokens() -> None:
    cfg = V2Config(max_context_chars=1234)
    apply_llm_context_token_budget(cfg, {})
    assert cfg.max_context_chars == 1234
