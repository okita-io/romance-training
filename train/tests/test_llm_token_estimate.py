"""Approximate LLM prompt token estimation for progress logs."""

from romance_factory.core.ollama_client import (
    _chars_to_tokens_heuristic,
    estimate_chat_messages_prompt_tokens,
    estimate_completions_prompt_tokens,
    estimate_ollama_prompt_tokens,
    estimate_payload_prompt_tokens,
)


def test_chars_heuristic():
    assert _chars_to_tokens_heuristic("") == 0
    assert _chars_to_tokens_heuristic("abcd") == 1
    assert _chars_to_tokens_heuristic("a" * 8) == 2


def test_chat_messages_empty():
    total, method, roles = estimate_chat_messages_prompt_tokens([])
    assert total == 0
    assert method == "empty"
    assert roles == {}


def test_chat_system_user():
    total, method, roles = estimate_chat_messages_prompt_tokens(
        [
            {"role": "system", "content": "x" * 40},
            {"role": "user", "content": "y" * 40},
        ]
    )
    assert roles["system"] == 10
    assert roles["user"] == 10
    # Total uses tiktoken when available (can be lower than the chars/4 role
    # totals) plus a small per-message overhead.
    assert total >= roles["system"] + roles["user"]
    assert "chars/4" in method or "tiktoken" in method


def test_payload_dispatch_completions():
    t, method, roles = estimate_payload_prompt_tokens({"prompt": "word " * 20})
    assert t > 0
    assert roles == {}
    assert "chars/4" in method or "tiktoken" in method


def test_ollama_estimate():
    t, method = estimate_ollama_prompt_tokens("sys", "user " * 10)
    assert t > 0
    assert "chars/4" in method or "tiktoken" in method
