"""Offline tests for the OpenRouter free-model cascade."""

from __future__ import annotations

import json

from tools.llm_client import LLMError
from tools.style_evaluation.openrouter_cascade import (
    cascade_ids,
    extract_consensus,
    is_free_model,
    load_council_passages,
    parameter_count_b,
    rank_free_text_models,
    should_fallback,
    summarize_results,
)


def _model(model_id: str, *, prompt: str = "0", completion: str = "0", ctx: int = 8192, **arch):
    return {
        "id": model_id,
        "name": model_id,
        "canonical_slug": model_id,
        "context_length": ctx,
        "pricing": {"prompt": prompt, "completion": completion},
        "architecture": {
            "input_modalities": arch.get("inputs", ["text"]),
            "output_modalities": arch.get("outputs", ["text"]),
        },
    }


def test_rank_free_models_largest_first() -> None:
    models = [
        _model("liquid/lfm-2.5-2.6b:free", ctx=65536),
        _model("nvidia/nemotron-3-ultra-550b-a55b:free", ctx=1000000),
        _model("google/gemma-4-31b-it:free", ctx=262144),
        _model("paid/huge-70b", prompt="0.001", completion="0.002"),
        _model("nvidia/nemotron-3.5-content-safety:free"),
        _model("google/lyria-3-pro-preview", ctx=1048576, outputs=["text", "audio"]),
    ]
    ranked = rank_free_text_models(models)
    ids = [m["id"] for m in ranked]
    assert ids[0] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert ids[1] == "google/gemma-4-31b-it:free"
    assert ids[2] == "liquid/lfm-2.5-2.6b:free"
    assert "paid/huge-70b" not in ids
    assert "content-safety" not in "".join(ids)
    assert "lyria" not in "".join(ids)


def test_parameter_count_moe_and_active() -> None:
    assert parameter_count_b(_model("nvidia/nemotron-3-ultra-550b-a55b:free")) == 550.0
    assert parameter_count_b(_model("google/gemma-4-26b-a4b-it:free")) == 26.0
    assert parameter_count_b(_model("liquid/lfm-2.5-2.6b:free")) == 2.6


def test_is_free_zero_or_suffix() -> None:
    assert is_free_model(_model("x:free"))
    assert not is_free_model(_model("x", prompt="0.1", completion="0"))


def test_extract_consensus_and_sample(tmp_path) -> None:
    rec = {
        "text": "Alice opened the door. " * 20,
        "metadata": {
            "title": "Alice",
            "style_council": {
                "tone": {"value": "comedic", "consensus": True, "agree_n": 3},
                "register": {"value": "colloquial", "consensus": False},
                "pov": {"value": "third_limited", "consensus": True, "agree_n": 2},
            },
        },
    }
    labels = extract_consensus(rec, ["tone", "register", "pov"])
    assert labels == {"tone": "comedic", "pov": "third_limited"}
    path = tmp_path / "_council_x_out.jsonl"
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    sample = load_council_passages([path], fields=["tone", "pov"], limit=5, seed=1, min_fields=2)
    assert len(sample) == 1
    assert sample[0]["gold"]["tone"] == "comedic"


def test_should_fallback_rate_limit() -> None:
    assert should_fallback(LLMError("HTTP 429 from https://openrouter.ai/api/v1: rate limit"))
    assert should_fallback(LLMError("HTTP 402 from https://openrouter.ai/api/v1: payment"))
    assert not should_fallback(LLMError("HTTP 401 from https://openrouter.ai/api/v1: bad key"))


def test_summarize_agree_rate() -> None:
    rows = [
        {"field": "tone", "gold": "comedic", "label": "comedic", "model": "a"},
        {"field": "tone", "gold": "comedic", "label": "melancholic", "model": "b"},
        {"field": "tone", "gold": "comedic", "label": None, "error": "HTTP 429", "model": "c"},
    ]
    report = summarize_results(rows)
    assert report["n_labelled"] == 2
    assert report["agree"] == 1
    assert report["agree_rate"] == 0.5


def test_cascade_ids_dedupes() -> None:
    models = [_model("google/gemma-4-31b-it:free"), _model("google/gemma-4-31b-it:free")]
    assert cascade_ids(models) == ["google/gemma-4-31b-it:free"]
