"""Tests for graceful KeyboardInterrupt handling in run_pipeline."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tools.style_classification.pass_config import ALL_LLM_FIELDS
from tools.style_classification.run_pipeline import run


def _sample_record(chunk_index: int) -> dict:
    return {
        "text": f"Chunk {chunk_index} prose with enough words to classify. " * 4,
        "metadata": {"source": "test", "chunk_index": chunk_index},
    }


def _enriched(record: dict) -> dict:
    meta = dict(record.get("metadata") or {})
    meta["style_profile"] = {field: "value" for field in ALL_LLM_FIELDS}
    return {"text": record["text"], "metadata": meta}


@patch("tools.style_classification.run_pipeline._enrich")
def test_keyboard_interrupt_compacts_partial_output(mock_enrich, tmp_path) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(_sample_record(i)) for i in range(3)) + "\n",
        encoding="utf-8",
    )

    calls = 0

    def enrich_side_effect(record, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return _enriched(record)

    mock_enrich.side_effect = enrich_side_effect

    with pytest.raises(SystemExit) as exc_info:
        run(
            input_path=input_path,
            output_path=output_path,
            use_llm=True,
            workers=1,
            resume=False,
            pass_mode="full",
            quiet=True,
        )

    assert exc_info.value.code == 130
    lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["metadata"]["chunk_index"] == 0


@patch("tools.style_classification.run_pipeline._enrich")
def test_keyboard_interrupt_cancels_thread_pool(mock_enrich, tmp_path) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(_sample_record(i)) for i in range(4)) + "\n",
        encoding="utf-8",
    )

    calls = 0

    def enrich_side_effect(record, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return _enriched(record)

    mock_enrich.side_effect = enrich_side_effect

    with pytest.raises(SystemExit) as exc_info:
        run(
            input_path=input_path,
            output_path=output_path,
            use_llm=True,
            workers=4,
            resume=False,
            pass_mode="both",
            quiet=True,
        )

    assert exc_info.value.code == 130
    lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) < 4
    assert len(lines) == len(set(lines))
