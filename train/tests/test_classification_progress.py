"""Tests for manual Phase 2 progress reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.incremental.classification_progress import (
    default_manual_output,
    format_progress_report,
    measure_classification_progress,
    resolve_output_path,
)
from tools.style_classification.pass_config import ALL_LLM_FIELDS
from tools.style_classification.run_pipeline import _chunk_record


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _sample_record(chunk_index: int = 0) -> dict:
    return {
        "text": f"Chunk {chunk_index} prose with enough words to classify. " * 4,
        "metadata": {"source": "test", "chunk_index": chunk_index},
    }


def _enriched(record: dict) -> dict:
    meta = dict(record.get("metadata") or {})
    meta["style_profile"] = {field: "value" for field in ALL_LLM_FIELDS}
    return {"text": record["text"], "metadata": meta}


def test_default_manual_output_path() -> None:
    path = default_manual_output("literotica_stories", 0)
    assert path.name == "literotica_stories_deep_seg_000.jsonl"
    assert path.parent.name == "romance_corpus"


def test_measure_progress_no_output(tmp_path: Path) -> None:
    input_path = tmp_path / "seg_000.jsonl"
    _write_jsonl(
        input_path,
        [{"text": "One short sentence.", "source": "test"}],
    )
    progress = measure_classification_progress(
        input_path,
        None,
        pass_mode="both",
        corpus="test_corpus",
        segment_index=0,
    )
    assert progress.source_rows == 1
    assert progress.pipeline_chunks >= 1
    assert progress.complete == 0
    assert progress.pending == progress.pipeline_chunks
    assert progress.percent_complete == 0.0


def test_measure_progress_with_partial_output(tmp_path: Path) -> None:
    input_path = tmp_path / "seg_000.jsonl"
    output_path = tmp_path / "out.jsonl"
    records = [_sample_record(0), _sample_record(1)]
    _write_jsonl(input_path, records)

    progress_empty = measure_classification_progress(
        input_path, output_path, pass_mode="both"
    )
    assert progress_empty.complete == 0
    assert progress_empty.pending == progress_empty.pipeline_chunks

    chunks = [c for record in records for c in _chunk_record(record)]
    _write_jsonl(output_path, [_enriched(chunks[0])])

    progress_partial = measure_classification_progress(
        input_path, output_path, pass_mode="both"
    )
    assert progress_partial.complete == 1
    assert progress_partial.pending == progress_partial.pipeline_chunks - 1
    assert progress_partial.output_unique == 1


def test_format_progress_report_includes_pending(tmp_path: Path) -> None:
    input_path = tmp_path / "seg_000.jsonl"
    _write_jsonl(input_path, [{"text": "Hello world.", "source": "x"}])
    progress = measure_classification_progress(
        input_path,
        tmp_path / "missing.jsonl",
        corpus="demo",
        segment_index=3,
    )
    report = format_progress_report(progress)
    assert "demo - seg_003" in report
    assert "pending:" in report


def test_resolve_output_prefers_romance_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.incremental.classification_progress as mod

    corpus_root = tmp_path / "train" / "romance_corpus"
    corpus_root.mkdir(parents=True)
    styled = corpus_root / "foo_deep_seg_001.jsonl"
    styled.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    resolved = resolve_output_path("foo", 1, None)
    assert resolved == styled
