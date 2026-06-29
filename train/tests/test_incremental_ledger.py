"""Tests for incremental segment splitting and ledger."""

import json
from pathlib import Path

from tools.incremental.ledger import Ledger, segment_id
from tools.incremental.segment_jsonl import segment_jsonl


def test_segment_jsonl_respects_max_bytes(tmp_path: Path) -> None:
    src = tmp_path / "source.jsonl"
    lines = [json.dumps({"text": "word " * 200, "metadata": {"i": i}}) for i in range(40)]
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_dir = tmp_path / "segments"
    max_bytes = 5000
    parts = segment_jsonl(src, out_dir, max_bytes=max_bytes)

    assert len(parts) >= 2
    for part in parts:
        assert part.path.is_file()
        assert part.bytes <= max_bytes or part.rows == 1
        assert part.rows >= 1


def test_ledger_segment_lifecycle(tmp_path: Path, monkeypatch) -> None:
    import tools.incremental.ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "LEDGER_PATH", tmp_path / "ledger.json")

    ledger = Ledger()
    ledger.register_input_segment(
        "horror_novel_chunks",
        0,
        tmp_path / "seg_000.jsonl",
        bytes=1000,
        rows=10,
    )
    seg = ledger.get_segment(segment_id("horror_novel_chunks", 0))
    assert seg is not None
    assert seg["classification_status"] == "pending"

    ledger.mark_classified(
        segment_id("horror_novel_chunks", 0),
        tmp_path / "styled_000.jsonl",
        pass_fast=True,
        pass_deep=False,
    )
    seg = ledger.get_segment(segment_id("horror_novel_chunks", 0))
    assert seg["classification_status"] == "classified"
    assert seg["training_status"] == "available"

    available = ledger.available_for_training("horror_novel_chunks")
    assert len(available) == 1

    ledger.allocate_segments("horror_novel_chunks", [segment_id("horror_novel_chunks", 0)], "batch_001")
    assert ledger.available_for_training("horror_novel_chunks") == []

    ledger.create_batch(
        "batch_001",
        max_mb_per_corpus=50,
        segments_by_corpus={"horror_novel_chunks": [segment_id("horror_novel_chunks", 0)]},
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
    )
    ledger.mark_batch_trained("batch_001", "run_001")
    seg = ledger.get_segment(segment_id("horror_novel_chunks", 0))
    assert seg["training_status"] == "trained"
