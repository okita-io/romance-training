"""Tests for multi-grain chunking and 50 MB segment packing."""

from __future__ import annotations

import json
from pathlib import Path

from tools.incremental.ledger import corpus_segments_dir, segment_id
from tools.incremental.segment_jsonl import segment_jsonl
from tools.style_classification.chunk_text import (
    chunk_by_sentences,
    chunk_record_multigrain,
    split_sentences,
)


def _long_prose(*, sentences: int = 80, words_per: int = 12) -> str:
    parts = []
    for i in range(sentences):
        body = " ".join(f"word{i}_{j}" for j in range(words_per))
        parts.append(f"{body.capitalize()}.")
    return " ".join(parts)


class TestChunkRecordMultigrain:
    def test_emits_all_grains_with_parent_links(self):
        text = _long_prose(sentences=100, words_per=15)
        record = {
            "text": text,
            "metadata": {"source": "demo_book", "chunk_index": 3},
        }
        grains = chunk_record_multigrain(record, span_words=300, act_words=1000)

        assert grains["sentence"]
        assert grains["span"]
        assert grains["act"]
        assert len(grains["sentence"]) == len(split_sentences(text))

        for sent in grains["sentence"]:
            m = sent["metadata"]
            assert m["grain"] == "sentence"
            assert m["sentence_id"]
            assert m["parent_span_id"]
            assert m["parent_act_id"]
            assert m["legacy_chunk_index"] == 3
            assert m["parent_legacy_chunk_id"] == "demo_book:legacy:0003"

        span_ids = {s["metadata"]["span_id"] for s in grains["span"]}
        act_ids = {a["metadata"]["act_id"] for a in grains["act"]}
        for span in grains["span"]:
            m = span["metadata"]
            assert m["grain"] == "span"
            assert m["parent_act_id"] in act_ids
            assert m["word_count"] >= 30 or len(grains["span"]) == 1

        for act in grains["act"]:
            assert act["metadata"]["grain"] == "act"
            assert act["metadata"]["act_id"] in act_ids

        for sent in grains["sentence"]:
            assert sent["metadata"]["parent_span_id"] in span_ids

    def test_sentence_boundaries_preserved(self):
        text = _long_prose(sentences=60, words_per=20)
        grains = chunk_record_multigrain(
            {"text": text, "source": "x"},
            span_words=200,
            act_words=600,
        )
        for grain in ("span", "act"):
            for row in grains[grain]:
                assert row["text"].endswith((".", "!", "?"))

    def test_short_text_single_windows(self):
        text = "One short sentence. Another short sentence."
        grains = chunk_record_multigrain(
            {"text": text, "metadata": {"source": "tiny"}},
            span_words=300,
            act_words=1000,
        )
        assert len(grains["sentence"]) == 2
        assert len(grains["span"]) == 1
        assert len(grains["act"]) == 1
        assert grains["span"][0]["metadata"]["parent_act_id"] == grains["act"][0]["metadata"]["act_id"]


class TestSegmentPacking:
    def test_many_small_rows_respect_max_bytes(self, tmp_path: Path):
        src = tmp_path / "small.jsonl"
        rows = [{"text": f"row {i}", "metadata": {"i": i}} for i in range(500)]
        with src.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        out = tmp_path / "segs"
        # Tiny budget forces multiple segments
        parts = segment_jsonl(src, out, max_bytes=2_000, prefix="seg")
        assert len(parts) > 1
        for info in parts:
            assert info.bytes <= 2_000 or info.rows == 1
            assert info.path.is_file()
            assert info.path.stat().st_size == info.bytes or info.bytes > 0

        total_rows = sum(info.rows for info in parts)
        assert total_rows == 500

    def test_grain_segment_paths(self):
        from tools.incremental.ledger import ROOT

        assert corpus_segments_dir("gutenberg_fiction", "input") == (
            ROOT / "train" / "incremental" / "segments" / "gutenberg_fiction" / "input"
        )
        assert corpus_segments_dir("gutenberg_fiction", "input", grain="span") == (
            ROOT / "train" / "incremental" / "segments" / "gutenberg_fiction" / "span" / "input"
        )
        assert segment_id("gutenberg_fiction", 0) == "gutenberg_fiction/seg_000"
        assert segment_id("gutenberg_fiction", 1, grain="span") == "gutenberg_fiction/span/seg_001"


class TestLegacyChunkerUnchanged:
    def test_chunk_by_sentences_still_works(self):
        text = _long_prose(sentences=40, words_per=20)
        chunks = chunk_by_sentences(text, target_words=150, overlap_sentences=2)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.endswith(".")
