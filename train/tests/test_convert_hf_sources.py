"""Tests for HF source → unified corpus conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.data_preparation.unified_corpus import (
    normalize_record,
    pick_text_field,
    slug_from_repo_id,
    word_count,
)


def test_slug_from_repo_id() -> None:
    assert slug_from_repo_id("TristanBehrens/lovecraftcorpus") == "lovecraftcorpus"


def test_pick_text_field_prefers_common_names() -> None:
    row = {"meta": {"source": "x.txt"}, "text": "word " * 40}
    assert pick_text_field(row) == "text"


def test_normalize_record_filters_short_text() -> None:
    assert (
        normalize_record(
            "too short",
            source_dataset="author/set",
            source_slug="set",
            min_words=30,
        )
        is None
    )


def test_normalize_record_shape() -> None:
    text = " ".join(["word"] * 40)
    record = normalize_record(
        text,
        source_dataset="TristanBehrens/lovecraftcorpus",
        source_slug="lovecraftcorpus",
        genres=["horror"],
        author="H.P. Lovecraft",
        source_file="unnamable.txt",
        record_index=1,
    )
    assert record is not None
    assert record["text"] == text
    assert record["metadata"]["source"] == "hf:lovecraftcorpus"
    assert record["metadata"]["word_count"] == word_count(text)


def test_convert_lovecraft_sample(tmp_path: Path) -> None:
    from tools.data_preparation.convert_hf_sources import convert_dataset

    dataset_dir = tmp_path / "TristanBehrens__lovecraftcorpus"
    dataset_dir.mkdir()
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest_dir.joinpath("lovecraftcorpus.json").write_text(
        json.dumps(
            {
                "repo_id": "TristanBehrens/lovecraftcorpus",
                "slug": "lovecraftcorpus",
                "author": "H.P. Lovecraft",
                "genres": ["horror"],
                "text_field": "text",
                "metadata_field": "meta",
                "field_mapping": {"source_file": "source"},
                "min_words": 30,
            }
        ),
        encoding="utf-8",
    )

    sample = {
        "text": " ".join(["word"] * 40),
        "meta": {"source": "sample.txt"},
    }
    dataset_dir.joinpath("train.jsonl").write_text(
        json.dumps(sample) + "\n",
        encoding="utf-8",
    )

    import tools.data_preparation.convert_hf_sources as mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod, "MANIFESTS", manifest_dir)
    try:
        records = convert_dataset(dataset_dir)
    finally:
        monkeypatch.undo()

    assert len(records) == 1
    assert records[0]["metadata"]["author"] == "H.P. Lovecraft"
    assert records[0]["metadata"]["source_file"] == "sample.txt"
