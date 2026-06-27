"""Tests for manifest-backed HF parquet conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.data_preparation.convert_hf_parquet import convert_dataset


def test_convert_horror_novel_chunks_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.data_preparation.convert_hf_parquet as mod

    manifest = {
        "repo_id": "molbal/horror-novel-chunks",
        "slug": "horror_novel_chunks",
        "genres": ["horror"],
        "files": "**/*.parquet",
        "text_field": "chunk",
        "source_file_field": "source",
        "min_words": 30,
    }
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest_dir.joinpath("horror_novel_chunks.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    hf_dir = tmp_path / "hf" / "molbal__horror-novel-chunks"
    hf_dir.mkdir(parents=True)
    parquet_path = hf_dir / "train.parquet"
    parquet_path.touch()

    rows = [
        {
            "chunk": " ".join(["word"] * 40),
            "source": "Dracula.txt",
        }
    ]

    def fake_iter_parquet(path: Path):
        assert path == parquet_path
        yield from rows

    monkeypatch.setattr(mod, "MANIFESTS", manifest_dir)
    monkeypatch.setattr(mod, "HF_ROOT", tmp_path / "hf")
    monkeypatch.setattr(mod, "iter_parquet", fake_iter_parquet)

    books, skipped_lang, skipped_plays, stories, chunks = convert_dataset(
        manifest, english_only=False
    )
    assert len(skipped_lang) == 0
    assert len(skipped_plays) == 0
    assert len(stories) == 1
    assert len(chunks) == 1
    assert stories[0]["metadata"]["source_file"] == "Dracula.txt"
    assert stories[0]["metadata"]["genres"] == ["horror"]
