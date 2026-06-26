"""Tests for romance parquet → story/chunk extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.data_preparation.split_romance_parquet import (
    REPO_ID,
    DATASET_SLUG,
    author_slug,
    build_index,
    chunk_stories,
    row_to_story_record,
    title_slug,
    write_by_author,
)


def test_author_slug() -> None:
    assert author_slug("Elliott, Emily") == "elliott_emily"
    assert author_slug("Cartland, Barbara") == "cartland_barbara"


def test_title_slug() -> None:
    assert title_slug("Tomorrow's Promise") == "tomorrow_s_promise"


def test_row_to_story_record() -> None:
    manifest = {
        "repo_id": "diltdicker/romance_books_32K",
        "slug": "romance_books_32k",
        "genres": ["romance"],
        "text_field": "description",
        "genre_tags_field": "genres",
        "min_words": 30,
        "extra_fields": ["id", "pub_month", "isbn13"],
    }
    row = {
        "id": 25041,
        "pub_month": "Jan-1985",
        "title": "Tomorrow's Promise",
        "author": "Elliott, Emily",
        "isbn13": 9780440187370.0,
        "description": " ".join(["word"] * 40),
        "genres": {"romance": 1, "contemporary-romance": 1, "category-romance": 1},
    }
    record = row_to_story_record(
        row,
        manifest=manifest,
        record_index=0,
        split="train",
        source_file="romance_data-v2-32K-train.parquet",
    )
    assert record is not None
    meta = record["metadata"]
    assert meta["author"] == "Elliott, Emily"
    assert meta["author_slug"] == "elliott_emily"
    assert meta["story_id"] == 25041
    assert meta["story_key"] == "elliott_emily:25041"
    assert meta["title"] == "Tomorrow's Promise"
    assert "romance" in meta["genres"]


def test_row_to_story_record_filters_short_text() -> None:
    manifest = {"text_field": "description", "min_words": 30, "slug": "romance_books_32k"}
    row = {"id": 1, "description": "too short", "author": "A, B", "genres": {}}
    assert (
        row_to_story_record(
            row,
            manifest=manifest,
            record_index=0,
            split="train",
            source_file="test.parquet",
        )
        is None
    )


def test_write_by_author(tmp_path: Path) -> None:
    stories = [
        {
            "text": "a " * 40,
            "metadata": {"author_slug": "elliott_emily", "story_id": 1},
        },
        {
            "text": "b " * 40,
            "metadata": {"author_slug": "elliott_emily", "story_id": 2},
        },
        {
            "text": "c " * 40,
            "metadata": {"author_slug": "cartland_barbara", "story_id": 3},
        },
    ]
    counts = write_by_author(stories, tmp_path)
    assert counts == {"cartland_barbara": 1, "elliott_emily": 2}
    elliott_path = tmp_path / "by_author" / "elliott_emily" / "stories.jsonl"
    assert elliott_path.exists()
    lines = elliott_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_chunk_stories_preserves_story_metadata() -> None:
    story = {
        "text": " ".join(["Sentence one here."] * 80),
        "metadata": {
            "source": "hf:romance_books_32k",
            "story_id": 99,
            "story_key": "author:99",
            "author": "Author, Name",
            "author_slug": "author_name",
            "title": "Sample",
            "split": "train",
        },
    }
    chunks = chunk_stories([story], target_words=100, overlap_sentences=1)
    assert len(chunks) >= 2
    assert all(c["metadata"]["story_id"] == 99 for c in chunks)
    assert all(c["metadata"]["author_slug"] == "author_name" for c in chunks)


def test_build_index() -> None:
    stories = [
        {"metadata": {"split": "train", "author_slug": "a"}},
        {"metadata": {"split": "train", "author_slug": "b"}},
        {"metadata": {"split": "test", "author_slug": "a"}},
    ]
    index = build_index(stories, {"a": 2, "b": 1}, english_only=True, skipped_non_english=3)
    assert index["story_count"] == 3
    assert index["author_count"] == 2
    assert index["english_only"] is True
    assert index["skipped_non_english"] == 3
    assert index["splits"] == {"test": 1, "train": 2}


def test_iter_story_records_skips_non_english(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.data_preparation import split_romance_parquet as mod

    manifest = {
        "repo_id": REPO_ID,
        "slug": DATASET_SLUG,
        "text_field": "description",
        "min_words": 30,
    }
    rows = [
        {
            "id": 1,
            "title": "English Book",
            "author": "Author, One",
            "description": " ".join(["She said that he would love her forever."] * 8),
            "genres": {"romance": 1},
        },
        {
            "id": 2,
            "title": "Spanish Book",
            "author": "Autor, Dos",
            "description": " ".join(
                ["Anna Harris No me olvides. Ella dijo que el amor era para siempre."] * 8
            ),
            "genres": {"romance": 1},
        },
    ]

    def fake_iter_parquet(_path: Path):
        yield from rows

    monkeypatch.setattr(mod, "iter_parquet", fake_iter_parquet)
    monkeypatch.setattr(mod, "parquet_path", lambda _split: Path("fake.parquet"))

    skipped: list[dict] = []
    stories = list(
        mod.iter_story_records(
            "train",
            manifest=manifest,
            english_only=True,
            skipped=skipped,
        )
    )
    assert len(stories) == 1
    assert stories[0]["metadata"]["story_id"] == 1
    assert len(skipped) == 1
    assert skipped[0]["id"] == 2
