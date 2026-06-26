#!/usr/bin/env python3
"""
Extract romance novel blurbs from HF parquet into story- and author-organized JSONL.

Each parquet row is one book (story). The `description` field is the prose passage
to classify in Phase 2. Output is ready for sentence-aware chunking and style markup.

Usage:
    python tools/data_preparation/split_romance_parquet.py
    python tools/data_preparation/split_romance_parquet.py --chunk --by-author
    python tools/data_preparation/split_romance_parquet.py --split test --limit 100
    python tools/data_preparation/split_romance_parquet.py --chunk --include-all-languages
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.convert_hf_sources import iter_parquet, load_manifest
from tools.data_preparation.language_filter import classify_language, detect_language
from tools.data_preparation.unified_corpus import (
    genres_from_tag_map,
    normalize_genres,
    normalize_record,
    slug_from_repo_id,
)

HF_DATASET_DIR = ROOT / "source-data" / "hf" / "diltdicker__romance_books_32K"
DEFAULT_OUTPUT = ROOT / "source-data" / "processed" / "romance_books_32k"
REPO_ID = "diltdicker/romance_books_32K"
DATASET_SLUG = slug_from_repo_id(REPO_ID)

PARQUET_BY_SPLIT = {
    "train": "train/romance_data-v2-32K-train.parquet",
    "test": "test/romance_data-v2-32K-test.parquet",
}


def author_slug(author: str) -> str:
    """Elliott, Emily -> elliott_emily."""
    author = author.strip()
    if not author:
        return "unknown_author"
    if "," in author:
        last, first = author.split(",", 1)
        parts = [last.strip(), first.strip()]
    else:
        parts = author.split()
    slug = "_".join(p.lower() for p in parts if p)
    slug = re.sub(r"[^a-z0-9_]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown_author"


def title_slug(title: str) -> str:
    slug = title.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:80] or "untitled"


def parquet_path(split: str) -> Path:
    rel = PARQUET_BY_SPLIT.get(split)
    if not rel:
        raise ValueError(f"Unknown split {split!r}; choose from {sorted(PARQUET_BY_SPLIT)}")
    path = HF_DATASET_DIR / rel
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet not found: {path.relative_to(ROOT)}. "
            f"Download with: python tools/data_preparation/download_hf_dataset.py {REPO_ID}"
        )
    return path


def row_to_story_record(
    row: dict[str, Any],
    *,
    manifest: dict[str, Any],
    record_index: int,
    split: str,
    source_file: str,
) -> dict[str, Any] | None:
    text = row.get(manifest.get("text_field", "description"), "")
    if not isinstance(text, str) or not text.strip():
        return None

    author = row.get("author")
    if isinstance(author, str):
        author = author.strip()
    else:
        author = None

    title = row.get("title")
    if isinstance(title, str):
        title = title.strip()
    else:
        title = None

    genres = normalize_genres(manifest.get("genres"))
    tag_field = manifest.get("genre_tags_field")
    if tag_field and tag_field in row:
        genres.extend(genres_from_tag_map(row[tag_field]))
    genres = sorted(set(genres))

    story_id = row.get("id")
    slug = author_slug(author or "")
    story_key = f"{slug}:{story_id}" if story_id is not None else f"{slug}:{record_index}"

    extra: dict[str, Any] = {}
    for key in manifest.get("extra_fields") or ("id", "pub_month", "isbn13"):
        if key in row and row[key] is not None:
            extra[key] = row[key]
    extra["split"] = split
    extra["story_id"] = story_id
    extra["story_key"] = story_key
    extra["author_slug"] = slug
    if title:
        extra["title_slug"] = title_slug(title)

    record = normalize_record(
        text,
        source_dataset=manifest.get("repo_id", REPO_ID),
        source_slug=manifest.get("slug", DATASET_SLUG),
        genres=genres,
        author=author,
        title=title,
        source_file=source_file,
        record_index=record_index,
        extra=extra,
        min_words=int(manifest.get("min_words", 30)),
    )
    if record is None:
        return None

    meta = record["metadata"]
    meta["story_id"] = story_id
    meta["story_key"] = story_key
    meta["author_slug"] = slug
    meta["split"] = split
    return record


def iter_story_records(
    split: str,
    *,
    manifest: dict[str, Any],
    limit: int | None = None,
    english_only: bool = True,
    skipped: list[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    path = parquet_path(split)
    source_file = path.name
    text_field = manifest.get("text_field", "description")
    for record_index, row in enumerate(iter_parquet(path)):
        if limit is not None and record_index >= limit:
            break
        text = row.get(text_field, "")
        if english_only and isinstance(text, str) and text.strip():
            lang_class = classify_language(text)
            if lang_class != "en":
                if skipped is not None:
                    skipped.append({
                        "record_index": record_index,
                        "split": split,
                        "id": row.get("id"),
                        "title": row.get("title"),
                        "author": row.get("author"),
                        "language": detect_language(text) or lang_class,
                        "word_count": len(text.split()),
                        "reason": "non_english" if lang_class == "non_en" else "unknown_language",
                    })
                continue
        record = row_to_story_record(
            row,
            manifest=manifest,
            record_index=record_index,
            split=split,
            source_file=source_file,
        )
        if record:
            yield record


def chunk_stories(
    stories: list[dict[str, Any]],
    *,
    target_words: int = 500,
    overlap_sentences: int = 2,
) -> list[dict[str, Any]]:
    from tools.style_classification.chunk_text import chunk_record

    chunks: list[dict[str, Any]] = []
    for story in stories:
        for piece in chunk_record(
            story,
            target_words=target_words,
            overlap_sentences=overlap_sentences,
        ):
            meta = dict(piece.get("metadata", {}))
            story_meta = story.get("metadata", {})
            meta.setdefault("story_id", story_meta.get("story_id"))
            meta.setdefault("story_key", story_meta.get("story_key"))
            meta.setdefault("author", story_meta.get("author"))
            meta.setdefault("author_slug", story_meta.get("author_slug"))
            meta.setdefault("title", story_meta.get("title"))
            meta.setdefault("split", story_meta.get("split"))
            piece["metadata"] = meta
            chunks.append(piece)
    return chunks


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_by_author(stories: list[dict[str, Any]], output_dir: Path) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for story in stories:
        slug = story.get("metadata", {}).get("author_slug", "unknown_author")
        grouped[slug].append(story)

    counts: dict[str, int] = {}
    author_root = output_dir / "by_author"
    for slug, records in sorted(grouped.items()):
        path = author_root / slug / "stories.jsonl"
        write_jsonl(records, path)
        counts[slug] = len(records)
    return counts


def build_index(
    stories: list[dict[str, Any]],
    author_counts: dict[str, int] | None,
    *,
    english_only: bool = True,
    skipped_non_english: int = 0,
) -> dict[str, Any]:
    splits: dict[str, int] = defaultdict(int)
    for story in stories:
        split = story.get("metadata", {}).get("split", "unknown")
        splits[split] += 1
    return {
        "dataset": REPO_ID,
        "slug": DATASET_SLUG,
        "english_only": english_only,
        "story_count": len(stories),
        "skipped_non_english": skipped_non_english,
        "author_count": len(author_counts or {}),
        "splits": dict(sorted(splits.items())),
        "authors": dict(sorted((author_counts or {}).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        choices=sorted(PARQUET_BY_SPLIT),
        dest="splits",
        help="Parquet split to convert (default: train)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--by-author",
        action="store_true",
        help="Also write source-data/processed/.../by_author/<slug>/stories.jsonl",
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Write chunks.jsonl with sentence-boundary chunks for Phase 2",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=500,
        help="Target words per chunk when --chunk is set (default: 500)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N rows per split (for testing)",
    )
    parser.add_argument(
        "--include-all-languages",
        action="store_true",
        help="Keep non-English blurbs (default: English only)",
    )
    args = parser.parse_args()

    splits = args.splits or ["train"]
    manifest = load_manifest(DATASET_SLUG) or {
        "repo_id": REPO_ID,
        "slug": DATASET_SLUG,
        "genres": ["romance"],
        "text_field": "description",
        "genre_tags_field": "genres",
        "min_words": 30,
        "extra_fields": ["id", "pub_month", "isbn13"],
    }

    stories: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    english_only = not args.include_all_languages
    for split in splits:
        path = parquet_path(split)
        print(f"Reading {path.relative_to(ROOT)} …")
        split_stories = list(
            iter_story_records(
                split,
                manifest=manifest,
                limit=args.limit,
                english_only=english_only,
                skipped=skipped,
            )
        )
        print(f"  → {len(split_stories)} English stories")
        stories.extend(split_stories)

    if not stories:
        raise SystemExit("No stories extracted — check parquet path and min_words filter.")

    output_dir = args.output
    stories_path = output_dir / "stories.jsonl"
    write_jsonl(stories, stories_path)
    print(f"Wrote {len(stories)} stories to {stories_path.relative_to(ROOT)}")
    if skipped:
        skipped_path = output_dir / "skipped_non_english.jsonl"
        write_jsonl(skipped, skipped_path)
        print(f"Skipped {len(skipped)} non-English / unknown-language blurbs")

    author_counts: dict[str, int] | None = None
    if args.by_author:
        author_counts = write_by_author(stories, output_dir)
        print(f"Wrote {len(author_counts)} author directories under {output_dir.relative_to(ROOT) / 'by_author'}")

    if args.chunk:
        chunks = chunk_stories(stories, target_words=args.chunk_words)
        chunks_path = output_dir / "chunks.jsonl"
        write_jsonl(chunks, chunks_path)
        print(f"Wrote {len(chunks)} chunks to {chunks_path.relative_to(ROOT)}")

    index = build_index(
        stories,
        author_counts,
        english_only=english_only,
        skipped_non_english=len(skipped),
    )
    index_path = output_dir / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Index: {index['story_count']} stories, {index['author_count']} authors")


if __name__ == "__main__":
    main()
