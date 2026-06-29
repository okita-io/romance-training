#!/usr/bin/env python3
"""
Convert a manifest-backed HF parquet dataset into processed JSONL for Phase 2.

Reads field mapping from source-data/manifests/<slug>.json. Supports BookRix
url+text corpora (Korshuk), full-book chunking, and row-level pre-chunked data.

Usage:
    python tools/data_preparation/download_hf_dataset.py molbal/horror-novel-chunks
    python tools/data_preparation/convert_hf_parquet.py --dataset horror_novel_chunks

    python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/fiction-books
    python tools/data_preparation/convert_hf_parquet.py --dataset fiction_books_korshuk --chunk

    python tools/data_preparation/download_hf_dataset.py ppirli/Gutenberg-Fiction
    python tools/data_preparation/convert_hf_parquet.py --dataset gutenberg_fiction --chunk
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.bookrix_metadata import parse_bookrix_metadata
from tools.data_preparation.convert_hf_sources import iter_parquet, list_data_files, load_manifest
from tools.data_preparation.gutenberg_corpus import (
    clean_gutenberg_prose,
    detect_gutenberg_play,
    iter_book_chunks,
    title_slug,
)
from tools.data_preparation.language_filter import classify_language, detect_language
from tools.data_preparation.prose_filter import filter_records
from tools.data_preparation.unified_corpus import normalize_record, repo_dir_name

MANIFESTS = ROOT / "source-data" / "manifests"
HF_ROOT = ROOT / "source-data" / "hf"
PROCESSED_ROOT = ROOT / "source-data" / "processed"


def author_slug(author: str | None) -> str:
    if not author:
        return "unknown_author"
    slug = re.sub(r"[^a-z0-9]+", "_", author.lower()).strip("_")
    return slug or "unknown_author"


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_manifest(dataset: str) -> tuple[str, dict[str, Any]]:
    """Resolve --dataset slug or repo tail to (slug, manifest dict)."""
    direct = MANIFESTS / f"{dataset}.json"
    if direct.exists():
        manifest = load_manifest(dataset)
        assert manifest is not None
        return dataset, manifest

    for path in sorted(MANIFESTS.glob("*.json")):
        manifest = load_manifest(path.stem)
        if manifest and (
            manifest.get("slug") == dataset
            or manifest.get("repo_id", "").split("/")[-1] == dataset
        ):
            return path.stem, manifest

    raise SystemExit(
        f"No manifest for {dataset!r}. Add source-data/manifests/{dataset}.json"
    )


def parquet_paths(manifest: dict[str, Any]) -> list[Path]:
    repo_id = manifest["repo_id"]
    dataset_dir = HF_ROOT / repo_dir_name(repo_id)
    if not dataset_dir.exists():
        raise SystemExit(
            f"Dataset not downloaded: {dataset_dir.relative_to(ROOT)}\n"
            f"Run: python tools/data_preparation/download_hf_dataset.py {repo_id}"
        )
    files = list_data_files(dataset_dir, manifest)
    if not files:
        raise SystemExit(f"No parquet files under {dataset_dir.relative_to(ROOT)}")
    return files


def row_to_book(
    row: dict[str, Any],
    *,
    manifest: dict[str, Any],
    record_index: int,
) -> dict[str, Any] | None:
    text_field = manifest["text_field"]
    text = row.get(text_field, "")
    if not isinstance(text, str) or not text.strip():
        return None

    if manifest.get("clean_prose"):
        text = clean_gutenberg_prose(text, strip_toc=manifest.get("strip_toc", True))
        if not text.strip():
            return None

    title_field = manifest.get("title_field")
    author_field = manifest.get("author_field")
    source_file_field = manifest.get("source_file_field")

    title = row.get(title_field) if title_field else None
    author = row.get(author_field) if author_field else None
    source_file = row.get(source_file_field) if source_file_field else None

    url = row.get("url", "")
    if manifest.get("bookrix_metadata") and isinstance(url, str):
        parsed_author, parsed_title, source_url = parse_bookrix_metadata(url, text)
        author = author or parsed_author
        title = title or parsed_title
        url = source_url

    story_id = record_index
    slug = author_slug(author if isinstance(author, str) else None)
    title_str = title if isinstance(title, str) else f"record_{story_id}"

    extra: dict[str, Any] = {}
    for key in manifest.get("extra_fields") or []:
        if key in row:
            extra[key] = row[key]
    if url:
        extra["url"] = url
    if source_file:
        extra["source_file"] = source_file

    return {
        "text": text,
        "author": author if isinstance(author, str) else None,
        "title": title_str,
        "story_id": story_id,
        "story_key": f"{slug}:{story_id}",
        "author_slug": slug,
        "title_slug": title_slug(title_str),
        "record_index": record_index,
        "extra": extra,
    }


def filter_english_books(
    books: list[dict[str, Any]],
    *,
    english_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not english_only:
        return books, []

    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for book in books:
        text = book.get("text", "")
        lang_class = classify_language(text)
        if lang_class != "en":
            skipped.append({
                "story_id": book["story_id"],
                "title": book.get("title"),
                "author": book.get("author"),
                "language": detect_language(text) or lang_class,
                "word_count": len(text.split()),
                "reason": "non_english" if lang_class == "non_en" else "unknown_language",
            })
            continue
        kept.append(book)
    return kept, skipped


def filter_plays(
    books: list[dict[str, Any]],
    *,
    exclude_plays: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not exclude_plays:
        return books, []

    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for book in books:
        reason = detect_gutenberg_play(book.get("title"), book.get("text", ""))
        if reason:
            skipped.append({
                "story_id": book["story_id"],
                "title": book.get("title"),
                "author": book.get("author"),
                "word_count": len(book.get("text", "").split()),
                "reason": "play",
                "play_signal": reason,
            })
            continue
        kept.append(book)
    return kept, skipped


def book_to_story(
    book: dict[str, Any],
    *,
    manifest: dict[str, Any],
    source_file: str,
) -> dict[str, Any] | None:
    slug = manifest["slug"]
    repo_id = manifest["repo_id"]
    genres = manifest.get("genres") or []
    min_words = int(manifest.get("min_words", 30))
    row_source = book.get("extra", {}).get("source_file")
    file_label = row_source if isinstance(row_source, str) and row_source else source_file

    record = normalize_record(
        book["text"],
        source_dataset=repo_id,
        source_slug=slug,
        genres=genres,
        author=book.get("author"),
        title=book.get("title"),
        source_file=file_label,
        record_index=book["record_index"],
        extra={
            "story_id": book["story_id"],
            "story_key": book["story_key"],
            "author_slug": book["author_slug"],
            "title_slug": book["title_slug"],
            **book.get("extra", {}),
        },
        min_words=min_words,
    )
    if record is None:
        return None
    record["metadata"]["story_id"] = book["story_id"]
    record["metadata"]["story_key"] = book["story_key"]
    record["metadata"]["author_slug"] = book["author_slug"]
    return record


def books_to_chunks(
    books: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    source_file: str,
    chunk_words: int,
) -> list[dict[str, Any]]:
    slug = manifest["slug"]
    repo_id = manifest["repo_id"]
    genres = manifest.get("genres") or []
    chunk_books = [{"text": b["text"], "title": b["title"], "author": b["author"]} for b in books]
    chunks: list[dict[str, Any]] = []

    for chunk in iter_book_chunks(chunk_books, target_words=chunk_words):
        meta = chunk["metadata"]
        source_book = books[meta["story_id"]]
        record = normalize_record(
            chunk["text"],
            source_dataset=repo_id,
            source_slug=slug,
            genres=genres,
            author=meta.get("author"),
            title=meta.get("title"),
            source_file=source_file,
            record_index=meta["chunk_index"],
            extra={
                **meta,
                "story_key": source_book["story_key"],
                "author_slug": source_book["author_slug"],
                **source_book.get("extra", {}),
                "chunk_boundary": "sentence",
            },
            min_words=30,
        )
        if record:
            record["metadata"].update({
                "story_id": source_book["story_id"],
                "story_key": source_book["story_key"],
                "author_slug": source_book["author_slug"],
            })
            chunks.append(record)
    return chunks


def convert_dataset(
    manifest: dict[str, Any],
    *,
    chunk: bool = False,
    chunk_words: int = 500,
    english_only: bool = True,
    exclude_plays: bool = False,
    limit: int | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    paths = parquet_paths(manifest)
    books: list[dict[str, Any]] = []
    record_index = 0

    for path in paths:
        for row in iter_parquet(path):
            book = row_to_book(row, manifest=manifest, record_index=record_index)
            record_index += 1
            if book:
                book["_source_parquet"] = path.name
                books.append(book)

    if limit:
        books = books[:limit]

    books, skipped_lang = filter_english_books(books, english_only=english_only)
    books, skipped_plays = filter_plays(books, exclude_plays=exclude_plays)

    stories: list[dict[str, Any]] = []
    for book in books:
        story = book_to_story(book, manifest=manifest, source_file=book["_source_parquet"])
        if story:
            stories.append(story)

    if chunk:
        by_file: dict[str, list[dict[str, Any]]] = {}
        for book in books:
            by_file.setdefault(book["_source_parquet"], []).append(book)
        chunks: list[dict[str, Any]] = []
        for source_file, file_books in by_file.items():
            chunks.extend(
                books_to_chunks(
                    file_books,
                    manifest=manifest,
                    source_file=source_file,
                    chunk_words=chunk_words,
                )
            )
    else:
        chunks = stories

    return books, skipped_lang, skipped_plays, stories, chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Manifest slug (e.g. horror_novel_chunks, fiction_books_korshuk)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--chunk", action="store_true", help="Sentence-aware ~500-word chunks")
    parser.add_argument("--chunk-words", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-all-languages",
        action="store_true",
        help="Keep non-English rows (default: English only)",
    )
    parser.add_argument(
        "--include-plays",
        action="store_true",
        help="Keep plays/dramas (default: exclude when manifest exclude_plays is true)",
    )
    parser.add_argument(
        "--skip-prose-filter",
        action="store_true",
        help="Do not drop publisher-catalog / TOC / transcriber chunks at convert time",
    )
    args = parser.parse_args()

    slug, manifest = find_manifest(args.dataset)
    output_dir = args.output or (PROCESSED_ROOT / manifest["slug"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    exclude_plays = manifest.get("exclude_plays", False) and not args.include_plays

    books, skipped_lang, skipped_plays, stories, chunks = convert_dataset(
        manifest,
        chunk=args.chunk,
        chunk_words=args.chunk_words,
        english_only=not args.include_all_languages,
        exclude_plays=exclude_plays,
        limit=args.limit,
    )

    skipped_prose: dict[str, int] = {}
    if not args.skip_prose_filter:
        chunks, skipped_prose = filter_records(chunks)

    write_jsonl(stories, output_dir / "stories.jsonl")
    write_jsonl(chunks, output_dir / "chunks.jsonl")
    print(f"Wrote {len(stories)} stories -> {(output_dir / 'stories.jsonl').relative_to(ROOT)}")
    print(f"Wrote {len(chunks)} chunks -> {(output_dir / 'chunks.jsonl').relative_to(ROOT)}")
    if skipped_prose:
        total_dropped = sum(skipped_prose.values())
        print(f"Prose filter dropped {total_dropped} non-narrative chunks at convert time")
    if skipped_lang:
        write_jsonl(skipped_lang, output_dir / "skipped_non_english.jsonl")
        print(f"Skipped {len(skipped_lang)} non-English / unknown-language rows")
    if skipped_plays:
        write_jsonl(skipped_plays, output_dir / "skipped_plays.jsonl")
        print(f"Skipped {len(skipped_plays)} plays / dramas")

    index = {
        "dataset": manifest["repo_id"],
        "slug": manifest["slug"],
        "english_only": not args.include_all_languages,
        "exclude_plays": exclude_plays,
        "chunked": args.chunk,
        "story_count": len(stories),
        "chunk_count": len(chunks),
        "skipped_non_english": len(skipped_lang),
        "skipped_plays": len(skipped_plays),
        "prose_filter_dropped": sum(skipped_prose.values()),
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
