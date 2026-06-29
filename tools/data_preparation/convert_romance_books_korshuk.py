#!/usr/bin/env python3
"""
Convert AlekseyKorshuk/romance-books (BookRix full novels) into chunked JSONL.

Each parquet row is one full book (~3.2k median words). Output is ready for
Phase 2 style classification.

Usage:
    python tools/data_preparation/download_hf_dataset.py AlekseyKorshuk/romance-books
    python tools/data_preparation/convert_romance_books_korshuk.py --chunk
    python tools/data_preparation/convert_romance_books_korshuk.py --chunk --include-all-languages
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
from tools.data_preparation.convert_hf_sources import iter_parquet
from tools.data_preparation.gutenberg_corpus import iter_book_chunks, title_slug
from tools.data_preparation.language_filter import classify_language, detect_language
from tools.data_preparation.prose_filter import filter_records
from tools.data_preparation.unified_corpus import normalize_record

REPO_ID = "AlekseyKorshuk/romance-books"
DATASET_SLUG = "romance_books_korshuk"
HF_DIR = ROOT / "source-data" / "hf" / "AlekseyKorshuk__romance-books"
PARQUET = HF_DIR / "data" / "train-00000-of-00001.parquet"
DEFAULT_OUTPUT = ROOT / "source-data" / "processed" / "romance_books_korshuk"


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


def iter_books(
    path: Path,
    *,
    min_words: int = 100,
    english_only: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    books: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for record_index, row in enumerate(iter_parquet(path)):
        text = row.get("text", "")
        url = row.get("url", "")
        if not isinstance(text, str) or len(text.split()) < min_words:
            continue

        if english_only:
            lang_class = classify_language(text)
            if lang_class != "en":
                skipped.append({
                    "record_index": record_index,
                    "url": url,
                    "language": detect_language(text) or lang_class,
                    "word_count": len(text.split()),
                    "reason": "non_english" if lang_class == "non_en" else "unknown_language",
                })
                continue

        author, title, source_url = parse_bookrix_metadata(url, text)
        story_id = record_index
        slug = author_slug(author)
        books.append({
            "text": text,
            "author": author,
            "title": title,
            "url": source_url,
            "story_id": story_id,
            "story_key": f"{slug}:{story_id}",
            "author_slug": slug,
            "title_slug": title_slug(title or f"book_{story_id}"),
            "record_index": record_index,
        })

    return books, skipped


def book_to_story(book: dict[str, Any], *, source_file: str) -> dict[str, Any] | None:
    record = normalize_record(
        book["text"],
        source_dataset=REPO_ID,
        source_slug=DATASET_SLUG,
        genres=["romance", "bookrix"],
        author=book.get("author"),
        title=book.get("title"),
        source_file=source_file,
        record_index=book["record_index"],
        extra={
            "story_id": book["story_id"],
            "story_key": book["story_key"],
            "author_slug": book["author_slug"],
            "title_slug": book["title_slug"],
            "url": book.get("url"),
        },
        min_words=100,
    )
    if record is None:
        return None
    record["metadata"]["story_id"] = book["story_id"]
    record["metadata"]["story_key"] = book["story_key"]
    record["metadata"]["author_slug"] = book["author_slug"]
    record["metadata"]["url"] = book.get("url")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk", action="store_true")
    parser.add_argument("--chunk-words", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-all-languages",
        action="store_true",
        help="Keep non-English books (default: English only)",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else (ROOT / args.input)
    output_dir = args.output if args.output.is_absolute() else (ROOT / args.output)

    if not input_path.exists():
        raise SystemExit(
            f"Parquet not found: {input_path.relative_to(ROOT)}\n"
            f"Download: python tools/data_preparation/download_hf_dataset.py {REPO_ID}"
        )

    books, skipped = iter_books(input_path, english_only=not args.include_all_languages)
    if args.limit:
        books = books[: args.limit]

    stories = [book_to_story(b, source_file=input_path.name) for b in books]
    stories = [s for s in stories if s]
    write_jsonl(stories, output_dir / "stories.jsonl")
    print(f"Wrote {len(stories)} English books")
    if skipped:
        write_jsonl(skipped, output_dir / "skipped_non_english.jsonl")
        print(f"Skipped {len(skipped)} non-English / unknown-language books")

    chunk_count: int | None = None
    skipped_prose: dict[str, int] = {}
    if args.chunk:
        chunk_books = [{"text": b["text"], "title": b["title"], "author": b["author"]} for b in books]
        chunks: list[dict[str, Any]] = []
        for chunk in iter_book_chunks(chunk_books, target_words=args.chunk_words):
            meta = chunk["metadata"]
            source_book = books[meta["story_id"]]
            record = normalize_record(
                chunk["text"],
                source_dataset=REPO_ID,
                source_slug=DATASET_SLUG,
                genres=["romance", "bookrix"],
                author=meta.get("author"),
                title=meta.get("title"),
                source_file=input_path.name,
                record_index=meta["chunk_index"],
                extra={
                    **meta,
                    "story_key": source_book["story_key"],
                    "author_slug": source_book["author_slug"],
                    "url": source_book.get("url"),
                    "chunk_boundary": "sentence",
                },
                min_words=30,
            )
            if record:
                record["metadata"].update({
                    "story_id": source_book["story_id"],
                    "story_key": source_book["story_key"],
                    "author_slug": source_book["author_slug"],
                    "url": source_book.get("url"),
                })
                chunks.append(record)
        chunks, skipped_prose = filter_records(chunks)
        write_jsonl(chunks, output_dir / "chunks.jsonl")
        chunk_count = len(chunks)
        print(f"Wrote {chunk_count} chunks → {(output_dir / 'chunks.jsonl').relative_to(ROOT)}")
        if skipped_prose:
            print(f"Prose filter dropped {sum(skipped_prose.values())} non-narrative chunks")

    index = {
        "dataset": REPO_ID,
        "slug": DATASET_SLUG,
        "english_only": not args.include_all_languages,
        "story_count": len(stories),
        "skipped_non_english": len(skipped),
        "chunk_count": chunk_count,
        "prose_filter_dropped": sum(skipped_prose.values()) if args.chunk else 0,
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
