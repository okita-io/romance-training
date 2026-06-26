#!/usr/bin/env python3
"""
Convert Dwaraka Gothic Gutenberg corpus into chunked JSONL for Phase 2.

The HF dataset ships as TRAINING_CORPUS.txt — 12 Gothic novels concatenated.
This script splits on Gutenberg markers, strips boilerplate, and writes
sentence-aware chunks ready for style classification.

Usage:
    python tools/data_preparation/download_hf_dataset.py \\
        Dwaraka/Training_Dataset_of_Project_Gutebberg_Gothic_Fiction
    python tools/data_preparation/convert_gutenberg_gothic.py --chunk
    python tools/data_preparation/convert_gutenberg_gothic.py --chunk --include-all-languages
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.gutenberg_corpus import (
    iter_book_chunks,
    split_gutenberg_corpus,
    title_slug,
)
from tools.data_preparation.language_filter import classify_language, detect_language
from tools.data_preparation.unified_corpus import normalize_record, slug_from_repo_id

REPO_ID = "Dwaraka/Training_Dataset_of_Project_Gutebberg_Gothic_Fiction"
DATASET_SLUG = "gutenberg_gothic_fiction"
HF_DIR = ROOT / "source-data" / "hf" / "Dwaraka__Training_Dataset_of_Project_Gutebberg_Gothic_Fiction"
CORPUS_FILE = HF_DIR / "TRAINING_CORPUS.txt"
DEFAULT_OUTPUT = ROOT / "source-data" / "processed" / "gutenberg_gothic_fiction"


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def filter_english_books(
    books: list[dict[str, str]],
    *,
    english_only: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if not english_only:
        return books, []

    kept: list[dict[str, str]] = []
    skipped: list[dict[str, Any]] = []
    for idx, book in enumerate(books):
        text = book.get("text", "")
        lang_class = classify_language(text)
        if lang_class != "en":
            skipped.append({
                "book_index": idx,
                "title": book.get("title"),
                "author": book.get("author"),
                "language": detect_language(text) or lang_class,
                "word_count": len(text.split()),
                "reason": "non_english" if lang_class == "non_en" else "unknown_language",
            })
            continue
        kept.append(book)
    return kept, skipped


def books_to_stories(books: list[dict[str, str]], *, source_file: str) -> list[dict[str, Any]]:
    stories: list[dict[str, Any]] = []
    for idx, book in enumerate(books):
        record = normalize_record(
            book["text"],
            source_dataset=REPO_ID,
            source_slug=DATASET_SLUG,
            genres=["gothic", "fiction", "project_gutenberg"],
            author=book.get("author"),
            title=book["title"],
            source_file=source_file,
            record_index=idx,
            extra={
                "story_id": idx,
                "story_key": f"{title_slug(book['title'])}:{idx}",
                "title_slug": title_slug(book["title"]),
            },
            min_words=100,
        )
        if record:
            meta = record["metadata"]
            meta["story_id"] = idx
            meta["story_key"] = f"{title_slug(book['title'])}:{idx}"
            stories.append(record)
    return stories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CORPUS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk", action="store_true", help="Write chunks.jsonl for Phase 2")
    parser.add_argument("--chunk-words", type=int, default=500)
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
            f"Corpus not found: {input_path.relative_to(ROOT)}\n"
            f"Download: python tools/data_preparation/download_hf_dataset.py {REPO_ID}"
        )

    text = input_path.read_text(encoding="utf-8", errors="replace")
    all_books = split_gutenberg_corpus(text)
    if not all_books:
        raise SystemExit("No books extracted from corpus.")

    books, skipped = filter_english_books(
        all_books,
        english_only=not args.include_all_languages,
    )
    if not books:
        raise SystemExit("No English books remaining after language filter.")

    stories = books_to_stories(books, source_file=input_path.name)
    write_jsonl(stories, output_dir / "stories.jsonl")
    print(f"Wrote {len(stories)} English books ({sum(len(b['text'].split()) for b in books):,} words)")
    if skipped:
        write_jsonl(skipped, output_dir / "skipped_non_english.jsonl")
        print(f"Skipped {len(skipped)} non-English / unknown-language books")

    chunk_count: int | None = None
    if args.chunk:
        chunks: list[dict[str, Any]] = []
        for chunk in iter_book_chunks(books, target_words=args.chunk_words):
            record = normalize_record(
                chunk["text"],
                source_dataset=REPO_ID,
                source_slug=DATASET_SLUG,
                genres=["gothic", "fiction", "project_gutenberg"],
                author=chunk["metadata"].get("author"),
                title=chunk["metadata"].get("title"),
                source_file=input_path.name,
                record_index=chunk["metadata"]["chunk_index"],
                extra={
                    **chunk["metadata"],
                    "chunk_boundary": "sentence",
                },
                min_words=30,
            )
            if record:
                record["metadata"].update(chunk["metadata"])
                chunks.append(record)
        write_jsonl(chunks, output_dir / "chunks.jsonl")
        chunk_count = len(chunks)
        print(f"Wrote {chunk_count} chunks → {(output_dir / 'chunks.jsonl').relative_to(ROOT)}")

    index = {
        "dataset": REPO_ID,
        "slug": DATASET_SLUG,
        "english_only": not args.include_all_languages,
        "book_count": len(stories),
        "skipped_non_english": len(skipped),
        "chunk_count": chunk_count,
        "titles": [b["title"] for b in books],
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
