#!/usr/bin/env python3
"""
Batch prose cleanup for Phase 2 inputs and styled outputs.

Removes publisher catalogs, transcriber blocks, TOC indexes, and similar
non-narrative chunks from:
  - source-data/processed/*/chunks.jsonl
  - train/romance_corpus/*_styled.jsonl (optional)

Run this before starting Phase 2 on new collections, and on nearly-finished
styled corpora so resume does not re-classify junk chunks.

Usage:
    # Preview all processed chunk files
    python tools/data_preparation/clean_processed_corpora.py --report-only

    # Clean processed inputs in place (keeps .jsonl.bak backups)
    python tools/data_preparation/clean_processed_corpora.py --processed --in-place

    # Clean horror styled output + its source chunks
    python tools/data_preparation/clean_processed_corpora.py \\
        --styled train/romance_corpus/horror_styled.jsonl \\
        --slug horror_novel_chunks \\
        --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.prose_filter import classify_chunk_prose
from tools.data_preparation.unified_corpus import normalize_prose_text

PROCESSED_ROOT = ROOT / "source-data" / "processed"
STYLED_DIR = ROOT / "train" / "romance_corpus"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def filter_jsonl_file(
    input_path: Path,
    output_path: Path,
    *,
    min_words: int,
    progress_every: int,
    skipped_path: Path | None = None,
    reflow_ocr: bool = True,
) -> tuple[int, int, Counter[str]]:
    kept = 0
    dropped = 0
    reasons: Counter[str] = Counter()
    skipped_fh = skipped_path.open("w", encoding="utf-8") if skipped_path else None
    t0 = time.time()

    try:
        with input_path.open(encoding="utf-8") as in_fh, output_path.open("w", encoding="utf-8") as out_fh:
            for line_no, line in enumerate(in_fh, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text") or ""
                if reflow_ocr:
                    text = normalize_prose_text(text, reflow_ocr=True, strip_front_matter=True)
                    record = dict(record)
                    record["text"] = text
                    metadata = dict(record.get("metadata") or {})
                    metadata["word_count"] = len(text.split())
                    metadata["text_reflowed"] = True
                    record["metadata"] = metadata
                quality = classify_chunk_prose(text, min_words=min_words)

                if quality.verdict == "non_prose":
                    dropped += 1
                    if quality.reason:
                        reasons[quality.reason] += 1
                    if skipped_fh:
                        meta = dict(record.get("metadata") or {})
                        meta["prose_filter"] = {
                            "line": line_no - 1,
                            "reason": quality.reason,
                            "narrative_word_ratio": round(quality.narrative_word_ratio, 4),
                        }
                        record["metadata"] = meta
                        skipped_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                kept += 1
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                if progress_every and kept % progress_every == 0:
                    elapsed = time.time() - t0
                    rate = (kept + dropped) / max(elapsed, 0.01)
                    print(
                        f"  ... {kept + dropped:,} scanned ({rate:,.0f}/s), "
                        f"{kept:,} kept, {dropped:,} dropped",
                        flush=True,
                    )
    finally:
        if skipped_fh:
            skipped_fh.close()

    return kept, dropped, reasons


def update_index_json(dir_path: Path, *, chunk_count: int, dropped: int) -> None:
    index_path = dir_path / "index.json"
    if not index_path.is_file():
        return
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    index["chunk_count"] = chunk_count
    index["prose_filter_dropped"] = dropped
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def clean_one(
    path: Path,
    *,
    in_place: bool,
    backup: bool,
    min_words: int,
    progress_every: int,
    write_skipped: bool,
    reflow_ocr: bool = True,
) -> tuple[int, int, Counter[str]] | None:
    if not path.is_file():
        print(f"skip (missing): {path.relative_to(ROOT)}")
        return None

    output_path = path
    temp_path = path.with_suffix(path.suffix + ".tmp")
    if in_place:
        output_path = temp_path
    else:
        print(f"skip (use --in-place to write): {path.relative_to(ROOT)}")
        return None

    skipped_path = None
    if write_skipped:
        skipped_path = path.with_name(path.stem + "_skipped_non_prose.jsonl")

    print(f"clean: {path.relative_to(ROOT)}")
    kept, dropped, reasons = filter_jsonl_file(
        path,
        output_path,
        min_words=min_words,
        progress_every=progress_every,
        skipped_path=skipped_path,
        reflow_ocr=reflow_ocr,
    )

    if in_place:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(path, backup_path)
                print(f"  backup -> {backup_path.relative_to(ROOT)}")
        try:
            temp_path.replace(path)
        except OSError as exc:
            alt = path.with_name(path.stem + "_clean" + path.suffix)
            temp_path.replace(alt)
            print(
                f"  could not replace locked file ({exc}); "
                f"wrote {alt.relative_to(ROOT)} instead — swap when pipeline is stopped"
            )
            return kept, dropped, reasons

    if path.parent.name and (path.name == "chunks.jsonl"):
        update_index_json(path.parent, chunk_count=kept, dropped=dropped)

    total = kept + dropped
    print(
        f"  done: {total:,} total -> {kept:,} kept, {dropped:,} dropped "
        f"({100 * dropped / total:.2f}%)" if total else "  done: empty"
    )
    if reasons:
        for reason, count in reasons.most_common(5):
            print(f"    {reason}: {count:,}")
    if skipped_path and skipped_path.exists():
        print(f"  skipped log -> {skipped_path.relative_to(ROOT)}")
    return kept, dropped, reasons


def discover_processed_chunk_files(*, slugs: list[str] | None) -> list[Path]:
    if not PROCESSED_ROOT.is_dir():
        return []
    paths: list[Path] = []
    for child in sorted(PROCESSED_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if slugs and child.name not in slugs:
            continue
        chunk_path = child / "chunks.jsonl"
        if chunk_path.is_file():
            paths.append(chunk_path)
    return paths


def discover_styled_files(explicit: list[Path] | None) -> list[Path]:
    if explicit:
        return explicit
    if not STYLED_DIR.is_dir():
        return []
    return sorted(STYLED_DIR.glob("*_styled.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch non-prose cleanup for corpus JSONL files.")
    parser.add_argument(
        "--processed",
        action="store_true",
        help="Clean source-data/processed/*/chunks.jsonl",
    )
    parser.add_argument(
        "--styled",
        nargs="*",
        type=Path,
        default=None,
        metavar="PATH",
        help="Clean styled JSONL (default: train/romance_corpus/*_styled.jsonl when flag set with no paths)",
    )
    parser.add_argument(
        "--slug",
        action="append",
        default=None,
        help="Limit --processed to one or more dataset slugs (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Skip slug(s) under --processed (e.g. gutenberg_fiction for a separate run)",
    )
    parser.add_argument("--in-place", action="store_true", help="Replace files in place")
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Keep .jsonl.bak before replacing (default: on)",
    )
    parser.add_argument("--no-backup", action="store_true", help="Do not write .jsonl.bak backups")
    parser.add_argument("--report-only", action="store_true", help="Scan and print stats only")
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="Log progress every N records (default: 50000)",
    )
    parser.add_argument(
        "--write-skipped",
        action="store_true",
        help="Write dropped rows to *_skipped_non_prose.jsonl",
    )
    parser.add_argument(
        "--no-reflow",
        action="store_true",
        help="Skip OCR line-wrap / drop-cap reflow",
    )
    args = parser.parse_args()

    if not args.processed and args.styled is None and not args.report_only:
        args.processed = True
        args.styled = []

    targets: list[Path] = []
    if args.processed:
        slugs = args.slug
        paths = discover_processed_chunk_files(slugs=slugs)
        excludes = set(args.exclude or [])
        targets.extend(p for p in paths if p.parent.name not in excludes)

    if args.styled is not None:
        styled_paths = [
            p if p.is_absolute() else ROOT / p for p in args.styled
        ] if args.styled else discover_styled_files(None)
        targets.extend(styled_paths)

    if not targets:
        raise SystemExit("No JSONL targets found. Use --processed and/or --styled.")

    totals_kept = 0
    totals_dropped = 0
    all_reasons: Counter[str] = Counter()

    for path in targets:
        if args.report_only:
            kept, dropped, reasons = filter_jsonl_file(
                path,
                path.with_suffix(".report.tmp"),
                min_words=args.min_words,
                progress_every=args.progress_every,
                skipped_path=None,
                reflow_ocr=not args.no_reflow,
            )
            path.with_suffix(".report.tmp").unlink(missing_ok=True)
        else:
            if not args.in_place:
                raise SystemExit("--in-place is required to write cleaned files.")
            result = clean_one(
                path,
                in_place=True,
                backup=not args.no_backup,
                min_words=args.min_words,
                progress_every=args.progress_every,
                write_skipped=args.write_skipped,
                reflow_ocr=not args.no_reflow,
            )
            if result is None:
                continue
            kept, dropped, reasons = result

        totals_kept += kept
        totals_dropped += dropped
        all_reasons.update(reasons)
        if args.report_only:
            total = kept + dropped
            print(
                f"{path.relative_to(ROOT)}: {total:,} total, "
                f"{dropped:,} would drop ({100 * dropped / total:.2f}%)" if total else f"{path}: empty"
            )

    print("\n=== summary ===")
    grand = totals_kept + totals_dropped
    print(f"Files:  {len(targets)}")
    print(f"Total:  {grand:,}")
    print(f"Keep:   {totals_kept:,}")
    print(f"Drop:   {totals_dropped:,} ({100 * totals_dropped / grand:.2f}%)" if grand else "Drop:   0")
    if all_reasons:
        print("Reasons:")
        for reason, count in all_reasons.most_common():
            print(f"  {reason}: {count:,}")


if __name__ == "__main__":
    main()
