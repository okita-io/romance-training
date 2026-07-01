#!/usr/bin/env python3
"""
Expand processed corpus chunks to Phase 2 pipeline size (~500-word sentence chunks).

Phase 2 (`run_pipeline.py`) re-chunks every input row before classification. Source
`chunks.jsonl` row counts are often much smaller than the styled output (e.g. horror
5.5k -> ~16k). This script materializes that expansion and writes an inventory with
time estimates.

Usage:
    # Count pipeline rows for all processed corpora (no files written)
    python tools/data_preparation/build_pipeline_chunks.py --report-only

    # Write expanded JSONL for one collection
    python tools/data_preparation/build_pipeline_chunks.py --write --slug horror_novel_chunks

    # Full inventory JSON for planning
    python tools/data_preparation/build_pipeline_chunks.py --report-only \\
        --reference-lines 16410 --reference-days 2.75
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED_ROOT = ROOT / "source-data" / "processed"
INVENTORY_PATH = PROCESSED_ROOT / "pipeline_inventory.json"

from tools.data_preparation.paths import IN_REPO_CORPUS, PIPELINE_CHUNKS_DIR  # noqa: E402

# Legacy styled output names (run_pipeline --output)
STYLED_OUTPUT_ALIASES: dict[str, str] = {
    "horror_novel_chunks": "horror_styled.jsonl",
}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def discover_chunk_files(*, slugs: list[str] | None) -> list[tuple[str, Path]]:
    if not PROCESSED_ROOT.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for child in sorted(PROCESSED_ROOT.iterdir()):
        if not child.is_dir():
            continue
        slug = child.name
        if slugs and slug not in slugs:
            continue
        chunk_path = child / "chunks.jsonl"
        if chunk_path.is_file():
            out.append((slug, chunk_path))
    return out


def load_index(slug_dir: Path) -> dict[str, Any]:
    index_path = slug_dir / "index.json"
    if not index_path.is_file():
        return {}
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def pipeline_output_path(slug: str) -> Path:
    return PIPELINE_CHUNKS_DIR / f"{slug}_pipeline_chunks.jsonl"


def styled_output_path(slug: str) -> Path:
    name = STYLED_OUTPUT_ALIASES.get(slug, f"{slug}_styled.jsonl")
    return IN_REPO_CORPUS / name


def expand_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    from tools.data_preparation.unified_corpus import normalize_prose_text
    from tools.style_classification.run_pipeline import _chunk_record

    text = record.get("text") or ""
    if text:
        record = {**record, "text": normalize_prose_text(text)}
    return _chunk_record(record)


def process_corpus(
    slug: str,
    input_path: Path,
    *,
    write: bool,
    write_max_rows: int | None,
    progress_every: int,
    fast_if_unchunked: bool = True,
) -> dict[str, Any]:
    index = load_index(input_path.parent)
    output_path = pipeline_output_path(slug)
    styled_path = styled_output_path(slug)

    use_fast = False
    if fast_if_unchunked and not write:
        sample_ratio = _sample_expansion_ratio(input_path, sample_size=80)
        use_fast = sample_ratio <= 1.05
        if use_fast:
            print(f"  fast count (~{sample_ratio:.2f}x sample expansion)")

    source_rows = 0
    pipeline_rows = 0
    split_source_rows = 0
    word_total = 0
    t0 = time.time()

    out_fh = None
    if write:
        if write_max_rows is not None and write_max_rows <= 0:
            write = False
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_fh = output_path.open("w", encoding="utf-8")

    try:
        for record in iter_jsonl(input_path):
            source_rows += 1
            if use_fast:
                pipeline_rows += 1
            else:
                word_total += len((record.get("text") or "").split())
                expanded = expand_record(record)
                if len(expanded) > 1:
                    split_source_rows += 1
                pipeline_rows += len(expanded)
                if out_fh is not None:
                    for chunk in expanded:
                        out_fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        if write_max_rows is not None and pipeline_rows >= write_max_rows:
                            out_fh.close()
                            out_fh = None
                            write = False
                            break

            if progress_every and source_rows % progress_every == 0:
                elapsed = time.time() - t0
                rate = source_rows / max(elapsed, 0.01)
                ratio = pipeline_rows / max(source_rows, 1)
                print(
                    f"  {slug}: {source_rows:,} source -> {pipeline_rows:,} pipeline "
                    f"({ratio:.2f}x) | {rate:,.0f} source rows/s",
                    flush=True,
                )
    finally:
        if out_fh is not None:
            out_fh.close()

    elapsed = time.time() - t0
    ratio = pipeline_rows / max(source_rows, 1)
    median_words = word_total // max(source_rows, 1) if word_total else None
    rechunk_needed = ratio > 1.05

    result: dict[str, Any] = {
        "slug": slug,
        "source_chunks": source_rows,
        "pipeline_chunks": pipeline_rows,
        "expansion_ratio": round(ratio, 3),
        "rechunk_needed": rechunk_needed,
        "split_source_rows": split_source_rows,
        "median_source_words": median_words,
        "fast_count": use_fast,
        "source_path": str(input_path.relative_to(ROOT)).replace("\\", "/"),
        "pipeline_chunks_path": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        "styled_output_path": str(styled_path.relative_to(ROOT)).replace("\\", "/"),
        "styled_lines_existing": _count_lines(styled_path) if styled_path.is_file() else 0,
        "convert_chunked": index.get("chunked"),
        "dataset": index.get("dataset"),
        "elapsed_sec": round(elapsed, 1),
    }
    if write and output_path.is_file():
        result["pipeline_chunks_written"] = _count_lines(output_path)
    elif not rechunk_needed:
        result["note"] = (
            "Source chunks are already ~500 words; Phase 2 expansion is ~1:1. "
            "Use chunks.jsonl directly as pipeline input."
        )
    return result


def _count_lines(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def add_estimates(
    entries: list[dict[str, Any]],
    *,
    reference_lines: int,
    reference_days: float,
) -> None:
    if reference_lines <= 0 or reference_days <= 0:
        return
    lines_per_day = reference_lines / reference_days
    for entry in entries:
        pipeline = entry.get("pipeline_chunks", 0)
        entry["eta_days_at_reference"] = round(pipeline / lines_per_day, 1)
        entry["reference_throughput"] = {
            "lines": reference_lines,
            "days": reference_days,
            "lines_per_day": round(lines_per_day),
        }


def print_summary(entries: list[dict[str, Any]]) -> None:
    if not entries:
        print("No processed chunk files found.")
        return

    print(f"\n{'slug':<24} {'source':>12} {'pipeline':>12} {'ratio':>7} {'eta days':>9}")
    print("-" * 70)
    total_source = 0
    total_pipeline = 0
    for entry in entries:
        total_source += entry["source_chunks"]
        total_pipeline += entry["pipeline_chunks"]
        eta = entry.get("eta_days_at_reference")
        eta_s = f"{eta:,.1f}" if eta is not None else "-"
        print(
            f"{entry['slug']:<24} "
            f"{entry['source_chunks']:>12,} "
            f"{entry['pipeline_chunks']:>12,} "
            f"{entry['expansion_ratio']:>6.2f}x "
            f"{eta_s:>9}"
        )
    print("-" * 70)
    print(
        f"{'TOTAL':<24} {total_source:>12,} {total_pipeline:>12,} "
        f"{total_pipeline / max(total_source, 1):>6.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand processed chunks to Phase 2 pipeline size and estimate run time.",
    )
    parser.add_argument(
        "--slug",
        action="append",
        default=None,
        help="Limit to one or more dataset slugs (repeatable)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write train/staging/pipeline_chunks/<slug>_pipeline_chunks.jsonl",
    )
    parser.add_argument(
        "--write-max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Stop writing after N pipeline rows (counting still runs to completion unless --limit-source)",
    )
    parser.add_argument(
        "--write-all",
        action="store_true",
        help="Write pipeline JSONL even when source chunks are already ~500 words",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Count and write inventory only (default when --write is not set)",
    )
    parser.add_argument(
        "--reference-lines",
        type=int,
        default=16_410,
        help="Reference styled line count for ETA (default: horror_styled.jsonl)",
    )
    parser.add_argument(
        "--reference-days",
        type=float,
        default=2.75,
        help="Days to classify --reference-lines (default: 2.75)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="Log progress every N source rows (default: 50000)",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=INVENTORY_PATH,
        help=f"Write inventory JSON (default: {INVENTORY_PATH.relative_to(ROOT)})",
    )
    args = parser.parse_args()

    if not args.write:
        args.report_only = True

    corpora = discover_chunk_files(slugs=args.slug)
    if not corpora:
        print(f"No chunks.jsonl under {PROCESSED_ROOT.relative_to(ROOT)}")
        sys.exit(1)

    print(f"Scanning {len(corpora)} corpus/corpora …")
    entries: list[dict[str, Any]] = []
    for slug, path in corpora:
        print(f"\n{slug} <- {path.relative_to(ROOT)}")
        index = load_index(path.parent)
        pre_count = index.get("chunk_count")
        if pre_count is not None:
            print(f"  index.json chunk_count: {pre_count:,}")

        do_write = args.write and (not args.slug or slug in args.slug)
        if do_write and not args.write_all:
            sample_ratio = _sample_expansion_ratio(path, sample_size=100)
            if sample_ratio <= 1.05:
                print(f"  ~1:1 expansion ({sample_ratio:.2f}x sample) — skipping write")
                do_write = False

        entry = process_corpus(
            slug,
            path,
            write=do_write,
            write_max_rows=args.write_max_rows,
            progress_every=args.progress_every,
        )
        entries.append(entry)
        if entry.get("note"):
            print(f"  {entry['note']}")
        if do_write and entry.get("pipeline_chunks_written"):
            print(f"  wrote {entry['pipeline_chunks_written']:,} -> {entry['pipeline_chunks_path']}")

    add_estimates(entries, reference_lines=args.reference_lines, reference_days=args.reference_days)

    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": {
            "lines": args.reference_lines,
            "days": args.reference_days,
            "description": "Horror fast-pass baseline unless overridden",
        },
        "corpora": entries,
        "totals": {
            "source_chunks": sum(e["source_chunks"] for e in entries),
            "pipeline_chunks": sum(e["pipeline_chunks"] for e in entries),
        },
    }
    args.inventory.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nInventory -> {args.inventory.relative_to(ROOT)}")
    print_summary(entries)


def _sample_expansion_ratio(path: Path, *, sample_size: int) -> float:
    import random

    reservoir: list[dict[str, Any]] = []
    for i, record in enumerate(iter_jsonl(path)):
        if i < sample_size:
            reservoir.append(record)
        else:
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = record
    if not reservoir:
        return 1.0
    source = len(reservoir)
    pipeline = sum(len(expand_record(r)) for r in reservoir)
    return pipeline / source


if __name__ == "__main__":
    main()
