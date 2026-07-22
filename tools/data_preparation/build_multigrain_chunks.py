#!/usr/bin/env python3
"""
Build multi-grain JSONL (sentence / span / act) from processed corpus chunks.

Does not overwrite legacy ~500w pipeline chunks. Output lands under
``train/staging/multigrain/<slug>/{sentence,span,act}.jsonl``.

Usage:
    python tools/data_preparation/build_multigrain_chunks.py --report-only
    python tools/data_preparation/build_multigrain_chunks.py --write --slug gutenberg_fiction
    python tools/data_preparation/build_multigrain_chunks.py --write --slug horror_novel_chunks \\
        --span-words 300 --act-words 1000
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

from tools.data_preparation.paths import MULTIGRAIN_DIR, SOURCE_DATA  # noqa: E402
from tools.style_classification.chunk_text import GRAINS, chunk_record_multigrain  # noqa: E402

PROCESSED_ROOT = SOURCE_DATA / "processed"
INVENTORY_PATH = PROCESSED_ROOT / "multigrain_inventory.json"


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


def grain_output_path(slug: str, grain: str) -> Path:
    return MULTIGRAIN_DIR / slug / f"{grain}.jsonl"


def process_corpus(
    slug: str,
    input_path: Path,
    *,
    write: bool,
    span_words: int,
    act_words: int,
    overlap_sentences: int,
    progress_every: int,
    write_max_rows: int | None,
) -> dict[str, Any]:
    from tools.data_preparation.unified_corpus import normalize_prose_text

    counts = {g: 0 for g in GRAINS}
    source_rows = 0
    t0 = time.time()

    out_fhs: dict[str, Any] = {}
    if write:
        out_dir = MULTIGRAIN_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        for grain in GRAINS:
            out_fhs[grain] = grain_output_path(slug, grain).open("w", encoding="utf-8")

    try:
        for record in iter_jsonl(input_path):
            source_rows += 1
            text = record.get("text") or ""
            if text:
                record = {**record, "text": normalize_prose_text(text)}

            grains = chunk_record_multigrain(
                record,
                span_words=span_words,
                act_words=act_words,
                overlap_sentences=overlap_sentences,
            )
            for grain in GRAINS:
                rows = grains[grain]
                counts[grain] += len(rows)
                fh = out_fhs.get(grain)
                if fh is not None:
                    for row in rows:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            if write_max_rows is not None and counts["span"] >= write_max_rows:
                break

            if progress_every and source_rows % progress_every == 0:
                elapsed = time.time() - t0
                rate = source_rows / max(elapsed, 0.01)
                print(
                    f"  {slug}: {source_rows:,} source | "
                    f"sent={counts['sentence']:,} span={counts['span']:,} act={counts['act']:,} "
                    f"| {rate:,.0f} source rows/s",
                    flush=True,
                )
    finally:
        for fh in out_fhs.values():
            fh.close()

    elapsed = time.time() - t0
    result: dict[str, Any] = {
        "slug": slug,
        "source_chunks": source_rows,
        "grain_counts": counts,
        "expansion": {
            g: round(counts[g] / max(source_rows, 1), 3) for g in GRAINS
        },
        "span_words": span_words,
        "act_words": act_words,
        "overlap_sentences": overlap_sentences,
        "source_path": str(input_path.relative_to(ROOT)).replace("\\", "/"),
        "outputs": {
            g: str(grain_output_path(slug, g).relative_to(ROOT)).replace("\\", "/")
            for g in GRAINS
        },
        "elapsed_sec": round(elapsed, 1),
        "written": write,
    }
    return result


def print_summary(entries: list[dict[str, Any]]) -> None:
    if not entries:
        print("No processed chunk files found.")
        return
    print(
        f"\n{'slug':<24} {'source':>10} {'sentence':>10} {'span':>10} {'act':>10}"
    )
    print("-" * 70)
    totals = {"source": 0, "sentence": 0, "span": 0, "act": 0}
    for entry in entries:
        c = entry["grain_counts"]
        totals["source"] += entry["source_chunks"]
        for g in GRAINS:
            totals[g] += c[g]
        print(
            f"{entry['slug']:<24} "
            f"{entry['source_chunks']:>10,} "
            f"{c['sentence']:>10,} "
            f"{c['span']:>10,} "
            f"{c['act']:>10,}"
        )
    print("-" * 70)
    print(
        f"{'TOTAL':<24} {totals['source']:>10,} "
        f"{totals['sentence']:>10,} {totals['span']:>10,} {totals['act']:>10,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sentence/span/act multigrain JSONL from processed chunks.",
    )
    parser.add_argument("--slug", action="append", default=None)
    parser.add_argument("--write", action="store_true", help="Write staging multigrain JSONL")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--span-words", type=int, default=300)
    parser.add_argument("--act-words", type=int, default=1000)
    parser.add_argument("--overlap-sentences", type=int, default=2)
    parser.add_argument("--write-max-rows", type=int, default=None, metavar="N")
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=INVENTORY_PATH,
        help=f"Inventory JSON (default: {INVENTORY_PATH})",
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
        do_write = bool(args.write and (not args.slug or slug in args.slug))
        entry = process_corpus(
            slug,
            path,
            write=do_write,
            span_words=args.span_words,
            act_words=args.act_words,
            overlap_sentences=args.overlap_sentences,
            progress_every=args.progress_every,
            write_max_rows=args.write_max_rows,
        )
        entries.append(entry)
        if do_write:
            print(
                f"  wrote sent={entry['grain_counts']['sentence']:,} "
                f"span={entry['grain_counts']['span']:,} "
                f"act={entry['grain_counts']['act']:,} -> {MULTIGRAIN_DIR / slug}"
            )

    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "span_words": args.span_words,
        "act_words": args.act_words,
        "corpora": entries,
    }
    args.inventory.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nInventory -> {args.inventory.relative_to(ROOT)}")
    print_summary(entries)


if __name__ == "__main__":
    main()
