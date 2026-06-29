#!/usr/bin/env python3
"""
Scan web-fiction JSONL (e.g. literotica_stories) for style-training quality.

Estimates how many story rows are worth classifying before committing to a
multi-million-line Phase 2 run. Can filter the corpus to keep-tier rows only.

Usage:
    # Fast estimate on 2k random stories
    python tools/data_preparation/scan_fiction_quality.py \\
        --input source-data/processed/literotica_stories/chunks_clean.jsonl \\
        --sample 2000

    # Filter full corpus — keep tier only, replace chunks.jsonl in dataset dir
    python tools/data_preparation/scan_fiction_quality.py \\
        --input source-data/processed/literotica_stories/chunks_clean.jsonl \\
        --in-place --keep-tier keep --fast-language
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, TextIO

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.fiction_quality import FictionQuality, classify_fiction_quality

DEFAULT_INPUT = ROOT / "source-data" / "processed" / "literotica_stories" / "chunks.jsonl"
DEFAULT_REPORT = ROOT / "source-data" / "processed" / "literotica_stories" / "quality_report.json"
DEFAULT_EXPANSION = 11.55


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def reservoir_sample(path: Path, n: int, *, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for i, record in enumerate(iter_jsonl(path)):
        if i < n:
            sample.append(record)
        else:
            j = rng.randint(0, i)
            if j < n:
                sample[j] = record
    return sample


def annotate(record: dict[str, Any], quality: FictionQuality) -> dict[str, Any]:
    out = dict(record)
    meta = dict(out.get("metadata") or {})
    meta["fiction_quality"] = {
        "tier": quality.tier,
        "score": quality.score,
        "reasons": list(quality.reasons),
        "word_count": quality.word_count,
        "unique_word_ratio": quality.unique_word_ratio,
        "top_bigram_ratio": quality.top_bigram_ratio,
        "html_char_ratio": quality.html_char_ratio,
        "language": quality.language,
    }
    out["metadata"] = meta
    return out


def tier_emits(quality: FictionQuality, keep_tier: str | None) -> bool:
    if keep_tier is None:
        return True
    if keep_tier == "keep":
        return quality.tier == "keep"
    if keep_tier == "review":
        return quality.tier != "drop"
    if keep_tier == "keep+review":
        return quality.tier != "drop"
    return True


def update_index_json(
    dir_path: Path,
    *,
    chunk_count: int,
    fiction_quality_skipped: int,
    fiction_quality_review: int,
    fiction_quality_drop: int,
) -> None:
    index_path = dir_path / "index.json"
    if not index_path.is_file():
        return
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    index["chunk_count"] = chunk_count
    index["fiction_quality_skipped"] = fiction_quality_skipped
    index["fiction_quality_review"] = fiction_quality_review
    index["fiction_quality_drop"] = fiction_quality_drop
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def process_records(
    records: Iterator[dict[str, Any]],
    *,
    keep_tier: str | None,
    fast_language: bool,
    out_fh: TextIO | None,
    skipped_fh: TextIO | None,
    flag_only: bool,
    progress_every: int,
) -> tuple[Counter[str], Counter[str], int, int]:
    tiers: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    total = 0
    kept_out = 0
    t0 = time.time()

    for record in records:
        total += 1
        quality = classify_fiction_quality(record.get("text") or "", fast_language=fast_language)
        tiers[quality.tier] += 1
        for reason in quality.reasons:
            reasons[reason] += 1

        annotated = annotate(record, quality)
        if tier_emits(quality, keep_tier):
            if out_fh is not None:
                payload = annotated if flag_only else record
                out_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                kept_out += 1
        elif skipped_fh is not None:
            skipped_fh.write(json.dumps(annotated, ensure_ascii=False) + "\n")

        if progress_every and total % progress_every == 0:
            elapsed = time.time() - t0
            print(
                f"  ... {total:,} scanned ({total / max(elapsed, 0.01):,.0f}/s) "
                f"| keep {tiers['keep']:,} review {tiers['review']:,} drop {tiers['drop']:,}"
                + (f" | wrote {kept_out:,}" if out_fh else ""),
                flush=True,
            )

    return tiers, reasons, total, kept_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan web-fiction corpus for training quality.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None, help="Write kept rows here")
    parser.add_argument("--skipped-output", type=Path, default=None, help="Write review+drop rows")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample", type=int, default=None, metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--flag-only", action="store_true", help="Attach fiction_quality metadata to output")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write keep-tier rows to <dataset>/chunks.jsonl (requires --keep-tier)",
    )
    parser.add_argument(
        "--keep-tier",
        choices=("keep", "review", "keep+review"),
        default=None,
        help="Emit only rows at or above this tier when writing output",
    )
    parser.add_argument(
        "--fast-language",
        action="store_true",
        default=True,
        help="Skip langdetect when English heuristics are strong (default: on)",
    )
    parser.add_argument("--no-fast-language", action="store_true")
    parser.add_argument("--expansion-ratio", type=float, default=DEFAULT_EXPANSION)
    parser.add_argument("--progress-every", type=int, default=25_000)
    args = parser.parse_args()

    if args.no_fast_language:
        args.fast_language = False

    args.input = args.input.resolve()
    if args.output:
        args.output = args.output.resolve()
    if args.skipped_output:
        args.skipped_output = args.skipped_output.resolve()
    args.report = args.report.resolve()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    dataset_dir = args.input.parent
    if args.in_place:
        if not args.keep_tier:
            raise SystemExit("--in-place requires --keep-tier")
        args.output = dataset_dir / "chunks.jsonl.tmp"
        if args.skipped_output is None:
            args.skipped_output = dataset_dir / "chunks_quality_skipped.jsonl"

    if args.sample:
        records: Iterator[dict[str, Any]] = iter(reservoir_sample(args.input, args.sample, seed=args.seed))
        mode = f"sample_{args.sample}"
    else:
        records = iter_jsonl(args.input)
        mode = "full"

    print(f"Scanning {args.input.relative_to(ROOT)} …", flush=True)
    if args.keep_tier:
        print(f"  keep tier filter: {args.keep_tier}", flush=True)

    tiers: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    total = 0
    kept_out = 0

    if not args.report_only and (args.output or args.in_place):
        out_fh = args.output.open("w", encoding="utf-8") if args.output else None
        skipped_fh = (
            args.skipped_output.open("w", encoding="utf-8") if args.skipped_output else None
        )
        try:
            tiers, reasons, total, kept_out = process_records(
                records,
                keep_tier=args.keep_tier,
                fast_language=args.fast_language,
                out_fh=out_fh,
                skipped_fh=skipped_fh,
                flag_only=args.flag_only,
                progress_every=args.progress_every,
            )
        finally:
            if out_fh:
                out_fh.close()
            if skipped_fh:
                skipped_fh.close()

        if args.in_place and args.output and args.output.is_file():
            final_path = dataset_dir / "chunks.jsonl"
            try:
                args.output.replace(final_path)
                print(f"  replaced {final_path.relative_to(ROOT)} ({kept_out:,} rows)")
            except OSError as exc:
                alt = dataset_dir / "chunks_quality_kept.jsonl"
                args.output.replace(alt)
                print(
                    f"  could not replace locked file ({exc}); "
                    f"wrote {alt.relative_to(ROOT)} instead"
                )
            update_index_json(
                dataset_dir,
                chunk_count=kept_out,
                fiction_quality_skipped=tiers.get("review", 0) + tiers.get("drop", 0),
                fiction_quality_review=tiers.get("review", 0),
                fiction_quality_drop=tiers.get("drop", 0),
            )
        if args.skipped_output and args.skipped_output.is_file():
            skipped_n = tiers.get("review", 0) + tiers.get("drop", 0)
            print(f"  skipped -> {args.skipped_output.relative_to(ROOT)} ({skipped_n:,} rows)")
    else:
        tiers, reasons, total, _ = process_records(
            records,
            keep_tier=None,
            fast_language=args.fast_language,
            out_fh=None,
            skipped_fh=None,
            flag_only=False,
            progress_every=args.progress_every,
        )

    def project(count: int) -> int:
        return int(count * args.expansion_ratio)

    keep = tiers.get("keep", 0)
    review = tiers.get("review", 0)
    drop = tiers.get("drop", 0)

    print(f"\n=== quality summary ({mode}, n={total:,}) ===")
    for tier in ("keep", "review", "drop"):
        n = tiers.get(tier, 0)
        pct = 100 * n / max(total, 1)
        print(f"  {tier:6s}: {n:>8,} ({pct:5.1f}%)  ~{project(n):>10,} pipeline chunks")
    print(f"  keep+review: {keep + review:,} ({100*(keep+review)/max(total,1):.1f}%)")
    if reasons:
        print("\nTop reasons:")
        for reason, count in reasons.most_common(8):
            print(f"  {reason}: {count:,}")

    report = {
        "input": str(args.input.relative_to(ROOT)).replace("\\", "/"),
        "mode": mode,
        "keep_tier_filter": args.keep_tier,
        "total": total,
        "kept_output": kept_out if args.output and not args.report_only else None,
        "tiers": dict(tiers),
        "tier_pct": {k: round(100 * v / max(total, 1), 2) for k, v in tiers.items()},
        "reasons": dict(reasons.most_common(20)),
        "expansion_ratio": args.expansion_ratio,
        "projected_pipeline_chunks": {
            "keep": project(keep),
            "review": project(review),
            "drop": project(drop),
            "keep_plus_review": project(keep + review),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport -> {args.report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
