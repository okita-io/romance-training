#!/usr/bin/env python3
"""
Clean train/romance_corpus for Phase 3 training.

- Remove disallowed artifacts (e.g. combined_styled.jsonl)
- Deduplicate segment files (last wins, same key as run_pipeline resume)
- Drop rows missing style_profile
- Rename legacy *_deep_seg_NNN.jsonl → *_styled_seg_NNN.jsonl

Usage:
    python tools/data_preparation/cleanup_romance_corpus.py --report-only
    python tools/data_preparation/cleanup_romance_corpus.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.paths import IN_REPO_CORPUS
from tools.style_classification.pass_config import pass_complete
from tools.style_classification.run_pipeline import _record_key, _rewrite_output

DEEP_SEG_RE = re.compile(r"^(.+)_deep_seg_(\d{3})\.jsonl$", re.IGNORECASE)
STYLED_SEG_RE = re.compile(r"^[a-z0-9_]+_styled_seg_\d{3}\.jsonl$", re.IGNORECASE)
REMOVE_NAMES = frozenset({"combined_styled.jsonl"})


def _has_style_profile(record: dict) -> bool:
    profile = (record.get("metadata") or {}).get("style_profile")
    return isinstance(profile, dict) and bool(profile)


def _is_training_ready(record: dict) -> bool:
    profile = (record.get("metadata") or {}).get("style_profile")
    if not isinstance(profile, dict) or not profile:
        return False
    return pass_complete(profile, "both")


def _load_deduped(records_iter, *, strict: bool) -> tuple[dict[str, dict], list[str], int, int, int]:
    by_key: dict[str, dict] = {}
    order: list[str] = []
    total = 0
    dropped_no_profile = 0
    dropped_incomplete = 0

    for record in records_iter:
        total += 1
        profile = (record.get("metadata") or {}).get("style_profile")
        if not isinstance(profile, dict) or not profile:
            dropped_no_profile += 1
            continue
        if strict and not pass_complete(profile, "both"):
            dropped_incomplete += 1
            continue
        key = _record_key(record)
        if key not in by_key:
            order.append(key)
        by_key[key] = record

    return by_key, order, total, dropped_no_profile, dropped_incomplete


def _iter_records(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clean_segment_file(path: Path, *, dry_run: bool, strict: bool) -> dict[str, int]:
    match = DEEP_SEG_RE.match(path.name)
    if not match:
        raise ValueError(f"not a legacy deep_seg file: {path.name}")

    corpus, seg = match.group(1), match.group(2)
    target = path.with_name(f"{corpus}_styled_seg_{seg}.jsonl")

    by_key, order, total, dropped_no_profile, dropped_incomplete = _load_deduped(
        _iter_records(path), strict=strict
    )
    unique = len(order)
    dropped_dupes = total - dropped_no_profile - dropped_incomplete - unique

    stats = {
        "lines_read": total,
        "unique_kept": unique,
        "dropped_duplicates": dropped_dupes,
        "dropped_no_profile": dropped_no_profile,
        "dropped_incomplete": dropped_incomplete,
    }

    if dry_run:
        return stats

    tmp = target.with_suffix(target.suffix + ".tmp")
    _rewrite_output(tmp, by_key, order)
    tmp.replace(target)
    if path != target and path.is_file():
        path.unlink()
    return stats


def clean_styled_file(path: Path, *, dry_run: bool, strict: bool) -> dict[str, int]:
    by_key, order, total, dropped_no_profile, dropped_incomplete = _load_deduped(
        _iter_records(path), strict=strict
    )
    unique = len(order)
    dropped_dupes = total - dropped_no_profile - dropped_incomplete - unique

    stats = {
        "lines_read": total,
        "unique_kept": unique,
        "dropped_duplicates": dropped_dupes,
        "dropped_no_profile": dropped_no_profile,
        "dropped_incomplete": dropped_incomplete,
    }

    if dry_run or (
        dropped_dupes == 0 and dropped_no_profile == 0 and dropped_incomplete == 0
    ):
        return stats

    tmp = path.with_suffix(path.suffix + ".tmp")
    _rewrite_output(tmp, by_key, order)
    tmp.replace(path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean train/romance_corpus segment JSONL files.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print planned changes without writing files",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Drop rows missing any pass-both LLM field (training-ready only)",
    )
    args = parser.parse_args()

    if not IN_REPO_CORPUS.is_dir():
        raise SystemExit(f"Not found: {IN_REPO_CORPUS}")

    for name in sorted(REMOVE_NAMES):
        path = IN_REPO_CORPUS / name
        if path.is_file():
            print(f"remove: {path.relative_to(ROOT)}")
            if not args.report_only:
                path.unlink()

    segment_files = sorted(
        p for p in IN_REPO_CORPUS.glob("*_deep_seg_*.jsonl") if p.is_file()
    )
    styled_files = sorted(
        p
        for p in IN_REPO_CORPUS.glob("*_styled_seg_*.jsonl")
        if p.is_file() and STYLED_SEG_RE.match(p.name)
    )

    if not segment_files and not styled_files:
        print("No segment JSONL files to clean.")
        return

    def _print_stats(path: Path, target: str, stats: dict[str, int]) -> None:
        print(
            f"{path.name} -> {target}: "
            f"{stats['lines_read']:,} read, "
            f"{stats['unique_kept']:,} kept, "
            f"{stats['dropped_duplicates']:,} dupes, "
            f"{stats['dropped_no_profile']:,} no profile, "
            f"{stats['dropped_incomplete']:,} incomplete"
        )

    for path in segment_files:
        match = DEEP_SEG_RE.match(path.name)
        assert match is not None
        target_name = f"{match.group(1)}_styled_seg_{match.group(2)}.jsonl"
        stats = clean_segment_file(path, dry_run=args.report_only, strict=args.strict)
        _print_stats(path, target_name, stats)

    for path in styled_files:
        stats = clean_styled_file(path, dry_run=args.report_only, strict=args.strict)
        _print_stats(path, path.name, stats)

    if args.report_only:
        print("(report only — no files changed)")
    else:
        print("romance_corpus cleanup complete.")


if __name__ == "__main__":
    main()
