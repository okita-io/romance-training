#!/usr/bin/env python3
"""Validate that train/romance_corpus contains only training-ready styled JSONL."""

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
from tools.style_classification.pass_config import ALL_LLM_FIELDS, pass_complete

ALLOWED_STYLED_RE = re.compile(
    r"^[a-z0-9_]+_styled(?:_seg_\d{3})?\.jsonl$",
    re.IGNORECASE,
)
# Legacy name from manual --output paths; rename to *_styled_seg_NNN.jsonl when convenient.
LEGACY_DEEP_SEG_RE = re.compile(
    r"^[a-z0-9_]+_deep_seg_\d{3}\.jsonl$",
    re.IGNORECASE,
)
FORBIDDEN_SUFFIXES = (".bak", ".tmp", ".json")
FORBIDDEN_NAME_PARTS = ("pipeline_chunks", "temp", "combined", "deduped", "reflowed", "clean", "flagged")


def _is_allowed_file(name: str) -> bool:
    if not name.endswith(".jsonl"):
        return False
    if any(part in name for part in FORBIDDEN_NAME_PARTS):
        return False
    return bool(ALLOWED_STYLED_RE.match(name) or LEGACY_DEEP_SEG_RE.match(name))


def validate_file(path: Path, *, strict: bool, sample: int) -> list[str]:
    issues: list[str] = []
    total = 0
    incomplete = 0
    missing_profile = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            if sample and total > sample:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                issues.append(f"{path.name}: line {total} invalid JSON")
                continue
            profile = (record.get("metadata") or {}).get("style_profile")
            if not isinstance(profile, dict) or not profile:
                missing_profile += 1
                continue
            if strict and not pass_complete(profile, "both"):
                incomplete += 1

    if total == 0:
        issues.append(f"{path.name}: empty file")
    if missing_profile:
        issues.append(f"{path.name}: {missing_profile}/{total} rows missing style_profile")
    if incomplete:
        issues.append(f"{path.name}: {incomplete}/{total} rows incomplete (--pass both fields)")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate train/romance_corpus contents.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require all LLM fields on every row (pass both complete)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Validate only first N rows per file (0 = all)",
    )
    args = parser.parse_args()

    if not IN_REPO_CORPUS.is_dir():
        raise SystemExit(f"Not found: {IN_REPO_CORPUS}")

    issues: list[str] = []
    styled_files: list[Path] = []

    for entry in sorted(IN_REPO_CORPUS.iterdir()):
        name = entry.name
        if entry.is_dir():
            issues.append(f"unexpected directory: {name}/")
            continue
        if name == "README.md":
            continue
        if name.endswith(FORBIDDEN_SUFFIXES) and not name.endswith(".jsonl"):
            issues.append(f"non-training file: {name}")
            continue
        if name.endswith(".jsonl"):
            if not _is_allowed_file(name):
                issues.append(f"disallowed JSONL name: {name} (expected *_styled.jsonl or *_styled_seg_NNN.jsonl)")
                continue
            styled_files.append(entry)
        else:
            issues.append(f"unexpected file: {name}")

    for path in styled_files:
        issues.extend(validate_file(path, strict=args.strict, sample=args.sample))

    if issues:
        print("romance_corpus validation FAILED:\n")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    print(f"romance_corpus OK ({len(styled_files)} styled file(s))")
    for path in styled_files:
        lines = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"  {path.name}: {lines:,} rows")
        if not args.strict:
            file_issues = validate_file(path, strict=True, sample=0)
            for issue in file_issues:
                if "incomplete" in issue:
                    print(f"    note: {issue} (use --strict to fail, or resume --pass deep)")


if __name__ == "__main__":
    main()
