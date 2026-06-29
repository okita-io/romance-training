#!/usr/bin/env python3
"""
Deduplicate styled corpus JSONL using the same key as run_pipeline.py resume.

Keeps the last occurrence of each record (matches Phase 2 resume semantics).
Use after interrupted runs or when append-mode left duplicate lines.

Usage:
    python tools/data_preparation/dedup_corpus_jsonl.py \\
        --input train/romance_corpus/horror_styled.jsonl \\
        --report-only

    python tools/data_preparation/dedup_corpus_jsonl.py \\
        --input train/romance_corpus/horror_styled.jsonl \\
        --output train/romance_corpus/horror_styled_deduped.jsonl

    python tools/data_preparation/dedup_corpus_jsonl.py \\
        --input train/romance_corpus/horror_styled.jsonl --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.style_classification.run_pipeline import _record_key, _rewrite_output


def load_index(path: Path) -> tuple[dict[str, dict], list[str], int]:
  by_key: dict[str, dict] = {}
  order: list[str] = []
  total = 0
  with path.open(encoding="utf-8") as fh:
    for line in fh:
      line = line.strip()
      if not line:
        continue
      total += 1
      try:
        record = json.loads(line)
      except json.JSONDecodeError:
        continue
      key = _record_key(record)
      if key not in by_key:
        order.append(key)
      by_key[key] = record
  return by_key, order, total


def main() -> None:
  parser = argparse.ArgumentParser(description="Deduplicate styled JSONL (last wins per record key).")
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--in-place", action="store_true")
  parser.add_argument("--report-only", action="store_true")
  args = parser.parse_args()

  if not args.input.is_file():
    raise SystemExit(f"Not found: {args.input}")

  by_key, order, total = load_index(args.input)
  unique = len(order)
  dropped = total - unique

  print(f"Lines read:   {total:,}")
  print(f"Unique keys:  {unique:,}")
  print(f"Duplicates:   {dropped:,}")

  if dropped and args.report_only:
    # Find which keys were duplicated
    key_counts: Counter[str] = Counter()
    with args.input.open(encoding="utf-8") as fh:
      for line in fh:
        if line.strip():
          try:
            key_counts[_record_key(json.loads(line))] += 1
          except json.JSONDecodeError:
            pass
    for key, count in key_counts.most_common():
      if count > 1:
        print(f"  {count}x {key[:90]}...")

  if args.report_only or dropped == 0:
    return

  out = args.output
  if args.in_place:
    out = args.input.with_suffix(args.input.suffix + ".tmp")
  if out is None:
    out = args.input.with_name(args.input.stem + "_deduped.jsonl")

  _rewrite_output(out, by_key, order)
  if args.in_place:
    out.replace(args.input)
    print(f"Replaced {args.input.relative_to(ROOT)}")
  else:
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
  main()
