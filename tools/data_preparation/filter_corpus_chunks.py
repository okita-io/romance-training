#!/usr/bin/env python3
"""
Scan or filter classified corpus JSONL — remove non-narrative Gutenberg chunks.

Detects publisher catalogs, transcriber credits, table-of-contents blocks,
illustration lists, and similar non-prose material embedded in chunked corpora.

Usage:
    # Report only
    python tools/data_preparation/filter_corpus_chunks.py \\
        --input train/romance_corpus/horror_styled.jsonl

    # Write cleaned corpus (drops non-prose chunks)
    python tools/data_preparation/filter_corpus_chunks.py \\
        --input train/romance_corpus/horror_styled.jsonl \\
        --output train/romance_corpus/horror_styled_clean.jsonl

    # Flag in metadata instead of dropping (writes all rows)
    python tools/data_preparation/filter_corpus_chunks.py \\
        --input train/romance_corpus/horror_styled.jsonl \\
        --output train/romance_corpus/horror_styled_flagged.jsonl \\
        --flag-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.prose_filter import ProseQuality, classify_chunk_prose
from tools.data_preparation.unified_corpus import normalize_prose_text


def annotate_record(record: dict[str, Any], quality: ProseQuality) -> dict[str, Any]:
    out = dict(record)
    metadata = dict(out.get("metadata") or {})
    metadata["prose_quality"] = {
        "keep": quality.verdict == "prose",
        "reason": quality.reason,
        "narrative_ratio": round(quality.narrative_ratio, 4),
        "narrative_word_ratio": round(quality.narrative_word_ratio, 4),
        "signals": list(quality.signals),
    }
    out["metadata"] = metadata
    return out


def process_jsonl(
    input_path: Path,
    output_path: Path | None,
    *,
    flag_only: bool,
    min_words: int,
    reflow_ocr: bool = True,
) -> tuple[int, int, Counter[str]]:
    kept = 0
    dropped = 0
    reasons: Counter[str] = Counter()

    out_fh = output_path.open("w", encoding="utf-8") if output_path else None
    try:
        with input_path.open(encoding="utf-8") as in_fh:
            for line in in_fh:
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
                    if flag_only and out_fh:
                        out_fh.write(json.dumps(annotate_record(record, quality), ensure_ascii=False) + "\n")
                    continue

                kept += 1
                if out_fh:
                    payload = annotate_record(record, quality) if flag_only else record
                    out_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        if out_fh:
            out_fh.close()

    return kept, dropped, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter non-prose chunks from corpus JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL (required unless --report-only)",
    )
    parser.add_argument(
        "--flag-only",
        action="store_true",
        help="Keep all rows; add metadata.prose_quality instead of dropping",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print stats only; do not write output",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=30,
        help="Minimum words for a chunk to count as prose (default: 30)",
    )
    parser.add_argument(
        "--show-dropped",
        type=int,
        default=8,
        metavar="N",
        help="Print N example dropped chunk indices (default: 8)",
    )
    parser.add_argument(
        "--no-reflow",
        action="store_true",
        help="Skip OCR line-wrap / drop-cap reflow before filtering",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    output_path: Path | None = None
    if not args.report_only:
        if args.output is None:
            raise SystemExit("--output is required unless --report-only")
        output_path = args.output if args.output.is_absolute() else ROOT / args.output

    kept, dropped, reasons = process_jsonl(
        input_path,
        output_path,
        flag_only=args.flag_only,
        min_words=args.min_words,
        reflow_ocr=not args.no_reflow,
    )
    total = kept + dropped

    print(f"Input:  {input_path}")
    if output_path:
        mode = "flagged" if args.flag_only else "filtered"
        print(f"Output: {output_path} ({mode})")
    print(f"Total:  {total:,}")
    print(f"Keep:   {kept:,} ({100 * kept / total:.1f}%)" if total else "Keep:   0")
    print(f"Drop:   {dropped:,} ({100 * dropped / total:.1f}%)" if total else "Drop:   0")
    if reasons:
        print("\nDrop reasons:")
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count:,}")

    if args.show_dropped and dropped:
        print(f"\nExample dropped chunks (first {args.show_dropped}):")
        shown = 0
        with input_path.open(encoding="utf-8") as fh:
            for index, line in enumerate(fh):
                record = json.loads(line)
                quality = classify_chunk_prose(record.get("text") or "", min_words=args.min_words)
                if quality.verdict != "non_prose":
                    continue
                preview = (record.get("text") or "").replace("\n", " ")[:120]
                print(f"  [{index}] {quality.reason}: {preview}…")
                shown += 1
                if shown >= args.show_dropped:
                    break


if __name__ == "__main__":
    main()
