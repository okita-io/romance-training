#!/usr/bin/env python3
"""
Reflow OCR hard wraps in corpus JSONL (in place or to a new file).

Usage:
    python tools/data_preparation/reflow_corpus_jsonl.py \\
        --input train/romance_corpus/horror_styled.jsonl \\
        --output train/romance_corpus/horror_styled_reflowed.jsonl

    python tools/data_preparation/reflow_corpus_jsonl.py \\
        --input train/romance_corpus/horror_styled.jsonl --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.reflow_prose import reflow_ocr_prose


def reflow_file(
    input_path: Path,
    output_path: Path,
    *,
    progress_every: int,
) -> tuple[int, int]:
    changed = 0
    total = 0
    t0 = time.time()
    with input_path.open(encoding="utf-8") as in_fh, output_path.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            original = record.get("text") or ""
            reflowed = reflow_ocr_prose(original)
            total += 1
            if reflowed != original:
                changed += 1
                record = dict(record)
                record["text"] = reflowed
                metadata = dict(record.get("metadata") or {})
                metadata["word_count"] = len(reflowed.split())
                metadata["text_reflowed"] = True
                record["metadata"] = metadata
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if progress_every and total % progress_every == 0:
                rate = total / max(time.time() - t0, 0.01)
                print(f"  ... {total:,} ({rate:,.0f}/s), {changed:,} changed", flush=True)
    return total, changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflow OCR line wraps in JSONL corpus files.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    if args.in_place:
        output_path = input_path.with_suffix(input_path.suffix + ".tmp")
    elif args.output:
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
    else:
        raise SystemExit("Provide --output or --in-place")

    print(f"Reflow: {input_path.relative_to(ROOT)}")
    total, changed = reflow_file(input_path, output_path, progress_every=args.progress_every)
    print(f"Done: {total:,} records, {changed:,} changed ({100 * changed / total:.1f}%)")

    if args.in_place:
        if not args.no_backup:
            backup = input_path.with_suffix(input_path.suffix + ".pre_reflow.bak")
            if not backup.exists():
                shutil.copy2(input_path, backup)
                print(f"Backup -> {backup.relative_to(ROOT)}")
        try:
            output_path.replace(input_path)
        except OSError as exc:
            alt = input_path.with_name(input_path.stem + "_reflowed" + input_path.suffix)
            output_path.replace(alt)
            print(f"Could not replace locked file ({exc}); wrote {alt.relative_to(ROOT)}")
    else:
        print(f"Output -> {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
