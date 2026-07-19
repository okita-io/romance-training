#!/usr/bin/env python3
"""
Human-readable review of council-classified passages.

Reads a styled JSONL produced with ``--llm-mode council`` and prints, per record,
the **text segment under review** followed by each metric's three judges (label +
quoted evidence), the arbitrator's decision + rationale, and the final label — so
you can compare the judgements against the subject being judged.

Usage:
  python tools/style_classification/review_council.py \\
    --input train/romance_corpus/_council_test_gutenberg_seg000_deep_out.jsonl \\
    --limit 5

  # Write a markdown report instead of stdout:
  python tools/style_classification/review_council.py \\
    --input <styled.jsonl> --output review.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]


def _iter_records(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def _record_title(meta: dict[str, Any], idx: int) -> str:
    bits = [str(meta.get("title") or meta.get("source") or "record")]
    author = meta.get("author")
    if author:
        bits.append(f"by {author}")
    return f"[{idx}] " + " ".join(bits)


def _wrap(text: str, width: int = 100) -> str:
    import textwrap

    out: list[str] = []
    for para in text.split("\n"):
        out.append("\n".join(textwrap.wrap(para, width=width)) if para.strip() else "")
    return "\n".join(out)


def format_record(
    rec: dict[str, Any],
    idx: int,
    *,
    fields: list[str] | None,
    text_chars: int,
) -> str:
    meta = rec.get("metadata", {})
    council = meta.get("style_council", {})
    profile = meta.get("style_profile", {})
    text = rec.get("text", "")

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(_record_title(meta, idx))
    wc = meta.get("word_count") or len(text.split())
    lines.append(f"words={wc}")
    lines.append("-" * 100)
    shown = text if text_chars <= 0 else text[:text_chars]
    lines.append("PASSAGE UNDER REVIEW:")
    lines.append(_wrap(shown))
    if text_chars > 0 and len(text) > text_chars:
        lines.append(f"… (+{len(text) - text_chars} more chars)")
    lines.append("-" * 100)

    keys = fields if fields else sorted(council.keys())
    if not council:
        lines.append("(no style_council on this record — was it classified with --llm-mode council?)")
    for field in keys:
        cm = council.get(field)
        if cm is None:
            continue
        final = cm.get("value")
        method = cm.get("method", "?")
        lines.append(f"### {field}  ->  {final!r}   [{method}]")
        for v in cm.get("votes", []):
            ev = v.get("evidence")
            ev_str = f'  "{ev}"' if ev else ""
            lines.append(f"    judge[{v.get('variant'):<10}] {str(v.get('label')):<20}{ev_str}")
        arb = cm.get("arbitration")
        if arb:
            lines.append(f"    ARBITER    {str(arb.get('value')):<20}  {arb.get('rationale') or ''}")
        # Show computable/profile value if it disagrees with council value (sanity).
        prof_val = profile.get(field)
        if prof_val is not None and prof_val != final:
            lines.append(f"    (profile={prof_val!r})")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Readable review of council classifications")
    p.add_argument("--input", type=Path, required=True, help="Styled JSONL (council mode)")
    p.add_argument("--output", type=Path, default=None, help="Write report here (default: stdout)")
    p.add_argument("--limit", type=int, default=None, help="Only first N records")
    p.add_argument("--fields", nargs="*", default=None, help="Restrict to these metric ids")
    p.add_argument(
        "--text-chars",
        type=int,
        default=0,
        help="Truncate the shown passage to N chars (0 = full text)",
    )
    args = p.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    chunks = [
        format_record(rec, i, fields=args.fields, text_chars=args.text_chars)
        for i, rec in enumerate(_iter_records(args.input, args.limit))
    ]
    report = "\n".join(chunks)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote review → {args.output} ({len(chunks)} records)")
    else:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
