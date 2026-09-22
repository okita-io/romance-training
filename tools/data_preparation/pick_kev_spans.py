#!/usr/bin/env python3
"""
Sample Kev-sized spans from the bulk styled corpus for re-classification.

Picks random passages from ``*_styled_seg_*.jsonl``, then cuts a sentence-
bounded window that fits Jev/Kev's 384-token training state (~1400 chars).
Old style_profile labels are dropped: a subset span is not the same object.

Glyph dumps, footnotes, credits, screenplay sluglines, and other non-narrative
chunks are skipped so the mix stays English prose.

Usage:
    python tools/data_preparation/pick_kev_spans.py --n 80
    python tools/data_preparation/pick_kev_spans.py --n 80 --report-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.language_filter import (  # noqa: E402
    english_word_ratio,
    has_non_latin_script,
)
from tools.data_preparation.prose_filter import classify_chunk_prose  # noqa: E402
from tools.style_classification.chunk_text import (  # noqa: E402
    KEV_MIN_WORDS,
    KEV_STATE_CHARS,
    random_kev_span,
)

CORPUS_DIR = ROOT / "train" / "romance_corpus"
DEFAULT_OUT = ROOT / "train" / "style_training" / "kev" / "spans" / "span_mix.jsonl"
_STYLED_RE = re.compile(r"^(?P<slug>.+)_styled_seg_")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def discover_styled_paths(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    return sorted(
        p
        for p in corpus_dir.glob("*_styled_seg_*.jsonl")
        if p.is_file() and not p.name.startswith("_")
    )


def corpus_slug(path: Path) -> str:
    match = _STYLED_RE.match(path.name)
    return match.group("slug") if match else path.stem


def _text_ok(text: str, min_words: int) -> bool:
    return span_reject_reason(text, min_words=min_words) is None


def span_reject_reason(text: str, *, min_words: int) -> str | None:
    """Return a drop reason, or None if the span looks like English narrative."""
    raw = (text or "").strip()
    if not raw:
        return "too_short"
    if len(raw) > 50_000:
        return "unbounded_dump"
    if len(raw.split()) < min_words:
        return "too_short"
    if has_non_latin_script(raw):
        return "non_latin_script"
    if english_word_ratio(raw) < 0.04:
        return "non_english"
    prose = classify_chunk_prose(raw, min_words=min_words)
    if prose.verdict != "prose":
        return prose.reason or "non_prose"
    return None


def reservoir_sample(
    path: Path,
    k: int,
    rng: random.Random,
    *,
    min_words: int,
) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    seen = 0
    for record in iter_jsonl(path):
        text = record.get("text") or ""
        if not _text_ok(text, min_words):
            continue
        seen += 1
        if len(sample) < k:
            sample.append(record)
            continue
        j = rng.randrange(seen)
        if j < k:
            sample[j] = record
    return sample


def _span_id(text: str, slug: str) -> str:
    digest = hashlib.sha256(f"{slug}|{' '.join(text.casefold().split())}".encode()).hexdigest()
    return digest[:12]


def to_span_record(
    record: dict[str, Any],
    *,
    slug: str,
    source_file: str,
    rng: random.Random,
    max_chars: int,
    min_words: int,
) -> dict[str, Any] | None:
    original = (record.get("text") or "").strip()
    if not original:
        return None
    span, truncated = random_kev_span(
        original, rng, max_chars=max_chars, min_words=min_words
    )
    if not _text_ok(span, min_words) and truncated:
        return None
    parent = dict(record.get("metadata") or {})
    extra = dict(parent.get("extra") or {})
    parent.pop("style_profile", None)
    parent.pop("style_council", None)
    parent.update(
        {
            "source_corpus": slug,
            "source_file": source_file,
            "grain": "kev_span",
            "span_id": _span_id(span, slug),
            "parent_word_count": len(original.split()),
            "parent_chars": len(original),
            "span_word_count": len(span.split()),
            "span_chars": len(span),
            "truncated_to_kev": truncated,
            "title": parent.get("title") or extra.get("title_slug") or "",
            "author": parent.get("author") or "",
            "story_key": parent.get("story_key") or extra.get("story_key") or "",
        }
    )
    return {"text": span, "metadata": parent}


def pick_spans(
    paths: list[Path],
    *,
    n: int,
    seed: int,
    max_chars: int,
    min_words: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    if not paths:
        return [], {"files": [], "kept": 0}
    per_file = max(1, (n + len(paths) - 1) // len(paths))
    pool: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        drawn = reservoir_sample(path, per_file * 6, rng, min_words=min_words)
        rng.shuffle(drawn)
        for record in drawn:
            pool.append((path, record))
    rng.shuffle(pool)

    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_corpus: Counter[str] = Counter()
    truncated = 0
    skipped_short = 0
    skipped_vet: Counter[str] = Counter()
    for path, record in pool:
        if len(kept) >= n:
            break
        slug = corpus_slug(path)
        span = to_span_record(
            record,
            slug=slug,
            source_file=path.name,
            rng=rng,
            max_chars=max_chars,
            min_words=min_words,
        )
        if span is None:
            skipped_short += 1
            continue
        reason = span_reject_reason(span["text"], min_words=min_words)
        if reason:
            skipped_vet[reason] += 1
            continue
        key = span["metadata"]["span_id"]
        if key in seen:
            continue
        seen.add(key)
        kept.append(span)
        by_corpus[slug] += 1
        if span["metadata"]["truncated_to_kev"]:
            truncated += 1

    stats = {
        "files": [p.name for p in paths],
        "requested": n,
        "kept": len(kept),
        "truncated": truncated,
        "skipped_short": skipped_short,
        "skipped_vet": dict(skipped_vet),
        "by_corpus": dict(by_corpus),
        "max_chars": max_chars,
        "min_words": min_words,
        "seed": seed,
        "char_max": max((len(r["text"]) for r in kept), default=0),
        "word_mean": round(
            sum(r["metadata"]["span_word_count"] for r in kept) / len(kept), 1
        )
        if kept
        else 0,
    }
    return kept, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=80, help="How many unique Kev spans to keep")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-chars", type=int, default=KEV_STATE_CHARS)
    ap.add_argument("--min-words", type=int, default=KEV_MIN_WORDS)
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report-only", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = discover_styled_paths(args.corpus_dir)
    if not paths:
        print(f"no *_styled_seg_*.jsonl under {args.corpus_dir}", file=sys.stderr)
        return 1
    spans, stats = pick_spans(
        paths,
        n=args.n,
        seed=args.seed,
        max_chars=args.max_chars,
        min_words=args.min_words,
    )
    print(
        f"kept {stats['kept']}/{stats['requested']} spans from {len(paths)} files; "
        f"truncated {stats['truncated']}; max {stats['char_max']} chars; "
        f"mean {stats['word_mean']} words"
    )
    if stats.get("skipped_vet"):
        dropped = ", ".join(
            f"{reason}={count}" for reason, count in sorted(stats["skipped_vet"].items())
        )
        print(f"  dropped by vet: {dropped}")
    for slug, count in sorted(stats["by_corpus"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {slug}: {count}")
    if args.report_only:
        return 0 if spans else 1
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for row in spans:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = out.with_suffix(".report.json")
    report.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {report}")
    return 0 if spans else 1


if __name__ == "__main__":
    raise SystemExit(main())
