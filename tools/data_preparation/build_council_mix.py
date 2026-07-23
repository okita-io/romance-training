#!/usr/bin/env python3
"""
Build a stratified, interleaved council pilot / gold mix from span segments.

Pulls unlabeled (or already-built) span rows from each corpus in
``training_mix_corpora``, samples N per corpus (or a weighted share of --total),
then **interleaves** them so consecutive lines rotate across fiction, Literotica,
Gutenberg, fantasy, horror, etc.

Output is ready for:
    python tools/style_classification/run_pipeline.py \\
      --input train/staging/council_mix/<name>/mix.jsonl \\
      --output train/romance_corpus/<name>_council.jsonl \\
      --llm-mode council --no-rechunk --pass deep

Usage:
    # Dry-run: how many spans are available per corpus
    python tools/data_preparation/build_council_mix.py --report-only

    # Equal count: 40 spans from each mix corpus, interleaved
    python tools/data_preparation/build_council_mix.py --write --per-corpus 40

    # Fixed total, equal shares across available corpora
    python tools/data_preparation/build_council_mix.py --write --total 280 --seed 7

    # Weighted by corpora.json mix_weight (falls back to equal if unset)
    python tools/data_preparation/build_council_mix.py --write --total 300 --mode weighted
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.paths import STAGING_ROOT  # noqa: E402
from tools.incremental.ledger import (  # noqa: E402
    corpus_segments_dir,
    load_corpora_config,
    multigrain_input_path,
)

COUNCIL_MIX_ROOT = STAGING_ROOT / "council_mix"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _record_id(record: dict[str, Any], corpus: str, ordinal: int) -> str:
    meta = record.get("metadata") or {}
    for key in ("span_id", "act_id", "sentence_id", "chunk_id", "id"):
        val = meta.get(key) or record.get(key)
        if val:
            return str(val)
    story = meta.get("story_id") or record.get("story_id") or "unknown"
    idx = meta.get("chunk_index", ordinal)
    return f"{corpus}:{story}:{idx}"


def _discover_span_sources(
    slug: str,
    *,
    grain: str,
) -> list[Path]:
    """Prefer packed segments; fall back to staging grain JSONL."""
    seg_dir = corpus_segments_dir(slug, "input", grain=grain)
    if seg_dir.is_dir():
        parts = sorted(seg_dir.glob("seg_*.jsonl"))
        if parts:
            return parts
    staging = multigrain_input_path(slug, grain)
    if staging.is_file():
        return [staging]
    return []


def _load_corpus_rows(
    slug: str,
    *,
    grain: str,
    max_scan: int | None,
) -> list[dict[str, Any]]:
    paths = _discover_span_sources(slug, grain=grain)
    if not paths:
        return []
    rows: list[dict[str, Any]] = []
    for path in paths:
        for record in iter_jsonl(path):
            text = (record.get("text") or "").strip()
            if not text:
                continue
            meta = dict(record.get("metadata") or {})
            meta.setdefault("source_corpus", slug)
            meta.setdefault("grain", grain)
            record = {**record, "metadata": meta}
            rows.append(record)
            if max_scan is not None and len(rows) >= max_scan:
                return rows
    return rows


def _mix_weight(corpus_cfg: dict[str, Any], mode: str) -> float:
    if mode == "equal":
        return 1.0
    if "mix_weight" in corpus_cfg:
        return float(corpus_cfg["mix_weight"])
    # expansion_ratio is "rows per source chunk", not importance — invert lightly
    # so huge expanders don't dominate when mix_weight is unset.
    expansion = float(corpus_cfg.get("expansion_ratio") or 1.0)
    return 1.0 / max(expansion, 0.1)


def _allocate_counts(
    available: dict[str, int],
    weights: dict[str, float],
    *,
    per_corpus: int | None,
    total: int | None,
) -> dict[str, int]:
    slugs = [s for s, n in available.items() if n > 0]
    if not slugs:
        return {}

    if per_corpus is not None:
        return {s: min(per_corpus, available[s]) for s in slugs}

    if total is None:
        raise SystemExit("Specify --per-corpus N or --total N")

    raw_w = {s: max(weights.get(s, 1.0), 0.0) for s in slugs}
    wsum = sum(raw_w.values()) or float(len(slugs))
    # Largest-remainder so counts sum to total.
    ideal = {s: total * (raw_w[s] / wsum) for s in slugs}
    base = {s: int(ideal[s]) for s in slugs}
    remainders = sorted(
        ((ideal[s] - base[s], s) for s in slugs),
        reverse=True,
    )
    assigned = sum(base.values())
    for _, s in remainders:
        if assigned >= total:
            break
        base[s] += 1
        assigned += 1

    # Cap by availability; redistribute shortfall once.
    counts = {s: min(base[s], available[s]) for s in slugs}
    shortfall = total - sum(counts.values())
    if shortfall > 0:
        room = sorted(
            ((available[s] - counts[s], s) for s in slugs if available[s] > counts[s]),
            reverse=True,
        )
        i = 0
        while shortfall > 0 and room:
            _, s = room[i % len(room)]
            if counts[s] < available[s]:
                counts[s] += 1
                shortfall -= 1
            i += 1
            if i > total * 2:
                break
    return counts


def _sample_rows(
    rows: list[dict[str, Any]],
    k: int,
    *,
    rng: random.Random,
    strategy: str,
) -> list[dict[str, Any]]:
    if k <= 0 or not rows:
        return []
    if k >= len(rows):
        return list(rows)
    if strategy == "systematic":
        step = len(rows) / k
        idxs = [min(len(rows) - 1, int(i * step)) for i in range(k)]
        # Dedupe if collisions from rounding, then fill.
        seen: set[int] = set()
        chosen: list[int] = []
        for idx in idxs:
            if idx not in seen:
                seen.add(idx)
                chosen.append(idx)
        cursor = 0
        while len(chosen) < k:
            if cursor not in seen:
                seen.add(cursor)
                chosen.append(cursor)
            cursor += 1
            if cursor >= len(rows):
                break
        return [rows[i] for i in sorted(chosen)[:k]]
    return rng.sample(rows, k)


def interleave(by_corpus: dict[str, list[dict[str, Any]]], order: list[str]) -> list[dict[str, Any]]:
    queues = {s: deque(by_corpus[s]) for s in order if by_corpus.get(s)}
    mixed: list[dict[str, Any]] = []
    while queues:
        exhausted: list[str] = []
        for slug in order:
            q = queues.get(slug)
            if not q:
                if slug in queues:
                    exhausted.append(slug)
                continue
            mixed.append(q.popleft())
            if not q:
                exhausted.append(slug)
        for slug in exhausted:
            queues.pop(slug, None)
    return mixed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified interleaved council mix from span segments across corpora.",
    )
    parser.add_argument(
        "--slug",
        action="append",
        default=None,
        help="Limit to these corpora (repeatable). Default: training_mix_corpora.",
    )
    parser.add_argument("--grain", default="span", choices=("sentence", "span", "act"))
    parser.add_argument("--per-corpus", type=int, default=None, metavar="N")
    parser.add_argument("--total", type=int, default=None, metavar="N")
    parser.add_argument(
        "--mode",
        choices=("equal", "weighted"),
        default="equal",
        help="equal shares, or weighted by corpora.json mix_weight (else 1/expansion_ratio).",
    )
    parser.add_argument(
        "--sample",
        choices=("random", "systematic"),
        default="systematic",
        help="Within-corpus sampling (default: systematic stride across the file).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--name",
        default=None,
        help="Output folder name under train/staging/council_mix/ (default: timestamped).",
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--max-scan",
        type=int,
        default=None,
        metavar="N",
        help="Optional cap on rows loaded per corpus (faster reports on huge files).",
    )
    args = parser.parse_args()

    if args.write and args.report_only:
        raise SystemExit("Use either --write or --report-only, not both.")
    if not args.write:
        args.report_only = True

    cfg = load_corpora_config()
    corpora_cfg = cfg["corpora"]
    slugs = args.slug or list(cfg.get("training_mix_corpora") or corpora_cfg.keys())
    unknown = [s for s in slugs if s not in corpora_cfg]
    if unknown:
        raise SystemExit(f"Unknown corpus slug(s): {', '.join(unknown)}")

    print(f"Grain: {args.grain}")
    print(f"Corpora: {', '.join(slugs)}")
    print(f"Mode: {args.mode} | sample: {args.sample} | seed: {args.seed}")

    available: dict[str, int] = {}
    pools: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, list[str]] = {}

    for slug in slugs:
        paths = _discover_span_sources(slug, grain=args.grain)
        sources[slug] = [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths]
        if not paths:
            print(f"  {slug}: NO span source (build/pack first)")
            available[slug] = 0
            pools[slug] = []
            continue
        rows = _load_corpus_rows(slug, grain=args.grain, max_scan=args.max_scan)
        pools[slug] = rows
        available[slug] = len(rows)
        print(f"  {slug}: {len(rows):,} rows from {len(paths)} file(s)")

    usable = {s: n for s, n in available.items() if n > 0}
    if not usable:
        raise SystemExit(
            "No span rows found. Run prepare_bulk_segments.py --write first "
            "(requires source-data/processed/*/chunks.jsonl)."
        )

    weights = {
        s: _mix_weight(corpora_cfg[s], args.mode) for s in usable
    }
    if args.per_corpus is None and args.total is None:
        # Sensible report default: show what 40/corpus would look like.
        args.per_corpus = 40

    counts = _allocate_counts(
        available,
        weights,
        per_corpus=args.per_corpus,
        total=args.total,
    )
    print("\nAllocation:")
    for slug, n in counts.items():
        print(f"  {slug}: {n} / {available[slug]} (weight={weights.get(slug, 0):.3f})")

    rng = random.Random(args.seed)
    sampled: dict[str, list[dict[str, Any]]] = {}
    for slug, n in counts.items():
        sampled[slug] = _sample_rows(
            pools[slug], n, rng=rng, strategy=args.sample
        )

    order = [s for s in slugs if sampled.get(s)]
    mixed = interleave(sampled, order)
    print(f"\nInterleaved mix size: {len(mixed):,}")

    if args.report_only:
        print("(report-only — pass --write to materialize)")
        return

    name = args.name or time.strftime("mix_%Y%m%d_%H%M%S", time.gmtime())
    out_dir = COUNCIL_MIX_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    mix_path = out_dir / "mix.jsonl"
    ids_path = out_dir / "sampled_ids.jsonl"
    manifest_path = out_dir / "manifest.json"

    with mix_path.open("w", encoding="utf-8") as mix_fh, ids_path.open(
        "w", encoding="utf-8"
    ) as ids_fh:
        for i, record in enumerate(mixed):
            corpus = (record.get("metadata") or {}).get("source_corpus", "unknown")
            rid = _record_id(record, corpus, i)
            meta = dict(record.get("metadata") or {})
            meta["council_mix"] = name
            meta["council_mix_index"] = i
            meta["council_mix_id"] = rid
            row = {**record, "metadata": meta}
            mix_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            ids_fh.write(
                json.dumps(
                    {
                        "council_mix_id": rid,
                        "corpus": corpus,
                        "mix_index": i,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    per_corpus_written = defaultdict(int)
    for record in mixed:
        per_corpus_written[(record.get("metadata") or {}).get("source_corpus", "?")] += 1

    manifest = {
        "name": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grain": args.grain,
        "mode": args.mode,
        "sample": args.sample,
        "seed": args.seed,
        "per_corpus_request": args.per_corpus,
        "total_request": args.total,
        "rows": len(mixed),
        "counts": dict(per_corpus_written),
        "weights": weights,
        "sources": sources,
        "mix_path": str(mix_path.relative_to(ROOT)).replace("\\", "/"),
        "sampled_ids_path": str(ids_path.relative_to(ROOT)).replace("\\", "/"),
        "classify_hint": (
            f"python tools/style_classification/run_pipeline.py "
            f"--input {mix_path.relative_to(ROOT)} "
            f"--output train/romance_corpus/{name}_council.jsonl "
            f"--llm-mode council --no-rechunk --pass deep"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {len(mixed):,} rows -> {mix_path.relative_to(ROOT)}")
    print(f"Manifest -> {manifest_path.relative_to(ROOT)}")
    print("\nNext:")
    print(f"  {manifest['classify_hint']}")


if __name__ == "__main__":
    main()
