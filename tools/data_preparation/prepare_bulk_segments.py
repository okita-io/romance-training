#!/usr/bin/env python3
"""
Build multi-grain staging JSONL and pack ~50 MB bulk input segments.

Step 1 — materialize sentence / span / act rows from processed ``chunks.jsonl``
         into ``train/staging/multigrain/<slug>/{grain}.jsonl``.
Step 2 — pack each grain into ``train/incremental/segments/<slug>/<grain>/input/seg_*.jsonl``
         and register them in the incremental ledger.

Default grain is ``span`` (~300 words) — the council / span-editor teacher size.
Legacy ~500w trees are left untouched.

Usage:
    # Report what would be built (needs source-data/processed/*/chunks.jsonl)
    python tools/data_preparation/prepare_bulk_segments.py --dry-run

    # Build span staging + 50 MB segments for every training-mix corpus
    python tools/data_preparation/prepare_bulk_segments.py --write

    # One corpus, all grains
    python tools/data_preparation/prepare_bulk_segments.py --write \\
      --slug gutenberg_fiction --grain all

    # Staging only (skip 50 MB packing)
    python tools/data_preparation/prepare_bulk_segments.py --write --staging-only \\
      --slug horror_novel_chunks
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.build_multigrain_chunks import (  # noqa: E402
    INVENTORY_PATH,
    discover_chunk_files,
    print_summary,
    process_corpus,
)
from tools.data_preparation.paths import MULTIGRAIN_DIR  # noqa: E402
from tools.incremental.ledger import (  # noqa: E402
    GRAIN_NAMES,
    Ledger,
    corpus_segments_dir,
    load_corpora_config,
    multigrain_input_path,
)
from tools.incremental.segment_jsonl import segment_jsonl  # noqa: E402
def _resolve_slugs(args: argparse.Namespace, cfg: dict[str, Any]) -> list[str]:
    if args.slug:
        unknown = [s for s in args.slug if s not in cfg["corpora"]]
        if unknown:
            raise SystemExit(f"Unknown corpus slug(s): {', '.join(unknown)}")
        return list(args.slug)
    return list(cfg.get("training_mix_corpora") or cfg["corpora"].keys())


def _resolve_grains(grain_arg: str) -> list[str]:
    if grain_arg == "all":
        return list(GRAIN_NAMES)
    if grain_arg not in GRAIN_NAMES:
        raise SystemExit(f"Unknown grain: {grain_arg} (choose {', '.join(GRAIN_NAMES)}, all)")
    return [grain_arg]


def _pack_grain(
    slug: str,
    grain: str,
    *,
    max_bytes: int,
    ledger: Ledger,
) -> dict[str, Any]:
    input_path = multigrain_input_path(slug, grain)
    if not input_path.is_file():
        return {
            "slug": slug,
            "grain": grain,
            "skipped": True,
            "reason": f"missing {input_path.relative_to(ROOT)}",
        }

    out_dir = corpus_segments_dir(slug, "input", grain=grain)
    parts = segment_jsonl(input_path, out_dir, max_bytes=max_bytes)
    for info in parts:
        ledger.register_input_segment(
            slug,
            info.index,
            info.path,
            bytes=info.bytes,
            rows=info.rows,
            grain=grain,
        )
    return {
        "slug": slug,
        "grain": grain,
        "skipped": False,
        "segments": len(parts),
        "rows": sum(p.rows for p in parts),
        "bytes": sum(p.bytes for p in parts),
        "out_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "source": str(input_path.relative_to(ROOT)).replace("\\", "/"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build multigrain staging JSONL and pack ~50 MB bulk segments "
            "(default grain: span)."
        ),
    )
    parser.add_argument(
        "--slug",
        action="append",
        default=None,
        help="Corpus slug (repeatable). Default: all training_mix_corpora.",
    )
    parser.add_argument(
        "--grain",
        default="span",
        choices=(*GRAIN_NAMES, "all"),
        help="Which grain(s) to pack into segments (default: span).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write staging JSONL and/or segments (without this: dry-run report only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for omitting --write (report only).",
    )
    parser.add_argument(
        "--staging-only",
        action="store_true",
        help="Build staging multigrain JSONL but do not pack 50 MB segments.",
    )
    parser.add_argument(
        "--pack-only",
        action="store_true",
        help="Skip staging rebuild; pack existing train/staging/multigrain JSONL only.",
    )
    parser.add_argument("--span-words", type=int, default=300)
    parser.add_argument("--act-words", type=int, default=1000)
    parser.add_argument("--overlap-sentences", type=int, default=2)
    parser.add_argument("--max-mb", type=float, default=50.0, help="Segment byte budget (default 50).")
    parser.add_argument("--write-max-rows", type=int, default=None, metavar="N")
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    args = parser.parse_args()

    if args.dry_run and args.write:
        raise SystemExit("Use either --write or --dry-run, not both.")
    do_write = bool(args.write) and not args.dry_run

    cfg = load_corpora_config()
    slugs = _resolve_slugs(args, cfg)
    grains = _resolve_grains(args.grain)
    max_bytes = int(args.max_mb * 1024 * 1024)

    print(f"Corpora: {', '.join(slugs)}")
    print(f"Grains to pack: {', '.join(grains)}")
    print(f"Mode: {'WRITE' if do_write else 'DRY-RUN'}")
    if args.pack_only:
        print("Pack-only: skipping staging rebuild")
    if args.staging_only:
        print("Staging-only: skipping 50 MB segment packing")

    staging_entries: list[dict[str, Any]] = []
    if not args.pack_only:
        discovered = dict(discover_chunk_files(slugs=slugs))
        missing = [s for s in slugs if s not in discovered]
        if missing:
            print(
                "\nMissing processed chunks.jsonl for:\n  "
                + "\n  ".join(missing)
                + f"\nExpected under source-data/processed/<slug>/chunks.jsonl"
            )
            if not discovered:
                raise SystemExit(1)
            print("Continuing with corpora that have chunks …")

        for slug in slugs:
            path = discovered.get(slug)
            if path is None:
                continue
            print(f"\n[staging] {slug} <- {path.relative_to(ROOT)}")
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
            staging_entries.append(entry)
            if do_write:
                print(
                    f"  wrote sent={entry['grain_counts']['sentence']:,} "
                    f"span={entry['grain_counts']['span']:,} "
                    f"act={entry['grain_counts']['act']:,} -> {MULTIGRAIN_DIR / slug}"
                )

        if staging_entries:
            args.inventory.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "span_words": args.span_words,
                "act_words": args.act_words,
                "corpora": staging_entries,
            }
            if do_write:
                args.inventory.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                print(f"\nInventory -> {args.inventory.relative_to(ROOT)}")
            print_summary(staging_entries)

    pack_results: list[dict[str, Any]] = []
    if not args.staging_only:
        ledger = Ledger() if do_write else None
        print(f"\n[pack] max_mb={args.max_mb}")
        for slug in slugs:
            for grain in grains:
                staging_path = multigrain_input_path(slug, grain)
                if not staging_path.is_file():
                    print(f"  skip {slug}/{grain}: missing staging file")
                    pack_results.append(
                        {
                            "slug": slug,
                            "grain": grain,
                            "skipped": True,
                            "reason": "missing staging",
                        }
                    )
                    continue
                if not do_write:
                    size_mb = staging_path.stat().st_size / (1024 * 1024)
                    print(
                        f"  would pack {slug}/{grain} "
                        f"({size_mb:.1f} MB) -> "
                        f"{corpus_segments_dir(slug, 'input', grain=grain).relative_to(ROOT)}"
                    )
                    pack_results.append(
                        {
                            "slug": slug,
                            "grain": grain,
                            "skipped": False,
                            "dry_run": True,
                            "source_mb": round(size_mb, 2),
                        }
                    )
                    continue
                assert ledger is not None
                result = _pack_grain(slug, grain, max_bytes=max_bytes, ledger=ledger)
                pack_results.append(result)
                if result.get("skipped"):
                    print(f"  skip {slug}/{grain}: {result.get('reason')}")
                else:
                    mb = result["bytes"] / (1024 * 1024)
                    print(
                        f"  {slug}/{grain}: {result['segments']} segment(s), "
                        f"{result['rows']:,} rows, {mb:.1f} MB -> {result['out_dir']}"
                    )
        if do_write and ledger is not None:
            ledger.save()
            print(f"\nLedger -> {ledger.path.relative_to(ROOT)}")

    written = sum(1 for r in pack_results if not r.get("skipped") and not r.get("dry_run"))
    print(
        f"\nDone. staging_corpora={len(staging_entries)} "
        f"packed={written} dry_run={not do_write}"
    )


if __name__ == "__main__":
    main()
