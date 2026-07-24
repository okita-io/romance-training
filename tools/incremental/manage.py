#!/usr/bin/env python3
"""
Incremental training workflow: 50 MB segments + ledger.

Split corpora into manageable JSONL segments, classify segment-by-segment,
build mixed training batches (e.g. 50 MB horror + 50 MB literotica + …),
and track what is pending, classified, available, or already trained.

Usage:
    # Dashboard
    python tools/incremental/manage.py status

    # Split all mix corpora into ~50 MB input segments
    python tools/incremental/manage.py segment --all

    # Import already-finished horror_styled.jsonl as classified segments
    python tools/incremental/manage.py import-styled --corpus horror_novel_chunks

    # Classify next pending segment
    python tools/incremental/manage.py classify-next --corpus literotica_stories --pass fast --workers 4

    # Progress for a manual run_pipeline job (output in train/romance_corpus/)
    python tools/incremental/manage.py classify-progress --corpus literotica_stories --segment 0
    python tools/incremental/manage.py classify-progress --corpus literotica_stories --segment 0 --sync-ledger

    # Build mixed training batch (50 MB per corpus from available styled segments)
    python tools/incremental/manage.py build-batch --max-mb 50

    # After training on mistral_style_lora (or Silver Siren 12B)
    python tools/incremental/manage.py mark-trained --batch batch_001 --run run_001
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.incremental.classification_progress import (
    format_progress_report,
    measure_classification_progress,
    resolve_output_path,
)
from tools.incremental.ledger import (
    BATCHES_ROOT,
    GRAIN_NAMES,
    INCREMENTAL_ROOT,
    Ledger,
    corpus_input_path,
    corpus_segments_dir,
    load_corpora_config,
    multigrain_input_path,
    segment_id,
)
from tools.incremental.segment_jsonl import segment_jsonl


def _resolve_grains(grain_arg: str | None) -> list[str | None]:
    """Return grain list for segment/classify. ``None`` entry = legacy ~500w path."""
    if grain_arg is None:
        return [None]
    if grain_arg == "all":
        return list(GRAIN_NAMES)
    if grain_arg not in GRAIN_NAMES:
        raise SystemExit(f"Unknown grain: {grain_arg} (choose {', '.join(GRAIN_NAMES)}, all)")
    return [grain_arg]


def _next_batch_id(ledger: Ledger) -> str:
    existing = [k for k in ledger.batches if k.startswith("batch_")]
    n = len(existing) + 1
    return f"batch_{n:03d}"


def cmd_classify_progress(args: argparse.Namespace) -> None:
    cfg = load_corpora_config()
    grain = getattr(args, "grain", None)
    if grain is not None and grain not in GRAIN_NAMES:
        raise SystemExit(f"Unknown grain: {grain}")
    segments: list[tuple[str | None, int | None, Path, Path | None]] = []

    if args.input:
        if not args.input.is_file():
            raise SystemExit(f"Input not found: {args.input}")
        segments.append((args.corpus, args.segment, args.input, args.output))
    elif args.corpus:
        slug = args.corpus
        if slug not in cfg["corpora"]:
            raise SystemExit(f"Unknown corpus: {slug}")
        input_dir = corpus_segments_dir(slug, "input", grain=grain)
        if not input_dir.is_dir():
            raise SystemExit(f"No input segments: {input_dir}")

        if args.all_segments:
            paths = sorted(input_dir.glob("seg_*.jsonl"))
        else:
            idx = 0 if args.segment is None else args.segment
            path = input_dir / f"seg_{idx:03d}.jsonl"
            if not path.is_file():
                raise SystemExit(f"Input segment not found: {path}")
            paths = [path]

        for path in paths:
            idx = int(path.stem.split("_")[1])
            out = (
                args.output
                if (args.output and len(paths) == 1)
                else resolve_output_path(slug, idx, None, grain=grain)
            )
            segments.append((slug, idx, path, out))
    else:
        raise SystemExit("Specify --corpus or --input")

    ledger = Ledger() if args.sync_ledger else None
    pass_fast = args.pass_mode in ("fast", "full", "both")
    pass_deep = args.pass_mode in ("deep", "full", "both")
    no_rechunk = grain is not None

    for slug, idx, input_path, output_path in segments:
        progress = measure_classification_progress(
            input_path,
            output_path,
            pass_mode=args.pass_mode,
            corpus=slug,
            segment_index=idx,
            no_rechunk=no_rechunk,
        )
        label = f"{slug}/{grain}" if grain else slug
        if args.all_segments and args.quiet:
            print(
                f"{label}/seg_{idx:03d}: "
                f"{progress.complete:,}/{progress.pipeline_chunks:,} "
                f"({progress.percent_complete:.1f}%)"
            )
        else:
            print(format_progress_report(progress))
            print()

        if ledger is None or slug is None or idx is None:
            continue

        seg_id = segment_id(slug, idx, grain=grain)
        if progress.pending == 0 and progress.pipeline_chunks > 0:
            if output_path is None:
                continue
            ledger.mark_classified(seg_id, output_path, pass_fast=pass_fast, pass_deep=pass_deep)
        elif progress.complete > 0:
            seg = ledger.get_segment(seg_id) or {}
            seg.update(
                {
                    "id": seg_id,
                    "corpus": slug,
                    "segment_index": idx,
                    "grain": grain,
                    "input_path": str(input_path.relative_to(ROOT)).replace("\\", "/"),
                    "classification_status": "in_progress",
                    "training_status": "unavailable",
                }
            )
            if output_path is not None:
                seg["styled_path"] = str(output_path.relative_to(ROOT)).replace("\\", "/")
            seg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ledger.segments[seg_id] = seg

    if ledger is not None:
        ledger.save()
        print(f"Ledger updated -> {ledger.path.relative_to(ROOT)}")


def cmd_status(_: argparse.Namespace) -> None:
    ledger = Ledger()
    summary = ledger.summary()
    print(json.dumps(summary, indent=2))
    print()
    cfg = load_corpora_config()
    for slug in cfg.get("training_mix_corpora", []):
        label = cfg["corpora"][slug].get("label", slug)
        stats = summary["corpora"].get(slug, {})
        print(
            f"{label} ({slug}): "
            f"{stats.get('classification_classified', 0)} classified, "
            f"{stats.get('classification_pending', 0)} pending | "
            f"{stats.get('training_available', 0)} available for training, "
            f"{stats.get('training_trained', 0)} trained"
        )


def cmd_segment(args: argparse.Namespace) -> None:
    cfg = load_corpora_config()
    corpora = cfg.get("training_mix_corpora", []) if args.all else args.corpus
    if not corpora:
        raise SystemExit("Specify --corpus SLUG or --all")

    max_bytes = int(args.max_mb * 1024 * 1024)
    grains = _resolve_grains(args.grain)
    ledger = Ledger()

    for slug in corpora:
        if slug not in cfg["corpora"]:
            raise SystemExit(f"Unknown corpus: {slug}")

        for grain in grains:
            if grain is None:
                input_path = args.input or corpus_input_path(slug)
            else:
                input_path = args.input or multigrain_input_path(slug, grain)
            if not input_path.is_file():
                label = f"{slug}/{grain}" if grain else slug
                print(f"skip {label}: missing {input_path.relative_to(ROOT)}")
                continue

            out_dir = corpus_segments_dir(slug, "input", grain=grain)
            label = f"{slug}/{grain}" if grain else slug
            print(f"segment {label} <- {input_path.relative_to(ROOT)}")
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
            print(f"  -> {len(parts)} segments in {out_dir.relative_to(ROOT)}")
    ledger.save()
    print(f"\nLedger -> {ledger.path.relative_to(ROOT)}")


def cmd_import_styled(args: argparse.Namespace) -> None:
    cfg = load_corpora_config()
    slug = args.corpus
    if slug not in cfg["corpora"]:
        raise SystemExit(f"Unknown corpus: {slug}")

    corpus_cfg = cfg["corpora"][slug]
    styled_src = ROOT / (args.input or corpus_cfg.get("import_styled", ""))
    if not styled_src.is_file():
        raise SystemExit(f"Styled file not found: {styled_src}")

    max_bytes = int(args.max_mb * 1024 * 1024)
    out_dir = corpus_segments_dir(slug, "styled")
    print(f"import styled {slug} <- {styled_src.relative_to(ROOT)}")
    parts = segment_jsonl(styled_src, out_dir, max_bytes=max_bytes)

    ledger = Ledger()
    for info in parts:
        rec = ledger.register_styled_segment(
            slug,
            info.index,
            info.path,
            input_path=None,
            bytes=info.bytes,
            rows=info.rows,
            classified=True,
        )
        seg = ledger.segments[rec.id]
        seg["pass_fast"] = True
        seg["pass_deep"] = args.mark_deep
    ledger.save()
    print(f"  -> {len(parts)} styled segments ({sum(p.rows for p in parts):,} rows)")
    print(f"Ledger -> {ledger.path.relative_to(ROOT)}")


def cmd_classify_next(args: argparse.Namespace) -> None:
    grain = args.grain
    if grain is not None and grain not in GRAIN_NAMES:
        raise SystemExit(f"Unknown grain: {grain}")

    ledger = Ledger()
    seg = ledger.next_pending(args.corpus, grain=grain)
    if not seg:
        label = f"{args.corpus}/{grain}" if grain else args.corpus
        print(f"No pending segments for {label}")
        return

    seg_id = seg["id"]
    grain = seg.get("grain") or grain
    input_rel = seg.get("input_path")
    if not input_rel:
        raise SystemExit(f"Segment {seg_id} has no input_path")
    input_path = ROOT / input_rel

    # Prefer romance_corpus naming for grain runs; keep incremental styled/ for legacy.
    if grain:
        romance_dir = ROOT / "train" / "romance_corpus"
        romance_dir.mkdir(parents=True, exist_ok=True)
        styled_path = romance_dir / f"{args.corpus}_{grain}_styled_seg_{seg['segment_index']:03d}.jsonl"
        styled_dir = corpus_segments_dir(args.corpus, "styled", grain=grain)
        styled_dir.mkdir(parents=True, exist_ok=True)
    else:
        styled_dir = corpus_segments_dir(args.corpus, "styled")
        styled_dir.mkdir(parents=True, exist_ok=True)
        styled_path = styled_dir / f"seg_{seg['segment_index']:03d}.jsonl"

    rows = seg.get("rows", "?")
    size_mb = (seg.get("bytes") or 0) / (1024 * 1024)
    resumed = seg.get("classification_status") == "in_progress"
    print(
        f"Segment {seg_id} — {rows} rows, {size_mb:.1f} MB"
        + (" (resuming interrupted run)" if resumed else "")
    )
    print(f"  input  -> {input_path.relative_to(ROOT)}")
    print(f"  output -> {styled_path.relative_to(ROOT)}")

    seg["classification_status"] = "in_progress"
    seg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ledger.segments[seg_id] = seg
    ledger.save()

    cmd = [
        sys.executable,
        str(ROOT / "tools/style_classification/run_pipeline.py"),
        "--pass",
        args.pass_mode,
        "--workers",
        str(args.workers),
        "--input",
        str(input_path),
        "--output",
        str(styled_path),
    ]
    if grain:
        cmd.append("--no-rechunk")
    if args.quiet:
        cmd.append("--quiet")
    if getattr(args, "field_batch_size", None) is not None:
        cmd.extend(["--field-batch-size", str(args.field_batch_size)])
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Segment {seg_id} failed — status left as in_progress; rerun classify-next to resume.")
        raise

    pass_fast = args.pass_mode in ("fast", "full", "both")
    pass_deep = args.pass_mode in ("deep", "full", "both")
    ledger.mark_classified(seg_id, styled_path, pass_fast=pass_fast, pass_deep=pass_deep)
    ledger.save()
    print(f"Classified {seg_id} -> {styled_path.relative_to(ROOT)}")


def _pick_segments_for_budget(
    available: list[dict[str, Any]],
    max_bytes: int,
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    total = 0
    for seg in available:
        seg_bytes = int(seg.get("bytes") or 0)
        if picked and total + seg_bytes > max_bytes:
            break
        if not picked and seg_bytes > max_bytes:
            picked.append(seg)
            break
        if total + seg_bytes <= max_bytes:
            picked.append(seg)
            total += seg_bytes
    return picked


def cmd_build_batch(args: argparse.Namespace) -> None:
    from tools.training_formats.generate_instruction_pairs import generate

    cfg = load_corpora_config()
    corpora = cfg.get("training_mix_corpora", [])
    max_bytes = int(args.max_mb * 1024 * 1024)
    ledger = Ledger()
    batch_id = args.batch_id or _next_batch_id(ledger)
    batch_dir = BATCHES_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    segments_by_corpus: dict[str, list[str]] = {}
    combined_styled = batch_dir / "styled_combined.jsonl"

    with combined_styled.open("w", encoding="utf-8") as out_fh:
        for slug in corpora:
            available = ledger.available_for_training(slug)
            picked = _pick_segments_for_budget(available, max_bytes)
            if not picked:
                print(f"  {slug}: no available styled segments")
                continue
            seg_ids = [p["id"] for p in picked]
            segments_by_corpus[slug] = seg_ids
            ledger.allocate_segments(slug, seg_ids, batch_id)
            total_mb = sum(p.get("bytes", 0) for p in picked) / (1024 * 1024)
            print(f"  {slug}: {len(picked)} segment(s), {total_mb:.1f} MB")
            for p in picked:
                styled_path = ROOT / p["styled_path"]
                with styled_path.open(encoding="utf-8") as in_fh:
                    shutil.copyfileobj(in_fh, out_fh)

    if not segments_by_corpus:
        raise SystemExit("No styled segments available — classify or import-styled first.")

    train_path, val_path = generate(
        combined_styled,
        batch_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    manifest = {
        "batch_id": batch_id,
        "max_mb_per_corpus": args.max_mb,
        "segments": segments_by_corpus,
        "styled_combined": str(combined_styled.relative_to(ROOT)).replace("\\", "/"),
        "train_path": str(train_path.relative_to(ROOT)).replace("\\", "/"),
        "validation_path": str(val_path.relative_to(ROOT)).replace("\\", "/"),
        "status": "ready",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    ledger.create_batch(
        batch_id,
        max_mb_per_corpus=args.max_mb,
        segments_by_corpus=segments_by_corpus,
        train_path=train_path,
        val_path=val_path,
    )
    ledger.save()

    print(f"\nBatch {batch_id} ready:")
    print(f"  train -> {train_path.relative_to(ROOT)}")
    print(f"  val   -> {val_path.relative_to(ROOT)}")
    print("\nPoint train/train_config.toml paths.data_dir at this batch dir, then:")
    print(f"  python train/train_qwen_unsloth.py")
    print(f"\nAfter training: python tools/incremental/manage.py mark-trained --batch {batch_id} --run run_001")


def cmd_mark_trained(args: argparse.Namespace) -> None:
    ledger = Ledger()
    if args.batch not in ledger.batches:
        raise SystemExit(f"Unknown batch: {args.batch}")
    ledger.mark_batch_trained(
        args.batch,
        args.run,
        model_base=args.model_base or "",
        output_dir=args.output_dir or "",
    )
    ledger.save()
    print(f"Marked {args.batch} as trained (run {args.run})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental segment + training ledger")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show ledger summary")
    p_status.set_defaults(func=cmd_status)

    p_seg = sub.add_parser("segment", help="Split corpus input into ~50 MB segments")
    p_seg.add_argument("--corpus", action="append", default=None)
    p_seg.add_argument("--all", action="store_true")
    p_seg.add_argument("--max-mb", type=int, default=50)
    p_seg.add_argument(
        "--grain",
        default=None,
        choices=(*GRAIN_NAMES, "all"),
        help="Pack multigrain staging JSONL (sentence/span/act/all). Omit for legacy ~500w input.",
    )
    p_seg.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override input JSONL (legacy path or a single grain file)",
    )
    p_seg.set_defaults(func=cmd_segment)

    p_imp = sub.add_parser("import-styled", help="Segment an existing styled JSONL as classified")
    p_imp.add_argument("--corpus", required=True)
    p_imp.add_argument("--input", type=Path, default=None)
    p_imp.add_argument("--max-mb", type=int, default=50)
    p_imp.add_argument("--mark-deep", action="store_true", help="Mark segments as deep-pass complete")
    p_imp.set_defaults(func=cmd_import_styled)

    p_cls = sub.add_parser(
        "classify-next",
        help="Classify the next pending segment (one segment per invocation)",
    )
    p_cls.add_argument("--corpus", required=True)
    p_cls.add_argument("--pass", dest="pass_mode", default="both", choices=("fast", "deep", "full", "both"))
    p_cls.add_argument("--workers", type=int, default=4)
    p_cls.add_argument("--quiet", action="store_true")
    p_cls.add_argument(
        "--field-batch-size",
        type=int,
        default=3,
        metavar="N",
        help="Joint mode: max labels per LLM call (default 3). 0 = one call per pass",
    )
    p_cls.add_argument(
        "--grain",
        default=None,
        choices=GRAIN_NAMES,
        help="Classify next pending multigrain segment (passes --no-rechunk to the pipeline)",
    )
    p_cls.set_defaults(func=cmd_classify_next)

    p_prog = sub.add_parser(
        "classify-progress",
        help="Show Phase 2 progress for a segment (works with manual run_pipeline output)",
    )
    p_prog.add_argument("--corpus", default=None)
    p_prog.add_argument("--segment", type=int, default=None, help="Segment index (default: 0)")
    p_prog.add_argument("--all-segments", action="store_true", help="Report every input segment")
    p_prog.add_argument("--input", type=Path, default=None)
    p_prog.add_argument("--output", type=Path, default=None)
    p_prog.add_argument(
        "--pass",
        dest="pass_mode",
        default="both",
        choices=("fast", "deep", "full", "both"),
    )
    p_prog.add_argument(
        "--sync-ledger",
        action="store_true",
        help="Update ledger.json from output progress (in_progress or classified)",
    )
    p_prog.add_argument("--quiet", action="store_true", help="One line per segment with --all-segments")
    p_prog.add_argument(
        "--grain",
        default=None,
        choices=GRAIN_NAMES,
        help="Report progress for a multigrain segment tree (implies no-rechunk counting)",
    )
    p_prog.set_defaults(func=cmd_classify_progress)

    p_batch = sub.add_parser("build-batch", help="Mixed training batch from available styled segments")
    p_batch.add_argument("--max-mb", type=int, default=50)
    p_batch.add_argument("--batch-id", default=None)
    p_batch.add_argument("--val-fraction", type=float, default=0.1)
    p_batch.add_argument("--seed", type=int, default=42)
    p_batch.set_defaults(func=cmd_build_batch)

    p_done = sub.add_parser("mark-trained", help="Record a completed training run")
    p_done.add_argument("--batch", required=True)
    p_done.add_argument("--run", required=True)
    p_done.add_argument("--model-base", default="")
    p_done.add_argument("--output-dir", default="")
    p_done.set_defaults(func=cmd_mark_trained)

    args = parser.parse_args()
    INCREMENTAL_ROOT.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
