#!/usr/bin/env python3
"""
Phase 2: Run the style classification pipeline over a JSONL corpus.

Reads JSONL (text + metadata), adds style_profile to metadata, writes enriched JSONL.
Supports resuming interrupted runs — already-processed records are skipped.
Each classified chunk is appended and flushed immediately (safe to interrupt).
On Ctrl+C, pending worker tasks are cancelled, the output is compacted from the
in-memory index (one line per chunk), then the process exits. Deep/both also
compact on normal completion.

Usage:
    # Classify existing Gutenberg romance corpus (fast, no LLM)
    python tools/style_classification/run_pipeline.py --no-llm

    # Full classification with LLM on all records (single model, all fields)
    python tools/style_classification/run_pipeline.py

    # Two-pass hybrid (see source/multi-pass.md)
    # Pass 1 — small model, ~4 workers
    python tools/style_classification/run_pipeline.py --pass fast --workers 4 \\
        --input source-data/processed/horror_novel_chunks/chunks.jsonl \\
        --output train/romance_corpus/horror_styled.jsonl

    # Pass 2 — same or larger model (--pass deep), or use --pass both for one model, two calls/chunk
    python tools/style_classification/run_pipeline.py --pass deep --workers 2 \\
        --input source-data/processed/horror_novel_chunks/chunks.jsonl \\
        --output train/romance_corpus/horror_styled.jsonl

    # Same model for both field sets (2 LLM calls/chunk, no swap in LM Studio)
    python tools/style_classification/run_pipeline.py --pass both --workers 4 \\
        --input source-data/processed/horror_novel_chunks/chunks.jsonl \\
        --output train/romance_corpus/horror_styled.jsonl

    # LLM on a 20% sample (good balance of speed vs. coverage)
    python tools/style_classification/run_pipeline.py --llm-sample-rate 0.2

    # Parallel workers (computable-only mode)
    python tools/style_classification/run_pipeline.py --no-llm --workers 8
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
import random
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "train" / "romance_corpus" / "gutenberg_romance.jsonl"
DEFAULT_OUTPUT = ROOT / "train" / "romance_corpus" / "gutenberg_styled.jsonl"

CHUNK_WORDS = 500
CHUNK_OVERLAP_SENTENCES = 2
THROUGHPUT_WINDOW = 30
RUN_LOG_EVERY = 50
RUN_LOG_DIR = ROOT / "train" / "incremental" / "logs"


class _ProgressTracker:
    """Wall-clock throughput from completion timestamps (works with parallel workers)."""

    def __init__(self, total: int, *, window: int = THROUGHPUT_WINDOW) -> None:
        self.total = total
        self.processed = 0
        self.t0 = time.time()
        self.window = window
        self._times: deque[float] = deque(maxlen=window)
        self._last: float | None = None

    def mark_done(self) -> tuple[float | None, float, float]:
        """Return (seconds since last completion, rec/s, eta seconds)."""
        now = time.time()
        interval = None if self._last is None else now - self._last
        self._last = now
        self._times.append(now)
        self.processed += 1

        if len(self._times) >= 2:
            span = self._times[-1] - self._times[0]
            throughput = (len(self._times) - 1) / max(span, 0.01)
        else:
            throughput = self.processed / max(now - self.t0, 0.01)

        remain = self.total - self.processed
        eta_sec = remain / max(throughput, 0.001)
        return interval, throughput, eta_sec

    def snapshot(self) -> tuple[int, float, float]:
        """Return (processed, rec/s, eta seconds) using the same throughput logic."""
        remain = self.total - self.processed
        now = time.time()
        if len(self._times) >= 2:
            span = self._times[-1] - self._times[0]
            throughput = (len(self._times) - 1) / max(span, 0.01)
        elif self.processed >= 1:
            throughput = self.processed / max(now - self.t0, 0.01)
        else:
            throughput = 0.001
        eta_sec = remain / max(throughput, 0.001)
        return self.processed, throughput, eta_sec

    def format_progress(self, interval: float | None, throughput: float, eta_sec: float) -> str:
        parts = [f"classified {self.processed}/{self.total}", f"{self.total - self.processed} remain"]
        if interval is not None:
            parts.append(f"interval {interval:.1f}s")
        parts.append(f"{throughput:.2f} rec/s")
        parts.append(f"~{eta_sec / 60:.0f} min left")
        return " | ".join(parts)


def _chunk_record(
    record: dict,
    *,
    target_words: int = CHUNK_WORDS,
    overlap_sentences: int = CHUNK_OVERLAP_SENTENCES,
) -> list[dict]:
    """Split a long record into sentence-boundary chunks of ~target_words."""
    from tools.style_classification.chunk_text import chunk_record

    return chunk_record(
        record,
        target_words=target_words,
        overlap_sentences=overlap_sentences,
    )


def _record_key(record: dict) -> str:
    """Stable resume identity for a chunk.

    Must match saved rows: ``_enrich`` runs ``normalize_prose_text`` before write,
    so the signature is taken from normalized text (otherwise resume misses work).
    """
    from tools.data_preparation.unified_corpus import normalize_prose_text

    m = record.get("metadata", {})
    source = m.get("source", record.get("source", ""))
    chunk = m.get("chunk_index", 0)
    text_sig = normalize_prose_text(record.get("text", "") or "")[:60]
    return f"{source}|{chunk}|{text_sig}"


def _record_label(record: dict) -> str:
    """Short human-readable label for progress logs."""
    m = record.get("metadata", {})
    parts: list[str] = []
    if m.get("source"):
        parts.append(str(m["source"]))
    if m.get("chunk_index") is not None:
        parts.append(f"chunk:{m['chunk_index']}")
    elif m.get("story_key"):
        parts.append(str(m["story_key"]))
    if m.get("source_file"):
        parts.append(str(m["source_file"]))
    if m.get("title"):
        parts.append(str(m["title"])[:60])
    if not parts:
        preview = record.get("text", "")[:50].replace("\n", " ").strip()
        parts.append(f'"{preview}..."')
    return " | ".join(parts)


def _profile_hint(record: dict) -> str:
    profile = record.get("metadata", {}).get("style_profile", {})
    if not isinstance(profile, dict):
        return ""
    register = profile.get("register")
    if register:
        return f"register={register}"
    tone = profile.get("tone")
    if tone:
        return f"tone={tone}"
    return ""


def _load_output_index(path: Path) -> tuple[dict[str, dict], list[str]]:
    """Load output JSONL into key -> record (last wins) preserving first-seen order."""
    by_key: dict[str, dict] = {}
    order: list[str] = []
    if not path.exists():
        return by_key, order
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _record_key(record)
            if key not in by_key:
                order.append(key)
            by_key[key] = record
    return by_key, order


def _existing_profile(
    record: dict,
    output_index: dict[str, dict],
    key: str,
) -> dict[str, Any]:
    if key in output_index:
        profile = output_index[key].get("metadata", {}).get("style_profile", {})
        if isinstance(profile, dict):
            return profile
    profile = record.get("metadata", {}).get("style_profile", {})
    return profile if isinstance(profile, dict) else {}


def _should_skip(
    key: str,
    record: dict,
    output_index: dict[str, dict],
    pass_mode: str,
    resume: bool,
) -> bool:
    if not resume:
        return False
    from tools.style_classification.pass_config import PassMode, pass_complete

    profile = _existing_profile(record, output_index, key)
    mode: PassMode = pass_mode if pass_mode in ("full", "fast", "deep", "both") else "full"
    if mode == "full":
        return bool(profile)
    return pass_complete(profile, mode)


def _rewrite_output(path: Path, by_key: dict[str, dict], order: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for key in order:
            record = by_key.get(key)
            if record is not None:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _default_run_log_path(output_path: Path) -> Path:
    try:
        output_path.resolve().relative_to(ROOT)
    except ValueError:
        return output_path.with_suffix(output_path.suffix + ".events.jsonl")
    return RUN_LOG_DIR / f"{output_path.stem}.events.jsonl"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _append_run_log(log_path: Path | None, event: dict[str, Any]) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        **event,
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _merge_output_from_disk(
    path: Path,
    by_key: dict[str, dict],
    order: list[str],
) -> tuple[int, int]:
    """Merge on-disk rows missing from memory before compact rewrite.

    Keep in-memory values when a key already exists (this session's results win).
    Only add keys that exist on disk but not in memory (e.g. prior-run rows).
    """
    disk_index, disk_order = _load_output_index(path)
    before = len(order)
    for key in disk_order:
        if key in by_key:
            continue
        order.append(key)
        by_key[key] = disk_index[key]
    return before, len(order)


def _enrich(
    record: dict,
    rubric: dict | None,
    use_llm: bool,
    llm_model: str,
    pass_mode: str,
    prior_profile: dict[str, Any] | None,
    llm_mode: str = "joint",
    council_arbitrate: bool = True,
    field_batch_size: int = 3,
) -> dict:
    text = record.get("text", "")
    from tools.data_preparation.unified_corpus import normalize_prose_text

    text = normalize_prose_text(text)
    if not text or len(text.split()) < 30:
        return record

    try:
        from tools.style_classification.classify_passage import classify

        council_meta: dict[str, Any] = {}
        profile = classify(
            text,
            rubric=rubric,
            use_llm=use_llm,
            llm_model=llm_model,
            pass_mode=pass_mode,
            prior_profile=prior_profile,
            llm_mode=llm_mode,
            council_meta_out=council_meta if llm_mode == "council" else None,
            council_arbitrate=council_arbitrate,
            field_batch_size=field_batch_size,
        )
        out = dict(record)
        out["text"] = text
        meta = dict(out.get("metadata", {}))
        meta["word_count"] = len(text.split())
        meta["style_profile"] = profile
        if council_meta:
            existing = dict(meta.get("style_council", {}))
            existing.update(council_meta)
            meta["style_council"] = existing
        out["metadata"] = meta
        return out
    except Exception as exc:
        sys.stderr.write(f"  Warning: classification error: {exc}\n")
        return record


def run(
    input_path: Path,
    output_path: Path,
    use_llm: bool = True,
    llm_model: str = "llama3.1:8b",
    llm_sample_rate: float = 1.0,
    workers: int = 1,
    limit: int | None = None,
    resume: bool = True,
    seed: int = 42,
    quiet: bool = False,
    pass_mode: str = "full",
    llm_mode: str = "joint",
    council_arbitrate: bool = True,
    field_batch_size: int = 3,
    run_log_path: Path | None = None,
    run_log_enabled: bool = True,
    target_words: int = CHUNK_WORDS,
    no_rechunk: bool = False,
) -> None:
    random.seed(seed)
    run_id = uuid4().hex
    if not run_log_enabled:
        run_log_path = None
    elif run_log_path is None:
        run_log_path = _default_run_log_path(output_path)

    from tools.style_classification.pass_config import suggested_workers

    # Load rubric (optional; enhances LLM prompt context)
    rubric: dict | None = None
    rubric_path = ROOT / "source" / "style_rubric.json"
    if rubric_path.exists():
        from tools.style_classification.classify_passage import load_rubric

        rubric = load_rubric(rubric_path)
        print(f"Rubric loaded: {len(rubric.get('dimensions', []))} dimensions")
    else:
        print("No rubric found — run extract_rubric.py first for best results")

    print(f"Pass mode: {pass_mode}")
    if llm_mode == "joint":
        if field_batch_size and field_batch_size > 0:
            print(f"Field batching: ≤{field_batch_size} labels per LLM call")
        else:
            print("Field batching: off (one JSON call per pass)")
    if llm_mode == "council":
        arb = "with arbitrator on split votes" if council_arbitrate else "majority only"
        print(f"LLM mode: council (single-metric, 3-judge, {arb})")
    else:
        print(f"LLM mode: {llm_mode}")
    hint = suggested_workers(pass_mode if pass_mode in ("full", "fast", "deep", "both") else "full")
    if use_llm and hint and workers == 1:
        print(f"Tip: --pass {pass_mode} often runs well with --workers {hint}")

    # Read input
    print(f"Reading {input_path} …")
    records: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if limit:
        records = records[:limit]

    # Auto-chunk any full-book records before classification (unless pre-sized)
    pre_chunk = len(records)
    if no_rechunk:
        print(f"Skipping re-chunk (--no-rechunk); treating {pre_chunk} rows as units")
    else:
        records = [
            chunk
            for r in records
            for chunk in _chunk_record(r, target_words=target_words)
        ]
        if len(records) != pre_chunk:
            print(
                f"Chunked {pre_chunk} records -> {len(records)} chunks "
                f"({target_words}-word, {CHUNK_OVERLAP_SENTENCES}-sentence overlap)"
            )
    print(f"Total records: {len(records)}")

    output_index, output_order = _load_output_index(output_path)
    skipped = sum(
        1 for r in records if _should_skip(_record_key(r), r, output_index, pass_mode, resume)
    )
    if skipped:
        print(f"Resuming — {skipped} already complete for pass={pass_mode}")

    work_items: list[tuple[str, dict]] = []
    for record in records:
        key = _record_key(record)
        if _should_skip(key, record, output_index, pass_mode, resume):
            continue
        work_items.append((key, record))

    print(f"Records to classify: {len(work_items)}")
    _append_run_log(
        run_log_path,
        {
            "event": "run_started",
            "run_id": run_id,
            "pid": os.getpid(),
            "input_path": _display_path(input_path),
            "output_path": _display_path(output_path),
            "pass_mode": pass_mode,
            "llm_mode": llm_mode,
            "council_arbitrate": council_arbitrate,
            "workers": workers,
            "use_llm": use_llm,
            "llm_model": llm_model,
            "resume": resume,
            "source_records": pre_chunk,
            "pipeline_chunks": len(records),
            "already_complete": skipped,
            "records_to_classify": len(work_items),
            "output_unique_at_start": len(output_order),
        },
    )
    if not work_items:
        print("Nothing to do.")
        _append_run_log(
            run_log_path,
            {
                "event": "nothing_to_do",
                "run_id": run_id,
                "already_complete": skipped,
                "pipeline_chunks": len(records),
                "output_unique": len(output_order),
            },
        )
        return

    # Decide which records get LLM analysis
    llm_indices: set[int] = set()
    if use_llm:
        if llm_sample_rate >= 1.0:
            llm_indices = set(range(len(work_items)))
        else:
            n = int(len(work_items) * llm_sample_rate)
            llm_indices = set(random.sample(range(len(work_items)), n))
        print(
            f"LLM analysis: {len(llm_indices)}/{len(work_items)} records "
            f"({100 * len(llm_indices) // max(len(work_items), 1)}%)"
        )
        if workers > 1:
            print(f"Parallel LLM workers: {workers}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress = _ProgressTracker(total=len(work_items))
    compact_at_end = pass_mode in ("deep", "both")
    interrupted = False

    def _log_classified(result: dict) -> None:
        interval, throughput, eta_sec = progress.mark_done()
        label = _record_label(result)
        hint = _profile_hint(result)
        suffix = f" | {hint}" if hint else ""
        print(f"[{progress.format_progress(interval, throughput, eta_sec)}] {label}{suffix}", flush=True)

    def _flush_progress() -> None:
        done, throughput, eta_sec = progress.snapshot()
        print(
            f"  {done:>6}/{progress.total} "
            f"| {throughput:>5.2f} rec/s "
            f"| ~{eta_sec / 60:>4.0f} min remaining",
            flush=True,
        )

    def _process_item(idx: int, key: str, record: dict) -> None:
        do_llm = use_llm and (idx in llm_indices)
        prior = _existing_profile(record, output_index, key) if pass_mode in ("deep", "both") else None
        result = _enrich(
            record,
            rubric,
            do_llm,
            llm_model,
            pass_mode,
            prior,
            llm_mode,
            council_arbitrate,
            field_batch_size,
        )
        _write_result(key, result)

    with open(output_path, "a", encoding="utf-8") as out_fh:
        write_lock = threading.Lock()

        def _write_result(key: str, result: dict) -> None:
            output_index[key] = result
            if key not in output_order:
                output_order.append(key)
            with write_lock:
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_fh.flush()
                if quiet:
                    progress.mark_done()
                    if progress.processed % (50 if use_llm else 200) == 0:
                        _flush_progress()
                else:
                    _log_classified(result)
                if progress.processed % RUN_LOG_EVERY == 0:
                    _append_run_log(
                        run_log_path,
                        {
                            "event": "progress",
                            "run_id": run_id,
                            "processed_this_run": progress.processed,
                            "records_to_classify": progress.total,
                            "remaining_this_run": progress.total - progress.processed,
                            "output_unique_in_memory": len(output_order),
                        },
                    )

        if workers > 1:
            pool = ThreadPoolExecutor(max_workers=workers)
            futures = {
                pool.submit(_process_item, i, key, record): i
                for i, (key, record) in enumerate(work_items)
            }
            try:
                for future in as_completed(futures):
                    future.result()
            except KeyboardInterrupt:
                interrupted = True
                _append_run_log(
                    run_log_path,
                    {
                        "event": "interrupted",
                        "run_id": run_id,
                        "processed_this_run": progress.processed,
                        "records_to_classify": progress.total,
                        "output_unique_in_memory": len(output_order),
                    },
                )
                print(
                    "\nInterrupted — cancelling pending tasks "
                    "(in-flight LLM calls may finish briefly) …",
                    flush=True,
                )
                pool.shutdown(wait=False, cancel_futures=True)
            else:
                pool.shutdown(wait=True)
        else:
            try:
                for i, (key, record) in enumerate(work_items):
                    _process_item(i, key, record)
            except KeyboardInterrupt:
                interrupted = True
                _append_run_log(
                    run_log_path,
                    {
                        "event": "interrupted",
                        "run_id": run_id,
                        "processed_this_run": progress.processed,
                        "records_to_classify": progress.total,
                        "output_unique_in_memory": len(output_order),
                    },
                )
                print("\nInterrupted.", flush=True)

    if interrupted:
        if output_index:
            before_merge, after_merge = _merge_output_from_disk(output_path, output_index, output_order)
            _rewrite_output(output_path, output_index, output_order)
            print(f"Compacted {output_path} ({len(output_order)} unique records)")
            _append_run_log(
                run_log_path,
                {
                    "event": "compacted_after_interrupt",
                    "run_id": run_id,
                    "processed_this_run": progress.processed,
                    "records_to_classify": progress.total,
                    "output_unique_before_disk_merge": before_merge,
                    "output_unique_after_disk_merge": after_merge,
                    "output_unique_after_compact": len(output_order),
                },
            )
        print(
            f"\nStopped after {progress.processed}/{progress.total} records this session."
        )
        print("Resume: rerun the same command (skips chunks already complete).")
        if not output_index:
            print("No new records were written.")
        else:
            print(
                "If duplicate lines remain from earlier append-only runs:\n"
                f"  python tools/data_preparation/dedup_corpus_jsonl.py "
                f"--input {output_path} --in-place"
            )
        raise SystemExit(130)

    if compact_at_end:
        before_merge, after_merge = _merge_output_from_disk(output_path, output_index, output_order)
        _rewrite_output(output_path, output_index, output_order)
        print(f"Compacted {output_path} ({len(output_order)} records)")
        _append_run_log(
            run_log_path,
            {
                "event": "compacted_after_completion",
                "run_id": run_id,
                "output_unique_before_disk_merge": before_merge,
                "output_unique_after_disk_merge": after_merge,
                "output_unique_after_compact": len(output_order),
            },
        )

    elapsed = time.time() - progress.t0
    print(f"\nDone. {progress.processed} records in {elapsed / 60:.1f} min -> {output_path}")
    _append_run_log(
        run_log_path,
        {
            "event": "completed",
            "run_id": run_id,
            "processed_this_run": progress.processed,
            "records_to_classify": progress.total,
            "elapsed_seconds": round(elapsed, 3),
            "output_unique": len(output_order),
        },
    )
    print("Next: python tools/training_formats/generate_instruction_pairs.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Style-classify a JSONL corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-llm", action="store_true", help="Computable metrics only (fast)")
    parser.add_argument("--model", default=None, help="Model name (defaults to LLM_MODEL env var)")
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM API base URL (default: LLM_BASE_URL or localhost:1234/v1)",
    )
    parser.add_argument(
        "--llm-sample-rate",
        type=float,
        default=1.0,
        help="Fraction of records to run LLM on (0.0–1.0). Default: 1.0",
    )
    parser.add_argument(
        "--pass",
        dest="pass_mode",
        choices=("full", "fast", "deep", "both"),
        default="full",
        help="LLM pass: fast (pass 1 fields), deep (pass 2 fields), both (1+2 same model), full (all fields one call)",
    )
    parser.add_argument(
        "--llm-mode",
        dest="llm_mode",
        choices=("joint", "council"),
        default="joint",
        help="joint (batched JSON calls of ≤3 labels) or council (single-metric 3-judge vote per field)",
    )
    parser.add_argument(
        "--field-batch-size",
        type=int,
        default=3,
        metavar="N",
        help="Joint mode: max labels per LLM call (default 3). Use 0 for one call per pass (legacy).",
    )
    parser.add_argument(
        "--no-arbitrate",
        dest="council_arbitrate",
        action="store_false",
        default=True,
        help="Council only: skip the arbitrator on split votes (plain majority)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel threads. Typical: 4 for --pass fast|both, 2 for --pass deep (match LM Studio slots).",
    )
    parser.add_argument("--limit", type=int, help="Process only first N records (for testing)")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite output")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Log every 50/200 records instead of each classified chunk",
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        default=None,
        help="Append structured run events to this JSONL file (default: train/incremental/logs/<output>.events.jsonl)",
    )
    parser.add_argument(
        "--no-run-log",
        action="store_true",
        help="Disable structured run event logging",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-words",
        type=int,
        default=CHUNK_WORDS,
        help=f"Sentence-boundary chunk size when re-chunking (default: {CHUNK_WORDS})",
    )
    parser.add_argument(
        "--no-rechunk",
        action="store_true",
        help="Treat each input row as one unit (for prebuilt multigrain / sized segments)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.base_url:
        import tools.llm_client as _lc

        _lc.DEFAULT_BASE_URL = args.base_url

    from tools.llm_client import DEFAULT_MODEL

    model = args.model or DEFAULT_MODEL

    if args.no_resume and args.output.exists():
        args.output.unlink()

    run(
        input_path=args.input,
        output_path=args.output,
        use_llm=not args.no_llm,
        llm_model=model,
        llm_sample_rate=args.llm_sample_rate,
        workers=args.workers,
        limit=args.limit,
        resume=not args.no_resume,
        seed=args.seed,
        quiet=args.quiet,
        pass_mode=args.pass_mode,
        llm_mode=args.llm_mode,
        council_arbitrate=args.council_arbitrate,
        field_batch_size=args.field_batch_size,
        run_log_path=args.run_log,
        run_log_enabled=not args.no_run_log,
        target_words=args.target_words,
        no_rechunk=args.no_rechunk,
    )


if __name__ == "__main__":
    main()
