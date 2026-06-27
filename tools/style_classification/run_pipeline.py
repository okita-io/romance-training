#!/usr/bin/env python3
"""
Phase 2: Run the style classification pipeline over a JSONL corpus.

Reads JSONL (text + metadata), adds style_profile to metadata, writes enriched JSONL.
Supports resuming interrupted runs — already-processed records are skipped.

Usage:
    # Classify existing Gutenberg romance corpus (fast, no LLM)
    python tools/style_classification/run_pipeline.py --no-llm

    # Full classification with LLM on all records
    python tools/style_classification/run_pipeline.py

    # LLM on a 20% sample (good balance of speed vs. coverage)
    python tools/style_classification/run_pipeline.py --llm-sample-rate 0.2

    # Custom input/output
    python tools/style_classification/run_pipeline.py \\
        --input data/corpus/sources/project_gutenberg/train.jsonl \\
        --output data/corpus/training/processed/gutenberg_styled.jsonl

    # Parallel workers (computable-only mode)
    python tools/style_classification/run_pipeline.py --no-llm --workers 8

    # Parallel LLM requests (LM Studio with multiple slots, e.g. --workers 4)
    python tools/style_classification/run_pipeline.py --workers 4 \\
        --input source-data/processed/horror_novel_chunks/chunks.jsonl \\
        --output train/romance_corpus/horror_styled.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "train" / "romance_corpus" / "gutenberg_romance.jsonl"
DEFAULT_OUTPUT = ROOT / "train" / "romance_corpus" / "gutenberg_styled.jsonl"

CHUNK_WORDS = 500
CHUNK_OVERLAP_SENTENCES = 2
THROUGHPUT_WINDOW = 30


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


def _chunk_record(record: dict) -> list[dict]:
    """Split a long record into ~500-word sentence-boundary chunks."""
    from tools.style_classification.chunk_text import chunk_record

    return chunk_record(
        record,
        target_words=CHUNK_WORDS,
        overlap_sentences=CHUNK_OVERLAP_SENTENCES,
    )


def _record_key(record: dict) -> str:
    m = record.get("metadata", {})
    source = m.get("source", record.get("source", ""))
    chunk = m.get("chunk_index", 0)
    text_sig = record.get("text", "")[:60]
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
    return ""


def _enrich(
    record: dict,
    rubric: dict | None,
    use_llm: bool,
    llm_model: str,
) -> dict:
    text = record.get("text", "")
    if not text or len(text.split()) < 30:
        return record

    try:
        from tools.style_classification.classify_passage import classify
        profile = classify(text, rubric=rubric, use_llm=use_llm, llm_model=llm_model)
        out = dict(record)
        meta = dict(out.get("metadata", {}))
        meta["style_profile"] = profile
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
) -> None:
    random.seed(seed)

    # Load rubric (optional; enhances LLM prompt context)
    rubric: dict | None = None
    rubric_path = ROOT / "source" / "style_rubric.json"
    if rubric_path.exists():
        from tools.style_classification.classify_passage import load_rubric
        rubric = load_rubric(rubric_path)
        print(f"Rubric loaded: {len(rubric.get('dimensions', []))} dimensions")
    else:
        print("No rubric found — run extract_rubric.py first for best results")

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

    # Auto-chunk any full-book records before classification
    pre_chunk = len(records)
    records = [chunk for r in records for chunk in _chunk_record(r)]
    if len(records) != pre_chunk:
        print(f"Chunked {pre_chunk} records -> {len(records)} chunks ({CHUNK_WORDS}-word, {CHUNK_OVERLAP_SENTENCES}-sentence overlap)")
    print(f"Total records: {len(records)}")

    # Resume: collect already-processed keys
    done_keys: set[str] = set()
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r.get("metadata", {}).get("style_profile"):
                            done_keys.add(_record_key(r))
                    except json.JSONDecodeError:
                        pass
        if done_keys:
            print(f"Resuming — {len(done_keys)} already classified")

    to_process = [r for r in records if _record_key(r) not in done_keys]
    print(f"Records to classify: {len(to_process)}")
    if not to_process:
        print("Nothing to do.")
        return

    # Decide which records get LLM analysis
    llm_indices: set[int] = set()
    if use_llm:
        if llm_sample_rate >= 1.0:
            llm_indices = set(range(len(to_process)))
        else:
            n = int(len(to_process) * llm_sample_rate)
            llm_indices = set(random.sample(range(len(to_process)), n))
        print(
            f"LLM analysis: {len(llm_indices)}/{len(to_process)} records "
            f"({100 * len(llm_indices) // len(to_process)}%)"
        )
        if workers > 1:
            print(f"Parallel LLM workers: {workers}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress = _ProgressTracker(total=len(to_process))

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

    with open(output_path, "a", encoding="utf-8") as out_fh:
        write_lock = threading.Lock()

        def _write_result(result: dict) -> None:
            with write_lock:
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                if quiet:
                    progress.mark_done()
                    if progress.processed % (50 if use_llm else 200) == 0:
                        _flush_progress()
                else:
                    _log_classified(result)
                out_fh.flush()

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _enrich,
                        record,
                        rubric,
                        use_llm and (i in llm_indices),
                        llm_model,
                    ): i
                    for i, record in enumerate(to_process)
                }
                for future in as_completed(futures):
                    _write_result(future.result())
        else:
            for i, record in enumerate(to_process):
                do_llm = use_llm and (i in llm_indices)
                _write_result(_enrich(record, rubric, do_llm, llm_model))

    elapsed = time.time() - progress.t0
    print(f"\nDone. {progress.processed} records in {elapsed / 60:.1f} min -> {output_path}")
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
    parser.add_argument("--base-url", default=None, help="LLM API base URL (default: LLM_BASE_URL or localhost:1234/v1)")
    parser.add_argument(
        "--llm-sample-rate", type=float, default=1.0,
        help="Fraction of records to run LLM on (0.0–1.0). Default: 1.0",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel threads (LLM or computable). Match LM Studio concurrent slots (e.g. 4).",
    )
    parser.add_argument("--limit", type=int, help="Process only first N records (for testing)")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite output")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Log every 50/200 records instead of each classified chunk",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Apply base-url override before any LLM calls
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
    )


if __name__ == "__main__":
    main()
