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
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "train" / "romance_corpus" / "gutenberg_romance.jsonl"
DEFAULT_OUTPUT = ROOT / "train" / "romance_corpus" / "gutenberg_styled.jsonl"

CHUNK_WORDS = 500
CHUNK_OVERLAP_SENTENCES = 2


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
        print(f"Chunked {pre_chunk} records → {len(records)} chunks ({CHUNK_WORDS}-word, {CHUNK_OVERLAP_SENTENCES}-sentence overlap)")
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    processed = 0

    def _flush_progress() -> None:
        elapsed = time.time() - t0
        rate = processed / max(elapsed, 0.01)
        remaining = (len(to_process) - processed) / max(rate, 0.001)
        print(
            f"  {processed:>6}/{len(to_process)} "
            f"| {rate:>5.1f} rec/s "
            f"| ~{remaining / 60:>4.0f} min remaining",
            flush=True,
        )

    with open(output_path, "a", encoding="utf-8") as out_fh:
        if workers > 1 and not use_llm:
            # Parallel path — computable metrics only, thread-safe
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_enrich, r, rubric, False, llm_model): i
                    for i, r in enumerate(to_process)
                }
                for future in as_completed(futures):
                    result = future.result()
                    out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed += 1
                    if processed % 200 == 0:
                        _flush_progress()
                        out_fh.flush()
        else:
            # Sequential path — required when Ollama is involved (single GPU)
            for i, record in enumerate(to_process):
                do_llm = use_llm and (i in llm_indices)
                result = _enrich(record, rubric, do_llm, llm_model)
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed += 1
                if processed % 100 == 0:
                    _flush_progress()
                    out_fh.flush()

    elapsed = time.time() - t0
    print(f"\nDone. {processed} records in {elapsed / 60:.1f} min → {output_path}")
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
    parser.add_argument("--workers", type=int, default=1, help="Threads (computable-only mode)")
    parser.add_argument("--limit", type=int, help="Process only first N records (for testing)")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite output")
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
    )


if __name__ == "__main__":
    main()
