"""Classification progress for manual or ledger-backed Phase 2 runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ClassificationProgress:
    corpus: str | None
    segment_index: int | None
    input_path: Path
    output_path: Path | None
    pass_mode: str
    source_rows: int
    pipeline_chunks: int
    complete: int
    pending: int
    output_lines: int
    output_unique: int
    output_duplicates: int

    @property
    def percent_complete(self) -> float:
        if self.pipeline_chunks <= 0:
            return 0.0
        return 100.0 * self.complete / self.pipeline_chunks


def default_manual_output(corpus: str, segment_index: int) -> Path:
    """Default styled path for direct ``run_pipeline.py`` segment runs."""
    return ROOT / "train" / "romance_corpus" / f"{corpus}_styled_seg_{segment_index:03d}.jsonl"


def resolve_output_path(
    corpus: str | None,
    segment_index: int | None,
    output: Path | None,
) -> Path | None:
    if output is not None:
        return output
    if corpus is not None and segment_index is not None:
        corpus_dir = ROOT / "train" / "romance_corpus"
        for name in (
            f"{corpus}_styled_seg_{segment_index:03d}.jsonl",
            f"{corpus}_deep_seg_{segment_index:03d}.jsonl",
        ):
            candidate = corpus_dir / name
            if candidate.is_file():
                return candidate
        styled = (
            ROOT
            / "train"
            / "incremental"
            / "segments"
            / corpus
            / "styled"
            / f"seg_{segment_index:03d}.jsonl"
        )
        if styled.is_file():
            return styled
    return None


def _iter_source_records(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def measure_classification_progress(
    input_path: Path,
    output_path: Path | None,
    *,
    pass_mode: str = "both",
    corpus: str | None = None,
    segment_index: int | None = None,
) -> ClassificationProgress:
    from tools.style_classification.pass_config import PassMode, pass_complete
    from tools.style_classification.run_pipeline import (
        _chunk_record,
        _existing_profile,
        _load_output_index,
        _record_key,
        _should_skip,
    )

    source_records = _iter_source_records(input_path)
    chunks = [chunk for record in source_records for chunk in _chunk_record(record)]

    output_index: dict[str, dict] = {}
    output_order: list[str] = []
    if output_path is not None and output_path.is_file():
        output_index, output_order = _load_output_index(output_path)

    mode: PassMode = pass_mode if pass_mode in ("full", "fast", "deep", "both") else "both"
    complete = 0
    for record in chunks:
        key = _record_key(record)
        if _should_skip(key, record, output_index, mode, resume=True):
            complete += 1

    output_lines = 0
    if output_path is not None and output_path.is_file():
        with output_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    output_lines += 1

    return ClassificationProgress(
        corpus=corpus,
        segment_index=segment_index,
        input_path=input_path,
        output_path=output_path,
        pass_mode=mode,
        source_rows=len(source_records),
        pipeline_chunks=len(chunks),
        complete=complete,
        pending=len(chunks) - complete,
        output_lines=output_lines,
        output_unique=len(output_index),
        output_duplicates=max(0, output_lines - len(output_index)),
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def format_progress_report(progress: ClassificationProgress) -> str:
    lines: list[str] = []
    title_parts: list[str] = []
    if progress.corpus:
        title_parts.append(progress.corpus)
    if progress.segment_index is not None:
        title_parts.append(f"seg_{progress.segment_index:03d}")
    title = " - ".join(title_parts) if title_parts else progress.input_path.name

    lines.append(title)
    lines.append(f"  input:  {_display_path(progress.input_path)} ({progress.source_rows:,} source rows)")
    if progress.output_path:
        if progress.output_path.is_file():
            lines.append(
                f"  output: {_display_path(progress.output_path)} "
                f"({progress.output_unique:,} unique"
                + (
                    f", {progress.output_duplicates:,} duplicate lines"
                    if progress.output_duplicates
                    else ""
                )
                + ")"
            )
        else:
            lines.append(f"  output: {_display_path(progress.output_path)} (not created yet)")
    else:
        lines.append("  output: (none)")

    lines.append(f"  pass {progress.pass_mode}: {progress.complete:,}/{progress.pipeline_chunks:,} complete")
    lines.append(
        f"  pending: {progress.pending:,} ({progress.percent_complete:.1f}% done)"
    )
    if progress.output_duplicates:
        lines.append(
            "  tip: python tools/data_preparation/dedup_corpus_jsonl.py "
            f"--input {progress.output_path} --in-place"
        )
    return "\n".join(lines)
