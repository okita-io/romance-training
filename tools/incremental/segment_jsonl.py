"""Split JSONL files into size-bounded segments (record boundaries preserved)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class SegmentInfo:
    index: int
    path: Path
    bytes: int
    rows: int


def iter_jsonl_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def segment_jsonl(
    input_path: Path,
    output_dir: Path,
    *,
    max_bytes: int = 50 * 1024 * 1024,
    prefix: str = "seg",
) -> list[SegmentInfo]:
    """Write JSONL segments of at most max_bytes each (whole records only)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    segments: list[SegmentInfo] = []
    seg_index = 0
    current_bytes = 0
    current_rows = 0
    out_fh = None
    out_path: Path | None = None

    def flush_segment() -> None:
        nonlocal out_fh, out_path, current_bytes, current_rows, seg_index
        if out_fh is None:
            return
        out_fh.close()
        out_fh = None
        segments.append(
            SegmentInfo(seg_index, out_path, current_bytes, current_rows)  # type: ignore[arg-type]
        )
        seg_index += 1
        current_bytes = 0
        current_rows = 0

    def start_segment() -> None:
        nonlocal out_fh, out_path
        out_path = output_dir / f"{prefix}_{seg_index:03d}.jsonl"
        out_fh = out_path.open("w", encoding="utf-8")

    for line in iter_jsonl_lines(input_path):
        payload = line + "\n"
        line_bytes = len(payload.encode("utf-8"))

        if out_fh is None:
            start_segment()
        elif current_bytes > 0 and current_bytes + line_bytes > max_bytes:
            flush_segment()
            start_segment()

        assert out_fh is not None
        out_fh.write(payload)
        current_bytes += line_bytes
        current_rows += 1

    flush_segment()
    return segments


def count_jsonl_rows(path: Path) -> int:
    return sum(1 for _ in iter_jsonl_lines(path))
