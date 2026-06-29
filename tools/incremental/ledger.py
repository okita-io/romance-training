"""Incremental training ledger — segment and batch state."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]
INCREMENTAL_ROOT = ROOT / "train" / "incremental"
CORPORA_CONFIG = INCREMENTAL_ROOT / "corpora.json"
LEDGER_PATH = INCREMENTAL_ROOT / "ledger.json"
SEGMENTS_ROOT = INCREMENTAL_ROOT / "segments"
BATCHES_ROOT = INCREMENTAL_ROOT / "batches"

ClassificationStatus = Literal["pending", "in_progress", "classified", "failed"]
TrainingStatus = Literal["unavailable", "available", "allocated", "trained"]
BatchStatus = Literal["ready", "trained"]


@dataclass
class SegmentRecord:
    id: str
    corpus: str
    segment_index: int
    input_path: str | None = None
    styled_path: str | None = None
    bytes: int = 0
    rows: int = 0
    classification_status: ClassificationStatus = "pending"
    training_status: TrainingStatus = "unavailable"
    pass_fast: bool = False
    pass_deep: bool = False
    batch_id: str | None = None
    training_run_id: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class BatchRecord:
    id: str
    status: BatchStatus = "ready"
    max_mb_per_corpus: int = 50
    segments: dict[str, list[str]] = field(default_factory=dict)
    train_path: str = ""
    validation_path: str = ""
    training_run_id: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class TrainingRunRecord:
    id: str
    batch_id: str
    model_base: str = ""
    output_dir: str = ""
    notes: str = ""
    created_at: str = ""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_corpora_config() -> dict[str, Any]:
    return json.loads(CORPORA_CONFIG.read_text(encoding="utf-8"))


def corpus_input_path(corpus: str) -> Path:
    cfg = load_corpora_config()
    rel = cfg["corpora"][corpus]["input"]
    return ROOT / rel


def corpus_segments_dir(corpus: str, stage: str) -> Path:
    return SEGMENTS_ROOT / corpus / stage


def segment_id(corpus: str, index: int) -> str:
    return f"{corpus}/seg_{index:03d}"


class Ledger:
    def __init__(self, path: Path = LEDGER_PATH) -> None:
        self.path = path
        self.data: dict[str, Any] = {
            "version": 1,
            "updated_at": _now(),
            "segments": {},
            "batches": {},
            "training_runs": {},
        }
        if path.is_file():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = _now()
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

    @property
    def segments(self) -> dict[str, dict[str, Any]]:
        return self.data.setdefault("segments", {})

    @property
    def batches(self) -> dict[str, dict[str, Any]]:
        return self.data.setdefault("batches", {})

    @property
    def training_runs(self) -> dict[str, dict[str, Any]]:
        return self.data.setdefault("training_runs", {})

    def upsert_segment(self, record: SegmentRecord) -> None:
        record.updated_at = _now()
        if not record.created_at:
            record.created_at = record.updated_at
        self.segments[record.id] = asdict(record)

    def get_segment(self, seg_id: str) -> dict[str, Any] | None:
        return self.segments.get(seg_id)

    def register_input_segment(
        self,
        corpus: str,
        index: int,
        path: Path,
        *,
        bytes: int,
        rows: int,
    ) -> SegmentRecord:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        rec = SegmentRecord(
            id=segment_id(corpus, index),
            corpus=corpus,
            segment_index=index,
            input_path=rel,
            bytes=bytes,
            rows=rows,
            classification_status="pending",
            training_status="unavailable",
        )
        self.upsert_segment(rec)
        return rec

    def register_styled_segment(
        self,
        corpus: str,
        index: int,
        styled_path: Path,
        input_path: Path | None,
        *,
        bytes: int,
        rows: int,
        classified: bool = True,
    ) -> SegmentRecord:
        rec = SegmentRecord(
            id=segment_id(corpus, index),
            corpus=corpus,
            segment_index=index,
            input_path=str(input_path.relative_to(ROOT)).replace("\\", "/") if input_path else None,
            styled_path=str(styled_path.relative_to(ROOT)).replace("\\", "/"),
            bytes=bytes,
            rows=rows,
            classification_status="classified" if classified else "pending",
            training_status="available" if classified else "unavailable",
        )
        self.upsert_segment(rec)
        return rec

    def mark_classified(self, seg_id: str, styled_path: Path, *, pass_fast: bool, pass_deep: bool) -> None:
        seg = self.segments[seg_id]
        seg["styled_path"] = str(styled_path.relative_to(ROOT)).replace("\\", "/")
        seg["classification_status"] = "classified"
        seg["training_status"] = "available"
        seg["pass_fast"] = pass_fast
        seg["pass_deep"] = pass_deep
        seg["updated_at"] = _now()
        self.segments[seg_id] = seg

    def next_pending(self, corpus: str) -> dict[str, Any] | None:
        pending = [
            s for s in self.segments.values()
            if s.get("corpus") == corpus and s.get("classification_status") == "pending"
        ]
        if not pending:
            return None
        pending.sort(key=lambda s: s.get("segment_index", 0))
        return pending[0]

    def available_for_training(self, corpus: str) -> list[dict[str, Any]]:
        out = [
            s for s in self.segments.values()
            if s.get("corpus") == corpus
            and s.get("classification_status") == "classified"
            and s.get("training_status") == "available"
            and s.get("styled_path")
        ]
        out.sort(key=lambda s: s.get("segment_index", 0))
        return out

    def allocate_segments(self, corpus: str, seg_ids: list[str], batch_id: str) -> None:
        for seg_id in seg_ids:
            seg = self.segments[seg_id]
            seg["training_status"] = "allocated"
            seg["batch_id"] = batch_id
            seg["updated_at"] = _now()

    def create_batch(
        self,
        batch_id: str,
        *,
        max_mb_per_corpus: int,
        segments_by_corpus: dict[str, list[str]],
        train_path: Path,
        val_path: Path,
    ) -> BatchRecord:
        rec = BatchRecord(
            id=batch_id,
            max_mb_per_corpus=max_mb_per_corpus,
            segments=segments_by_corpus,
            train_path=str(train_path.relative_to(ROOT)).replace("\\", "/"),
            validation_path=str(val_path.relative_to(ROOT)).replace("\\", "/"),
            created_at=_now(),
            updated_at=_now(),
        )
        self.batches[batch_id] = asdict(rec)
        return rec

    def mark_batch_trained(self, batch_id: str, run_id: str, *, model_base: str = "", output_dir: str = "") -> None:
        batch = self.batches[batch_id]
        batch["status"] = "trained"
        batch["training_run_id"] = run_id
        batch["updated_at"] = _now()

        for corpus, seg_ids in batch.get("segments", {}).items():
            for sid in seg_ids:
                seg = self.segments.get(sid)
                if not seg:
                    continue
                seg["training_status"] = "trained"
                seg["training_run_id"] = run_id
                seg["updated_at"] = _now()

        self.training_runs[run_id] = asdict(
            TrainingRunRecord(
                id=run_id,
                batch_id=batch_id,
                model_base=model_base,
                output_dir=output_dir,
                created_at=_now(),
            )
        )

    def summary(self) -> dict[str, Any]:
        cfg = load_corpora_config()
        corpora = cfg.get("training_mix_corpora") or list(cfg["corpora"].keys())
        by_corpus: dict[str, dict[str, int]] = {}
        for slug in corpora:
            segs = [s for s in self.segments.values() if s.get("corpus") == slug]
            by_corpus[slug] = {
                "segments_total": len(segs),
                "classification_pending": sum(1 for s in segs if s.get("classification_status") == "pending"),
                "classification_in_progress": sum(1 for s in segs if s.get("classification_status") == "in_progress"),
                "classification_classified": sum(1 for s in segs if s.get("classification_status") == "classified"),
                "training_available": sum(1 for s in segs if s.get("training_status") == "available"),
                "training_allocated": sum(1 for s in segs if s.get("training_status") == "allocated"),
                "training_trained": sum(1 for s in segs if s.get("training_status") == "trained"),
                "bytes_classified": sum(s.get("bytes", 0) for s in segs if s.get("classification_status") == "classified"),
                "bytes_available": sum(
                    s.get("bytes", 0) for s in segs if s.get("training_status") == "available"
                ),
            }
        return {
            "ledger": str(self.path.relative_to(ROOT)).replace("\\", "/"),
            "batches_ready": sum(1 for b in self.batches.values() if b.get("status") == "ready"),
            "batches_trained": sum(1 for b in self.batches.values() if b.get("status") == "trained"),
            "training_runs": len(self.training_runs),
            "corpora": by_corpus,
        }
