"""Tests for training checkpoint discovery helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from romance_factory.training_checkpoints import (
    checkpoint_global_step,
    find_latest_checkpoint,
    resolve_resume_checkpoint,
)


def _make_checkpoint(root: Path, step: int, *, emergency: bool = False) -> Path:
    suffix = f"{step}-emergency" if emergency else str(step)
    ckpt = root / f"checkpoint-{suffix}"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_model.safetensors").write_bytes(b"x")
    (ckpt / "trainer_state.json").write_text(
        json.dumps({"global_step": step}),
        encoding="utf-8",
    )
    return ckpt


def test_find_latest_checkpoint_picks_highest_step(tmp_path: Path) -> None:
    out = tmp_path / "lora"
    out.mkdir()
    _make_checkpoint(out, 250)
    _make_checkpoint(out, 500)

    latest = find_latest_checkpoint(out)
    assert latest is not None
    assert latest.name == "checkpoint-500"


def test_find_latest_checkpoint_ignores_incomplete_dirs(tmp_path: Path) -> None:
    out = tmp_path / "lora"
    out.mkdir()
    incomplete = out / "checkpoint-999"
    incomplete.mkdir()
    _make_checkpoint(out, 250)

    latest = find_latest_checkpoint(out)
    assert latest is not None
    assert latest.name == "checkpoint-250"


def test_resolve_resume_true_uses_latest(tmp_path: Path) -> None:
    out = tmp_path / "lora"
    out.mkdir()
    ckpt = _make_checkpoint(out, 250)

    resolved = resolve_resume_checkpoint(True, out)
    assert resolved == str(ckpt.resolve())


def test_resolve_resume_true_returns_none_when_missing(tmp_path: Path) -> None:
    out = tmp_path / "lora"
    out.mkdir()
    assert resolve_resume_checkpoint(True, out) is None


def test_resolve_resume_explicit_path(tmp_path: Path) -> None:
    out = tmp_path / "lora"
    out.mkdir()
    ckpt = _make_checkpoint(out, 250)

    resolved = resolve_resume_checkpoint(str(ckpt), out)
    assert resolved == str(ckpt.resolve())


def test_resolve_resume_missing_path_raises(tmp_path: Path) -> None:
    out = tmp_path / "lora"
    out.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(out / "checkpoint-404", out)


def test_checkpoint_global_step_from_state(tmp_path: Path) -> None:
    ckpt = _make_checkpoint(tmp_path, 250)
    assert checkpoint_global_step(ckpt) == 250


def test_checkpoint_global_step_from_name_when_state_missing(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint-500"
    ckpt.mkdir()
    (ckpt / "adapter_model.safetensors").write_bytes(b"x")
    assert checkpoint_global_step(ckpt) == 500
