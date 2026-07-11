"""Helpers for Hugging Face Trainer checkpoint discovery and resume."""

from __future__ import annotations

import json
import re
from pathlib import Path

_CHECKPOINT_DIR = re.compile(r"^checkpoint-(\d+)(?:-(?:emergency|oom))?$")


def _checkpoint_step_from_name(name: str) -> int | None:
    match = _CHECKPOINT_DIR.match(name)
    if not match:
        return None
    return int(match.group(1))


def _has_model_weights(checkpoint_dir: Path) -> bool:
    return any(
        (checkpoint_dir / fname).is_file()
        for fname in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "pytorch_model.bin",
            "model.safetensors",
        )
    )


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the highest-step checkpoint directory under output_dir, if any."""
    root = Path(output_dir)
    if not root.is_dir():
        return None

    best: tuple[int, Path] | None = None
    for path in root.iterdir():
        if not path.is_dir():
            continue
        step = _checkpoint_step_from_name(path.name)
        if step is None or not _has_model_weights(path):
            continue
        if best is None or step > best[0]:
            best = (step, path)
    return best[1] if best else None


def checkpoint_global_step(checkpoint_dir: str | Path) -> int | None:
    """Read global_step from trainer_state.json, else parse checkpoint-N name."""
    path = Path(checkpoint_dir)
    state_file = path / "trainer_state.json"
    if state_file.is_file():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        else:
            step = data.get("global_step")
            if isinstance(step, int):
                return step
    return _checkpoint_step_from_name(path.name)


def resolve_resume_checkpoint(
    resume: bool | str | Path | None,
    output_dir: str | Path,
) -> str | None:
    """
    Normalize --resume handling.

    - None -> fresh training
    - True -> latest checkpoint under output_dir
    - path -> explicit checkpoint directory
    """
    if resume is None:
        return None
    if resume is True:
        latest = find_latest_checkpoint(output_dir)
        return str(latest.resolve()) if latest is not None else None

    path = Path(resume)
    if not path.is_dir():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    if not _has_model_weights(path):
        raise FileNotFoundError(
            f"resume checkpoint missing model weights: {path}"
        )
    return str(path.resolve())
