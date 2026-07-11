"""Optional terminal training charts for Unsloth SFT runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import TrainerCallback


def parse_report_to(raw: str) -> tuple[str, bool]:
    """Split Hugging Face report_to from optional ``cli`` chart logging."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    use_cli = "cli" in parts
    hf_parts = [p for p in parts if p != "cli"]
    if hf_parts:
        return hf_parts[0] if len(hf_parts) == 1 else ",".join(hf_parts), use_cli
    return ("none" if use_cli else raw), use_cli


class CliChartLoggerCallback(TrainerCallback):
    """Log loss/LR to terminal charts when ``cli-charts`` is installed."""

    def __init__(self, output_dir: str | Path) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self._callback: TrainerCallback | None = None
        try:
            from cli_charts import TrainingChartCallback  # type: ignore

            self._callback = TrainingChartCallback(
                save_dir=str(self.output_dir / "cli_charts")
            )
        except Exception:
            self._callback = None

    def _delegate(self, event: str, *args: Any, **kwargs: Any) -> None:
        if self._callback is not None and hasattr(self._callback, event):
            getattr(self._callback, event)(*args, **kwargs)

    def on_train_begin(self, *args: Any, **kwargs: Any) -> None:
        self._delegate("on_train_begin", *args, **kwargs)

    def on_log(self, *args: Any, **kwargs: Any) -> None:
        self._delegate("on_log", *args, **kwargs)

    def on_train_end(self, *args: Any, **kwargs: Any) -> None:
        self._delegate("on_train_end", *args, **kwargs)
