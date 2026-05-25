"""Central corpus paths for romance-training (re-exported for scripts and configs)."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from data_preparation.paths import (  # noqa: E402
    CORPUS_ROOT,
    FINAL_COMBINED_TRAIN,
    FINAL_COMBINED_VAL,
    IN_REPO_CORPUS,
    PROCESSED,
    REPO_ROOT,
    SOURCES,
    TRAINING,
    fiction1b_source,
    project_gutenberg_source,
)

DEFAULT_TRAIN = TRAINING["youtube_combined"] / "train.jsonl"
DEFAULT_VAL = TRAINING["youtube_combined"] / "validation.jsonl"
FICTION1B_ROOT = SOURCES["fiction1b_enhanced"]
HEAT_SUBSETS_ROOT = CORPUS_ROOT / "training" / "subsets" / "by_heat"
