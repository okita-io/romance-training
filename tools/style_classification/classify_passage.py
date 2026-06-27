"""
Classify a single prose passage — combines computable and LLM metrics
into a unified style_profile dict.

Usage as a library:
    from tools.style_classification.classify_passage import classify, load_rubric
    profile = classify("It was a dark and stormy night...")

Usage as a script:
    echo "Your passage here." | python tools/style_classification/classify_passage.py
    python tools/style_classification/classify_passage.py --file passage.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUBRIC_PATH = ROOT / "source" / "style_rubric.json"


def load_rubric(path: Path = RUBRIC_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Rubric not found at {path}.\n"
            "Run: python tools/style_extraction/extract_rubric.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def classify(
    text: str,
    rubric: dict | None = None,
    use_llm: bool = True,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """
    Return a full style_profile for the given text passage.

    Args:
        text:      Prose passage to classify.
        rubric:    Pre-loaded rubric dict (loaded from disk if None).
        use_llm:   Run LLM-based semantic metrics (slower; ~2-5s per passage).
        llm_model: Ollama / LM Studio model name (defaults to LLM_MODEL env).

    Returns:
        Flat dict of all computed metrics.
    """
    from tools.llm_client import DEFAULT_MODEL
    from tools.style_classification.metrics_computable import compute

    if llm_model is None:
        llm_model = DEFAULT_MODEL

    profile: dict[str, Any] = compute(text)

    if use_llm:
        from tools.style_classification.metrics_llm import assess
        if rubric is None:
            try:
                rubric = load_rubric()
            except FileNotFoundError:
                rubric = None
        profile.update(assess(text, model=llm_model, rubric=rubric))

    return profile


def _main() -> None:
    parser = argparse.ArgumentParser(description="Classify a prose passage for style")
    parser.add_argument("--file", type=Path, help="Text file to classify (default: stdin)")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--model", default=None, help="LLM model (default: LLM_MODEL env)")
    args = parser.parse_args()

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    from tools.llm_client import DEFAULT_MODEL

    profile = classify(text, use_llm=not args.no_llm, llm_model=args.model or DEFAULT_MODEL)
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    _main()
