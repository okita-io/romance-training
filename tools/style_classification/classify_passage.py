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
    pass_mode: str = "full",
    prior_profile: dict[str, Any] | None = None,
    llm_mode: str = "joint",
    council_meta_out: dict[str, Any] | None = None,
    council_arbitrate: bool = True,
    field_batch_size: int = 3,
) -> dict[str, Any]:
    """
    Return a full style_profile for the given text passage.

    Args:
        text:           Prose passage to classify.
        rubric:         Pre-loaded rubric dict (loaded from disk if None).
        use_llm:        Run LLM-based semantic metrics (slower; ~2-5s per passage).
        llm_model:      Ollama / LM Studio model name (defaults to LLM_MODEL env).
        pass_mode:      "full" | "fast" | "deep" | "both" — see source/multi-pass.md.
        prior_profile:  Existing style_profile for deep/both merge (pass 1 labels).
        llm_mode:       "joint" (batched JSON calls) or "council" (single-metric
                        3-judge vote + arbitration per field).
        council_meta_out: When provided and llm_mode="council", populated with the
                        per-field vote breakdown (mutated in place).
        council_arbitrate: When council, invoke the arbitrator on split votes
                        (default True). False falls back to plain majority.
        field_batch_size: Joint mode only. ``>=3`` uses semantic batches of ≤3
                        labels; ``0`` = one LLM call per pass (legacy).

    Returns:
        Flat dict of all computed metrics.
    """
    from tools.llm_client import DEFAULT_MODEL
    from tools.style_classification.metrics_computable import compute
    from tools.style_classification.pass_config import (
        ALL_LLM_FIELDS,
        DEFAULT_FIELD_BATCH_SIZE,
        PASS1_LLM_FIELDS,
        PassMode,
        batches_for_pass,
        fields_for_pass,
        pass_complete,
    )

    if llm_model is None:
        llm_model = DEFAULT_MODEL
    if field_batch_size is None:
        field_batch_size = DEFAULT_FIELD_BATCH_SIZE

    mode: PassMode = pass_mode if pass_mode in ("full", "fast", "deep", "both") else "full"
    prior = prior_profile or {}

    profile: dict[str, Any] = compute(text)

    if not use_llm:
        if prior:
            profile.update({k: v for k, v in prior.items() if k not in profile or k in PASS1_LLM_FIELDS})
        return profile

    from tools.style_classification.metrics_llm import assess

    if rubric is None:
        try:
            rubric = load_rubric()
        except FileNotFoundError:
            rubric = None

    council = llm_mode == "council"
    if council:
        from tools.style_classification.metric_council import assess_fields_council

    def _get_llm(pass_key: PassMode, prior_fields: dict[str, Any] | None) -> dict[str, Any]:
        """Fetch LLM fields for one pass via joint (batched) or council dispatch."""
        if council:
            fields = fields_for_pass(pass_key) or ALL_LLM_FIELDS
            partial, meta = assess_fields_council(
                text,
                fields,
                model=llm_model,
                rubric=rubric,
                prior=prior_fields,
                arbitrate_split=council_arbitrate,
            )
            if council_meta_out is not None:
                council_meta_out.update(meta)
            return partial

        # Joint: one or more small JSON calls (≤3 fields when batching on).
        merged: dict[str, Any] = {}
        running_prior = dict(prior_fields or {})
        for batch in batches_for_pass(pass_key, field_batch_size=field_batch_size):
            partial = assess(
                text,
                model=llm_model,
                rubric=rubric,
                pass_mode=pass_key,
                fields=batch,
                prior=running_prior if running_prior else None,
            )
            merged.update(partial)
            for key, val in partial.items():
                if key != "evidence" and val is not None:
                    running_prior[key] = val
        return merged

    if mode == "both":
        if prior:
            profile.update({k: v for k, v in prior.items() if v is not None})

        if not pass_complete(profile, "fast"):
            profile.update(_get_llm("fast", None))

        if not pass_complete(profile, "deep"):
            prior_for_deep = {k: profile[k] for k in PASS1_LLM_FIELDS if k in profile}
            profile.update(_get_llm("deep", prior_for_deep))

        if prior:
            profile.update({k: v for k, v in prior.items() if k in PASS1_LLM_FIELDS and v is not None})
        return profile

    if mode == "deep" and prior:
        profile.update({k: v for k, v in prior.items() if k not in profile})

    llm_fields = _get_llm(mode, prior if mode == "deep" else None)
    profile.update(llm_fields)

    if mode == "deep" and prior:
        profile.update({k: v for k, v in prior.items() if k in PASS1_LLM_FIELDS})

    return profile


def _main() -> None:
    parser = argparse.ArgumentParser(description="Classify a prose passage for style")
    parser.add_argument("--file", type=Path, help="Text file to classify (default: stdin)")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--model", default=None, help="LLM model (default: LLM_MODEL env)")
    parser.add_argument(
        "--pass",
        dest="pass_mode",
        choices=("full", "fast", "deep", "both"),
        default="full",
        help="LLM pass mode (default: full). both = fast+deep field sets, same model",
    )
    parser.add_argument(
        "--llm-mode",
        dest="llm_mode",
        choices=("joint", "council"),
        default="joint",
        help="joint = batched JSON calls; council = single-metric 3-judge vote per field",
    )
    parser.add_argument(
        "--field-batch-size",
        type=int,
        default=3,
        metavar="N",
        help="Joint mode: max labels per LLM call (default 3). 0 = one call per pass",
    )
    parser.add_argument(
        "--no-arbitrate",
        dest="arbitrate",
        action="store_false",
        default=True,
        help="Council only: skip the arbitrator on split votes (plain majority)",
    )
    args = parser.parse_args()

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    from tools.llm_client import DEFAULT_MODEL

    council_meta: dict[str, Any] = {}
    profile = classify(
        text,
        use_llm=not args.no_llm,
        llm_model=args.model or DEFAULT_MODEL,
        pass_mode=args.pass_mode,
        llm_mode=args.llm_mode,
        council_meta_out=council_meta if args.llm_mode == "council" else None,
        council_arbitrate=args.arbitrate,
        field_batch_size=args.field_batch_size,
    )
    print(json.dumps(profile, indent=2))
    if council_meta:
        print(json.dumps({"style_council": council_meta}, indent=2), file=sys.stderr)


if __name__ == "__main__":
    _main()
