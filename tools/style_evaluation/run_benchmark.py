#!/usr/bin/env python3
"""
Run the Ashenmere style benchmark: generate scenes, classify, record deltas.

Pass 1 (optional): world / roster documentation prompt.
Pass 2 (default): 7 plots × 3 scene types = 21 generations with rubric-targeted style.

Each result line records model, prompts, generated text, classification, and delta vs target.

Usage:
    # Generate + classify (needs LLM for both generation and classification)
    python tools/style_evaluation/run_benchmark.py \\
        --label baseline_llama10b \\
        --output eval/style_benchmark/results/baseline_llama10b.jsonl

    # Prompts only (no LLM calls)
    python tools/style_evaluation/run_benchmark.py --dry-run

    # Classify existing generations from a prior run
    python tools/style_evaluation/run_benchmark.py \\
        --reclassify eval/style_benchmark/results/baseline_llama10b.jsonl

    # Compare two completed runs
    python tools/style_evaluation/run_benchmark.py compare \\
        eval/style_benchmark/results/baseline.jsonl \\
        eval/style_benchmark/results/batch001_finetuned.jsonl

    # Summarize one run
    python tools/style_evaluation/run_benchmark.py summarize \\
        eval/style_benchmark/results/baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.llm_client import DEFAULT_MODEL, complete  # noqa: E402
from tools.style_evaluation.benchmark import (  # noqa: E402
    DEFAULT_FIXTURE,
    aggregate_run,
    build_scene_prompt,
    build_setup_prompt,
    compare_runs,
    compute_delta,
    iter_benchmark_cases,
    load_fixture,
    load_results_jsonl,
    make_run_id,
)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _classify_text(text: str, *, model: str, pass_mode: str) -> dict[str, Any]:
    from tools.style_classification.classify_passage import classify, load_rubric

    rubric = load_rubric()
    return classify(
        text,
        rubric=rubric,
        use_llm=True,
        llm_model=model,
        pass_mode=pass_mode,
    )


def cmd_run(args: argparse.Namespace) -> None:
    fixture = load_fixture(Path(args.fixture))
    run_id = make_run_id(args.label)
    out_path = Path(args.output)
    plot_map = {p["id"]: p for p in fixture["plots"]}

    if args.setup_only:
        system, user = build_setup_prompt(fixture)
        record: dict[str, Any] = {
            "run_id": run_id,
            "label": args.label,
            "phase": "setup",
            "model": args.model,
            "classifier_model": args.classifier_model or args.model,
            "timestamp": time.time(),
            "prompt_system": system,
            "prompt_user": user,
        }
        if args.dry_run:
            record["generated_text"] = None
            print(json.dumps(record, indent=2))
            _append_jsonl(out_path, record)
            return

        record["generated_text"] = complete(user, system=system, model=args.model)
        _append_jsonl(out_path, record)
        print(f"Wrote setup record → {out_path}")
        return

    cases = iter_benchmark_cases(fixture)
    if args.plot:
        cases = [c for c in cases if c["plot_id"] == args.plot]
    if args.scene:
        cases = [c for c in cases if c["scene_type"] == args.scene]
    if args.limit:
        cases = cases[: args.limit]

    classifier = args.classifier_model or args.model
    print(f"Run {run_id}: {len(cases)} cases → {out_path}")
    print(f"Generator: {args.model} | Classifier: {classifier} (pass={args.classify_pass})")

    for i, case in enumerate(cases, 1):
        plot = plot_map[case["plot_id"]]
        system, user = build_scene_prompt(fixture, plot, case["scene_type"])

        record = {
            "run_id": run_id,
            "label": args.label,
            "phase": "scene",
            "plot_id": case["plot_id"],
            "plot_title": case["plot_title"],
            "scene_type": case["scene_type"],
            "scene_label": case["scene_label"],
            "style_label": case["style_label"],
            "style_target": case["style_target"],
            "female_lead": case["female_lead"],
            "male_lead": case["male_lead"],
            "model": args.model,
            "classifier_model": classifier,
            "classify_pass": args.classify_pass,
            "timestamp": time.time(),
            "prompt_system": system,
            "prompt_user": user,
        }

        if args.dry_run:
            record["generated_text"] = None
            print(f"[{i}/{len(cases)}] dry-run {case['plot_id']} / {case['scene_type']}")
            _append_jsonl(out_path, record)
            continue

        print(f"[{i}/{len(cases)}] generating {case['plot_id']} / {case['scene_type']} …")
        text = complete(user, system=system, model=args.model)
        record["generated_text"] = text

        if not args.no_classify and text.strip():
            print(f"    classifying …")
            profile = _classify_text(text, model=classifier, pass_mode=args.classify_pass)
            record["classification"] = profile
            record["delta"] = compute_delta(case["style_target"], profile)

        _append_jsonl(out_path, record)
        print(f"    → match {record.get('delta', {}).get('match_score', 'n/a')}")

    if not args.dry_run:
        summary = aggregate_run(load_results_jsonl(out_path))
        summary_path = out_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Done. Summary → {summary_path}")


def cmd_reclassify(args: argparse.Namespace) -> None:
    path = Path(args.reclassify)
    records = load_results_jsonl(path)
    classifier = args.classifier_model or args.model
    updated: list[dict[str, Any]] = []

    for i, rec in enumerate(records, 1):
        text = rec.get("generated_text") or ""
        if not text.strip():
            updated.append(rec)
            continue
        print(f"[{i}/{len(records)}] reclassify {rec.get('plot_id')} / {rec.get('scene_type')}")
        profile = _classify_text(text, model=classifier, pass_mode=args.classify_pass)
        rec = dict(rec)
        rec["classifier_model"] = classifier
        rec["classify_pass"] = args.classify_pass
        rec["classification"] = profile
        if rec.get("style_target"):
            rec["delta"] = compute_delta(rec["style_target"], profile)
        updated.append(rec)

    _write_jsonl(path, updated)
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(aggregate_run(updated), indent=2), encoding="utf-8")
    print(f"Updated {path}")


def cmd_compare(args: argparse.Namespace) -> None:
    baseline = load_results_jsonl(Path(args.baseline))
    candidate = load_results_jsonl(Path(args.candidate))
    report = compare_runs(baseline, candidate)
    out = Path(args.output) if args.output else None
    text = json.dumps(report, indent=2)
    if out:
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(text)


def cmd_summarize(args: argparse.Namespace) -> None:
    records = load_results_jsonl(Path(args.path))
    print(json.dumps(aggregate_run(records), indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ashenmere style benchmark runner")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Generate benchmark scenes (default command)")
    run.set_defaults(command="run")
    run.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    run.add_argument(
        "--output",
        default=str(ROOT / "eval" / "style_benchmark" / "results" / "latest.jsonl"),
    )
    run.add_argument("--label", default="benchmark", help="Tag for this run (baseline, batch_001, …)")
    run.add_argument("--model", default=DEFAULT_MODEL, help="Generation model")
    run.add_argument(
        "--classifier-model",
        default=None,
        help="Classification model (default: same as --model)",
    )
    run.add_argument(
        "--classify-pass",
        default="both",
        choices=["full", "fast", "deep", "both"],
        help="Pass mode for style classification",
    )
    run.add_argument("--plot", help="Run one plot id only (e.g. plot_03)")
    run.add_argument("--scene", help="Run one scene type only (opening, climax_reveal, romantic_encounter)")
    run.add_argument("--limit", type=int, help="Max cases")
    run.add_argument("--setup-only", action="store_true", help="Pass 1 world/roster prompt only")
    run.add_argument("--dry-run", action="store_true", help="Emit prompts without calling LLM")
    run.add_argument("--no-classify", action="store_true", help="Generate only; skip classification")

    reclass = sub.add_parser("reclassify", help="Re-run classification on existing results")
    reclass.add_argument("reclassify", type=Path)
    reclass.add_argument("--model", default=DEFAULT_MODEL)
    reclass.add_argument("--classifier-model", default=None)
    reclass.add_argument("--classify-pass", default="both", choices=["full", "fast", "deep", "both"])

    cmp = sub.add_parser("compare", help="Compare two benchmark result files")
    cmp.add_argument("baseline", type=Path)
    cmp.add_argument("candidate", type=Path)
    cmp.add_argument("--output", type=Path, default=None)

    summ = sub.add_parser("summarize", help="Aggregate stats for one result file")
    summ.add_argument("path", type=Path)

    # Default-command flags (when subcommand omitted)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument(
        "--output",
        default=str(ROOT / "eval" / "style_benchmark" / "results" / "latest.jsonl"),
    )
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--classifier-model", default=None)
    parser.add_argument("--classify-pass", default="both", choices=["full", "fast", "deep", "both"])
    parser.add_argument("--plot", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-classify", action="store_true")
    parser.add_argument("--reclassify", type=Path, default=None)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.reclassify is not None:
        cmd_reclassify(args)
        return

    command = getattr(args, "command", None) or "run"
    if command == "run":
        cmd_run(args)
    elif command == "reclassify":
        cmd_reclassify(args)
    elif command == "compare":
        cmd_compare(args)
    elif command == "summarize":
        cmd_summarize(args)
    else:
        parser.error(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
