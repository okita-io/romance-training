#!/usr/bin/env python3
"""
Phase 5A — Evaluator agree/reject bake-off.

Uses the Gemma 4 style GGUF (via LM Studio / OpenAI-compatible API) as the judge
against held-out gold ``style_profile`` labels.

Commands:
  build-set   Sample gold + trap passages into an eval JSONL
  run         Classify the eval set with the judge model; write agree/reject
  summarize   Print hit rates + go/no-go against thresholds

Example (on the 3090 once the GGUF is loaded in LM Studio):

  export LLM_BASE_URL=http://localhost:1234/v1
  export LLM_MODEL=gemma4-style-step3000-q4_k_m
  export LLM_DISABLE_THINKING=1

  python tools/style_evaluation/bakeoff_5a.py build-set
  python tools/style_evaluation/bakeoff_5a.py run \\
    --eval-set eval/bakeoff_5a/eval_set.jsonl \\
    --output eval/bakeoff_5a/results/gemma4_step3000_q4.jsonl
  python tools/style_evaluation/bakeoff_5a.py summarize \\
    eval/bakeoff_5a/results/gemma4_step3000_q4.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CORPUS_DIR = ROOT / "train" / "romance_corpus"
DEFAULT_EVAL_DIR = ROOT / "eval" / "bakeoff_5a"
DEFAULT_EVAL_SET = DEFAULT_EVAL_DIR / "eval_set.jsonl"
DEFAULT_RESULTS_DIR = DEFAULT_EVAL_DIR / "results"

# Gate fields from docs/PHASE5_STYLE_STEERING.md §5A
GATE_FIELDS: tuple[str, ...] = (
    "tone",
    "pov",
    "register",
    "free_indirect_discourse",
    "figurative_density",
)

# Broader LLM fields for reporting (same set as style benchmark deltas)
from tools.style_evaluation.benchmark import COMPARE_FIELDS  # noqa: E402

DEFAULT_THRESHOLDS: dict[str, float] = {
    "json_parse_rate": 0.90,
    "gate_exact_rate": 0.60,  # all gate fields exact-match
    "tone": 0.65,
    "pov": 0.70,
    "register": 0.65,
    "free_indirect_discourse": 0.55,
    "figurative_density": 0.55,
}

# Trap heuristics (rare / conflicting profiles)
TRAP_RULES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("mixed_pov", {"pov": "mixed"}),
    ("heavy_fid", {"free_indirect_discourse": "heavy"}),
    ("archaic_register", {"register": "archaic"}),
    ("first_person", {"pov": "first_person"}),
    ("comedic_tone", {"tone": "comedic"}),
    ("low_figurative", {"figurative_density": "low"}),
)


def _is_scalar_label(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and not value.startswith("{")


def _profile_ok(profile: dict[str, Any]) -> bool:
    if not profile:
        return False
    for key in GATE_FIELDS:
        if not _is_scalar_label(profile.get(key)):
            return False
    return True


def _record_id(source: str, text: str, index: int) -> str:
    digest = hashlib.sha1(f"{source}|{index}|{text[:200]}".encode("utf-8")).hexdigest()[:12]
    return f"{digest}"


def _iter_styled_jsonl(paths: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield path, i, rec


def _extract_candidate(path: Path, index: int, rec: dict[str, Any]) -> dict[str, Any] | None:
    text = rec.get("text")
    if not isinstance(text, str) or len(text.split()) < 80:
        return None
    meta = rec.get("metadata") or {}
    profile = meta.get("style_profile") or {}
    if not _profile_ok(profile):
        return None
    gold = {k: profile[k] for k in COMPARE_FIELDS if _is_scalar_label(profile.get(k))}
    if len(gold) < len(GATE_FIELDS):
        return None
    source = str(meta.get("source") or meta.get("source_dataset") or path.name)
    return {
        "id": _record_id(source, text, index),
        "source": source,
        "source_file": path.name,
        "source_line": index,
        "text": text,
        "gold_profile": gold,
        "word_count": len(text.split()),
    }


def _matches_trap(profile: dict[str, Any], rule: dict[str, Any]) -> bool:
    return all(profile.get(k) == v for k, v in rule.items())


def build_eval_set(
    corpus_paths: list[Path],
    *,
    n_gold: int = 30,
    n_traps: int = 10,
    seed: int = 42,
    max_scan: int = 80_000,
) -> list[dict[str, Any]]:
    """Sample diverse gold passages + trap passages from styled JSONL."""
    rng = random.Random(seed)
    gold_pool: list[dict[str, Any]] = []
    trap_pools: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in TRAP_RULES}

    scanned = 0
    for path, index, rec in _iter_styled_jsonl(corpus_paths):
        scanned += 1
        if scanned > max_scan:
            break
        cand = _extract_candidate(path, index, rec)
        if cand is None:
            continue
        profile = cand["gold_profile"]
        trapped = False
        for name, rule in TRAP_RULES:
            if _matches_trap(profile, rule):
                trap_pools[name].append(cand)
                trapped = True
                break
        if not trapped:
            gold_pool.append(cand)

    # Prefer rarer trap categories first, then fill.
    traps: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    order = sorted(trap_pools.keys(), key=lambda k: len(trap_pools[k]))
    for name in order:
        pool = trap_pools[name]
        if not pool:
            continue
        rng.shuffle(pool)
        pick = pool[0]
        if pick["id"] in seen_ids:
            continue
        entry = dict(pick)
        entry["set"] = "trap"
        entry["trap_reason"] = name
        traps.append(entry)
        seen_ids.add(pick["id"])
        if len(traps) >= n_traps:
            break

    # Fill remaining traps from any trap pool
    if len(traps) < n_traps:
        leftover: list[tuple[str, dict[str, Any]]] = []
        for name, pool in trap_pools.items():
            for cand in pool:
                if cand["id"] in seen_ids:
                    continue
                leftover.append((name, cand))
        rng.shuffle(leftover)
        for name, cand in leftover:
            entry = dict(cand)
            entry["set"] = "trap"
            entry["trap_reason"] = name
            traps.append(entry)
            seen_ids.add(cand["id"])
            if len(traps) >= n_traps:
                break

    # Stratify gold a bit by tone/pov/register
    rng.shuffle(gold_pool)
    gold: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for cand in gold_pool:
        if cand["id"] in seen_ids:
            continue
        key = (
            f"{cand['gold_profile'].get('tone')}|"
            f"{cand['gold_profile'].get('pov')}|"
            f"{cand['gold_profile'].get('register')}"
        )
        buckets.setdefault(key, []).append(cand)

    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    # Round-robin across buckets for diversity
    while len(gold) < n_gold and bucket_keys:
        progressed = False
        for key in list(bucket_keys):
            if not buckets[key]:
                bucket_keys.remove(key)
                continue
            cand = buckets[key].pop()
            if cand["id"] in seen_ids:
                continue
            entry = dict(cand)
            entry["set"] = "gold"
            entry["trap_reason"] = None
            gold.append(entry)
            seen_ids.add(cand["id"])
            progressed = True
            if len(gold) >= n_gold:
                break
        if not progressed:
            break

    if len(gold) < n_gold:
        raise RuntimeError(
            f"Only found {len(gold)} gold passages (need {n_gold}). "
            "Widen corpus paths or raise --max-scan."
        )
    if len(traps) < n_traps:
        print(
            f"Warning: only found {len(traps)} trap passages (wanted {n_traps}).",
            file=sys.stderr,
        )

    out = gold[:n_gold] + traps[:n_traps]
    rng.shuffle(out)
    return out


def _exact_match(gold: Any, pred: Any) -> bool:
    return gold == pred


# Mild near-miss maps for a secondary "relaxed" score (not the go/no-go gate).
_RELAXED_NEIGHBORS: dict[str, dict[str, set[str]]] = {
    "figurative_density": {
        "low": {"low", "moderate"},
        "moderate": {"low", "moderate", "high"},
        "high": {"moderate", "high"},
    },
    "free_indirect_discourse": {
        "none": {"none", "sparse"},
        "sparse": {"none", "sparse", "moderate"},
        "moderate": {"sparse", "moderate", "heavy"},
        "heavy": {"moderate", "heavy"},
    },
    "narrative_distance": {
        "intimate": {"intimate", "moderate"},
        "moderate": {"intimate", "moderate", "distant"},
        "distant": {"moderate", "distant"},
    },
}


def _relaxed_match(field: str, gold: Any, pred: Any) -> bool:
    if gold == pred:
        return True
    neighbors = _RELAXED_NEIGHBORS.get(field, {}).get(str(gold))
    if neighbors is None:
        return False
    return pred in neighbors


def score_prediction(
    gold_profile: dict[str, Any],
    pred_profile: dict[str, Any],
    *,
    parse_ok: bool,
) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    exact_hits = 0
    exact_compared = 0
    relaxed_hits = 0
    gate_exact = True
    gate_compared = 0

    for key in COMPARE_FIELDS:
        g = gold_profile.get(key)
        if g is None:
            continue
        p = pred_profile.get(key)
        exact = _exact_match(g, p)
        relaxed = _relaxed_match(key, g, p)
        fields[key] = {
            "gold": g,
            "pred": p,
            "exact": exact,
            "relaxed": relaxed,
        }
        exact_compared += 1
        if exact:
            exact_hits += 1
        if relaxed:
            relaxed_hits += 1
        if key in GATE_FIELDS:
            gate_compared += 1
            if not exact:
                gate_exact = False

    if gate_compared < len(GATE_FIELDS):
        gate_exact = False

    agree = bool(parse_ok and gate_exact)
    return {
        "parse_ok": parse_ok,
        "agree": agree,
        "gate_exact": gate_exact,
        "exact_match_count": exact_hits,
        "exact_compared_count": exact_compared,
        "exact_match_score": round(exact_hits / exact_compared, 4) if exact_compared else 0.0,
        "relaxed_match_score": round(relaxed_hits / exact_compared, 4) if exact_compared else 0.0,
        "fields": fields,
    }


def run_bakeoff(
    eval_set: list[dict[str, Any]],
    *,
    model: str,
    pass_mode: str = "full",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    from tools.style_classification.classify_passage import load_rubric
    from tools.style_classification.metrics_llm import assess_detailed

    try:
        rubric = load_rubric()
    except FileNotFoundError:
        rubric = None

    results: list[dict[str, Any]] = []
    items = eval_set[:limit] if limit else eval_set
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['id']} ({item['set']}) …", flush=True)
        pred, meta = assess_detailed(
            item["text"],
            model=model,
            rubric=rubric,
            pass_mode=pass_mode,  # type: ignore[arg-type]
        )
        # Drop non-LLM computables from comparison noise; assess returns LLM fields only.
        score = score_prediction(item["gold_profile"], pred, parse_ok=bool(meta.get("parse_ok")))
        record = {
            "id": item["id"],
            "set": item["set"],
            "trap_reason": item.get("trap_reason"),
            "source": item.get("source"),
            "source_file": item.get("source_file"),
            "word_count": item.get("word_count"),
            "gold_profile": item["gold_profile"],
            "pred_profile": {k: pred.get(k) for k in COMPARE_FIELDS if k in pred},
            "judge_meta": meta,
            "score": score,
            "model": model,
            "pass_mode": pass_mode,
            "judged_at": datetime.now(timezone.utc).isoformat(),
        }
        verdict = "AGREE" if score["agree"] else "REJECT"
        print(
            f"    → {verdict} parse_ok={score['parse_ok']} "
            f"exact={score['exact_match_score']:.2f} gate={score['gate_exact']}",
            flush=True,
        )
        results.append(record)
    return results


def summarize(results: list[dict[str, Any]], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    n = len(results)
    if n == 0:
        return {"n": 0, "go": False, "reason": "empty_results"}

    parse_ok = sum(1 for r in results if r.get("score", {}).get("parse_ok"))
    agree = sum(1 for r in results if r.get("score", {}).get("agree"))
    gate_exact = sum(1 for r in results if r.get("score", {}).get("gate_exact"))

    field_hits: Counter[str] = Counter()
    field_n: Counter[str] = Counter()
    for r in results:
        for key, info in (r.get("score") or {}).get("fields", {}).items():
            field_n[key] += 1
            if info.get("exact"):
                field_hits[key] += 1

    field_hit_rate = {
        k: round(field_hits[k] / field_n[k], 4) for k in sorted(field_n) if field_n[k]
    }

    by_set: dict[str, dict[str, Any]] = {}
    for set_name in ("gold", "trap"):
        subset = [r for r in results if r.get("set") == set_name]
        if not subset:
            continue
        by_set[set_name] = {
            "n": len(subset),
            "agree_rate": round(
                sum(1 for r in subset if r["score"]["agree"]) / len(subset), 4
            ),
            "parse_ok_rate": round(
                sum(1 for r in subset if r["score"]["parse_ok"]) / len(subset), 4
            ),
            "mean_exact_match_score": round(
                sum(r["score"]["exact_match_score"] for r in subset) / len(subset), 4
            ),
        }

    json_parse_rate = parse_ok / n
    gate_exact_rate = gate_exact / n
    agree_rate = agree / n

    checks: dict[str, dict[str, Any]] = {
        "json_parse_rate": {
            "value": round(json_parse_rate, 4),
            "threshold": thresholds["json_parse_rate"],
            "pass": json_parse_rate >= thresholds["json_parse_rate"],
        },
        "gate_exact_rate": {
            "value": round(gate_exact_rate, 4),
            "threshold": thresholds["gate_exact_rate"],
            "pass": gate_exact_rate >= thresholds["gate_exact_rate"],
        },
    }
    for field in GATE_FIELDS:
        thr = thresholds.get(field, 0.55)
        val = field_hit_rate.get(field, 0.0)
        checks[f"field:{field}"] = {
            "value": val,
            "threshold": thr,
            "pass": val >= thr,
        }

    go = all(c["pass"] for c in checks.values())
    report = {
        "n": n,
        "json_parse_rate": round(json_parse_rate, 4),
        "agree_rate": round(agree_rate, 4),
        "gate_exact_rate": round(gate_exact_rate, 4),
        "field_hit_rate": field_hit_rate,
        "by_set": by_set,
        "checks": checks,
        "go": go,
        "gate_fields": list(GATE_FIELDS),
        "thresholds": thresholds,
        "model": results[0].get("model") if results else None,
    }
    return report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _default_corpus_paths(corpus_dir: Path) -> list[Path]:
    paths = sorted(corpus_dir.glob("*_styled_seg_*.jsonl"))
    if not paths:
        paths = sorted(corpus_dir.glob("*styled*.jsonl"))
    return paths


def cmd_build_set(args: argparse.Namespace) -> None:
    corpus_dir = Path(args.corpus_dir)
    paths = [Path(p) for p in args.corpus] if args.corpus else _default_corpus_paths(corpus_dir)
    if not paths:
        raise SystemExit(f"No styled JSONL found under {corpus_dir}")
    print(f"Scanning {len(paths)} files (seed={args.seed}) …")
    rows = build_eval_set(
        paths,
        n_gold=args.n_gold,
        n_traps=args.n_traps,
        seed=args.seed,
        max_scan=args.max_scan,
    )
    out = Path(args.output)
    write_jsonl(out, rows)
    n_gold = sum(1 for r in rows if r["set"] == "gold")
    n_trap = sum(1 for r in rows if r["set"] == "trap")
    trap_reasons = Counter(r.get("trap_reason") for r in rows if r["set"] == "trap")
    print(f"Wrote {len(rows)} passages → {out}")
    print(f"  gold={n_gold} traps={n_trap} reasons={dict(trap_reasons)}")


def cmd_run(args: argparse.Namespace) -> None:
    from tools.llm_client import DEFAULT_MODEL

    model = args.model or DEFAULT_MODEL
    eval_set = read_jsonl(Path(args.eval_set))
    if not eval_set:
        raise SystemExit(f"Empty eval set: {args.eval_set}")
    print(f"Judge model: {model}")
    print(f"Eval set: {args.eval_set} ({len(eval_set)} passages)")
    print(f"Pass mode: {args.pass_mode}")
    results = run_bakeoff(
        eval_set,
        model=model,
        pass_mode=args.pass_mode,
        limit=args.limit,
    )
    out = Path(args.output)
    write_jsonl(out, results)

    agrees = [r for r in results if r["score"]["agree"]]
    rejects = [r for r in results if not r["score"]["agree"]]
    agree_path = out.with_name(out.stem + ".agree.jsonl")
    reject_path = out.with_name(out.stem + ".reject.jsonl")
    write_jsonl(agree_path, agrees)
    write_jsonl(reject_path, rejects)

    report = summarize(results)
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote results → {out}")
    print(f"  agree={len(agrees)} reject={len(rejects)}")
    print(f"  summary → {summary_path}")
    _print_summary(report)


def cmd_summarize(args: argparse.Namespace) -> None:
    results = read_jsonl(Path(args.results))
    report = summarize(results)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_summary(report)


def _print_summary(report: dict[str, Any]) -> None:
    print("\n=== Phase 5A bake-off summary ===")
    print(f"n={report['n']}  model={report.get('model')}")
    print(f"JSON parse rate: {report['json_parse_rate']:.2%}")
    print(f"Gate exact rate: {report['gate_exact_rate']:.2%}  (agree={report['agree_rate']:.2%})")
    print("Field hit rates:")
    for k, v in report.get("field_hit_rate", {}).items():
        marker = " *" if k in GATE_FIELDS else ""
        print(f"  {k}: {v:.2%}{marker}")
    if report.get("by_set"):
        print("By set:")
        for name, info in report["by_set"].items():
            print(
                f"  {name}: n={info['n']} agree={info['agree_rate']:.2%} "
                f"parse={info['parse_ok_rate']:.2%} mean_exact={info['mean_exact_match_score']:.2%}"
            )
    print("Checks:")
    for name, info in report.get("checks", {}).items():
        status = "PASS" if info["pass"] else "FAIL"
        print(f"  [{status}] {name}: {info['value']:.2%} (need ≥ {info['threshold']:.0%})")
    print()
    if report.get("go"):
        print("GO — promote Gemma GGUF as Phase 2 classifier; unlock 5B / 6A.")
    else:
        print("NO-GO — do not promote yet; inspect rejects / add targeted SFT.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 5A evaluator bake-off")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-set", help="Sample gold + trap eval passages")
    b.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    b.add_argument("--corpus", nargs="*", default=None, help="Explicit styled JSONL paths")
    b.add_argument("--n-gold", type=int, default=30)
    b.add_argument("--n-traps", type=int, default=10)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--max-scan", type=int, default=80_000)
    b.add_argument("--output", type=Path, default=DEFAULT_EVAL_SET)
    b.set_defaults(func=cmd_build_set)

    r = sub.add_parser("run", help="Judge eval set with Gemma GGUF via LM Studio")
    r.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    r.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "gemma4_step3000_bakeoff.jsonl",
    )
    r.add_argument("--model", default=None, help="LM Studio model id (default: LLM_MODEL)")
    r.add_argument(
        "--pass-mode",
        choices=("full", "fast", "deep", "both"),
        default="full",
        help="Classification pass mode (default: full — one JSON call)",
    )
    r.add_argument("--limit", type=int, default=None, help="Smoke-test on first N passages")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("summarize", help="Summarize an existing results JSONL")
    s.add_argument("results", type=Path)
    s.add_argument("--output", type=Path, default=None)
    s.set_defaults(func=cmd_summarize)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
