#!/usr/bin/env python3
"""
OpenRouter free-model cascade for double-checking style-council labels.

Pulls the live free catalog, ranks chat models largest → smallest, then asks
each model the same single-metric council prompt. Failed/rate-limited models
fall through to the next. A cooldown sits between requests so the free tier
is not hammered.

Usage:
    python tools/style_evaluation/openrouter_cascade.py list
    python tools/style_evaluation/openrouter_cascade.py audit \\
        --passages 6 --votes 2 --cooldown 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.llm_client import (  # noqa: E402
    LLMError,
    OPENROUTER_BASE_URL,
    complete,
    openrouter_api_key,
    openrouter_headers,
)
from tools.style_classification.metric_council import (  # noqa: E402
    _parse_json,
    build_judge_prompts,
    resolve_metric,
)
from tools.style_classification.pass_config import ALL_LLM_FIELDS  # noqa: E402

CORPUS_DIR = ROOT / "train" / "romance_corpus"
OUT_DIR = ROOT / "eval" / "openrouter_audit"
MODELS_URL = OPENROUTER_BASE_URL.rstrip("/") + "/models"

# Skip non-chat / non-prose endpoints even when priced at zero.
_SKIP_SUBSTR = (
    "content-safety",
    "lyria",
    "whisper",
    "tts",
    "embed",
    "rerank",
    "moderation",
    "imagen",
    "flux",
    "sdxl",
    "sora",
    "kling",
    "veo",
    "image-edit",
)

_FALLBACK_HTTP = {402, 404, 408, 429, 502, 503, 529}
_LAST_CALL = 0.0


def _pace(cooldown: float) -> None:
    """Sleep so consecutive OpenRouter calls are at least ``cooldown`` seconds apart."""
    global _LAST_CALL
    if cooldown <= 0:
        _LAST_CALL = time.monotonic()
        return
    now = time.monotonic()
    wait = cooldown - (now - _LAST_CALL) if _LAST_CALL else 0.0
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL = time.monotonic()

# Gate fields plus the Pass-2 viewpoint set the current council files actually cover.
DEFAULT_FIELDS: tuple[str, ...] = (
    "tone",
    "pov",
    "register",
    "free_indirect_discourse",
    "figurative_density",
    "mind_style",
    "narrative_distance",
)

_PARAM_RE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*[bB]\b")
_MOE_RE = re.compile(r"(?i)(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*[bB]")


def _zero_price(value: Any) -> bool:
    try:
        return float(value or 0) == 0.0
    except (TypeError, ValueError):
        return False


def is_free_model(model: dict[str, Any]) -> bool:
    pricing = model.get("pricing") or {}
    if _zero_price(pricing.get("prompt")) and _zero_price(pricing.get("completion")):
        return True
    return str(model.get("id") or "").endswith(":free")


def is_text_chat_model(model: dict[str, Any]) -> bool:
    model_id = str(model.get("id") or "").lower()
    if any(token in model_id for token in _SKIP_SUBSTR):
        return False
    arch = model.get("architecture") or {}
    outputs = [str(x).lower() for x in (arch.get("output_modalities") or ["text"])]
    if "text" not in outputs:
        return False
    if "audio" in outputs and "text" in outputs and "lyria" in model_id:
        return False
    inputs = [str(x).lower() for x in (arch.get("input_modalities") or ["text"])]
    return "text" in inputs


def parameter_count_b(model: dict[str, Any]) -> float:
    blob = " ".join(
        str(model.get(k) or "") for k in ("id", "name", "canonical_slug", "hugging_face_id")
    ).replace("-", " ")
    values: list[float] = []
    moe = _MOE_RE.search(blob)
    if moe:
        values.append(float(moe.group(1)) * float(moe.group(2)))
    values.extend(float(m.group(1)) for m in _PARAM_RE.finditer(blob))
    return max(values) if values else 0.0


def rank_free_text_models(models: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Largest named parameter count first; unknown sizes follow by context length."""
    ranked: list[dict[str, Any]] = []
    for model in models:
        if not is_free_model(model) or not is_text_chat_model(model):
            continue
        row = dict(model)
        row["_params_b"] = parameter_count_b(model)
        row["_context"] = int(model.get("context_length") or 0)
        ranked.append(row)
    ranked.sort(key=lambda m: (-m["_params_b"], -m["_context"], str(m.get("id") or "")))
    return ranked


def fetch_openrouter_models(*, timeout: int = 30) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {openrouter_api_key()}", **openrouter_headers()}
    req = urllib.request.Request(MODELS_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    models = body.get("data") or []
    if not isinstance(models, list):
        raise LLMError("OpenRouter /models did not return a data list")
    return models


def cascade_ids(models: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for model in rank_free_text_models(models):
        model_id = str(model.get("id") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        out.append(model_id)
    return out


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def discover_council_paths(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    return sorted(p for p in corpus_dir.glob("_council_*_out.jsonl") if p.is_file())


def extract_consensus(record: dict[str, Any], fields: Iterable[str]) -> dict[str, str]:
    meta = record.get("metadata") or {}
    council = meta.get("style_council") or {}
    out: dict[str, str] = {}
    for field in fields:
        entry = council.get(field) if isinstance(council, dict) else None
        if not isinstance(entry, dict) or not entry.get("consensus"):
            continue
        value = entry.get("value")
        if isinstance(value, str) and value:
            out[field] = value
    return out


def load_council_passages(
    paths: list[Path],
    *,
    fields: Iterable[str],
    limit: int,
    seed: int,
    min_fields: int = 2,
) -> list[dict[str, Any]]:
    by_hash: dict[str, dict[str, Any]] = {}
    wanted = tuple(fields)
    for path in paths:
        for record in _iter_jsonl(path):
            text = (record.get("text") or "").strip()
            if not text:
                continue
            labels = extract_consensus(record, wanted)
            if len(labels) < min_fields:
                continue
            key = hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()
            meta = record.get("metadata") or {}
            extra = meta.get("extra") or {}
            candidate = {
                "id": key[:12],
                "text": text,
                "gold": labels,
                "title": str(meta.get("title") or extra.get("title_slug") or ""),
                "author": str(meta.get("author") or ""),
                "source_file": path.name,
            }
            previous = by_hash.get(key)
            if previous is None or len(labels) > len(previous["gold"]):
                by_hash[key] = candidate
    records = list(by_hash.values())
    rng = random.Random(seed)
    # Stratify: at most two passages per title so Alice does not dominate.
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_title[rec["title"] or rec["id"]].append(rec)
    pool: list[dict[str, Any]] = []
    for group in by_title.values():
        rng.shuffle(group)
        pool.extend(group[:2])
    rng.shuffle(pool)
    return pool[:limit]


def _http_status(error: LLMError) -> int | None:
    match = re.search(r"HTTP (\d{3}) ", str(error))
    return int(match.group(1)) if match else None


def should_fallback(error: LLMError) -> bool:
    status = _http_status(error)
    if status in _FALLBACK_HTTP:
        return True
    text = str(error).lower()
    return any(
        token in text
        for token in ("rate-limit", "rate limit", "temporarily", "overloaded", "timeout", "empty model")
    )


def judge_field(
    *,
    text: str,
    field: str,
    model: str,
    timeout: int = 90,
    cooldown: float = 0.0,
) -> dict[str, Any]:
    metric = resolve_metric(field)
    if metric is None:
        return {"label": None, "parse_ok": False, "error": f"unknown field {field}", "model": model}
    system, user = build_judge_prompts(metric, text, variant="definition")
    allowed = set(metric["values"])
    vote: dict[str, Any] = {
        "model": model,
        "field": field,
        "label": None,
        "raw_label": None,
        "evidence": None,
        "parse_ok": False,
        "error": None,
        "fallback": False,
    }
    _pace(cooldown)
    try:
        raw = complete(
            user,
            system=system,
            model=model,
            base_url=OPENROUTER_BASE_URL,
            api_key=openrouter_api_key(),
            extra_headers=openrouter_headers(),
            max_tokens=1536,
            temperature=0.05,
            timeout=timeout,
            max_retries=0,
        )
    except LLMError as exc:
        vote["error"] = str(exc)
        vote["fallback"] = should_fallback(exc)
        if _http_status(exc) == 429 and cooldown >= 0:
            time.sleep(max(15.0, cooldown))
        return vote
    parsed = _parse_json(raw)
    if not parsed:
        vote["error"] = "json_parse_failed"
        vote["fallback"] = True
        vote["raw"] = raw[:400]
        return vote
    vote["parse_ok"] = True
    raw_label = parsed.get(field)
    vote["raw_label"] = raw_label
    evidence = parsed.get("evidence")
    if isinstance(evidence, str):
        vote["evidence"] = evidence[:200]
    if isinstance(raw_label, str) and raw_label in allowed:
        vote["label"] = raw_label
    else:
        vote["error"] = "label_not_allowed"
        vote["fallback"] = True
    return vote


def collect_votes(
    *,
    text: str,
    field: str,
    cascade: list[str],
    votes_needed: int,
    cooldown: float,
    timeout: int,
    skip_models: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Walk the cascade until ``votes_needed`` valid labels are collected."""
    got: list[dict[str, Any]] = []
    skip = skip_models if skip_models is not None else set()
    labelled = 0
    for model in cascade:
        if model in skip:
            continue
        if labelled >= votes_needed:
            break
        vote = judge_field(
            text=text,
            field=field,
            model=model,
            timeout=timeout,
            cooldown=cooldown,
        )
        got.append(vote)
        if vote.get("label"):
            labelled += 1
            continue
        status = _http_status(LLMError(vote.get("error") or ""))
        # Only retire a model for the rest of the run when it is unauthorized or missing.
        if status in {401, 403, 404}:
            skip.add(model)
    return got


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [r for r in rows if r.get("label") and r.get("gold")]
    matches = [r for r in comparable if r["label"] == r["gold"]]
    by_field: dict[str, Counter] = defaultdict(Counter)
    by_model: dict[str, Counter] = defaultdict(Counter)
    for row in comparable:
        key = "agree" if row["label"] == row["gold"] else "disagree"
        by_field[row["field"]][key] += 1
        by_model[row["model"]][key] += 1
        by_field[row["field"]][str(row["label"])] += 1
    return {
        "n_rows": len(rows),
        "n_labelled": len(comparable),
        "agree": len(matches),
        "agree_rate": (len(matches) / len(comparable)) if comparable else None,
        "by_field": {k: dict(v) for k, v in sorted(by_field.items())},
        "by_model": {k: dict(v) for k, v in sorted(by_model.items())},
        "errors": dict(Counter(r.get("error") or "ok" for r in rows if not r.get("label"))),
    }


def _print_cascade(cascade: list[str], ranked: list[dict[str, Any]]) -> None:
    by_id = {m["id"]: m for m in ranked}
    print(f"{len(cascade)} free text models, largest → smallest:\n")
    for i, model_id in enumerate(cascade, 1):
        meta = by_id.get(model_id) or {}
        params = meta.get("_params_b") or 0
        ctx = meta.get("_context") or 0
        size = f"{params:g}B" if params else "?"
        print(f"  {i:2d}. {size:<8} ctx={ctx:<8} {model_id}")


def cmd_list(args: argparse.Namespace) -> int:
    models = fetch_openrouter_models()
    ranked = rank_free_text_models(models)
    ids = cascade_ids(models)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_catalog": len(models),
        "cascade": ids,
        "models": [
            {
                "id": m["id"],
                "params_b": m["_params_b"],
                "context_length": m["_context"],
                "name": m.get("name"),
            }
            for m in ranked
        ],
    }
    out = Path(args.out) if args.out else OUT_DIR / "cascade.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_cascade(ids, ranked)
    print(f"\nwrote {out}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    fields = tuple(args.fields) if args.fields else DEFAULT_FIELDS
    fields = tuple(f for f in fields if f in ALL_LLM_FIELDS)
    council_paths = args.council or discover_council_paths()
    passages = load_council_passages(
        council_paths,
        fields=fields,
        limit=args.passages,
        seed=args.seed,
        min_fields=args.min_fields,
    )
    if not passages:
        print("no council passages with consensus labels found", file=sys.stderr)
        return 1

    if args.cascade_file:
        saved = json.loads(Path(args.cascade_file).read_text(encoding="utf-8"))
        cascade = list(saved.get("cascade") or [])
        ranked = [
            {
                "id": m.get("id"),
                "_params_b": float(m.get("params_b") or m.get("_params_b") or 0),
                "_context": int(m.get("context_length") or m.get("_context") or 0),
            }
            for m in (saved.get("models") or [])
            if m.get("id")
        ]
    else:
        models = fetch_openrouter_models()
        ranked = rank_free_text_models(models)
        cascade = cascade_ids(models)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "cascade.json").write_text(
            json.dumps(
                {
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "cascade": cascade,
                    "models": [
                        {"id": m["id"], "params_b": m["_params_b"], "context_length": m["_context"]}
                        for m in ranked
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if args.max_models and args.max_models > 0:
        cascade = cascade[: args.max_models]
    _print_cascade(cascade, ranked)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out) if args.out else OUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    questions = [(p, field) for p in passages for field in fields if field in p["gold"]]
    if args.max_questions:
        questions = questions[: args.max_questions]

    print(
        f"\n{len(passages)} passages, {len(questions)} questions, "
        f"{args.votes} vote(s), cooldown {args.cooldown}s"
    )
    rows: list[dict[str, Any]] = []
    skip_models: set[str] = set()
    for i, (passage, field) in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {passage['id']} {field} gold={passage['gold'][field]}", flush=True)
        votes = collect_votes(
            text=passage["text"],
            field=field,
            cascade=cascade,
            votes_needed=args.votes,
            cooldown=args.cooldown,
            timeout=args.timeout,
            skip_models=skip_models,
        )
        for vote in votes:
            row = {
                "passage_id": passage["id"],
                "title": passage["title"],
                "author": passage["author"],
                "field": field,
                "gold": passage["gold"][field],
                **vote,
            }
            rows.append(row)
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            mark = "agree" if vote.get("label") == passage["gold"][field] else (
                "fallback" if vote.get("fallback") else "disagree"
            )
            print(
                f"    {vote.get('model')}: {vote.get('label') or vote.get('error')} ({mark})",
                flush=True,
            )

    report = summarize_results(rows)
    report["passages"] = len(passages)
    report["cascade"] = cascade
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rate = report["agree_rate"]
    rate_s = "n/a" if rate is None else f"{rate:.1%}"
    print(f"\nagree {report['agree']}/{report['n_labelled']} ({rate_s})")
    print(f"wrote {results_path}")
    print(f"wrote {run_dir / 'summary.json'}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Fetch and rank free OpenRouter chat models")
    p_list.add_argument("--out", type=Path, default=None)
    p_list.set_defaults(func=cmd_list)

    p_audit = sub.add_parser("audit", help="Compare council labels against the free cascade")
    p_audit.add_argument("--passages", type=int, default=6)
    p_audit.add_argument("--votes", type=int, default=2, help="Successful models per question")
    p_audit.add_argument("--cooldown", type=float, default=10.0)
    p_audit.add_argument("--timeout", type=int, default=90)
    p_audit.add_argument("--seed", type=int, default=7)
    p_audit.add_argument("--min-fields", type=int, default=2)
    p_audit.add_argument("--max-models", type=int, default=8, help="Cap cascade depth (0 = all)")
    p_audit.add_argument("--max-questions", type=int, default=16)
    p_audit.add_argument("--fields", nargs="*", default=None)
    p_audit.add_argument("--council", nargs="*", type=Path, default=None)
    p_audit.add_argument("--cascade-file", type=Path, default=None)
    p_audit.add_argument("--out", type=Path, default=None)
    p_audit.set_defaults(func=cmd_audit)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
