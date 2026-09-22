#!/usr/bin/env python3
"""
Re-classify Kev spans with a size-weighted teacher panel.

Default panel is the two local, not-rate-limited judges:
  - qwen3.8-27b-sglang at http://127.0.0.1:8888/v1  (27B)
  - hemmingway-1        at http://10.0.1.8:1234/v1  (Qwen3.8-27B writing fine-tune)

Optional ``--with-openrouter`` adds the free cascade (550B / 120B / …) with
cooldowns. Vote weights default to log2(params+1) so a 550B teacher cannot
erase both 27B locals.

Usage:
    python tools/style_evaluation/panel_reclassify.py \\
        --input train/style_training/kev/spans/span_mix.jsonl \\
        --fields tone register pov --limit 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.llm_client import (  # noqa: E402
    DEFAULT_API_KEY,
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
    vote_weight,
    weighted_council_vote,
)
from tools.style_classification.pass_config import ALL_LLM_FIELDS  # noqa: E402
from tools.style_evaluation.openrouter_cascade import should_fallback  # noqa: E402

DEFAULT_INPUT = ROOT / "train" / "style_training" / "kev" / "spans" / "span_mix.jsonl"
DEFAULT_FIELDS = (
    "tone",
    "register",
    "pov",
    "figurative_density",
    "free_indirect_discourse",
    "mind_style",
    "narrative_distance",
    "sentence_complexity",
)


@dataclass(frozen=True)
class Judge:
    name: str
    model: str
    base_url: str
    params_b: float
    api_key: str = DEFAULT_API_KEY
    cooldown: float = 0.0
    timeout: int = 120
    openrouter: bool = False


LOCAL_JUDGES: tuple[Judge, ...] = (
    Judge(
        name="hemmingway-1",
        model="hemmingway-1",
        base_url="http://10.0.1.8:1234/v1",
        params_b=27.0,  # Altworld/Hemmingway-1 is a Qwen3.8-27B writing fine-tune
        api_key="lm-studio",
        cooldown=0.0,
        timeout=120,
    ),
    Judge(
        name="qwen3.8-27b-sglang",
        model="qwen3.8-27b-sglang",
        base_url="http://127.0.0.1:8888/v1",
        params_b=27.0,
        api_key="sglang",
        cooldown=0.0,
        timeout=180,
    ),
)

OPENROUTER_JUDGES: tuple[Judge, ...] = (
    Judge(
        name="nemotron-ultra-550b",
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url=OPENROUTER_BASE_URL,
        params_b=550.0,
        cooldown=10.0,
        timeout=90,
        openrouter=True,
    ),
    Judge(
        name="nemotron-super-120b",
        model="nvidia/nemotron-3-super-120b-a12b:free",
        base_url=OPENROUTER_BASE_URL,
        params_b=120.0,
        cooldown=10.0,
        timeout=90,
        openrouter=True,
    ),
)


def build_panel(*, with_openrouter: bool) -> list[Judge]:
    panel = list(LOCAL_JUDGES)
    if with_openrouter:
        panel.extend(OPENROUTER_JUDGES)
    return panel


def ask_judge(
    judge: Judge,
    *,
    text: str,
    field: str,
    weight_mode: str,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    metric = resolve_metric(field)
    vote: dict[str, Any] = {
        "judge": judge.name,
        "model": judge.model,
        "params_b": judge.params_b,
        "weight": vote_weight(judge.params_b, weight_mode),
        "label": None,
        "raw_label": None,
        "evidence": None,
        "parse_ok": False,
        "error": None,
        "fallback": False,
    }
    if metric is None:
        vote["error"] = f"unknown field {field}"
        return vote
    system, user = build_judge_prompts(metric, text, variant="definition")
    allowed = set(metric["values"])
    if judge.cooldown > 0:
        time.sleep(judge.cooldown)
    headers = openrouter_headers() if judge.openrouter else None
    api_key = openrouter_api_key() if judge.openrouter else judge.api_key
    try:
        raw = complete(
            user,
            system=system,
            model=judge.model,
            base_url=judge.base_url,
            api_key=api_key,
            extra_headers=headers,
            max_tokens=max_tokens,
            temperature=0.05,
            timeout=judge.timeout,
            max_retries=0,
        )
    except LLMError as exc:
        vote["error"] = str(exc)
        vote["fallback"] = should_fallback(exc)
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


def classify_span(
    record: dict[str, Any],
    *,
    fields: list[str],
    panel: list[Judge],
    weight_mode: str,
    min_share: float,
) -> dict[str, Any]:
    text = (record.get("text") or "").strip()
    meta = dict(record.get("metadata") or {})
    profile: dict[str, str] = {}
    council: dict[str, Any] = {}
    for field in fields:
        votes: list[dict[str, Any]] = []
        for judge in panel:
            votes.append(ask_judge(judge, text=text, field=field, weight_mode=weight_mode))
        decided = weighted_council_vote(votes, min_share=min_share, min_voters=2)
        council[field] = decided
        if decided.get("value"):
            profile[field] = decided["value"]
        print(
            f"    {field}: {decided.get('value') or 'no-consensus'} "
            f"share={decided.get('weight_share')} n={decided.get('n_valid')}",
            flush=True,
        )
    meta["style_profile"] = profile
    meta["style_council"] = council
    meta["panel"] = {
        "judges": [j.name for j in panel],
        "weight_mode": weight_mode,
        "min_share": min_share,
        "classified_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"text": text, "metadata": meta}


def _completed_span_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            span_id = (rec.get("metadata") or {}).get("span_id")
            if span_id:
                done.add(str(span_id))
    return done
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fields", nargs="*", default=list(DEFAULT_FIELDS))
    ap.add_argument("--limit", type=int, default=0, help="Classify only the first N spans (0 = all)")
    ap.add_argument("--weight-mode", choices=("log", "linear", "equal"), default="log")
    ap.add_argument("--min-share", type=float, default=0.5)
    ap.add_argument("--with-openrouter", action="store_true")
    ap.add_argument("--local-only", action="store_true", help="Alias for default (no OpenRouter)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.is_file():
        print(f"missing input {args.input}", file=sys.stderr)
        return 1
    fields = [f for f in args.fields if f in ALL_LLM_FIELDS]
    panel = build_panel(with_openrouter=args.with_openrouter)
    print("panel:")
    for judge in panel:
        print(
            f"  {judge.name:24s} {judge.params_b:g}B  w={vote_weight(judge.params_b, args.weight_mode):.2f}  {judge.base_url}"
        )
    out = args.out
    if out is None:
        suffix = ".panel.jsonl"
        out = args.input.with_name(args.input.stem + suffix)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_skip = 0
    done = _completed_span_ids(out)
    mode = "a" if done else "w"
    if done:
        print(f"resuming {out}: {len(done)} spans already written")
    with args.input.open(encoding="utf-8") as fh, out.open(mode, encoding="utf-8") as dest:
        for line in fh:
            if not line.strip():
                continue
            n_in += 1
            if args.limit and n_out >= args.limit:
                break
            record = json.loads(line)
            span_id = str((record.get("metadata") or {}).get("span_id") or n_in)
            if span_id in done:
                n_skip += 1
                continue
            print(f"[{n_out + 1}] {span_id}", flush=True)
            classified = classify_span(
                record,
                fields=fields,
                panel=panel,
                weight_mode=args.weight_mode,
                min_share=args.min_share,
            )
            dest.write(json.dumps(classified, ensure_ascii=False) + "\n")
            dest.flush()
            n_out += 1
            done.add(span_id)
    print(f"wrote {n_out} new records ({n_skip} skipped) to {out}")
    return 0 if (n_out or n_skip) else 1


if __name__ == "__main__":
    raise SystemExit(main())
