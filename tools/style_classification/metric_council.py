"""
Single-metric teacher council for style classification.

Instead of asking one LLM call for the whole ``style_profile`` schema, the
council judges **one metric at a time** with a clear single goal, using three
slightly different prompt framings (definition / evidence / contrast) as three
"teachers". Each teacher cites supporting evidence. When they are unanimous the
label stands; when they split (or a teacher fails to produce a valid label) a
senior **arbitrator** reviews their evidence plus the passage and decides.

The council reuses the same loaded model for all three variants — diversity
comes from prompt framing, not from swapping weights (LM Studio serves one
model at a time). Temperature stays low so variance is driven by framing.

Public API:
  resolve_metric(field_id, rubric=None)      -> metric spec dict | None
  build_judge_prompts(metric, text, variant) -> (system, user)
  judge_once(metric, text, variant, ...)     -> vote dict
  council_vote(votes)                        -> {value, agree_n, consensus, votes}
  vote_weight(params_b, mode)                -> float
  weighted_council_vote(votes)               -> {value, weight_share, consensus, votes}
  build_arbitrator_prompts(metric, text, votes) -> (system, user)
  arbitrate(metric, text, votes, ...)        -> {value, rationale, parse_ok, error}
  assess_metric_council(text, field, ...)    -> (label | None, meta)
  assess_fields_council(text, fields, ...)   -> (profile_partial, council_meta)
"""

from __future__ import annotations

import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from llm_client import DEFAULT_MODEL, LLMError, complete as llm_complete  # noqa: E402

ANALYSIS_SYSTEM_PATH = ROOT / "source" / "extracted" / "style_analysis_system.json"

# Prompt framings — three "teachers" applying the same metric from different angles.
COUNCIL_VARIANTS: tuple[str, ...] = ("definition", "evidence", "contrast")

_BASE_SYSTEM = (
    "You are a literary stylistician trained in Geoffrey Leech and Mick Short's "
    "*Style in Fiction* framework. You judge ONE style metric at a time on a prose "
    "passage. Return ONLY valid JSON — no explanation, no markdown."
)

_VARIANT_SYSTEM: dict[str, str] = {
    "definition": (
        _BASE_SYSTEM
        + " Apply the metric's definition strictly and literally. Choose the single "
        "allowed label whose definition best fits the passage."
    ),
    "evidence": (
        _BASE_SYSTEM
        + " First locate the single strongest linguistic cue in the passage for this "
        "metric, then choose the allowed label that the evidence supports."
    ),
    "contrast": (
        _BASE_SYSTEM
        + " Consider the two most plausible labels, briefly rule out the nearest "
        "distractor, then choose the allowed label that remains."
    ),
}

_ARBITRATOR_SYSTEM = (
    "You are the senior arbitrator on a panel of literary stylisticians trained in "
    "Geoffrey Leech and Mick Short's *Style in Fiction*. Three junior judges have each "
    "labelled ONE metric on a passage and cited supporting evidence. Weigh their evidence "
    "against the passage itself and decide the single most defensible allowed label. You "
    "may overrule the majority when its evidence is weak, and you may choose a label no "
    "judge picked if the passage warrants it. If no allowed label is defensible, return "
    "null. Return ONLY valid JSON — no explanation outside the JSON, no markdown."
)


@lru_cache(maxsize=1)
def _load_analysis_system() -> dict[str, Any]:
    if not ANALYSIS_SYSTEM_PATH.exists():
        return {}
    try:
        return json.loads(ANALYSIS_SYSTEM_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _metric_specs(rubric: dict | None = None) -> dict[str, dict[str, Any]]:
    """Index every judgeable metric id -> spec (values, definition, prompt)."""
    specs: dict[str, dict[str, Any]] = {}

    def _add(entry: dict[str, Any]) -> None:
        mid = entry.get("id")
        values = entry.get("values")
        if not mid or not isinstance(values, list) or not values:
            return
        if mid in specs:
            return
        specs[mid] = {
            "id": mid,
            "name": entry.get("name", mid),
            "definition": entry.get("definition", ""),
            "analysis_prompt": entry.get("analysis_prompt", ""),
            "scoring": dict(entry.get("scoring") or {}),
            "values": list(values),
        }

    # Rubric dimensions take priority when present (project-specific tuning).
    if rubric:
        for dim in rubric.get("dimensions", []):
            if dim.get("computation") == "llm":
                _add(dim)

    system = _load_analysis_system()
    for dim in system.get("llm_dimensions", []):
        _add(dim)
    for principle in system.get("textual_principles", []):
        _add(principle)

    return specs


def resolve_metric(field_id: str, rubric: dict | None = None) -> dict[str, Any] | None:
    """Return the metric spec for ``field_id`` or None if it has no allowed values."""
    return _metric_specs(rubric).get(field_id)


def build_judge_prompts(
    metric: dict[str, Any],
    text: str,
    *,
    variant: str,
    knowledge: str = "",
) -> tuple[str, str]:
    """Build (system, user) prompts for a single-metric judge."""
    system = _VARIANT_SYSTEM.get(variant, _VARIANT_SYSTEM["definition"])

    mid = metric["id"]
    values = metric["values"]
    allowed = ", ".join(json.dumps(v) for v in values)
    goal = metric.get("analysis_prompt") or metric.get("definition") or f"Classify {mid}."

    scoring = metric.get("scoring") or {}
    scoring_block = ""
    if scoring:
        # Prefer per-label glosses; otherwise low/mid/high
        label_keys = [k for k in scoring if k not in ("low", "mid", "high")]
        keys = [v for v in values if v in scoring] if label_keys else ["low", "mid", "high"]
        bits = [f"  - {k}: {scoring[k]}" for k in keys if k in scoring]
        if bits:
            scoring_block = "Label guide:\n" + "\n".join(bits) + "\n"

    knowledge_block = f"\nReference (Leech & Short):\n{knowledge.strip()}\n" if knowledge.strip() else ""

    user = f"""Metric: {metric.get('name', mid)} ({mid})
Definition: {metric.get('definition', '')}
Goal: {goal}
Allowed labels: [{allowed}]
{scoring_block}{knowledge_block}
Judge ONLY this metric for the passage below. Return JSON with exactly these keys:
{{"{mid}": one of [{allowed}], "evidence": "<short quote or phrase from the passage>"}}

Passage:
{text}"""

    return system, user


def _parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def judge_once(
    metric: dict[str, Any],
    text: str,
    *,
    variant: str,
    model: str = DEFAULT_MODEL,
    knowledge: str = "",
) -> dict[str, Any]:
    """Run one judge (one prompt variant). Returns a vote dict.

    Vote keys: variant, label (valid label or None), raw_label, evidence, parse_ok, error.
    """
    mid = metric["id"]
    values = set(metric["values"])
    system, user = build_judge_prompts(metric, text, variant=variant, knowledge=knowledge)

    vote: dict[str, Any] = {
        "variant": variant,
        "label": None,
        "raw_label": None,
        "evidence": None,
        "parse_ok": False,
        "error": None,
    }

    try:
        raw = llm_complete(
            user,
            system=system,
            model=model,
            max_tokens=256,
            temperature=0.05,
        )
    except LLMError as exc:
        vote["error"] = str(exc)
        return vote

    parsed = _parse_json(raw)
    if not parsed:
        vote["error"] = "json_parse_failed"
        return vote

    vote["parse_ok"] = True
    raw_label = parsed.get(mid)
    vote["raw_label"] = raw_label
    ev = parsed.get("evidence")
    if isinstance(ev, str):
        vote["evidence"] = ev[:200]
    if isinstance(raw_label, str) and raw_label in values:
        vote["label"] = raw_label
    else:
        vote["error"] = "label_not_allowed"
    return vote


def council_vote(votes: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine judge votes by majority (>=2 agreeing valid labels).

    Ties or fewer than 2 agreeing valid labels -> no consensus (value=None).
    """
    counts: dict[str, int] = {}
    for v in votes:
        label = v.get("label")
        if isinstance(label, str):
            counts[label] = counts.get(label, 0) + 1

    value: str | None = None
    agree_n = 0
    if counts:
        top_label, top_n = max(counts.items(), key=lambda kv: kv[1])
        # Reject ambiguous ties (two labels sharing the top count).
        tied = sum(1 for n in counts.values() if n == top_n)
        if top_n >= 2 and tied == 1:
            value = top_label
            agree_n = top_n

    return {
        "value": value,
        "agree_n": agree_n,
        "consensus": value is not None,
        "parse_ok": any(v.get("parse_ok") for v in votes),
        "votes": votes,
    }


def vote_weight(params_b: float, mode: str = "log") -> float:
    """Map a parameter count to a council vote weight.

    ``log`` (default) keeps a 550B teacher from erasing two 27B locals.
    ``linear`` is raw billions. ``equal`` ignores size.
    """
    if mode == "equal":
        return 1.0
    size = max(float(params_b or 0.0), 0.0)
    if size <= 0:
        return 1.0
    if mode == "linear":
        return size
    return math.log2(size + 1.0)


def weighted_council_vote(
    votes: list[dict[str, Any]],
    *,
    min_share: float = 0.5,
    min_voters: int = 2,
) -> dict[str, Any]:
    """Combine labelled votes with per-voter ``weight`` (defaults to 1).

    Consensus requires at least ``min_voters`` valid labels and a unique
    winner whose share of total weight is >= ``min_share``.
    """
    tallies: dict[str, float] = {}
    counts: dict[str, int] = {}
    total = 0.0
    n_valid = 0
    for vote in votes:
        label = vote.get("label")
        if not isinstance(label, str) or not label:
            continue
        weight = float(vote.get("weight") or 1.0)
        if weight <= 0:
            continue
        tallies[label] = tallies.get(label, 0.0) + weight
        counts[label] = counts.get(label, 0) + 1
        total += weight
        n_valid += 1

    value: str | None = None
    agree_n = 0
    share = 0.0
    if tallies and total > 0:
        top_label, top_w = max(tallies.items(), key=lambda kv: kv[1])
        tied = sum(1 for w in tallies.values() if abs(w - top_w) < 1e-9)
        share = top_w / total
        if n_valid >= min_voters and tied == 1 and share >= min_share:
            value = top_label
            agree_n = counts[top_label]

    return {
        "value": value,
        "agree_n": agree_n,
        "weight_share": round(share, 4) if total else 0.0,
        "weight_total": round(total, 4),
        "weight_by_label": {k: round(v, 4) for k, v in sorted(tallies.items())},
        "consensus": value is not None,
        "parse_ok": any(v.get("parse_ok") for v in votes),
        "n_valid": n_valid,
        "votes": votes,
    }


def _knowledge_for(field_id: str, text: str) -> str:
    """Best-effort compact Leech & Short snippet for this metric (offline-safe)."""
    try:
        from tools.style_classification.style_knowledge import format_context, retrieve

        chunks = retrieve(text, dimension_id=field_id, k=1)
        return format_context(chunks, max_chars=1200)
    except Exception:
        return ""


def build_arbitrator_prompts(
    metric: dict[str, Any],
    text: str,
    votes: list[dict[str, Any]],
    *,
    knowledge: str = "",
) -> tuple[str, str]:
    """Build (system, user) prompts for the arbitrator reviewing the judges."""
    mid = metric["id"]
    values = metric["values"]
    allowed = ", ".join(json.dumps(v) for v in values)

    judge_lines: list[str] = []
    for v in votes:
        label = v.get("label")
        evidence = v.get("evidence")
        label_str = json.dumps(label) if label is not None else "null (no valid label)"
        ev_str = json.dumps(evidence) if isinstance(evidence, str) else "null"
        judge_lines.append(f"- Judge [{v.get('variant')}]: label={label_str}, evidence={ev_str}")
    judges_block = "\n".join(judge_lines) if judge_lines else "- (no judge produced a usable label)"

    knowledge_block = f"\nReference (Leech & Short):\n{knowledge.strip()}\n" if knowledge.strip() else ""

    scoring = metric.get("scoring") or {}
    scoring_block = ""
    if scoring:
        label_keys = [k for k in scoring if k not in ("low", "mid", "high")]
        keys = [v for v in values if v in scoring] if label_keys else ["low", "mid", "high"]
        bits = [f"  - {k}: {scoring[k]}" for k in keys if k in scoring]
        if bits:
            scoring_block = "Label guide:\n" + "\n".join(bits) + "\n"

    user = f"""Metric: {metric.get('name', mid)} ({mid})
Definition: {metric.get('definition', '')}
Allowed labels: [{allowed}]
{scoring_block}{knowledge_block}
Three junior judges reviewed the passage for this metric:
{judges_block}

As the senior arbitrator, weigh the evidence against the passage and decide the single most
defensible label. Return JSON with exactly these keys:
{{"{mid}": one of [{allowed}] or null, "rationale": "<one sentence citing the deciding evidence>"}}

Passage:
{text}"""

    return _ARBITRATOR_SYSTEM, user


def arbitrate(
    metric: dict[str, Any],
    text: str,
    votes: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    knowledge: str = "",
) -> dict[str, Any]:
    """Arbitrator reviews the judges' evidence and decides. Returns a decision dict.

    Keys: value (valid label or None), rationale, parse_ok, error.
    """
    mid = metric["id"]
    values = set(metric["values"])
    system, user = build_arbitrator_prompts(metric, text, votes, knowledge=knowledge)

    decision: dict[str, Any] = {"value": None, "rationale": None, "parse_ok": False, "error": None}
    try:
        raw = llm_complete(user, system=system, model=model, max_tokens=384, temperature=0.0)
    except LLMError as exc:
        decision["error"] = str(exc)
        return decision

    parsed = _parse_json(raw)
    if not parsed:
        decision["error"] = "json_parse_failed"
        return decision

    decision["parse_ok"] = True
    rationale = parsed.get("rationale")
    if isinstance(rationale, str):
        decision["rationale"] = rationale[:300]
    label = parsed.get(mid)
    if isinstance(label, str) and label in values:
        decision["value"] = label
    elif label is None:
        decision["error"] = "arbitrator_abstained"
    else:
        decision["error"] = "label_not_allowed"
    return decision


def assess_metric_council(
    text: str,
    field: str,
    *,
    model: str = DEFAULT_MODEL,
    rubric: dict | None = None,
    variants: tuple[str, ...] = COUNCIL_VARIANTS,
    arbitrate_split: bool = True,
    arbitrator_model: str | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Judge a single metric with the full council + optional arbitration.

    The three judges vote. If they are unanimous the label stands. Otherwise (split
    or any judge failing to produce a valid label), the arbitrator reviews their
    evidence and the passage and decides — unless ``arbitrate_split`` is False, in
    which case plain majority is used.

    Returns (label | None, meta). ``meta`` carries: value, method, consensus,
    parse_ok, majority {value, agree_n}, arbitration {value, rationale} | None, votes.
    """
    metric = resolve_metric(field, rubric)
    if metric is None:
        return None, {"value": None, "method": "unknown_metric", "consensus": False,
                      "parse_ok": False, "error": "unknown_metric",
                      "majority": None, "arbitration": None, "votes": []}

    knowledge = _knowledge_for(field, text)
    votes = [
        judge_once(metric, text, variant=variant, model=model, knowledge=knowledge)
        for variant in variants
    ]
    majority = council_vote(votes)

    labels = [v.get("label") for v in votes]
    unanimous = bool(labels) and all(l is not None for l in labels) and len(set(labels)) == 1

    arbitration: dict[str, Any] | None = None
    if unanimous:
        value = majority["value"]
        method = "unanimous"
    elif arbitrate_split:
        arbitration = arbitrate(
            metric, text, votes, model=arbitrator_model or model, knowledge=knowledge
        )
        value = arbitration["value"]
        method = "arbitrated"
    else:
        value = majority["value"]
        method = "majority"

    meta = {
        "value": value,
        "method": method,
        "consensus": value is not None,
        "parse_ok": bool(majority["parse_ok"] or (arbitration or {}).get("parse_ok")),
        "majority": {"value": majority["value"], "agree_n": majority["agree_n"]},
        "arbitration": arbitration,
        "votes": votes,
    }
    return value, meta


def assess_fields_council(
    text: str,
    fields: Any,
    *,
    model: str = DEFAULT_MODEL,
    rubric: dict | None = None,
    prior: dict[str, Any] | None = None,
    variants: tuple[str, ...] = COUNCIL_VARIANTS,
    arbitrate_split: bool = True,
    arbitrator_model: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the council (+ arbitration) over each requested field.

    Returns (profile_partial, council_meta):
      profile_partial: {field: decided_label} for fields that resolved to a label
      council_meta:    {field: {value, method, agree_n, consensus, arbitration, votes}}

    Fields already present (non-None) in ``prior`` are reused, not re-judged.
    Truncates over-long passages the same way as the joint judge.
    """
    prior = prior or {}
    words = text.split()
    if len(words) > 1200:
        text = " ".join(words[:1200]) + "…"

    profile: dict[str, Any] = {}
    meta: dict[str, Any] = {}

    for field in fields:
        prior_val = prior.get(field)
        if isinstance(prior_val, str) and prior_val:
            profile[field] = prior_val
            meta[field] = {"value": prior_val, "method": "reused_prior", "agree_n": 0,
                           "consensus": True, "parse_ok": True, "reused_prior": True,
                           "arbitration": None, "votes": []}
            continue

        label, field_meta = assess_metric_council(
            text, field, model=model, rubric=rubric, variants=variants,
            arbitrate_split=arbitrate_split, arbitrator_model=arbitrator_model,
        )
        arb = field_meta.get("arbitration")
        meta[field] = {
            "value": field_meta["value"],
            "method": field_meta["method"],
            "agree_n": field_meta.get("majority", {}).get("agree_n", 0),
            "consensus": field_meta["consensus"],
            "parse_ok": field_meta["parse_ok"],
            "arbitration": (
                {"value": arb.get("value"), "rationale": arb.get("rationale")}
                if arb else None
            ),
            "votes": [
                {kk: vv for kk, vv in vote.items() if kk in ("variant", "label", "evidence")}
                for vote in field_meta.get("votes", [])
            ],
        }
        if label is not None:
            profile[field] = label

    return profile, meta
