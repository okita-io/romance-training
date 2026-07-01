"""
Style benchmark helpers — prompt construction and profile deltas.

See eval/style_benchmark/README.md and fixture.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / "eval" / "style_benchmark" / "fixture.json"

# LLM + textual-principle keys used for delta scoring (matches pass_config).
COMPARE_FIELDS: tuple[str, ...] = (
    "lexical_complexity",
    "register",
    "sentence_complexity",
    "figurative_density",
    "cohesion",
    "pov",
    "narrative_distance",
    "free_indirect_discourse",
    "tone",
    "mind_style",
    "segmentation",
    "prose_rhythm",
    "end_focus",
    "subordination_salience",
    "textual_relations",
    "climax",
)

_LAYER_HINTS: dict[str, str] = {
    "lexical_complexity": "Lexis — vocabulary complexity and semantic fields",
    "register": "Lexis — situational register",
    "sentence_complexity": "Grammar — parataxis vs hypotaxis",
    "figurative_density": "Figures — metaphor, simile, foregrounding",
    "cohesion": "Cohesion — links between sentences",
    "pov": "Viewpoint — narrative person and access",
    "narrative_distance": "Viewpoint — psychological proximity",
    "free_indirect_discourse": "Viewpoint — narrator/character voice blend",
    "tone": "Context — affective and attitudinal register",
    "mind_style": "Viewpoint — conceptual worldview in language",
    "segmentation": "Textual dynamics — graphic/tone units",
    "prose_rhythm": "Textual dynamics — rhythmic tempo",
    "end_focus": "Textual dynamics — end-focus salience",
    "subordination_salience": "Textual dynamics — foreground vs background",
    "textual_relations": "Textual dynamics — given/new information",
    "climax": "Textual dynamics — climactic build",
}


def load_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scene_type_map(fixture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in fixture["scene_types"]}


def style_target_lines(style_target: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in COMPARE_FIELDS:
        value = style_target.get(key)
        if value is None:
            continue
        hint = _LAYER_HINTS.get(key, key)
        lines.append(f"- {key}: {value} ({hint})")
    return lines


def build_system_prompt() -> str:
    return (
        "You are a literary fiction writer trained in Geoffrey Leech and Mick Short's "
        "*Style in Fiction* framework. Write original prose that deliberately embodies "
        "the requested stylistic dimensions — not by naming them, but through observable "
        "linguistic choices in lexis, grammar, figures, cohesion, viewpoint, and textual dynamics."
    )


def build_scene_prompt(
    fixture: dict[str, Any],
    plot: dict[str, Any],
    scene_type_id: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a single benchmark scene."""
    scene = scene_type_map(fixture)[scene_type_id]
    world = fixture["world"]
    lo, hi = scene["word_range"]

    style_block = "\n".join(style_target_lines(plot["style_target"]))

    user = f"""World: {world["name"]} — {world["setting"]}

Plot: {plot["title"]}
Summary: {plot["summary"]}
Female lead: {plot["female_lead"]}
Male lead: {plot["male_lead"]}

Scene task ({scene["label"]}):
{scene["instruction"]}

Target length: {lo}–{hi} words.

Style profile to embody ("{plot["style_label"]}"):
{style_block}

Write only the prose passage. No title, no meta-commentary, no bullet lists."""

    return build_system_prompt(), user


def build_setup_prompt(fixture: dict[str, Any]) -> tuple[str, str]:
    """Pass-1 style prompt: document world roster (optional generation check)."""
    world = fixture["world"]
    female = ", ".join(fixture["female_leads"])
    male = ", ".join(fixture["male_leads"])
    plot_lines = "\n".join(
        f"{p['id']}. {p['title']} — {p['female_lead']} / {p['male_lead']}: {p['summary']}"
        for p in fixture["plots"]
    )

    user = f"""Fantasy world bible excerpt for {world["name"]}.

Setting: {world["setting"]}
Era: {world["era"]}

Female lead roster: {female}
Male lead roster: {male}

Seven plot seeds (keep names and continuity):
{plot_lines}

Write a concise 300-word world primer that a writer could use to keep continuity across all seven stories. Mention each lead pair once."""

    return build_system_prompt(), user


def compute_delta(
    target: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    """Field-level match report between target and classified profile."""
    fields: dict[str, dict[str, Any]] = {}
    matches = 0
    compared = 0

    for key in COMPARE_FIELDS:
        t_val = target.get(key)
        a_val = actual.get(key)
        if t_val is None:
            continue
        compared += 1
        matched = a_val == t_val
        if matched:
            matches += 1
        fields[key] = {
            "target": t_val,
            "actual": a_val,
            "match": matched,
        }

    score = (matches / compared) if compared else 0.0
    return {
        "fields": fields,
        "match_count": matches,
        "compared_count": compared,
        "match_score": round(score, 4),
    }


def make_run_id(label: str | None = None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = (label or "run").replace(" ", "_").replace("/", "-")[:40]
    return f"{ts}_{safe}"


def iter_benchmark_cases(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    """All plot × scene_type combinations."""
    cases: list[dict[str, Any]] = []
    for plot in fixture["plots"]:
        for scene in fixture["scene_types"]:
            cases.append(
                {
                    "plot_id": plot["id"],
                    "plot_title": plot["title"],
                    "scene_type": scene["id"],
                    "scene_label": scene["label"],
                    "style_label": plot["style_label"],
                    "style_target": plot["style_target"],
                    "female_lead": plot["female_lead"],
                    "male_lead": plot["male_lead"],
                }
            )
    return cases


def aggregate_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize match scores across a completed benchmark run."""
    if not records:
        return {"count": 0}

    scores = [r["delta"]["match_score"] for r in records if r.get("delta")]
    by_plot: dict[str, list[float]] = {}
    by_scene: dict[str, list[float]] = {}
    by_field: dict[str, list[bool]] = {k: [] for k in COMPARE_FIELDS}

    for rec in records:
        delta = rec.get("delta")
        if not delta:
            continue
        pid = rec["plot_id"]
        sid = rec["scene_type"]
        by_plot.setdefault(pid, []).append(delta["match_score"])
        by_scene.setdefault(sid, []).append(delta["match_score"])
        for key, info in delta.get("fields", {}).items():
            by_field.setdefault(key, []).append(info["match"])

    def _mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "count": len(records),
        "mean_match_score": _mean(scores),
        "by_plot": {k: _mean(v) for k, v in sorted(by_plot.items())},
        "by_scene": {k: _mean(v) for k, v in sorted(by_scene.items())},
        "field_hit_rate": {
            k: round(sum(v) / len(v), 4) if v else 0.0
            for k, v in sorted(by_field.items())
            if v
        },
    }


def compare_runs(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare two benchmark result sets keyed by plot_id + scene_type."""
    def _key(r: dict[str, Any]) -> tuple[str, str]:
        return (r["plot_id"], r["scene_type"])

    base_map = {_key(r): r for r in baseline}
    cand_map = {_key(r): r for r in candidate}
    keys = sorted(set(base_map) & set(cand_map))

    rows: list[dict[str, Any]] = []
    for key in keys:
        b = base_map[key]
        c = cand_map[key]
        b_score = b.get("delta", {}).get("match_score", 0.0)
        c_score = c.get("delta", {}).get("match_score", 0.0)
        rows.append(
            {
                "plot_id": key[0],
                "scene_type": key[1],
                "baseline_score": b_score,
                "candidate_score": c_score,
                "delta": round(c_score - b_score, 4),
                "baseline_model": b.get("model"),
                "candidate_model": c.get("model"),
            }
        )

    improved = sum(1 for r in rows if r["delta"] > 0)
    regressed = sum(1 for r in rows if r["delta"] < 0)
    mean_delta = round(sum(r["delta"] for r in rows) / len(rows), 4) if rows else 0.0

    return {
        "paired_count": len(rows),
        "mean_score_delta": mean_delta,
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(rows) - improved - regressed,
        "rows": rows,
        "baseline_summary": aggregate_run(baseline),
        "candidate_summary": aggregate_run(candidate),
    }


def load_results_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
