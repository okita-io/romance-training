"""
Style benchmark helpers — prompt construction and profile deltas.

Prompt shape mirrors romance-factory ``PromptBuilder.build_act_generation_prompt``
(Phase 7 act prose): same system voice, length budget, verbosity / narrative-purpose
contracts, and ``format_style_targets``-style block. See eval/style_benchmark/README.md.

Character names are assigned per run via :func:`materialize_fixture_for_run`.
"""

from __future__ import annotations

import copy
import json
import random
import re
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

# Mirrors prompt_engineering/system_prompts.json → act_generation.
_RF_ACT_GENERATION_SYSTEM = (
    "You are a prose generation engine. You output ONLY narrative fiction prose. "
    "You never address the reader or user. You never explain what you are writing or why. "
    "You never include introductions like 'Here is the scene' or 'I wrote this for you'. "
    "You never include meta-commentary, bullet points, or summaries. "
    "Your entire output is the scene text as it would appear in a published romance novel "
    "— nothing else. You write one act at scene scale: follow the length budget in the user "
    "prompt and do not produce tens of thousands of words in a single response."
)

_VERBOSITY_BLOCKS: dict[int, str] = {
    0: """LEVEL 0 (TERSE):
- 5–10 word sentences; one idea each
- Dialogue and action carry the scene
- No interior monologue unless the beat requires it""",
    1: """LEVEL 1 (MODERATE):
- 10–18 word sentences; selective detail
- Balance dialogue, action, and brief interiority
- Steady pacing; no scene-length bloat""",
    2: """LEVEL 2 (ELABORATE):
- 15–30 word sentences when the beat earns it
- Layered sensory and emotional texture allowed
- Reflection and setting detail OK; no filler""",
}

_NARRATIVE_PURPOSE_BLOCKS: dict[int, str] = {
    0: """LEVEL 0 (WORLD-FORWARD):
- Situation, rules, and stakes lead; romance is subtext or deferred
- Ground the reader in place, faction, and constraint before couple chemistry""",
    1: """LEVEL 1 (BALANCED):
- Relationship tension and world texture share the scene
- Neither the couple nor the setting should disappear""",
    2: """LEVEL 2 (ROMANCE-FORWARD):
- The relationship engine leads scene turns: desire, fear, misunderstanding, consent, proximity, and choice between leads shape what happens next.
- World and setting appear as texture that touches emotion and conflict — avoid freestanding exposition blocks that do not press the couple's arc in this act.""",
}

# Syllable pools for deterministic per-run fantasy names (Ashenmere register).
_FEMALE_GIVEN = (
    "Ael", "Mira", "Ves", "Bron", "Elo", "Sab", "Iso", "Cal", "Ner", "Thal",
    "Ser", "Lys", "Cor", "Fen", "Ryn", "Ash", "Mor", "Sel", "Vey", "Kael",
)
_MALE_GIVEN = (
    "Cas", "Thor", "Ald", "Ron", "Dar", "Evan", "Luc", "Garr", "Tor", "Mal",
    "Bren", "Cor", "Fin", "Hal", "Jor", "Kael", "Lorn", "Merr", "Os", "Perr",
)
_NAME_SUFFIX = (
    "iana", "ira", "pera", "onte", "owen", "able", "olde", "essa", "ara", "yn",
    "ian", "ne", "ric", "an", "ius", "ander", "ien", "row", "ell", "as",
)
_FAMILY = (
    "Thornweave", "Solacewright", "Kael", "Ashford", "Duskmire", "Wynter",
    "Ravencrest", "Vale", "Blackwell", "Fenn", "Greywing", "Holt", "Storm",
    "Merrow", "Saltmere", "Tideborn", "Ashwick", "Brinemark", "Cinderfold",
    "Deeproot", "Emberline", "Foghart", "Gullmere", "Harrowgate", "Ironspool",
    "Jettwater", "Kestrel", "Lowtide", "Mistral", "Nightreef",
)

_NAME_TEMPLATE_RE = re.compile(r"\{\{(female_lead|male_lead)\}\}")


def load_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scene_type_map(fixture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in fixture["scene_types"]}


def verbosity_level_for_word_range(lo: int, hi: int) -> int:
    """Map benchmark scene word_range to RF verbosity level 0–2."""
    mid = (lo + hi) / 2
    if mid <= 200:
        return 0
    if mid <= 325:
        return 1
    return 2


def verbosity_word_budget(level: int) -> tuple[int, int]:
    """Same bands as romance_factory.generate.verbosity.VERBOSITY_WORD_BUDGETS."""
    budgets = {0: (150, 250), 1: (250, 400), 2: (400, 600)}
    return budgets.get(max(0, min(2, level)), budgets[1])


def format_act_length_budget_block(level: int) -> str:
    """Mirror romance_factory.generate.verbosity.format_act_length_budget_block."""
    lv = max(0, min(2, level))
    min_w, max_w = verbosity_word_budget(lv)
    return (
        "## Rough-draft length budget (Phase 7 — one act only)\n\n"
        "This call must produce **a single act** (one outline segment), not a "
        "full chapter or novella.\n\n"
        f"- **Verbosity level {lv} budget:** aim for **{min_w}–{max_w} words** "
        "of in-world scene prose (dialogue, action, interiority — not beat-sheet "
        "recap or outline voice).\n"
        f"- **Hard cap:** do not exceed **{max_w} words**. Near the cap, "
        "accelerate to the act's final beat and close cleanly.\n"
        "- **Stop rule:** when the outline beats are on the page, **end**. "
        "Do not pad, spiral, or add fresh cycles of unrelated material."
    )


def format_verbosity_prompt_block(level: int) -> str:
    """Mirror romance_factory.generate.verbosity.format_verbosity_prompt_block."""
    lv = max(0, min(2, level))
    body = _VERBOSITY_BLOCKS.get(lv, _VERBOSITY_BLOCKS[1])
    min_w, max_w = verbosity_word_budget(lv)
    return (
        f"Write this act at **VERBOSITY LEVEL {lv}** ({min_w}–{max_w} words).\n\n"
        f"{body}\n\n"
        "Match this density exactly. Do not exceed the word budget or pad with "
        "extra description, explanation, or rhetorical flourish."
    )


def format_narrative_purpose_prompt_block(level: int) -> str:
    """Mirror romance_factory.generate.narrative_purpose.format_narrative_purpose_prompt_block."""
    lv = max(0, min(2, level))
    body = _NARRATIVE_PURPOSE_BLOCKS.get(lv, _NARRATIVE_PURPOSE_BLOCKS[1])
    return (
        f"You must honor NARRATIVE PURPOSE LEVEL {lv} for this act.\n\n"
        f"{body}\n\n"
        "If this purpose level ever conflicts with a generic genre reminder, "
        "follow the act outline and this NARRATIVE PURPOSE LEVEL first."
    )


def _sentence_length_band(style_target: dict[str, Any], verbosity: int) -> tuple[int, int]:
    """Derive mean sentence length range from style + verbosity (outline beat proxy)."""
    sc = str(style_target.get("sentence_complexity") or "moderate")
    if sc == "simple_paratactic":
        base = (8, 14)
    elif sc == "complex_hypotactic":
        base = (22, 32)
    else:
        base = (14, 22)
    if verbosity == 0:
        return (max(8, base[0] - 2), max(base[0], base[1] - 4))
    if verbosity == 2:
        return (base[0] + 2, base[1] + 4)
    return base


def _lexical_density_band(style_target: dict[str, Any]) -> tuple[float, float]:
    lc = str(style_target.get("lexical_complexity") or "neutral")
    if lc == "simple_colloquial":
        return (0.35, 0.45)
    if lc == "complex_literary":
        return (0.55, 0.65)
    return (0.45, 0.55)


def format_style_targets_prompt_block(
    style_target: dict[str, Any],
    *,
    verbosity: int,
) -> str:
    """Mirror romance_factory.style.targets.format_style_targets_prompt_block."""
    st = dict(style_target)
    sl = _sentence_length_band(st, verbosity)
    ld = _lexical_density_band(st)
    lines = [
        "## Style targets (Leech & Short — from outline beat)",
        "Write at these linguistic targets for this act. Match the **profile**, not a reference plot.",
    ]
    for key, label in (
        ("register", "Register"),
        ("pov", "POV"),
        ("narrative_distance", "Narrative distance"),
        ("free_indirect_discourse", "Free indirect discourse"),
        ("sentence_complexity", "Sentence complexity"),
    ):
        if st.get(key):
            lines.append(f"- **{label}:** {st[key]}")
    lines.append(f"- **Mean sentence length (words):** {sl[0]}–{sl[1]}")
    lines.append(f"- **Lexical density:** {ld[0]:.2f}–{ld[1]:.2f}")
    for key in ("end_focus", "prose_rhythm", "segmentation", "figurative_density", "tone"):
        if st.get(key):
            lines.append(f"- **{key.replace('_', ' ').title()}:** {st[key]}")
    # Benchmark-only rubric dimensions (classifier still scores these).
    extra = [
        k
        for k in (
            "lexical_complexity",
            "cohesion",
            "mind_style",
            "subordination_salience",
            "textual_relations",
            "climax",
        )
        if st.get(k)
    ]
    if extra:
        lines.append("\n**Additional stylistic targets (rubric):**")
        for key in extra:
            lines.append(f"- **{key.replace('_', ' ').title()}:** {st[key]}")
    return "\n".join(lines)


def _make_unique_name(rng: random.Random, given_pool: tuple[str, ...], used: set[str]) -> str:
    for _ in range(200):
        given = rng.choice(given_pool) + rng.choice(_NAME_SUFFIX)
        family = rng.choice(_FAMILY)
        name = f"{given.title()} {family}"
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError("Could not generate a unique name after 200 attempts")


def generate_run_roster(seed: str, *, plot_count: int = 7) -> tuple[list[str], list[str]]:
    """Return (female_leads, male_leads) — one pair per plot, unique within the run."""
    rng = random.Random(seed)
    used: set[str] = set()
    females = [_make_unique_name(rng, _FEMALE_GIVEN, used) for _ in range(plot_count)]
    males = [_make_unique_name(rng, _MALE_GIVEN, used) for _ in range(plot_count)]
    return females, males


def _substitute_names(text: str, *, female_lead: str, male_lead: str) -> str:
    return (
        text.replace("{{female_lead}}", female_lead).replace("{{male_lead}}", male_lead)
    )


def materialize_fixture_for_run(
    fixture: dict[str, Any],
    seed: str,
    *,
    name_mode: str = "llm",
    name_model: str | None = None,
    max_name_retries: int = 100,
    fixed_roster: tuple[list[str], list[str]] | None = None,
    complete_fn: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Deep-copy fixture with per-run character names and resolved plot summaries."""
    out = copy.deepcopy(fixture)
    naming_report: dict[str, Any] | None = None

    if fixed_roster is not None:
        females, males = fixed_roster
        naming_report = {
            "mode": "fixed",
            "name_seed": seed,
            "total_characters": len(females) + len(males),
            "total_attempts": 0,
            "mean_attempts": 0.0,
        }
    elif name_mode == "syllable" or dry_run:
        females, males = generate_run_roster(seed, plot_count=len(out["plots"]))
        naming_report = {
            "mode": "syllable" if not dry_run else "dry_run_syllable",
            "name_seed": seed,
            "total_characters": len(females) + len(males),
            "total_attempts": 0,
            "mean_attempts": 0.0,
        }
    else:
        from tools.style_evaluation.character_namer import generate_llm_roster

        females, males, report = generate_llm_roster(
            out,
            name_seed=seed,
            model=name_model or "local-model",
            max_retries=max_name_retries,
            complete_fn=complete_fn,
            dry_run=dry_run,
        )
        naming_report = report.to_dict()

    out["run_seed"] = seed
    out["female_leads"] = list(females)
    out["male_leads"] = list(males)
    out["naming"] = naming_report

    for i, plot in enumerate(out["plots"]):
        female = females[i]
        male = males[i]
        plot["female_lead"] = female
        plot["male_lead"] = male
        template = plot.get("summary_template") or plot.get("summary", "")
        plot["summary"] = _substitute_names(template, female_lead=female, male_lead=male)
    return out


def build_system_prompt() -> str:
    return _RF_ACT_GENERATION_SYSTEM


def build_scene_prompt(
    fixture: dict[str, Any],
    plot: dict[str, Any],
    scene_type_id: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a single benchmark scene."""
    scene = scene_type_map(fixture)[scene_type_id]
    world = fixture["world"]
    chapter = int(scene.get("chapter") or 1)
    act = int(scene.get("act") or 10)
    rf_level = int(scene.get("romance_focus_level") or 1)
    lo, hi = scene["word_range"]
    verbosity = verbosity_level_for_word_range(lo, hi)
    v_min, v_max = verbosity_word_budget(verbosity)
    style_target = plot["style_target"]

    sections: list[str] = [
        f"Write act {act} of chapter {chapter}. "
        "Output ONLY the prose — start directly with the narrative. "
        "No preamble, no explanation, no 'here is the scene'.",
        format_act_length_budget_block(verbosity),
        (
            "## World Lore (retrieved)\n\n"
            f"**{world['name']}** — {world['setting']}\n\n"
            f"Era: {world['era']}"
        ),
        (
            "## Act Outline\n\n"
            f"**Plot:** {plot['title']}\n"
            f"**Summary:** {plot['summary']}\n"
            f"**Female lead:** {plot['female_lead']}\n"
            f"**Male lead:** {plot['male_lead']}\n\n"
            f"**Scene task ({scene['label']}):** {scene['instruction']}"
        ),
        "## Verbosity contract\n\n" + format_verbosity_prompt_block(verbosity),
        "## Narrative purpose contract\n\n" + format_narrative_purpose_prompt_block(rf_level),
        format_style_targets_prompt_block(style_target, verbosity=verbosity),
        "\n".join(
            [
                "## Constraints",
                "",
                f"- **LENGTH:** Honor the rough-draft length budget — **{v_min}–{v_max} words** "
                f"for verbosity level {verbosity}; stop at the natural scene end.",
                "- Stay faithful to the act outline and character names above.",
                "- Maintain consistent character voices and relationship dynamics.",
                "- Do not introduce characters or plot points not in the outline.",
                "- **IN-SCENE THROUGH THE LAST LINE:** Close with what characters perceive, do, "
                "feel, or say — not meta summary or jacket-copy narration.",
            ]
        ),
    ]

    user = "\n\n".join(sections)
    return build_system_prompt(), user


def build_setup_prompt(fixture: dict[str, Any]) -> tuple[str, str]:
    """Pass-1 style prompt: document world roster (optional generation check)."""
    world = fixture["world"]
    females = fixture.get("female_leads") or []
    males = fixture.get("male_leads") or []
    female = ", ".join(females)
    male = ", ".join(males)
    plot_lines = "\n".join(
        f"{p['id']}. {p['title']} — {p.get('female_lead', '?')} / {p.get('male_lead', '?')}: {p.get('summary', '')}"
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


# Backward-compatible alias for tests that referenced the old hint format.
def style_target_lines(style_target: dict[str, Any]) -> list[str]:
    block = format_style_targets_prompt_block(style_target, verbosity=1)
    return [ln for ln in block.splitlines() if ln.startswith("- **")]


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


def _case_key(r: dict[str, Any]) -> tuple[str, str]:
    return (r["plot_id"], r["scene_type"])


def _field_hit_rate_delta(
    earlier: dict[str, float],
    later: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Per-dimension hit-rate change between two aggregate summaries."""
    out: dict[str, dict[str, Any]] = {}
    for key in COMPARE_FIELDS:
        before = earlier.get(key)
        after = later.get(key)
        if before is None and after is None:
            continue
        before_v = before if before is not None else 0.0
        after_v = after if after is not None else 0.0
        delta = round(after_v - before_v, 4)
        out[key] = {
            "before": before_v,
            "after": after_v,
            "delta": delta,
            "direction": "improved" if delta > 0 else ("regressed" if delta < 0 else "unchanged"),
        }
    return out


def session_summary(
    records: list[dict[str, Any]],
    *,
    label: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Aggregate one benchmark run plus identifying metadata."""
    agg = aggregate_run(records)
    meta: dict[str, Any] = {
        "label": label,
        "source": source,
        "run_id": None,
        "model": None,
        "classifier_model": None,
        "classify_pass": None,
        "name_seed": None,
        "naming": None,
    }
    for rec in records:
        if rec.get("phase") == "setup":
            continue
        for key in ("run_id", "model", "classifier_model", "classify_pass", "name_seed", "naming"):
            if meta.get(key) is None and rec.get(key) is not None:
                meta[key] = rec[key]
        if meta["label"] is None and rec.get("label"):
            meta["label"] = rec["label"]
    if meta["label"] is None and source:
        meta["label"] = Path(source).stem
    return {**meta, **agg}


def compare_runs(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare two benchmark result sets keyed by plot_id + scene_type."""
    base_map = {_case_key(r): r for r in baseline}
    cand_map = {_case_key(r): r for r in candidate}
    keys = sorted(set(base_map) & set(cand_map))

    rows: list[dict[str, Any]] = []
    field_improved: dict[str, int] = {k: 0 for k in COMPARE_FIELDS}
    field_regressed: dict[str, int] = {k: 0 for k in COMPARE_FIELDS}
    field_unchanged: dict[str, int] = {k: 0 for k in COMPARE_FIELDS}

    for key in keys:
        b = base_map[key]
        c = cand_map[key]
        b_delta = b.get("delta") or {}
        c_delta = c.get("delta") or {}
        b_score = b_delta.get("match_score", 0.0)
        c_score = c_delta.get("match_score", 0.0)
        field_changes: dict[str, dict[str, Any]] = {}
        for dim, b_info in b_delta.get("fields", {}).items():
            c_info = c_delta.get("fields", {}).get(dim) or {}
            b_match = bool(b_info.get("match"))
            c_match = bool(c_info.get("match"))
            if c_match and not b_match:
                field_improved[dim] = field_improved.get(dim, 0) + 1
                change = "improved"
            elif b_match and not c_match:
                field_regressed[dim] = field_regressed.get(dim, 0) + 1
                change = "regressed"
            else:
                field_unchanged[dim] = field_unchanged.get(dim, 0) + 1
                change = "unchanged"
            field_changes[dim] = {
                "baseline_match": b_match,
                "candidate_match": c_match,
                "change": change,
            }
        rows.append(
            {
                "plot_id": key[0],
                "scene_type": key[1],
                "baseline_score": b_score,
                "candidate_score": c_score,
                "delta": round(c_score - b_score, 4),
                "baseline_model": b.get("model"),
                "candidate_model": c.get("model"),
                "field_changes": field_changes,
            }
        )

    improved = sum(1 for r in rows if r["delta"] > 0)
    regressed = sum(1 for r in rows if r["delta"] < 0)
    mean_delta = round(sum(r["delta"] for r in rows) / len(rows), 4) if rows else 0.0

    base_summary = aggregate_run(baseline)
    cand_summary = aggregate_run(candidate)

    return {
        "paired_count": len(rows),
        "mean_score_delta": mean_delta,
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(rows) - improved - regressed,
        "rows": rows,
        "baseline_summary": base_summary,
        "candidate_summary": cand_summary,
        "field_hit_rate_delta": _field_hit_rate_delta(
            base_summary.get("field_hit_rate", {}),
            cand_summary.get("field_hit_rate", {}),
        ),
        "field_case_counts": {
            "improved": {k: v for k, v in sorted(field_improved.items()) if v},
            "regressed": {k: v for k, v in sorted(field_regressed.items()) if v},
            "unchanged": {k: v for k, v in sorted(field_unchanged.items()) if v},
        },
    }


def compare_training_sessions(
    sessions: list[tuple[str, list[dict[str, Any]]]],
    *,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compare an ordered list of benchmark runs (baseline → training sessions).

    Each session is (label, records). Returns conformity trends, step deltas,
    per-field hit-rate trajectories, and each session vs the baseline.
    """
    if len(sessions) < 2:
        raise ValueError("compare_training_sessions requires at least two sessions")

    srcs = sources or [None] * len(sessions)
    summaries = [
        session_summary(records, label=label, source=src)
        for (label, records), src in zip(sessions, srcs)
    ]

    mean_scores = [s.get("mean_match_score", 0.0) for s in summaries]
    first_score = mean_scores[0]
    last_score = mean_scores[-1]
    overall_delta = round(last_score - first_score, 4)

    steps: list[dict[str, Any]] = []
    for i in range(len(sessions) - 1):
        from_label, from_records = sessions[i]
        to_label, to_records = sessions[i + 1]
        step = compare_runs(from_records, to_records)
        steps.append(
            {
                "from_label": from_label,
                "to_label": to_label,
                "mean_score_before": mean_scores[i],
                "mean_score_after": mean_scores[i + 1],
                "mean_score_delta": step["mean_score_delta"],
                "improved_cases": step["improved"],
                "regressed_cases": step["regressed"],
                "unchanged_cases": step["unchanged"],
                "field_hit_rate_delta": step["field_hit_rate_delta"],
            }
        )

    vs_baseline: list[dict[str, Any]] = []
    baseline_records = sessions[0][1]
    for i in range(1, len(sessions)):
        label, records = sessions[i]
        cmp = compare_runs(baseline_records, records)
        vs_baseline.append(
            {
                "label": label,
                "mean_match_score": summaries[i].get("mean_match_score", 0.0),
                "vs_baseline_delta": cmp["mean_score_delta"],
                "improved_cases": cmp["improved"],
                "regressed_cases": cmp["regressed"],
                "field_hit_rate_delta": cmp["field_hit_rate_delta"],
            }
        )

    field_trends: dict[str, dict[str, Any]] = {}
    for key in COMPARE_FIELDS:
        rates = [s.get("field_hit_rate", {}).get(key, 0.0) for s in summaries]
        if not any(rates):
            continue
        deltas = [round(rates[j + 1] - rates[j], 4) for j in range(len(rates) - 1)]
        total_delta = round(rates[-1] - rates[0], 4)
        field_trends[key] = {
            "hit_rates": rates,
            "step_deltas": deltas,
            "first_to_last_delta": total_delta,
            "direction": (
                "improved" if total_delta > 0 else ("regressed" if total_delta < 0 else "unchanged")
            ),
        }

    plot_trends: dict[str, dict[str, Any]] = {}
    plot_ids = sorted({pid for s in summaries for pid in s.get("by_plot", {})})
    for pid in plot_ids:
        scores = [s.get("by_plot", {}).get(pid, 0.0) for s in summaries]
        plot_trends[pid] = {
            "mean_match_scores": scores,
            "first_to_last_delta": round(scores[-1] - scores[0], 4),
        }

    scene_trends: dict[str, dict[str, Any]] = {}
    scene_ids = sorted({sid for s in summaries for sid in s.get("by_scene", {})})
    for sid in scene_ids:
        scores = [s.get("by_scene", {}).get(sid, 0.0) for s in summaries]
        scene_trends[sid] = {
            "mean_match_scores": scores,
            "first_to_last_delta": round(scores[-1] - scores[0], 4),
        }

    naming_trend: dict[str, Any] | None = None
    mean_attempts = [
        (s.get("naming") or {}).get("mean_attempts")
        for s in summaries
        if (s.get("naming") or {}).get("mean_attempts") is not None
    ]
    if len(mean_attempts) >= 2:
        naming_trend = {
            "mean_attempts": mean_attempts,
            "first_to_last_delta": round(mean_attempts[-1] - mean_attempts[0], 4),
        }

    improved_fields = sorted(k for k, v in field_trends.items() if v["first_to_last_delta"] > 0)
    regressed_fields = sorted(k for k, v in field_trends.items() if v["first_to_last_delta"] < 0)

    return {
        "session_count": len(sessions),
        "sessions": summaries,
        "conformity_trend": {
            "mean_match_scores": mean_scores,
            "first_to_last_delta": overall_delta,
            "direction": (
                "improved" if overall_delta > 0 else ("regressed" if overall_delta < 0 else "unchanged")
            ),
            "steps": steps,
        },
        "vs_baseline": vs_baseline,
        "field_trends": field_trends,
        "fields_improved": improved_fields,
        "fields_regressed": regressed_fields,
        "plot_trends": plot_trends,
        "scene_trends": scene_trends,
        "naming_trend": naming_trend,
    }


def format_sessions_report(report: dict[str, Any]) -> str:
    """Human-readable summary for terminal output."""
    lines: list[str] = []
    trend = report["conformity_trend"]
    lines.append(f"Training sessions: {report['session_count']}")
    lines.append("")
    lines.append("Mean conformity (match_score):")
    for sess in report["sessions"]:
        label = sess.get("label") or sess.get("source") or "?"
        score = sess.get("mean_match_score", 0.0)
        model = sess.get("model") or "?"
        lines.append(f"  {label:30s}  {score:.4f}  ({model})")
    lines.append("")
    lines.append(
        f"Overall {trend['direction']}: {trend['first_to_last_delta']:+.4f} "
        f"({trend['mean_match_scores'][0]:.4f} → {trend['mean_match_scores'][-1]:.4f})"
    )
    if trend["steps"]:
        lines.append("")
        lines.append("Step deltas:")
        for step in trend["steps"]:
            lines.append(
                f"  {step['from_label']} → {step['to_label']}: "
                f"{step['mean_score_delta']:+.4f} "
                f"({step['improved_cases']}↑ {step['regressed_cases']}↓ cases)"
            )
    if report.get("fields_improved") or report.get("fields_regressed"):
        lines.append("")
        if report["fields_improved"]:
            lines.append(f"Fields improved vs baseline: {', '.join(report['fields_improved'])}")
        if report["fields_regressed"]:
            lines.append(f"Fields regressed vs baseline: {', '.join(report['fields_regressed'])}")
    if report.get("naming_trend"):
        nt = report["naming_trend"]
        lines.append("")
        lines.append(
            f"Naming mean_attempts: {nt['mean_attempts'][0]:.2f} → "
            f"{nt['mean_attempts'][-1]:.2f} ({nt['first_to_last_delta']:+.2f})"
        )
    return "\n".join(lines)


def load_results_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
