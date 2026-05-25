"""Opt-in live tests against a **local** LLM (Ollama or OpenAI-compatible / LM Studio).

These calls hit the configured endpoint in ``romance_factory.core.config`` (same as the
pipeline). They are skipped unless you opt in:

    export ROMANCE_FACTORY_LOCAL_LLM_LIVE=1

Requirements:
  - ``LLM_BACKEND`` is ``ollama`` or ``openai_chat`` (not ``openrouter``).
  - A model is running and reachable (``ollama_url`` / ``MODEL_NAME`` in ``settings.yaml``
    or env).

Each **prompt variant** runs **N** full trials (default **10**; ``ROMANCE_FACTORY_LOCAL_LLM_TRIES_PER_PROMPT``).
The best variant maximizes **(suffix pass streak, total passes, longest pass run)** so stable late passes
(e.g. ``F,F,P,P,P``) beat flaky early passes (e.g. ``F,P,F,F,F``). Thresholds:
``ROMANCE_FACTORY_LOCAL_LLM_MIN_SUFFIX_STREAK`` (default 3), ``ROMANCE_FACTORY_LOCAL_LLM_MIN_TOTAL_PASSES``
(default ~60% of N). Use ``pytest -s`` for full traces.

Run::

    ROMANCE_FACTORY_LOCAL_LLM_LIVE=1 pytest train/tests/test_local_llm_live.py -v -m local_llm_live
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Sequence
from unittest.mock import MagicMock, patch

import pytest

import romance_factory.core.config as cfg
import romance_factory.core.ollama_client as ollama_client
from romance_factory.core.config import HEAT_LEVELS
from romance_factory.generate.agents.editorial import EditorialAgent
from romance_factory.generate.config_v2 import V2Config
from romance_factory.generate.diegetic_consistency import (
    build_diegetic_anchor_prompt,
    build_outline_diegetic_review_prompt,
    parse_llm_json_object,
)
from romance_factory.generate.models import (
    DocumentMetadata,
    EditorialResult,
    RetrievalResult,
)
from romance_factory.generate.outline_editorial import _heat_escalation_line
from romance_factory.generate.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Retry + reporting helpers
# ---------------------------------------------------------------------------

_PROMPT_SNIP_LEN = 420


def tries_per_prompt() -> int:
    """Trials per prompt variant (default 10)."""
    raw = (os.environ.get("ROMANCE_FACTORY_LOCAL_LLM_TRIES_PER_PROMPT") or "10").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 10
    return max(1, min(n, 50))


def min_suffix_streak() -> int:
    """Minimum consecutive passes at the **end** of the trial block for the winning variant."""
    raw = (os.environ.get("ROMANCE_FACTORY_LOCAL_LLM_MIN_SUFFIX_STREAK") or "3").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 3


def min_total_passes(n: int) -> int:
    """Minimum total passes (out of *n*) for the winning variant."""
    raw = (os.environ.get("ROMANCE_FACTORY_LOCAL_LLM_MIN_TOTAL_PASSES") or "").strip()
    if raw:
        try:
            return max(1, min(int(raw), n))
        except ValueError:
            pass
    return max(1, (n * 6 + 9) // 10)


def _suffix_pass_streak(passes: list[bool]) -> int:
    k = 0
    for b in reversed(passes):
        if b:
            k += 1
        else:
            break
    return k


def _longest_pass_run(passes: list[bool]) -> int:
    best = cur = 0
    for b in passes:
        if b:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _variant_score_tuple(passes: list[bool]) -> tuple[int, int, int]:
    """Higher is better: trailing consistency, then volume, then any strong run."""
    return (
        _suffix_pass_streak(passes),
        sum(passes),
        _longest_pass_run(passes),
    )


def _snip(text: str, max_len: int = _PROMPT_SNIP_LEN) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


@dataclass
class LiveLLMRunReport:
    """Accumulated trace for one live-LLM scenario with optional retries."""

    scenario: str
    failure_count: int = 0
    attempt_lines: list[str] = field(default_factory=list)
    winning_attempt: str | None = None
    winning_system_prompt: str | None = None
    winning_user_prompt: str | None = None
    last_raw_response: str = ""

    def record_failure(
        self,
        attempt_name: str,
        *,
        system_prompt: str,
        user_prompt: str,
        reason: str,
        raw_preview: str,
    ) -> None:
        self.failure_count += 1
        self.attempt_lines.append(
            f"  [{attempt_name}] FAILED ({reason})\n"
            f"    system_prompt: {_snip(system_prompt)}\n"
            f"    user_prompt:   {_snip(user_prompt)}\n"
            f"    raw_preview:   {_snip(raw_preview, 280)!r}"
        )

    def record_success(
        self,
        attempt_name: str,
        *,
        system_prompt: str,
        user_prompt: str,
        raw_preview: str,
    ) -> None:
        self.winning_attempt = attempt_name
        self.winning_system_prompt = system_prompt
        self.winning_user_prompt = user_prompt
        self.attempt_lines.append(
            f"  [{attempt_name}] OK\n"
            f"    system_prompt: {_snip(system_prompt)}\n"
            f"    user_prompt:   {_snip(user_prompt)}\n"
            f"    raw_preview:   {_snip(raw_preview, 280)!r}"
        )

    def record_trial(
        self,
        attempt_name: str,
        passed: bool,
        *,
        system_prompt: str,
        user_prompt: str,
        reason: str,
        raw_preview: str,
    ) -> None:
        status = "PASS" if passed else "FAIL"
        if not passed:
            self.failure_count += 1
        self.attempt_lines.append(
            f"  [{attempt_name}] {status} ({reason})\n"
            f"    system_prompt: {_snip(system_prompt)}\n"
            f"    user_prompt:   {_snip(user_prompt)}\n"
            f"    raw_preview:   {_snip(raw_preview, 280)!r}"
        )

    def summary_header(self) -> str:
        return (
            f"\n=== {self.scenario} ===\n"
            f"total_fail_trials: {self.failure_count}\n"
            f"winning_variant: {self.winning_attempt!r}\n"
        )

    def full_trace(self) -> str:
        return self.summary_header() + "\n".join(self.attempt_lines)


def _generate_patched(
    user_prompt: str,
    *,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    with (
        patch.object(ollama_client._cfg, "LLM_STREAM", False),
        patch.object(ollama_client._cfg, "LLM_PROGRESS", False),
    ):
        return ollama_client.generate(
            user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def _variant_from_generate(
    name: str,
    sys_p: str,
    usr: str,
    mtok: int,
    temp: float,
    evaluate_raw: Callable[[str], tuple[bool, str]],
) -> tuple[str, str, str, Callable[[], tuple[bool, str, str]]]:
    """Build ``(name, sys, user, trial_fn)`` where *trial_fn* runs one generate + eval."""

    def trial_fn() -> tuple[bool, str, str]:
        try:
            raw = _generate_patched(
                usr, system_prompt=sys_p, max_tokens=mtok, temperature=temp,
            )
        except Exception as exc:
            return False, f"generate() raised: {exc}", ""
        ok, reason = evaluate_raw(raw or "")
        return ok, reason, _snip(raw, 500)

    return (name, sys_p, usr, trial_fn)


def run_live_llm_scored_variants(
    scenario: str,
    variants: Sequence[
        tuple[str, str, str, Callable[[], tuple[bool, str, str]]]
    ],
    *,
    report: LiveLLMRunReport | None = None,
    n_trials: int | None = None,
    threshold_min_suffix: int | None = None,
    threshold_min_total: int | None = None,
) -> LiveLLMRunReport:
    """Run *n_trials* calls per variant; pick best by (suffix streak, total, longest run).

    Each variant is ``(name, display_system_prompt, display_user_prompt, trial_fn)``.
    ``trial_fn`` returns ``(passed, reason, raw_preview)`` for one independent attempt.

    The highest-scoring variant must meet suffix/total thresholds (defaults from env +
    :func:`min_total_passes`; override per call for special cases like fixed-seed).
    """
    n = n_trials if n_trials is not None else tries_per_prompt()
    need_suf = (
        min_suffix_streak() if threshold_min_suffix is None else max(0, threshold_min_suffix)
    )
    need_tot = (
        min_total_passes(n) if threshold_min_total is None else max(1, min(threshold_min_total, n))
    )

    rpt = report or LiveLLMRunReport(scenario=scenario)
    if report is None:
        rpt.scenario = scenario
    else:
        rpt.attempt_lines.append(
            f"\n--- scored variant block: {scenario} ({n} trials each) ---\n"
        )

    best: tuple[tuple[int, int, int], str, list[bool], str, str, str] | None = None
    # score, name, passes, sys_p, usr_p, last_preview

    for name, sys_p, usr_p, trial_fn in variants:
        passes: list[bool] = []
        last_preview = ""
        for trial in range(1, n + 1):
            tag = f"{name} trial {trial}/{n}"
            try:
                ok, reason, preview = trial_fn()
            except Exception as exc:
                ok, reason, preview = False, f"trial_fn raised: {exc}", ""
            last_preview = preview or last_preview
            passes.append(ok)
            rpt.record_trial(
                tag,
                ok,
                system_prompt=sys_p,
                user_prompt=usr_p,
                reason=reason,
                raw_preview=preview,
            )
        sc = _variant_score_tuple(passes)
        seq = "".join("P" if p else "F" for p in passes)
        rpt.attempt_lines.append(
            f"  >> VARIANT_SCORE [{name}] suffix={sc[0]} total={sc[1]} "
            f"longest_run={sc[2]} sequence={seq}"
        )
        cand = (sc, name, passes, sys_p, usr_p, last_preview)
        if best is None or sc > best[0]:
            best = cand

    assert best is not None
    sc_b, name_b, passes_b, sys_b, usr_b, prev_b = best
    suf_b, tot_b, long_b = sc_b

    rpt.winning_attempt = (
        f"{name_b} score=(suffix={suf_b},total={tot_b},longest={long_b})"
    )
    rpt.winning_system_prompt = sys_b
    rpt.winning_user_prompt = usr_b
    rpt.last_raw_response = prev_b

    rpt.attempt_lines.append(
        f"\n  >> WINNER {rpt.winning_attempt}\n"
        f"  >> thresholds: min_suffix_streak={need_suf} min_total_passes={need_tot} (of {n})\n"
    )

    suffix_ok = suf_b >= need_suf if need_suf > 0 else True
    total_ok = tot_b >= need_tot
    if suffix_ok and total_ok:
        print(rpt.full_trace())
        return rpt

    print(rpt.full_trace())
    pytest.fail(
        f"{scenario}: best variant {name_b!r} has suffix={suf_b}, total={tot_b} "
        f"(need suffix>={need_suf} and total>={need_tot}).\n"
        + "\n".join(rpt.attempt_lines)
    )


# ---------------------------------------------------------------------------
# Skip / fixture
# ---------------------------------------------------------------------------


def _local_llm_live_enabled() -> bool:
    v = (os.environ.get("ROMANCE_FACTORY_LOCAL_LLM_LIVE") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _require_local_llm_live() -> None:
    if not _local_llm_live_enabled():
        pytest.skip(
            "Set ROMANCE_FACTORY_LOCAL_LLM_LIVE=1 to run local LLM live tests "
            "(Ollama or openai_chat / LM Studio)."
        )


def _require_non_openrouter_backend() -> None:
    b = (cfg.LLM_BACKEND or "").strip().lower()
    if b == "openrouter":
        pytest.skip(
            "Local LLM live tests expect LLM_BACKEND ollama or openai_chat; "
            "openrouter is excluded (use train/tests/test_openrouter_live.py)."
        )


def _smoke_generate() -> None:
    """One tiny call; skip if the stack is down."""
    try:
        with (
            patch.object(ollama_client._cfg, "LLM_STREAM", False),
            patch.object(ollama_client._cfg, "LLM_PROGRESS", False),
        ):
            text = ollama_client.generate(
                'Reply with exactly: OK',
                system_prompt="You follow instructions literally.",
                max_tokens=16,
                temperature=0.0,
            )
    except Exception as exc:
        pytest.skip(f"Local LLM not reachable: {exc}")
    if not (text or "").strip():
        pytest.skip("Local LLM returned empty text")


@pytest.fixture(scope="module")
def local_llm_live_ready():
    _require_local_llm_live()
    _require_non_openrouter_backend()
    _smoke_generate()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.local_llm_live
def test_local_llm_fixed_seed_reproducibility(local_llm_live_ready):
    """Same fixed seed + temperature 0 → identical paired outputs; scored like other live tests."""
    n = tries_per_prompt()
    prompts = [
        (
            "primary",
            "Reply with one word only, no punctuation.",
            "Pick a single made-up color name, one word only.",
        ),
        (
            "fallback_strict",
            "Output rules: respond with exactly one lowercase word, letters only, no punctuation.",
            "Invent a fictional color name. Respond with that single word only.",
        ),
        (
            "fallback_minimal",
            "You may only output one token.",
            "Color:",
        ),
    ]
    variants: list[tuple[str, str, str, Callable[[], tuple[bool, str, str]]]] = []
    for attempt_name, sys_p, usr in prompts:

        def _make_seed_trial(sn: str, su: str) -> Callable[[], tuple[bool, str, str]]:
            def trial() -> tuple[bool, str, str]:
                with (
                    patch.object(ollama_client._cfg, "LLM_STREAM", False),
                    patch.object(ollama_client._cfg, "LLM_PROGRESS", False),
                    patch.object(ollama_client._cfg, "LLM_SEED_MODE", "fixed"),
                    patch.object(ollama_client._cfg, "LLM_SEED_FIXED", 20260101),
                ):
                    try:
                        a = ollama_client.generate(
                            su, system_prompt=sn, max_tokens=12, temperature=0.0,
                        )
                        b = ollama_client.generate(
                            su, system_prompt=sn, max_tokens=12, temperature=0.0,
                        )
                    except Exception as exc:
                        return False, f"generate() raised: {exc}", ""
                if not (a.strip() and b.strip()):
                    return False, "empty response", f"a={a!r} b={b!r}"
                if a.strip() == b.strip():
                    return True, "paired_outputs_match", f"a=b={a.strip()!r}"
                return (
                    False,
                    f"mismatch {a.strip()!r} vs {b.strip()!r}",
                    f"a={a!r} b={b!r}",
                )

            return trial

        variants.append((attempt_name, sys_p, usr, _make_seed_trial(sys_p, usr)))

    # Reproducibility is often weaker than classification JSON; relax totals slightly.
    run_live_llm_scored_variants(
        "fixed_seed_reproducibility",
        variants,
        n_trials=n,
        threshold_min_suffix=2,
        threshold_min_total=max(3, (n * 4 + 9) // 10),
    )


@pytest.mark.local_llm_live
def test_local_llm_outline_diegetic_review_flags_anachronism(local_llm_live_ready):
    world = {
        "world_type": "fantasy",
        "setting_design": "River kingdoms before gunpowder; no modern technology.",
        "culture": "Heralds and wax seals; no printing presses or electronics.",
    }
    anchor = build_diegetic_anchor_prompt(world, catalog=None)
    beat = json.dumps(
        {
            "act_number": 1,
            "summary": (
                "Sir Mord uses his smartphone to GPS-navigate the sacred aqueduct "
                "in the year 412 of the river-kings."
            ),
            "characters_involved": ["Mord"],
            "emotional_tone": "urgent",
            "plot_function": "setup",
        },
        ensure_ascii=False,
    )
    sys_primary, usr_primary = build_outline_diegetic_review_prompt(
        1, 1, beat, anchor=anchor, chapter_outline_excerpt="",
    )

    def _eval_outline(raw: str) -> tuple[bool, str]:
        data = parse_llm_json_object(raw)
        issues = data.get("issues") if isinstance(data, dict) else None
        needs = data.get("needs_rewrite") if isinstance(data, dict) else None
        blob = json.dumps(data, ensure_ascii=False).lower()
        flagged = False
        if isinstance(issues, list) and issues:
            for it in issues:
                if not isinstance(it, dict):
                    continue
                t = str(it.get("type", "")).lower()
                expl = str(it.get("explanation", "")).lower()
                if "diegetic" in t or "phone" in expl or "smartphone" in expl or "gps" in expl:
                    flagged = True
                    break
        if needs is True:
            flagged = True
        if any(k in blob for k in ("smartphone", "phone", "gps", "anachron", "modern")):
            flagged = True
        if flagged:
            return True, "flagged anachronism"
        return False, f"no flag in parsed={data!r}"

    usr_fallback_1 = (
        f"{anchor}\n\n"
        "OUTLINE BEAT (JSON):\n"
        f"{beat}\n\n"
        'Return ONLY valid JSON: {{"issues":[...],"needs_rewrite":true/false,"rewrite_plan":""}}.\n'
        "The beat places a smartphone and GPS in a pre-gunpowder fantasy. "
        "You MUST list at least one diegetic_violation issue or set needs_rewrite true."
    )
    sys_fallback_1 = (
        "You are a setting continuity editor. Output one JSON object only, no markdown fences."
    )
    usr_fallback_2 = (
        "Fantasy world: no modern electronics. Beat summary: 'smartphone GPS in year 412 river-kings.'\n"
        "Reply JSON only: {\"anachronism_detected\": true/false, \"reason\": \"...\"}. "
        "anachronism_detected must be true."
    )
    sys_fallback_2 = "Return a single JSON object. No prose."

    def _eval_with_binary(raw: str) -> tuple[bool, str]:
        ok, reason = _eval_outline(raw)
        if ok:
            return True, reason
        data = parse_llm_json_object(raw)
        if isinstance(data, dict) and data.get("anachronism_detected") is True:
            return True, "binary fallback JSON"
        return False, reason

    variants = [
        _variant_from_generate(
            "outline_diegetic_primary",
            sys_primary,
            usr_primary,
            400,
            0.2,
            _eval_with_binary,
        ),
        _variant_from_generate(
            "outline_diegetic_fallback_explicit_json",
            sys_fallback_1,
            usr_fallback_1,
            450,
            0.15,
            _eval_with_binary,
        ),
        _variant_from_generate(
            "outline_diegetic_fallback_binary_json",
            sys_fallback_2,
            usr_fallback_2,
            200,
            0.0,
            _eval_with_binary,
        ),
    ]
    run_live_llm_scored_variants("outline_diegetic_anachronism", variants)


@pytest.mark.local_llm_live
def test_local_llm_editorial_act_validation_diegetic(local_llm_live_ready, tmp_path):
    """Editorial path with diegetic anchor; retries with a minimal JSON fallback prompt."""
    world_path = tmp_path / "world.json"
    world_path.write_text(
        json.dumps(
            {
                "world_type": "fantasy",
                "setting_design": "Pre-industrial river kingdoms; no gunpowder or electricity.",
                "culture": "Oral messages and river boats; no screens.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    bad_act = (
        "Elara adjusted the thermostat on the castle wall, then tweeted the "
        "herald about the invasion. The year was 300 by the old river reckoning."
    )

    def _query(collection: str, *_a, **_kw):
        meta = DocumentMetadata(type="act", chapter=1, act=1)
        if collection == "acts":
            return [
                RetrievalResult(text=bad_act, metadata=meta, similarity_score=1.0),
            ]
        return []

    engine = MagicMock()
    engine.query.side_effect = _query
    engine.store_document.return_value = None

    config = V2Config(
        story_path=str(tmp_path),
        enable_rubric_scoring=False,
        editorial_max_retries=1,
        enable_diegetic_act_validation=True,
        max_context_chars=4000,
        max_story_arc_chars=4000,
        max_previous_acts_chars=2000,
    )
    builder = PromptBuilder(
        max_context_chars=config.max_context_chars,
        max_story_arc_chars=config.max_story_arc_chars,
        max_previous_acts_chars=config.max_previous_acts_chars,
        num_chapters=config.num_chapters,
    )
    agent = EditorialAgent(
        engine, builder, config=config, story_path=str(tmp_path),
    )

    report = LiveLLMRunReport(scenario="editorial_diegetic")

    def editorial_ok(r: EditorialResult) -> tuple[bool, str]:
        joined = " ".join(
            f"{i.type} {i.explanation} {i.suggested_fix}".lower()
            for i in r.issues
        )
        if any(i.type == "diegetic_violation" for i in r.issues):
            return True, "diegetic_violation"
        if any(
            k in joined
            for k in (
                "thermostat",
                "tweet",
                "twitter",
                "anachron",
                "modern",
                "technology",
                "phone",
                "screen",
            )
        ):
            return True, "keyword in issues"
        if r.score < 5.0 and r.issues:
            return True, "low score with issues"
        return False, f"score={r.score} issues={[i.type for i in r.issues]}"

    anchor = build_diegetic_anchor_prompt(
        json.loads(world_path.read_text(encoding="utf-8")),
        catalog=None,
    )

    sys_fb1 = (
        "You return ONLY a JSON object with keys score (0-10), issues (array), "
        'rewrite_plan (string). Each issue: type, severity, location, explanation, suggested_fix.'
    )
    usr_fb1 = (
        f"{anchor}\n\n## Act to review\n{bad_act}\n\n"
        "The world is pre-industrial fantasy. Flag diegetic_violation for thermostat, tweet, "
        "or other modern tech. JSON only."
    )

    sys_fb2 = "JSON only. No markdown."
    usr_fb2 = (
        "Pre-industrial fantasy act mentions thermostat and tweeting. "
        'Return {"modern_tech_violation": true/false, "notes": "..."}. Must use true.'
    )

    def _eval_fallback_raw(raw: str) -> tuple[bool, str]:
        data = parse_llm_json_object(raw)
        if not isinstance(data, dict):
            return False, "not a dict"
        if data.get("modern_tech_violation") is True:
            return True, "fallback binary JSON"
        issues = data.get("issues")
        if isinstance(issues, list):
            blob = json.dumps(issues, ensure_ascii=False).lower()
            if "diegetic" in blob or "thermostat" in blob or "tweet" in blob:
                return True, "issues in JSON"
        return False, f"parsed keys={list(data.keys())}"

    act_excerpt = f"(act text excerpt) {_snip(bad_act, 200)}"
    sys_internal = (
        "(internal EditorialAgent.evaluate — PromptBuilder.build_act_validation_prompt)"
    )

    def editorial_trial() -> tuple[bool, str, str]:
        try:
            with (
                patch.object(ollama_client._cfg, "LLM_STREAM", False),
                patch.object(ollama_client._cfg, "LLM_PROGRESS", False),
            ):
                r = agent.evaluate(
                    chapter=1,
                    act=1,
                    is_last_act=False,
                    is_plot_twist=False,
                    story_state=None,
                    enable_progression_enforcement=False,
                )
        except Exception as exc:
            return False, f"evaluate() raised: {exc}", ""
        ok, reason = editorial_ok(r)
        preview = str([(x.type, _snip(x.explanation, 120)) for x in r.issues])
        return ok, reason, preview

    variants: list[tuple[str, str, str, Callable[[], tuple[bool, str, str]]]] = [
        ("editorial_agent_primary", sys_internal, act_excerpt, editorial_trial),
        _variant_from_generate(
            "editorial_json_fallback", sys_fb1, usr_fb1, 500, 0.15, _eval_fallback_raw,
        ),
        _variant_from_generate(
            "editorial_minimal_binary", sys_fb2, usr_fb2, 200, 0.0, _eval_fallback_raw,
        ),
    ]
    run_live_llm_scored_variants("editorial_diegetic", variants, report=report)


@pytest.mark.local_llm_live
def test_local_llm_heat_escalation_spectrum_ordering(local_llm_live_ready):
    lines = "\n\n".join(
        f"## {label}\n{_heat_escalation_line(label)}" for label in HEAT_LEVELS
    )
    keys_json = json.dumps(HEAT_LEVELS)
    usr_primary = (
        "You are scoring romance OUTLINE planning cues for implied on-page intimacy.\n"
        "Return ONLY one JSON object: {\"intensity\": {<heat_label>: <integer 1-5>}} "
        "where 1 = minimal / closed-door and 5 = explicit on-page intimacy clearly allowed.\n"
        f"Use exactly these keys (same spelling): {keys_json}.\n\n"
        f"{lines}"
    )
    sys_primary = "Respond with a single raw JSON object only. No markdown fences."

    sweet_line = _heat_escalation_line("sweet")
    explicit_line = _heat_escalation_line("explicit")
    usr_fb = (
        "Compare two romance outline cues.\n\n"
        f"SWEET:\n{sweet_line}\n\nEXPLICIT:\n{explicit_line}\n\n"
        "Return JSON {\"sweet\": <int 1-5>, \"explicit\": <int 1-5>} only. "
        "sweet must be less than explicit."
    )
    sys_fb = "Output one JSON object. Integers only for values."

    usr_fb2 = (
        "Rate implied on-page intimacy 1-5 for:\n"
        f"A (sweet): {sweet_line}\n"
        f"B (explicit): {explicit_line}\n"
        'JSON: {"A":n,"B":m} with A < B required.'
    )
    sys_fb2 = "JSON only."

    def _eval_heat(raw: str) -> tuple[bool, str]:
        data = parse_llm_json_object(raw)
        if not isinstance(data, dict):
            return False, "not a dict"
        intens = data.get("intensity")
        if isinstance(intens, dict):
            try:
                sweet = int(intens["sweet"])
                explicit = int(intens["explicit"])
            except (KeyError, TypeError, ValueError):
                return False, f"missing sweet/explicit in {intens!r}"
            if sweet < explicit:
                return True, "full spectrum map"
            return False, f"sweet={sweet} >= explicit={explicit}"
        a = data.get("A")
        b = data.get("B")
        if a is not None and b is not None:
            try:
                if int(a) < int(b):
                    return True, "A/B keys"
            except (TypeError, ValueError):
                pass
        s = data.get("sweet")
        e = data.get("explicit")
        if s is not None and e is not None:
            try:
                if int(s) < int(e):
                    return True, "sweet/explicit pair"
            except (TypeError, ValueError):
                pass
        return False, f"keys={list(data.keys())}"

    variants = [
        _variant_from_generate(
            "heat_batch_primary", sys_primary, usr_primary, 400, 0.1, _eval_heat,
        ),
        _variant_from_generate(
            "heat_pair_fallback", sys_fb, usr_fb, 300, 0.1, _eval_heat,
        ),
        _variant_from_generate(
            "heat_ab_fallback", sys_fb2, usr_fb2, 250, 0.0, _eval_heat,
        ),
    ]
    run_live_llm_scored_variants("heat_escalation_spectrum", variants)
