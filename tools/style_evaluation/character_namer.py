"""
RF-style lead naming for the style benchmark — LLM + overused-name gate with retry telemetry.

Mirrors romance-factory phase-4 lead naming (character_namer system prompt, JSON name/title,
overused first/surname lists, within-run collision checks). Tracks attempt counts so training
runs can compare how quickly the model escapes Elara / Isolde / etc.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OVERUSED_PATH = ROOT / "eval" / "style_benchmark" / "overused_llm_names.json"
DEFAULT_MAX_NAME_RETRIES = 100

# romance-factory prompt_engineering/system_prompts.json → character_namer
_CHARACTER_NAMER_SYSTEM = (
    "You name romance characters so names fit the story's genre, world, subgenre, and setting "
    "— not generic defaults from the wrong era or subculture. You ALWAYS respond with valid JSON "
    "only — no prose, no markdown fences, no explanation before or after the JSON object."
)

CHARACTER_NAME_AND_TITLE_JSON_RULES = (
    "- Respond with JSON containing exactly these top-level **string** fields: "
    "`name` and `character_title`.\n"
    "- `name` must be **only** the personal name the story uses in narration and dialogue "
    "attribution: given name plus family name (or one culturally appropriate personal name if "
    "mononyms fit). Typically **two to four words**. Do **not** put honorifics, ranks, job "
    "titles, commas, appositives, relationship phrases, or plot summary in `name`.\n"
    "- `character_title` holds optional in-world honorifics or formal styling, or an **empty "
    "string** when none apply.\n"
)

_TITLE_PREFIXES: frozenset[str] = frozenset({
    "admiral", "brother", "captain", "chief", "col", "colonel", "commander",
    "consul", "councillor", "count", "countess", "doctor", "dr", "duchess",
    "duke", "elder", "emperor", "empress", "ensign", "father", "gen",
    "general", "governor", "judge", "king", "lady", "lieutenant", "lord",
    "lt", "madam", "madame", "major", "marshal", "master", "miss", "mr",
    "mrs", "ms", "mother", "officer", "president", "prince", "princess",
    "prof", "professor", "queen", "regent", "reverend", "rev", "senator",
    "sergeant", "sgt", "sheriff", "sir", "sister", "specialist", "sultan",
    "vizier", "warden",
})

_DEFAULT_FIRST_LETTER_WEIGHTS: dict[str, float] = {
    "A": 9.0, "B": 8.0, "C": 8.0, "D": 7.0, "E": 1.0, "F": 7.0, "G": 7.0,
    "H": 7.0, "I": 6.0, "J": 8.0, "K": 7.0, "L": 7.0, "M": 8.0, "N": 7.0,
    "O": 6.0, "P": 7.0, "Q": 2.0, "R": 8.0, "S": 9.0, "T": 8.0, "U": 5.0,
    "V": 5.0, "W": 7.0, "X": 2.0, "Y": 5.0, "Z": 2.0,
}


@dataclass
class NameRejection:
    attempt: int
    name: str
    reason: str
    code: str = ""


@dataclass
class NamedCharacter:
    plot_id: str
    role: str
    name: str
    character_title: str
    attempts: int
    rejections: list[NameRejection] = field(default_factory=list)


@dataclass
class NamingRunReport:
    mode: str
    model: str | None
    max_retries: int
    name_seed: str
    characters: list[NamedCharacter] = field(default_factory=list)
    exhausted: bool = False

    @property
    def total_attempts(self) -> int:
        return sum(c.attempts for c in self.characters)

    @property
    def mean_attempts(self) -> float:
        if not self.characters:
            return 0.0
        return round(self.total_attempts / len(self.characters), 4)

    @property
    def max_attempts_used(self) -> int:
        return max((c.attempts for c in self.characters), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "model": self.model,
            "max_retries": self.max_retries,
            "name_seed": self.name_seed,
            "total_characters": len(self.characters),
            "total_attempts": self.total_attempts,
            "mean_attempts": self.mean_attempts,
            "max_attempts_used": self.max_attempts_used,
            "exhausted": self.exhausted,
            "characters": [
                {
                    "plot_id": c.plot_id,
                    "role": c.role,
                    "name": c.name,
                    "character_title": c.character_title,
                    "attempts": c.attempts,
                    "rejections": [
                        {
                            "attempt": r.attempt,
                            "name": r.name,
                            "reason": r.reason,
                            "code": r.code,
                        }
                        for r in c.rejections
                    ],
                }
                for c in self.characters
            ],
        }


def overused_names_path() -> Path:
    override = os.environ.get("STYLE_BENCHMARK_OVERUSED_NAMES_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    rf = ROOT.parent / "prompt_engineering" / "overused_llm_names.json"
    if rf.is_file():
        return rf
    return DEFAULT_OVERUSED_PATH


def load_overused_names() -> tuple[set[str], set[str]]:
    path = overused_names_path()
    doc: dict[str, Any] = {}
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            doc = {}
    first = {str(x).strip().lower() for x in (doc.get("names") or []) if str(x).strip()}
    last = {str(x).strip().lower() for x in (doc.get("surnames") or []) if str(x).strip()}
    # Always merge benchmark extras when using RF parent file.
    if path != DEFAULT_OVERUSED_PATH and DEFAULT_OVERUSED_PATH.is_file():
        try:
            extra = json.loads(DEFAULT_OVERUSED_PATH.read_text(encoding="utf-8"))
            first |= {str(x).strip().lower() for x in (extra.get("names") or []) if str(x).strip()}
            last |= {str(x).strip().lower() for x in (extra.get("surnames") or []) if str(x).strip()}
        except (OSError, json.JSONDecodeError):
            pass
    if not first:
        first = {"elara", "cassian", "kael", "lyra", "isolde", "isolide"}
    if not last:
        last = {"vance", "vale", "blackwood", "storm"}
    return first, last


def normalize_name(raw: str) -> str:
    s = str(raw or "").strip()
    return re.sub(r"\s+", " ", s)


def first_name_key(raw: str) -> str:
    n = normalize_name(raw)
    if not n:
        return ""
    parts = n.split()
    for p in parts:
        stripped = re.sub(r"[^\w]", "", p).lower()
        if stripped not in _TITLE_PREFIXES:
            return p.lower()
    return parts[-1].lower() if parts else ""


def last_name_key(raw: str) -> str:
    n = normalize_name(raw)
    if not n:
        return ""
    parts = n.split()
    if len(parts) >= 3 and parts[-2].lower() == "of":
        parts = parts[:-2]
    return parts[-1].lower() if parts else ""


def name_key(raw: str) -> str:
    return normalize_name(raw).lower()


@dataclass
class NameValidation:
    ok: bool
    reason: str
    code: str


def validate_personal_name(
    raw_name: str,
    *,
    reserved: set[str],
    overused_first: set[str],
    overused_last: set[str],
) -> NameValidation:
    nm = normalize_name(raw_name)
    if not nm:
        return NameValidation(False, "empty name", "empty")
    fn = first_name_key(nm)
    ln = last_name_key(nm)
    if fn and fn in overused_first:
        return NameValidation(False, f"overused first name: {fn}", "overused_first")
    if ln and ln in overused_last:
        return NameValidation(False, f"overused surname: {ln}", "overused_last")
    nk = name_key(nm)
    if nk in {name_key(x) for x in reserved}:
        return NameValidation(False, f"duplicate name in run: {nm}", "duplicate")
    if fn and fn in {first_name_key(x) for x in reserved if first_name_key(x)}:
        return NameValidation(False, f"duplicate first name in run: {fn}", "duplicate_first")
    return NameValidation(True, "", "ok")


def format_avoid_instruction() -> str:
    first, last = load_overused_names()
    sample_first = sorted(first)[:18]
    sample_last = sorted(last)[:12]
    return (
        "IMPORTANT: Do NOT use common AI-generated fiction first names "
        f"(including: {', '.join(sample_first)}, …). "
        f"Also avoid overused AI-fiction surnames (including: {', '.join(sample_last)}, …). "
        "Use names that belong in **Ashenmere** — folkloric, imperial, or salt-archipelago "
        "weight — not generic small-model romance defaults."
    )


def _parse_name_json(raw: str) -> tuple[str, str] | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\"name\"[^{}]*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    name = normalize_name(str(data.get("name") or ""))
    title = str(data.get("character_title") or "").strip()
    if not name:
        return None
    return name, title


def _pick_first_letter(rng: random.Random, exclude: set[str]) -> str:
    letters: list[str] = []
    weights: list[float] = []
    for ch, w in _DEFAULT_FIRST_LETTER_WEIGHTS.items():
        if ch in exclude or w <= 0:
            continue
        letters.append(ch)
        weights.append(w)
    if not letters:
        return "B"
    return rng.choices(letters, weights=weights, k=1)[0]


def build_naming_context_block(fixture: dict[str, Any]) -> str:
    world = fixture["world"]
    return (
        "=== GENRE, CLI, & WORLD (use with the profile to choose the name) ===\n\n"
        f"**Pipeline genre flag:** fantasy romance\n\n"
        f"**World:** {world['name']} — {world['setting']}\n"
        f"**Era:** {world['era']}\n\n"
        "**Naming style — weigh together with the character profile, station, and world:**\n"
        "- **Fantasy / mythic:** lean **legendary, folkloric, or invented with mythic weight** — "
        "names that echo regions, old tongues, or storybook grandeur; avoid bland contemporary "
        "Western defaults unless the character is clearly from a mundane pocket of the world.\n"
        "- **Setting alignment:** the full name should sound as if it belongs in the same **culture, "
        "caste, era, and naming tradition** implied by Ashenmere — fractured archipelago, salt-magic, "
        "storm citadels, Ash Court politics.\n"
    )


def build_lead_naming_user_prompt(
    *,
    context_block: str,
    character_brief: dict[str, Any],
    avoid_instruction: str,
    reserved_list: str,
    bad_tries: list[str],
    first_name_letter: str,
    other_lead_name: str | None,
) -> str:
    bad_note = ""
    if bad_tries:
        bad_note = (
            "\n\nThese `name` values were rejected (empty, collision, or overused list). "
            f"Pick a different valid `name`: {', '.join(bad_tries)}."
        )
    letter_line = ""
    if first_name_letter:
        letter_line = (
            f"- Provide a **given name** that starts with the letter **{first_name_letter}**. "
            "The family name may begin with any letter.\n"
        )
    other_line = ""
    if other_lead_name:
        other_line = (
            f"The other romance lead in this plot is already named: **{other_lead_name}**. "
            "Choose a name that fits the same setting and reads well with that name.\n\n"
        )
    return (
        "Choose **personal names** and optional **character_title** for this romance lead. "
        "Weigh **(1)** the character brief, **(2)** the world context, and **(3)** the avoidance rules.\n\n"
        f"{other_line}"
        f"{context_block}\n"
        "Rules:\n"
        f"{CHARACTER_NAME_AND_TITLE_JSON_RULES}"
        f"{letter_line}"
        f"- {avoid_instruction}\n"
        f"- Do not reuse any personal `name` already used in this benchmark run: {reserved_list}\n"
        f"- The new `name` must stay distinct from the other lead in this plot.{bad_note}\n\n"
        "=== CHARACTER BRIEF ===\n"
        f"{json.dumps(character_brief, indent=2, ensure_ascii=False)}"
    )


def _character_brief(plot: dict[str, Any], role: str) -> dict[str, Any]:
    story_role = "main_character" if role == "female_lead" else "love_interest"
    return {
        "story_role": story_role,
        "plot_id": plot["id"],
        "plot_title": plot["title"],
        "style_label": plot.get("style_label", ""),
        "plot_premise": plot.get("summary_template") or plot.get("summary", ""),
        "world": "Ashenmere — fantasy romance / salt-magic archipelago",
    }


def name_one_lead(
    *,
    plot: dict[str, Any],
    role: str,
    fixture: dict[str, Any],
    reserved: set[str],
    complete_fn: Callable[..., str],
    model: str,
    max_retries: int,
    rng: random.Random,
    other_lead_name: str | None = None,
    dry_run: bool = False,
) -> NamedCharacter:
    overused_first, overused_last = load_overused_names()
    avoid = format_avoid_instruction()
    context = build_naming_context_block(fixture)
    brief = _character_brief(plot, role)
    rejections: list[NameRejection] = []
    bad_tries: list[str] = []
    used_letters: set[str] = set()

    for attempt in range(1, max_retries + 1):
        letter = _pick_first_letter(rng, used_letters)
        reserved_list = ", ".join(sorted(reserved)) if reserved else "(none yet)"
        user = build_lead_naming_user_prompt(
            context_block=context,
            character_brief=brief,
            avoid_instruction=avoid,
            reserved_list=reserved_list,
            bad_tries=bad_tries[-8:],
            first_name_letter=letter,
            other_lead_name=other_lead_name,
        )
        if dry_run:
            placeholder = f"DryRun {plot['id']} {role}"
            return NamedCharacter(
                plot_id=plot["id"],
                role=role,
                name=placeholder,
                character_title="",
                attempts=0,
                rejections=[],
            )
        raw = complete_fn(user, system=_CHARACTER_NAMER_SYSTEM, model=model)
        parsed = _parse_name_json(raw)
        if not parsed:
            bad_tries.append("(unparseable JSON)")
            rejections.append(
                NameRejection(attempt, "(unparseable)", "model did not return valid JSON", "parse")
            )
            continue
        cand, title = parsed
        check = validate_personal_name(
            cand,
            reserved=reserved,
            overused_first=overused_first,
            overused_last=overused_last,
        )
        if not check.ok:
            bad_tries.append(f"{cand} ({check.reason})")
            rejections.append(NameRejection(attempt, cand, check.reason, check.code))
            continue
        return NamedCharacter(
            plot_id=plot["id"],
            role=role,
            name=cand,
            character_title=title,
            attempts=attempt,
            rejections=rejections,
        )

    return NamedCharacter(
        plot_id=plot["id"],
        role=role,
        name="",
        character_title="",
        attempts=max_retries,
        rejections=rejections,
    )


def generate_llm_roster(
    fixture: dict[str, Any],
    *,
    name_seed: str,
    model: str,
    max_retries: int = DEFAULT_MAX_NAME_RETRIES,
    complete_fn: Callable[..., str] | None = None,
    dry_run: bool = False,
) -> tuple[list[str], list[str], NamingRunReport]:
    """Name all 7 plot pairs via LLM; return (females, males, telemetry report)."""
    from tools.llm_client import complete as default_complete

    llm = complete_fn or default_complete
    rng = random.Random(name_seed)
    reserved: set[str] = set()
    females: list[str] = []
    males: list[str] = []
    report = NamingRunReport(
        mode="llm",
        model=model if not dry_run else None,
        max_retries=max_retries,
        name_seed=name_seed,
    )

    for plot in fixture["plots"]:
        female_char = name_one_lead(
            plot=plot,
            role="female_lead",
            fixture=fixture,
            reserved=reserved,
            complete_fn=llm,
            model=model,
            max_retries=max_retries,
            rng=rng,
            dry_run=dry_run,
        )
        report.characters.append(female_char)
        if not female_char.name:
            report.exhausted = True
            break
        reserved.add(female_char.name)

        male_char = name_one_lead(
            plot=plot,
            role="male_lead",
            fixture=fixture,
            reserved=reserved,
            complete_fn=llm,
            model=model,
            max_retries=max_retries,
            rng=rng,
            other_lead_name=female_char.name,
            dry_run=dry_run,
        )
        report.characters.append(male_char)
        if not male_char.name:
            report.exhausted = True
            break
        reserved.add(male_char.name)
        females.append(female_char.name)
        males.append(male_char.name)

    if report.exhausted and not dry_run:
        raise RuntimeError(
            f"Could not name all benchmark leads within {max_retries} retries per character. "
            f"See naming report: total_attempts={report.total_attempts}, "
            f"last rejections={report.characters[-1].rejections[-3:] if report.characters else []}"
        )
    return females, males, report


def aggregate_naming_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize naming telemetry across benchmark result records or run headers."""
    attempts = [int(r.get("total_attempts", 0)) for r in reports if r]
    means = [float(r.get("mean_attempts", 0)) for r in reports if r]
    if not attempts:
        return {"run_count": 0}
    return {
        "run_count": len(reports),
        "mean_total_attempts": round(sum(attempts) / len(attempts), 4),
        "mean_mean_attempts_per_character": round(sum(means) / len(means), 4) if means else 0.0,
        "max_total_attempts": max(attempts),
    }
