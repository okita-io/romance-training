"""
LLM-based semantic style metrics via OpenAI-compatible endpoint.
Works with LM Studio (default: localhost:1234) and Ollama (localhost:11434/v1).

Configure via env vars:
  LLM_BASE_URL   http://localhost:1234/v1  (LM Studio default)
  LLM_MODEL      model name as loaded in LM Studio
"""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from llm_client import DEFAULT_MODEL, LLMError, complete as llm_complete  # noqa: E402

ANALYSIS_SYSTEM_PATH = ROOT / "source" / "extracted" / "style_analysis_system.json"

_LEGACY_DEFAULTS: dict[str, Any] = {
    "register": "neutral_narrative",
    "pov": "third_limited",
    "narrative_distance": "moderate",
    "free_indirect_discourse": "none",
    "figurative_density": "low",
    "tone": "neutral",
    "temporal_structure": "linear",
    "sentence_variety": "moderate_variety",
    "dialogue_style": "none",
    "imagery_type": "none",
}


@lru_cache(maxsize=1)
def load_analysis_system(path: Path = ANALYSIS_SYSTEM_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _llm_dimensions(rubric: dict | None) -> list[dict[str, Any]]:
    if rubric:
        dims = [d for d in rubric.get("dimensions", []) if d.get("computation") == "llm"]
        if dims:
            return dims
    system = load_analysis_system()
    if system:
        return system.get("llm_dimensions", [])
    return []


def _default_for_dimension(dim: dict[str, Any]) -> Any:
    values = dim.get("values")
    if isinstance(values, list) and values:
        mid = len(values) // 2
        return values[mid]
    metric_type = dim.get("metric_type")
    if metric_type == "ordinal":
        return "moderate"
    if metric_type == "categorical":
        return "neutral"
    return None


def _defaults(rubric: dict | None = None) -> dict[str, Any]:
    out = dict(_LEGACY_DEFAULTS)
    for dim in _llm_dimensions(rubric):
        dim_id = dim.get("id")
        if dim_id and dim_id not in out:
            default = _default_for_dimension(dim)
            if default is not None:
                out[dim_id] = default

    principles: list[dict[str, Any]] = []
    if rubric:
        principles = rubric.get("textual_principles") or []
    else:
        system = load_analysis_system()
        if system:
            principles = system.get("textual_principles") or []
    for principle in principles:
        pid = principle.get("id")
        values = principle.get("values")
        if pid and pid not in out and isinstance(values, list) and values:
            out[pid] = values[len(values) // 2]
    return out


def _schema_lines(dims: list[dict[str, Any]], principles: list[dict[str, Any]]) -> str:
    lines: list[str] = ["Return a JSON object with exactly these keys and allowed values:", "{"]
    for dim in dims:
        dim_id = dim.get("id")
        values = dim.get("values")
        if not dim_id:
            continue
        if isinstance(values, list):
            allowed = ", ".join(json.dumps(v) for v in values)
            lines.append(f'  "{dim_id}": one of [{allowed}],')
        else:
            lines.append(f'  "{dim_id}": string,')
    for principle in principles:
        pid = principle.get("id")
        values = principle.get("values")
        if pid and isinstance(values, list):
            allowed = ", ".join(json.dumps(v) for v in values)
            lines.append(f'  "{pid}": one of [{allowed}],')
    lines.append('  "evidence": optional brief object mapping dimension ids to quoted phrases from the passage')
    lines.append("}")
    return "\n".join(lines)


def _build_system_prompt(rubric: dict | None) -> str:
    system = load_analysis_system()
    if system and system.get("system_prompt"):
        return system["system_prompt"] + " Return ONLY valid JSON — no explanation, no markdown."
    return (
        "You are a literary stylistician trained in Leech & Short's Style in Fiction. "
        "Analyse prose and return structured JSON. Return ONLY valid JSON — no explanation, no markdown."
    )


def _build_user_prompt(text: str, rubric: dict | None, rubric_context: str) -> str:
    dims = _llm_dimensions(rubric)
    principles = []
    if rubric:
        principles = rubric.get("textual_principles") or []
    elif load_analysis_system():
        principles = (load_analysis_system() or {}).get("textual_principles") or []

    schema = _schema_lines(dims, principles)
    return f"""Analyse this prose passage using the Leech & Short framework.

{rubric_context}

{schema}

Base judgments on the passage text and rubric definitions above. Return ONLY valid JSON.

Passage:
{text}"""


def assess(
    text: str,
    model: str = DEFAULT_MODEL,
    rubric: dict | None = None,
) -> dict[str, Any]:
    """
    Run LLM analysis on a passage and return semantic style metrics.
    Falls back to defaults on any connection error so the pipeline never stalls.
    """
    words = text.split()
    if len(words) > 1200:
        text = " ".join(words[:1200]) + "…"

    rubric_context = ""
    if rubric is not None:
        from tools.style_classification.style_knowledge import build_classification_context
        rubric_context = build_classification_context(text, rubric=rubric, knowledge_k=2)
    if not rubric_context.strip():
        rubric_context = "(No rubric reference loaded — apply general literary stylistics.)"

    defaults = _defaults(rubric)
    max_tokens = min(512, 64 + 24 * len(defaults))

    try:
        raw = llm_complete(
            _build_user_prompt(text, rubric, rubric_context),
            system=_build_system_prompt(rubric),
            model=model,
            max_tokens=max_tokens,
            temperature=0.05,
            stop=["\n\n"],
        )
    except LLMError as exc:
        sys.stderr.write(f"  LLM metrics skipped: {exc}\n")
        return defaults

    result = _parse_json(raw)
    if not result:
        return defaults

    merged = dict(defaults)
    merged.update({k: v for k, v in result.items() if k in defaults or k == "evidence"})
    return merged


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
