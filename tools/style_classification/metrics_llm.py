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

from .pass_config import PASS1_LLM_FIELDS, PASS2_LLM_FIELDS, PassMode, fields_for_pass  # noqa: E402

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


def _textual_principles(rubric: dict | None) -> list[dict[str, Any]]:
    if rubric:
        return rubric.get("textual_principles") or []
    system = load_analysis_system()
    if system:
        return system.get("textual_principles") or []
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


def _defaults(
    rubric: dict | None = None,
    *,
    fields: frozenset[str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if fields is None:
        out = dict(_LEGACY_DEFAULTS)

    for dim in _llm_dimensions(rubric):
        dim_id = dim.get("id")
        if not dim_id:
            continue
        if fields is not None and dim_id not in fields:
            continue
        if dim_id not in out:
            default = _default_for_dimension(dim)
            if default is not None:
                out[dim_id] = default

    for principle in _textual_principles(rubric):
        pid = principle.get("id")
        values = principle.get("values")
        if not pid or pid in out:
            continue
        if fields is not None and pid not in fields:
            continue
        if isinstance(values, list) and values:
            out[pid] = values[len(values) // 2]

    return out


def _schema_lines(
    dims: list[dict[str, Any]],
    principles: list[dict[str, Any]],
    *,
    fields: frozenset[str] | None = None,
) -> str:
    lines: list[str] = ["Return a JSON object with exactly these keys and allowed values:", "{"]
    for dim in dims:
        dim_id = dim.get("id")
        values = dim.get("values")
        if not dim_id:
            continue
        if fields is not None and dim_id not in fields:
            continue
        if isinstance(values, list):
            allowed = ", ".join(json.dumps(v) for v in values)
            lines.append(f'  "{dim_id}": one of [{allowed}],')
        else:
            lines.append(f'  "{dim_id}": string,')
    for principle in principles:
        pid = principle.get("id")
        values = principle.get("values")
        if not pid or not isinstance(values, list):
            continue
        if fields is not None and pid not in fields:
            continue
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


def _prior_block(prior: dict[str, Any], context_fields: frozenset[str] | None) -> str:
    if not prior:
        return ""
    skip = {"evidence"}
    if context_fields is not None:
        relevant = {
            k: v for k, v in prior.items() if k in context_fields and k not in skip and v is not None
        }
    else:
        relevant = {k: v for k, v in prior.items() if k not in skip and v is not None}
    if not relevant:
        return ""
    payload = json.dumps(relevant, ensure_ascii=False, indent=2)
    return (
        "Prior classification from an earlier pass (use as context; refine only if the passage contradicts):\n"
        f"{payload}\n"
    )


def _build_user_prompt(
    text: str,
    rubric: dict | None,
    rubric_context: str,
    *,
    fields: frozenset[str] | None = None,
    prior: dict[str, Any] | None = None,
) -> str:
    dims = _llm_dimensions(rubric)
    principles = _textual_principles(rubric)
    schema = _schema_lines(dims, principles, fields=fields)
    prior_context = PASS1_LLM_FIELDS if fields == PASS2_LLM_FIELDS else fields
    prior_text = _prior_block(prior or {}, prior_context)

    return f"""Analyse this prose passage using the Leech & Short framework.

{rubric_context}

{prior_text}{schema}

Base judgments on the passage text and rubric definitions above. Return ONLY valid JSON.

Passage:
{text}"""


def assess(
    text: str,
    model: str = DEFAULT_MODEL,
    rubric: dict | None = None,
    *,
    pass_mode: PassMode = "full",
    fields: frozenset[str] | None = None,
    prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run LLM analysis on a passage and return semantic style metrics.

    Args:
        pass_mode: "full" (all fields), "fast" (pass 1), "deep" (pass 2), or "both" (pass 1 then pass 2).
        fields:    Restrict to these keys (overrides pass_mode when set).
        prior:     Pass 1 labels injected into the prompt for pass 2.

    Falls back to defaults on any connection error so the pipeline never stalls.
    """
    active_fields = fields if fields is not None else fields_for_pass(pass_mode)

    words = text.split()
    if len(words) > 1200:
        text = " ".join(words[:1200]) + "…"

    rubric_context = ""
    if rubric is not None:
        from tools.style_classification.style_knowledge import build_classification_context
        rubric_context = build_classification_context(text, rubric=rubric, knowledge_k=2)
    if not rubric_context.strip():
        rubric_context = "(No rubric reference loaded — apply general literary stylistics.)"

    defaults = _defaults(rubric, fields=active_fields)
    max_tokens = 8192 if active_fields is None else 2048

    try:
        raw = llm_complete(
            _build_user_prompt(
                text,
                rubric,
                rubric_context,
                fields=active_fields,
                prior=prior,
            ),
            system=_build_system_prompt(rubric),
            model=model,
            max_tokens=max_tokens,
            temperature=0.05,
        )
    except LLMError as exc:
        sys.stderr.write(f"  LLM metrics skipped: {exc}\n")
        return defaults

    result = _parse_json(raw)
    if not result:
        return defaults

    merged = dict(defaults)
    allowed = set(defaults) | {"evidence"}
    merged.update({k: v for k, v in result.items() if k in allowed})
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
