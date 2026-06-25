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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from llm_client import DEFAULT_MODEL, LLMError, complete as llm_complete  # noqa: E402

# Single-pass prompt — all semantic metrics in one call to minimise inference overhead.
_SYSTEM = (
    "You are a literary stylistician. Analyse prose and return structured JSON. "
    "Return ONLY valid JSON — no explanation, no markdown."
)

_PROMPT = """Analyse this prose passage for stylistic characteristics using the Leech & Short framework.

{rubric_context}

Return a JSON object with exactly these keys and allowed values:

{{
  "register": one of ["formal_literary", "formal_technical", "neutral_narrative", "colloquial", "dialect", "archaic"],
  "pov": one of ["first_person", "second_person", "third_limited", "third_omniscient", "mixed"],
  "narrative_distance": one of ["intimate", "moderate", "distant"],
  "free_indirect_discourse": one of ["none", "sparse", "moderate", "heavy"],
  "figurative_density": one of ["low", "moderate", "high"],
  "tone": one of ["neutral", "lyrical", "sardonic", "melancholic", "comedic", "tense", "contemplative"],
  "temporal_structure": one of ["linear", "retrospective", "prospective", "fragmented"],
  "sentence_variety": one of ["uniform", "moderate_variety", "high_variety"],
  "dialogue_style": one of ["none", "naturalistic", "stylized", "period"],
  "imagery_type": one of ["none", "visual", "sensory_mixed", "abstract", "nature", "urban"]
}}

Base judgments on the passage text and the rubric definitions above. Return ONLY valid JSON.

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

    try:
        raw = llm_complete(
            _PROMPT.format(text=text, rubric_context=rubric_context),
            system=_SYSTEM,
            model=model,
            max_tokens=256,
            temperature=0.05,
            stop=["\n\n"],
        )
    except LLMError as exc:
        sys.stderr.write(f"  LLM metrics skipped: {exc}\n")
        return _defaults()

    result = _parse_json(raw)
    return result if result else _defaults()


def _parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _defaults() -> dict[str, Any]:
    return {
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
