"""
LLM-based semantic style metrics via Ollama.
Handles dimensions that require interpretation, not just counting.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"

# Single-pass prompt — all semantic metrics in one call to minimise overhead.
_PROMPT = """Analyze this prose passage for stylistic characteristics.
Return ONLY a valid JSON object with exactly these keys and allowed values:

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

Passage:
{text}

JSON:"""


def assess(
    text: str,
    model: str = DEFAULT_MODEL,
    rubric: dict | None = None,
) -> dict[str, Any]:
    """
    Run LLM analysis on a passage and return semantic style metrics.
    Falls back to defaults on any error so the pipeline never stalls.
    """
    # Truncate to ~1200 words — sufficient for style cues, avoids slow inference
    words = text.split()
    if len(words) > 1200:
        text = " ".join(words[:1200]) + "…"

    prompt = _PROMPT.format(text=text)
    raw = _call_ollama(prompt, model)
    result = _parse_json(raw)
    if result:
        return result
    return _defaults()


def _call_ollama(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "num_predict": 256,
            "stop": ["\n\n", "Passage:"],
        },
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read()).get("response", "")
    except Exception:
        return ""


def _parse_json(raw: str) -> dict | None:
    # Try direct parse first
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Extract first JSON object
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
