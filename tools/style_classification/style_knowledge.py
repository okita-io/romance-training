"""Retrieve Leech & Short knowledge chunks for rubric-aware classification."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_PATH = ROOT / "source" / "extracted" / "style_knowledge.jsonl"
RUBRIC_PATH = ROOT / "source" / "style_rubric.json"

# Map style_profile dimension ids to rubric categories for retrieval routing.
_DIMENSION_CATEGORIES: dict[str, str] = {
    "lexical_density": "lexical",
    "type_token_ratio": "lexical",
    "avg_word_length": "lexical",
    "register": "lexical",
    "sentence_length_mean": "grammatical",
    "sentence_length_std": "grammatical",
    "subordination_ratio": "grammatical",
    "coordination_ratio": "grammatical",
    "passive_rate": "grammatical",
    "nominalization_ratio": "grammatical",
    "dialogue_ratio": "grammatical",
    "punctuation_density": "grammatical",
    "avg_dependency_depth": "grammatical",
    "figurative_density": "figurative",
    "pov": "viewpoint",
    "narrative_distance": "viewpoint",
    "free_indirect_discourse": "viewpoint",
    "tone": "context",
    "temporal_structure": "context",
    "sentence_variety": "grammatical",
    "dialogue_style": "grammatical",
    "imagery_type": "figurative",
}

_TOKEN_RE = re.compile(r"[a-z]{3,}")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


@lru_cache(maxsize=1)
def load_knowledge(path: Path = KNOWLEDGE_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


@lru_cache(maxsize=1)
def load_rubric(path: Path = RUBRIC_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _score_chunk(
    chunk: dict[str, Any],
    *,
    query_tokens: set[str],
    categories: set[str],
    dimension_id: str | None,
) -> float:
    title = chunk.get("title", "")
    text = chunk.get("text", "")
    chunk_tokens = _tokenize(title + " " + text)
    overlap = len(query_tokens & chunk_tokens)

    score = float(overlap)
    chunk_cats = set(chunk.get("categories", []))
    if categories & chunk_cats:
        score += 5.0
    if dimension_id and dimension_id.replace("_", " ") in text.lower():
        score += 8.0
    if dimension_id and dimension_id.replace("_", " ") in title.lower():
        score += 4.0
    if chunk.get("has_mermaid"):
        score += 0.5
    return score


def retrieve(
    query: str,
    *,
    categories: list[str] | None = None,
    dimension_id: str | None = None,
    k: int = 3,
    knowledge: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return top-k knowledge chunks by keyword overlap and category match."""
    kb = knowledge if knowledge is not None else load_knowledge()
    if not kb:
        return []

    query_tokens = _tokenize(query)
    if dimension_id:
        query_tokens |= _tokenize(dimension_id.replace("_", " "))

    cat_set = set(categories or [])
    if dimension_id and not cat_set:
        cat = _DIMENSION_CATEGORIES.get(dimension_id)
        if cat:
            cat_set = {cat}

    scored = [
        (chunk, _score_chunk(chunk, query_tokens=query_tokens, categories=cat_set, dimension_id=dimension_id))
        for chunk in kb
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, s in scored[:k] if s > 0]


def format_context(chunks: list[dict[str, Any]], max_chars: int = 4000) -> str:
    """Format retrieved chunks as prompt context."""
    if not chunks:
        return ""

    parts: list[str] = []
    used = 0
    for chunk in chunks:
        title = chunk.get("title", "Section")
        page = chunk.get("page")
        header = f"### {title}"
        if page:
            header += f" (p. {page})"
        body = chunk.get("text", "")
        block = f"{header}\n\n{body}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                parts.append(block[:remaining] + "…")
            break
        parts.append(block)
        used += len(block)

    return "\n\n---\n\n".join(parts)


def rubric_dimension_summary(rubric: dict[str, Any] | None, dimension_ids: list[str]) -> str:
    """Build compact rubric definitions for the requested dimensions."""
    if not rubric:
        return ""

    by_id = {d["id"]: d for d in rubric.get("dimensions", []) if d.get("id")}
    lines: list[str] = []
    for dim_id in dimension_ids:
        dim = by_id.get(dim_id)
        if not dim:
            continue
        scoring = dim.get("scoring", {})
        score_bits = []
        for level in ("low", "mid", "high"):
            if level in scoring:
                score_bits.append(f"  {level}: {scoring[level]}")
        values = dim.get("values")
        values_str = ""
        if isinstance(values, list):
            values_str = f" Allowed: {', '.join(values)}."
        elif isinstance(values, str):
            values_str = f" Scale: {values}."
        lines.append(
            f"- **{dim_id}** ({dim.get('name', dim_id)}): {dim.get('definition', '')}{values_str}"
        )
        if score_bits:
            lines.append("\n".join(score_bits))

    return "\n".join(lines)


def build_classification_context(
    passage: str,
    *,
    rubric: dict[str, Any] | None = None,
    knowledge_k: int = 2,
) -> str:
    """Assemble Leech & Short reference context for LLM style classification."""
    rubric = rubric or load_rubric()
    llm_dims = [
        "register", "pov", "narrative_distance", "free_indirect_discourse",
        "figurative_density", "tone", "temporal_structure",
        "sentence_variety", "dialogue_style", "imagery_type",
    ]

    sections: list[str] = []

    rubric_block = rubric_dimension_summary(rubric, llm_dims)
    if rubric_block:
        sections.append("## Rubric dimensions\n\n" + rubric_block)

    chunks = retrieve(passage, k=knowledge_k)
    if chunks:
        kb_block = format_context(chunks, max_chars=3500)
        sections.append("## Reference excerpts (Leech & Short)\n\n" + kb_block)

    return "\n\n".join(sections)
