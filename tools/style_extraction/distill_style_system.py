#!/usr/bin/env python3
"""
Distill Style in Fiction (parsed markdown) into an LLM-consumable analysis system.

Filters OCR/front-matter noise, builds a RAG knowledge base, extracts the Section 3.1
checklist, and writes a structured rubric anchored in Leech & Short's framework
(lexis, syntax, textual relations, segmentation, climax, mind style, etc.).

Usage:
    python tools/style_extraction/distill_style_system.py
    python tools/style_extraction/distill_style_system.py --input source/Style-in-Fiction.parsed.md
    python tools/style_extraction/distill_style_system.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.style_extraction.manuscript_parser import (
    extract_checklist_items,
    parse_manuscript,
    sections_to_knowledge_records,
)

DEFAULT_INPUT = ROOT / "source" / "Style-in-Fiction.parsed.md"
KNOWLEDGE_OUTPUT = ROOT / "source" / "extracted" / "style_knowledge.jsonl"
RUBRIC_OUTPUT = ROOT / "source" / "style_rubric.json"
ANALYSIS_SYSTEM_OUTPUT = ROOT / "source" / "extracted" / "style_analysis_system.json"

TEXTUAL_PRINCIPLES: list[dict[str, Any]] = [
    {
        "id": "end_focus",
        "name": "Principle of End-Focus",
        "source_section": "7.3",
        "definition": "Given information tends to precede new information; the end of a tone or graphic unit carries communicative salience.",
        "analysis_prompt": "Does the passage place new or emphatic information at clause/sentence ends?",
        "metric_type": "ordinal",
        "values": ["weak", "neutral", "strong"],
    },
    {
        "id": "segmentation",
        "name": "Segmentation and Graphic Units",
        "source_section": "7.4",
        "definition": "Linear text is chunked into tone/graphic units that parcel information for the reader.",
        "analysis_prompt": "How is the passage segmented (short vs long sentences, heavy vs light punctuation)?",
        "metric_type": "ordinal",
        "values": ["minimal", "balanced", "heavy"],
    },
    {
        "id": "prose_rhythm",
        "name": "Rhythm of Prose",
        "source_section": "7.4.1",
        "definition": "Patterns in graphic-unit length and stress create implicit tempo.",
        "analysis_prompt": "Is there a perceptible rhythmic pattern in sentence/graphic-unit length?",
        "metric_type": "ordinal",
        "values": ["flat", "moderate", "marked"],
    },
    {
        "id": "climax",
        "name": "Principle of Climax",
        "source_section": "7.5.2",
        "definition": "'Last is most important' — final position tends to be the major information focus.",
        "analysis_prompt": "Does the passage build toward a final focal element?",
        "metric_type": "ordinal",
        "values": ["anticlimactic", "neutral", "climactic"],
    },
    {
        "id": "subordination_salience",
        "name": "Subordination and Backgrounding",
        "source_section": "7.5.1",
        "definition": "Subordinate clauses background circumstantial information; coordination gives equal weight.",
        "analysis_prompt": "Are key events foregrounded or backgrounded via subordination/coordination?",
        "metric_type": "ordinal",
        "values": ["flat", "balanced", "strategic"],
    },
    {
        "id": "textual_relations",
        "name": "Textual Relations (Given/New)",
        "source_section": "6.4.4",
        "definition": "How clauses link via reference, ellipsis, coordination — distinguishing old vs new information.",
        "analysis_prompt": "How are pronouns, substitution, and conjunctions used?",
        "metric_type": "ordinal",
        "values": ["immature", "standard", "artful"],
    },
]

CORE_DIMENSIONS: list[dict[str, Any]] = [
    {
        "id": "lexical_complexity",
        "category": "lexical",
        "name": "Lexical Complexity",
        "source_section": "3.1.A",
        "definition": "Vocabulary simplicity/complexity, register, collocation, semantic fields.",
        "computation": "llm",
        "metric_type": "ordinal",
        "values": ["simple_colloquial", "neutral", "complex_literary"],
        "scoring": {"low": "Simple colloquial", "mid": "Standard narrative", "high": "Complex literary"},
    },
    {
        "id": "lexical_density",
        "category": "lexical",
        "name": "Lexical Density",
        "definition": "Proportion of content words to total words.",
        "computation": "computable",
        "metric_type": "continuous",
        "values": "0.0 – 1.0",
        "scoring": {"low": "< 0.40", "mid": "0.40–0.60", "high": "> 0.60"},
    },
    {
        "id": "register",
        "category": "lexical",
        "name": "Register",
        "definition": "Situation-appropriate language variety.",
        "computation": "llm",
        "metric_type": "categorical",
        "values": ["formal_literary", "formal_technical", "neutral_narrative", "colloquial", "dialect", "archaic"],
        "scoring": {"low": "colloquial", "mid": "neutral", "high": "formal_literary"},
    },
    {
        "id": "sentence_complexity",
        "category": "grammatical",
        "name": "Sentence Complexity",
        "source_section": "3.1.B.2",
        "definition": "Simple vs complex structure; coordination vs subordination vs parataxis.",
        "computation": "llm",
        "metric_type": "ordinal",
        "values": ["simple_paratactic", "moderate", "complex_hypotactic"],
        "scoring": {"low": "Simple paratactic", "mid": "Mixed", "high": "Complex hypotactic"},
    },
    {
        "id": "sentence_length_mean",
        "category": "grammatical",
        "name": "Mean Sentence Length",
        "definition": "Average words per sentence.",
        "computation": "computable",
        "metric_type": "continuous",
        "values": "word count",
        "scoring": {"low": "< 12", "mid": "12–25", "high": "> 25"},
    },
    {
        "id": "subordination_ratio",
        "category": "grammatical",
        "name": "Subordination Ratio",
        "definition": "Proxy for hypotactic complexity.",
        "computation": "computable",
        "metric_type": "continuous",
        "values": "0.0 – 0.1",
        "scoring": {"low": "< 0.01", "mid": "0.01–0.03", "high": "> 0.03"},
    },
    {
        "id": "figurative_density",
        "category": "figurative",
        "name": "Figurative Language",
        "source_section": "3.1.C",
        "definition": "Tropes and schemes: metaphor, simile, irony, parallelism.",
        "computation": "llm",
        "metric_type": "ordinal",
        "values": ["low", "moderate", "high"],
        "scoring": {"low": "Plain", "mid": "Occasional", "high": "Dense"},
    },
    {
        "id": "cohesion",
        "category": "cohesion",
        "name": "Cohesion",
        "source_section": "3.1.D.1",
        "definition": "Links between sentences: conjunctions, reference, substitution, ellipsis.",
        "computation": "llm",
        "metric_type": "ordinal",
        "values": ["loose", "standard", "tight"],
        "scoring": {"low": "Loose", "mid": "Standard", "high": "Tight or artful"},
    },
    {
        "id": "mind_style",
        "category": "viewpoint",
        "name": "Mind Style",
        "source_section": "6.1",
        "definition": "Worldview encoded in language — conceptualisation of experience.",
        "computation": "llm",
        "metric_type": "ordinal",
        "values": ["standard", "distinct", "deviant"],
        "scoring": {"low": "Standard", "mid": "Distinct", "high": "Deviant"},
    },
    {
        "id": "pov",
        "category": "viewpoint",
        "name": "Narrative Point of View",
        "source_section": "3.1.D.2",
        "definition": "Grammatical person and access to interiority.",
        "computation": "llm",
        "metric_type": "categorical",
        "values": ["first_person", "second_person", "third_limited", "third_omniscient", "mixed"],
        "scoring": {},
    },
    {
        "id": "narrative_distance",
        "category": "viewpoint",
        "name": "Narrative Distance",
        "definition": "Psychological proximity of narrator to characters and events.",
        "computation": "llm",
        "metric_type": "ordinal",
        "values": ["intimate", "moderate", "distant"],
        "scoring": {"low": "distant", "mid": "moderate", "high": "intimate"},
    },
    {
        "id": "free_indirect_discourse",
        "category": "viewpoint",
        "name": "Free Indirect Discourse",
        "source_section": "3.1.D.2",
        "definition": "Blending of narrator and character voice.",
        "computation": "llm",
        "metric_type": "ordinal",
        "values": ["none", "sparse", "moderate", "heavy"],
        "scoring": {},
    },
    {
        "id": "tone",
        "category": "context",
        "name": "Tone",
        "source_section": "3.1.D.2",
        "definition": "Affective and attitudinal register.",
        "computation": "llm",
        "metric_type": "categorical",
        "values": ["neutral", "lyrical", "sardonic", "melancholic", "comedic", "tense", "contemplative"],
        "scoring": {},
    },
    {
        "id": "dialogue_ratio",
        "category": "grammatical",
        "name": "Dialogue Ratio",
        "definition": "Proportion of direct speech.",
        "computation": "computable",
        "metric_type": "continuous",
        "values": "0.0 – 1.0",
        "scoring": {"low": "< 0.05", "mid": "0.05–0.30", "high": "> 0.30"},
    },
]

FRAMEWORK: dict[str, Any] = {
    "levels": [
        {"id": "semantic", "label": "Semantics"},
        {"id": "syntax", "label": "Syntax / lexigrammar"},
        {"id": "phonology", "label": "Phonology"},
        {"id": "graphology", "label": "Graphology"},
    ],
    "parts": [
        {"id": "part_one", "title": "Approaches and Methods", "chapters": "1–4"},
        {"id": "part_two", "title": "Aspects of Style", "chapters": "5–12"},
    ],
    "analysis_layers": [
        {"id": "lexical", "label": "Lexis", "source": "3.1.A"},
        {"id": "grammatical", "label": "Grammar and syntax", "source": "3.1.B"},
        {"id": "figurative", "label": "Figures of speech", "source": "3.1.C"},
        {"id": "cohesion", "label": "Cohesion and context", "source": "3.1.D"},
        {"id": "textual", "label": "Textual dynamics", "source": "7.1–7.5"},
        {"id": "viewpoint", "label": "Viewpoint and mind style", "source": "6.1–6.4"},
    ],
}


def _build_llm_system_prompt(checklist_count: int, principle_ids: list[str], dimension_ids: list[str]) -> str:
    return f"""You are a literary stylistician trained in Geoffrey Leech and Mick Short's *Style in Fiction* framework.

Analyse prose using these layers:
1. Lexis — vocabulary, register, semantic fields, collocation
2. Grammar/syntax — sentence types, complexity, clause structure
3. Figures — schemes and tropes, foregrounding
4. Cohesion & context — reference, conjunction, speech presentation, tone
5. Textual dynamics — segmentation, end-focus, climax, given/new information
6. Viewpoint — POV, narrative distance, mind style, free indirect discourse

Apply {checklist_count} Section 3.1 checklist prompts where relevant.
Score textual principles: {", ".join(principle_ids)}.
Return JSON for dimensions: {", ".join(dimension_ids)}.
Ground judgments in observable linguistic evidence from the passage."""


def distill(input_path: Path, *, min_section_chars: int = 200) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    text = input_path.read_text(encoding="utf-8", errors="replace")
    sections = parse_manuscript(text, min_section_chars=min_section_chars)
    if not sections:
        raise ValueError(f"No sections extracted from {input_path}")

    knowledge = sections_to_knowledge_records(sections, source_file=input_path.name)

    checklist: list[dict[str, Any]] = []
    for section in sections:
        if section.is_checklist or section.section_id == "3.1":
            checklist.extend(extract_checklist_items(section))

    llm_dims = [d for d in CORE_DIMENSIONS if d.get("computation") == "llm"]
    llm_dim_ids = [d["id"] for d in llm_dims]

    rubric: dict[str, Any] = {
        "version": "2.0",
        "source": "Style in Fiction — Leech & Short (2nd ed., parsed manuscript)",
        "source_file": input_path.name,
        "framework": FRAMEWORK,
        "textual_principles": TEXTUAL_PRINCIPLES,
        "checklist": checklist,
        "categories": {
            "lexical": "Vocabulary and register (Section 3.1.A)",
            "grammatical": "Syntax and sentence structure (Section 3.1.B)",
            "figurative": "Figures of speech (Section 3.1.C)",
            "cohesion": "Cohesion and textual relations (Section 3.1.D, Ch. 7)",
            "context": "Tone and speech presentation (Section 3.1.D)",
            "viewpoint": "POV and mind style (Ch. 6, 10)",
            "textual": "Segmentation, rhythm, climax (Ch. 7)",
        },
        "dimensions": CORE_DIMENSIONS,
        "stats": {
            "sections": len(sections),
            "knowledge_chunks": len(knowledge),
            "checklist_items": len(checklist),
            "dimensions": len(CORE_DIMENSIONS),
            "textual_principles": len(TEXTUAL_PRINCIPLES),
        },
    }

    analysis_system: dict[str, Any] = {
        "version": "1.0",
        "source": rubric["source"],
        "system_prompt": _build_llm_system_prompt(
            len(checklist),
            [p["id"] for p in TEXTUAL_PRINCIPLES],
            llm_dim_ids,
        ),
        "analysis_protocol": [
            "Read for overall literary effect.",
            "Survey lexis (checklist A).",
            "Survey grammar (checklist B).",
            "Note figures and foregrounding (checklist C).",
            "Assess cohesion and context (checklist D).",
            "Evaluate textual dynamics: segmentation, end-focus, climax.",
            "Assess viewpoint and mind style.",
            "Return JSON scores with brief evidence.",
        ],
        "textual_principles": TEXTUAL_PRINCIPLES,
        "llm_dimensions": llm_dims,
        "computable_dimensions": [d for d in CORE_DIMENSIONS if d.get("computation") == "computable"],
        "checklist_sample": checklist[:12],
    }

    return knowledge, rubric, analysis_system


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--knowledge-output", type=Path, default=KNOWLEDGE_OUTPUT)
    parser.add_argument("--rubric-output", type=Path, default=RUBRIC_OUTPUT)
    parser.add_argument("--analysis-output", type=Path, default=ANALYSIS_SYSTEM_OUTPUT)
    parser.add_argument("--min-section-chars", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    if args.rubric_output.exists() and not args.force:
        raise SystemExit(f"{args.rubric_output} exists — use --force to overwrite")

    print(f"Distilling {args.input.relative_to(ROOT)} …")
    knowledge, rubric, analysis_system = distill(args.input, min_section_chars=args.min_section_chars)

    args.knowledge_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.knowledge_output, "w", encoding="utf-8") as fh:
        for rec in knowledge:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    args.rubric_output.write_text(json.dumps(rubric, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.analysis_output.write_text(json.dumps(analysis_system, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stats = rubric["stats"]
    print(f"  Sections: {stats['sections']} | Chunks: {stats['knowledge_chunks']} | Checklist: {stats['checklist_items']}")
    print(f"  Rubric → {args.rubric_output.relative_to(ROOT)}")
    print(f"  Analysis system → {args.analysis_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
