#!/usr/bin/env python3
"""
Phase 1: Extract style taxonomy from Style in Fiction (Leech & Short) PDF.
Produces: source/style_rubric.json

Usage:
    python tools/style_extraction/extract_rubric.py
    python tools/style_extraction/extract_rubric.py --skip-pdf   # reuse existing markdown
    python tools/style_extraction/extract_rubric.py --model llama3.2:latest
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT / "source" / "Style-in-Fiction.pdf"
EXTRACTED_DIR = ROOT / "source" / "extracted"
RUBRIC_PATH = ROOT / "source" / "style_rubric.json"

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "local-model")

sys.path.insert(0, str(ROOT / "tools"))
from llm_client import LLMError, check_connection, complete as llm_complete  # noqa: E402

STYLE_KEYWORDS = [
    "style", "lexic", "gramm", "syntax", "rhetoric", "figur",
    "narrat", "viewpoint", "voice", "register", "cohes", "diction",
    "sentence", "prose", "metaphor", "irony", "point of view",
]

EXTRACTION_PROMPT = """You are analyzing "Style in Fiction" by Geoffrey Leech and Mick Short.
Extract ALL style dimensions, metrics, and criteria defined or discussed in this section.

For each style dimension found, produce a JSON object with:
- "id": snake_case identifier (e.g. "lexical_density", "narrative_distance")
- "category": one of [lexical, grammatical, figurative, cohesion, context, viewpoint]
- "name": human-readable name
- "definition": concise definition from the text (1-2 sentences)
- "computation": "computable" if measurable from text statistics, "llm" if requires semantic judgment
- "metric_type": "continuous" (numeric 0-1 scale), "categorical" (named classes), or "ordinal" (low/mid/high)
- "values": array of class names if categorical, or scale description string if continuous/ordinal
- "scoring": object with "low", "mid", "high" keys describing each level

Return ONLY a valid JSON array. If no style dimensions are found, return [].

SECTION TITLE: {title}

CONTENT:
{content}

JSON array of style dimensions:"""


def convert_pdf(pdf_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {pdf_path.name} to markdown via marker...")

    result = subprocess.run(
        ["marker_single", str(pdf_path), "--output_dir", str(output_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"marker_single stderr:\n{result.stderr[:2000]}", file=sys.stderr)
        raise RuntimeError(f"marker_single failed (exit {result.returncode})")

    # Locate generated markdown
    for candidate in [
        output_dir / pdf_path.stem / f"{pdf_path.stem}.md",
        output_dir / f"{pdf_path.stem}.md",
    ]:
        if candidate.exists():
            print(f"Markdown at: {candidate}")
            return candidate

    md_files = list(output_dir.rglob("*.md"))
    if md_files:
        print(f"Markdown at: {md_files[0]}")
        return md_files[0]

    raise FileNotFoundError(f"No markdown output found under {output_dir}")


def split_sections(text: str) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    current_title = "Preface"
    current_lines: list[str] = []

    for line in text.splitlines():
        m = re.match(r"^(#{1,3})\s+(.+)", line)
        if m:
            if current_lines:
                sections.append({"title": current_title, "content": "\n".join(current_lines)})
            current_title = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({"title": current_title, "content": "\n".join(current_lines)})

    return sections


def is_style_relevant(section: dict[str, str]) -> bool:
    combined = (section["title"] + " " + section["content"][:300]).lower()
    return any(kw in combined for kw in STYLE_KEYWORDS)


def call_llm(prompt: str, model: str) -> str:
    try:
        return llm_complete(
            prompt,
            system="You are an expert in linguistics and literary stylistics. Return only valid JSON.",
            model=model,
            max_tokens=4096,
            temperature=0.05,
        )
    except LLMError as exc:
        print(f"  LLM error: {exc}", file=sys.stderr)
        return ""


def extract_dims_from_section(section: dict[str, str], model: str) -> list[dict]:
    content = section["content"]
    if len(content) < 400:
        return []

    # Trim to ~8000 chars so we don't exceed context
    if len(content) > 8000:
        content = content[:8000] + "\n[...truncated]"

    prompt = EXTRACTION_PROMPT.format(title=section["title"], content=content)
    response = call_llm(prompt, model)
    if not response:
        return []

    # Extract JSON array from response
    for pattern in [r"\[[\s\S]*?\]", r"\[[\s\S]*"]:
        m = re.search(pattern, response)
        if m:
            try:
                dims = json.loads(m.group())
                return [d for d in dims if isinstance(d, dict) and d.get("id")]
            except json.JSONDecodeError:
                pass

    return []


def merge_dims(extracted: list[list[dict]]) -> list[dict]:
    seen: dict[str, dict] = {}
    for batch in extracted:
        for dim in batch:
            dim_id = dim.get("id", "")
            if not dim_id:
                continue
            if dim_id not in seen:
                seen[dim_id] = dim
            else:
                # Keep whichever has the longer definition
                if len(dim.get("definition", "")) > len(seen[dim_id].get("definition", "")):
                    seen[dim_id] = dim
    return list(seen.values())


# Seed dimensions from Leech & Short's core checklist — used to fill gaps when
# the LLM extraction misses well-known metrics.
SEED_DIMS: list[dict[str, Any]] = [
    {
        "id": "lexical_density",
        "category": "lexical",
        "name": "Lexical Density",
        "definition": "Proportion of content words (nouns, verbs, adjectives, adverbs) to total words.",
        "computation": "computable",
        "metric_type": "continuous",
        "values": "0.0 – 1.0",
        "scoring": {
            "low": "< 0.40: Conversational, function-word-heavy prose",
            "mid": "0.40–0.60: Balanced narrative register",
            "high": "> 0.60: Dense, information-rich or formal text",
        },
    },
    {
        "id": "type_token_ratio",
        "category": "lexical",
        "name": "Vocabulary Richness (TTR)",
        "definition": "Ratio of unique word forms (types) to total words (tokens).",
        "computation": "computable",
        "metric_type": "continuous",
        "values": "0.0 – 1.0",
        "scoring": {
            "low": "< 0.30: Highly repetitive vocabulary",
            "mid": "0.30–0.55: Moderate variety",
            "high": "> 0.55: Rich, varied lexicon",
        },
    },
    {
        "id": "register",
        "category": "lexical",
        "name": "Register",
        "definition": "The variety of language appropriate to a particular situation — from formal/literary to colloquial.",
        "computation": "llm",
        "metric_type": "categorical",
        "values": ["formal_literary", "formal_technical", "neutral_narrative", "colloquial", "dialect", "archaic"],
        "scoring": {
            "low": "colloquial or dialect",
            "mid": "neutral_narrative",
            "high": "formal_literary or archaic",
        },
    },
    {
        "id": "sentence_length_mean",
        "category": "grammatical",
        "name": "Mean Sentence Length",
        "definition": "Average number of words per sentence.",
        "computation": "computable",
        "metric_type": "continuous",
        "values": "word count (typically 5–50)",
        "scoring": {
            "low": "< 12: Short, punchy sentences",
            "mid": "12–25: Standard narrative length",
            "high": "> 25: Long, complex sentences",
        },
    },
    {
        "id": "subordination_ratio",
        "category": "grammatical",
        "name": "Subordination Ratio",
        "definition": "Proportion of subordinating conjunctions to total tokens — a proxy for syntactic complexity.",
        "computation": "computable",
        "metric_type": "continuous",
        "values": "0.0 – 0.1 (typical range)",
        "scoring": {
            "low": "< 0.01: Simple, paratactic style",
            "mid": "0.01–0.03: Moderate subordination",
            "high": "> 0.03: Heavily subordinated, hypotactic prose",
        },
    },
    {
        "id": "passive_rate",
        "category": "grammatical",
        "name": "Passive Voice Rate",
        "definition": "Proportion of passive constructions among verb phrases.",
        "computation": "computable",
        "metric_type": "continuous",
        "values": "0.0 – 1.0",
        "scoring": {
            "low": "< 0.05: Active, direct narration",
            "mid": "0.05–0.15: Moderate passive use",
            "high": "> 0.15: Passive-heavy, impersonal or evasive tone",
        },
    },
    {
        "id": "figurative_density",
        "category": "figurative",
        "name": "Figurative Language Density",
        "definition": "Frequency of tropes (metaphor, simile, metonymy, irony) and rhetorical schemes.",
        "computation": "llm",
        "metric_type": "ordinal",
        "values": ["low", "moderate", "high"],
        "scoring": {
            "low": "Plain, unadorned prose",
            "mid": "Occasional figurative language",
            "high": "Dense figurative texture; foregrounded imagery",
        },
    },
    {
        "id": "pov",
        "category": "viewpoint",
        "name": "Narrative Point of View",
        "definition": "The grammatical person and access to interiority of the narrator.",
        "computation": "llm",
        "metric_type": "categorical",
        "values": ["first_person", "second_person", "third_limited", "third_omniscient", "mixed"],
        "scoring": {
            "low": "second_person (rare, experimental)",
            "mid": "third_limited",
            "high": "first_person or third_omniscient",
        },
    },
    {
        "id": "narrative_distance",
        "category": "viewpoint",
        "name": "Narrative Distance",
        "definition": "Psychological and emotional proximity of the narrator to the story world.",
        "computation": "llm",
        "metric_type": "ordinal",
        "values": ["intimate", "moderate", "distant"],
        "scoring": {
            "low": "distant: cold, observational narration",
            "mid": "moderate",
            "high": "intimate: deep interior access",
        },
    },
    {
        "id": "free_indirect_discourse",
        "category": "viewpoint",
        "name": "Free Indirect Discourse",
        "definition": "Blending of narrator voice with character thought/speech without explicit attribution.",
        "computation": "llm",
        "metric_type": "ordinal",
        "values": ["none", "sparse", "moderate", "heavy"],
        "scoring": {
            "low": "none: clear narrator/character separation",
            "mid": "sparse or moderate",
            "high": "heavy: voices deeply fused",
        },
    },
    {
        "id": "tone",
        "category": "context",
        "name": "Tone",
        "definition": "The affective and attitudinal register of the prose.",
        "computation": "llm",
        "metric_type": "categorical",
        "values": ["neutral", "lyrical", "sardonic", "melancholic", "comedic", "tense", "contemplative"],
        "scoring": {
            "low": "neutral or comedic",
            "mid": "contemplative or tense",
            "high": "lyrical or sardonic",
        },
    },
    {
        "id": "temporal_structure",
        "category": "context",
        "name": "Temporal Structure",
        "definition": "The relationship between story time and discourse time — whether narration is linear, retrospective, or fragmented.",
        "computation": "llm",
        "metric_type": "categorical",
        "values": ["linear", "retrospective", "prospective", "fragmented"],
        "scoring": {
            "low": "linear",
            "mid": "retrospective",
            "high": "fragmented",
        },
    },
    {
        "id": "dialogue_ratio",
        "category": "grammatical",
        "name": "Dialogue Ratio",
        "definition": "Proportion of the passage occupied by direct speech.",
        "computation": "computable",
        "metric_type": "continuous",
        "values": "0.0 – 1.0",
        "scoring": {
            "low": "< 0.05: Predominantly narration",
            "mid": "0.05–0.30: Mixed",
            "high": "> 0.30: Dialogue-dominant",
        },
    },
    {
        "id": "nominalization_ratio",
        "category": "grammatical",
        "name": "Nominalization Ratio",
        "definition": "Proportion of abstract nouns derived from verbs/adjectives (-tion, -ness, -ment, -ity), indicating formal or bureaucratic register.",
        "computation": "computable",
        "metric_type": "continuous",
        "values": "0.0 – 0.2 (typical range)",
        "scoring": {
            "low": "< 0.02: Verbal, dynamic prose",
            "mid": "0.02–0.06: Moderate abstraction",
            "high": "> 0.06: Heavily nominalized, formal register",
        },
    },
]


def build_rubric(pdf_path: Path, model: str, skip_pdf: bool = False) -> dict[str, Any]:
    if skip_pdf:
        md_files = list(EXTRACTED_DIR.rglob("*.md"))
        if not md_files:
            raise FileNotFoundError("No extracted markdown found. Run without --skip-pdf first.")
        md_path = md_files[0]
        print(f"Using existing markdown: {md_path}")
    else:
        md_path = convert_pdf(pdf_path, EXTRACTED_DIR)

    text = md_path.read_text(encoding="utf-8", errors="replace")
    sections = split_sections(text)
    relevant = [s for s in sections if is_style_relevant(s)]
    print(f"Sections: {len(sections)} total, {len(relevant)} style-relevant")

    all_extracted: list[list[dict]] = []
    for i, section in enumerate(relevant):
        print(f"  [{i+1:3d}/{len(relevant)}] {section['title'][:70]}")
        dims = extract_dims_from_section(section, model)
        if dims:
            print(f"         → {len(dims)} dimensions")
            all_extracted.append(dims)

    llm_dims = merge_dims(all_extracted)
    print(f"\nLLM-extracted dimensions: {len(llm_dims)}")

    # Merge seed dims for any ids the LLM missed
    seed_ids = {d["id"] for d in llm_dims}
    supplemented = llm_dims + [d for d in SEED_DIMS if d["id"] not in seed_ids]
    print(f"After seeding gaps: {len(supplemented)} dimensions total")

    counts = Counter(d.get("category", "unknown") for d in supplemented)

    return {
        "version": "1.0",
        "source": "Style in Fiction — Leech & Short (2nd ed.)",
        "pdf": pdf_path.name,
        "categories": {
            "lexical": "Vocabulary, register, and word-choice patterns",
            "grammatical": "Sentence structure, length, and syntactic complexity",
            "figurative": "Tropes, schemes, and rhetorical devices",
            "cohesion": "Textual coherence, reference, and connectivity",
            "context": "Setting, time, place, and social world",
            "viewpoint": "Narrative voice, perspective, and psychological distance",
        },
        "category_counts": dict(counts),
        "dimensions": supplemented,
    }


def main() -> None:
    from llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL as _DEFAULT_MODEL, check_connection

    parser = argparse.ArgumentParser(
        description="Extract style rubric from Style in Fiction PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "LM Studio (default):  --base-url http://localhost:1234/v1\n"
            "Ollama:               --base-url http://localhost:11434/v1\n"
        ),
    )
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--output", type=Path, default=RUBRIC_PATH)
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Model name as shown in LM Studio / Ollama")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LLM API base URL")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF conversion, use existing markdown")
    parser.add_argument("--force", action="store_true", help="Overwrite existing rubric")
    args = parser.parse_args()

    # Set base URL for the shared client
    import llm_client
    llm_client.DEFAULT_BASE_URL = args.base_url

    # Verify LLM is reachable before doing expensive PDF conversion
    print(f"Checking LLM at {args.base_url} …")
    try:
        models = check_connection(args.base_url)
        print(f"  Available models: {models or ['(none listed — model name required)']}")
    except Exception as exc:
        print(f"\n  LLM not reachable: {exc}", file=sys.stderr)
        print("  Start LM Studio and enable the local server, or start Ollama.", file=sys.stderr)
        sys.exit(1)

    if args.output.exists() and not args.force:
        print(f"\nRubric already exists: {args.output}")
        print("Use --force to regenerate.")
        sys.exit(0)

    if not args.pdf.exists() and not args.skip_pdf:
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    rubric = build_rubric(args.pdf, model=args.model, skip_pdf=args.skip_pdf)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rubric, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nRubric saved → {args.output}")
    print(f"Dimensions: {len(rubric['dimensions'])}")
    print("Category breakdown:")
    for cat, n in sorted(rubric["category_counts"].items()):
        print(f"  {cat}: {n}")
    print("\nReview source/style_rubric.json, then run:")
    print("  python tools/style_classification/run_pipeline.py --help")


if __name__ == "__main__":
    main()
