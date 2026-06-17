#!/usr/bin/env python3
"""
Phase 3: Generate multi-task instruction pairs for style classifier training.

Reads enriched JSONL (with style_profile), writes two task types:
  1. classification  — "Classify the style of this passage" → JSON profile
  2. judgment        — "Analyze [dimension] of this passage" → natural language

Rewrite pairs (Phase 3B) require a separate generation step with a frontier LLM.

Usage:
    python tools/training_formats/generate_instruction_pairs.py
    python tools/training_formats/generate_instruction_pairs.py \\
        --input train/romance_corpus/gutenberg_styled.jsonl \\
        --output-dir train/style_training
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "train" / "romance_corpus" / "gutenberg_styled.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "train" / "style_training"

# --- Instruction templates (randomised to increase diversity) ---

_CLF_INSTRUCTIONS = [
    "Classify the stylistic dimensions of this prose passage.",
    "Analyze and classify the style of the following passage.",
    "Provide a complete style profile for this text.",
    "Using the Leech-Short stylistics framework, classify this passage.",
    "What are the key stylistic metrics of this prose?",
    "Identify the style characteristics of the following text.",
    "Produce a structured style analysis of this passage.",
]

_JUDGMENT_TEMPLATES: dict[str, list[str]] = {
    "lexical_density": [
        "How lexically dense is this passage?",
        "Analyze the lexical density of this prose.",
        "What proportion of this text consists of content words?",
    ],
    "register": [
        "What register is this passage written in?",
        "Describe the linguistic register of this prose.",
        "Is this text formal, colloquial, or somewhere between?",
    ],
    "pov": [
        "What narrative point of view is used in this passage?",
        "Identify the narrative voice and perspective here.",
        "From whose perspective is this passage narrated?",
    ],
    "tone": [
        "What is the overall tone of this passage?",
        "Describe the emotional and attitudinal tone of this prose.",
        "Analyze the tonal qualities of this text.",
    ],
    "sentence_length_mean": [
        "Comment on the sentence length patterns in this passage.",
        "How does sentence length shape the rhythm of this prose?",
        "Analyze how sentence length contributes to the style.",
    ],
    "figurative_density": [
        "How much figurative language is present in this passage?",
        "Rate the density of metaphor, simile, and other tropes.",
        "Analyze the use of figurative language in this text.",
    ],
    "free_indirect_discourse": [
        "Is free indirect discourse present in this passage?",
        "Analyze the blending of narrator and character voice.",
        "How does the narrative voice relate to character consciousness?",
    ],
    "passive_rate": [
        "Analyze the use of passive voice in this passage.",
        "How does the balance of active and passive constructions affect this prose?",
    ],
    "narrative_distance": [
        "How intimate or distant is the narrative perspective here?",
        "Analyze the narrative distance in this passage.",
        "How close does the narration bring us to the characters?",
    ],
    "type_token_ratio": [
        "Comment on the vocabulary richness of this passage.",
        "How varied is the word choice in this text?",
        "Analyze lexical variety and repetition patterns.",
    ],
    "subordination_ratio": [
        "How syntactically complex is this prose?",
        "Analyze the use of subordination and sentence complexity.",
        "Is this prose more paratactic (simple) or hypotactic (complex)?",
    ],
    "dialogue_ratio": [
        "How much of this passage consists of dialogue?",
        "Analyze the balance of narration and direct speech.",
    ],
    "tone": [
        "What tone does the author strike in this passage?",
        "Identify and analyze the dominant tone.",
    ],
    "temporal_structure": [
        "How is time handled in this passage — linear, retrospective, or fragmented?",
        "Analyze the temporal structure of this narrative.",
    ],
    "nominalization_ratio": [
        "Analyze the degree of nominalization in this prose.",
        "How much does this text rely on abstract nouns derived from verbs or adjectives?",
    ],
}

_NATURAL_LANGUAGE_EXPLANATIONS: dict[str, dict[str, str]] = {
    "register": {
        "formal_literary": (
            "The register is formal literary — elevated diction, complex syntax, "
            "and a vocabulary associated with serious fiction or essay prose."
        ),
        "formal_technical": (
            "The register is formal technical, employing specialist terminology "
            "and impersonal constructions typical of expository writing."
        ),
        "neutral_narrative": (
            "The register is neutral narrative — neither conspicuously elevated nor colloquial; "
            "the unmarked default of mainstream fiction."
        ),
        "colloquial": (
            "The register is colloquial, reflecting everyday speech patterns: "
            "contractions, informal vocabulary, and conversational rhythms."
        ),
        "dialect": (
            "The register is dialect-marked, with regional vocabulary, "
            "non-standard grammar, and phonologically influenced spelling."
        ),
        "archaic": (
            "The register is archaic, employing vocabulary and constructions "
            "associated with earlier historical periods."
        ),
    },
    "pov": {
        "first_person": (
            "Narrated in the first person. The 'I' voice creates intimacy "
            "and subjective immediacy, limiting the reader to one consciousness."
        ),
        "second_person": (
            "Narrated in the second person ('you'). Unusual in literary fiction, "
            "this creates a charged, direct address that implicates the reader in the action."
        ),
        "third_limited": (
            "Third-person limited narration: the narrator adheres to one character's "
            "perspective and inner life, without omniscient access to others."
        ),
        "third_omniscient": (
            "Third-person omniscient narration: the narrator can move freely between "
            "multiple characters' inner lives and survey the narrative world from above."
        ),
        "mixed": (
            "The narrative point of view is mixed or shifts across the passage, "
            "blending different modes of access and distance."
        ),
    },
    "narrative_distance": {
        "intimate": (
            "The narrative distance is intimate — the prose immerses us deeply in "
            "a character's perception and feeling, minimising the narrator's mediating presence."
        ),
        "moderate": (
            "Narrative distance is moderate: the narrator mediates experience "
            "without being obtrusive, balancing interiority with external description."
        ),
        "distant": (
            "The narrative distance is marked: the narrator observes characters "
            "from outside, offering description and event without deep psychological access."
        ),
    },
    "free_indirect_discourse": {
        "none": (
            "No free indirect discourse is detected. Narrator and character "
            "voices remain clearly distinct."
        ),
        "sparse": (
            "Sparse free indirect discourse: the narrator occasionally slides into "
            "character idiom or thought, but the technique is subtle."
        ),
        "moderate": (
            "Moderate free indirect discourse: narrator and character perspectives blend "
            "regularly, creating productive ambiguity about whose language we are reading."
        ),
        "heavy": (
            "Heavy free indirect discourse: the narration is saturated with character "
            "thought and feeling, nearly dissolving the narrator's independent voice."
        ),
    },
    "figurative_density": {
        "low": "The prose is relatively plain, with little recourse to figurative language or rhetorical embellishment.",
        "moderate": "Figurative language appears occasionally — metaphors, similes, or rhetorical schemes punctuate the prose without dominating it.",
        "high": "The prose is densely figurative: metaphor, simile, and other tropes are frequent and central to the texture of the writing.",
    },
    "tone": {
        "neutral": "The tone is neutral and unaffected — the prose does not impose a strong emotional coloring.",
        "lyrical": "The tone is lyrical: the prose attends to sound, image, and rhythm in ways that approach poetic intensity.",
        "sardonic": "The tone is sardonic — ironic, dry, and detached, with an implied critique beneath the surface.",
        "melancholic": "A pervasive melancholy inflects the prose: loss, longing, or regret color the narration and imagery.",
        "comedic": "The tone is comedic: wit, incongruity, or comic timing shape the prose.",
        "tense": "The tone is tense and charged — the prose creates suspense or unease through rhythm, diction, and scene management.",
        "contemplative": "The tone is contemplative: the prose lingers over experience, thought, or sensation in a meditative mode.",
    },
}


def _format_computable_explanation(dimension: str, value: Any) -> str:
    if dimension == "lexical_density" and isinstance(value, float):
        level = "low" if value < 0.4 else "high" if value > 0.6 else "moderate"
        return (
            f"Lexical density is {value:.2f}, indicating {level} information density. "
            + (
                "The prose is conversational, with function words outweighing content words."
                if level == "low"
                else "The prose packs a high proportion of content words, creating a dense reading texture."
                if level == "high"
                else "Content and function words are balanced, typical of mainstream narrative prose."
            )
        )

    if dimension == "sentence_length_mean" and isinstance(value, (int, float)):
        if value < 12:
            return f"Mean sentence length is {value:.1f} words — short, punchy sentences that drive pace and urgency."
        if value > 28:
            return f"Mean sentence length is {value:.1f} words — long, complex sentences that build sustained rhythm and accumulate detail."
        return f"Mean sentence length is {value:.1f} words — within the standard range for narrative prose."

    if dimension == "passive_rate" and isinstance(value, float):
        pct = value * 100
        if pct < 5:
            return f"Passive constructions account for {pct:.1f}% of verb phrases — the prose is active and direct."
        if pct > 15:
            return f"Passive constructions are frequent ({pct:.1f}%), lending an impersonal or evasive quality."
        return f"Passive voice is used moderately ({pct:.1f}%), unremarkable for this genre."

    if dimension == "type_token_ratio" and isinstance(value, float):
        level = "low" if value < 0.35 else "high" if value > 0.55 else "moderate"
        return (
            f"Type-token ratio is {value:.3f} — vocabulary richness is {level}. "
            + (
                "Repetition is a notable feature." if level == "low"
                else "The lexicon is varied and wide-ranging." if level == "high"
                else "Word variety is typical of sustained narrative prose."
            )
        )

    if dimension == "subordination_ratio" and isinstance(value, float):
        if value < 0.01:
            return f"Subordination ratio is {value:.3f} — paratactic, co-ordinating style (Hemingwayesque simplicity)."
        if value > 0.03:
            return f"Subordination ratio is {value:.3f} — hypotactic, clause-heavy syntax with embedded subordination."
        return f"Subordination ratio is {value:.3f} — moderate syntactic complexity."

    return f"{dimension.replace('_', ' ').title()}: {value}."


def _make_judgment_output(dimension: str, value: Any) -> str:
    dim_name = dimension.replace("_", " ").title()

    # Try natural language lookup first
    nl = _NATURAL_LANGUAGE_EXPLANATIONS.get(dimension, {})
    if isinstance(value, str) and value in nl:
        explanation = nl[value]
    else:
        explanation = _format_computable_explanation(dimension, value)

    return f"**{dim_name}**: {value}\n\n{explanation}"


def _classification_output(profile: dict) -> str:
    keep_keys = {
        "lexical_density", "type_token_ratio", "avg_word_length",
        "sentence_length_mean", "sentence_length_std",
        "subordination_ratio", "coordination_ratio", "passive_rate",
        "nominalization_ratio", "dialogue_ratio", "punctuation_density",
        "avg_dependency_depth", "flesch_reading_ease", "gunning_fog",
        "register", "pov", "narrative_distance", "free_indirect_discourse",
        "figurative_density", "tone", "temporal_structure",
        "sentence_variety", "dialogue_style", "imagery_type",
    }
    return json.dumps(
        {k: v for k, v in profile.items() if k in keep_keys},
        indent=2,
    )


def generate(
    input_path: Path,
    output_dir: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
    judgments_per_record: int = 3,
) -> tuple[Path, Path]:
    random.seed(seed)

    print(f"Reading {input_path} …")
    records: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    styled = [r for r in records if r.get("metadata", {}).get("style_profile")]
    print(f"Records with style_profile: {len(styled)} / {len(records)}")

    all_pairs: list[dict] = []

    for i, record in enumerate(styled):
        text = record.get("text", "")
        profile: dict = record.get("metadata", {}).get("style_profile", {})
        source = record.get("metadata", {}).get("source", "unknown")

        if not text or not profile:
            continue

        # --- Classification pair ---
        all_pairs.append({
            "instruction": random.choice(_CLF_INSTRUCTIONS),
            "input": text,
            "output": _classification_output(profile),
            "task_type": "classification",
            "source": source,
        })

        # --- Judgment pairs ---
        available = [d for d in _JUDGMENT_TEMPLATES if d in profile]
        selected = random.sample(available, min(judgments_per_record, len(available)))
        for dim in selected:
            all_pairs.append({
                "instruction": random.choice(_JUDGMENT_TEMPLATES[dim]),
                "input": text,
                "output": _make_judgment_output(dim, profile[dim]),
                "task_type": "judgment",
                "dimension": dim,
                "source": source,
            })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(styled)} records processed ({len(all_pairs)} pairs so far)")

    print(f"\nTotal pairs: {len(all_pairs)}")
    by_type: dict[str, int] = {}
    for p in all_pairs:
        t = p.get("task_type", "?")
        by_type[t] = by_type.get(t, 0) + 1
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}")

    random.shuffle(all_pairs)
    n_val = int(len(all_pairs) * val_fraction)
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"

    with open(train_path, "w", encoding="utf-8") as fh:
        for p in train_pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as fh:
        for p in val_pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nTrain : {len(train_pairs):>6} pairs → {train_path}")
    print(f"Val   : {len(val_pairs):>6} pairs → {val_path}")
    print("\nNext: update train/train_config.toml paths.data_dir, then train.")
    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate style instruction pairs for training")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--judgments-per-record", type=int, default=3,
        help="Number of judgment dimensions sampled per record (default: 3)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        print("Run the classification pipeline first:", file=sys.stderr)
        print("  python tools/style_classification/run_pipeline.py", file=sys.stderr)
        sys.exit(1)

    generate(
        input_path=args.input,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        judgments_per_record=args.judgments_per_record,
    )


if __name__ == "__main__":
    main()
