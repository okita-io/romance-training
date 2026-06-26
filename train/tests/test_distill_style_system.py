"""Tests for Style in Fiction manuscript parsing and distillation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.style_extraction.manuscript_parser import (
    extract_checklist_items,
    parse_manuscript,
)
from tools.style_extraction.distill_style_system import distill


SAMPLE_PARSED = """\
<PAGE>
Contents
Introduction 1
<PAGE>
Introduction
Aim
An earlier book in this series was written with the aim of showing the student of English that examining the language of a literary text can be a means to a fuller understanding and appreciation of the writer's artistic achievement. The present book is written with the same aim in mind, this time taking prose fiction, not poetry, as the object of study.
<PAGE>
CHAPTER 3
A method of analysis
and some examples
This chapter has the practical purpose of showing how the apparatus of linguistic description can be used in analysing the style of a prose text.
3.1 A checklist of linguistic and stylistic categories
The categories are placed under four general headings.
A: Lexical categories
1 GENERAL. Is the vocabulary simple or complex? formal or colloquial?
2 NOUNS. Are the nouns abstract or concrete?
B: Grammatical categories
1 SENTENCE TYPES. Does the author use only statements?
C: Figures of speech, etc.
1 GRAMMATICAL AND LEXICAL. Are there any cases of formal and structural repetition?
D: Context and cohesion
1 COHESION. Does the text contain logical or other links between sentences?
<PAGE>
7.5.2 The principle of climax: 'last is most important'
We have now encountered two principles of saliency, the principle of end-focus, which is phonological, and the principle of subordination, which is syntactic. They operate on different levels, and are independent of one another. However, they are linked by a further phonological principle, which has important repercussions in syntax, and which we call the PRINCIPLE OF CLIMAX:
In a sequence of interrelated tone units, the final position tends to be the major focus of information.
This principle can be seen as an extension of the end-focus principle, for it says for a sequence of tone-units what the end-focus principle says for individual tone-units, that 'last is most important'.
"""


def test_parse_manuscript_skips_front_matter() -> None:
    sections = parse_manuscript(SAMPLE_PARSED, min_section_chars=80)
    titles = [s.title for s in sections]
    assert "Contents" not in titles
    assert any("Introduction" in t or s.section_id for s in sections for t in [s.title])


def test_parse_manuscript_detects_checklist_section() -> None:
    sections = parse_manuscript(SAMPLE_PARSED, min_section_chars=80)
    checklist_sections = [s for s in sections if s.is_checklist or s.section_id == "3.1"]
    assert checklist_sections
    items = extract_checklist_items(checklist_sections[0])
    assert any("vocabulary simple or complex" in i["prompt"].lower() for i in items)
    assert any(i["group"] == "a" for i in items)


def test_parse_manuscript_tags_textual_section() -> None:
    sections = parse_manuscript(SAMPLE_PARSED, min_section_chars=80)
    climax = [s for s in sections if "climax" in s.title.lower()]
    assert climax
    assert "textual" in climax[0].categories


def test_distill_writes_rubric_shape(tmp_path: Path) -> None:
    src = tmp_path / "sample.parsed.md"
    src.write_text(SAMPLE_PARSED, encoding="utf-8")
    knowledge, rubric, analysis = distill(src, min_section_chars=80)
    assert knowledge
    assert rubric["version"] == "2.0"
    assert rubric["checklist"]
    assert rubric["textual_principles"]
    assert analysis["system_prompt"]
    assert any(d["id"] == "climax" for d in rubric["textual_principles"])
