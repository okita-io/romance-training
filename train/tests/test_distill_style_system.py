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


SAMPLE_PAGE_BREAK = """\
<PAGE>
Contents
Introduction 1
<PAGE>
Introduction
Aim
An earlier book in this series was written with the aim of showing the student of English that examining the language of a literary text can be a means to a fuller understanding and appreciation of the writer's artistic achievement.
<PAGE>
Style in Fiction
Some ordinary body text that continues the introduction with enough length to keep the section alive across pages.
<PAGE>
A method of analysis and some examples
More body text on a later page, again long enough to matter for the parser.
<PAGE>
3.1 A checklist of linguistic and stylistic categories
A: Lexical categories
[For notes (i–xiv) on the categories see pp. 66–7]
1 GENERAL. Is the vocabulary simple or complex?
B: Grammatical categories
4 CLAUSE STRUCTURE. Is there anything significant about clause elements?
5 NOUN PHRASES. Are they relatively simple or complex?
6 VERB PHRASES. Are there any significant departures from the use of the simple past tense? For example, notice occurrences and functions of the present tense; of the perfective
62
<PAGE>
A method of analysis and some examples
aspect (e.g. has/had appeared); of modal auxiliaries (e.g. can, must, would, etc.). Look out for phrasal verbs and how they are used.
7 OTHER PHRASE TYPES. Is there anything to be said about other phrase types?
D: Context and cohesion
1 COHESION \\( ^{[43]} \\) . Does the text contain logical or other links between sentences?
<PAGE>
A method of analysis and some examples
3.2 Notes on the categories
Some notes text follows here so the section has content beyond the heading itself.
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


def _checklist_items(sample: str) -> list[dict]:
    sections = parse_manuscript(sample, min_section_chars=40)
    items: list[dict] = []
    for s in sections:
        if s.is_checklist or s.section_id == "3.1":
            items.extend(extract_checklist_items(s))
    return items


def test_checklist_items_are_not_merged() -> None:
    items = _checklist_items(SAMPLE_PAGE_BREAK)
    subs = [i["subgroup"] for i in items]
    assert "Clause structure" in subs
    assert "Noun phrases" in subs
    assert "Verb phrases" in subs
    assert "Other phrase types" in subs
    clause = next(i for i in items if i["subgroup"] == "Clause structure")
    assert "NOUN PHRASES" not in clause["prompt"]
    assert "VERB PHRASES" not in clause["prompt"]


def test_running_header_stripped_across_page_break() -> None:
    items = _checklist_items(SAMPLE_PAGE_BREAK)
    verb = next(i for i in items if i["subgroup"] == "Verb phrases")
    assert "A method of analysis" not in verb["prompt"]
    assert "of the perfective aspect (e.g. has/had appeared)" in verb["prompt"]


def test_checklist_skips_notes_stub_and_cleans_markers() -> None:
    items = _checklist_items(SAMPLE_PAGE_BREAK)
    assert not any("For notes (i–xiv)" in i["prompt"] for i in items)
    cohesion = next(i for i in items if i["subgroup"] == "Cohesion")
    assert cohesion["prompt"].startswith("Does the text contain")


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
