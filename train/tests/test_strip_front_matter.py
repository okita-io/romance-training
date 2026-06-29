"""Tests for Gutenberg front-matter stripping."""

from __future__ import annotations

from tools.data_preparation.strip_front_matter import (
    find_lone_roman_chapter_splits,
    strip_front_matter,
)


def test_lone_roman_i_between_paragraphs() -> None:
    raw = (
        "THE CASTLE OF OTRANTO\n\n"
        "BY HORACE WALPOLE\n\n"
        "I\n\n"
        "It was the best of times, and the old castle stood upon the hill "
        "watching over the valley below with silent patience."
    )
    result = strip_front_matter(raw, min_words=10)
    assert result.stripped
    assert result.reason == "roman_chapter_marker"
    assert result.text.startswith("It was the best")
    assert result.text.split("\n\n")[0] != "I"


def test_lone_roman_ix_between_paragraphs() -> None:
    raw = (
        "Some title page line\n\n"
        "Another imprint line\n\n"
        "IX\n\n"
        "The ninth chapter opened with rain against the window panes, and "
        "everyone in the house seemed to feel the weight of the coming storm."
    )
    assert find_lone_roman_chapter_splits(raw) == ["IX"]
    result = strip_front_matter(raw, min_words=10)
    assert result.stripped
    assert result.text.startswith("The ninth chapter")


def test_no_strip_when_narrative_from_start() -> None:
    raw = (
        "The morning sun shone brightly over Raglan Castle. Dorothy walked slowly "
        "through the courtyard, her thoughts far from the feast preparations."
    )
    result = strip_front_matter(raw)
    assert not result.stripped
    assert result.text == raw


def test_strip_preface_and_contents_before_story() -> None:
    raw = (
        "THE MORNING POST says: clever first novels.\n\n"
        "PREFACE\n\n"
        "These stories have been written in the hopes of giving some pleasant "
        "qualms to their reader, so that anyone may cast a glance into the corners.\n\n"
        "CONTENTS\n\n"
        "THE ROOM IN THE TOWER 1\n\n"
        "THE ROOM IN THE TOWER\n\n"
        "It is probable that everybody who is at all a constant dreamer has had "
        "at least one experience of an event or a sequence of circumstances which "
        "have come to his mind in sleep being subsequently realised in the material world."
    )
    result = strip_front_matter(raw, min_words=10)
    assert result.stripped
    assert result.text.startswith("It is probable")


def test_pronoun_i_paragraph_not_treated_as_chapter() -> None:
    raw = (
        "I went to the market that morning and bought bread, cheese, and a bottle "
        "of wine before returning home through the narrow streets."
    )
    result = strip_front_matter(raw)
    assert not result.stripped
