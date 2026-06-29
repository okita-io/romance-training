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


def test_strip_bierce_transcriber_toc_and_preface() -> None:
    raw = (
        "Transcribed from the 1918 Boni and Liveright edition by David Price, email "
        "ccx074@pglaf.org\n\n"
        "[Picture: Public domain cover]\n\n"
        "PRESENT AT A HANGING AND OTHER GHOST STORIES\n\n"
        "By Ambrose Bierce\n\n"
        "CONTENTS\n\n"
        "THE WAYS OF GHOSTS                                         PAGE PRESENT AT A HANGING "
        "327 A COLD GREETING 331 THE MIDDLE TOE 335\n\n"
        "THE WAYS OF GHOSTS\n\n"
        "_My peculiar relation to the writer of the following narratives is such that I "
        "must ask the reader to overlook the absence of explanation as to how these papers "
        "passed into my possession._\n\n"
        "PRESENT AT A HANGING\n\n"
        "AN old man named Daniel Baker, living near Lebanon, Iowa, was suspected "
        "by his neighbors of having murdered a peddler who had obtained permission "
        "to pass the night at his house. This was in 1853."
    )
    result = strip_front_matter(raw, min_words=10)
    assert result.stripped
    assert result.text.startswith("AN old man named Daniel Baker")
    assert "ccx074@pglaf.org" not in result.text
    assert "PRESENT AT A HANGING 327" not in result.text
    assert "_My peculiar relation" not in result.text


def test_strip_publisher_contents_and_chapter_epigraph() -> None:
    raw = (
        'Author _of_ "Ten Minute Stories," "Julius Le Vallon," "The Wave," etc. [Illustration]\n\n'
        "NEW YORK E. P. DUTTON & CO. 681 FIFTH AVENUE\n\n"
        "COPYRIGHT, 1917, BY E. P. DUTTON & CO. Printed in the United States of America\n\n"
        "CONTENTS\n\n"
        "CHAPTER                              PAGE\n\n"
        "I. THE TRYST                       1\n\n"
        "II. THE TOUCH OF PAN               16\n\n"
        "DAY AND NIGHT STORIES\n\n"
        "I\n\n"
        "THE TRYST\n\n"
        '"_Je suis la première au rendez-vous. Je vous attends._"\n\n'
        "As he got out of the train at the little wayside station he remembered the conversation "
        "as if it had been yesterday, instead of fifteen years ago--and his heart went thumping "
        "against his ribs so violently that he almost heard it."
    )
    result = strip_front_matter(raw, min_words=10)
    assert result.stripped
    assert result.text.startswith("As he got out of the train")
    assert "DUTTON" not in result.text
    assert "I. THE TRYST" not in result.text
    assert "Je suis la première" not in result.text


def test_strip_victorian_title_page_and_illustrations_list() -> None:
    raw = (
        "TALES OF THE WONDER CLUB. BY DRYASDUST. VOL. III. ILLUSTRATED BY JOHN JELLICOE "
        "and VAL PRINCE, AFTER DESIGNS BY THE AUTHOR. HARRISON & SONS, 59, PALL MALL, "
        "_Booksellers to the Queen and H.R.H. the Prince of Wales._\n\n"
        "_All rights reserved._\n\n"
        "LONDON: PRINTED BY A. HUDSON AND CO., 160 WANDSWORTH ROAD, S.W. LIST OF "
        "ILLUSTRATIONS. PAGE THE ABDUCTION             _Frontispiece_ THE FIRE "
        "                   _Title Page_\n\n"
        "[Illustration: THE CURIOSITY SHOP]\n\n"
        "PREFACE TO VOL. III. Before taking leave of his readers, the author would inform "
        "them that at the commencement of these Tales, the earlier ones dating some thirty "
        "years back, nothing was further from his intentions than to continue the series."
    )
    result = strip_front_matter(raw, min_words=10)
    assert result.stripped
    assert result.text.startswith("Before taking leave of his readers")
    assert "PALL MALL" not in result.text
    assert "LIST OF ILLUSTRATIONS" not in result.text
    assert "PREFACE TO VOL" not in result.text
