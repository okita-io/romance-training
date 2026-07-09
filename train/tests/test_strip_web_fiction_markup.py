"""Tests for Literotica-style HTML and author-note stripping."""

from __future__ import annotations

from tools.data_preparation.strip_web_fiction_markup import (
    strip_html_markup,
    strip_leading_author_notes,
    strip_web_fiction_markup,
)

INSTALLMENT_NOTE = (
    "<I>[At last, the next installment has been written. I should have previously "
    "advised that this story is fiction but was inspired by an actual relationship.]</I>\n\n"
    "I woke the next morning around 7:30 a.m."
)

AUTHORS_NOTE_EM = (
    "<em>[Author's note: this is far from what I usually write, and likely far from "
    "what is generally found on Literotica. It is, essentially, a re-telling of what "
    "happens when someone ignores warnings and trigger signs and almost falls prey to a "
    "blackmailing hacker.]</em>\n\n"
    "I discovered a long time ago that I enjoyed gay sex in addition to hetero sex."
)

AUTHORS_NOTE_I_STARS = (
    "<I>Author's Note: This one's a bit short, so my apologies. Sorry also, about the "
    "long waits between installments, but I do like to see how everyone reacts to each "
    "chapter before I go ahead and submit the updates.</I>\n\n***\n\n"
    "Rose was restless that night."
)

META_PREAMBLE = (
    "Story was inspired by already existing stories... no original work was stolen "
    "in the making of this tale.\n\n"
    "________________________________\n\n"
    "\"And another one...\" I said in disgust"
)

HTML_EMPHASIS = (
    "It is why we learn and practice crew resource management philosophy, <i>da</i>? "
    "We must rely on skills and insights of every member of this crew. <i>That</i> is how."
)


def test_strip_html_tags_and_entities() -> None:
    cleaned = strip_html_markup(HTML_EMPHASIS)
    assert "<" not in cleaned
    assert cleaned == (
        "It is why we learn and practice crew resource management philosophy, da? "
        "We must rely on skills and insights of every member of this crew. That is how."
    )


def test_strip_installment_author_note_in_i_tag() -> None:
    result = strip_web_fiction_markup(INSTALLMENT_NOTE)
    assert result.stripped
    assert result.text.startswith("I woke the next morning")
    assert "<I>" not in result.text
    assert "installment" not in result.text.lower()


def test_strip_authors_note_in_em_tag() -> None:
    result = strip_web_fiction_markup(AUTHORS_NOTE_EM)
    assert result.stripped
    assert result.text.startswith("I discovered a long time ago")
    assert "Author's note" not in result.text


def test_strip_authors_note_with_stars_divider() -> None:
    result = strip_web_fiction_markup(AUTHORS_NOTE_I_STARS)
    assert result.stripped
    assert result.text.startswith("Rose was restless")
    assert "***" not in result.text


def test_strip_meta_preamble_before_underscore_divider() -> None:
    result = strip_web_fiction_markup(META_PREAMBLE)
    assert result.stripped
    assert result.text.startswith('"And another one..."')
    assert "inspired by already existing" not in result.text.lower()


def test_strip_leading_author_notes_only() -> None:
    result = strip_leading_author_notes(AUTHORS_NOTE_EM)
    assert result.stripped
    assert "<em>" not in result.text
    assert result.text.startswith("I discovered a long time ago")
