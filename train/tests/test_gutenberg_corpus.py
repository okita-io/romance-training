"""Tests for Gutenberg corpus splitting."""

from __future__ import annotations

from tools.data_preparation.gutenberg_corpus import (
    clean_gutenberg_prose,
    detect_gutenberg_play,
    is_gutenberg_play,
    split_gutenberg_corpus,
    strip_gutenberg_boilerplate,
    title_slug,
)

SAMPLE = """\
The Project Gutenberg EBook of Nightmare Abbey, by Thomas Love Peacock

Title: Nightmare Abbey
Author: Thomas Love Peacock

*** START OF THIS PROJECT GUTENBERG EBOOK NIGHTMARE ABBEY ***

[Illustration: Front cover]

NIGHTMARE ABBEY

By Thomas Love Peacock

CHAPTER I

Once upon a time there lived a philosopher in a secluded abbey.
The philosopher spoke at length about the vanity of human wishes.
He walked through the corridors with measured steps and dark thoughts.

*** END OF THIS PROJECT GUTENBERG EBOOK NIGHTMARE ABBEY ***
"""

ST_GEORGE_SAMPLE = """\
This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever.  You may copy it, give it away or


Title: St. George and St. Michael

Author: George MacDonald

Language: English

Produced by Charles Aldarondo, Charles Franks and the
Distributed Proofreading Team


ST. GEORGE AND ST. MICHAEL

CONTENTS OF VOL. I.

CHAPTER I. DOROTHY AND RICHARD.

CHAPTER II. RICHARD AND HIS FATHER.


CHAPTER I. DOROTHY AND RICHARD.

The morning sun shone brightly over Raglan Castle.
"""


def test_strip_gutenberg_boilerplate() -> None:
    body = strip_gutenberg_boilerplate(SAMPLE)
    assert "START OF" not in body
    assert "END OF" not in body
    assert "philosopher" in body


def test_clean_gutenberg_prose() -> None:
    body = clean_gutenberg_prose(SAMPLE)
    assert "Illustration" not in body
    assert "Title:" not in body
    assert "philosopher" in body
    assert "CHAPTER I" in body


def test_clean_gutenberg_prose_strips_credits_and_toc() -> None:
    body = clean_gutenberg_prose(ST_GEORGE_SAMPLE)
    assert "Produced by" not in body
    assert "Distributed Proofreading" not in body
    assert "CONTENTS OF VOL" not in body
    assert "morning sun shone brightly" in body


def test_split_gutenberg_corpus() -> None:
    books = split_gutenberg_corpus(SAMPLE)
    assert len(books) == 1
    assert books[0]["title"] == "Nightmare Abbey"
    assert books[0]["author"] == "Thomas Love Peacock"
    assert "philosopher" in books[0]["text"]


def test_title_slug() -> None:
    assert title_slug("The Castle of Otranto") == "the_castle_of_otranto"


def test_detect_gutenberg_play_shakespeare() -> None:
    assert detect_gutenberg_play("The Tragedy of Romeo and Juliet", "") == "title"


def test_detect_gutenberg_play_keeps_novelization() -> None:
    title = "The Round-Up: A Romance of Arizona; Novelized from Edmund Day's Melodrama"
    assert not is_gutenberg_play(title, "")


def test_detect_gutenberg_play_body_signals() -> None:
    text = "\n".join([
        "DRAMATIS PERSONAE",
        "ACT I",
        "SCENE I",
        "[Enter KING]",
        "[Enter QUEEN]",
        "[Exit KING]",
    ] * 5)
    assert detect_gutenberg_play("Some Obscure Work", text) == "body"


def test_filter_plays_excludes_shakespeare() -> None:
    from tools.data_preparation.convert_hf_parquet import filter_plays

    books = [
        {
            "story_id": 0,
            "title": "The Tragedy of Hamlet",
            "author": "Shakespeare, William",
            "text": "word " * 200,
        },
        {
            "story_id": 1,
            "title": "Pride and Prejudice",
            "author": "Austen, Jane",
            "text": "word " * 200,
        },
    ]
    kept, skipped = filter_plays(books, exclude_plays=True)
    assert len(kept) == 1
    assert kept[0]["title"] == "Pride and Prejudice"
    assert len(skipped) == 1
    assert skipped[0]["play_signal"] == "title"
