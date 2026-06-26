"""Tests for Gutenberg corpus splitting."""

from __future__ import annotations

from tools.data_preparation.gutenberg_corpus import (
    split_gutenberg_corpus,
    strip_gutenberg_boilerplate,
    title_slug,
)

SAMPLE = """\
The Project Gutenberg EBook of Nightmare Abbey, by Thomas Love Peacock

Title: Nightmare Abbey
Author: Thomas Love Peacock

*** START OF THIS PROJECT GUTENBERG EBOOK NIGHTMARE ABBEY ***

NIGHTMARE ABBEY

By Thomas Love Peacock

CHAPTER I

Once upon a time there lived a philosopher in a secluded abbey.
The philosopher spoke at length about the vanity of human wishes.
He walked through the corridors with measured steps and dark thoughts.

*** END OF THIS PROJECT GUTENBERG EBOOK NIGHTMARE ABBEY ***
"""


def test_strip_gutenberg_boilerplate() -> None:
    body = strip_gutenberg_boilerplate(SAMPLE)
    assert "START OF" not in body
    assert "END OF" not in body
    assert "philosopher" in body


def test_split_gutenberg_corpus() -> None:
    books = split_gutenberg_corpus(SAMPLE)
    assert len(books) == 1
    assert books[0]["title"] == "Nightmare Abbey"
    assert books[0]["author"] == "Thomas Love Peacock"
    assert "philosopher" in books[0]["text"]


def test_title_slug() -> None:
    assert title_slug("The Castle of Otranto") == "the_castle_of_otranto"
