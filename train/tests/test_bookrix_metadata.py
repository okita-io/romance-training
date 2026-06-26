"""Tests for BookRix metadata parsing."""

from __future__ import annotations

from tools.data_preparation.bookrix_metadata import (
    parse_bookrix_header,
    parse_bookrix_metadata,
    parse_bookrix_url,
)


def test_parse_bookrix_url() -> None:
    author, title = parse_bookrix_url(
        "https://www.bookrix.com/_ebook-nidhi-agrawal-a-cute-love-story/"
    )
    assert author == "Nidhi Agrawal"
    assert "Cute Love Story" in title


def test_parse_bookrix_header() -> None:
    text = "Lorelei Sutton A Howl In The Night       Suicidal Beginnings    The wind swirls"
    author, title = parse_bookrix_header(text)
    assert author == "Lorelei Sutton"
    assert title == "A Howl In The Night"


def test_parse_bookrix_metadata_prefers_header() -> None:
    author, title, url = parse_bookrix_metadata(
        "https://www.bookrix.com/_ebook-katy-wong-arranged/",
        "Katy Wong Arranged (Completed)        Prologue",
    )
    assert author == "Katy Wong"
    assert "Arranged" in (title or "")
