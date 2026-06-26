"""Tests for Gothic Gutenberg conversion helpers."""

from __future__ import annotations

from tools.data_preparation.convert_gutenberg_gothic import filter_english_books

ENGLISH = {
    "title": "Frankenstein",
    "author": "Shelley, Mary",
    "text": (
        "It was on a dreary night of November that I beheld the accomplishment of my toils. "
        "The rain pattered dismally against the panes, and my candle was nearly burnt out."
    )
    * 20,
}

SPANISH = {
    "title": "Amor Eterno",
    "author": "Autor, Desconocido",
    "text": (
        "Anna Harris No me olvides. Ella dijo que el amor era para siempre "
        "y que no podía vivir sin él en la mansión oscura."
    )
    * 20,
}


def test_filter_english_books_keeps_english() -> None:
    kept, skipped = filter_english_books([ENGLISH, SPANISH])
    assert len(kept) == 1
    assert kept[0]["title"] == "Frankenstein"
    assert len(skipped) == 1
    assert skipped[0]["title"] == "Amor Eterno"


def test_filter_english_books_disabled() -> None:
    kept, skipped = filter_english_books([ENGLISH, SPANISH], english_only=False)
    assert len(kept) == 2
    assert skipped == []
