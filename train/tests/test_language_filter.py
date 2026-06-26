"""Tests for English language filtering."""

from __future__ import annotations

from tools.data_preparation.language_filter import (
    classify_language,
    has_non_latin_script,
    is_english_text,
)

ENGLISH = (
    "She had come from North Carolina, newly divorced, to start over. "
    "The rain fell in torrents and the wind was cold. "
    "He said that he would love her forever, but she did not believe him."
)

ARABIC = "قصص تونسية احببت ارمله و عراض مملح و تحس عينيه الذابلة"

HINDI = "अभिषेक दळवी कदाचित हेच आहे प्रेम sample लेखकाची ही कथाही अशाच दोन पात्रांशी निगडित आहे"

SPANISH = (
    "Anna Harris No me olvides. Ella dijo que el amor era para siempre "
    "y que no podía vivir sin él en la mansión."
)


def test_english_passes() -> None:
    assert is_english_text(ENGLISH)
    assert classify_language(ENGLISH) == "en"


def test_non_latin_script_rejected() -> None:
    assert has_non_latin_script(ARABIC)
    assert classify_language(ARABIC) == "non_en"
    assert not is_english_text(ARABIC)


def test_hindi_rejected() -> None:
    assert not is_english_text(HINDI)


def test_spanish_rejected() -> None:
    assert not is_english_text(SPANISH)
