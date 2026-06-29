"""Tests for Creative Commons license and back-matter stripping."""

from __future__ import annotations

from tools.data_preparation.strip_license_agreement import strip_license_agreement


def test_strip_note_about_this_book_suffix() -> None:
    raw = (
        "========== i booked us ringside seats at the polynesian luau, riding high on a fresh "
        "round of sympathy whuffie, and dan and i drank a dozen lapu-lapus. "
        "========================================== a note about this book, february 12, 2004: "
        "========================================== as you will see, when you read the text beneath "
        "this section, i released this book under the terms of a creative commons license. "
        "6. limitation on liability. except to the extent required by applicable law, in no event "
        "will licensor be liable to you on any legal theory for any special, incidental, "
        "consequential, punitive or exemplary damages arising out of this license."
    )
    result = strip_license_agreement(raw)
    assert result.stripped
    assert result.reason == "note_about_this_book"
    assert result.text.startswith("i booked us ringside seats")
    assert "limitation on liability" not in result.text.lower()
    assert "creative commons license" not in result.text.lower()


def test_strip_cc_legal_code_block() -> None:
    raw = (
        "my name is marcus yallow. i was tortured by my country, but i still love it here. "
        "thank you all for giving me the tools to think and write about these ideas. "
        "creative commons creative commons legal code attribution-noncommercial-sharealike "
        "3.0 unported creative commons corporation is not a law firm and does not provide "
        "legal services. 6. limitation on liability. except to the extent required by applicable "
        "law, in no event will licensor be liable to you on any legal theory for any special, "
        "incidental, consequential, punitive or exemplary damages arising out of this license "
        "or the use of the work, even if licensor has been advised of the possibility of such "
        "damages. creative commons may be contacted at http://creativecommons.org/."
    )
    result = strip_license_agreement(raw)
    assert result.stripped
    assert result.reason == "cc_legal_code"
    assert result.text.startswith("my name is marcus yallow")
    assert result.text.endswith("these ideas.")
    assert "licensor be liable" not in result.text.lower()


def test_keep_narrative_creative_commons_mention() -> None:
    raw = (
        "He hosted it on the internet archive's alexandria mirror in egypt, where they'd host "
        "anything for free so long as you'd put it under the creative commons license, which "
        "let anyone remix it and share it."
    )
    result = strip_license_agreement(raw)
    assert not result.stripped
    assert result.text == raw


def test_strip_publisher_copyright_footer() -> None:
    raw = (
        "She closed the door behind her and walked into the rain. "
        "text: copyright 2003 cory doctorow doctorow@craphound.com "
        "tor books, january 2003 isbn: 0765304368 all rights reserved."
    )
    result = strip_license_agreement(raw)
    assert result.stripped
    assert result.reason == "publisher_copyright_footer"
    assert result.text == "She closed the door behind her and walked into the rain."


def test_strip_afterword_suffix() -> None:
    raw = (
        '"Ange, I\'ve never thought more clearly in my whole life." She kissed me then, '
        "and I kissed her back, and it was some time before we went out for that burrito. "
        "afterword by bruce schneier i'm a security technologist. my job is making people secure. "
        "bibliography no writer creates from scratch -- we all engage in what isaac newton called "
        '"standing on the shoulders of giants." acknowledgments this book owes a tremendous debt.'
    )
    result = strip_license_agreement(raw)
    assert result.stripped
    assert result.reason == "afterword"
    assert result.text.endswith("burrito")
    assert "bruce schneier" not in result.text.lower()
    assert "bibliography" not in result.text.lower()


def test_drop_fiction_disclaimer_only_chunk() -> None:
    raw = (
        "this novel is entirely a work of fiction. the names, characters, and incidents "
        "portrayed in it are the work of the author's imagination. any resemblance to actual "
        "persons is coincidental. about the author jane doe lives in london."
    )
    result = strip_license_agreement(raw)
    assert result.stripped
    assert result.reason == "fiction_disclaimer_only"
    assert result.text == ""


def test_keep_narrative_acknowledgment_in_dialogue() -> None:
    raw = (
        '"A most unnecessary acknowledgment, my dear child--it is patent to the dullest observer. '
        'But, now, Edith--look here--this is serious, mind!"'
    )
    result = strip_license_agreement(raw)
    assert not result.stripped
    assert result.text == raw


def test_strip_biography_bibliography_after_the_end() -> None:
    raw = (
        '"God bless you both!" "God bless you, Paul!" the end. * * * * * * '
        "biography and bibliography mrs. adeline dutton whitney was born in boston, "
        "september 15, 1824, and published many novels."
    )
    result = strip_license_agreement(raw)
    assert result.stripped
    assert result.reason == "biography_bibliography"
    assert result.text.endswith('"God bless you, Paul!"')
    assert "whitney" not in result.text.lower()
