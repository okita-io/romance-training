"""Tests for chapter dedication and boilerplate stripping."""

from __future__ import annotations

from tools.data_preparation.strip_dedication import strip_dedication_and_boilerplate


def test_strip_bakka_chapter_dedication() -> None:
    raw = (
        "this chapter is dedicated to bakkaphoenix books in toronto, canada. bakka is the oldest "
        "science fiction bookstore in the world, and it made me the mutant i am today. i wandered in "
        "for the first time around the age of 10 and asked for some recommendations. "
        "[[bakkaphoenix books: http://www.bakkaphoenixbooks.com/ 697 queen street west, "
        "toronto on canada m6j1e6, +1 416 963 9993]] "
        "tanya huff took me back into the used section and pressed a copy of h. beam piper's "
        "\"little fuzzy\" into my hands, and changed my life forever. "
        "[[bakkaphoenix books: http://www.bakkaphoenixbooks.com/]] "
        "i'm a senior at cesar chavez high in san francisco's sunny mission district, and that "
        "makes me one of the most surveilled people in the world. my name is marcus yallow, but "
        "back when this story starts, i was going by w1n5t0n. pronounced \"winston.\" "
        "not pronounced \"double-you-one-enn-five-tee-zero-enn\" -- unless you're a clueless "
        "disciplinary officer who's far enough behind the curve that you still call the internet "
        "\"the information superhighway.\" i know just such a clueless person, and his name is "
        "fred benson, one of three vice-principals at cesar chavez."
    )
    result = strip_dedication_and_boilerplate(raw)
    assert result.stripped
    assert result.text.startswith("i'm a senior at cesar chavez high")
    assert "bakkaphoenix" not in result.text.lower()
    assert "[[bakkaphoenix" not in result.text


def test_strip_amazon_mid_chapter_dedication() -> None:
    raw = (
        "this chapter is dedicated to amazon.com, the largest internet bookseller in the world. "
        "amazon is amazing -- a \"store\" where you can get practically any book ever published. "
        "amazon has always treated me like gold -- the founder, jeff bezos, even posted a "
        "reader-review for my first novel. "
        "[[amazon: http://www.amazon.com/exec/obidos/asin/0765319853/downandoutint-20]] "
        "amazon is amazing -- a \"store\" where you can get practically any book ever published. "
        "amazon has always treated me like gold -- the founder, jeff bezos, even posted a "
        "reader-review for my first novel. "
        "[[amazon: http://www.amazon.com/exec/obidos/asin/0765319853/downandoutint-20]] "
        "\"i'm thinking of majoring in physics when i go to berkeley,\" darryl said. his dad taught "
        "at the university of california at berkeley, which meant he'd get free tuition when he "
        "was old enough to be admitted. and there'd never been any question in darryl's household "
        "about whether he'd go."
    )
    result = strip_dedication_and_boilerplate(raw)
    assert result.stripped
    assert result.text.startswith('"i\'m thinking of majoring in physics')
    assert "jeff bezos" not in result.text.lower()


def test_no_strip_normal_chapter() -> None:
    raw = (
        "The morning sun was barely over the hills when Mara left the croft with a basket on her arm "
        "and a purpose in her stride that had not troubled her sleep for many weeks before."
    )
    result = strip_dedication_and_boilerplate(raw)
    assert not result.stripped
    assert result.text == raw


def test_strip_free_ebooks_bookrix_boilerplate() -> None:
    raw = (
        "m.b. julien anthology complex bookrix gmbh & co. kg 81371 munich start this book was "
        "distributed courtesy of: for your own unlimited reading and free ebooks today, visit: "
        "http://www.free-ebooks.net share this ebook with anyone and everyone automatically by "
        "selecting any of the options below: to show your appreciation to the author and help "
        "others have wonderful reading experiences and find helpful information too, we'd be very "
        "grateful if you'd kindly post your comments for this book here. copyright information "
        "free-ebooks.net respects the intellectual property of others. when a book's copyright "
        "owner submits their work to free-ebooks.net, they are granting us permission to distribute "
        "such material. unless otherwise stated in this book, this permission is not passed onto "
        "others. as such, redistributing this book without the copyright owner's permission can "
        "constitute copyright infringement. if you believe that your work has been used in a manner "
        "that constitutes copyright infringement, please follow our notice and procedure for making "
        "claims of copyright infringement as seen in our terms of service here: "
        "http://www.free-ebooks.net/tos.html composition 1, part 1"
    )
    result = strip_dedication_and_boilerplate(raw)
    assert result.stripped
    assert result.reason == "ebook_boilerplate"
    assert result.text == ""
    assert "free-ebooks.net" not in result.text


def test_strip_colon_prefixed_chapter_title() -> None:
    raw = (
        ": this anthologic life last night, i had a dream. i'm walking alongside a row of "
        "parked cars in broad daylight, peering through each car's driver seat window as i "
        "pass by. this goes on for a while until i realize that none of the cars have drivers."
    )
    result = strip_dedication_and_boilerplate(raw)
    assert result.stripped
    assert result.reason == "colon_chapter_title"
    assert result.text.startswith("last night, i had a dream")


def test_drop_epilogue_contents_back_matter() -> None:
    raw = (
        "epilogue contents - about this book read this first introduction do something great britain "
        "other editions the copyright thing donations and a word to teachers and librarians "
        "creative commons read this first this book is distributed under a creative commons "
        "attribution-noncommercial-sharealike 3.0 license. that means: you are free: "
        "* to share — to copy, distribute and transmit the work"
    )
    result = strip_dedication_and_boilerplate(raw)
    assert result.stripped
    assert result.reason == "back_matter_toc"
    assert result.text == ""


def test_keep_narrative_epilogue() -> None:
    raw = (
        "epilogue the following september i learned of another tragedy. while i was on extended "
        "assignment for my new job, word came that russell perished in a blaze."
    )
    result = strip_dedication_and_boilerplate(raw)
    assert not result.stripped
    assert result.text.startswith("epilogue the following september")


def test_strip_bookrix_publication_trailer() -> None:
    raw = (
        "She closed the door behind her and walked into the rain. "
        "Whatever happened next, she was finally ready for it. "
        "publication date: november 25th 2010 https://www.bookrix.com/-love09"
    )
    result = strip_dedication_and_boilerplate(raw)
    assert result.stripped
    assert result.reason == "bookrix_publication_trailer"
    assert result.text.endswith("ready for it.")
    assert "bookrix.com" not in result.text
    assert "publication date" not in result.text.lower()


def test_strip_bookrix_publication_trailer_with_isbn() -> None:
    raw = (
        "The footsteps faded into silence. "
        "publication date: september 5th 2016 https://www.bookrix.com/-ktae0fcbd24dc75 "
        "isbn: 978-3-7396-7198-7"
    )
    result = strip_dedication_and_boilerplate(raw)
    assert result.stripped
    assert result.text == "The footsteps faded into silence."


def test_strip_free_ebooks_trailer_after_narrative() -> None:
    raw = (
        "you slowly begin to realize that the sun is the same, in a relative way, "
        "but you're older. shorter of breath, and one day closer to death. "
        "this book was distributed courtesy of: for your own unlimited reading and free ebooks today, visit: "
        "http://www.free-ebooks.net share this ebook with anyone and everyone automatically by "
        "selecting any of the options below: to show your appreciation to the author and help "
        "others have wonderful reading experiences and find helpful information too, we'd be very "
        "grateful if you'd kindly post your comments for this book here. copyright information "
        "free-ebooks.net respects the intellectual property of others. when a book's copyright "
        "owner submits their work to free-ebooks.net, they are granting us permission to distribute "
        "such material. unless otherwise stated in this book, this permission is not passed onto "
        "others. as such, redistributing this book without the copyright owner's permission can "
        "constitute copyright infringement. if you believe that your work has been used in a manner "
        "that constitutes copyright infringement, please follow our notice and procedure for making "
        "claims of copyright infringement as seen in our terms of service here: "
        "http://www.free-ebooks.net/tos.html publisher: bookrix gmbh & co. kg implerstraße 24 "
        "81371 munich germany"
    )
    result = strip_dedication_and_boilerplate(raw)
    assert result.stripped
    assert result.reason == "ebook_boilerplate_trailer"
    assert result.text.endswith("one day closer to death.")
    assert "free-ebooks.net" not in result.text.lower()
    assert "bookrix" not in result.text.lower()
