"""Tests for Victorian ``. section title`` stripping."""

from __future__ import annotations

from tools.data_preparation.strip_victorian_section_title import strip_victorian_section_title


def test_strip_period_delimited_title() -> None:
    raw = (
        ". a daughter of the picts. the morning sun was barely over the hills when "
        "mara left the croft with a basket on her arm and a purpose in her stride "
        "that had not troubled her sleep for many weeks."
    )
    result = strip_victorian_section_title(raw)
    assert result.stripped
    assert result.title == "a daughter of the picts"
    assert result.text.startswith("the morning sun was barely")


def test_strip_blanket_washing_title() -> None:
    raw = (
        ". the blanket washing. in a very genteel lodging-house, in the very genteel "
        "neighborhood of russell square, early in the afternoon of a dull autumn day, "
        "two ladies sat together over their tea."
    )
    result = strip_victorian_section_title(raw)
    assert result.stripped
    assert result.text.startswith("in a very genteel lodging-house")


def test_strip_scarlet_letter_editorial_errata() -> None:
    raw = (
        ". conclusion addendum book cover custom house plot summary understanding the "
        "scarlet letter revision statement the first major revision objective was to "
        "reduce the multitudinous use of commas. nathaniel hawthorne's the scarlet "
        "letter was downloaded from project gutenberg. the custom house in my native "
        "town of salem, at the head of what half a century ago in the days of old king "
        "derby was a bustling wharf, there stood a spacious building of brick."
    )
    result = strip_victorian_section_title(raw)
    assert result.stripped
    assert result.text.startswith("the custom house in my native town of salem")
    assert "revision statement" not in result.text.lower()


def test_strip_conclusion_section_title() -> None:
    raw = (
        ". conclusion. the awaking of the boys was of the most pleasant character. "
        "the sky had cleared and the sunlight penetrated between the branches from "
        "which the autumn leaves were falling slowly upon the mossy ground below."
    )
    result = strip_victorian_section_title(raw)
    assert result.stripped
    assert result.text.startswith("the awaking of the boys")


def test_strip_dickens_style_without_title_period() -> None:
    raw = (
        ". family affairs as the city clocks struck nine on monday morning, "
        "mrs clennam was wheeled by jeremiah flintwinch of the cut-down bedstead "
        "into her own room, where she sat all day in a wheeled chair."
    )
    result = strip_victorian_section_title(raw)
    assert result.stripped
    assert result.title == "family affairs"
    assert result.text.startswith("as the city clocks struck nine")


def test_no_strip_without_dot_prefix() -> None:
    raw = "The morning sun was barely over the hills when mara left the croft."
    result = strip_victorian_section_title(raw)
    assert not result.stripped
    assert result.text == raw


def test_strip_quoted_epigraph_title() -> None:
    raw = (
        '. in the twilight. "no words can be strong enough to reprehend your conduct, victor. '
        "you have acted disgracefully; you have broken every law of honor and decency in this house, "
        'and I can never forgive you for what you have done tonight."'
    )
    result = strip_victorian_section_title(raw)
    assert result.stripped
    assert result.title == "in the twilight"
    assert result.text.startswith('"no words can be strong')


def test_no_strip_when_body_too_short() -> None:
    raw = ". short title. only a few words here."
    result = strip_victorian_section_title(raw)
    assert not result.stripped
