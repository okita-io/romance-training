"""Tests for PDF page transcription quality checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "style_extraction"))

from page_quality import clean_model_output, validate_page_markdown


GOOD_PAGE = """## ENGLISH LANGUAGE SERIES

### General Editor:
Randolph Quirk

- Complex Words in English, Valerie Adams
- An Introduction to Modern English Word-Formation, Valerie Adams
"""

REASONING_PAGE = """We,,,,,, in,,,,,,, so I need to transcribe the page into markdown following the given rules.
Let's start by identifying the headings.
First, the heading: "Style in Fiction" is at the top, so #.
Then the first paragraph...
"""

REPEAT_PAGE = "We have 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19"


class TestPageQuality:
    def test_good_page_passes(self):
        report = validate_page_markdown(GOOD_PAGE)
        assert report.ok

    def test_reasoning_page_fails(self):
        report = validate_page_markdown(REASONING_PAGE)
        assert not report.ok
        assert any("reasoning" in i for i in report.issues)

    def test_repeat_page_fails(self):
        report = validate_page_markdown(REPEAT_PAGE)
        assert not report.ok

    def test_too_short_fails(self):
        report = validate_page_markdown("We")
        assert not report.ok
        assert any("too short" in i for i in report.issues)

    def test_clean_strips_meta_preamble(self):
        raw = REASONING_PAGE + "\n\n" + GOOD_PAGE
        cleaned = clean_model_output(raw)
        report = validate_page_markdown(cleaned)
        assert report.ok
