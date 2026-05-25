"""Unit tests for ManuscriptLoader."""

from __future__ import annotations

import os
import re
import tempfile

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from romance_factory.generate.phrase_detection.manuscript_loader import (
    ManuscriptLoader,
)
from romance_factory.generate.phrase_detection.models import ChapterSegment


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _write_tmp(content: str) -> str:
    """Write *content* to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


# ------------------------------------------------------------------
# FileNotFoundError / IOError
# ------------------------------------------------------------------

class TestManuscriptLoaderErrors:
    def test_file_not_found(self):
        loader = ManuscriptLoader()
        with pytest.raises(FileNotFoundError, match="not_a_real_file"):
            loader.load("/tmp/not_a_real_file.txt")

    def test_unreadable_file(self, tmp_path):
        p = tmp_path / "locked.txt"
        p.write_text("hello")
        p.chmod(0o000)
        loader = ManuscriptLoader()
        try:
            with pytest.raises(IOError):
                loader.load(str(p))
        finally:
            p.chmod(0o644)


# ------------------------------------------------------------------
# No-heading fallback
# ------------------------------------------------------------------

class TestNoHeadingFallback:
    def test_no_headings_single_segment(self):
        text = "Some prose without any chapter headings.\nAnother line."
        path = _write_tmp(text)
        try:
            segments = ManuscriptLoader().load(path)
            assert len(segments) == 1
            seg = segments[0]
            assert seg.chapter_index == 0
            assert seg.title == ""
            assert seg.text == text
            assert seg.global_char_offset == 0
        finally:
            os.unlink(path)

    def test_empty_manuscript(self):
        path = _write_tmp("")
        try:
            segments = ManuscriptLoader().load(path)
            assert len(segments) == 1
            assert segments[0].text == ""
            assert segments[0].title == ""
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# Chapter splitting
# ------------------------------------------------------------------

class TestChapterSplitting:
    def test_two_chapters(self):
        text = "Chapter 1\nFirst chapter body.\nChapter 2\nSecond chapter body.\n"
        path = _write_tmp(text)
        try:
            segments = ManuscriptLoader().load(path)
            assert len(segments) == 2
            assert segments[0].title == "Chapter 1"
            assert "First chapter body." in segments[0].text
            assert segments[1].title == "Chapter 2"
            assert "Second chapter body." in segments[1].text
        finally:
            os.unlink(path)

    def test_case_insensitive_headings(self):
        text = "CHAPTER 1\nBody one.\nchapter 2\nBody two.\n"
        path = _write_tmp(text)
        try:
            segments = ManuscriptLoader().load(path)
            assert len(segments) == 2
        finally:
            os.unlink(path)

    def test_text_before_first_heading(self):
        text = "Prologue text.\nChapter 1\nBody.\n"
        path = _write_tmp(text)
        try:
            segments = ManuscriptLoader().load(path)
            assert len(segments) == 2
            assert segments[0].title == ""
            assert "Prologue" in segments[0].text
            assert segments[1].title == "Chapter 1"
        finally:
            os.unlink(path)

    def test_custom_pattern(self):
        text = "Part I\nBody one.\nPart II\nBody two.\n"
        loader = ManuscriptLoader(chapter_pattern=r"(?i)^Part\s+[IVX]+")
        path = _write_tmp(text)
        try:
            segments = loader.load(path)
            assert len(segments) == 2
            assert segments[0].title == "Part I"
            assert segments[1].title == "Part II"
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# Round-trip property (example-based)
# ------------------------------------------------------------------

class TestRoundTrip:
    def test_round_trip_no_headings(self):
        text = "Just some text.\nMore text here."
        path = _write_tmp(text)
        try:
            segments = ManuscriptLoader().load(path)
            reconstructed = "".join(seg.title + seg.text for seg in segments)
            assert reconstructed == text
        finally:
            os.unlink(path)

    def test_round_trip_with_headings(self):
        text = "Chapter 1\nBody one.\nChapter 2\nBody two.\n"
        path = _write_tmp(text)
        try:
            segments = ManuscriptLoader().load(path)
            reconstructed = "".join(seg.title + seg.text for seg in segments)
            assert reconstructed == text
        finally:
            os.unlink(path)

    def test_round_trip_with_prologue(self):
        text = "Prologue.\nChapter 1\nBody.\n"
        path = _write_tmp(text)
        try:
            segments = ManuscriptLoader().load(path)
            reconstructed = "".join(seg.title + seg.text for seg in segments)
            assert reconstructed == text
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# global_char_offset correctness
# ------------------------------------------------------------------

class TestGlobalCharOffset:
    def test_offsets_point_to_segment_start(self):
        text = "Chapter 1\nBody one.\nChapter 2\nBody two.\n"
        path = _write_tmp(text)
        try:
            segments = ManuscriptLoader().load(path)
            for seg in segments:
                # The segment's title+text should start at global_char_offset
                combined = seg.title + seg.text
                actual = text[seg.global_char_offset:seg.global_char_offset + len(combined)]
                assert actual == combined, (
                    f"Segment {seg.chapter_index}: expected {combined!r} "
                    f"at offset {seg.global_char_offset}, got {actual!r}"
                )
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# Hypothesis strategies for Property 1
# ------------------------------------------------------------------

# Prose text that does NOT accidentally match "Chapter N" at line start
_SAFE_PROSE = st.text(
    alphabet=st.sampled_from(
        "abcdefghijklmnopqrstuvwxyz "
        "ABDEFGIJKLMNOPQRSTUVWXYZ"  # no 'C' or 'H' to avoid "Chapter"
        "0123456789.,;:!?'-\n"
    ),
    min_size=0,
    max_size=120,
).filter(
    lambda t: not re.search(r"(?i)^chapter\s+\d+", t, re.MULTILINE)
)

# A chapter heading line like "Chapter 3" or "CHAPTER 12"
_CHAPTER_HEADING = st.integers(min_value=1, max_value=99).map(
    lambda n: f"Chapter {n}"
)


@st.composite
def manuscript_with_headings(draw):
    """Generate a manuscript string with zero or more chapter headings.

    Returns a tuple of (manuscript_text, num_headings).
    The text is guaranteed to have each heading on its own line start.
    """
    num_headings = draw(st.integers(min_value=0, max_value=5))

    parts: list[str] = []
    # Optionally add prose before the first heading
    pre_text = draw(_SAFE_PROSE)
    if pre_text:
        # Ensure it ends with a newline so the heading starts on a new line
        if not pre_text.endswith("\n"):
            pre_text += "\n"
        parts.append(pre_text)

    for i in range(num_headings):
        heading = draw(_CHAPTER_HEADING)
        parts.append(heading)
        # Body text after the heading (always starts with \n)
        body = draw(_SAFE_PROSE)
        body_with_newline = "\n" + body
        # Ensure body ends with \n if there's another heading coming
        if i < num_headings - 1 and not body_with_newline.endswith("\n"):
            body_with_newline += "\n"
        parts.append(body_with_newline)

    manuscript = "".join(parts)

    # If manuscript is empty and no headings, that's fine (empty manuscript case)
    return manuscript, num_headings


# ------------------------------------------------------------------
# Property 1: Chapter Splitting Round-Trip
# Feature: repeated-phrase-detection, Property 1: Chapter Splitting Round-Trip
# ------------------------------------------------------------------

class TestChapterSplittingRoundTripProperty:
    """**Validates: Requirements 1.1, 1.2, 1.3, 1.5**"""

    @given(data=manuscript_with_headings())
    @settings(max_examples=100)
    def test_round_trip_reconstruction(self, data):
        """For any manuscript text with zero or more chapter headings,
        splitting and re-concatenating SHALL reproduce the original text exactly.

        Feature: repeated-phrase-detection, Property 1: Chapter Splitting Round-Trip
        """
        manuscript_text, num_headings = data

        # Write manuscript to a temp file
        path = _write_tmp(manuscript_text)
        try:
            loader = ManuscriptLoader()
            segments = loader.load(path)

            # Property 1a: Round-trip reconstruction
            reconstructed = "".join(seg.title + seg.text for seg in segments)
            assert reconstructed == manuscript_text, (
                f"Round-trip failed.\n"
                f"Original ({len(manuscript_text)} chars): {manuscript_text!r}\n"
                f"Reconstructed ({len(reconstructed)} chars): {reconstructed!r}\n"
                f"Segments: {[(s.title, s.text) for s in segments]}"
            )
        finally:
            os.unlink(path)

    @given(data=manuscript_with_headings())
    @settings(max_examples=100)
    def test_segment_count(self, data):
        """Number of segments SHALL equal number of headings found (or 1 if none),
        plus 1 if there's text before the first heading.

        Feature: repeated-phrase-detection, Property 1: Chapter Splitting Round-Trip
        """
        manuscript_text, num_headings = data

        path = _write_tmp(manuscript_text)
        try:
            loader = ManuscriptLoader()
            segments = loader.load(path)

            if num_headings == 0:
                # No headings → single segment
                assert len(segments) == 1, (
                    f"Expected 1 segment for no headings, got {len(segments)}"
                )
            else:
                # Count headings actually found in the text
                pattern = re.compile(r"(?i)^chapter\s+\d+", re.MULTILINE)
                actual_headings = len(pattern.findall(manuscript_text))

                # Check if there's text before the first heading
                first_match = pattern.search(manuscript_text)
                has_pre_text = first_match is not None and first_match.start() > 0

                expected_segments = actual_headings + (1 if has_pre_text else 0)
                assert len(segments) == expected_segments, (
                    f"Expected {expected_segments} segments "
                    f"({actual_headings} headings + {'pre-text' if has_pre_text else 'no pre-text'}), "
                    f"got {len(segments)}.\nText: {manuscript_text!r}"
                )
        finally:
            os.unlink(path)

    @given(data=manuscript_with_headings())
    @settings(max_examples=100)
    def test_global_char_offset_correctness(self, data):
        """global_char_offset correctness: for each segment,
        original_text[seg.global_char_offset:seg.global_char_offset + len(seg.title + seg.text)]
        SHALL equal seg.title + seg.text.

        Feature: repeated-phrase-detection, Property 1: Chapter Splitting Round-Trip
        """
        manuscript_text, num_headings = data

        path = _write_tmp(manuscript_text)
        try:
            loader = ManuscriptLoader()
            segments = loader.load(path)

            for seg in segments:
                combined = seg.title + seg.text
                end = seg.global_char_offset + len(combined)
                actual = manuscript_text[seg.global_char_offset:end]
                assert actual == combined, (
                    f"Segment {seg.chapter_index}: offset {seg.global_char_offset} "
                    f"expected {combined!r}, got {actual!r}"
                )
        finally:
            os.unlink(path)
