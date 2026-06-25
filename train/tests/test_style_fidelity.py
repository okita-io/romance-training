"""Tests for sentence-aware chunking and style knowledge retrieval."""

from __future__ import annotations

import pytest

from tools.style_classification.chunk_text import chunk_by_sentences, split_sentences
from tools.style_classification.style_knowledge import (
    build_classification_context,
    format_context,
    retrieve,
    rubric_dimension_summary,
)


SAMPLE_PROSE = (
    "It was a dark and stormy night. The rain fell in torrents. "
    "Valancy wakened early, in the lifeless hour before dawn. "
    "She had not slept very well. One does not sleep well, sometimes, "
    "when one is twenty-nine on the morrow. "
    "But of course appearances should be kept up. "
    "The tears came into her eyes as she lay there alone."
)


class TestSplitSentences:
    def test_splits_on_period(self):
        sents = split_sentences("First sentence. Second sentence.")
        assert len(sents) == 2
        assert sents[0].startswith("First")

    def test_single_fragment_returns_one(self):
        assert split_sentences("no terminal punctuation here") == ["no terminal punctuation here"]


class TestChunkBySentences:
    def test_short_text_unchanged(self):
        chunks = chunk_by_sentences(SAMPLE_PROSE, target_words=500)
        assert len(chunks) == 1
        assert chunks[0] == SAMPLE_PROSE

    def test_boundaries_are_complete_sentences(self):
        long_text = " ".join([SAMPLE_PROSE] * 20)
        chunks = chunk_by_sentences(long_text, target_words=80, overlap_sentences=1)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.endswith((".", "!", "?")), f"chunk does not end on sentence boundary: {chunk[-40:]!r}"

    def test_overlap_shares_sentences(self):
        long_text = " ".join([SAMPLE_PROSE] * 15)
        chunks = chunk_by_sentences(long_text, target_words=60, overlap_sentences=2)
        if len(chunks) < 2:
            pytest.skip("not enough chunks")
        first_sents = split_sentences(chunks[0])
        second_sents = split_sentences(chunks[1])
        assert first_sents[-1] in second_sents or first_sents[-2] in second_sents


class TestStyleKnowledge:
    def test_retrieve_scores_category_match(self, tmp_path: Path):
        kb = [
            {
                "id": "a",
                "title": "Lexical density in fiction",
                "text": "## Lexical density\n\nContent words versus function words.",
                "categories": ["lexical"],
            },
            {
                "id": "b",
                "title": "Unrelated chapter",
                "text": "## Other\n\nSomething about cooking.",
                "categories": ["general"],
            },
        ]
        hits = retrieve(
            "lexical density content words",
            categories=["lexical"],
            dimension_id="lexical_density",
            knowledge=kb,
        )
        assert hits and hits[0]["id"] == "a"

    def test_format_context_respects_max_chars(self):
        chunks = [{"title": "T", "text": "x" * 5000}]
        out = format_context(chunks, max_chars=200)
        assert len(out) <= 210

    def test_rubric_dimension_summary(self):
        rubric = {
            "dimensions": [
                {
                    "id": "register",
                    "name": "Register",
                    "definition": "Situation-appropriate language variety.",
                    "values": ["formal_literary", "colloquial"],
                    "scoring": {"low": "colloquial", "mid": "neutral", "high": "formal"},
                }
            ]
        }
        summary = rubric_dimension_summary(rubric, ["register"])
        assert "register" in summary
        assert "colloquial" in summary

    def test_build_classification_context_with_rubric(self):
        rubric = {
            "dimensions": [
                {
                    "id": "tone",
                    "name": "Tone",
                    "definition": "Affective register.",
                    "values": ["neutral", "lyrical"],
                    "scoring": {},
                }
            ]
        }
        ctx = build_classification_context("She wept silently.", rubric=rubric, knowledge_k=0)
        assert "tone" in ctx.lower()
