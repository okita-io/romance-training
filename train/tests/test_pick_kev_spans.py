"""Tests for Kev-span sampling and sentence-bounded windows."""

from __future__ import annotations

import json
import random
from pathlib import Path

from tools.data_preparation.pick_kev_spans import pick_spans, to_span_record
from tools.style_classification.chunk_text import KEV_STATE_CHARS, random_kev_span, split_sentences


LONG = (
    "It was a dark and stormy night. The rain fell in torrents. "
    "Valancy wakened early, in the lifeless hour before dawn. "
    "She had not slept very well. One does not sleep well, sometimes, "
    "when one is twenty-nine on the morrow. "
    "But of course appearances should be kept up. "
    "The tears came into her eyes as she lay there alone. "
) * 12


def test_short_text_not_truncated() -> None:
    text = "She opened the door and waited."
    span, truncated = random_kev_span(text, random.Random(1), max_chars=1400)
    assert span == text
    assert truncated is False


def test_long_text_fits_budget_and_sentence_boundary() -> None:
    span, truncated = random_kev_span(LONG, random.Random(3), max_chars=400, min_words=20)
    assert truncated
    assert len(span) <= 400
    assert span.endswith((".", "!", "?", "…"))


def test_to_span_record_strips_old_labels() -> None:
    rec = {
        "text": LONG,
        "metadata": {
            "title": "Demo",
            "author": "X",
            "style_profile": {"tone": "melancholic"},
            "style_council": {"tone": {"value": "melancholic", "consensus": True}},
        },
    }
    span = to_span_record(
        rec,
        slug="gutenberg_fiction",
        source_file="gutenberg_fiction_styled_seg_000.jsonl",
        rng=random.Random(0),
        max_chars=500,
        min_words=20,
    )
    assert span is not None
    assert "style_profile" not in span["metadata"]
    assert "style_council" not in span["metadata"]
    assert span["metadata"]["truncated_to_kev"] is True
    assert span["metadata"]["span_chars"] <= 500
    assert span["metadata"]["source_corpus"] == "gutenberg_fiction"


def test_pick_spans_stratifies(tmp_path: Path) -> None:
    a = tmp_path / "gutenberg_fiction_styled_seg_000.jsonl"
    b = tmp_path / "horror_novel_chunks_styled_seg_000.jsonl"
    row = {"text": LONG, "metadata": {"title": "T"}}
    a.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    b.write_text(json.dumps(row) + "\n", encoding="utf-8")
    spans, stats = pick_spans(
        [a, b], n=2, seed=1, max_chars=KEV_STATE_CHARS, min_words=20
    )
    assert len(spans) == 2
    assert stats["kept"] == 2
    assert all(len(s["text"]) <= KEV_STATE_CHARS for s in spans)
