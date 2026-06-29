"""Tests for web-fiction quality heuristics."""

from tools.data_preparation.fiction_quality import classify_fiction_quality, strip_html


def test_strip_html():
    raw = "<i>All characters are 18.</i> She walked into the room."
    assert "All characters" in strip_html(raw)
    assert "<i>" not in strip_html(raw)


def test_keep_narrative_story():
    parts = [
        "The morning air smelled of rain and old books as Clara crossed the courtyard.",
        "She paused at the fountain, listening to sparrows quarrel in the ivy.",
        "Her companion waited by the gate, tapping an impatient rhythm against the stone.",
        "Neither of them spoke about the letter, though both had read it twice.",
        "Instead they discussed the harvest, the price of grain, and the mayor's latest decree.",
        "A cart rattled past, laden with apples, wool, and gossip from the neighboring village.",
        "Clara adjusted her shawl and wondered whether the road would ever feel familiar again.",
    ]
    text = " ".join(parts * 30)
    q = classify_fiction_quality(text)
    assert q.tier in ("keep", "review")
    assert q.language == "en"
    assert "too_short" not in q.reasons


def test_drop_too_short():
    q = classify_fiction_quality("A short note with barely any words here.")
    assert q.tier == "drop"
    assert "too_short" in q.reasons


def test_review_repetitive():
    unit = "he moaned softly as she moved closer and whispered his name again "
    text = unit * 80
    q = classify_fiction_quality(text)
    assert q.tier in ("review", "drop")
    assert "repetitive" in q.reasons or "low_vocab" in q.reasons
