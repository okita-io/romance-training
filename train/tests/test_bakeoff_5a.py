"""Unit tests for Phase 5A bake-off scoring (offline)."""

from __future__ import annotations

from tools.style_evaluation.bakeoff_5a import (
    GATE_FIELDS,
    score_prediction,
    summarize,
)


def test_score_prediction_agree_on_gate_match() -> None:
    gold = {k: "x" for k in GATE_FIELDS}
    gold.update({"tone": "lyrical", "pov": "first_person", "register": "colloquial",
                 "free_indirect_discourse": "heavy", "figurative_density": "high",
                 "cohesion": "tight"})
    pred = dict(gold)
    score = score_prediction(gold, pred, parse_ok=True)
    assert score["agree"] is True
    assert score["gate_exact"] is True
    assert score["parse_ok"] is True


def test_score_prediction_reject_on_tone_mismatch() -> None:
    gold = {
        "tone": "lyrical",
        "pov": "first_person",
        "register": "colloquial",
        "free_indirect_discourse": "heavy",
        "figurative_density": "high",
    }
    pred = dict(gold)
    pred["tone"] = "sardonic"
    score = score_prediction(gold, pred, parse_ok=True)
    assert score["agree"] is False
    assert score["gate_exact"] is False
    assert score["fields"]["tone"]["exact"] is False


def test_score_prediction_reject_on_parse_fail() -> None:
    gold = {
        "tone": "lyrical",
        "pov": "first_person",
        "register": "colloquial",
        "free_indirect_discourse": "heavy",
        "figurative_density": "high",
    }
    score = score_prediction(gold, gold, parse_ok=False)
    assert score["agree"] is False


def test_summarize_go_nogo() -> None:
    def _row(agree: bool, parse_ok: bool = True) -> dict:
        gold = {
            "tone": "lyrical",
            "pov": "first_person",
            "register": "colloquial",
            "free_indirect_discourse": "heavy",
            "figurative_density": "high",
        }
        pred = dict(gold)
        if not agree:
            pred["tone"] = "sardonic"
        score = score_prediction(gold, pred, parse_ok=parse_ok)
        return {
            "set": "gold",
            "model": "test-model",
            "score": score,
        }

    # All agree → go
    results = [_row(True) for _ in range(10)]
    report = summarize(results)
    assert report["go"] is True
    assert report["agree_rate"] == 1.0

    # Mostly reject → no-go
    results = [_row(False) for _ in range(10)]
    report = summarize(results)
    assert report["go"] is False
