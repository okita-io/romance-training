"""Tests for Kev System One conversion of style-council labels."""

from __future__ import annotations

import json
from pathlib import Path

from tools.training_formats.convert_kev_style import (
    CHOICE_FIELDS,
    extract_consensus_labels,
    metric_specs,
    question_bank,
    to_kev_record,
    truncate_state,
)


ALICE = (
    "Alice opened the door and found that it led into a small passage, "
    "not much larger than a rat-hole: she knelt down and looked along the passage "
    "into the loveliest garden you ever saw."
)


def _council_record(**fields: dict) -> dict:
    profile = {k: v["value"] for k, v in fields.items()}
    return {
        "text": ALICE,
        "metadata": {
            "title": "Alice's Adventures in Wonderland",
            "author": "Carroll, Lewis",
            "style_profile": profile,
            "style_council": fields,
        },
    }


def test_metric_specs_cover_llm_fields() -> None:
    specs = metric_specs()
    assert "register" in specs
    assert "climax" in specs
    assert "tone" in specs
    assert specs["register"]["metric_type"] == "categorical"
    assert "third_limited" in specs["pov"]["values"]


def test_question_bank_choice_vs_score() -> None:
    bank = question_bank()
    assert bank["register"]["type"] == "choice"
    assert "neutral_narrative" in bank["register"]["criteria"]
    assert bank["figurative_density"]["type"] == "score"
    assert isinstance(bank["figurative_density"]["criteria"], list)
    assert bank["tone"]["type"] == "choice"
    for field in CHOICE_FIELDS:
        assert bank[field]["type"] == "choice"
    # scoring.low/high for narrative_distance names the opposite label; do not invert.
    assert bank["narrative_distance"]["criteria"][0].startswith("intimate")
    assert "distant" not in bank["narrative_distance"]["criteria"][0]
    assert bank["figurative_density"]["criteria"][0].startswith("low")


def test_extract_consensus_skips_disagreement() -> None:
    specs = metric_specs()
    record = _council_record(
        register={
            "value": "neutral_narrative",
            "agree_n": 3,
            "consensus": True,
        },
        tone={
            "value": "melancholic",
            "agree_n": 1,
            "consensus": False,
        },
        climax={
            "value": "not_a_label",
            "agree_n": 3,
            "consensus": True,
        },
    )
    labels = extract_consensus_labels(record, specs)
    assert set(labels) == {"register"}
    assert labels["register"][0] == "neutral_narrative"


def test_to_kev_record_score_index_and_choice_name() -> None:
    specs = metric_specs()
    bank = question_bank(specs)
    record = _council_record(
        register={
            "value": "colloquial",
            "agree_n": 2,
            "consensus": True,
        },
        figurative_density={
            "value": "high",
            "agree_n": 3,
            "consensus": True,
        },
    )
    labels = extract_consensus_labels(record, specs)
    converted = to_kev_record(record, labels, specs, bank, src="council")
    assert converted is not None
    assert converted["questions"]["register"]["label"] == "colloquial"
    assert converted["questions"]["figurative_density"]["label"] == 2
    assert converted["_meta"]["n_questions"] == 2


def test_truncate_state_prefers_sentence_boundary() -> None:
    text = "First sentence is short. " + ("Later words keep going. " * 80)
    clipped, truncated = truncate_state(text, max_chars=80)
    assert truncated
    assert len(clipped) <= 80
    assert clipped.startswith("First sentence is short.")
    assert clipped.endswith(".")


def test_convert_cli_writes_council_jsonl(tmp_path: Path) -> None:
    from tools.training_formats.convert_kev_style import main

    src = tmp_path / "council_out.jsonl"
    record = _council_record(
        pov={"value": "third_limited", "agree_n": 3, "consensus": True},
        cohesion={"value": "tight", "agree_n": 2, "consensus": True},
    )
    src.write_text(json.dumps(record) + "\n", encoding="utf-8")
    out = tmp_path / "kev"
    assert main(["--council", str(src), "--out-dir", str(out)]) == 0
    rows = [json.loads(line) for line in (out / "council.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["questions"]["pov"]["label"] == "third_limited"
    assert rows[0]["questions"]["cohesion"]["label"] == 2
    workload = json.loads((out / "workload.json").read_text())
    assert "register" in workload["questions"]
    assert "figurative_density" in workload["questions"]
