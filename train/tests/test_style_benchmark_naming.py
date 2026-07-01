"""Tests for RF-style benchmark character naming and retry telemetry."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.style_evaluation.benchmark import load_fixture, materialize_fixture_for_run
from tools.style_evaluation.character_namer import (
    NameValidation,
    generate_llm_roster,
    load_overused_names,
    validate_personal_name,
)

FIXTURE = Path(__file__).resolve().parents[2] / "eval" / "style_benchmark" / "fixture.json"


def test_overused_list_includes_isolde_and_elara() -> None:
    first, _ = load_overused_names()
    assert "elara" in first
    assert "isolde" in first
    assert "isolide" in first


def test_validate_rejects_elara() -> None:
    first, last = load_overused_names()
    check = validate_personal_name(
        "Elara Thornweave",
        reserved=set(),
        overused_first=first,
        overused_last=last,
    )
    assert not check.ok
    assert check.code == "overused_first"


def test_validate_accepts_uncommon_name() -> None:
    first, last = load_overused_names()
    check = validate_personal_name(
        "Phaedra Saltwick",
        reserved=set(),
        overused_first=first,
        overused_last=last,
    )
    assert check.ok


def test_llm_namer_retries_until_valid() -> None:
    fx = load_fixture(FIXTURE)
    fx = {**fx, "plots": fx["plots"][:1]}
    female_calls = 0
    male_calls = 0

    def fake_complete(user: str, *, system: str = "", model: str = "", **_: object) -> str:
        nonlocal female_calls, male_calls
        if '"story_role": "main_character"' in user:
            female_calls += 1
            if female_calls == 1:
                return '{"name": "Elara Vance", "character_title": ""}'
            return '{"name": "Phaedra Saltwick", "character_title": ""}'
        male_calls += 1
        return '{"name": "Torven Saltmark", "character_title": ""}'

    females, males, report = generate_llm_roster(
        fx,
        name_seed="retry-test",
        model="test-model",
        max_retries=100,
        complete_fn=fake_complete,
    )
    assert females == ["Phaedra Saltwick"]
    assert males == ["Torven Saltmark"]
    assert report.characters[0].attempts == 2
    assert len(report.characters[0].rejections) == 1
    assert report.characters[0].rejections[0].code == "overused_first"
    assert report.total_attempts == 3


def test_llm_namer_exhaust_raises() -> None:
    fx = load_fixture(FIXTURE)

    def always_elara(_user: str, *, system: str = "", model: str = "", **_: object) -> str:
        return '{"name": "Elara Vance", "character_title": ""}'

    with pytest.raises(RuntimeError, match="Could not name all benchmark leads"):
        generate_llm_roster(
            {"plots": fx["plots"][:1], "world": fx["world"]},
            name_seed="fail-test",
            model="test-model",
            max_retries=5,
            complete_fn=always_elara,
        )


def test_materialize_fixed_roster_for_tests() -> None:
    fx = load_fixture(FIXTURE)
    fx["plots"] = fx["plots"][:1]
    fixed = (["A Test"], ["B Test"])
    out = materialize_fixture_for_run(fx, "seed", fixed_roster=fixed)
    assert out["plots"][0]["female_lead"] == "A Test"
    assert out["plots"][0]["male_lead"] == "B Test"
    assert out["naming"]["mode"] == "fixed"
