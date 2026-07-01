"""Tests for style benchmark fixture and delta scoring."""

from __future__ import annotations

from pathlib import Path

from tools.style_evaluation.benchmark import (
    COMPARE_FIELDS,
    aggregate_run,
    build_scene_prompt,
    build_setup_prompt,
    compare_runs,
    compute_delta,
    iter_benchmark_cases,
    load_fixture,
    style_target_lines,
)

FIXTURE = Path(__file__).resolve().parents[2] / "eval" / "style_benchmark" / "fixture.json"


def test_fixture_has_seven_plots_and_rosters() -> None:
    fx = load_fixture(FIXTURE)
    assert len(fx["female_leads"]) == 7
    assert len(fx["male_leads"]) == 7
    assert len(fx["plots"]) == 7
    assert len(fx["scene_types"]) == 3
    assert len(iter_benchmark_cases(fx)) == 21


def test_each_plot_has_full_style_target() -> None:
    fx = load_fixture(FIXTURE)
    for plot in fx["plots"]:
        target = plot["style_target"]
        for key in COMPARE_FIELDS:
            assert key in target, f"{plot['id']} missing {key}"


def test_scene_prompt_includes_style_and_characters() -> None:
    fx = load_fixture(FIXTURE)
    plot = fx["plots"][0]
    system, user = build_scene_prompt(fx, plot, "opening")
    assert "Leech" in system
    assert plot["female_lead"] in user
    assert plot["male_lead"] in user
    assert plot["style_label"] in user
    assert "lexical_complexity" in user


def test_setup_prompt_lists_all_plots() -> None:
    fx = load_fixture(FIXTURE)
    _, user = build_setup_prompt(fx)
    for plot in fx["plots"]:
        assert plot["title"] in user


def test_compute_delta_exact_match() -> None:
    target = {k: "x" for k in COMPARE_FIELDS}
    actual = dict(target)
    actual["tone"] = "wrong"
    delta = compute_delta(target, actual)
    assert delta["match_count"] == len(COMPARE_FIELDS) - 1
    assert delta["match_score"] < 1.0
    assert delta["fields"]["tone"]["match"] is False


def test_aggregate_and_compare_runs() -> None:
    base = [
        {
            "plot_id": "plot_01",
            "scene_type": "opening",
            "model": "base",
            "delta": {"match_score": 0.5, "fields": {}},
        }
    ]
    cand = [
        {
            "plot_id": "plot_01",
            "scene_type": "opening",
            "model": "ft",
            "delta": {"match_score": 0.75, "fields": {}},
        }
    ]
    summary = aggregate_run(base)
    assert summary["mean_match_score"] == 0.5

    report = compare_runs(base, cand)
    assert report["paired_count"] == 1
    assert report["mean_score_delta"] == 0.25
    assert report["improved"] == 1


def test_style_target_lines_cover_rubric_layers() -> None:
    fx = load_fixture(FIXTURE)
    lines = style_target_lines(fx["plots"][3]["style_target"])
    joined = "\n".join(lines)
    assert "figurative_density" in joined
    assert "Viewpoint" in joined
    assert "Textual dynamics" in joined
