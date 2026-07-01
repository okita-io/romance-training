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
    generate_run_roster,
    iter_benchmark_cases,
    load_fixture,
    materialize_fixture_for_run,
    style_target_lines,
)

FIXTURE = Path(__file__).resolve().parents[2] / "eval" / "style_benchmark" / "fixture.json"

# Stable roster for prompt assertions (reproducible across machines).
FIXED_ROSTER = (
    [
        "Aeliana Thornweave",
        "Mira Solacewright",
        "Vespera Kael",
        "Bronte Ashford",
        "Elowen Duskmire",
        "Sable Wynter",
        "Isolde Ravencrest",
    ],
    [
        "Cassian Vale",
        "Thorne Blackwell",
        "Aldric Fenn",
        "Ronan Greywing",
        "Darius Holt",
        "Evander Storm",
        "Lucien Merrow",
    ],
)


def _materialized(seed: str = "test-seed") -> dict:
    return materialize_fixture_for_run(
        load_fixture(FIXTURE),
        seed,
        fixed_roster=FIXED_ROSTER,
    )


def test_fixture_has_seven_plots_and_three_scenes() -> None:
    fx = _materialized()
    assert len(fx["plots"]) == 7
    assert len(fx["scene_types"]) == 3
    assert len(iter_benchmark_cases(fx)) == 21


def test_each_plot_has_full_style_target() -> None:
    fx = _materialized()
    for plot in fx["plots"]:
        target = plot["style_target"]
        for key in COMPARE_FIELDS:
            assert key in target, f"{plot['id']} missing {key}"


def test_run_roster_unique_per_seed() -> None:
    f1, m1 = generate_run_roster("seed-a")
    f2, m2 = generate_run_roster("seed-b")
    assert f1 != f2 or m1 != m2
    assert len(set(f1 + m1)) == 14
    f1b, m1b = generate_run_roster("seed-a")
    assert f1 == f1b and m1 == m1b


def test_scene_prompt_matches_romance_factory_shape() -> None:
    fx = _materialized()
    plot = fx["plots"][0]
    system, user = build_scene_prompt(fx, plot, "opening")
    assert "prose generation engine" in system
    assert "published romance novel" in system
    assert "Write act 10 of chapter 1" in user
    assert "## Rough-draft length budget" in user
    assert "## Verbosity contract" in user
    assert "## Narrative purpose contract" in user
    assert "## Style targets (Leech & Short" in user
    assert plot["female_lead"] in user
    assert plot["male_lead"] in user
    assert plot["style_label"] not in user  # RF uses rubric block, not style_label string


def test_setup_prompt_lists_all_plots() -> None:
    fx = _materialized()
    _, user = build_setup_prompt(fx)
    for plot in fx["plots"]:
        assert plot["title"] in user
        assert plot["female_lead"] in user


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
    fx = _materialized()
    lines = style_target_lines(fx["plots"][3]["style_target"])
    joined = "\n".join(lines)
    assert "Figurative Density" in joined
    assert "POV" in joined
