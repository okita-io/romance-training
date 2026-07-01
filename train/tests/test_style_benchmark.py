"""Tests for style benchmark fixture and delta scoring."""

from __future__ import annotations

from pathlib import Path

from tools.style_evaluation.benchmark import (
    COMPARE_FIELDS,
    aggregate_run,
    build_scene_prompt,
    build_setup_prompt,
    compare_runs,
    compare_training_sessions,
    compute_delta,
    format_sessions_report,
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


def _bench_record(
    plot_id: str,
    scene_type: str,
    *,
    match_score: float,
    label: str,
    model: str,
    field_matches: dict[str, bool] | None = None,
) -> dict:
    fields = {}
    for key in COMPARE_FIELDS:
        matched = (field_matches or {}).get(key, match_score >= 0.5)
        fields[key] = {"target": "t", "actual": "t" if matched else "x", "match": matched}
    return {
        "plot_id": plot_id,
        "scene_type": scene_type,
        "label": label,
        "model": model,
        "delta": {"match_score": match_score, "fields": fields},
    }


def test_compare_training_sessions_trend() -> None:
    cases = [
        ("plot_01", "opening"),
        ("plot_01", "climax_reveal"),
        ("plot_02", "opening"),
    ]
    baseline = [
        _bench_record(pid, sid, match_score=0.4, label="baseline", model="base")
        for pid, sid in cases
    ]
    mid = [
        _bench_record(pid, sid, match_score=0.55, label="batch_001", model="ft1")
        for pid, sid in cases
    ]
    final = [
        _bench_record(
            pid,
            sid,
            match_score=0.7,
            label="batch_002",
            model="ft2",
            field_matches={"register": True, "pov": True, "tone": True},
        )
        for pid, sid in cases
    ]

    report = compare_training_sessions(
        [("baseline", baseline), ("batch_001", mid), ("batch_002", final)]
    )
    trend = report["conformity_trend"]
    assert trend["direction"] == "improved"
    assert trend["first_to_last_delta"] == 0.3
    assert len(trend["steps"]) == 2
    assert trend["steps"][0]["mean_score_delta"] == 0.15
    assert len(report["vs_baseline"]) == 2
    assert report["vs_baseline"][-1]["vs_baseline_delta"] == 0.3
    assert "register" in report["field_trends"]
    assert report["field_trends"]["register"]["direction"] == "improved"

    text = format_sessions_report(report)
    assert "batch_002" in text
    assert "improved" in text.lower()


def test_compare_runs_field_hit_rate_delta() -> None:
    base = [
        _bench_record(
            "plot_01",
            "opening",
            match_score=0.25,
            label="b",
            model="base",
            field_matches={"register": False, "pov": False},
        )
    ]
    cand = [
        _bench_record(
            "plot_01",
            "opening",
            match_score=0.75,
            label="c",
            model="ft",
            field_matches={k: True for k in COMPARE_FIELDS},
        )
    ]
    report = compare_runs(base, cand)
    assert "field_hit_rate_delta" in report
    assert report["field_hit_rate_delta"]["register"]["delta"] > 0
