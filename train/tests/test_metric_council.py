"""Offline unit tests for the single-metric teacher council (no live LLM)."""

from __future__ import annotations

from unittest.mock import patch

from tools.style_classification import metric_council as mc


# ---------------------------------------------------------------------------
# resolve_metric / build_judge_prompts
# ---------------------------------------------------------------------------


def test_resolve_metric_known_fields() -> None:
    reg = mc.resolve_metric("register")
    assert reg is not None
    assert reg["id"] == "register"
    assert "neutral_narrative" in reg["values"]

    # Textual principle (not an llm_dimension) should also resolve.
    climax = mc.resolve_metric("climax")
    assert climax is not None
    assert "climactic" in climax["values"]


def test_resolve_metric_unknown() -> None:
    assert mc.resolve_metric("not_a_real_metric") is None


def test_build_judge_prompts_single_key_only() -> None:
    metric = mc.resolve_metric("register")
    system, user = mc.build_judge_prompts(metric, "Some prose.", variant="definition")

    # Only the target metric key appears in the schema, no other dimensions.
    assert '"register"' in user
    assert "mind_style" not in user
    assert "tone" not in user
    # All allowed labels are offered.
    for value in metric["values"]:
        assert value in user
    assert "Some prose." in user
    assert isinstance(system, str) and system


def test_pov_and_fid_rubric_have_detailed_guides() -> None:
    from tools.style_classification.classify_passage import load_rubric

    rubric = load_rubric()
    by_id = {d["id"]: d for d in rubric["dimensions"]}
    for mid in ("pov", "free_indirect_discourse"):
        dim = by_id[mid]
        assert len(dim["definition"]) > 80
        assert dim.get("analysis_prompt")
        scoring = dim.get("scoring") or {}
        for value in dim["values"]:
            assert value in scoring, f"{mid} missing scoring gloss for {value}"


def test_build_judge_prompts_includes_fid_label_guide() -> None:
    from tools.style_classification.classify_passage import load_rubric

    rubric = load_rubric()
    metric = mc.resolve_metric("free_indirect_discourse", rubric=rubric)
    assert metric is not None
    _, user = mc.build_judge_prompts(
        metric,
        "She looked away. Was he lying again?",
        variant="definition",
    )
    assert "Label guide" in user
    assert "WITHOUT a reporting clause" in user or "reporting" in user.lower()
    assert "heavy" in user


def test_build_arbitrator_prompts_include_pov_guide() -> None:
    from tools.style_classification.classify_passage import load_rubric

    rubric = load_rubric()
    metric = mc.resolve_metric("pov", rubric=rubric)
    votes = [
        {"variant": "definition", "label": "first_person", "evidence": "I walked"},
        {"variant": "evidence", "label": "third_limited", "evidence": "she felt"},
        {"variant": "contrast", "label": "first_person", "evidence": "my hand"},
    ]
    _, user = mc.build_arbitrator_prompts(metric, "I walked on. She felt cold.", votes)
    assert "Label guide" in user
    assert "third_limited" in user
    assert "Judge [definition]" in user


def test_build_judge_prompts_variants_differ() -> None:
    metric = mc.resolve_metric("tone")
    systems = {
        v: mc.build_judge_prompts(metric, "x", variant=v)[0]
        for v in mc.COUNCIL_VARIANTS
    }
    # Three distinct framings.
    assert len(set(systems.values())) == 3


# ---------------------------------------------------------------------------
# council_vote
# ---------------------------------------------------------------------------


def _vote(label, parse_ok=True):
    return {"variant": "x", "label": label, "parse_ok": parse_ok}


def test_council_vote_majority() -> None:
    result = mc.council_vote([_vote("lyrical"), _vote("lyrical"), _vote("tense")])
    assert result["value"] == "lyrical"
    assert result["agree_n"] == 2
    assert result["consensus"] is True


def test_council_vote_unanimous() -> None:
    result = mc.council_vote([_vote("tight"), _vote("tight"), _vote("tight")])
    assert result["value"] == "tight"
    assert result["agree_n"] == 3


def test_council_vote_three_way_tie_no_consensus() -> None:
    result = mc.council_vote([_vote("a"), _vote("b"), _vote("c")])
    assert result["value"] is None
    assert result["consensus"] is False


def test_council_vote_single_valid_no_majority() -> None:
    # Only one valid label (< 2 agreeing) -> no consensus.
    result = mc.council_vote([_vote("lyrical"), _vote(None), _vote(None)])
    assert result["value"] is None
    assert result["consensus"] is False


def test_council_vote_two_valid_agree() -> None:
    result = mc.council_vote([_vote("lyrical"), _vote("lyrical"), _vote(None)])
    assert result["value"] == "lyrical"
    assert result["agree_n"] == 2


def test_council_vote_parse_ok_propagates() -> None:
    result = mc.council_vote([_vote(None, parse_ok=False), _vote(None, parse_ok=True), _vote(None, parse_ok=False)])
    assert result["parse_ok"] is True
    assert result["value"] is None


# ---------------------------------------------------------------------------
# judge_once — label validation
# ---------------------------------------------------------------------------


def test_judge_once_valid_label() -> None:
    metric = mc.resolve_metric("cohesion")
    with patch.object(mc, "llm_complete", return_value='{"cohesion": "tight", "evidence": "and then"}'):
        vote = mc.judge_once(metric, "prose", variant="definition")
    assert vote["label"] == "tight"
    assert vote["parse_ok"] is True
    assert vote["evidence"] == "and then"


def test_judge_once_invalid_label_rejected() -> None:
    metric = mc.resolve_metric("cohesion")
    with patch.object(mc, "llm_complete", return_value='{"cohesion": "super-glue"}'):
        vote = mc.judge_once(metric, "prose", variant="definition")
    assert vote["label"] is None
    assert vote["parse_ok"] is True
    assert vote["error"] == "label_not_allowed"


def test_judge_once_bad_json() -> None:
    metric = mc.resolve_metric("cohesion")
    with patch.object(mc, "llm_complete", return_value="not json at all"):
        vote = mc.judge_once(metric, "prose", variant="definition")
    assert vote["label"] is None
    assert vote["parse_ok"] is False
    assert vote["error"] == "json_parse_failed"


# ---------------------------------------------------------------------------
# assess_fields_council
# ---------------------------------------------------------------------------


def test_assess_fields_council_majority(monkeypatch) -> None:
    calls: list[str] = []

    def fake_judge(metric, text, *, variant, model=mc.DEFAULT_MODEL, knowledge=""):
        calls.append(f"{metric['id']}:{variant}")
        # register -> majority "colloquial"; tone -> 3-way tie (no consensus)
        table = {
            "register": {"definition": "colloquial", "evidence": "colloquial", "contrast": "archaic"},
            "tone": {"definition": "lyrical", "evidence": "tense", "contrast": "comedic"},
        }
        label = table[metric["id"]][variant]
        return {"variant": variant, "label": label, "evidence": "e", "parse_ok": True}

    monkeypatch.setattr(mc, "judge_once", fake_judge)
    # arbitrate_split=False -> pure majority, no arbitrator/network for the tie.
    profile, meta = mc.assess_fields_council(
        "passage text", ["register", "tone"], arbitrate_split=False
    )

    assert profile == {"register": "colloquial"}  # tone omitted (no consensus)
    assert meta["register"]["consensus"] is True
    assert meta["register"]["method"] == "majority"
    assert meta["tone"]["consensus"] is False
    # 3 variants x 2 fields = 6 judge calls
    assert len(calls) == 6


def test_assess_fields_council_reuses_prior(monkeypatch) -> None:
    def fail_judge(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("judge_once should not run for prior-supplied field")

    monkeypatch.setattr(mc, "judge_once", fail_judge)
    profile, meta = mc.assess_fields_council(
        "passage", ["register"], prior={"register": "archaic"}
    )
    assert profile == {"register": "archaic"}
    assert meta["register"]["reused_prior"] is True


# ---------------------------------------------------------------------------
# classify(..., llm_mode="council") integration (mocked judges + computables)
# ---------------------------------------------------------------------------


def test_classify_council_path(monkeypatch) -> None:
    import tools.style_classification.metrics_computable as computable

    monkeypatch.setattr(computable, "compute", lambda text: {"lexical_density": 0.5})

    def fake_judge(metric, text, *, variant, model=mc.DEFAULT_MODEL, knowledge=""):
        # Unanimous label = metric id's first allowed value.
        return {
            "variant": variant,
            "label": metric["values"][0],
            "evidence": "cue",
            "parse_ok": True,
        }

    monkeypatch.setattr(mc, "judge_once", fake_judge)

    from tools.style_classification.classify_passage import classify

    council_meta: dict = {}
    profile = classify(
        "A long enough prose passage " * 20,
        use_llm=True,
        pass_mode="fast",
        llm_mode="council",
        council_meta_out=council_meta,
    )

    # Computable field preserved, and council-voted fields present.
    assert profile["lexical_density"] == 0.5
    assert "register" in profile
    assert council_meta  # per-field votes recorded
    assert council_meta["register"]["consensus"] is True
    assert council_meta["register"]["method"] == "unanimous"


# ---------------------------------------------------------------------------
# Arbitrator
# ---------------------------------------------------------------------------


def test_build_arbitrator_prompts_lists_judges_and_passage() -> None:
    metric = mc.resolve_metric("tone")
    votes = [
        {"variant": "definition", "label": "lyrical", "evidence": "soft golden light"},
        {"variant": "evidence", "label": "tense", "evidence": "heart racing"},
        {"variant": "contrast", "label": None, "evidence": None},
    ]
    system, user = mc.build_arbitrator_prompts(metric, "PASSAGE-BODY-XYZ", votes)
    assert "arbitrator" in system.lower()
    # Each judge's label + evidence surfaced for review.
    assert "lyrical" in user and "soft golden light" in user
    assert "tense" in user and "heart racing" in user
    assert "definition" in user and "evidence" in user and "contrast" in user
    # Allowed labels + rationale key + the passage present.
    assert "rationale" in user
    assert "PASSAGE-BODY-XYZ" in user
    for value in metric["values"]:
        assert value in user


def test_arbitrate_valid_decision() -> None:
    metric = mc.resolve_metric("tone")
    votes = [{"variant": "definition", "label": "lyrical", "evidence": "x"}]
    payload = '{"tone": "melancholic", "rationale": "elegiac diction dominates"}'
    with patch.object(mc, "llm_complete", return_value=payload):
        d = mc.arbitrate(metric, "prose", votes)
    assert d["value"] == "melancholic"
    assert d["rationale"] == "elegiac diction dominates"
    assert d["parse_ok"] is True


def test_arbitrate_abstains_on_null() -> None:
    metric = mc.resolve_metric("tone")
    with patch.object(mc, "llm_complete", return_value='{"tone": null, "rationale": "ambiguous"}'):
        d = mc.arbitrate(metric, "prose", [])
    assert d["value"] is None
    assert d["error"] == "arbitrator_abstained"


def test_arbitrate_rejects_invalid_label() -> None:
    metric = mc.resolve_metric("tone")
    with patch.object(mc, "llm_complete", return_value='{"tone": "whimsical"}'):
        d = mc.arbitrate(metric, "prose", [])
    assert d["value"] is None
    assert d["error"] == "label_not_allowed"


def test_assess_metric_council_unanimous_skips_arbitrator(monkeypatch) -> None:
    def fake_judge(metric, text, *, variant, model=mc.DEFAULT_MODEL, knowledge=""):
        return {"variant": variant, "label": "tight", "evidence": "e", "parse_ok": True}

    def fail_arbitrate(*a, **k):  # pragma: no cover - must not run when unanimous
        raise AssertionError("arbitrator must not run on unanimous votes")

    monkeypatch.setattr(mc, "judge_once", fake_judge)
    monkeypatch.setattr(mc, "arbitrate", fail_arbitrate)
    value, meta = mc.assess_metric_council("prose", "cohesion")
    assert value == "tight"
    assert meta["method"] == "unanimous"
    assert meta["arbitration"] is None


def test_assess_metric_council_split_invokes_arbitrator(monkeypatch) -> None:
    labels = {"definition": "loose", "evidence": "tight", "contrast": "standard"}

    def fake_judge(metric, text, *, variant, model=mc.DEFAULT_MODEL, knowledge=""):
        return {"variant": variant, "label": labels[variant], "evidence": "e", "parse_ok": True}

    def fake_arbitrate(metric, text, votes, *, model=mc.DEFAULT_MODEL, knowledge=""):
        return {"value": "tight", "rationale": "dense reference chains", "parse_ok": True, "error": None}

    monkeypatch.setattr(mc, "judge_once", fake_judge)
    monkeypatch.setattr(mc, "arbitrate", fake_arbitrate)
    value, meta = mc.assess_metric_council("prose", "cohesion")
    assert value == "tight"
    assert meta["method"] == "arbitrated"
    assert meta["arbitration"]["value"] == "tight"
    assert meta["arbitration"]["rationale"] == "dense reference chains"


def test_assess_metric_council_arbitrates_all_none(monkeypatch) -> None:
    # All judges fail to produce a valid label -> arbitrator still decides from passage.
    def fake_judge(metric, text, *, variant, model=mc.DEFAULT_MODEL, knowledge=""):
        return {"variant": variant, "label": None, "evidence": None, "parse_ok": False}

    def fake_arbitrate(metric, text, votes, *, model=mc.DEFAULT_MODEL, knowledge=""):
        return {"value": "standard", "rationale": "recovered", "parse_ok": True, "error": None}

    monkeypatch.setattr(mc, "judge_once", fake_judge)
    monkeypatch.setattr(mc, "arbitrate", fake_arbitrate)
    value, meta = mc.assess_metric_council("prose", "cohesion")
    assert value == "standard"
    assert meta["method"] == "arbitrated"
