"""Tests for --pass both (fast + deep field sets, same model)."""

from unittest.mock import patch

from tools.style_classification.classify_passage import classify
from tools.style_classification.pass_config import (
    ALL_LLM_FIELDS,
    PASS1_LLM_FIELDS,
    PASS2_LLM_FIELDS,
    pass_complete,
    suggested_workers,
)


def test_pass_complete_both_requires_all_fields() -> None:
    partial = {f: "x" for f in PASS1_LLM_FIELDS}
    assert pass_complete(partial, "fast")
    assert not pass_complete(partial, "both")

    full = {f: "x" for f in ALL_LLM_FIELDS}
    assert pass_complete(full, "both")


def test_suggested_workers_both() -> None:
    assert suggested_workers("both") == 4


@patch("tools.style_classification.metrics_llm.assess")
@patch("tools.style_classification.metrics_computable.compute")
def test_classify_both_calls_fast_then_deep(mock_compute, mock_assess) -> None:
    mock_compute.return_value = {"lexical_density": 0.5, "sentence_length_mean": 12.0}

    def fake_assess(text, *, model, rubric, pass_mode, prior=None):
        if pass_mode == "fast":
            return {f: f"fast_{f}" for f in PASS1_LLM_FIELDS}
        return {f: f"deep_{f}" for f in PASS2_LLM_FIELDS}

    mock_assess.side_effect = fake_assess

    profile = classify(
        "She walked through the garden. The air was cold.",
        use_llm=True,
        llm_model="test-model",
        pass_mode="both",
        rubric=None,
    )

    assert mock_assess.call_count == 2
    assert mock_assess.call_args_list[0].kwargs["pass_mode"] == "fast"
    assert mock_assess.call_args_list[1].kwargs["pass_mode"] == "deep"
    assert pass_complete(profile, "both")
    assert profile["register"] == "fast_register"
    assert profile["tone"] == "deep_tone"


@patch("tools.style_classification.metrics_llm.assess")
@patch("tools.style_classification.metrics_computable.compute")
def test_classify_both_resumes_from_fast_only(mock_compute, mock_assess) -> None:
    mock_compute.return_value = {"lexical_density": 0.4}

    prior_fast = {f: f"prior_{f}" for f in PASS1_LLM_FIELDS}

    def fake_assess(text, *, model, rubric, pass_mode, prior=None):
        assert pass_mode == "deep"
        assert prior is not None
        return {f: f"deep_{f}" for f in PASS2_LLM_FIELDS}

    mock_assess.side_effect = fake_assess

    profile = classify(
        "A longer passage with enough words to classify properly here.",
        use_llm=True,
        pass_mode="both",
        prior_profile=prior_fast,
        rubric=None,
    )

    mock_assess.assert_called_once()
    assert pass_complete(profile, "both")
    assert profile["cohesion"] == "prior_cohesion"
    assert profile["mind_style"] == "deep_mind_style"
