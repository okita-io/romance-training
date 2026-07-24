"""Tests for --pass both (fast + deep field sets, same model) and field batching."""

from unittest.mock import patch

from tools.style_classification.classify_passage import classify
from tools.style_classification.pass_config import (
    ALL_FIELD_BATCHES,
    ALL_LLM_FIELDS,
    PASS1_FIELD_BATCHES,
    PASS1_LLM_FIELDS,
    PASS2_FIELD_BATCHES,
    PASS2_LLM_FIELDS,
    batches_for_pass,
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


def test_batches_for_pass_semantic() -> None:
    assert len(batches_for_pass("fast", field_batch_size=3)) == len(PASS1_FIELD_BATCHES)
    assert len(batches_for_pass("deep", field_batch_size=3)) == len(PASS2_FIELD_BATCHES)
    assert len(batches_for_pass("full", field_batch_size=3)) == len(ALL_FIELD_BATCHES)
    assert all(len(b) <= 3 for b in batches_for_pass("fast", field_batch_size=3))


def test_batches_for_pass_legacy_one_shot() -> None:
    fast = batches_for_pass("fast", field_batch_size=0)
    assert len(fast) == 1
    assert fast[0] == PASS1_LLM_FIELDS
    deep = batches_for_pass("deep", field_batch_size=0)
    assert len(deep) == 1
    assert deep[0] == PASS2_LLM_FIELDS


@patch("tools.style_classification.metrics_llm.assess")
@patch("tools.style_classification.metrics_computable.compute")
def test_classify_both_legacy_two_calls(mock_compute, mock_assess) -> None:
    mock_compute.return_value = {"lexical_density": 0.5, "sentence_length_mean": 12.0}

    def fake_assess(text, *, model, rubric, pass_mode, fields=None, prior=None):
        if pass_mode == "fast":
            return {f: f"fast_{f}" for f in (fields or PASS1_LLM_FIELDS)}
        return {f: f"deep_{f}" for f in (fields or PASS2_LLM_FIELDS)}

    mock_assess.side_effect = fake_assess

    profile = classify(
        "She walked through the garden. The air was cold.",
        use_llm=True,
        llm_model="test-model",
        pass_mode="both",
        rubric=None,
        field_batch_size=0,
    )

    assert mock_assess.call_count == 2
    assert mock_assess.call_args_list[0].kwargs["pass_mode"] == "fast"
    assert mock_assess.call_args_list[1].kwargs["pass_mode"] == "deep"
    assert pass_complete(profile, "both")
    assert profile["register"] == "fast_register"
    assert profile["tone"] == "deep_tone"


@patch("tools.style_classification.metrics_llm.assess")
@patch("tools.style_classification.metrics_computable.compute")
def test_classify_both_batched_six_calls(mock_compute, mock_assess) -> None:
    mock_compute.return_value = {"lexical_density": 0.5, "sentence_length_mean": 12.0}

    def fake_assess(text, *, model, rubric, pass_mode, fields=None, prior=None):
        assert fields is not None
        assert len(fields) <= 3
        return {f: f"{pass_mode}_{f}" for f in fields}

    mock_assess.side_effect = fake_assess

    profile = classify(
        "She walked through the garden. The air was cold.",
        use_llm=True,
        llm_model="test-model",
        pass_mode="both",
        rubric=None,
        field_batch_size=3,
    )

    expected = len(PASS1_FIELD_BATCHES) + len(PASS2_FIELD_BATCHES)
    assert mock_assess.call_count == expected
    fast_calls = [c for c in mock_assess.call_args_list if c.kwargs["pass_mode"] == "fast"]
    deep_calls = [c for c in mock_assess.call_args_list if c.kwargs["pass_mode"] == "deep"]
    assert len(fast_calls) == len(PASS1_FIELD_BATCHES)
    assert len(deep_calls) == len(PASS2_FIELD_BATCHES)
    # Within-pass prior accumulates: second fast batch should see first batch keys
    second_fast_prior = fast_calls[1].kwargs.get("prior") or {}
    for key in PASS1_FIELD_BATCHES[0]:
        assert key in second_fast_prior
    # Deep batches get Pass1 labels in prior
    assert deep_calls[0].kwargs.get("prior")
    assert "register" in deep_calls[0].kwargs["prior"]
    assert pass_complete(profile, "both")
    assert profile["sentence_complexity"] == "fast_sentence_complexity"
    assert profile["tone"] == "deep_tone"
    assert profile["free_indirect_discourse"] == "deep_free_indirect_discourse"


@patch("tools.style_classification.metrics_llm.assess")
@patch("tools.style_classification.metrics_computable.compute")
def test_classify_both_resumes_from_fast_only(mock_compute, mock_assess) -> None:
    mock_compute.return_value = {"lexical_density": 0.4}

    prior_fast = {f: f"prior_{f}" for f in PASS1_LLM_FIELDS}

    def fake_assess(text, *, model, rubric, pass_mode, fields=None, prior=None):
        assert pass_mode == "deep"
        assert prior is not None
        return {f: f"deep_{f}" for f in (fields or PASS2_LLM_FIELDS)}

    mock_assess.side_effect = fake_assess

    profile = classify(
        "A longer passage with enough words to classify properly here.",
        use_llm=True,
        pass_mode="both",
        prior_profile=prior_fast,
        rubric=None,
        field_batch_size=0,
    )

    mock_assess.assert_called_once()
    assert pass_complete(profile, "both")
    assert profile["cohesion"] == "prior_cohesion"
    assert profile["mind_style"] == "deep_mind_style"
