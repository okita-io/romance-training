"""Property-based tests for v2 editorial rubric scoring.

Uses Hypothesis to verify universal correctness properties of the rubric
scoring integration into the v2 pipeline.

Feature: v2-editorial-rubric-scoring
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from romance_factory.generate.config_v2 import V2Config
from romance_factory.story_core.editorial_rules import _parse_flexible_score
from romance_factory.generate.agents.rubric_grader import RubricGrader


# ── Strategies ──────────────────────────────────────────────────────────────

# Valid threshold in [0.0, 10.0]
valid_threshold = st.floats(
    min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
)

# Invalid threshold: below 0.0
below_range = st.floats(
    max_value=-0.001, allow_nan=False, allow_infinity=False,
    min_value=-1e6,
)

# Invalid threshold: above 10.0
above_range = st.floats(
    min_value=10.001, allow_nan=False, allow_infinity=False,
    max_value=1e6,
)

# Any out-of-range threshold
invalid_threshold = st.one_of(below_range, above_range)


# ── Property 7: Config validation bounds ───────────────────────────────────
# Feature: v2-editorial-rubric-scoring, Property 7: Config validation bounds
# **Validates: Requirements 8.5**
#
# For any float value outside [0.0, 10.0] assigned to rubric_pass_threshold
# or rubric_per_rule_min, V2Config.validate() SHALL raise ValueError.
# For any float value within [0.0, 10.0], validation SHALL succeed
# (assuming all other fields are valid).


class TestConfigValidationBounds:
    """Property 7: Config validation bounds."""

    @given(value=invalid_threshold)
    @settings(max_examples=100)
    def test_rubric_pass_threshold_outside_range_raises(self, value: float) -> None:
        """**Validates: Requirements 8.5**

        For any float outside [0.0, 10.0] assigned to rubric_pass_threshold,
        V2Config.validate() SHALL raise ValueError.
        """
        cfg = V2Config(rubric_pass_threshold=value)
        with pytest.raises(ValueError, match="rubric_pass_threshold"):
            cfg.validate()

    @given(value=invalid_threshold)
    @settings(max_examples=100)
    def test_rubric_per_rule_min_outside_range_raises(self, value: float) -> None:
        """**Validates: Requirements 8.5**

        For any float outside [0.0, 10.0] assigned to rubric_per_rule_min,
        V2Config.validate() SHALL raise ValueError.
        """
        cfg = V2Config(rubric_per_rule_min=value)
        with pytest.raises(ValueError, match="rubric_per_rule_min"):
            cfg.validate()

    @given(pass_thresh=valid_threshold, rule_min=valid_threshold)
    @settings(max_examples=100)
    def test_rubric_thresholds_in_range_pass_validation(
        self, pass_thresh: float, rule_min: float
    ) -> None:
        """**Validates: Requirements 8.5**

        For any float values within [0.0, 10.0] assigned to both
        rubric_pass_threshold and rubric_per_rule_min, V2Config.validate()
        SHALL succeed (all other fields use valid defaults).
        """
        cfg = V2Config(
            rubric_pass_threshold=pass_thresh,
            rubric_per_rule_min=rule_min,
        )
        cfg.validate()  # Should not raise


# ── Strategies for score values ─────────────────────────────────────────────

# Integer scores in [0, 10]
integer_scores = st.integers(min_value=0, max_value=10)

# Single-decimal float scores in [0.0, 10.0], e.g. 0.0, 0.1, ..., 9.9, 10.0
single_decimal_scores = st.integers(min_value=0, max_value=100).map(lambda n: n / 10.0)

# Any valid score: integer or single-decimal float
any_score = st.one_of(integer_scores, single_decimal_scores)


# ── Property 3: Flexible score parsing round-trip ──────────────────────────
# Feature: v2-editorial-rubric-scoring, Property 3: Flexible score parsing round-trip


class TestFlexibleScoreParsingRoundTrip:
    """Property 3: Flexible score parsing round-trip."""

    @given(score=any_score)
    @settings(max_examples=100)
    def test_int_input(self, score: int | float) -> None:
        """**Validates: Requirements 3.1**

        _parse_flexible_score returns the original value when given an int.
        """
        int_val = int(score) if isinstance(score, int) or score == int(score) else score
        if isinstance(score, int) or score == int(score):
            parsed, method = _parse_flexible_score(int(score))
            assert parsed == float(int(score))

    @given(score=any_score)
    @settings(max_examples=100)
    def test_float_input(self, score: int | float) -> None:
        """**Validates: Requirements 3.1**

        _parse_flexible_score returns the original value when given a float.
        """
        parsed, method = _parse_flexible_score(float(score))
        assert parsed == float(score)

    @given(score=any_score)
    @settings(max_examples=100)
    def test_string_input(self, score: int | float) -> None:
        """**Validates: Requirements 3.1**

        _parse_flexible_score returns the original value when given str(value).
        """
        parsed, method = _parse_flexible_score(str(score))
        assert parsed == float(score)

    @given(score=any_score)
    @settings(max_examples=100)
    def test_fraction_format(self, score: int | float) -> None:
        """**Validates: Requirements 3.1**

        _parse_flexible_score returns the original value for "X/10" format.
        """
        fraction_str = f"{score}/10"
        parsed, method = _parse_flexible_score(fraction_str)
        assert parsed == float(score)

    @given(score=any_score)
    @settings(max_examples=100)
    def test_out_of_format(self, score: int | float) -> None:
        """**Validates: Requirements 3.1**

        _parse_flexible_score returns the original value for "X out of 10" format.
        """
        out_of_str = f"{score} out of 10"
        parsed, method = _parse_flexible_score(out_of_str)
        assert parsed == float(score)


# ── Strategies for JSON objects ─────────────────────────────────────────────

# Simple JSON-safe values: ints and short strings (no control chars or backslashes
# that could break JSON serialization round-trips).
json_safe_values = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "S", "Z"),
            blacklist_characters='\\"\n\r\t',
        ),
        min_size=0,
        max_size=20,
    ),
)

# Simple JSON objects: dicts with 1-5 string keys and int/string values.
simple_json_objects = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N"),
            blacklist_characters='\\"\n\r\t',
        ),
        min_size=1,
        max_size=10,
    ),
    values=json_safe_values,
    min_size=1,
    max_size=5,
)

# Non-brace prefix text: letters, digits, spaces, punctuation — no { or } or [ or ]
non_brace_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Z"),
        blacklist_characters="{}[]`",
    ),
    min_size=1,
    max_size=50,
)


# ── Property 4: JSON extraction from wrapped payloads ──────────────────────
# Feature: v2-editorial-rubric-scoring, Property 4: JSON extraction from wrapped payloads


class TestJsonExtractionFromWrappedPayloads:
    """Property 4: JSON extraction from wrapped payloads."""

    @given(obj=simple_json_objects)
    @settings(max_examples=100)
    def test_bare_json_object_extracts_correctly(self, obj: dict) -> None:
        """**Validates: Requirements 3.2**

        For any valid JSON object string, _extract_json SHALL extract a
        string that json.loads() can parse into an equivalent object.
        """
        raw = json.dumps(obj)
        extracted = RubricGrader._extract_json(raw)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert parsed == obj

    @given(obj=simple_json_objects)
    @settings(max_examples=100)
    def test_json_fenced_block_extracts_correctly(self, obj: dict) -> None:
        """**Validates: Requirements 3.2**

        For any valid JSON object string wrapped in ```json ... ``` fences,
        _extract_json SHALL extract a string that json.loads() can parse
        into an equivalent object.
        """
        raw = "```json\n" + json.dumps(obj) + "\n```"
        extracted = RubricGrader._extract_json(raw)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert parsed == obj

    @given(obj=simple_json_objects, prefix=non_brace_text)
    @settings(max_examples=100)
    def test_prepended_non_brace_text_extracts_correctly(
        self, obj: dict, prefix: str
    ) -> None:
        """**Validates: Requirements 3.2**

        For any valid JSON object string prepended with arbitrary non-brace
        text, _extract_json SHALL extract a string that json.loads() can
        parse into an equivalent object.
        """
        raw = prefix + json.dumps(obj)
        extracted = RubricGrader._extract_json(raw)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert parsed == obj

    @given(obj=simple_json_objects, suffix=non_brace_text)
    @settings(max_examples=100)
    def test_appended_trailing_text_extracts_correctly(
        self, obj: dict, suffix: str
    ) -> None:
        """**Validates: Requirements 3.2**

        For any valid JSON object string with trailing text appended,
        _extract_json SHALL extract a string that json.loads() can parse
        into an equivalent object.
        """
        raw = json.dumps(obj) + suffix
        extracted = RubricGrader._extract_json(raw)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert parsed == obj


# ── Imports for Property 1 ──────────────────────────────────────────────────

from unittest.mock import MagicMock

from romance_factory.story_core.editorial_rules import (
    ALL_RULES,
    CATEGORY_WEIGHTS,
    EDITORIAL_PASS_FAIL,
    EditorialReport,
    RuleCategory,
    Severity,
    compute_report,
    cliffhanger_editorial_weight_for_profile,
    CHAPTER_CLIFFHANGERS_AND_PAYOFFS,
)

_N_EDITORIAL_RULES = len(ALL_RULES)

from romance_factory.generate.agents.editorial import EditorialAgent
from romance_factory.generate.models import (
    DocumentMetadata,
    EditorialIssue,
)
from romance_factory.generate.prompt_builder import PromptBuilder


# ── Strategies for Property 1 ──────────────────────────────────────────────

# Score per rule: float in [0, 10]
rule_score_value = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Optional author_profile with cliffhanger_editorial_weight
optional_author_profile = st.one_of(
    st.none(),
    st.just({}),  # profile without cliffhanger weight
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False).map(
        lambda w: {"cliffhanger_editorial_weight": w}
    ),
)


# ── Property 1: Score computation correctness ──────────────────────────────
# Feature: v2-editorial-rubric-scoring, Property 1: Score computation correctness


class TestScoreComputationCorrectness:
    """Property 1: Score computation correctness."""

    @given(
        scores=st.lists(
            rule_score_value, min_size=_N_EDITORIAL_RULES, max_size=_N_EDITORIAL_RULES
        ),
        author_profile=optional_author_profile,
    )
    @settings(max_examples=100)
    def test_category_and_overall_scores(
        self, scores: list[float], author_profile: dict | None
    ) -> None:
        """**Validates: Requirements 4.1, 4.2, 4.3**

        For any set of one score per rubric rule (each in [0, 10]) and any
        author_profile (with or without cliffhanger_editorial_weight),
        compute_report() SHALL produce category scores equal to the weighted
        average of rule scores within each category (using each rule's weight,
        with ENG-06 using the effective weight from
        cliffhanger_editorial_weight_for_profile), and an overall score equal
        to the weighted average of category scores using CATEGORY_WEIGHTS.
        """
        # Build raw_scores list matching ALL_RULES order
        raw_scores = [
            {"rule_id": rule.id, "score": score, "notes": "test"}
            for rule, score in zip(ALL_RULES, scores)
        ]

        report = compute_report(
            chapter_num=1,
            raw_scores=raw_scores,
            author_profile=author_profile,
        )

        # Independently compute expected category scores
        eng06_effective_weight = cliffhanger_editorial_weight_for_profile(author_profile)

        cat_accum: dict[RuleCategory, tuple[float, float]] = {}
        for rule, score in zip(ALL_RULES, scores):
            weight = (
                eng06_effective_weight
                if rule.id == CHAPTER_CLIFFHANGERS_AND_PAYOFFS.id
                else rule.weight
            )
            ws, wt = cat_accum.get(rule.category, (0.0, 0.0))
            cat_accum[rule.category] = (ws + score * weight, wt + weight)

        expected_cat_scores: dict[RuleCategory, float] = {}
        for cat, (ws, wt) in cat_accum.items():
            expected_cat_scores[cat] = ws / wt if wt else 0.0

        # Assert category scores match
        for cat in expected_cat_scores:
            assert cat in report.category_scores, f"Missing category {cat}"
            assert report.category_scores[cat] == pytest.approx(
                expected_cat_scores[cat], abs=1e-9
            ), f"Category {cat.value}: expected {expected_cat_scores[cat]}, got {report.category_scores[cat]}"

        # Independently compute expected overall score
        weighted_overall = sum(
            expected_cat_scores.get(cat, 0.0) * CATEGORY_WEIGHTS.get(cat, 0.0)
            for cat in RuleCategory
        )
        total_cat_weight = sum(
            CATEGORY_WEIGHTS.get(cat, 0.0)
            for cat in expected_cat_scores
        )
        expected_overall = weighted_overall / total_cat_weight if total_cat_weight else 0.0

        assert report.overall_score == pytest.approx(
            expected_overall, abs=1e-9
        ), f"Overall: expected {expected_overall}, got {report.overall_score}"


# ── Strategies for Property 2 ──────────────────────────────────────────────

# RAG diagnostic score in [0, 10]
rag_score_st = st.floats(
    min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
)

# Rubric overall score in [0, 10]
rubric_overall_score_st = st.floats(
    min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
)

# Severity for editorial issues
severity_st = st.sampled_from(["BLOCKING", "MAJOR", "MINOR"])

# Generate a list of EditorialIssue with controlled severities
editorial_issue_st = st.builds(
    EditorialIssue,
    type=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
    severity=severity_st,
    location=st.just("test location"),
    explanation=st.just("test explanation"),
    suggested_fix=st.just("test fix"),
)

# List of RAG issues (0 to 5)
rag_issues_st = st.lists(editorial_issue_st, min_size=0, max_size=5)

# Blocking rule IDs: subset of BLOCKING-severity rules
_BLOCKING_RULE_IDS = [r.id for r in ALL_RULES if r.severity == Severity.BLOCKING]

# Generate blocking_failures as a subset of actual blocking rule IDs
blocking_failures_st = st.lists(
    st.sampled_from(_BLOCKING_RULE_IDS) if _BLOCKING_RULE_IDS else st.nothing(),
    min_size=0,
    max_size=min(3, len(_BLOCKING_RULE_IDS)),
    unique=True,
)

# Config thresholds
rubric_pass_threshold_st = st.floats(
    min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
)
passing_score_threshold_st = st.floats(
    min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
)


def _make_editorial_agent(
    passing_score_threshold: float = 6.0,
    rubric_pass_threshold: float = 4.5,
    rubric_per_rule_min: float = 4.5,
) -> EditorialAgent:
    """Create a minimal EditorialAgent with mocked dependencies for testing _combine_results."""
    mock_engine = MagicMock()
    prompt_builder = PromptBuilder()
    config = V2Config(
        enable_rubric_scoring=False,  # Don't instantiate RubricGrader
        passing_score_threshold=passing_score_threshold,
        rubric_pass_threshold=rubric_pass_threshold,
        rubric_per_rule_min=rubric_per_rule_min,
    )
    return EditorialAgent(mock_engine, prompt_builder, config)


def _make_rubric_report(
    overall_score: float,
    blocking_failures: list[str],
    rubric_pass_threshold: float,
) -> EditorialReport:
    """Create an EditorialReport with controlled values.

    Sets `passed` consistently: True only when overall_score >= threshold
    AND no blocking failures.
    """
    passed = overall_score >= rubric_pass_threshold and len(blocking_failures) == 0
    return EditorialReport(
        chapter_number=1,
        overall_score=overall_score,
        passed=passed,
        blocking_failures=list(blocking_failures),
    )


# ── Property 2: Combined pass/fail correctness ────────────────────────────
# Feature: v2-editorial-rubric-scoring, Property 2: Combined pass/fail correctness


class TestCombinedPassFailCorrectness:
    """Property 2: Combined pass/fail correctness."""

    @given(
        rubric_overall=rubric_overall_score_st,
        rag_score=rag_score_st,
        blocking_failures=blocking_failures_st,
        rag_issues=rag_issues_st,
    )
    @settings(max_examples=100)
    def test_combined_pass_fail_matches_four_conditions(
        self,
        rubric_overall: float,
        rag_score: float,
        blocking_failures: list[str],
        rag_issues: list[EditorialIssue],
    ) -> None:
        """**Validates: Requirements 5.1, 5.2, 5.3**

        For any rubric overall score, RAG diagnostic score, set of per-rule
        rubric scores, and set of RAG issues, the combined pass/fail decision
        SHALL be True only when ALL of:
          (a) rubric overall score >= rubric_pass_threshold
          (b) RAG score >= passing_score_threshold
          (c) no BLOCKING-severity rubric rule scores below rubric_per_rule_min
          (d) no BLOCKING-severity RAG issues exist
        """
        # Use the YAML-defined thresholds that _combine_results actually uses
        # (rubric_report.passed uses EDITORIAL_PASS_FAIL thresholds)
        rubric_threshold = float(EDITORIAL_PASS_FAIL.chapter_overall_min)
        rag_threshold = 6.0  # V2Config default passing_score_threshold

        agent = _make_editorial_agent(
            passing_score_threshold=rag_threshold,
        )

        rubric_report = _make_rubric_report(
            overall_score=rubric_overall,
            blocking_failures=blocking_failures,
            rubric_pass_threshold=rubric_threshold,
        )

        metadata = DocumentMetadata(type="editorial", chapter=1, act=1)

        result = agent._combine_results(
            rag_score=rag_score,
            rag_issues=rag_issues,
            rag_rewrite_plan="",
            rubric_report=rubric_report,
            metadata=metadata,
        )

        # Independently compute expected combined_passed
        cond_a = rubric_overall >= rubric_threshold  # rubric overall passes
        cond_b = rag_score >= rag_threshold           # RAG score passes
        cond_c = len(blocking_failures) == 0          # no blocking rubric failures
        cond_d = not any(i.severity == "BLOCKING" for i in rag_issues)  # no blocking RAG issues

        expected_combined = cond_a and cond_b and cond_c and cond_d

        assert result.combined_passed == expected_combined, (
            f"combined_passed={result.combined_passed}, expected={expected_combined} | "
            f"cond_a(rubric>={rubric_threshold})={cond_a} "
            f"cond_b(rag>={rag_threshold})={cond_b} "
            f"cond_c(no_blocking_rubric)={cond_c} "
            f"cond_d(no_blocking_rag)={cond_d}"
        )

    @given(
        rag_score=rag_score_st,
        rag_issues=rag_issues_st,
    )
    @settings(max_examples=100)
    def test_combined_pass_fail_without_rubric_report(
        self,
        rag_score: float,
        rag_issues: list[EditorialIssue],
    ) -> None:
        """**Validates: Requirements 5.1, 5.2, 5.3**

        When rubric_report is None (rubric disabled or parse failure),
        combined_passed depends only on RAG score and RAG issues.
        rubric_passed defaults to True and no blocking rubric failures exist.
        """
        rag_threshold = 6.0
        agent = _make_editorial_agent(passing_score_threshold=rag_threshold)

        metadata = DocumentMetadata(type="editorial", chapter=1, act=1)

        result = agent._combine_results(
            rag_score=rag_score,
            rag_issues=rag_issues,
            rag_rewrite_plan="",
            rubric_report=None,
            metadata=metadata,
        )

        has_blocking_rag = any(i.severity == "BLOCKING" for i in rag_issues)
        expected = rag_score >= rag_threshold and not has_blocking_rag

        assert result.combined_passed == expected, (
            f"combined_passed={result.combined_passed}, expected={expected} | "
            f"rag_score={rag_score}, rag_threshold={rag_threshold}, "
            f"has_blocking_rag={has_blocking_rag}"
        )
        assert result.rubric_passed is True


# ── Property 9: Combined score is minimum ──────────────────────────────────
# Feature: v2-editorial-rubric-scoring, Property 9: Combined score is minimum


class TestCombinedScoreIsMinimum:
    """Property 9: Combined score is minimum."""

    @given(
        rubric_overall=rubric_overall_score_st,
        rag_score=rag_score_st,
    )
    @settings(max_examples=100)
    def test_editorial_score_is_min_when_rubric_exists(
        self,
        rubric_overall: float,
        rag_score: float,
    ) -> None:
        """**Validates: Requirements 10.3**

        For any rubric overall score R and RAG diagnostic score D (both in
        [0, 10]), the editorial_score stored in DocumentMetadata SHALL equal
        min(R, D) when rubric_report is not None.
        """
        agent = _make_editorial_agent()

        rubric_report = _make_rubric_report(
            overall_score=rubric_overall,
            blocking_failures=[],
            rubric_pass_threshold=4.5,
        )

        metadata = DocumentMetadata(type="editorial", chapter=1, act=1)

        result = agent._combine_results(
            rag_score=rag_score,
            rag_issues=[],
            rag_rewrite_plan="",
            rubric_report=rubric_report,
            metadata=metadata,
        )

        expected = min(rubric_overall, rag_score)
        assert result.metadata.editorial_score == pytest.approx(expected, abs=1e-9), (
            f"editorial_score={result.metadata.editorial_score}, "
            f"expected=min({rubric_overall}, {rag_score})={expected}"
        )

    @given(rag_score=rag_score_st)
    @settings(max_examples=100)
    def test_editorial_score_equals_rag_when_rubric_is_none(
        self,
        rag_score: float,
    ) -> None:
        """**Validates: Requirements 10.3**

        When rubric is disabled or parsing failed (rubric_report is None),
        editorial_score SHALL equal the RAG diagnostic score D.
        """
        agent = _make_editorial_agent()

        metadata = DocumentMetadata(type="editorial", chapter=1, act=1)

        result = agent._combine_results(
            rag_score=rag_score,
            rag_issues=[],
            rag_rewrite_plan="",
            rubric_report=None,
            metadata=metadata,
        )

        assert result.metadata.editorial_score == pytest.approx(rag_score, abs=1e-9), (
            f"editorial_score={result.metadata.editorial_score}, "
            f"expected={rag_score} (rubric_report is None)"
        )


# ── Strategies for Property 5 ──────────────────────────────────────────────

# Rule IDs used in rubric-style guidance entries
_RUBRIC_RULE_IDS = [r.id for r in ALL_RULES]

# Severity keywords for rubric entries (lowercase in the parenthetical)
_RUBRIC_SEVERITIES = ["blocking", "major", "minor"]

# Severity tags for RAG entries (uppercase in brackets)
_RAG_SEVERITY_TAGS = ["BLOCKING", "MAJOR", "MINOR"]


def _rubric_entry(rule_id: str, score: float, severity: str, description: str) -> str:
    """Build a rubric-style rewrite_guidance entry.

    Format: [CRAFT-01] Show Don't Tell (scored 3.5/10, blocking): Rewrite ...
    """
    return f"[{rule_id}] {description} (scored {score:.1f}/10, {severity}): Rewrite to fix {rule_id}"


def _rag_entry(index: int, severity_tag: str, issue_type: str, location: str) -> str:
    """Build a RAG-style rewrite_plan entry.

    Format: 1. [MAJOR] Fix missing_cliffhanger at end of act: Add suspense
    """
    return f"{index}. [{severity_tag}] Fix {issue_type} at {location}: Suggested fix text"


# Strategy: generate a rubric guidance entry with a controlled severity
rubric_guidance_entry_st = st.tuples(
    st.sampled_from(_RUBRIC_RULE_IDS),
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    st.sampled_from(_RUBRIC_SEVERITIES),
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
).map(lambda t: _rubric_entry(t[0], t[1], t[2], t[3]))

# Strategy: generate a RAG rewrite plan entry with a controlled severity
rag_guidance_entry_st = st.tuples(
    st.integers(min_value=1, max_value=20),
    st.sampled_from(_RAG_SEVERITY_TAGS),
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
).map(lambda t: _rag_entry(t[0], t[1], t[2], t[3]))

# Lists of rubric and RAG entries (1-5 each to ensure non-empty)
rubric_entries_st = st.lists(rubric_guidance_entry_st, min_size=1, max_size=5)
rag_entries_st = st.lists(rag_guidance_entry_st, min_size=1, max_size=5)


# ── Property 5: Rewrite guidance merge ordering ───────────────────────────
# Feature: v2-editorial-rubric-scoring, Property 5: Rewrite guidance merge ordering


class TestRewriteGuidanceMergeOrdering:
    """Property 5: Rewrite guidance merge ordering."""

    @given(
        rubric_entries=rubric_entries_st,
        rag_entries=rag_entries_st,
    )
    @settings(max_examples=100)
    def test_blocking_before_major_before_minor(
        self,
        rubric_entries: list[str],
        rag_entries: list[str],
    ) -> None:
        """**Validates: Requirements 7.1, 7.2**

        For any failed rubric report with rewrite_guidance and any RAG
        rewrite_plan, the merged rewrite plan SHALL contain all BLOCKING-
        severity entries before any MAJOR or MINOR entries, and all MAJOR
        entries before any MINOR entries.
        """
        rubric_report = EditorialReport(
            chapter_number=1,
            overall_score=2.0,
            passed=False,
            rewrite_guidance="\n".join(rubric_entries),
        )
        rag_rewrite_plan = "\n".join(rag_entries)

        merged = EditorialAgent._merge_rewrite_guidance(
            rubric_report, rag_rewrite_plan, [],
        )

        merged_lines = [line for line in merged.split("\n") if line.strip()]
        severities = [
            EditorialAgent._classify_severity(line) for line in merged_lines
        ]

        # Verify non-decreasing severity order (0=BLOCKING, 1=MAJOR, 2=MINOR, 3=unknown)
        for i in range(len(severities) - 1):
            assert severities[i] <= severities[i + 1], (
                f"Ordering violation at index {i}: severity {severities[i]} "
                f"followed by {severities[i + 1]}.\n"
                f"  Entry[{i}]: {merged_lines[i]}\n"
                f"  Entry[{i+1}]: {merged_lines[i + 1]}\n"
                f"  Full severity sequence: {severities}"
            )

    @given(
        rubric_entries=rubric_entries_st,
        rag_entries=rag_entries_st,
    )
    @settings(max_examples=100)
    def test_rubric_before_rag_within_same_severity(
        self,
        rubric_entries: list[str],
        rag_entries: list[str],
    ) -> None:
        """**Validates: Requirements 7.1, 7.2**

        For any failed rubric report with rewrite_guidance and any RAG
        rewrite_plan, within each severity tier the rubric guidance entries
        SHALL appear before RAG guidance entries.
        """
        rubric_report = EditorialReport(
            chapter_number=1,
            overall_score=2.0,
            passed=False,
            rewrite_guidance="\n".join(rubric_entries),
        )
        rag_rewrite_plan = "\n".join(rag_entries)

        merged = EditorialAgent._merge_rewrite_guidance(
            rubric_report, rag_rewrite_plan, [],
        )

        merged_lines = [line for line in merged.split("\n") if line.strip()]

        # Build sets of rubric and RAG entries for identification
        rubric_set = {e.strip() for e in rubric_entries}

        # Group merged entries by severity tier
        tiers: dict[int, list[str]] = {}
        for line in merged_lines:
            sev = EditorialAgent._classify_severity(line)
            tiers.setdefault(sev, []).append(line)

        # Within each tier, all rubric entries must come before all RAG entries
        for sev, tier_entries in tiers.items():
            seen_rag = False
            for entry in tier_entries:
                is_rubric = entry in rubric_set
                if is_rubric and seen_rag:
                    assert False, (
                        f"Rubric entry found after RAG entry in severity tier {sev}.\n"
                        f"  Rubric entry: {entry}\n"
                        f"  Full tier: {tier_entries}"
                    )
                if not is_rubric:
                    seen_rag = True


# ── Property 6: Guidance deduplication ─────────────────────────────────────
# Feature: v2-editorial-rubric-scoring, Property 6: Guidance deduplication


# Strategy: generate overlapping rubric + RAG entries that share some issue types.
# We pick a shared set of rule IDs, then build rubric entries using those IDs
# and RAG entries whose issue_type matches the lowered rule ID.

_OVERLAP_RULE_IDS = st.lists(
    st.sampled_from(_RUBRIC_RULE_IDS),
    min_size=1,
    max_size=5,
    unique=True,
)


@st.composite
def overlapping_guidance_entries(draw):
    """Generate rubric and RAG entries that share at least one issue type.

    Returns (rubric_entries, rag_entries) where some RAG entries use the
    same issue type (lowered rule_id) as rubric entries.
    """
    # Draw shared rule IDs that will appear in both rubric and RAG entries
    shared_ids = draw(_OVERLAP_RULE_IDS)

    # Build rubric entries: some from shared IDs, plus optional extras
    rubric_entries = []
    for rule_id in shared_ids:
        score = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        severity = draw(st.sampled_from(_RUBRIC_SEVERITIES))
        desc = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))))
        rubric_entries.append(_rubric_entry(rule_id, score, severity, desc))

    # Optionally add extra non-overlapping rubric entries
    extra_rubric = draw(st.lists(rubric_guidance_entry_st, min_size=0, max_size=3))
    rubric_entries.extend(extra_rubric)

    # Build RAG entries: some using shared rule IDs as issue_type (lowered)
    rag_entries = []
    for rule_id in shared_ids:
        idx = draw(st.integers(min_value=1, max_value=20))
        sev_tag = draw(st.sampled_from(_RAG_SEVERITY_TAGS))
        location = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))))
        # Use the lowered rule_id as the issue_type so _extract_issue_type
        # will produce the same key for both rubric and RAG entries
        rag_entries.append(_rag_entry(idx, sev_tag, rule_id.lower(), location))

    # Optionally add extra non-overlapping RAG entries
    extra_rag = draw(st.lists(rag_guidance_entry_st, min_size=0, max_size=3))
    rag_entries.extend(extra_rag)

    return rubric_entries, rag_entries


class TestGuidanceDeduplication:
    """Property 6: Guidance deduplication."""

    @given(data=overlapping_guidance_entries())
    @settings(max_examples=100)
    def test_deduplicated_guidance_has_at_most_one_entry_per_issue_type(
        self,
        data: tuple[list[str], list[str]],
    ) -> None:
        """**Validates: Requirements 7.3**

        For any set of rubric failed-rule types and RAG issue types where
        overlapping types exist, the merged rewrite guidance SHALL contain
        at most one entry per issue type.
        """
        rubric_entries, rag_entries = data

        # Combine rubric first (so rubric wins on dedup), then RAG
        combined = rubric_entries + rag_entries

        deduplicated = EditorialAgent._deduplicate_guidance(combined)

        # Extract issue types from deduplicated result
        issue_types = [
            EditorialAgent._extract_issue_type(entry) for entry in deduplicated
        ]

        # Verify uniqueness: no duplicate issue types
        assert len(issue_types) == len(set(issue_types)), (
            f"Duplicate issue types found in deduplicated guidance.\n"
            f"  Issue types: {issue_types}\n"
            f"  Duplicates: {[t for t in issue_types if issue_types.count(t) > 1]}\n"
            f"  Deduplicated entries: {deduplicated}"
        )


# ── Strategies for Property 8 ──────────────────────────────────────────────

# Strategy: generate a full set of per-rule raw_scores for compute_report
raw_scores_for_report_st = st.lists(
    rule_score_value, min_size=_N_EDITORIAL_RULES, max_size=_N_EDITORIAL_RULES,
).map(
    lambda scores: [
        {"rule_id": rule.id, "score": score, "notes": "test note"}
        for rule, score in zip(ALL_RULES, scores)
    ]
)

# RAG score in [0, 10]
rag_score_for_serial_st = st.floats(
    min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False,
)


@st.composite
def editorial_result_with_rubric(draw):
    """Generate an EditorialResult with a non-None rubric_report from compute_report().

    Uses compute_report() with random per-rule scores to produce realistic
    EditorialReport instances, then wraps them in an EditorialResult.
    """
    raw_scores = draw(raw_scores_for_report_st)
    rag_score = draw(rag_score_for_serial_st)
    rag_issues = draw(rag_issues_st)

    # Compute a realistic rubric report via the real compute_report function
    rubric_report = compute_report(
        chapter_num=1,
        raw_scores=raw_scores,
        author_profile=None,
    )

    metadata = DocumentMetadata(type="editorial", chapter=1, act=1)

    # Use _combine_results to get a properly constructed EditorialResult
    agent = _make_editorial_agent()
    result = agent._combine_results(
        rag_score=rag_score,
        rag_issues=rag_issues,
        rag_rewrite_plan="test rewrite plan",
        rubric_report=rubric_report,
        metadata=metadata,
    )

    return result


# ── Property 8: Serialization completeness ─────────────────────────────────
# Feature: v2-editorial-rubric-scoring, Property 8: Serialization completeness


class TestSerializationCompleteness:
    """Property 8: Serialization completeness."""

    @given(result=editorial_result_with_rubric())
    @settings(max_examples=100)
    def test_serialized_payload_contains_all_rubric_keys(
        self,
        result: "EditorialResult",
    ) -> None:
        """**Validates: Requirements 10.1**

        For any EditorialResult with a non-None rubric_report, the serialized
        feedback payload SHALL contain keys rubric_overall_score,
        rubric_category_scores, rubric_rule_scores, rubric_passed,
        combined_passed, and combined_score, and the rubric_rule_scores array
        SHALL have one entry per rule in the report.
        """
        # Precondition: rubric_report is non-None
        assert result.rubric_report is not None

        # Serialize
        serialized = EditorialAgent._serialize_feedback(result)
        payload = json.loads(serialized)

        # Verify all required keys exist
        required_keys = {
            "rubric_overall_score",
            "rubric_category_scores",
            "rubric_rule_scores",
            "rubric_passed",
            "combined_passed",
            "combined_score",
        }
        missing_keys = required_keys - set(payload.keys())
        assert not missing_keys, (
            f"Missing required keys in serialized payload: {missing_keys}\n"
            f"  Present keys: {sorted(payload.keys())}"
        )

        # Verify rubric_rule_scores has one entry per rule in the report
        expected_rule_count = len(result.rubric_report.rule_scores)
        actual_rule_count = len(payload["rubric_rule_scores"])
        assert actual_rule_count == expected_rule_count, (
            f"rubric_rule_scores count mismatch: "
            f"expected {expected_rule_count} (from report), "
            f"got {actual_rule_count} (in serialized payload)"
        )
