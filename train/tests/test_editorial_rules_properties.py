"""Property-based tests for editorial_rules.py — Tasks 6.2 and 6.3.

Uses hypothesis to verify universal correctness properties of
compute_report, EditorialReport, and rewrite_guidance generation
across arbitrary score combinations.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from romance_factory.story_core.editorial_rules import (
    ALL_RULES,
    EDITORIAL_PASS_FAIL,
    RULES_BY_ID,
    EditorialReport,
    RuleScore,
    Severity,
    compute_report,
)

_PER_RULE_MIN = float(EDITORIAL_PASS_FAIL.per_rule_min)
_CHAPTER_OVERALL_MIN = float(EDITORIAL_PASS_FAIL.chapter_overall_min)


# ── Strategies ──────────────────────────────────────────────────────────────

# Valid score in [0, 10].
score_value = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Generate a dict of {rule_id: score} for all known rules.
all_rule_scores_strategy = st.fixed_dictionaries(
    {r.id: score_value for r in ALL_RULES}
)


def _raw_scores_from_dict(score_dict: dict[str, float]) -> list[dict]:
    """Convert {rule_id: score} dict to the raw_scores format compute_report expects."""
    return [
        {"rule_id": rid, "score": s, "notes": "auto-generated"}
        for rid, s in score_dict.items()
    ]


# ── Property 17: Editorial Scoring Consistency ─────────────────────────────
# **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
#
# For any EditorialReport, all rule scores SHALL be between 0 and 10,
# any rule with score below per_rule_min SHALL have passed set to false,
# any report containing a BLOCKING severity failure SHALL have passed
# set to false regardless of overall score, and any report with
# overall_score below chapter_overall_min SHALL have needs_rewrite
# return true.


class TestEditorialScoringConsistency:
    """Property 17: Editorial Scoring Consistency."""

    @given(score_dict=all_rule_scores_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_all_rule_scores_between_0_and_10(
        self, score_dict: dict[str, float]
    ) -> None:
        """**Validates: Requirement 8.1**

        For any EditorialReport produced by compute_report, every
        RuleScore.score SHALL be between 0 and 10.
        """
        raw = _raw_scores_from_dict(score_dict)
        report = compute_report(1, raw)

        for rs in report.rule_scores:
            assert 0.0 <= rs.score <= 10.0, (
                f"Rule {rs.rule_id} score {rs.score} outside [0, 10]"
            )

    @given(score_dict=all_rule_scores_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_score_below_threshold_marks_rule_failed(
        self, score_dict: dict[str, float]
    ) -> None:
        """**Validates: Requirement 8.2**

        For any EditorialReport, any rule with score below per_rule_min SHALL have
        passed set to false.
        """
        raw = _raw_scores_from_dict(score_dict)
        report = compute_report(1, raw)

        for rs in report.rule_scores:
            if rs.score < _PER_RULE_MIN:
                assert rs.passed is False, (
                    f"Rule {rs.rule_id} scored {rs.score} (< {_PER_RULE_MIN}) but passed=True"
                )
            else:
                assert rs.passed is True, (
                    f"Rule {rs.rule_id} scored {rs.score} (>= {_PER_RULE_MIN}) but passed=False"
                )

    @given(score_dict=all_rule_scores_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_blocking_failure_auto_fails_report(
        self, score_dict: dict[str, float]
    ) -> None:
        """**Validates: Requirement 8.3**

        For any EditorialReport containing a BLOCKING severity failure,
        the report SHALL have passed set to false regardless of overall score.
        """
        raw = _raw_scores_from_dict(score_dict)
        report = compute_report(1, raw)

        has_blocking_failure = any(
            rs.score < _PER_RULE_MIN
            and RULES_BY_ID[rs.rule_id].severity == Severity.BLOCKING
            for rs in report.rule_scores
        )

        if has_blocking_failure:
            assert report.passed is False, (
                "Report passed despite BLOCKING severity failure"
            )

    @given(score_dict=all_rule_scores_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_below_threshold_needs_rewrite(
        self, score_dict: dict[str, float]
    ) -> None:
        """**Validates: Requirement 8.4**

        For any EditorialReport with overall_score below
        chapter_overall_min (and no other pass condition), needs_rewrite SHALL return true.
        """
        raw = _raw_scores_from_dict(score_dict)
        report = compute_report(1, raw)

        if report.overall_score < _CHAPTER_OVERALL_MIN:
            assert report.needs_rewrite is True, (
                f"overall_score={report.overall_score} < threshold="
                f"{_CHAPTER_OVERALL_MIN} but needs_rewrite is False"
            )

    @given(score_dict=all_rule_scores_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_blocking_failures_list_matches_rule_scores(
        self, score_dict: dict[str, float]
    ) -> None:
        """**Validates: Requirements 8.2, 8.3**

        The blocking_failures list SHALL contain exactly the rule_ids of
        BLOCKING-severity rules that scored below per_rule_min.
        """
        raw = _raw_scores_from_dict(score_dict)
        report = compute_report(1, raw)

        expected_blocking = {
            rs.rule_id
            for rs in report.rule_scores
            if rs.score < _PER_RULE_MIN
            and RULES_BY_ID[rs.rule_id].severity == Severity.BLOCKING
        }

        assert set(report.blocking_failures) == expected_blocking, (
            f"blocking_failures={report.blocking_failures}, "
            f"expected={expected_blocking}"
        )


# ── Property 18: Editorial Report Targets Only Failed Rules ────────────────
# **Validates: Requirement 8.5**
#
# For any EditorialReport where the text fails review, the rewrite_guidance
# SHALL reference only rules where passed is false. Rules that passed SHALL
# not appear in the rewrite guidance.


class TestEditorialReportTargetsOnlyFailedRules:
    """Property 18: Editorial Report Targets Only Failed Rules."""

    @given(score_dict=all_rule_scores_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_rewrite_guidance_only_mentions_failed_rules(
        self, score_dict: dict[str, float]
    ) -> None:
        """**Validates: Requirement 8.5**

        For any EditorialReport, the rewrite_guidance SHALL reference only
        rules where passed is false. Passing rules SHALL not appear.
        """
        raw = _raw_scores_from_dict(score_dict)
        report = compute_report(1, raw)

        failed_ids = {rs.rule_id for rs in report.rule_scores if not rs.passed}
        passed_ids = {rs.rule_id for rs in report.rule_scores if rs.passed}

        guidance = report.rewrite_guidance

        # Every failed rule should appear in guidance (if there are any)
        for rid in failed_ids:
            assert rid in guidance, (
                f"Failed rule {rid} not found in rewrite_guidance"
            )

        # No passing rule should appear in guidance
        for rid in passed_ids:
            assert rid not in guidance, (
                f"Passing rule {rid} found in rewrite_guidance"
            )

    @given(score_dict=all_rule_scores_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_all_passing_scores_produce_empty_guidance(
        self, score_dict: dict[str, float]
    ) -> None:
        """**Validates: Requirement 8.5**

        When all rules pass (score >= per_rule_min), rewrite_guidance SHALL be empty.
        """
        high_scores = {rid: max(s, _PER_RULE_MIN) for rid, s in score_dict.items()}
        raw = _raw_scores_from_dict(high_scores)
        report = compute_report(1, raw)

        assert report.rewrite_guidance == "", (
            f"Expected empty rewrite_guidance when all rules pass, "
            f"got: {report.rewrite_guidance[:100]}"
        )

    @given(score_dict=all_rule_scores_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_mixed_scores_guidance_excludes_passed(
        self, score_dict: dict[str, float]
    ) -> None:
        """**Validates: Requirement 8.5**

        For any mix of passing and failing scores, rewrite_guidance SHALL
        not contain rule IDs of rules that passed.
        """
        raw = _raw_scores_from_dict(score_dict)
        report = compute_report(1, raw)

        passed_ids = {rs.rule_id for rs in report.rule_scores if rs.passed}

        for rid in passed_ids:
            assert rid not in report.rewrite_guidance, (
                f"Passed rule {rid} should not appear in rewrite_guidance"
            )
