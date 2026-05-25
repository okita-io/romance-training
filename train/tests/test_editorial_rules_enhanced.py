"""Tests for enhanced editorial_rules.py — Task 6.1.

Validates:
- RuleCategory and Severity enums
- EditorialRule, RuleScore, EditorialReport dataclasses
- Scoring logic: 0–10 per rule, fail below pass_fail.per_rule_min, BLOCKING auto-fail
- Weighted overall score computation against pass_fail.chapter_overall_min
- EditorialReport.needs_rewrite() and rewrite_guidance targeting only failed rules
"""

from romance_factory.story_core.editorial_rules import (
    ALL_RULES,
    CATEGORY_WEIGHTS,
    EDITORIAL_PASS_FAIL,
    RULES_BY_ID,
    EditorialReport,
    EditorialRule,
    RuleCategory,
    RuleScore,
    Severity,
    compute_report,
)

_PER_RULE_MIN = float(EDITORIAL_PASS_FAIL.per_rule_min)


class TestRuleCategoryEnum:
    def test_all_categories_present(self):
        expected = {"craft", "romance", "engagement", "consistency", "voice"}
        actual = {c.value for c in RuleCategory}
        assert actual == expected

    def test_category_values(self):
        assert RuleCategory.CRAFT.value == "craft"
        assert RuleCategory.ROMANCE.value == "romance"
        assert RuleCategory.ENGAGEMENT.value == "engagement"
        assert RuleCategory.CONSISTENCY.value == "consistency"
        assert RuleCategory.VOICE.value == "voice"


class TestSeverityEnum:
    def test_all_severities_present(self):
        expected = {"blocking", "major", "minor"}
        actual = {s.value for s in Severity}
        assert actual == expected

    def test_severity_values(self):
        assert Severity.BLOCKING.value == "blocking"
        assert Severity.MAJOR.value == "major"
        assert Severity.MINOR.value == "minor"


class TestEditorialRule:
    def test_rule_has_required_fields(self):
        rule = RULES_BY_ID["CRAFT-01"]
        assert isinstance(rule, EditorialRule)
        assert rule.name
        assert isinstance(rule.category, RuleCategory)
        assert isinstance(rule.severity, Severity)
        assert isinstance(rule.weight, float)
        assert rule.description


class TestRuleScore:
    def test_backward_compat_without_rule(self):
        """RuleScore can be created with just rule_id (backward compat)."""
        rs = RuleScore(rule_id="CRAFT-01", score=8.0, passed=True, notes="good")
        assert rs.rule_id == "CRAFT-01"
        assert rs.score == 8.0
        assert rs.passed is True
        assert rs.rule is None

    def test_with_rule_object(self):
        """RuleScore can include the resolved EditorialRule."""
        rule = RULES_BY_ID["CRAFT-01"]
        rs = RuleScore(rule_id="CRAFT-01", score=8.0, passed=True, notes="good", rule=rule)
        assert rs.rule is rule
        assert rs.rule.name == rule.name

    def test_manual_rule_score_below_threshold_fails(self):
        rs = RuleScore(rule_id="CRAFT-01", score=_PER_RULE_MIN - 0.1, passed=False)
        assert rs.passed is False

    def test_manual_rule_score_at_threshold_passes(self):
        rs = RuleScore(rule_id="CRAFT-01", score=_PER_RULE_MIN, passed=True)
        assert rs.passed is True


class TestEditorialReport:
    def test_needs_rewrite_when_not_passed(self):
        report = EditorialReport(chapter_number=1, passed=False)
        assert report.needs_rewrite is True

    def test_no_rewrite_when_passed(self):
        report = EditorialReport(chapter_number=1, passed=True)
        assert report.needs_rewrite is False

    def test_rewrite_guidance_default_empty(self):
        report = EditorialReport(chapter_number=1)
        assert report.rewrite_guidance == ""

    def test_rewrite_guidance_is_string(self):
        report = EditorialReport(chapter_number=1, rewrite_guidance="Fix craft issues")
        assert isinstance(report.rewrite_guidance, str)


class TestComputeReport:
    def _all_passing_scores(self):
        return [
            {"rule_id": r.id, "score": 8, "notes": "good"}
            for r in ALL_RULES
        ]

    def _all_failing_scores(self):
        return [
            {"rule_id": r.id, "score": 3, "notes": "poor"}
            for r in ALL_RULES
        ]

    def test_all_passing_scores_pass(self):
        """All scores well above per-rule min with no blocking failures should pass."""
        report = compute_report(1, self._all_passing_scores())
        assert report.passed is True
        assert report.needs_rewrite is False
        assert len(report.blocking_failures) == 0
        assert report.rewrite_guidance == ""

    def test_all_failing_scores_fail(self):
        """All scores below per-rule min should fail and generate rewrite_guidance."""
        report = compute_report(1, self._all_failing_scores())
        assert report.passed is False
        assert report.needs_rewrite is True
        assert report.rewrite_guidance != ""

    def test_score_below_per_rule_min_marks_rule_failed(self):
        """A score below per_rule_min should mark the rule as failed (Req 8.2)."""
        scores = [
            {
                "rule_id": "CRAFT-01",
                "score": max(0.0, _PER_RULE_MIN - 1.0),
                "notes": "telling not showing",
            }
        ]
        report = compute_report(1, scores)
        assert len(report.rule_scores) == 1
        assert report.rule_scores[0].passed is False

    def test_score_at_per_rule_min_marks_rule_passed(self):
        """A score of exactly per_rule_min should pass that rule (Req 8.1)."""
        scores = [{"rule_id": "CRAFT-01", "score": _PER_RULE_MIN, "notes": "acceptable"}]
        report = compute_report(1, scores)
        assert report.rule_scores[0].passed is True

    def test_blocking_failure_auto_fails(self):
        """A BLOCKING rule failure auto-fails regardless of overall score (Req 8.3)."""
        # Find a blocking rule
        blocking_rule = None
        for r in ALL_RULES:
            if r.severity == Severity.BLOCKING:
                blocking_rule = r
                break
        assert blocking_rule is not None, "No BLOCKING rule found in editorial rules"

        # Give all rules high scores except the blocking one
        scores = []
        for r in ALL_RULES:
            if r.id == blocking_rule.id:
                scores.append({"rule_id": r.id, "score": 3, "notes": "failed"})
            else:
                scores.append({"rule_id": r.id, "score": 10, "notes": "perfect"})

        report = compute_report(1, scores)
        assert report.passed is False
        assert blocking_rule.id in report.blocking_failures

    def test_rule_field_populated_on_rule_scores(self):
        """compute_report should populate the rule field on each RuleScore."""
        scores = [
            {"rule_id": "CRAFT-01", "score": 8, "notes": "good"},
            {"rule_id": "ROM-01", "score": 5, "notes": "weak"},
        ]
        report = compute_report(1, scores)
        for rs in report.rule_scores:
            assert rs.rule is not None
            assert rs.rule.id == rs.rule_id

    def test_rewrite_guidance_targets_only_failed_rules(self):
        """rewrite_guidance should only mention rules that failed (Req 8.5)."""
        scores = [
            {"rule_id": "CRAFT-01", "score": 4, "notes": "too much telling"},
            {"rule_id": "CRAFT-02", "score": 9, "notes": "excellent sensory"},
            {"rule_id": "ROM-01", "score": 3, "notes": "no chemistry"},
        ]
        report = compute_report(1, scores)
        guidance = report.rewrite_guidance
        # Failed rules should appear
        assert "CRAFT-01" in guidance
        assert "ROM-01" in guidance
        # Passing rule should NOT appear
        assert "CRAFT-02" not in guidance

    def test_weighted_overall_score_computation(self):
        """Overall score should be a weighted average across categories (Req 8.4)."""
        scores = self._all_passing_scores()
        report = compute_report(1, scores)
        assert 0.0 <= report.overall_score <= 10.0
        # With all 8s, overall should be around 8
        assert report.overall_score > 7.0

    def test_unknown_rule_id_ignored(self):
        """Unknown rule_ids in raw_scores should be silently ignored."""
        scores = [{"rule_id": "UNKNOWN-99", "score": 5, "notes": "?"}]
        report = compute_report(1, scores)
        assert len(report.rule_scores) == 0

    def test_empty_scores_produce_empty_report(self):
        """Empty raw_scores should produce a report with 0 overall and not passed."""
        report = compute_report(1, [])
        assert report.overall_score == 0.0
        assert report.passed is False
        assert report.rewrite_guidance == ""
