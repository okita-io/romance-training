"""Property-based tests for Agent Output Structure and Editorial Enforcement.

Feature: lancedb-rag-pipeline-v2, Properties 7 and 8

Tests:
  - Property 7: Agent Output Structure Completeness
  - Property 8: Cliffhanger and Plot Twist Editorial Enforcement
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from romance_factory.generate.agents.act_generation import ActGenerationAgent
from romance_factory.generate.agents.editorial import EditorialAgent
from romance_factory.generate.models import (
    DocumentMetadata,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retrieval_result(text: str = "some context") -> RetrievalResult:
    """Build a simple RetrievalResult with sensible defaults."""
    return RetrievalResult(
        text=text,
        metadata=DocumentMetadata(type="act"),
        similarity_score=0.9,
    )


def _mock_lancedb_engine() -> MagicMock:
    """Create a MagicMock LanceDBEngine whose query() returns a single result
    and store_document() is a no-op."""
    engine = MagicMock()
    engine.query.return_value = [
        _make_retrieval_result("Retrieved context for testing."),
    ]
    engine.store_document.return_value = None
    return engine


def _mock_prompt_builder() -> MagicMock:
    """Create a MagicMock PromptBuilder that returns deterministic prompts."""
    builder = MagicMock()
    builder.build_act_generation_prompt.return_value = (
        "Write act prose here.",
        "You are a romance novelist.",
    )
    builder.build_act_validation_prompt.return_value = (
        "Evaluate this act.",
        "You are an editorial reviewer.",
    )
    return builder


# ---------------------------------------------------------------------------
# Property 7: Agent Output Structure Completeness
# ---------------------------------------------------------------------------


class TestAgentOutputStructureCompleteness:
    """Feature: lancedb-rag-pipeline-v2, Property 7: Agent Output Structure Completeness

    **Validates: Requirements 4.3, 5.2, 5.5, 6.3, 13.3**

    For any agent operation (act generation, editorial evaluation, or rewrite),
    the result SHALL contain non-empty text and complete metadata with all
    required fields. For editorial evaluations, the score SHALL be in the range
    [0.0, 10.0] and each issue SHALL include type, severity, location,
    explanation, and suggested_fix. When issues are present, a non-empty
    rewrite_plan SHALL be included.
    """

    # ------------------------------------------------------------------
    # ActGenerationAgent output structure
    # ------------------------------------------------------------------

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
        prose_body=st.text(min_size=10, max_size=300),
        summary=st.text(min_size=1, max_size=100),
        emotional_tone=st.text(min_size=1, max_size=50),
        plot_function=st.text(min_size=1, max_size=50),
        char_name=st.text(min_size=1, max_size=30),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @patch("romance_factory.generate.agents.act_generation.generate")
    def test_act_result_has_non_empty_text_and_complete_metadata(
        self,
        mock_generate: MagicMock,
        chapter: int,
        act: int,
        prose_body: str,
        summary: str,
        emotional_tone: str,
        plot_function: str,
        char_name: str,
    ) -> None:
        """ActResult must have non-empty text and metadata with type, chapter, act."""
        # Build a deterministic LLM response with embedded JSON metadata
        meta_block = json.dumps({
            "characters_involved": [char_name],
            "emotional_tone": emotional_tone,
            "plot_function": plot_function,
            "summary": summary,
            "foreshadowing_created": [],
            "relationship_changes": [],
        })
        llm_response = f"{prose_body}\n\n```json\n{meta_block}\n```"
        mock_generate.return_value = llm_response

        engine = _mock_lancedb_engine()
        builder = _mock_prompt_builder()
        agent = ActGenerationAgent(lancedb_engine=engine, prompt_builder=builder)

        result = agent.generate(chapter=chapter, act=act)

        # Non-empty text
        assert result.text, "ActResult.text must be non-empty"

        # Metadata completeness
        assert result.metadata.type == "act"
        assert result.metadata.chapter == chapter
        assert result.metadata.act == act
        assert isinstance(result.metadata.characters_involved, list)
        assert isinstance(result.metadata.emotional_tone, str)
        assert isinstance(result.metadata.plot_function, str)
        assert isinstance(result.metadata.summary, str)

    # ------------------------------------------------------------------
    # EditorialAgent output structure — score range and issue fields
    # ------------------------------------------------------------------

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
        score=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        issue_type=st.sampled_from([
            "continuity", "pacing", "motivation", "slop",
            "anti_pattern", "repetition", "genre_drift",
            "outline_deviation",
        ]),
        severity=st.sampled_from(["BLOCKING", "MAJOR", "MINOR", "INFO"]),
        location=st.text(min_size=1, max_size=50),
        explanation=st.text(min_size=1, max_size=100),
        suggested_fix=st.text(min_size=1, max_size=100),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @patch("romance_factory.generate.agents.editorial.generate")
    def test_editorial_result_score_in_range_and_issues_complete(
        self,
        mock_generate: MagicMock,
        chapter: int,
        act: int,
        score: float,
        issue_type: str,
        severity: str,
        location: str,
        explanation: str,
        suggested_fix: str,
    ) -> None:
        """EditorialResult.score must be in [0.0, 10.0] and each issue must
        have type, severity, location, explanation, and suggested_fix."""
        editorial_json = json.dumps({
            "score": score,
            "issues": [
                {
                    "type": issue_type,
                    "severity": severity,
                    "location": location,
                    "explanation": explanation,
                    "suggested_fix": suggested_fix,
                },
            ],
            "rewrite_plan": "Fix the identified issues.",
        })
        mock_generate.return_value = f"```json\n{editorial_json}\n```"

        engine = _mock_lancedb_engine()
        builder = _mock_prompt_builder()
        agent = EditorialAgent(lancedb_engine=engine, prompt_builder=builder)

        result = agent.evaluate(chapter=chapter, act=act)

        # Score in valid range
        assert 0.0 <= result.score <= 10.0, (
            f"Score {result.score} out of range [0.0, 10.0]"
        )

        # Each issue has all required fields
        for issue in result.issues:
            assert issue.type, "issue.type must be non-empty"
            assert issue.severity, "issue.severity must be non-empty"
            assert isinstance(issue.location, str), "issue.location must be a string"
            assert isinstance(issue.explanation, str), "issue.explanation must be a string"
            assert isinstance(issue.suggested_fix, str), "issue.suggested_fix must be a string"

        # Metadata completeness
        assert result.metadata.type == "editorial"
        assert result.metadata.chapter == chapter
        assert result.metadata.act == act

    # ------------------------------------------------------------------
    # EditorialAgent — rewrite_plan non-empty when issues present
    # ------------------------------------------------------------------

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @patch("romance_factory.generate.agents.editorial.generate")
    def test_editorial_rewrite_plan_non_empty_when_issues_present(
        self,
        mock_generate: MagicMock,
        chapter: int,
        act: int,
    ) -> None:
        """When issues are present, rewrite_plan SHALL be non-empty."""
        # Return issues but NO rewrite_plan from LLM — agent should build one
        editorial_json = json.dumps({
            "score": 4.0,
            "issues": [
                {
                    "type": "pacing",
                    "severity": "MAJOR",
                    "location": "paragraph 3",
                    "explanation": "Pacing is too slow.",
                    "suggested_fix": "Tighten the dialogue.",
                },
            ],
            "rewrite_plan": "",
        })
        mock_generate.return_value = f"```json\n{editorial_json}\n```"

        engine = _mock_lancedb_engine()
        builder = _mock_prompt_builder()
        agent = EditorialAgent(lancedb_engine=engine, prompt_builder=builder)

        result = agent.evaluate(chapter=chapter, act=act)

        assert len(result.issues) >= 1, "Expected at least one issue"
        assert result.rewrite_plan, (
            "rewrite_plan must be non-empty when issues are present"
        )


# ---------------------------------------------------------------------------
# Property 8: Cliffhanger and Plot Twist Editorial Enforcement
# ---------------------------------------------------------------------------


class TestCliffhangerPlotTwistEditorialEnforcement:
    """Feature: lancedb-rag-pipeline-v2, Property 8: Cliffhanger and Plot Twist Editorial Enforcement

    **Validates: Requirements 5.3, 5.4, 13.4, 13.5**

    For any act evaluated by the Editorial Agent: if the act is the last act
    of a chapter and does not end on a cliffhanger or unresolved suspense beat,
    the Editorial Agent SHALL raise a MAJOR severity issue of type
    "missing_cliffhanger". If the act is designated as a plot twist point and
    the twist is weak or missing, the Editorial Agent SHALL raise a MAJOR
    severity issue of type "weak_plot_twist".
    """

    # ------------------------------------------------------------------
    # Missing cliffhanger enforcement on last acts
    # ------------------------------------------------------------------

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
        # Act text that does NOT end with a cliffhanger indicator (?, ..., …, !)
        act_text=st.text(min_size=5, max_size=200).filter(
            lambda t: t.rstrip() != ""
            and not t.rstrip().endswith("?")
            and not t.rstrip().endswith("...")
            and not t.rstrip().endswith("\u2026")
            and not t.rstrip().endswith("!")
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @patch("romance_factory.generate.agents.editorial.generate")
    def test_major_missing_cliffhanger_on_last_act(
        self,
        mock_generate: MagicMock,
        chapter: int,
        act: int,
        act_text: str,
    ) -> None:
        """Last acts without a cliffhanger must trigger a MAJOR
        'missing_cliffhanger' issue."""
        # LLM returns clean editorial with no issues (no cliffhanger issue)
        editorial_json = json.dumps({
            "score": 7.0,
            "issues": [],
            "rewrite_plan": "",
        })
        mock_generate.return_value = f"```json\n{editorial_json}\n```"

        # Mock engine: query for acts returns the non-cliffhanger text
        engine = _mock_lancedb_engine()
        engine.query.side_effect = lambda collection, query_text, **kwargs: (
            [RetrievalResult(
                text=act_text,
                metadata=DocumentMetadata(type="act", chapter=chapter, act=act),
                similarity_score=0.95,
            )]
            if collection == "acts"
            else [_make_retrieval_result("context")]
        )

        builder = _mock_prompt_builder()
        agent = EditorialAgent(lancedb_engine=engine, prompt_builder=builder)

        result = agent.evaluate(
            chapter=chapter, act=act, is_last_act=True,
        )

        # Must contain a MAJOR missing_cliffhanger issue
        cliffhanger_issues = [
            i for i in result.issues
            if i.type == "missing_cliffhanger"
        ]
        assert len(cliffhanger_issues) >= 1, (
            f"Expected MAJOR 'missing_cliffhanger' issue for last act "
            f"ending with: {act_text[-30:]!r}"
        )
        for ci in cliffhanger_issues:
            assert ci.severity == "MAJOR", (
                f"missing_cliffhanger severity must be MAJOR, got {ci.severity}"
            )

    # ------------------------------------------------------------------
    # No false positive when act ends with cliffhanger indicator
    # ------------------------------------------------------------------

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
        ending=st.sampled_from(["?", "...", "\u2026", "!"]),
        body=st.text(min_size=5, max_size=100),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @patch("romance_factory.generate.agents.editorial.generate")
    def test_no_false_cliffhanger_issue_when_indicator_present(
        self,
        mock_generate: MagicMock,
        chapter: int,
        act: int,
        ending: str,
        body: str,
    ) -> None:
        """When act text ends with a cliffhanger indicator, the agent should
        NOT inject a missing_cliffhanger issue (LLM didn't flag one either)."""
        act_text = body + ending

        editorial_json = json.dumps({
            "score": 8.0,
            "issues": [],
            "rewrite_plan": "",
        })
        mock_generate.return_value = f"```json\n{editorial_json}\n```"

        engine = _mock_lancedb_engine()
        engine.query.side_effect = lambda collection, query_text, **kwargs: (
            [RetrievalResult(
                text=act_text,
                metadata=DocumentMetadata(type="act", chapter=chapter, act=act),
                similarity_score=0.95,
            )]
            if collection == "acts"
            else [_make_retrieval_result("context")]
        )

        builder = _mock_prompt_builder()
        agent = EditorialAgent(lancedb_engine=engine, prompt_builder=builder)

        result = agent.evaluate(
            chapter=chapter, act=act, is_last_act=True,
        )

        # Should NOT have a missing_cliffhanger issue injected by post-processing
        injected_cliffhanger = [
            i for i in result.issues
            if i.type == "missing_cliffhanger"
        ]
        assert len(injected_cliffhanger) == 0, (
            f"Should not inject missing_cliffhanger when text ends with "
            f"{ending!r}, but found {len(injected_cliffhanger)} issue(s)"
        )

    # ------------------------------------------------------------------
    # Weak plot twist enforcement
    # ------------------------------------------------------------------

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @patch("romance_factory.generate.agents.editorial.generate")
    def test_major_weak_plot_twist_on_twist_point(
        self,
        mock_generate: MagicMock,
        chapter: int,
        act: int,
    ) -> None:
        """Acts designated as plot twist points must trigger a MAJOR
        'weak_plot_twist' issue when the LLM doesn't flag one."""
        # LLM returns clean editorial — no twist issue detected
        editorial_json = json.dumps({
            "score": 6.5,
            "issues": [],
            "rewrite_plan": "",
        })
        mock_generate.return_value = f"```json\n{editorial_json}\n```"

        engine = _mock_lancedb_engine()
        builder = _mock_prompt_builder()
        agent = EditorialAgent(lancedb_engine=engine, prompt_builder=builder)

        result = agent.evaluate(
            chapter=chapter, act=act, is_plot_twist=True,
        )

        # Must contain a MAJOR weak_plot_twist issue
        twist_issues = [
            i for i in result.issues
            if i.type == "weak_plot_twist"
        ]
        assert len(twist_issues) >= 1, (
            "Expected MAJOR 'weak_plot_twist' issue for twist point act"
        )
        for ti in twist_issues:
            assert ti.severity == "MAJOR", (
                f"weak_plot_twist severity must be MAJOR, got {ti.severity}"
            )

    # ------------------------------------------------------------------
    # Combined: last act + plot twist point
    # ------------------------------------------------------------------

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
        act_text=st.text(min_size=5, max_size=200).filter(
            lambda t: t.rstrip() != ""
            and not t.rstrip().endswith("?")
            and not t.rstrip().endswith("...")
            and not t.rstrip().endswith("\u2026")
            and not t.rstrip().endswith("!")
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @patch("romance_factory.generate.agents.editorial.generate")
    def test_both_cliffhanger_and_twist_enforced_together(
        self,
        mock_generate: MagicMock,
        chapter: int,
        act: int,
        act_text: str,
    ) -> None:
        """When an act is both the last act AND a plot twist point, both
        missing_cliffhanger (MAJOR) and weak_plot_twist (MAJOR) must
        be raised."""
        editorial_json = json.dumps({
            "score": 5.0,
            "issues": [],
            "rewrite_plan": "",
        })
        mock_generate.return_value = f"```json\n{editorial_json}\n```"

        engine = _mock_lancedb_engine()
        engine.query.side_effect = lambda collection, query_text, **kwargs: (
            [RetrievalResult(
                text=act_text,
                metadata=DocumentMetadata(type="act", chapter=chapter, act=act),
                similarity_score=0.95,
            )]
            if collection == "acts"
            else [_make_retrieval_result("context")]
        )

        builder = _mock_prompt_builder()
        agent = EditorialAgent(lancedb_engine=engine, prompt_builder=builder)

        result = agent.evaluate(
            chapter=chapter, act=act,
            is_last_act=True, is_plot_twist=True,
        )

        issue_types = [i.type for i in result.issues]
        assert "missing_cliffhanger" in issue_types, (
            "Expected 'missing_cliffhanger' when is_last_act=True"
        )
        assert "weak_plot_twist" in issue_types, (
            "Expected 'weak_plot_twist' when is_plot_twist=True"
        )

        # Verify rewrite_plan is non-empty since issues exist
        assert result.rewrite_plan, (
            "rewrite_plan must be non-empty when issues are present"
        )
