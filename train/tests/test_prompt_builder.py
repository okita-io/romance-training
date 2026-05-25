"""Property-based tests for the Prompt Builder.

Feature: lancedb-rag-pipeline-v2, Properties 4, 5, 6

Tests:
  - Property 4: Prompt Template Completeness
  - Property 5: Prompt Context Truncation
  - Property 6: Conditional Prompt Constraints
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from romance_factory.generate.models import (
    DocumentMetadata,
    EditorialIssue,
    RetrievalResult,
    RetrievedContext,
)
from romance_factory.generate.prompt_builder import PromptBuilder


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

def _make_result(text: str = "some context") -> RetrievalResult:
    """Build a simple RetrievalResult with sensible defaults."""
    return RetrievalResult(
        text=text,
        metadata=DocumentMetadata(type="act"),
        similarity_score=0.9,
    )


def _retrieval_result_strategy(
    text_strategy: st.SearchStrategy[str] | None = None,
) -> st.SearchStrategy[RetrievalResult]:
    """Strategy that produces a single RetrievalResult."""
    txt = text_strategy or st.text(min_size=1, max_size=200)
    return st.builds(
        RetrievalResult,
        text=txt,
        metadata=st.builds(DocumentMetadata, type=st.just("act")),
        similarity_score=st.floats(min_value=0.0, max_value=1.0),
    )


def _non_empty_result_list(
    text_strategy: st.SearchStrategy[str] | None = None,
) -> st.SearchStrategy[list[RetrievalResult]]:
    """Strategy that produces a list with at least one RetrievalResult."""
    return st.lists(
        _retrieval_result_strategy(text_strategy),
        min_size=1,
        max_size=3,
    )



# ---------------------------------------------------------------------------
# Property 4: Prompt Template Completeness
# ---------------------------------------------------------------------------


class TestPromptTemplateCompleteness:
    """Feature: lancedb-rag-pipeline-v2, Property 4: Prompt Template Completeness

    **Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.7**

    For any of the four prompt templates and any valid input (RetrievedContext
    with at least one non-empty section), the Prompt_Builder SHALL produce a
    non-empty string containing the task instruction and all required context
    sections for that template type.
    """

    @given(
        context=st.builds(
            RetrievedContext,
            author_profile=_non_empty_result_list(),
            character_web=_non_empty_result_list(),
            story_outline=_non_empty_result_list(),
            chapter_outline=_non_empty_result_list(),
            act_outline=_non_empty_result_list(),
            foreshadowing=_non_empty_result_list(),
            relationship_arcs=_non_empty_result_list(),
            previous_acts=_non_empty_result_list(),
        ),
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_act_generation_prompt_completeness(
        self,
        context: RetrievedContext,
        chapter: int,
        act: int,
    ) -> None:
        """build_act_generation_prompt produces a non-empty prompt with task
        instruction and required context sections."""
        builder = PromptBuilder(max_context_chars=8000)
        prompt, system_prompt = builder.build_act_generation_prompt(
            chapter=chapter, act=act, context=context,
        )

        assert isinstance(prompt, str) and len(prompt) > 0
        assert isinstance(system_prompt, str) and len(system_prompt) > 0

        # Must contain the task instruction (plain prose generation, no "Task:" label)
        assert "Write act" in prompt
        # Must contain context section headers
        assert "Author Profile" in prompt
        assert "Character Web" in prompt

    @given(
        context=st.builds(
            RetrievedContext,
            act_outline=_non_empty_result_list(),
            chapter_outline=_non_empty_result_list(),
            story_outline=_non_empty_result_list(),
            character_web=_non_empty_result_list(),
            foreshadowing=_non_empty_result_list(),
            previous_acts=_non_empty_result_list(),
            author_profile=_non_empty_result_list(),
        ),
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
        act_text=st.text(min_size=10, max_size=500),
    )
    @settings(max_examples=100)
    def test_act_validation_prompt_completeness(
        self,
        context: RetrievedContext,
        chapter: int,
        act: int,
        act_text: str,
    ) -> None:
        """build_act_validation_prompt produces a non-empty prompt with act text
        and context sections."""
        builder = PromptBuilder(max_context_chars=8000)
        prompt, system_prompt = builder.build_act_validation_prompt(
            chapter=chapter, act=act, act_text=act_text, context=context,
        )

        assert isinstance(prompt, str) and len(prompt) > 0
        assert isinstance(system_prompt, str) and len(system_prompt) > 0

        # Must contain the act text section and instructions
        assert "Act Text" in prompt
        assert "Instructions" in prompt
        assert "Character Web" in prompt

    @given(
        context=st.builds(
            RetrievedContext,
            author_profile=_non_empty_result_list(),
            character_web=_non_empty_result_list(),
            story_outline=_non_empty_result_list(),
            chapter_outline=_non_empty_result_list(),
            act_outline=_non_empty_result_list(),
            foreshadowing=_non_empty_result_list(),
            relationship_arcs=_non_empty_result_list(),
            previous_acts=_non_empty_result_list(),
        ),
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
        original_text=st.text(min_size=10, max_size=500),
        rewrite_plan=st.text(min_size=10, max_size=200),
    )
    @settings(max_examples=100)
    def test_act_rewrite_prompt_completeness(
        self,
        context: RetrievedContext,
        chapter: int,
        act: int,
        original_text: str,
        rewrite_plan: str,
    ) -> None:
        """build_act_rewrite_prompt produces a non-empty prompt with original
        text, issues, rewrite plan, and context."""
        builder = PromptBuilder(max_context_chars=8000)
        issues = [
            EditorialIssue(
                type="continuity",
                severity="MAJOR",
                location="paragraph 2",
                explanation="Timeline inconsistency",
                suggested_fix="Adjust the timeline reference",
            )
        ]
        prompt, system_prompt = builder.build_act_rewrite_prompt(
            chapter=chapter,
            act=act,
            original_text=original_text,
            issues=issues,
            rewrite_plan=rewrite_plan,
            context=context,
        )

        assert isinstance(prompt, str) and len(prompt) > 0
        assert isinstance(system_prompt, str) and len(system_prompt) > 0

        # Must contain original text section and rewrite plan header
        assert "Original Text" in prompt
        assert "Rewrite Plan" in prompt
        assert "Author Profile" in prompt
        assert "## Editorial issues" in prompt
        assert "Timeline inconsistency" in prompt
        assert "Adjust the timeline reference" in prompt

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        chapter_text=st.text(min_size=10, max_size=500),
    )
    @settings(max_examples=100)
    def test_chapter_cleanup_prompt_completeness(
        self,
        chapter: int,
        chapter_text: str,
    ) -> None:
        """build_chapter_cleanup_prompt produces a non-empty prompt with chapter
        text, cleanup rules, and author style."""
        builder = PromptBuilder(max_context_chars=8000)
        author_profile = {"voice": "warm and witty", "tropes": "slow burn"}
        prompt, system_prompt = builder.build_chapter_cleanup_prompt(
            chapter=chapter,
            chapter_text=chapter_text,
            author_profile=author_profile,
        )

        assert isinstance(prompt, str) and len(prompt) > 0
        assert isinstance(system_prompt, str) and len(system_prompt) > 0

        # Must contain chapter text and cleanup rules
        assert f"Chapter {chapter}" in prompt
        assert "Cleanup Rules" in prompt
        assert "Author Style" in prompt


# ---------------------------------------------------------------------------
# Property 5: Prompt Context Truncation
# ---------------------------------------------------------------------------


class TestPromptContextTruncation:
    """Feature: lancedb-rag-pipeline-v2, Property 5: Prompt Context Truncation

    **Validates: Requirements 3.6**

    For any retrieved context section of any length, the Prompt_Builder SHALL
    truncate each section to at most the configured maximum character limit.
    """

    @given(
        seed=st.text(min_size=100, max_size=500),
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_act_generation_context_truncated(
        self,
        seed: str,
        chapter: int,
        act: int,
    ) -> None:
        """Context sections in act generation prompts are truncated to
        max_context_chars."""
        # Build a long string by repeating the seed text
        long_text = (seed * 200)[:25000]
        max_chars = 500
        builder = PromptBuilder(
            max_context_chars=max_chars,
            max_story_arc_chars=max_chars,
            max_previous_acts_chars=max_chars,
        )

        context = RetrievedContext(
            author_profile=[_make_result(long_text)],
            character_web=[_make_result(long_text)],
            story_outline=[_make_result(long_text)],
            chapter_outline=[_make_result(long_text)],
            act_outline=[_make_result(long_text)],
            foreshadowing=[_make_result(long_text)],
            relationship_arcs=[_make_result(long_text)],
            previous_acts=[_make_result(long_text)],
        )

        prompt, _ = builder.build_act_generation_prompt(
            chapter=chapter, act=act, context=context,
        )

        # The truncation marker is "…[truncated]" (13 chars).
        # Each context section body should be at most max_chars + len(marker).
        marker_overhead = len("…[truncated]")
        sections = prompt.split("## ")
        for section in sections:
            if not section.strip():
                continue
            # Extract the body after the header line
            lines = section.split("\n", 1)
            if len(lines) < 2:
                continue
            header = lines[0].strip().lower()
            if header.startswith("constraints"):
                continue
            # Story outline adds digest + labels; checked separately in TestStoryOutlineArcBudget.
            if header.startswith("story outline"):
                continue
            # Fixed prompt contracts (not retrieved RAG text; not subject to max_context_chars).
            if header.startswith("rough-draft length budget"):
                continue
            if header.startswith("verbosity contract"):
                continue
            if header.startswith("narrative purpose contract"):
                continue
            if header.startswith("intimacy"):
                continue
            body = lines[1].strip()
            # Only check context sections (those that had long text injected)
            if body and len(body) > max_chars:
                assert len(body) <= max_chars + marker_overhead, (
                    f"Section body length {len(body)} exceeds "
                    f"max_context_chars ({max_chars}) + marker overhead "
                    f"({marker_overhead})"
                )

    @given(
        seed=st.text(min_size=100, max_size=500),
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
        act_text=st.text(min_size=10, max_size=200),
    )
    @settings(max_examples=100)
    def test_act_validation_context_truncated(
        self,
        seed: str,
        chapter: int,
        act: int,
        act_text: str,
    ) -> None:
        """Context sections in validation prompts are truncated to
        max_context_chars."""
        long_text = (seed * 200)[:25000]
        max_chars = 500
        builder = PromptBuilder(
            max_context_chars=max_chars,
            max_story_arc_chars=max_chars,
            max_previous_acts_chars=max_chars,
        )

        context = RetrievedContext(
            act_outline=[_make_result(long_text)],
            chapter_outline=[_make_result(long_text)],
            story_outline=[_make_result(long_text)],
            character_web=[_make_result(long_text)],
            foreshadowing=[_make_result(long_text)],
            previous_acts=[_make_result(long_text)],
            author_profile=[_make_result(long_text)],
        )

        prompt, _ = builder.build_act_validation_prompt(
            chapter=chapter, act=act, act_text=act_text, context=context,
        )

        # Verify the _truncate helper was applied: the truncation marker
        # should appear for every long context section.
        marker = "…[truncated]"
        # Count how many context sections we injected with long text (7 sections)
        assert prompt.count(marker) >= 7, (
            f"Expected at least 7 truncation markers, found "
            f"{prompt.count(marker)}"
        )

    @given(
        seed=st.text(min_size=100, max_size=500),
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
        original_text=st.text(min_size=10, max_size=200),
        rewrite_plan=st.text(min_size=10, max_size=200),
    )
    @settings(max_examples=100)
    def test_act_rewrite_context_truncated(
        self,
        seed: str,
        chapter: int,
        act: int,
        original_text: str,
        rewrite_plan: str,
    ) -> None:
        """Context sections in rewrite prompts are truncated to
        max_context_chars."""
        long_text = (seed * 200)[:25000]
        max_chars = 500
        builder = PromptBuilder(max_context_chars=max_chars)

        context = RetrievedContext(
            author_profile=[_make_result(long_text)],
            character_web=[_make_result(long_text)],
            story_outline=[_make_result(long_text)],
            chapter_outline=[_make_result(long_text)],
            act_outline=[_make_result(long_text)],
            foreshadowing=[_make_result(long_text)],
            relationship_arcs=[_make_result(long_text)],
            previous_acts=[_make_result(long_text)],
        )

        prompt, _ = builder.build_act_rewrite_prompt(
            chapter=chapter,
            act=act,
            original_text=original_text,
            issues=[],
            rewrite_plan=rewrite_plan,
            context=context,
        )

        marker = "…[truncated]"
        # Four retrieved context sections are truncated in rewrite prompts.
        assert prompt.count(marker) >= 4, (
            f"Expected at least 4 truncation markers, found "
            f"{prompt.count(marker)}"
        )


# ---------------------------------------------------------------------------
# Property 6: Conditional Prompt Constraints
# ---------------------------------------------------------------------------


class TestConditionalPromptConstraints:
    """Feature: lancedb-rag-pipeline-v2, Property 6: Conditional Prompt Constraints

    **Validates: Requirements 4.6, 4.7**

    For any act generation call where is_last_act is True, the generated prompt
    SHALL contain cliffhanger instruction text. For any act generation call
    where is_plot_twist is True, the generated prompt SHALL contain plot twist
    constraint text.
    """

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_cliffhanger_present_when_last_act(
        self,
        chapter: int,
        act: int,
    ) -> None:
        """When is_last_act=True, the prompt contains cliffhanger instructions."""
        builder = PromptBuilder(max_context_chars=8000)
        context = RetrievedContext(
            author_profile=[_make_result()],
            act_outline=[_make_result()],
        )

        prompt, _ = builder.build_act_generation_prompt(
            chapter=chapter,
            act=act,
            context=context,
            is_last_act=True,
            is_plot_twist=False,
        )

        assert "cliffhanger" in prompt.lower(), (
            "Expected 'cliffhanger' in prompt when is_last_act=True"
        )

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_plot_twist_present_when_is_plot_twist(
        self,
        chapter: int,
        act: int,
    ) -> None:
        """When is_plot_twist=True, the prompt contains plot twist constraints."""
        builder = PromptBuilder(max_context_chars=8000)
        context = RetrievedContext(
            author_profile=[_make_result()],
            act_outline=[_make_result()],
        )

        prompt, _ = builder.build_act_generation_prompt(
            chapter=chapter,
            act=act,
            context=context,
            is_last_act=False,
            is_plot_twist=True,
        )

        assert "plot twist" in prompt.lower(), (
            "Expected 'plot twist' in prompt when is_plot_twist=True"
        )

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_both_constraints_present(
        self,
        chapter: int,
        act: int,
    ) -> None:
        """When both flags are True, both constraint texts appear."""
        builder = PromptBuilder(max_context_chars=8000)
        context = RetrievedContext(
            author_profile=[_make_result()],
            act_outline=[_make_result()],
        )

        prompt, _ = builder.build_act_generation_prompt(
            chapter=chapter,
            act=act,
            context=context,
            is_last_act=True,
            is_plot_twist=True,
        )

        assert "cliffhanger" in prompt.lower(), (
            "Expected 'cliffhanger' in prompt when is_last_act=True"
        )
        assert "plot twist" in prompt.lower(), (
            "Expected 'plot twist' in prompt when is_plot_twist=True"
        )

    @given(
        chapter=st.integers(min_value=1, max_value=50),
        act=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_neither_constraint_when_both_false(
        self,
        chapter: int,
        act: int,
    ) -> None:
        """When both flags are False, neither cliffhanger nor plot twist
        constraint text appears in the Constraints section."""
        builder = PromptBuilder(max_context_chars=8000)
        context = RetrievedContext(
            author_profile=[_make_result()],
            act_outline=[_make_result()],
        )

        prompt, _ = builder.build_act_generation_prompt(
            chapter=chapter,
            act=act,
            context=context,
            is_last_act=False,
            is_plot_twist=False,
        )

        # Extract the Constraints section specifically
        constraints_start = prompt.find("## Constraints")
        assert constraints_start != -1, "Constraints section not found"

        # Find the next section header after Constraints
        next_section = prompt.find("## ", constraints_start + len("## Constraints"))
        if next_section == -1:
            constraints_section = prompt[constraints_start:]
        else:
            constraints_section = prompt[constraints_start:next_section]

        constraints_lower = constraints_section.lower()
        assert "cliffhanger" not in constraints_lower, (
            "Unexpected 'cliffhanger' in Constraints when is_last_act=False"
        )
        assert "plot twist" not in constraints_lower, (
            "Unexpected 'plot twist' in Constraints when is_plot_twist=False"
        )


# ---------------------------------------------------------------------------
# World Context Block formatting (world-building-system, Req 8.4)
# ---------------------------------------------------------------------------


class TestWorldContextBlockActGeneration:
    """World Context Block appears between character web and story outline."""

    def test_wcb_compact_layers_formatted_when_json(self) -> None:
        wcb_json = (
            '{"macro_world": {"involved": "Both leads", "location_ids": ["loc-hall"], '
            '"world_change": "Cold stone; tension rises", '
            '"why_involved": "Public confrontation"}, '
            '"meso_world": {"venue_ids": ["ven-hall"]}, '
            '"micro_world": {"stages": ['
            '{"stage_id": "stg-ch1-act1-001", "venue_id": "ven-hall", "stage_slug": "main", '
            '"characters": ["Lead A", "Lead B"], "contains": []}'
            "]}, "
            '"meso_story": {"arc_plot_changes": "Reputation at stake for arc"}, '
            '"micro_characters": {"character_state": "Lead A: rigid; Lead B: flushed"}}'
        )
        builder = PromptBuilder(max_context_chars=16000)
        context = RetrievedContext(
            author_profile=[_make_result("Author.")],
            character_web=[_make_result("Characters.")],
            story_outline=[_make_result("Outline.")],
            world_outline=[
                RetrievalResult(
                    text=wcb_json,
                    metadata=DocumentMetadata(type="world_context_block"),
                    similarity_score=0.95,
                ),
            ],
        )
        prompt, _ = builder.build_act_generation_prompt(
            chapter=1, act=1, context=context,
        )
        assert "## World Context Block (continuity capsule)" in prompt
        assert "### Scene snapshot (macro)" in prompt
        assert "**Location IDs:** loc-hall" in prompt
        assert "### Venues (meso geography)" in prompt
        assert "### Stages (micro geography)" in prompt
        assert "### Arc-relevant plot (meso)" in prompt
        assert "### Character state (micro)" in prompt

    def test_wcb_section_before_story_outline(self) -> None:
        builder = PromptBuilder(max_context_chars=8000)
        context = RetrievedContext(
            author_profile=[_make_result("A")],
            character_web=[_make_result("C")],
            story_outline=[_make_result("S")],
            world_outline=[_make_result('{"macro_world": {"location": "X"}}')],
        )
        prompt, _ = builder.build_act_generation_prompt(
            chapter=2, act=1, context=context,
        )
        wcb_pos = prompt.find("## World Context Block (this scene)")
        story_pos = prompt.find("## Story Outline")
        char_pos = prompt.find("## Character Web")
        assert wcb_pos != -1 and story_pos != -1
        assert char_pos < wcb_pos < story_pos


class TestStoryOutlineArcBudget:
    """Story outline section uses ``max_story_arc_chars``, not ``max_context_chars``."""

    def test_arc_section_allows_more_than_generic_cap(self) -> None:
        import json

        big_premise = "P" * 9000
        arc = {"title": "T", "premise": big_premise, "central_conflict": "c"}
        text = json.dumps(arc)
        r = RetrievalResult(
            text=text,
            metadata=DocumentMetadata(type="outline"),
            similarity_score=1.0,
        )
        builder = PromptBuilder(
            max_context_chars=400,
            max_story_arc_chars=50000,
            num_chapters=10,
        )
        section = builder._format_story_outline_section([r])
        assert "## Story Outline" in section
        assert "Arc digest" in section
        assert big_premise in section

    def test_truncate_zero_means_unlimited_for_helpers(self) -> None:
        builder = PromptBuilder(max_context_chars=400, max_story_arc_chars=0)
        body = "Z" * 5000
        assert builder._truncate(body, max_chars=0) == body


class TestEstablishmentActConstraints:
    """First act in establishment band gets extra grounding constraints."""

    def test_first_act_chapter_one_has_establishment_constraint(self) -> None:
        builder = PromptBuilder(max_context_chars=2000, num_chapters=10)
        ctx = RetrievedContext(
            author_profile=[_make_result("Author note.")],
            character_web=[_make_result("Web.")],
            story_outline=[_make_result('{"title":"x"}')],
            chapter_outline=[_make_result("Chapter.")],
            act_outline=[_make_result("Act beat.")],
        )
        prompt, _ = builder.build_act_generation_prompt(chapter=1, act=1, context=ctx)
        assert "ESTABLISHMENT (this act):" in prompt

    def test_act_two_same_chapter_no_duplicate_establishment_block(self) -> None:
        builder = PromptBuilder(max_context_chars=2000, num_chapters=10)
        ctx = RetrievedContext(
            author_profile=[_make_result("Author note.")],
            character_web=[_make_result("Web.")],
            story_outline=[_make_result('{"title":"x"}')],
            chapter_outline=[_make_result("Chapter.")],
            act_outline=[_make_result("Act beat.")],
        )
        prompt, _ = builder.build_act_generation_prompt(chapter=1, act=2, context=ctx)
        assert "ESTABLISHMENT (this act):" not in prompt

    def test_late_chapter_first_act_no_establishment(self) -> None:
        builder = PromptBuilder(max_context_chars=2000, num_chapters=10)
        ctx = RetrievedContext(
            author_profile=[_make_result("Author note.")],
            character_web=[_make_result("Web.")],
            story_outline=[_make_result('{"title":"x"}')],
            chapter_outline=[_make_result("Chapter.")],
            act_outline=[_make_result("Act beat.")],
        )
        prompt, _ = builder.build_act_generation_prompt(chapter=9, act=1, context=ctx)
        assert "ESTABLISHMENT (this act):" not in prompt


# ---------------------------------------------------------------------------
# Tests for build_rubric_grading_prompt (Req 2.1, 2.2, 2.3, 2.4)
# ---------------------------------------------------------------------------

from romance_factory.story_core.editorial_rules import (
    ALL_RULES,
    CATEGORY_WEIGHTS,
    RuleCategory,
    cliffhanger_editorial_weight_for_profile,
)


class TestBuildRubricGradingPrompt:
    """Unit tests for PromptBuilder.build_rubric_grading_prompt().

    Validates Requirements 2.1, 2.2, 2.3, 2.4.
    """

    def _build(
        self,
        act_text: str = "She looked across the room.",
        chapter: int = 3,
        act: int = 2,
        author_profile: dict | None = None,
        context: RetrievedContext | None = None,
    ) -> tuple[str, str]:
        builder = PromptBuilder(max_context_chars=8000)
        return builder.build_rubric_grading_prompt(
            act_text=act_text,
            chapter=chapter,
            act=act,
            author_profile=author_profile,
            context=context,
        )

    def test_contains_all_rule_ids(self) -> None:
        """Req 2.1: Prompt lists every loaded Editorial_Rules entry by id."""
        prompt, _ = self._build()
        for rule in ALL_RULES:
            assert rule.id in prompt, f"Rule {rule.id} not found in prompt"

    def test_contains_category_weights(self) -> None:
        """Req 2.2: Prompt includes Category_Weights."""
        prompt, _ = self._build()
        for cat, weight in CATEGORY_WEIGHTS.items():
            assert f"{weight:.0%}" in prompt, (
                f"Category weight {weight:.0%} for {cat.value} not in prompt"
            )

    def test_instructs_json_response_format(self) -> None:
        """Req 2.3: Prompt instructs LLM to return JSON with rule_id, score, notes."""
        prompt, _ = self._build()
        assert "rule_id" in prompt
        assert "score" in prompt
        assert "notes" in prompt
        assert "JSON" in prompt

    def test_includes_act_text(self) -> None:
        """Prompt includes the act text."""
        act_text = "The moonlight danced across her face."
        prompt, _ = self._build(act_text=act_text)
        assert act_text in prompt

    def test_includes_chapter_and_act_numbers(self) -> None:
        """Prompt includes chapter and act numbers."""
        prompt, _ = self._build(chapter=5, act=3)
        assert "chapter 5" in prompt.lower()
        assert "act 3" in prompt.lower()

    def test_includes_eng06_weight_with_author_profile(self) -> None:
        """Req 2.4: Prompt includes effective ENG-06 weight from author_profile."""
        profile = {"cliffhanger_editorial_weight": 0.30}
        prompt, _ = self._build(author_profile=profile)
        expected_weight = cliffhanger_editorial_weight_for_profile(profile)
        assert f"{expected_weight:.2f}" in prompt

    def test_includes_eng06_weight_without_author_profile(self) -> None:
        """ENG-06 weight section present even without author_profile."""
        prompt, _ = self._build(author_profile=None)
        assert "ENG-06" in prompt

    def test_returns_tuple_of_strings(self) -> None:
        """Method returns (prompt, system_prompt) tuple."""
        result = self._build()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_system_prompt_mentions_editor_role(self) -> None:
        """System prompt establishes the editor persona."""
        _, system_prompt = self._build()
        assert "editor" in system_prompt.lower()

    def test_includes_context_when_provided(self) -> None:
        """Prompt includes author profile and act outline from context."""
        context = RetrievedContext(
            author_profile=[_make_result("Author writes steamy romance")],
            act_outline=[_make_result("Act outline: tension builds")],
        )
        prompt, _ = self._build(context=context)
        assert "Author writes steamy romance" in prompt
        assert "Act outline: tension builds" in prompt

    def test_no_context_sections_when_none(self) -> None:
        """Prompt works fine without context."""
        prompt, _ = self._build(context=None)
        assert "## Act Text" in prompt
        assert "## Editorial Rubric" in prompt
