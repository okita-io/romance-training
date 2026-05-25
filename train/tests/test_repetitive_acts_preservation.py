"""Preservation property tests for the repetitive acts fix.

**Property 2: Preservation** — Non-Previous-Acts Behavior Unchanged

These tests MUST PASS on the current UNFIXED code. They establish baseline
behavior that must be preserved after the fix is applied.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from romance_factory.generate.agents.act_generation import ActGenerationAgent
from romance_factory.generate.config_v2 import V2Config
from romance_factory.generate.models import (
    DocumentMetadata,
    EditorialIssue,
    RetrievalResult,
    RetrievedContext,
)
from romance_factory.generate.prompt_builder import PromptBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(text: str = "Some context text.", **meta_kwargs) -> RetrievalResult:
    defaults = dict(type="act", chapter=1, act=1)
    defaults.update(meta_kwargs)
    return RetrievalResult(
        text=text,
        metadata=DocumentMetadata(**defaults),
        similarity_score=0.9,
    )


def _build_mock_engine():
    mock_engine = MagicMock()

    def _query_side_effect(collection, query_text, **kwargs):
        return [
            RetrievalResult(
                text=f"Result for {collection}",
                metadata=DocumentMetadata(type=collection),
                similarity_score=0.9,
            )
        ]

    mock_engine.query.side_effect = _query_side_effect
    return mock_engine


# ---------------------------------------------------------------------------
# Test 1 — Non-Act Collection Query Preservation
# ---------------------------------------------------------------------------


class TestNonActCollectionQueryPreservation:
    """For any (chapter, act) pair, queries to author_profile, character_web,
    story_outline, chapter_outline, act_outline, foreshadowing, and
    relationship_arcs must use identical arguments as the unfixed code.

    **Validates: Requirements 3.1**

    On UNFIXED code this MUST PASS — it captures the existing query behavior.
    """

    @given(
        chapter=st.integers(min_value=1, max_value=20),
        act=st.integers(min_value=1, max_value=7),
    )
    @settings(max_examples=50)
    def test_non_act_collection_queries_unchanged(
        self, chapter: int, act: int
    ) -> None:
        """All non-'acts' collection queries use the expected arguments."""
        mock_engine = _build_mock_engine()
        config = V2Config()
        builder = PromptBuilder(max_context_chars=8000)
        agent = ActGenerationAgent(
            lancedb_engine=mock_engine,
            prompt_builder=builder,
            config=config,
        )

        agent._retrieve_context(chapter, act)

        query_text = f"chapter {chapter} act {act}"

        # Collect all calls by collection name
        calls_by_collection: dict[str, list] = {}
        for c in mock_engine.query.call_args_list:
            coll = c[0][0]
            calls_by_collection.setdefault(coll, []).append(c)

        # --- author_profile ---
        ap_calls = calls_by_collection.get("author_profile", [])
        assert len(ap_calls) == 1
        ap = ap_calls[0]
        assert ap[0][1] == query_text
        assert ap[1].get("top_k") == config.retrieval_top_k_author_profile
        assert ap[1].get("metadata_filters") is None

        # --- character_web (character_web + relationship_arcs) ---
        cw_calls = calls_by_collection.get("character_web", [])
        assert len(cw_calls) == 2
        cw0 = cw_calls[0]
        assert cw0[0][1] == query_text
        assert cw0[1].get("top_k") == config.retrieval_top_k_character_web
        assert cw0[1].get("metadata_filters") is None
        cw1 = cw_calls[1]
        assert cw1[0][1] == f"relationship arcs chapter {chapter}"
        assert cw1[1].get("top_k") == config.retrieval_top_k_relationship_arcs
        assert cw1[1].get("metadata_filters") is None

        # --- story_outline (story_outline + chapter_outline) ---
        so_calls = calls_by_collection.get("story_outline", [])
        assert len(so_calls) == 2
        so0 = so_calls[0]
        assert so0[0][1] == query_text
        assert so0[1].get("metadata_filters") == {"type": "outline"}
        assert so0[1].get("top_k") == 1
        so1 = so_calls[1]
        assert so1[0][1] == query_text
        assert so1[1].get("metadata_filters") == {"chapter": chapter}
        assert so1[1].get("top_k") == 1

        # --- story_beats (act_outline + world_context_block + foreshadowing) ---
        sb_calls = calls_by_collection.get("story_beats", [])
        assert len(sb_calls) == 3
        sb0 = sb_calls[0]
        assert sb0[0][1] == query_text
        assert sb0[1].get("metadata_filters") == {"chapter": chapter, "act": act}
        assert sb0[1].get("top_k") == 1
        sb1 = sb_calls[1]
        assert sb1[0][1] == query_text
        assert sb1[1].get("metadata_filters") == {
            "chapter": chapter,
            "act": act,
            "type": "world_context_block",
        }
        assert sb1[1].get("top_k") == 1
        sb2 = sb_calls[2]
        assert sb2[0][1] == f"foreshadowing chapter {chapter}"
        assert sb2[1].get("top_k") == config.retrieval_top_k_foreshadowing
        assert sb2[1].get("metadata_filters") is None


# ---------------------------------------------------------------------------
# Test 2 — First Act Preservation
# ---------------------------------------------------------------------------


class TestFirstActPreservation:
    """For act=1 of any chapter, act generation proceeds normally.

    **Validates: Requirements 3.2**

    On UNFIXED code this MUST PASS.
    """

    @given(
        chapter=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=50)
    def test_first_act_retrieval_proceeds_normally(
        self, chapter: int
    ) -> None:
        """_retrieve_context(chapter, 1) completes and returns a valid
        RetrievedContext."""
        mock_engine = _build_mock_engine()
        config = V2Config()
        builder = PromptBuilder(max_context_chars=8000)
        agent = ActGenerationAgent(
            lancedb_engine=mock_engine,
            prompt_builder=builder,
            config=config,
        )

        context = agent._retrieve_context(chapter, 1)

        assert isinstance(context, RetrievedContext)
        assert isinstance(context.previous_acts, list)
        assert len(context.author_profile) >= 0
        assert len(context.character_web) >= 0

    @given(
        chapter=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=50)
    def test_first_act_prompt_builds_successfully(
        self, chapter: int
    ) -> None:
        """build_act_generation_prompt(chapter, 1, context) succeeds with
        empty previous_acts."""
        context = RetrievedContext(
            author_profile=[_make_result("Author profile text.")],
            character_web=[_make_result("Character web text.")],
            story_outline=[_make_result("Story outline text.")],
            chapter_outline=[_make_result("Chapter outline text.")],
            act_outline=[_make_result("Act outline text.")],
            foreshadowing=[_make_result("Foreshadowing text.")],
            relationship_arcs=[_make_result("Relationship arcs text.")],
            previous_acts=[],
        )

        builder = PromptBuilder(max_context_chars=8000)
        prompt, system_prompt = builder.build_act_generation_prompt(
            chapter=chapter, act=1, context=context,
        )

        assert isinstance(prompt, str) and len(prompt) > 0
        assert isinstance(system_prompt, str) and len(system_prompt) > 0
        assert f"act 1 of chapter {chapter}" in prompt.lower()


# ---------------------------------------------------------------------------
# Test 3 — Non-Previous-Acts Formatting Preservation
# ---------------------------------------------------------------------------


_varied_metadata_result = st.builds(
    RetrievalResult,
    text=st.text(min_size=1, max_size=200),
    metadata=st.builds(
        DocumentMetadata,
        type=st.sampled_from(["act", "outline", "beat", "character"]),
        chapter=st.integers(min_value=0, max_value=20),
        act=st.integers(min_value=0, max_value=7),
        characters_involved=st.lists(
            st.text(
                min_size=1, max_size=15,
                alphabet=st.characters(whitelist_categories=("L",)),
            ),
            max_size=3,
        ),
        emotional_tone=st.sampled_from(["tension", "joy", "sorrow", ""]),
        plot_function=st.sampled_from(["rising_action", "climax", ""]),
        summary=st.text(min_size=0, max_size=50),
    ),
    similarity_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)


class TestNonPreviousActsFormattingPreservation:
    """_format_context_section(title, results) for titles other than
    "Previous Acts" must produce identical output before and after fix.

    The unfixed behavior: concatenate r.text with "\\n\\n", truncate,
    wrap with "## {title}\\n\\n{body}".

    **Validates: Requirements 3.3**

    On UNFIXED code this MUST PASS.
    """

    @given(
        title=st.sampled_from([
            "Author Profile",
            "Character Web",
            "Story Outline",
            "Chapter Outline",
            "Act Outline",
            "Foreshadowing",
            "Relationship Arcs",
        ]),
        results=st.lists(_varied_metadata_result, min_size=1, max_size=5),
    )
    @settings(max_examples=50)
    def test_format_context_section_output_matches_expected(
        self, title: str, results: list[RetrievalResult]
    ) -> None:
        """_format_context_section produces the expected format."""
        builder = PromptBuilder(max_context_chars=8000)
        output = builder._format_context_section(title, results)

        # Reconstruct expected output from the known unfixed behavior
        parts = [r.text for r in results]
        body = "\n\n".join(parts)
        body = builder._truncate(body)
        expected = f"## {title}\n\n{body}"

        assert output == expected

    @given(
        title=st.sampled_from([
            "Author Profile",
            "Character Web",
            "Story Outline",
            "Chapter Outline",
            "Act Outline",
            "Foreshadowing",
            "Relationship Arcs",
        ]),
    )
    @settings(max_examples=20)
    def test_format_context_section_empty_returns_empty(
        self, title: str
    ) -> None:
        """_format_context_section returns empty string for empty results."""
        builder = PromptBuilder(max_context_chars=8000)
        output = builder._format_context_section(title, [])
        assert output == ""


# ---------------------------------------------------------------------------
# Test 4 — Validation/Rewrite/Cleanup Prompt Preservation
# ---------------------------------------------------------------------------


class TestPromptPreservation:
    """build_act_validation_prompt(), build_act_rewrite_prompt(), and
    build_chapter_cleanup_prompt() must produce identical outputs to the
    unfixed code when called with the same inputs.

    **Validates: Requirements 3.4**

    On UNFIXED code this MUST PASS.
    """

    @given(
        chapter=st.integers(min_value=1, max_value=20),
        act=st.integers(min_value=1, max_value=7),
        act_text=st.text(min_size=10, max_size=300),
    )
    @settings(max_examples=50)
    def test_validation_prompt_structure_preserved(
        self, chapter: int, act: int, act_text: str
    ) -> None:
        """build_act_validation_prompt produces expected structure."""
        context = RetrievedContext(
            act_outline=[_make_result("Act outline.")],
            chapter_outline=[_make_result("Chapter outline.")],
            story_outline=[_make_result("Story outline.")],
            character_web=[_make_result("Character web.")],
            foreshadowing=[_make_result("Foreshadowing.")],
            previous_acts=[
                _make_result("Previous act.", chapter=chapter, act=max(1, act - 1)),
            ],
            author_profile=[_make_result("Author profile.")],
        )

        builder = PromptBuilder(max_context_chars=8000)
        prompt, system_prompt = builder.build_act_validation_prompt(
            chapter=chapter, act=act, act_text=act_text, context=context,
        )

        assert f"Act Text (Chapter {chapter}, Act {act})" in prompt
        assert "## Instructions" in prompt
        assert "## Act Outline" in prompt
        assert "## Chapter Outline" in prompt
        assert "## Story Outline" in prompt
        assert "## Character Web" in prompt
        assert "## Foreshadowing" in prompt
        # Optional when context.world / context.world_outline are empty:
        # "## World Lore (retrieved)", "## Scene World Context Block"
        assert "## Previous Acts" in prompt
        assert "## Author Profile" in prompt
        assert "reviewer" in system_prompt.lower() or "editorial" in system_prompt.lower()

    @given(
        chapter=st.integers(min_value=1, max_value=20),
        act=st.integers(min_value=1, max_value=7),
        original_text=st.text(min_size=10, max_size=300),
    )
    @settings(max_examples=50)
    def test_rewrite_prompt_structure_preserved(
        self, chapter: int, act: int, original_text: str
    ) -> None:
        """build_act_rewrite_prompt produces expected structure."""
        context = RetrievedContext(
            author_profile=[_make_result("Author profile.")],
            character_web=[_make_result("Character web.")],
            act_outline=[_make_result("Act outline.")],
            previous_acts=[
                _make_result("Previous act.", chapter=chapter, act=max(1, act - 1)),
            ],
        )
        issues = [
            EditorialIssue(
                type="continuity",
                severity="MAJOR",
                location="paragraph 2",
                explanation="Timeline issue",
                suggested_fix="Fix the timeline",
            )
        ]

        builder = PromptBuilder(max_context_chars=8000)
        prompt, system_prompt = builder.build_act_rewrite_prompt(
            chapter=chapter,
            act=act,
            original_text=original_text,
            issues=issues,
            rewrite_plan="Fix continuity issues.",
            context=context,
        )

        assert "## Original Text" in prompt
        assert "## Editorial issues" in prompt
        assert "Timeline issue" in prompt
        assert "Fix the timeline" in prompt
        assert "## Author Profile" in prompt
        assert "## Character Web" in prompt
        assert "## Act Outline" in prompt
        assert "## Previous Acts" in prompt
        assert "continuity" in prompt.lower()
        assert "prose" in system_prompt.lower()

    @given(
        chapter=st.integers(min_value=1, max_value=20),
        chapter_text=st.text(min_size=10, max_size=300),
    )
    @settings(max_examples=50)
    def test_cleanup_prompt_structure_preserved(
        self, chapter: int, chapter_text: str
    ) -> None:
        """build_chapter_cleanup_prompt produces expected structure."""
        author_profile = {"voice": "warm and witty", "tropes": "slow burn"}

        builder = PromptBuilder(max_context_chars=8000)
        prompt, system_prompt = builder.build_chapter_cleanup_prompt(
            chapter=chapter,
            chapter_text=chapter_text,
            author_profile=author_profile,
        )

        assert f"## Chapter {chapter} Text" in prompt
        assert "## Cleanup Rules" in prompt
        assert "## Author Style" in prompt
        assert "mojibake" in prompt.lower()
        assert "cleanup" in system_prompt.lower() or "polish" in system_prompt.lower()
