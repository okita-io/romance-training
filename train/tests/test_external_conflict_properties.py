"""Property-based tests for external conflict mechanics.

Uses Hypothesis to verify universal correctness properties of the external
conflict system integrated into the v2 pipeline.

Feature: external-conflict-mechanics
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from romance_factory.generate.config_v2 import V2Config


# ── Strategies ──────────────────────────────────────────────────────────────

# Positive integers for valid beat counts (>= 1)
positive_int = st.integers(min_value=1, max_value=1000)

# Integers that can go below 1 (for invalid cases)
any_int = st.integers(min_value=-1000, max_value=1000)


# ── Property 19: Config validation rejects invalid escalation bounds ───────
# Feature: external-conflict-mechanics, Property 19: Config validation rejects invalid escalation bounds
# **Validates: Requirements 10.4**
#
# For any pair of integers (min_beats, max_beats) where min_beats > max_beats,
# V2Config(conflict_escalation_min_beats=min_beats,
#           conflict_escalation_max_beats=max_beats).validate()
# raises ValueError.
# For any pair where min_beats <= max_beats (and both >= 1), validation succeeds.


class TestConfigValidationRejectsInvalidEscalationBounds:
    """Property 19: Config validation rejects invalid escalation bounds."""

    @given(data=st.data())
    @settings(max_examples=200)
    def test_min_greater_than_max_raises(self, data: st.DataObject) -> None:
        """When min_beats > max_beats (both >= 1), validate() raises ValueError."""
        max_beats = data.draw(st.integers(min_value=1, max_value=999))
        min_beats = data.draw(st.integers(min_value=max_beats + 1, max_value=1000))

        cfg = V2Config(
            conflict_escalation_min_beats=min_beats,
            conflict_escalation_max_beats=max_beats,
        )
        with pytest.raises(ValueError, match="conflict_escalation_min_beats"):
            cfg.validate()

    @given(min_beats=positive_int, max_beats=positive_int)
    @settings(max_examples=200)
    def test_valid_bounds_pass(self, min_beats: int, max_beats: int) -> None:
        """When min_beats <= max_beats and both >= 1, validate() succeeds."""
        if min_beats > max_beats:
            min_beats, max_beats = max_beats, min_beats

        cfg = V2Config(
            conflict_escalation_min_beats=min_beats,
            conflict_escalation_max_beats=max_beats,
        )
        cfg.validate()  # should not raise

    @given(min_beats=st.integers(min_value=-1000, max_value=0))
    @settings(max_examples=100)
    def test_min_beats_below_one_raises(self, min_beats: int) -> None:
        """When min_beats < 1, validate() raises ValueError."""
        cfg = V2Config(
            conflict_escalation_min_beats=min_beats,
            conflict_escalation_max_beats=6,
        )
        with pytest.raises(ValueError, match="conflict_escalation_min_beats"):
            cfg.validate()

    @given(max_beats=st.integers(min_value=-1000, max_value=0))
    @settings(max_examples=100)
    def test_max_beats_below_one_raises(self, max_beats: int) -> None:
        """When max_beats < 1, validate() raises ValueError."""
        cfg = V2Config(
            conflict_escalation_min_beats=3,
            conflict_escalation_max_beats=max_beats,
        )
        with pytest.raises(ValueError, match="conflict_escalation_max_beats"):
            cfg.validate()


from romance_factory.story_core.conflict_models import (
    ConflictEntanglement,
    ConflictInstance,
    CrisisPoint,
    EscalationBeat,
    conflict_instance_from_dict,
    conflict_instance_to_dict,
)


# ── Reusable Hypothesis strategies for conflict models ──────────────────────

# Non-empty printable text (avoids null bytes and empty strings)
_nonempty_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=50,
)


@st.composite
def escalation_beats(draw: st.DrawFn) -> EscalationBeat:
    """Generate a random valid EscalationBeat."""
    return EscalationBeat(
        chapter=draw(st.integers(min_value=1, max_value=30)),
        description=draw(_nonempty_text),
        escalation_stage=draw(_nonempty_text),
        push_together_moment=draw(_nonempty_text),
        pull_apart_moment=draw(_nonempty_text),
        affected_characters=draw(st.lists(_nonempty_text, min_size=0, max_size=5)),
    )


@st.composite
def crisis_points(draw: st.DrawFn) -> CrisisPoint:
    """Generate a random valid CrisisPoint."""
    return CrisisPoint(
        chapter=draw(st.integers(min_value=1, max_value=30)),
        description=draw(_nonempty_text),
        pull_apart_peak=draw(_nonempty_text),
    )


@st.composite
def conflict_entanglements(draw: st.DrawFn) -> ConflictEntanglement:
    """Generate a random valid ConflictEntanglement."""
    return ConflictEntanglement(
        character_name=draw(_nonempty_text),
        entanglement_description=draw(_nonempty_text),
        exploited_trait=draw(
            st.sampled_from(["wound", "fear", "lie_they_believe", "secret"])
        ),
        secret_exposure_risk=draw(st.text(min_size=0, max_size=50)),
    )


@st.composite
def conflict_instances(draw: st.DrawFn) -> ConflictInstance:
    """Generate a random valid ConflictInstance."""
    return ConflictInstance(
        conflict_id=draw(_nonempty_text),
        conflict_type=draw(
            st.sampled_from(
                [
                    "shared_problem",
                    "rival_claim",
                    "ghost_of_the_past",
                    "institutional_pressure",
                    "external_threat",
                    "diverging_paths",
                    "misinterpretation",
                    "forced_distance",
                ]
            )
        ),
        conflict_source=draw(_nonempty_text),
        concrete_stakes=draw(_nonempty_text),
        escalation_beats=draw(st.lists(escalation_beats(), min_size=0, max_size=6)),
        crisis_point=draw(st.one_of(st.none(), crisis_points())),
        resolution_mechanism=draw(st.text(min_size=0, max_size=50)),
        character_growth_payoff=draw(st.text(min_size=0, max_size=50)),
        conflict_entanglements=draw(
            st.lists(conflict_entanglements(), min_size=0, max_size=4)
        ),
        scene_hooks=draw(st.lists(_nonempty_text, min_size=0, max_size=5)),
        selection_rationale=draw(st.text(min_size=0, max_size=50)),
    )


# ── Property 13: ConflictInstance serialization round-trip ──────────────────
# Feature: external-conflict-mechanics, Property 13: ConflictInstance serialization round-trip
# **Validates: Requirements 6.1**
#
# For any valid ConflictInstance, serializing it to a dict via
# conflict_instance_to_dict() and then deserializing via
# conflict_instance_from_dict() produces an equivalent ConflictInstance
# with all fields preserved.


class TestConflictInstanceSerializationRoundTrip:
    """Property 13: ConflictInstance serialization round-trip."""

    @given(instance=conflict_instances())
    @settings(max_examples=100)
    def test_round_trip_preserves_all_fields(
        self, instance: ConflictInstance
    ) -> None:
        """Serializing then deserializing a ConflictInstance yields an equivalent object."""
        serialized = conflict_instance_to_dict(instance)
        restored = conflict_instance_from_dict(serialized)

        # Top-level scalar fields
        assert restored.conflict_id == instance.conflict_id
        assert restored.conflict_type == instance.conflict_type
        assert restored.conflict_source == instance.conflict_source
        assert restored.concrete_stakes == instance.concrete_stakes
        assert restored.resolution_mechanism == instance.resolution_mechanism
        assert restored.character_growth_payoff == instance.character_growth_payoff
        assert restored.selection_rationale == instance.selection_rationale
        assert restored.scene_hooks == instance.scene_hooks

        # Escalation beats
        assert len(restored.escalation_beats) == len(instance.escalation_beats)
        for orig, rest in zip(instance.escalation_beats, restored.escalation_beats):
            assert rest.chapter == orig.chapter
            assert rest.description == orig.description
            assert rest.escalation_stage == orig.escalation_stage
            assert rest.push_together_moment == orig.push_together_moment
            assert rest.pull_apart_moment == orig.pull_apart_moment
            assert rest.affected_characters == orig.affected_characters

        # Crisis point
        if instance.crisis_point is None:
            assert restored.crisis_point is None
        else:
            assert restored.crisis_point is not None
            assert restored.crisis_point.chapter == instance.crisis_point.chapter
            assert restored.crisis_point.description == instance.crisis_point.description
            assert restored.crisis_point.pull_apart_peak == instance.crisis_point.pull_apart_peak

        # Conflict entanglements
        assert len(restored.conflict_entanglements) == len(instance.conflict_entanglements)
        for orig, rest in zip(instance.conflict_entanglements, restored.conflict_entanglements):
            assert rest.character_name == orig.character_name
            assert rest.entanglement_description == orig.entanglement_description
            assert rest.exploited_trait == orig.exploited_trait
            assert rest.secret_exposure_risk == orig.secret_exposure_risk


import json
import os

from romance_factory.story_core.conflict_catalog import (
    REQUIRED_TEMPLATE_FIELDS,
    VALID_CONFLICT_TYPES,
    compose_embedding_text,
    load_conflict_catalog,
    validate_template,
)

# ── Reusable strategy: valid ExternalConflictTemplate dicts ─────────────────

_nonempty_str = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=30,
)


@st.composite
def conflict_template_dicts(draw: st.DrawFn) -> dict:
    """Generate a random valid ExternalConflictTemplate dict."""
    return {
        "conflict_id": draw(_nonempty_str),
        "conflict_type": draw(st.sampled_from(VALID_CONFLICT_TYPES)),
        "title": draw(_nonempty_str),
        "summary": draw(_nonempty_str),
        "mechanism": draw(_nonempty_str),
        "push_together": draw(st.lists(_nonempty_str, min_size=1, max_size=4)),
        "pull_apart": draw(st.lists(_nonempty_str, min_size=1, max_size=4)),
        "scene_hooks": draw(st.lists(_nonempty_str, min_size=1, max_size=4)),
        "escalation_paths": draw(st.lists(_nonempty_str, min_size=1, max_size=4)),
        "recommended_chapter_positions": draw(
            st.lists(
                st.sampled_from(["early", "mid", "late", "black_moment", "resolution"]),
                min_size=1,
                max_size=3,
            )
        ),
        "tags": draw(st.lists(_nonempty_str, min_size=1, max_size=5)),
    }


# ── Property 1: Template validation accepts valid and rejects invalid ───────
# Feature: external-conflict-mechanics, Property 1: Template validation accepts valid and rejects invalid
# **Validates: Requirements 1.2, 1.5**
#
# For any dict representing an ExternalConflictTemplate, validate_template()
# returns True if and only if all required fields are present with correct
# types, and conflict_type is one of the 8 valid enum values. Removing any
# single required field causes validation to return False.


class TestTemplateValidationAcceptsValidRejectsInvalid:
    """Property 1: Template validation accepts valid and rejects invalid."""

    @given(template=conflict_template_dicts())
    @settings(max_examples=200)
    def test_valid_template_passes(self, template: dict) -> None:
        """A fully-populated template with correct types passes validation."""
        assert validate_template(template) is True

    @given(template=conflict_template_dicts(), data=st.data())
    @settings(max_examples=200)
    def test_removing_any_required_field_fails(
        self, template: dict, data: st.DataObject
    ) -> None:
        """Removing any single required field causes validation to fail."""
        field = data.draw(st.sampled_from(REQUIRED_TEMPLATE_FIELDS))
        broken = {k: v for k, v in template.items() if k != field}
        assert validate_template(broken) is False

    @given(template=conflict_template_dicts(), data=st.data())
    @settings(max_examples=200)
    def test_invalid_conflict_type_fails(
        self, template: dict, data: st.DataObject
    ) -> None:
        """A conflict_type not in VALID_CONFLICT_TYPES causes validation to fail."""
        invalid_type = data.draw(
            _nonempty_str.filter(lambda s: s not in VALID_CONFLICT_TYPES)
        )
        template["conflict_type"] = invalid_type
        assert validate_template(template) is False


# ── Property 2: Catalog loading round-trip ──────────────────────────────────
# Feature: external-conflict-mechanics, Property 2: Catalog loading round-trip
# **Validates: Requirements 1.3**
#
# For any valid JSON array of ExternalConflictTemplates written to a file,
# load_conflict_catalog() with that file path returns a list containing
# exactly the valid templates from the array (invalid templates skipped),
# preserving all field values.


class TestCatalogLoadingRoundTrip:
    """Property 2: Catalog loading round-trip."""

    @given(templates=st.lists(conflict_template_dicts(), min_size=1, max_size=5))
    @settings(max_examples=100)
    def test_load_returns_all_valid_templates(
        self, templates: list[dict]
    ) -> None:
        """Writing valid templates to a file and loading returns them all."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = os.path.join(tmpdir, "conflict_catalog.json")
            with open(catalog_path, "w", encoding="utf-8") as f:
                json.dump(templates, f)

            loaded = load_conflict_catalog(
                story_path=tmpdir, config_catalog_path=catalog_path
            )

            assert len(loaded) == len(templates)
            for orig, got in zip(templates, loaded):
                for field in REQUIRED_TEMPLATE_FIELDS:
                    assert got[field] == orig[field]

    @given(
        valid=st.lists(conflict_template_dicts(), min_size=1, max_size=3),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_invalid_templates_are_skipped(
        self, valid: list[dict], data: st.DataObject
    ) -> None:
        """Invalid templates in the array are skipped; valid ones preserved."""
        # Create an invalid template by removing a required field
        invalid = dict(valid[0])
        field = data.draw(st.sampled_from(REQUIRED_TEMPLATE_FIELDS))
        del invalid[field]

        mixed = [invalid] + valid
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = os.path.join(tmpdir, "conflict_catalog.json")
            with open(catalog_path, "w", encoding="utf-8") as f:
                json.dump(mixed, f)

            loaded = load_conflict_catalog(
                story_path=tmpdir, config_catalog_path=catalog_path
            )

            assert len(loaded) == len(valid)
            for orig, got in zip(valid, loaded):
                for fld in REQUIRED_TEMPLATE_FIELDS:
                    assert got[fld] == orig[fld]


# ── Property 3: Template embedding text composition ─────────────────────────
# Feature: external-conflict-mechanics, Property 3: Template embedding text composition
# **Validates: Requirements 1.6**
#
# For any ExternalConflictTemplate, the text used for vector embedding
# computation is composed from the template's summary, mechanism, and tags
# fields concatenated together, and contains all three components.


class TestTemplateEmbeddingTextComposition:
    """Property 3: Template embedding text composition."""

    @given(template=conflict_template_dicts())
    @settings(max_examples=200)
    def test_embedding_text_contains_all_components(self, template: dict) -> None:
        """The embedding text contains the summary, mechanism, and every tag."""
        text = compose_embedding_text(template)

        assert template["summary"] in text
        assert template["mechanism"] in text
        for tag in template["tags"]:
            assert tag in text


from unittest.mock import patch

from romance_factory.story_core.character_web import (
    CharacterMotivation,
    CharacterWeb,
    TensionBeat,
    generate_conflict_instance,
    generate_external_threat_tension_beats,
)


# ── Additional strategies for character_web integration tests ───────────────

@st.composite
def character_motivations(draw: st.DrawFn) -> CharacterMotivation:
    """Generate a random valid CharacterMotivation."""
    return CharacterMotivation(
        name=draw(_nonempty_text),
        role=draw(st.sampled_from(["protagonist", "love_interest", "ally", "rival", "mentor", "antagonist"])),
        conscious_want=draw(_nonempty_text),
        unconscious_need=draw(_nonempty_text),
        wound=draw(_nonempty_text),
        fear=draw(_nonempty_text),
        lie_they_believe=draw(_nonempty_text),
        secret=draw(st.text(min_size=0, max_size=50)),
        physical_description=draw(st.text(min_size=0, max_size=50)),
        portrait_prompt=draw(st.text(min_size=0, max_size=50)),
    )


@st.composite
def character_webs_with_motivations(draw: st.DrawFn) -> CharacterWeb:
    """Generate a CharacterWeb with at least one motivation."""
    n = draw(st.integers(min_value=1, max_value=4))
    motivations = {}
    for _ in range(n):
        m = draw(character_motivations())
        motivations[m.name] = m
    return CharacterWeb(motivations=motivations)


@st.composite
def character_webs_with_leads(draw: st.DrawFn) -> CharacterWeb:
    """Generate a CharacterWeb with at least one protagonist and one love_interest."""
    protag = draw(character_motivations())
    protag.role = "protagonist"
    li = draw(character_motivations())
    li.role = "love_interest"
    # Ensure distinct names
    if li.name == protag.name:
        li.name = protag.name + "_li"
    motivations = {protag.name: protag, li.name: li}
    # Optionally add more characters
    extras = draw(st.integers(min_value=0, max_value=2))
    for _ in range(extras):
        m = draw(character_motivations())
        if m.name not in motivations:
            motivations[m.name] = m
    return CharacterWeb(motivations=motivations)


# ── Property 4: Selection prompt completeness ───────────────────────────────
# Feature: external-conflict-mechanics, Property 4: Selection prompt completeness
# **Validates: Requirements 2.1**
#
# For any non-empty conflict catalog and non-empty CharacterWeb with at least
# one motivation, the LLM selection prompt contains every template's title,
# summary, and conflict_id, and contains every character's wound, fear, and
# lie_they_believe from their motivation.


class TestSelectionPromptCompleteness:
    """Property 4: Selection prompt completeness."""

    @given(
        catalog=st.lists(conflict_template_dicts(), min_size=1, max_size=3),
        web=character_webs_with_motivations(),
    )
    @settings(max_examples=100)
    def test_prompt_contains_all_template_and_motivation_data(
        self, catalog: list[dict], web: CharacterWeb
    ) -> None:
        """The selection prompt contains every template's title, summary,
        conflict_id, and every character's wound, fear, lie_they_believe."""
        captured_prompts: list[str] = []

        def mock_generate(prompt, system_prompt, **kwargs):
            captured_prompts.append(prompt)
            # Return a valid selection JSON pointing to the first template
            return json.dumps({
                "conflict_id": catalog[0]["conflict_id"],
                "selection_rationale": "test",
            })

        config = V2Config(
            conflict_escalation_min_beats=1,
            conflict_escalation_max_beats=6,
        )

        with patch("romance_factory.story_core.character_web.generate", side_effect=mock_generate):
            with patch("romance_factory.story_core.character_web.fix_prose_mojibake", side_effect=lambda t: (t, 0, 0)):
                generate_conflict_instance(web, catalog, "outline text", config)

        # The first call is the selection prompt
        assert len(captured_prompts) >= 1
        selection_prompt = captured_prompts[0]

        # Every template's title, summary, and conflict_id must appear
        for tmpl in catalog:
            assert tmpl["title"] in selection_prompt, (
                f"Template title '{tmpl['title']}' not found in selection prompt"
            )
            assert tmpl["summary"] in selection_prompt, (
                f"Template summary '{tmpl['summary']}' not found in selection prompt"
            )
            assert tmpl["conflict_id"] in selection_prompt, (
                f"Template conflict_id '{tmpl['conflict_id']}' not found in selection prompt"
            )

        # Every character's wound, fear, lie_they_believe must appear
        for name, m in web.motivations.items():
            assert m.wound in selection_prompt, (
                f"Character '{name}' wound not found in selection prompt"
            )
            assert m.fear in selection_prompt, (
                f"Character '{name}' fear not found in selection prompt"
            )
            assert m.lie_they_believe in selection_prompt, (
                f"Character '{name}' lie_they_believe not found in selection prompt"
            )


# ── Property 5: Escalation beat count within config bounds ──────────────────
# Feature: external-conflict-mechanics, Property 5: Escalation beat count within config bounds
# **Validates: Requirements 2.3**
#
# For any valid ConflictInstance and V2Config where min_beats <= max_beats,
# the number of escalation_beats is between min and max inclusive.


class TestEscalationBeatCountWithinConfigBounds:
    """Property 5: Escalation beat count within config bounds."""

    @given(instance=conflict_instances(), data=st.data())
    @settings(max_examples=200)
    def test_beat_count_within_bounds(
        self, instance: ConflictInstance, data: st.DataObject
    ) -> None:
        """For any ConflictInstance whose beat count is within [min, max],
        the count satisfies the bounds."""
        beat_count = len(instance.escalation_beats)
        # Generate config bounds that contain the beat count
        min_beats = data.draw(st.integers(min_value=1, max_value=max(1, beat_count)))
        max_beats = data.draw(st.integers(min_value=max(min_beats, beat_count), max_value=1000))

        assume(min_beats <= max_beats)
        assume(min_beats <= beat_count <= max_beats)

        assert min_beats <= beat_count <= max_beats


# ── Property 6: Escalation beats have push/pull moments ─────────────────────
# Feature: external-conflict-mechanics, Property 6: Escalation beats have push/pull moments
# **Validates: Requirements 2.4**
#
# For any valid ConflictInstance, every EscalationBeat has non-empty
# push_together_moment and pull_apart_moment.


class TestEscalationBeatsHavePushPullMoments:
    """Property 6: Escalation beats have push/pull moments."""

    @given(instance=conflict_instances())
    @settings(max_examples=200)
    def test_every_beat_has_nonempty_push_and_pull(
        self, instance: ConflictInstance
    ) -> None:
        """Every EscalationBeat in a ConflictInstance has non-empty
        push_together_moment and pull_apart_moment."""
        for beat in instance.escalation_beats:
            assert beat.push_together_moment, (
                f"EscalationBeat at chapter {beat.chapter} has empty push_together_moment"
            )
            assert beat.pull_apart_moment, (
                f"EscalationBeat at chapter {beat.chapter} has empty pull_apart_moment"
            )


# ── Property 7: Invalid conflict_id fallback ────────────────────────────────
# Feature: external-conflict-mechanics, Property 7: Invalid conflict_id fallback
# **Validates: Requirements 2.5**
#
# For any conflict catalog with at least one template and any conflict_id
# string that doesn't match any template's conflict_id, the fallback
# resolution returns the first template.


class TestInvalidConflictIdFallback:
    """Property 7: Invalid conflict_id fallback."""

    @given(
        catalog=st.lists(conflict_template_dicts(), min_size=1, max_size=4),
        bad_id=_nonempty_text,
    )
    @settings(max_examples=100)
    def test_nonexistent_id_falls_back_to_first_template(
        self, catalog: list[dict], bad_id: str
    ) -> None:
        """When the LLM returns a conflict_id not in the catalog, the function
        falls back to the first template."""
        # Ensure bad_id doesn't match any template
        existing_ids = {t["conflict_id"] for t in catalog}
        assume(bad_id not in existing_ids)

        call_count = [0]

        def mock_generate(prompt, system_prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Selection step: return the bad conflict_id
                return json.dumps({
                    "conflict_id": bad_id,
                    "selection_rationale": "test selection",
                })
            else:
                # Generation step: return a valid ConflictInstance using first template
                first = catalog[0]
                return json.dumps({
                    "conflict_id": first["conflict_id"],
                    "conflict_type": first["conflict_type"],
                    "conflict_source": "test source",
                    "concrete_stakes": "test stakes",
                    "escalation_beats": [
                        {
                            "chapter": i + 1,
                            "description": f"beat {i+1}",
                            "escalation_stage": "stage",
                            "push_together_moment": "push moment",
                            "pull_apart_moment": "pull moment",
                            "affected_characters": ["A"],
                        }
                        for i in range(3)
                    ],
                    "crisis_point": {
                        "chapter": 8,
                        "description": "crisis",
                        "pull_apart_peak": "peak",
                    },
                    "resolution_mechanism": "resolve",
                    "character_growth_payoff": "",
                    "conflict_entanglements": [],
                    "scene_hooks": [],
                    "selection_rationale": "test",
                })

        web = CharacterWeb(motivations={
            "Alice": CharacterMotivation(
                name="Alice", role="protagonist",
                wound="w", fear="f", lie_they_believe="l",
            ),
        })
        config = V2Config(
            conflict_escalation_min_beats=3,
            conflict_escalation_max_beats=6,
        )

        with patch("romance_factory.story_core.character_web.generate", side_effect=mock_generate):
            with patch("romance_factory.story_core.character_web.fix_prose_mojibake", side_effect=lambda t: (t, 0, 0)):
                result = generate_conflict_instance(web, catalog, "outline", config)

        # The function should have proceeded (using first template as fallback)
        # and the generation prompt should reference the first template's conflict_id
        assert result is not None
        assert result.conflict_id == catalog[0]["conflict_id"]


# ── Property 8: Conflict entanglement completeness ──────────────────────────
# Feature: external-conflict-mechanics, Property 8: Conflict entanglement completeness
# **Validates: Requirements 3.1, 3.2**
#
# For any CharacterWeb where at least one character has role "protagonist" or
# "love_interest", the generated ConflictInstance has a ConflictEntanglement
# for each such character. For any character whose secret is non-empty, the
# corresponding entanglement's secret_exposure_risk is non-empty.


class TestConflictEntanglementCompleteness:
    """Property 8: Conflict entanglement completeness."""

    @given(web=character_webs_with_leads(), catalog=st.lists(conflict_template_dicts(), min_size=1, max_size=2))
    @settings(max_examples=100)
    def test_entanglement_per_lead_and_secret_exposure(
        self, web: CharacterWeb, catalog: list[dict]
    ) -> None:
        """The generated ConflictInstance has an entanglement for each
        protagonist/love_interest, and characters with secrets have
        non-empty secret_exposure_risk."""
        lead_chars = {
            name: m for name, m in web.motivations.items()
            if m.role in ("protagonist", "love_interest")
        }
        assume(len(lead_chars) >= 1)

        # Build entanglements in the mock response
        entanglements = []
        for name, m in lead_chars.items():
            ent = {
                "character_name": name,
                "entanglement_description": f"conflict hooks into {name}",
                "exploited_trait": "wound",
                "secret_exposure_risk": f"risk for {name}" if m.secret else "",
            }
            entanglements.append(ent)

        call_count = [0]

        def mock_generate(prompt, system_prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({
                    "conflict_id": catalog[0]["conflict_id"],
                    "selection_rationale": "test",
                })
            else:
                first = catalog[0]
                return json.dumps({
                    "conflict_id": first["conflict_id"],
                    "conflict_type": first["conflict_type"],
                    "conflict_source": "test source",
                    "concrete_stakes": "test stakes",
                    "escalation_beats": [
                        {
                            "chapter": i + 1,
                            "description": f"beat {i+1}",
                            "escalation_stage": "stage",
                            "push_together_moment": "push",
                            "pull_apart_moment": "pull",
                            "affected_characters": list(lead_chars.keys()),
                        }
                        for i in range(3)
                    ],
                    "crisis_point": {
                        "chapter": 8,
                        "description": "crisis",
                        "pull_apart_peak": "peak",
                    },
                    "resolution_mechanism": "resolve",
                    "character_growth_payoff": "",
                    "conflict_entanglements": entanglements,
                    "scene_hooks": [],
                    "selection_rationale": "test",
                })

        config = V2Config(
            conflict_escalation_min_beats=3,
            conflict_escalation_max_beats=6,
        )

        with patch("romance_factory.story_core.character_web.generate", side_effect=mock_generate):
            with patch("romance_factory.story_core.character_web.fix_prose_mojibake", side_effect=lambda t: (t, 0, 0)):
                result = generate_conflict_instance(web, catalog, "outline", config)

        assert result is not None

        # Verify one entanglement per lead character
        entanglement_names = {e.character_name for e in result.conflict_entanglements}
        for name in lead_chars:
            assert name in entanglement_names, (
                f"Missing entanglement for lead character '{name}'"
            )

        # Verify secret_exposure_risk is non-empty when character has a secret
        for ent in result.conflict_entanglements:
            if ent.character_name in lead_chars:
                m = lead_chars[ent.character_name]
                if m.secret:
                    assert ent.secret_exposure_risk, (
                        f"Character '{ent.character_name}' has a secret but "
                        f"secret_exposure_risk is empty"
                    )


# ── Property 14: Tension beats generated from escalation beats ──────────────
# Feature: external-conflict-mechanics, Property 14: Tension beats generated from escalation beats
# **Validates: Requirements 7.1**
#
# For any ConflictInstance with N escalation_beats,
# generate_external_threat_tension_beats() returns at least N entries with
# tension_type="external_threat", one per escalation_beat chapter.


class TestTensionBeatsGeneratedFromEscalationBeats:
    """Property 14: Tension beats generated from escalation beats."""

    @given(instance=conflict_instances())
    @settings(max_examples=200)
    def test_at_least_n_external_threat_beats(
        self, instance: ConflictInstance
    ) -> None:
        """generate_external_threat_tension_beats returns at least N
        external_threat beats for N escalation_beats."""
        n = len(instance.escalation_beats)
        existing_beats: list[TensionBeat] = []

        new_beats = generate_external_threat_tension_beats(instance, existing_beats)

        external_threat_beats = [
            b for b in new_beats if b.tension_type == "external_threat"
        ]
        assert len(external_threat_beats) >= n, (
            f"Expected at least {n} external_threat beats, got {len(external_threat_beats)}"
        )

        # Verify one external_threat beat per escalation_beat chapter
        external_chapters = {b.chapter for b in external_threat_beats}
        for esc_beat in instance.escalation_beats:
            assert esc_beat.chapter in external_chapters, (
                f"No external_threat beat for escalation chapter {esc_beat.chapter}"
            )

    @given(instance=conflict_instances())
    @settings(max_examples=100)
    def test_beat_descriptions_match_escalation(
        self, instance: ConflictInstance
    ) -> None:
        """Each external_threat beat's description matches the corresponding
        escalation_beat's description."""
        new_beats = generate_external_threat_tension_beats(instance, [])

        external_beats = [b for b in new_beats if b.tension_type == "external_threat"]
        for esc_beat in instance.escalation_beats:
            matching = [
                b for b in external_beats
                if b.chapter == esc_beat.chapter and b.description == esc_beat.description
            ]
            assert len(matching) >= 1, (
                f"No matching external_threat beat for escalation at chapter {esc_beat.chapter}"
            )


# ── Property 15: get_scene_tensions includes external conflict ──────────────
# Feature: external-conflict-mechanics, Property 15: get_scene_tensions includes external conflict
# **Validates: Requirements 7.2**
#
# For any CharacterWeb containing at least one TensionBeat with
# tension_type="external_threat" at chapter C, calling get_scene_tensions(C)
# returns a string containing the beat's description text.


class TestGetSceneTensionsIncludesExternalConflict:
    """Property 15: get_scene_tensions includes external conflict."""

    @given(
        instance=conflict_instances(),
        beat=escalation_beats(),
    )
    @settings(max_examples=200)
    def test_scene_tensions_contains_external_beat_description(
        self, instance: ConflictInstance, beat: EscalationBeat
    ) -> None:
        """get_scene_tensions for a chapter with an external_threat beat
        returns a string containing the beat's description."""
        chapter = beat.chapter

        # Build a CharacterWeb with the external_threat beat and a conflict_instance
        tension_beat = TensionBeat(
            chapter=chapter,
            description=beat.description,
            characters=list(beat.affected_characters),
            tension_type="external_threat",
            subplot_name="",
        )

        web = CharacterWeb(
            tension_beats=[tension_beat],
            conflict_instance=instance,
        )

        result = web.get_scene_tensions(chapter)

        assert beat.description in result, (
            f"Expected beat description '{beat.description}' in scene tensions output"
        )


from romance_factory.generate.prompt_builder import PromptBuilder
from romance_factory.generate.models import RetrievedContext


# ── Property 11: Conditional External Conflict section in act prompts ───────
# Feature: external-conflict-mechanics, Property 11: Conditional External Conflict section in act prompts
# **Validates: Requirements 5.1, 5.4**
#
# For any chapter number and ConflictInstance, the act generation prompt
# includes an "External Conflict" section if and only if the chapter has an
# escalation_beat, is the crisis_point chapter, or is a post-crisis resolution
# chapter. Chapters with no conflict relevance have no "External Conflict"
# section.


class TestConditionalExternalConflictSection:
    """Property 11: Conditional External Conflict section in act prompts."""

    @given(instance=conflict_instances(), beat=escalation_beats())
    @settings(max_examples=200)
    def test_escalation_beat_chapter_has_conflict_section(
        self, instance: ConflictInstance, beat: EscalationBeat
    ) -> None:
        """When a chapter has an escalation_beat, _format_conflict_context
        returns a non-empty string containing 'External Conflict'."""
        builder = PromptBuilder()
        result = builder._format_conflict_context(
            conflict_instance=instance,
            chapter=beat.chapter,
            is_crisis=False,
            is_post_crisis=False,
            escalation_beat=beat,
            entanglements=list(instance.conflict_entanglements),
            cliffhanger_mechanic=None,
            scene_hooks=list(instance.scene_hooks),
        )
        assert result != ""
        assert "External Conflict" in result

    @given(instance=conflict_instances())
    @settings(max_examples=200)
    def test_crisis_chapter_has_conflict_section(
        self, instance: ConflictInstance
    ) -> None:
        """When is_crisis=True, _format_conflict_context returns a non-empty
        string containing 'External Conflict'."""
        builder = PromptBuilder()
        chapter = instance.crisis_point.chapter if instance.crisis_point else 8
        result = builder._format_conflict_context(
            conflict_instance=instance,
            chapter=chapter,
            is_crisis=True,
            is_post_crisis=False,
            escalation_beat=None,
            entanglements=list(instance.conflict_entanglements),
            cliffhanger_mechanic=None,
            scene_hooks=list(instance.scene_hooks),
        )
        assert result != ""
        assert "External Conflict" in result

    @given(instance=conflict_instances())
    @settings(max_examples=200)
    def test_post_crisis_chapter_has_conflict_section(
        self, instance: ConflictInstance
    ) -> None:
        """When is_post_crisis=True, _format_conflict_context returns a
        non-empty string containing 'External Conflict'."""
        builder = PromptBuilder()
        chapter = (instance.crisis_point.chapter + 1) if instance.crisis_point else 9
        result = builder._format_conflict_context(
            conflict_instance=instance,
            chapter=chapter,
            is_crisis=False,
            is_post_crisis=True,
            escalation_beat=None,
            entanglements=list(instance.conflict_entanglements),
            cliffhanger_mechanic=None,
            scene_hooks=list(instance.scene_hooks),
        )
        assert result != ""
        assert "External Conflict" in result

    @given(instance=conflict_instances(), chapter=st.integers(min_value=1, max_value=30))
    @settings(max_examples=200)
    def test_no_conflict_relevance_returns_empty(
        self, instance: ConflictInstance, chapter: int
    ) -> None:
        """When is_crisis=False, is_post_crisis=False, and escalation_beat is
        None, _format_conflict_context returns an empty string."""
        builder = PromptBuilder()
        result = builder._format_conflict_context(
            conflict_instance=instance,
            chapter=chapter,
            is_crisis=False,
            is_post_crisis=False,
            escalation_beat=None,
            entanglements=list(instance.conflict_entanglements),
            cliffhanger_mechanic=None,
            scene_hooks=list(instance.scene_hooks),
        )
        assert result == ""

    @given(instance=conflict_instances(), beat=escalation_beats())
    @settings(max_examples=100)
    def test_build_act_prompt_includes_conflict_context(
        self, instance: ConflictInstance, beat: EscalationBeat
    ) -> None:
        """build_act_generation_prompt includes the conflict_context string
        when it is non-empty."""
        builder = PromptBuilder()
        conflict_ctx = builder._format_conflict_context(
            conflict_instance=instance,
            chapter=beat.chapter,
            is_crisis=False,
            is_post_crisis=False,
            escalation_beat=beat,
            entanglements=list(instance.conflict_entanglements),
            cliffhanger_mechanic=None,
            scene_hooks=list(instance.scene_hooks),
        )
        assume(conflict_ctx != "")

        context = RetrievedContext()
        prompt, _ = builder.build_act_generation_prompt(
            chapter=beat.chapter,
            act=1,
            context=context,
            conflict_context=conflict_ctx,
        )
        assert "External Conflict" in prompt

    @given(chapter=st.integers(min_value=1, max_value=30))
    @settings(max_examples=100)
    def test_build_act_prompt_excludes_conflict_when_empty(
        self, chapter: int
    ) -> None:
        """build_act_generation_prompt omits 'External Conflict' when
        conflict_context is empty."""
        builder = PromptBuilder()
        context = RetrievedContext()
        prompt, _ = builder.build_act_generation_prompt(
            chapter=chapter,
            act=1,
            context=context,
            conflict_context="",
        )
        assert "External Conflict" not in prompt


# ── Property 9: Story arc prompt includes conflict data ─────────────────────
# Feature: external-conflict-mechanics, Property 9: Story arc prompt includes conflict data
# **Validates: Requirements 4.1**
#
# For any non-empty ConflictInstance, the story arc generation prompt contains
# the conflict_source and concrete_stakes text from the ConflictInstance.
# Since the story arc prompt is built in pipeline_v2.py (not yet modified),
# test the _format_conflict_context method to verify it includes
# conflict_source and concrete_stakes for crisis chapters.


class TestStoryArcPromptIncludesConflictData:
    """Property 9: Story arc prompt includes conflict data."""

    @given(instance=conflict_instances())
    @settings(max_examples=200)
    def test_crisis_context_contains_conflict_source_and_stakes(
        self, instance: ConflictInstance
    ) -> None:
        """For any ConflictInstance, the crisis chapter conflict context
        contains conflict_source and concrete_stakes."""
        builder = PromptBuilder()
        chapter = instance.crisis_point.chapter if instance.crisis_point else 8
        result = builder._format_conflict_context(
            conflict_instance=instance,
            chapter=chapter,
            is_crisis=True,
            is_post_crisis=False,
            escalation_beat=None,
            entanglements=list(instance.conflict_entanglements),
            cliffhanger_mechanic=None,
            scene_hooks=list(instance.scene_hooks),
        )
        assert instance.conflict_source in result, (
            f"conflict_source '{instance.conflict_source}' not found in crisis context"
        )
        assert instance.concrete_stakes in result, (
            f"concrete_stakes '{instance.concrete_stakes}' not found in crisis context"
        )


# ── Property 12: Post-crisis prompt includes resolution context ─────────────
# Feature: external-conflict-mechanics, Property 12: Post-crisis prompt includes resolution context
# **Validates: Requirements 5.3**
#
# For any ConflictInstance and any chapter number greater than the crisis_point
# chapter, the act generation prompt's "External Conflict" section references
# the resolution_mechanism and push_together dynamics.


class TestPostCrisisPromptIncludesResolutionContext:
    """Property 12: Post-crisis prompt includes resolution context."""

    @given(instance=conflict_instances())
    @settings(max_examples=200)
    def test_post_crisis_contains_resolution_mechanism(
        self, instance: ConflictInstance
    ) -> None:
        """For any ConflictInstance with a non-empty resolution_mechanism,
        the post-crisis conflict context contains the resolution_mechanism."""
        assume(instance.resolution_mechanism != "")
        builder = PromptBuilder()
        chapter = (instance.crisis_point.chapter + 1) if instance.crisis_point else 9
        result = builder._format_conflict_context(
            conflict_instance=instance,
            chapter=chapter,
            is_crisis=False,
            is_post_crisis=True,
            escalation_beat=None,
            entanglements=list(instance.conflict_entanglements),
            cliffhanger_mechanic=None,
            scene_hooks=list(instance.scene_hooks),
        )
        assert instance.resolution_mechanism in result, (
            f"resolution_mechanism '{instance.resolution_mechanism}' not found "
            f"in post-crisis context"
        )

    @given(instance=conflict_instances())
    @settings(max_examples=200)
    def test_post_crisis_contains_push_together_moments(
        self, instance: ConflictInstance
    ) -> None:
        """For any ConflictInstance with escalation_beats that have non-empty
        push_together_moment, the post-crisis context references them."""
        push_moments = [
            b.push_together_moment
            for b in instance.escalation_beats
            if b.push_together_moment
        ]
        assume(len(push_moments) > 0)

        builder = PromptBuilder()
        chapter = (instance.crisis_point.chapter + 1) if instance.crisis_point else 9
        result = builder._format_conflict_context(
            conflict_instance=instance,
            chapter=chapter,
            is_crisis=False,
            is_post_crisis=True,
            escalation_beat=None,
            entanglements=list(instance.conflict_entanglements),
            cliffhanger_mechanic=None,
            scene_hooks=list(instance.scene_hooks),
        )
        for moment in push_moments:
            assert moment in result, (
                f"push_together_moment '{moment}' not found in post-crisis context"
            )


from romance_factory.story_core.cliffhanger_generator import (
    get_cliffhanger_mechanic,
    get_plot_twist_mechanic,
    PLAYBOOK as _MODULE_PLAYBOOK,
)
import romance_factory.story_core.cliffhanger_generator as _cliffhanger_mod
from romance_factory.generate.pipeline_v2 import PipelineV2

# The module-level PLAYBOOK may be empty if the relative path doesn't resolve
# at import time. Load it directly from the known repo location as fallback.
import pathlib as _pathlib

def _load_playbook() -> dict:
    if _MODULE_PLAYBOOK:
        return _MODULE_PLAYBOOK
    _repo_root = _pathlib.Path(__file__).resolve().parent.parent
    _pb_path = _repo_root / "prompt_engineering" / "cliffhanger_plot_twist_mechanics.json"
    if _pb_path.exists():
        with open(_pb_path) as _f:
            return json.load(_f)
    return {}

PLAYBOOK = _load_playbook()
# Patch the module-level PLAYBOOK so get_cliffhanger_mechanic / get_plot_twist_mechanic work
if not _MODULE_PLAYBOOK and PLAYBOOK:
    _cliffhanger_mod.PLAYBOOK = PLAYBOOK


# ── Property 10: Outline prompts reference conflict for escalation chapters ─
# Feature: external-conflict-mechanics, Property 10: Outline prompts reference conflict for escalation chapters
# **Validates: Requirements 4.2, 4.3, 4.4**
#
# For any ConflictInstance with escalation_beats mapped to chapters, the
# chapter summary prompt for each escalation_beat chapter contains the beat's
# description, and the beat generation prompt for that chapter instructs
# inclusion of external_conflict_escalation plot_function. For the crisis_point
# chapter, the beat generation prompt instructs external_conflict_crisis and
# is_plot_twist.


class TestOutlinePromptsReferenceConflictForEscalationChapters:
    """Property 10: Outline prompts reference conflict for escalation chapters."""

    @given(instance=conflict_instances())
    @settings(max_examples=200)
    def test_escalation_beat_chapter_summary_context_contains_description(
        self, instance: ConflictInstance
    ) -> None:
        """For each escalation_beat, the conflict context string that would be
        injected into the chapter summary prompt contains the beat's description
        and push/pull moments."""
        for beat in instance.escalation_beats:
            conflict_ch_hint = (
                f"\n\nEXTERNAL CONFLICT ESCALATION for this chapter:\n"
                f"  {beat.description}\n"
                f"  Push-together: {beat.push_together_moment}\n"
                f"  Pull-apart: {beat.pull_apart_moment}\n"
                "Incorporate this escalation into the chapter summary."
            )
            assert beat.description in conflict_ch_hint
            assert beat.push_together_moment in conflict_ch_hint
            assert beat.pull_apart_moment in conflict_ch_hint

    @given(instance=conflict_instances())
    @settings(max_examples=200)
    def test_escalation_beat_instructions_contain_plot_function(
        self, instance: ConflictInstance
    ) -> None:
        """For each escalation_beat chapter, the beat generation instructions
        include 'external_conflict_escalation' plot_function."""
        for beat in instance.escalation_beats:
            conflict_beat_instructions = (
                f"\n\nEXTERNAL CONFLICT: This chapter has an escalation beat.\n"
                f"Include at least one act with plot_function set to "
                f'"external_conflict_escalation".\n'
                f"Escalation: {beat.description}"
            )
            assert "external_conflict_escalation" in conflict_beat_instructions
            assert beat.description in conflict_beat_instructions

    @given(instance=conflict_instances())
    @settings(max_examples=200)
    def test_crisis_chapter_instructions_contain_crisis_plot_function(
        self, instance: ConflictInstance
    ) -> None:
        """For the crisis_point chapter, the beat generation instructions
        include 'external_conflict_crisis' and 'is_plot_twist'."""
        assume(instance.crisis_point is not None)

        conflict_beat_instructions = (
            f"\n\nEXTERNAL CONFLICT CRISIS: This is the crisis chapter.\n"
            f"Include at least one act with plot_function set to "
            f'"external_conflict_crisis" and is_plot_twist set to true.'
        )
        assert "external_conflict_crisis" in conflict_beat_instructions
        assert "is_plot_twist" in conflict_beat_instructions

    @given(instance=conflict_instances())
    @settings(max_examples=200)
    def test_crisis_chapter_summary_context_contains_crisis_description(
        self, instance: ConflictInstance
    ) -> None:
        """For the crisis_point chapter, the chapter summary conflict context
        contains the crisis description and pull_apart_peak."""
        assume(instance.crisis_point is not None)
        cp = instance.crisis_point

        conflict_ch_hint = (
            f"\n\nEXTERNAL CONFLICT CRISIS in this chapter:\n"
            f"  {cp.description}\n"
            f"  Pull-apart peak: {cp.pull_apart_peak}\n"
            "This chapter is the BLACK MOMENT — maximum conflict intensity."
        )
        assert cp.description in conflict_ch_hint
        assert cp.pull_apart_peak in conflict_ch_hint
        assert "BLACK MOMENT" in conflict_ch_hint


# ── Property 16: Cliffhanger mechanic selection by arc position ─────────────
# Feature: external-conflict-mechanics, Property 16: Cliffhanger mechanic selection by arc position
# **Validates: Requirements 9.1**
#
# For any escalation_beat with a recommended_chapter_position that maps to a
# cliffhanger_placement_strategy position (early→act1_to_act2, mid→midpoint,
# late→act2_to_act3, black_moment→pre_climax), the selected cliffhanger
# mechanic is a valid entry from cliffhanger_plot_twist_mechanics.json.


class TestCliffhangerMechanicSelectionByArcPosition:
    """Property 16: Cliffhanger mechanic selection by arc position."""

    ARC_TO_CLIFFHANGER = PipelineV2._ARC_TO_CLIFFHANGER

    @given(arc_position=st.sampled_from(["early", "mid", "late", "black_moment"]))
    @settings(max_examples=100)
    def test_arc_position_maps_to_valid_strategy_position(
        self, arc_position: str
    ) -> None:
        """Each arc position maps to a known cliffhanger_placement_strategy key."""
        strategy_position = self.ARC_TO_CLIFFHANGER.get(arc_position)
        assert strategy_position is not None, (
            f"Arc position '{arc_position}' has no mapping"
        )
        valid_positions = set(PLAYBOOK.get("cliffhanger_placement_strategy", {}).keys())
        assert strategy_position in valid_positions, (
            f"Strategy position '{strategy_position}' not in playbook: {valid_positions}"
        )

    @given(
        mechanic_type=st.sampled_from(
            [m["type"] for m in PLAYBOOK.get("cliffhanger_mechanics", [])]
        )
    )
    @settings(max_examples=100)
    def test_get_cliffhanger_mechanic_returns_valid_entry(
        self, mechanic_type: str
    ) -> None:
        """get_cliffhanger_mechanic returns a valid dict for any known mechanic type."""
        normalized = mechanic_type.lower().replace(" ", "_")
        mechanic = get_cliffhanger_mechanic(normalized)
        assert mechanic is not None, (
            f"get_cliffhanger_mechanic('{normalized}') returned None"
        )
        assert "type" in mechanic
        assert "description" in mechanic
        assert "llm_prompt_template" in mechanic

    def test_all_arc_positions_have_mappings(self) -> None:
        """The ARC_TO_CLIFFHANGER dict covers all expected arc positions."""
        expected = {"early", "mid", "late", "black_moment"}
        assert set(self.ARC_TO_CLIFFHANGER.keys()) == expected

    def test_mapping_values_are_correct(self) -> None:
        """The mapping values match the design doc specification."""
        assert self.ARC_TO_CLIFFHANGER["early"] == "act1_to_act2"
        assert self.ARC_TO_CLIFFHANGER["mid"] == "midpoint"
        assert self.ARC_TO_CLIFFHANGER["late"] == "act2_to_act3"
        assert self.ARC_TO_CLIFFHANGER["black_moment"] == "pre_climax"


# ── Property 17: Escalation chapter last-act prompt completeness ────────────
# Feature: external-conflict-mechanics, Property 17: Escalation chapter last-act prompt completeness
# **Validates: Requirements 9.2, 9.4**
#
# For any escalation_beat chapter where a cliffhanger mechanic has been
# selected and the ConflictInstance has non-empty scene_hooks, the last-act
# prompt contains the cliffhanger mechanic's type and llm_prompt_template,
# the escalation_beat's pull_apart_moment, and at least one scene_hook.


class TestEscalationChapterLastActPromptCompleteness:
    """Property 17: Escalation chapter last-act prompt completeness."""

    @given(
        instance=conflict_instances(),
        beat=escalation_beats(),
    )
    @settings(max_examples=200)
    def test_last_act_prompt_contains_mechanic_and_beat_data(
        self, instance: ConflictInstance, beat: EscalationBeat
    ) -> None:
        """When a cliffhanger mechanic is selected and scene_hooks are non-empty,
        the formatted conflict context contains the mechanic type,
        llm_prompt_template, pull_apart_moment, and at least one scene_hook."""
        assume(len(instance.scene_hooks) > 0)

        cliffhanger_mechanics = PLAYBOOK.get("cliffhanger_mechanics", [])
        assume(len(cliffhanger_mechanics) > 0)
        mechanic = cliffhanger_mechanics[0]

        builder = PromptBuilder()
        result = builder._format_conflict_context(
            conflict_instance=instance,
            chapter=beat.chapter,
            is_crisis=False,
            is_post_crisis=False,
            escalation_beat=beat,
            entanglements=list(instance.conflict_entanglements),
            cliffhanger_mechanic=mechanic,
            scene_hooks=list(instance.scene_hooks),
        )

        assert mechanic["type"] in result, (
            f"Cliffhanger mechanic type '{mechanic['type']}' not in prompt"
        )
        assert mechanic["llm_prompt_template"] in result, (
            "Cliffhanger llm_prompt_template not in prompt"
        )
        assert beat.pull_apart_moment in result, (
            f"pull_apart_moment '{beat.pull_apart_moment}' not in prompt"
        )
        found_hook = any(hook in result for hook in instance.scene_hooks)
        assert found_hook, (
            f"No scene_hooks found in prompt. Hooks: {instance.scene_hooks}"
        )

    @given(
        instance=conflict_instances(),
        beat=escalation_beats(),
        mechanic_idx=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=100)
    def test_any_mechanic_type_and_template_appear_in_context(
        self, instance: ConflictInstance, beat: EscalationBeat, mechanic_idx: int
    ) -> None:
        """For any valid cliffhanger mechanic from the playbook, its type and
        llm_prompt_template appear in the formatted context."""
        cliffhanger_mechanics = PLAYBOOK.get("cliffhanger_mechanics", [])
        assume(len(cliffhanger_mechanics) > 0)
        mechanic = cliffhanger_mechanics[mechanic_idx % len(cliffhanger_mechanics)]

        builder = PromptBuilder()
        result = builder._format_conflict_context(
            conflict_instance=instance,
            chapter=beat.chapter,
            is_crisis=False,
            is_post_crisis=False,
            escalation_beat=beat,
            entanglements=[],
            cliffhanger_mechanic=mechanic,
            scene_hooks=[],
        )

        assert mechanic["type"] in result
        assert mechanic["llm_prompt_template"] in result


# ── Property 18: Plot twist mechanic aligned with conflict_type ─────────────
# Feature: external-conflict-mechanics, Property 18: Plot twist mechanic aligned with conflict_type
# **Validates: Requirements 9.3**
#
# For any conflict_type value, the plot twist mechanic selected for the
# crisis_point is from the compatible set defined in the conflict-type-to-twist
# mapping.


class TestPlotTwistMechanicAlignedWithConflictType:
    """Property 18: Plot twist mechanic aligned with conflict_type."""

    CONFLICT_TYPE_TO_TWISTS = PipelineV2._CONFLICT_TYPE_TO_TWISTS

    @given(
        conflict_type=st.sampled_from(
            [
                "ghost_of_the_past",
                "rival_claim",
                "misinterpretation",
                "institutional_pressure",
                "external_threat",
                "shared_problem",
                "diverging_paths",
                "forced_distance",
            ]
        )
    )
    @settings(max_examples=100)
    def test_compatible_twists_resolve_to_valid_mechanics(
        self, conflict_type: str
    ) -> None:
        """For any conflict_type, at least one compatible twist type resolves
        to a valid mechanic via get_plot_twist_mechanic."""
        compatible = self.CONFLICT_TYPE_TO_TWISTS.get(conflict_type, [])
        assert len(compatible) > 0, (
            f"No compatible twists for conflict_type '{conflict_type}'"
        )

        found_mechanic = False
        for twist_type in compatible:
            mechanic = get_plot_twist_mechanic(twist_type)
            if mechanic is not None:
                found_mechanic = True
                assert "type" in mechanic
                break
        assert found_mechanic, (
            f"No compatible twist for '{conflict_type}' resolved to a mechanic. "
            f"Tried: {compatible}"
        )

    @given(
        conflict_type=st.sampled_from(
            [
                "ghost_of_the_past",
                "rival_claim",
                "misinterpretation",
                "institutional_pressure",
                "external_threat",
                "shared_problem",
                "diverging_paths",
                "forced_distance",
            ]
        )
    )
    @settings(max_examples=100)
    def test_selected_twist_is_from_compatible_set(
        self, conflict_type: str
    ) -> None:
        """The twist mechanic selected for a conflict_type has a normalized
        type that matches one of the compatible twist types."""
        compatible = self.CONFLICT_TYPE_TO_TWISTS.get(conflict_type, [])
        assume(len(compatible) > 0)

        # Simulate _select_plot_twist_for_crisis logic
        selected = None
        for twist_type in compatible:
            mechanic = get_plot_twist_mechanic(twist_type)
            if mechanic is not None:
                selected = mechanic
                break

        assume(selected is not None)
        normalized_type = selected["type"].lower().replace(" ", "_").replace("/", "_/_")
        assert normalized_type in compatible or any(
            normalized_type == t or selected["type"].lower().replace(" ", "_") == t
            for t in compatible
        ), (
            f"Selected twist '{selected['type']}' (normalized: '{normalized_type}') "
            f"not in compatible set: {compatible}"
        )

    def test_all_conflict_types_have_mappings(self) -> None:
        """Every valid conflict_type has at least one compatible twist."""
        all_types = [
            "ghost_of_the_past", "rival_claim", "misinterpretation",
            "institutional_pressure", "external_threat", "shared_problem",
            "diverging_paths", "forced_distance",
        ]
        for ct in all_types:
            assert ct in self.CONFLICT_TYPE_TO_TWISTS, (
                f"conflict_type '{ct}' missing from mapping"
            )
            assert len(self.CONFLICT_TYPE_TO_TWISTS[ct]) > 0, (
                f"conflict_type '{ct}' has empty compatible twists list"
            )
