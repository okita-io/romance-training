"""llm_presets reasoning / thinking → OPENROUTER_PRESET_BODY."""

from romance_factory.core.presets import _openrouter_preset_body_from_spec


def test_reasoning_bool():
    assert _openrouter_preset_body_from_spec({"reasoning": True}) == {
        "reasoning": {"enabled": True}
    }
    assert _openrouter_preset_body_from_spec({"reasoning": False}) == {
        "reasoning": {"enabled": False}
    }


def test_thinking_alias_when_reasoning_absent():
    assert _openrouter_preset_body_from_spec({"thinking": True}) == {
        "reasoning": {"enabled": True}
    }
    assert _openrouter_preset_body_from_spec(
        {"reasoning": False, "thinking": True}
    ) == {"reasoning": {"enabled": False}}


def test_reasoning_dict_passthrough():
    assert _openrouter_preset_body_from_spec(
        {"reasoning": {"effort": "high", "max_tokens": 100}}
    ) == {"reasoning": {"effort": "high", "max_tokens": 100}}


def test_reasoning_string_effort():
    assert _openrouter_preset_body_from_spec({"reasoning": "low"}) == {
        "reasoning": {"effort": "low"}
    }


def test_empty_when_unset():
    assert _openrouter_preset_body_from_spec({}) == {}
