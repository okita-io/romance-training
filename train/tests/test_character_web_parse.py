"""Tests for character web JSON extraction (LLM output quirks)."""

from __future__ import annotations

import json

from romance_factory.story_core.character_web import (
    _first_json_object,
    _parse_character_web,
)


def test_first_json_object_with_markdown_fence():
    text = '```json\n{"motivations": [{"name": "A", "role": "x"}]}\n```'
    obj, err = _first_json_object(text)
    assert err is None
    assert obj is not None
    assert obj["motivations"][0]["name"] == "A"


def test_first_json_object_ignores_trailing_prose():
    text = 'Here you go:\n{"motivations": [], "relationships": []}\nHope this helps!'
    obj, err = _first_json_object(text)
    assert err is None
    assert obj == {"motivations": [], "relationships": []}


def test_parse_with_brace_inside_escaped_string():
    """Brace inside a JSON string must not confuse extraction (valid JSON)."""
    payload = {
        "motivations": [
            {
                "name": "Bob",
                "role": "ally",
                "secret": 'He said "done}" and left.',
            }
        ],
        "relationships": [],
    }
    text = json.dumps(payload)
    web, err = _parse_character_web(text)
    assert err is None
    assert "Bob" in web.motivations


def test_parse_empty_on_invalid_json():
    web, err = _parse_character_web("not json {")
    assert not web.motivations
    assert err is not None
