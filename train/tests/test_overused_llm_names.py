"""overused_llm_names.yaml loader and prompt fragments."""

from romance_factory.core import overused_llm_names as oln


def test_load_includes_elias_and_legacy_defaults():
    names = oln.load_overused_llm_character_names()
    assert "Elias" in names
    for x in ("Elara", "Kael", "Astra"):
        assert x in names


def test_load_surnames_includes_vance_and_defaults():
    surnames = oln.load_overused_llm_character_surnames()
    assert "Vance" in surnames
    for x in ("Vane", "Vale", "Vex", "Voss"):
        assert x in surnames


def test_format_character_instruction_contains_joined_names():
    text = oln.format_character_name_avoidance_instruction()
    assert "IMPORTANT:" in text
    assert "Elias" in text
    assert text.endswith("small-model habits.")


def test_format_character_instruction_contains_surnames():
    text = oln.format_character_name_avoidance_instruction()
    assert "surname" in text.lower()
    assert "Vance" in text


def test_pen_name_note_lists_same_catalog():
    text = oln.format_pen_name_avoidance_note()
    assert "pen_name" in text
    assert "Elias" in text
    assert "including:" in text


def test_pen_name_note_includes_surnames():
    text = oln.format_pen_name_avoidance_note()
    assert "surname" in text.lower()
    assert "Vance" in text


def test_clear_cache_roundtrip(tmp_path, monkeypatch):
    oln.clear_overused_llm_names_cache()
    p = tmp_path / "names.yaml"
    p.write_text(
        "names:\n  - ZZZTestOnlyName\nsurnames:\n  - ZZZTestOnlySurname\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ROMANCE_FACTORY_OVERUSED_LLM_NAMES_PATH", str(p))
    oln.clear_overused_llm_names_cache()
    try:
        assert oln.load_overused_llm_character_names() == ["ZZZTestOnlyName"]
        assert oln.load_overused_llm_character_surnames() == ["ZZZTestOnlySurname"]
    finally:
        oln.clear_overused_llm_names_cache()
