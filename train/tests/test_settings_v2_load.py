"""Tests for generate pipeline YAML loader (settings_v2_load.py).

Reads ``settings.yaml`` ``generate:`` (unified config); legacy paths are deprecated.
"""

from __future__ import annotations

from romance_factory.generate import settings_v2_load


def test_settings_v2_path_default_is_legacy_filename():
    p = settings_v2_load.settings_v2_path()
    assert p.name == "settings_v2.yaml"
    assert p.parent == settings_v2_load.REPO_ROOT


def test_settings_v2_path_env_override(tmp_path, monkeypatch):
    custom = tmp_path / "custom_v2.yaml"
    custom.write_text("top_k: 5\n", encoding="utf-8")
    monkeypatch.setenv("ROMANCE_FACTORY_V2_SETTINGS_PATH", str(custom))
    assert settings_v2_load.settings_v2_path() == custom.resolve()


def test_settings_v2_path_env_whitespace_ignored(monkeypatch):
    monkeypatch.setenv("ROMANCE_FACTORY_V2_SETTINGS_PATH", "   ")
    p = settings_v2_load.settings_v2_path()
    assert p.name == "settings_v2.yaml"


def test_load_uses_generate_section(monkeypatch):
    monkeypatch.setattr(
        "romance_factory.core.settings_load.load_settings_yaml",
        lambda: {"generate": {"embedding_model": "e5", "default_top_k": 3}},
    )
    result = settings_v2_load.load_settings_v2_yaml()
    assert result["embedding_model"] == "e5"
    assert result["default_top_k"] == 3


def test_load_v2_env_file_merges_on_top(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "romance_factory.core.settings_load.load_settings_yaml",
        lambda: {"generate": {"embedding_model": "e5", "default_top_k": 3}},
    )
    extra = tmp_path / "extra.yaml"
    extra.write_text("default_top_k: 99\n", encoding="utf-8")
    monkeypatch.setenv("ROMANCE_FACTORY_V2_SETTINGS_PATH", str(extra))
    result = settings_v2_load.load_settings_v2_yaml()
    assert result["embedding_model"] == "e5"
    assert result["default_top_k"] == 99


def test_load_legacy_settings_v2_file_merges(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_v2_load, "_legacy_v2_file_warned", False)
    monkeypatch.setattr(
        "romance_factory.core.settings_load.load_settings_yaml",
        lambda: {"generate": {"embedding_model": "e5"}},
    )
    monkeypatch.setattr(settings_v2_load, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings_v2_load, "REPO_ROOT", tmp_path)
    legacy = tmp_path / "settings_v2.yaml"
    legacy.write_text("default_top_k: 5\n", encoding="utf-8")
    result = settings_v2_load.load_settings_v2_yaml()
    assert result["embedding_model"] == "e5"
    assert result["default_top_k"] == 5


def test_load_invalid_v2_env_file_returns_partial(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "romance_factory.core.settings_load.load_settings_yaml",
        lambda: {"generate": {"embedding_model": "e5"}},
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text(":\n  - :\n    bad: [unterminated", encoding="utf-8")
    monkeypatch.setenv("ROMANCE_FACTORY_V2_SETTINGS_PATH", str(bad))
    result = settings_v2_load.load_settings_v2_yaml()
    assert result.get("embedding_model") == "e5"
    assert "warning" in capsys.readouterr().err.lower()


def test_load_non_dict_env_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "romance_factory.core.settings_load.load_settings_yaml",
        lambda: {},
    )
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- one\n- two\n", encoding="utf-8")
    monkeypatch.setenv("ROMANCE_FACTORY_V2_SETTINGS_PATH", str(list_yaml))
    assert settings_v2_load.load_settings_v2_yaml() == {}
