"""Tests for optional YAML settings loading (no dependency on a real settings file)."""

from __future__ import annotations

import os
from pathlib import Path

import romance_factory.core.settings_load as settings_load


def test_settings_path_default_points_under_repo():
    p = settings_load.settings_path()
    assert p.name == "settings.yaml"
    assert p.parent == settings_load.REPO_ROOT


def test_settings_path_env_override(tmp_path, monkeypatch):
    custom = tmp_path / "custom.yaml"
    custom.write_text("max_tokens: 1\n", encoding="utf-8")
    monkeypatch.setenv("ROMANCE_FACTORY_SETTINGS_PATH", str(custom))
    assert settings_load.settings_path() == custom.resolve()


def test_load_settings_yaml_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ROMANCE_FACTORY_SETTINGS_PATH", str(tmp_path / "nope.yaml"))
    assert settings_load.load_settings_yaml() == {}
