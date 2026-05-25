"""Unit tests for v2 CLI ``--world`` parsing and V2Config world-related fields."""

from __future__ import annotations

import pytest

from romance_factory.generate import cli as v2_main
from romance_factory.generate.config_v2 import V2Config, load_v2_config


@pytest.fixture
def empty_v2_yaml(monkeypatch):
    """Isolate tests from repo ``settings.yaml`` ``generate:`` (no accidental world/genre overrides)."""

    monkeypatch.setattr(
        "romance_factory.generate.settings_v2_load.load_settings_v2_yaml",
        lambda: {},
    )


def test_v2config_world_field_defaults():
    cfg = V2Config()
    assert cfg.world is None
    assert cfg.retrieval_top_k_world == 3
    assert cfg.world_setting_catalog is None


def test_cli_world_flag_absent():
    parser = v2_main._build_parser()
    args = parser.parse_args(["--story-path", "stories/x"])
    assert args.world is None


def test_cli_world_flag_present():
    parser = v2_main._build_parser()
    args = parser.parse_args(["--story-path", "stories/x", "--world", "fantasy"])
    assert args.world == "fantasy"


def test_cli_genre_and_world_together():
    parser = v2_main._build_parser()
    args = parser.parse_args(
        [
            "--story-path",
            "stories/x",
            "--genre",
            "paranormal_vampire",
            "--world",
            "fantasy",
        ]
    )
    assert args.genre == "paranormal_vampire"
    assert args.world == "fantasy"


def test_load_v2_config_world_none_when_omitted(empty_v2_yaml):
    cfg = load_v2_config(story_path="stories/x")
    assert cfg.world is None


def test_load_v2_config_world_from_cli(empty_v2_yaml):
    cfg = load_v2_config(story_path="stories/x", world="sci_fi")
    assert cfg.world == "sci_fi"


def test_load_v2_config_genre_and_world_from_cli(empty_v2_yaml):
    cfg = load_v2_config(
        story_path="stories/x",
        genre="paranormal_vampire",
        world="fantasy",
    )
    assert cfg.genre == "paranormal_vampire"
    assert cfg.world == "fantasy"


def test_load_v2_config_world_cli_overrides_yaml(monkeypatch):
    monkeypatch.setattr(
        "romance_factory.generate.settings_v2_load.load_settings_v2_yaml",
        lambda: {"world": "from_yaml"},
    )
    cfg = load_v2_config(story_path="stories/x", world="from_cli")
    assert cfg.world == "from_cli"


def test_load_v2_config_retrieval_top_k_world_default(empty_v2_yaml):
    cfg = load_v2_config(story_path="stories/x")
    assert cfg.retrieval_top_k_world == 3


def test_load_v2_config_world_setting_catalog_none_without_yaml_key(empty_v2_yaml):
    cfg = load_v2_config(story_path="stories/x")
    assert cfg.world_setting_catalog is None
