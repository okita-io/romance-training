"""Unit tests for V2Config validation and bridge method.

Validates: Requirements 9.1, 9.2, 9.3, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
"""

from __future__ import annotations

import pytest

from romance_factory.generate.config_v2 import V2Config
from romance_factory.generate.phrase_detection.config import PhraseDetectionConfig


# ---------------------------------------------------------------------------
# Validation — valid config
# ---------------------------------------------------------------------------


class TestV2ConfigValidation:
    """Test V2Config.validate() raises ValueError for invalid inputs."""

    def test_valid_defaults_pass_validation(self):
        """Default V2Config should pass validation without error."""
        cfg = V2Config()
        cfg.validate()  # should not raise

    def test_valid_custom_config_passes(self):
        """A custom config with all values in valid ranges passes."""
        cfg = V2Config(
            passing_score_threshold=5.0,
            min_acts_per_chapter=2,
            max_acts_per_chapter=4,
            min_words_per_chapter=2000,
            max_words_per_chapter=5000,
            min_words_per_act=300,
            max_words_per_act=600,
            phrase_similarity_threshold=0.5,
            temperature=1.2,
            phrase_min_ngram_words=3,
            phrase_max_ngram_words=10,
        )
        cfg.validate()

    def test_diegetic_max_llm_rounds_below_one_raises(self):
        cfg = V2Config(diegetic_max_llm_rounds=0)
        with pytest.raises(ValueError, match="diegetic_max_llm_rounds"):
            cfg.validate()

    # --- Surgical editor (V2Config.generate YAML) ---

    def test_surgical_max_act_change_pct_out_of_range_raises(self):
        cfg = V2Config(surgical_max_act_change_pct=100.1)
        with pytest.raises(ValueError, match="surgical_max_act_change_pct"):
            cfg.validate()

    def test_surgical_max_verify_rounds_out_of_range_raises(self):
        cfg = V2Config(surgical_max_verify_rounds=6)
        with pytest.raises(ValueError, match="surgical_max_verify_rounds"):
            cfg.validate()

    def test_surgical_batch_mode_invalid_raises(self):
        cfg = V2Config(surgical_batch_mode="nosuch")
        with pytest.raises(ValueError, match="surgical_batch_mode"):
            cfg.validate()

    def test_surgical_max_chars_per_replace_below_one_raises(self):
        cfg = V2Config(surgical_max_chars_per_replace=0)
        with pytest.raises(ValueError, match="surgical_max_chars_per_replace"):
            cfg.validate()

    def test_surgical_lancedb_replace_max_retries_out_of_range_raises(self):
        cfg = V2Config(surgical_lancedb_replace_max_retries=11)
        with pytest.raises(ValueError, match="surgical_lancedb_replace_max_retries"):
            cfg.validate()

    def test_surgical_defaults_validate(self):
        cfg = V2Config()
        assert cfg.enable_surgical_editing is True
        assert cfg.surgical_batch_mode == "transactional"
        assert cfg.surgical_lancedb_replace_max_retries == 3
        cfg.validate()

    # --- passing_score_threshold [0.0, 10.0] (Req 10.2) ---

    def test_passing_score_threshold_below_zero_raises(self):
        cfg = V2Config(passing_score_threshold=-0.1)
        with pytest.raises(ValueError, match="passing_score_threshold"):
            cfg.validate()

    def test_passing_score_threshold_above_ten_raises(self):
        cfg = V2Config(passing_score_threshold=10.1)
        with pytest.raises(ValueError, match="passing_score_threshold"):
            cfg.validate()

    def test_passing_score_threshold_boundary_zero_ok(self):
        V2Config(passing_score_threshold=0.0).validate()

    def test_passing_score_threshold_boundary_ten_ok(self):
        V2Config(passing_score_threshold=10.0).validate()

    # --- min_acts_per_chapter > max_acts_per_chapter (Req 10.3) ---

    def test_min_acts_greater_than_max_acts_raises(self):
        cfg = V2Config(min_acts_per_chapter=6, max_acts_per_chapter=5)
        with pytest.raises(ValueError, match="min_acts_per_chapter"):
            cfg.validate()

    def test_min_acts_equal_max_acts_ok(self):
        V2Config(min_acts_per_chapter=5, max_acts_per_chapter=5).validate()

    # --- min_words_per_chapter > max_words_per_chapter (Req 10.4) ---

    def test_min_words_chapter_greater_than_max_raises(self):
        cfg = V2Config(min_words_per_chapter=8000, max_words_per_chapter=3000)
        with pytest.raises(ValueError, match="min_words_per_chapter"):
            cfg.validate()

    def test_min_words_chapter_equal_max_ok(self):
        V2Config(min_words_per_chapter=5000, max_words_per_chapter=5000).validate()

    # --- min_words_per_act > max_words_per_act ---

    def test_min_words_act_greater_than_max_raises(self):
        cfg = V2Config(min_words_per_act=900, max_words_per_act=500)
        with pytest.raises(ValueError, match="min_words_per_act"):
            cfg.validate()

    def test_min_words_act_equal_max_ok(self):
        V2Config(min_words_per_act=600, max_words_per_act=600).validate()

    # --- phrase_similarity_threshold [0.0, 1.0] (Req 10.5) ---

    def test_phrase_similarity_below_zero_raises(self):
        cfg = V2Config(phrase_similarity_threshold=-0.01)
        with pytest.raises(ValueError, match="phrase_similarity_threshold"):
            cfg.validate()

    def test_phrase_similarity_above_one_raises(self):
        cfg = V2Config(phrase_similarity_threshold=1.01)
        with pytest.raises(ValueError, match="phrase_similarity_threshold"):
            cfg.validate()

    def test_phrase_similarity_boundary_zero_ok(self):
        V2Config(phrase_similarity_threshold=0.0).validate()

    def test_phrase_similarity_boundary_one_ok(self):
        V2Config(phrase_similarity_threshold=1.0).validate()

    # --- temperature >= 0.0 (Req 10.6) ---

    def test_negative_temperature_raises(self):
        cfg = V2Config(temperature=-0.1)
        with pytest.raises(ValueError, match="temperature"):
            cfg.validate()

    def test_temperature_zero_ok(self):
        V2Config(temperature=0.0).validate()

    # --- phrase_min_ngram_words >= 2 ---

    def test_phrase_min_ngram_below_two_raises(self):
        cfg = V2Config(phrase_min_ngram_words=1)
        with pytest.raises(ValueError, match="phrase_min_ngram_words"):
            cfg.validate()

    def test_phrase_min_ngram_exactly_two_ok(self):
        V2Config(phrase_min_ngram_words=2, phrase_max_ngram_words=2).validate()

    # --- phrase_max_ngram_words < phrase_min_ngram_words ---

    def test_phrase_max_ngram_below_min_raises(self):
        cfg = V2Config(phrase_min_ngram_words=5, phrase_max_ngram_words=3)
        with pytest.raises(ValueError, match="phrase_max_ngram_words"):
            cfg.validate()

    def test_phrase_max_ngram_equal_min_ok(self):
        V2Config(phrase_min_ngram_words=4, phrase_max_ngram_words=4).validate()


# ---------------------------------------------------------------------------
# Bridge method — to_phrase_detection_config() (Req 9.1, 9.2, 9.3)
# ---------------------------------------------------------------------------


class TestV2ConfigBridgeMethod:
    """Test V2Config.to_phrase_detection_config() maps fields correctly."""

    def test_bridge_returns_phrase_detection_config(self):
        cfg = V2Config()
        result = cfg.to_phrase_detection_config()
        assert isinstance(result, PhraseDetectionConfig)

    def test_bridge_maps_all_phrase_fields(self):
        """All phrase_* fields map to corresponding PhraseDetectionConfig fields."""
        cfg = V2Config(
            phrase_min_ngram_words=3,
            phrase_max_ngram_words=15,
            phrase_similarity_threshold=0.90,
            phrase_top_k_retrieval=25,
            phrase_max_clusters=60,
            phrase_context_sentences=3,
            phrase_output_suffix="_cleaned",
            phrase_chapter_heading_pattern=r"^CH\s+\d+",
        )
        pdc = cfg.to_phrase_detection_config()

        assert pdc.min_ngram_words == 3
        assert pdc.max_ngram_words == 15
        assert pdc.similarity_threshold == 0.90
        assert pdc.top_k_retrieval == 25
        assert pdc.max_clusters == 60
        assert pdc.context_sentences == 3
        assert pdc.output_suffix == "_cleaned"
        assert pdc.chapter_heading_pattern == r"^CH\s+\d+"

    def test_bridge_maps_embedding_model(self):
        """Req 9.3 — embedding_model is forwarded."""
        cfg = V2Config(embedding_model="all-minilm")
        pdc = cfg.to_phrase_detection_config()
        assert pdc.embedding_model == "all-minilm"

    def test_bridge_maps_db_path(self):
        """Req 9.3 — db_path is forwarded."""
        cfg = V2Config(db_path="/tmp/test_lance")
        pdc = cfg.to_phrase_detection_config()
        assert pdc.db_path == "/tmp/test_lance"

    def test_bridge_defaults_match_phrase_detection_defaults(self):
        """Default V2Config produces a PhraseDetectionConfig with matching defaults."""
        v2 = V2Config()
        pdc = v2.to_phrase_detection_config()
        pdc_default = PhraseDetectionConfig()

        assert pdc.min_ngram_words == pdc_default.min_ngram_words
        assert pdc.max_ngram_words == pdc_default.max_ngram_words
        assert pdc.similarity_threshold == pdc_default.similarity_threshold
        assert pdc.top_k_retrieval == pdc_default.top_k_retrieval
        assert pdc.max_clusters == pdc_default.max_clusters
        assert pdc.context_sentences == pdc_default.context_sentences
        assert pdc.output_suffix == pdc_default.output_suffix
        assert pdc.chapter_heading_pattern == pdc_default.chapter_heading_pattern
        assert pdc.embedding_model == pdc_default.embedding_model
        assert pdc.db_path == pdc_default.db_path


# ---------------------------------------------------------------------------
# load_v2_config() precedence chain (Req 4.1–4.5, 5.1–5.4)
# ---------------------------------------------------------------------------

import dataclasses
import os

from romance_factory.generate.config_v2 import load_v2_config

# All env var keys that could interfere with tests (one per scalar V2Config field).
_ALL_V2_ENV_KEYS = [
    f"ROMANCE_FACTORY_V2_{f.name.upper()}"
    for f in dataclasses.fields(V2Config)
]
_ALL_V2_ENV_KEYS.append("ROMANCE_FACTORY_V2_SETTINGS_PATH")


@pytest.fixture(autouse=False)
def _clean_v2_env(monkeypatch):
    """Remove every ROMANCE_FACTORY_V2_* env var so tests start from a clean slate."""
    for key in _ALL_V2_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _mock_yaml(monkeypatch, data: dict):
    """Monkeypatch load_settings_v2_yaml to return *data* instead of reading disk."""
    monkeypatch.setattr(
        "romance_factory.generate.settings_v2_load.load_settings_v2_yaml",
        lambda: data,
    )


class TestLoadV2ConfigPrecedence:
    """Scalars: CLI > LM Studio detected ``llm_context_tokens`` > YAML > default.

    ``ROMANCE_FACTORY_V2_*`` environment variables are ignored for non-secret fields.
    """

    # --- defaults when nothing provided ---

    def test_defaults_when_nothing_provided(self, monkeypatch, _clean_v2_env):
        """With no CLI or YAML, V2Config uses dataclass defaults."""
        _mock_yaml(monkeypatch, {})
        cfg = load_v2_config()
        assert cfg.default_top_k == 10
        assert cfg.temperature == 0.8
        assert cfg.embedding_model == "bge-large"
        assert cfg.llm_stream is True

    # --- YAML wins over default ---

    def test_yaml_wins_over_default(self, monkeypatch, _clean_v2_env):
        """YAML value is used when no CLI override is set."""
        _mock_yaml(monkeypatch, {"default_top_k": 42, "temperature": 0.5})
        cfg = load_v2_config()
        assert cfg.default_top_k == 42
        assert cfg.temperature == 0.5

    def test_yaml_surgical_fields(self, monkeypatch, _clean_v2_env):
        _mock_yaml(
            monkeypatch,
            {
                "enable_surgical_editing": False,
                "surgical_max_act_change_pct": 12.5,
                "surgical_batch_mode": "continue_on_failure",
                "surgical_lancedb_replace_max_retries": 0,
            },
        )
        cfg = load_v2_config()
        assert cfg.enable_surgical_editing is False
        assert cfg.surgical_max_act_change_pct == 12.5
        assert cfg.surgical_batch_mode == "continue_on_failure"
        assert cfg.surgical_lancedb_replace_max_retries == 0

    # --- env vars ignored (non-secrets come from YAML / CLI only) ---

    def test_env_ignored_for_scalars(self, monkeypatch, _clean_v2_env):
        """ROMANCE_FACTORY_V2_* does not override YAML."""
        _mock_yaml(monkeypatch, {"default_top_k": 42})
        monkeypatch.setenv("ROMANCE_FACTORY_V2_DEFAULT_TOP_K", "99")
        cfg = load_v2_config()
        assert cfg.default_top_k == 42

    # --- CLI wins over YAML, env, and default ---

    def test_cli_wins_over_yaml_and_env(self, monkeypatch, _clean_v2_env):
        """CLI kwarg takes highest precedence."""
        _mock_yaml(monkeypatch, {"default_top_k": 42})
        monkeypatch.setenv("ROMANCE_FACTORY_V2_DEFAULT_TOP_K", "99")
        cfg = load_v2_config(default_top_k=7)
        assert cfg.default_top_k == 7

    def test_cli_wins_over_yaml_no_env(self, monkeypatch, _clean_v2_env):
        """CLI kwarg beats YAML."""
        _mock_yaml(monkeypatch, {"default_top_k": 42})
        cfg = load_v2_config(default_top_k=7)
        assert cfg.default_top_k == 7

    def test_cli_wins_over_env_no_yaml(self, monkeypatch, _clean_v2_env):
        """CLI kwarg beats spurious env when YAML is empty."""
        _mock_yaml(monkeypatch, {})
        monkeypatch.setenv("ROMANCE_FACTORY_V2_DEFAULT_TOP_K", "99")
        cfg = load_v2_config(default_top_k=7)
        assert cfg.default_top_k == 7

    # --- None CLI args are skipped ---

    def test_none_cli_arg_skipped_env_ignored_use_default(self, monkeypatch, _clean_v2_env):
        """A None CLI kwarg is ignored; env var does not fill in (use default)."""
        _mock_yaml(monkeypatch, {})
        monkeypatch.setenv("ROMANCE_FACTORY_V2_DEFAULT_TOP_K", "55")
        cfg = load_v2_config(default_top_k=None)
        assert cfg.default_top_k == 10

    def test_none_cli_arg_skipped_falls_to_yaml(self, monkeypatch, _clean_v2_env):
        """A None CLI kwarg is ignored; YAML value is used."""
        _mock_yaml(monkeypatch, {"default_top_k": 42})
        cfg = load_v2_config(default_top_k=None)
        assert cfg.default_top_k == 42

    def test_none_cli_arg_skipped_falls_to_default(self, monkeypatch, _clean_v2_env):
        """A None CLI kwarg is ignored; dataclass default is used."""
        _mock_yaml(monkeypatch, {})
        cfg = load_v2_config(default_top_k=None)
        assert cfg.default_top_k == 10  # dataclass default

    # --- LM Studio detected context tokens ---

    def test_detected_llm_context_tokens_beats_yaml(self, monkeypatch, _clean_v2_env):
        from romance_factory.core import config as core_config

        _mock_yaml(monkeypatch, {"llm_context_tokens": 8000})
        core_config.set_detected_llm_context_tokens(128000)
        try:
            cfg = load_v2_config()
            assert cfg.llm_context_tokens == 128000
        finally:
            core_config.set_detected_llm_context_tokens(None)

    def test_cli_beats_detected_llm_context_tokens(self, monkeypatch, _clean_v2_env):
        from romance_factory.core import config as core_config

        _mock_yaml(monkeypatch, {"llm_context_tokens": 8000})
        core_config.set_detected_llm_context_tokens(128000)
        try:
            cfg = load_v2_config(llm_context_tokens=4096)
            assert cfg.llm_context_tokens == 4096
        finally:
            core_config.set_detected_llm_context_tokens(None)

    # --- validation ---

    def test_validation_called_invalid_threshold_raises(self, monkeypatch, _clean_v2_env):
        """load_v2_config raises ValueError when resolved config is invalid."""
        _mock_yaml(monkeypatch, {})
        with pytest.raises(ValueError, match="passing_score_threshold"):
            load_v2_config(passing_score_threshold=11.0)

    def test_validation_called_invalid_temperature_raises(self, monkeypatch, _clean_v2_env):
        """Negative temperature from YAML triggers validation error."""
        _mock_yaml(monkeypatch, {"temperature": -1.0})
        with pytest.raises(ValueError, match="temperature"):
            load_v2_config()

    def test_validation_called_invalid_min_max_acts_raises(self, monkeypatch, _clean_v2_env):
        """min_acts > max_acts from YAML triggers validation error."""
        _mock_yaml(monkeypatch, {"min_acts_per_chapter": 10, "max_acts_per_chapter": 2})
        with pytest.raises(ValueError, match="min_acts_per_chapter"):
            load_v2_config()

    # --- Complex structures from YAML ---

    def test_complex_structures_loaded_from_yaml(self, monkeypatch, _clean_v2_env):
        """genre_catalog and other complex fields come from YAML."""
        catalog = {"contemporary": {"theme": "modern love"}}
        _mock_yaml(monkeypatch, {"genre_catalog": catalog})
        cfg = load_v2_config()
        assert cfg.genre_catalog == catalog

    def test_complex_structures_none_when_absent(self, monkeypatch, _clean_v2_env):
        """Complex fields default to None when not in YAML."""
        _mock_yaml(monkeypatch, {})
        cfg = load_v2_config()
        assert cfg.genre_catalog is None
        assert cfg.common_tropes is None

    # --- callable with no arguments ---

    def test_callable_with_no_args(self, monkeypatch, _clean_v2_env):
        """load_v2_config() with no arguments returns valid V2Config."""
        _mock_yaml(monkeypatch, {})
        cfg = load_v2_config()
        assert isinstance(cfg, V2Config)

    # --- Mixed precedence ---

    def test_mixed_precedence_across_fields(self, monkeypatch, _clean_v2_env):
        """CLI beats YAML; YAML beats default; env vars ignored."""
        _mock_yaml(monkeypatch, {"temperature": 0.3, "max_tokens": 8000})
        monkeypatch.setenv("ROMANCE_FACTORY_V2_MAX_TOKENS", "9000")
        cfg = load_v2_config(embedding_model="custom-model")
        assert cfg.embedding_model == "custom-model"
        assert cfg.max_tokens == 8000
        assert cfg.temperature == 0.3
        assert cfg.default_top_k == 10
