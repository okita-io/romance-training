"""Property-based tests for PhraseDetectionConfig validation.

Feature: repeated-phrase-detection, Property 15: Configuration Validation

**Validates: Requirements 8.3**
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from romance_factory.generate.phrase_detection.config import PhraseDetectionConfig


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid config values
valid_min_ngram = st.integers(min_value=2, max_value=50)
valid_max_ngram_offset = st.integers(min_value=0, max_value=50)  # added to min
valid_similarity = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_top_k = st.integers(min_value=1, max_value=1000)
valid_max_clusters = st.integers(min_value=1, max_value=1000)


@st.composite
def valid_configs(draw):
    """Generate configs that should pass validation."""
    min_n = draw(valid_min_ngram)
    max_n = min_n + draw(valid_max_ngram_offset)
    return PhraseDetectionConfig(
        min_ngram_words=min_n,
        max_ngram_words=max_n,
        similarity_threshold=draw(valid_similarity),
        top_k_retrieval=draw(valid_top_k),
        max_clusters=draw(valid_max_clusters),
    )


# ---------------------------------------------------------------------------
# Property 15: Configuration Validation
# ---------------------------------------------------------------------------


class TestConfigValidationProperty15:
    """Property 15: Configuration Validation

    For any PhraseDetectionConfig where min_ngram_words < 2, or
    similarity_threshold is outside [0.0, 1.0], or max_ngram_words <
    min_ngram_words, or top_k_retrieval < 1, or max_clusters < 1, the
    pipeline SHALL raise a validation error before starting execution.

    Feature: repeated-phrase-detection, Property 15: Configuration Validation
    """

    @given(min_ngram=st.integers(max_value=1))
    @settings(max_examples=100)
    def test_min_ngram_words_below_2_raises(self, min_ngram: int):
        """min_ngram_words < 2 must raise ValueError."""
        cfg = PhraseDetectionConfig(min_ngram_words=min_ngram, max_ngram_words=max(min_ngram, 12))
        with pytest.raises(ValueError, match="min_ngram_words"):
            cfg.validate()

    @given(
        threshold=st.one_of(
            st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.01, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100)
    def test_similarity_threshold_out_of_range_raises(self, threshold: float):
        """similarity_threshold outside [0.0, 1.0] must raise ValueError."""
        cfg = PhraseDetectionConfig(similarity_threshold=threshold)
        with pytest.raises(ValueError, match="similarity_threshold"):
            cfg.validate()

    @given(data=st.data())
    @settings(max_examples=100)
    def test_max_ngram_below_min_ngram_raises(self, data):
        """max_ngram_words < min_ngram_words must raise ValueError."""
        min_n = data.draw(st.integers(min_value=2, max_value=100))
        max_n = data.draw(st.integers(max_value=min_n - 1))
        cfg = PhraseDetectionConfig(min_ngram_words=min_n, max_ngram_words=max_n)
        with pytest.raises(ValueError, match="max_ngram_words"):
            cfg.validate()

    @given(top_k=st.integers(max_value=0))
    @settings(max_examples=100)
    def test_top_k_retrieval_below_1_raises(self, top_k: int):
        """top_k_retrieval < 1 must raise ValueError."""
        cfg = PhraseDetectionConfig(top_k_retrieval=top_k)
        with pytest.raises(ValueError, match="top_k_retrieval"):
            cfg.validate()

    @given(max_clusters=st.integers(max_value=0))
    @settings(max_examples=100)
    def test_max_clusters_below_1_raises(self, max_clusters: int):
        """max_clusters < 1 must raise ValueError."""
        cfg = PhraseDetectionConfig(max_clusters=max_clusters)
        with pytest.raises(ValueError, match="max_clusters"):
            cfg.validate()

    @given(cfg=valid_configs())
    @settings(max_examples=100)
    def test_valid_config_does_not_raise(self, cfg: PhraseDetectionConfig):
        """A config with all values in valid ranges must not raise."""
        cfg.validate()  # should not raise
