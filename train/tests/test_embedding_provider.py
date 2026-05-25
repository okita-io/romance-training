"""Unit tests for EmbeddingProvider."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from romance_factory.generate.embedding_provider import (
    EmbeddingError,
    EmbeddingProvider,
    _MODEL_ALIASES,
    _resolve_model_id,
)


# ---------------------------------------------------------------------------
# Helper: build a mock SentenceTransformer class whose instances behave
# like real models (encode → numpy array, get_sentence_embedding_dimension).
# ---------------------------------------------------------------------------


def _mock_st_class(dim: int = 384):
    """Return a mock *class* that produces mock model instances."""
    mock_cls = MagicMock()
    instance = MagicMock()
    instance.get_sentence_embedding_dimension.return_value = dim
    instance.encode.return_value = np.random.rand(dim).astype("float32")
    mock_cls.return_value = instance
    return mock_cls, instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveModelId:
    """Alias resolution helper."""

    def test_bge_large_alias(self):
        assert _resolve_model_id("bge-large") == "BAAI/bge-large-en-v1.5"

    def test_e5_large_alias(self):
        assert _resolve_model_id("e5-large") == "intfloat/e5-large-v2"

    def test_unknown_passes_through(self):
        assert _resolve_model_id("org/custom") == "org/custom"


class TestEmbeddingProviderInit:
    """Constructor and model loading."""

    def test_alias_bge_large(self):
        mock_cls, _ = _mock_st_class(1024)
        with patch.dict(sys.modules, {"sentence_transformers": MagicMock(SentenceTransformer=mock_cls)}):
            provider = EmbeddingProvider("bge-large")
        mock_cls.assert_called_once_with("BAAI/bge-large-en-v1.5")
        assert provider.dimensionality == 1024

    def test_alias_e5_large(self):
        mock_cls, _ = _mock_st_class(1024)
        with patch.dict(sys.modules, {"sentence_transformers": MagicMock(SentenceTransformer=mock_cls)}):
            provider = EmbeddingProvider("e5-large")
        mock_cls.assert_called_once_with("intfloat/e5-large-v2")
        assert provider.dimensionality == 1024

    def test_direct_hf_id(self):
        mock_cls, _ = _mock_st_class(768)
        with patch.dict(sys.modules, {"sentence_transformers": MagicMock(SentenceTransformer=mock_cls)}):
            provider = EmbeddingProvider("some-org/custom-model")
        mock_cls.assert_called_once_with("some-org/custom-model")
        assert provider.dimensionality == 768

    def test_load_failure_raises_embedding_error(self):
        mock_cls = MagicMock(side_effect=RuntimeError("model not found"))
        with patch.dict(sys.modules, {"sentence_transformers": MagicMock(SentenceTransformer=mock_cls)}):
            with pytest.raises(EmbeddingError, match="Failed to load"):
                EmbeddingProvider("nonexistent/model")


class TestEmbed:
    """embed() method."""

    def test_returns_list_of_floats(self):
        dim = 384
        mock_cls, mock_model = _mock_st_class(dim)
        expected = np.ones(dim, dtype="float32")
        mock_model.encode.return_value = expected
        with patch.dict(sys.modules, {"sentence_transformers": MagicMock(SentenceTransformer=mock_cls)}):
            provider = EmbeddingProvider("bge-large")

        result = provider.embed("hello world")
        assert isinstance(result, list)
        assert len(result) == dim
        assert all(isinstance(v, float) for v in result)

    def test_dimensionality_matches_output(self):
        dim = 512
        mock_cls, mock_model = _mock_st_class(dim)
        mock_model.encode.return_value = np.zeros(dim, dtype="float32")
        with patch.dict(sys.modules, {"sentence_transformers": MagicMock(SentenceTransformer=mock_cls)}):
            provider = EmbeddingProvider("e5-large")

        vec = provider.embed("test")
        assert len(vec) == provider.dimensionality

    def test_encode_failure_raises_embedding_error(self):
        mock_cls, mock_model = _mock_st_class()
        mock_model.encode.side_effect = RuntimeError("OOM")
        with patch.dict(sys.modules, {"sentence_transformers": MagicMock(SentenceTransformer=mock_cls)}):
            provider = EmbeddingProvider("bge-large")

        with pytest.raises(EmbeddingError, match="Embedding failed"):
            provider.embed("some text")


class TestModelAliases:
    """Alias mapping coverage."""

    def test_known_aliases(self):
        assert "bge-large" in _MODEL_ALIASES
        assert "e5-large" in _MODEL_ALIASES
        assert _MODEL_ALIASES["bge-large"] == "BAAI/bge-large-en-v1.5"
        assert _MODEL_ALIASES["e5-large"] == "intfloat/e5-large-v2"
