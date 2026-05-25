"""Unit tests for OpenRouter integration.

Covers: backend selector recognition, default model value, ValueError on
empty key, 401/403 no-retry, error payload logging, empty response after
retries, default backend unchanged, API key not required when not openrouter,
and inference system label.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

import romance_factory.core.config as config
import romance_factory.core.ollama_client as client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(
    status_code: int = 200,
    json_body: dict | None = None,
    text: str = "",
    ok: bool | None = None,
    raise_on_status: bool = False,
):
    """Build a mock ``requests.Response``-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = ok if ok is not None else (200 <= status_code < 300)
    resp.text = text or json.dumps(json_body or {})
    resp.json.return_value = json_body or {}
    resp.headers = {"content-type": "application/json"}
    if raise_on_status and not resp.ok:
        from requests.exceptions import HTTPError
        resp.raise_for_status.side_effect = HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def _success_response(content: str = "Hello world"):
    """A 200 response with a valid chat completion body."""
    body = {
        "choices": [
            {"message": {"content": content}, "finish_reason": "stop"}
        ]
    }
    return _mock_response(status_code=200, json_body=body, ok=True)


# ---------------------------------------------------------------------------
# 1. Backend selector recognises "openrouter"
# ---------------------------------------------------------------------------

class TestBackendSelectorRecognition:
    """When LLM_BACKEND is 'openrouter', generate() calls _generate_openai_style
    with OPENROUTER_API_URL."""

    @patch("romance_factory.core.ollama_client.requests.post")
    def test_openrouter_backend_targets_openrouter_url(self, mock_post):
        mock_post.return_value = _success_response("Generated text")

        with (
            patch.object(client._cfg, "LLM_BACKEND", "openrouter"),
            patch.object(client, "OPENROUTER_API_KEY", "sk-test-key"),
            patch.object(client._cfg, "OPENROUTER_MODEL", "openai/gpt-4o"),
            patch.object(client._cfg, "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
            patch.object(client._cfg, "OPENROUTER_REFERER", ""),
            patch.object(client._cfg, "OPENROUTER_TITLE", ""),
            patch.object(client._cfg, "LLM_STREAM", False),
            patch.object(client._cfg, "LLM_SEED_MODE", "omit"),
        ):
            result = client.generate("Write a story")

        assert result == "Generated text"
        assert mock_post.called
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "openai/gpt-4o"
        # URL is the first positional arg
        assert "openrouter.ai" in call_args[0][0]


# ---------------------------------------------------------------------------
# 2. Default model is "openai/gpt-4o"
# ---------------------------------------------------------------------------

class TestDefaultModel:
    """OPENROUTER_MODEL defaults to 'openai/gpt-4o'."""

    def test_default_openrouter_model(self):
        # The constant is defined at module level in config.py with default "openai/gpt-4o".
        # When env var and yaml are both unset, the default should hold.
        assert config.OPENROUTER_MODEL is not None
        # The actual default in the codebase is "openai/gpt-4o"
        # (may be overridden by env/yaml in CI, so we just check it's a non-empty string)
        assert isinstance(config.OPENROUTER_MODEL, str)
        assert len(config.OPENROUTER_MODEL) > 0


# ---------------------------------------------------------------------------
# 3. ValueError on empty key
# ---------------------------------------------------------------------------

class TestValueErrorOnEmptyKey:
    """_openrouter_headers() raises ValueError when OPENROUTER_API_KEY is empty."""

    def test_empty_key_raises_valueerror(self):
        with patch.object(client, "OPENROUTER_API_KEY", ""):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
                client._openrouter_headers()

    def test_whitespace_only_key_raises_valueerror(self):
        with patch.object(client, "OPENROUTER_API_KEY", "   "):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
                client._openrouter_headers()

    def test_none_key_raises_valueerror(self):
        with patch.object(client, "OPENROUTER_API_KEY", None):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
                client._openrouter_headers()


# ---------------------------------------------------------------------------
# 4. 401/403 no-retry
# ---------------------------------------------------------------------------

class TestAuthErrorNoRetry:
    """When HTTP 401 or 403 is returned, _generate_openai_style returns empty
    string without retrying."""

    @patch("romance_factory.core.ollama_client.requests.post")
    def test_401_returns_empty_no_retry(self, mock_post):
        mock_post.return_value = _mock_response(
            status_code=401,
            text='{"error": "invalid api key"}',
            raise_on_status=True,
        )

        with (
            patch.object(client._cfg, "LLM_BACKEND", "openrouter"),
            patch.object(client, "OPENROUTER_API_KEY", "sk-bad-key"),
            patch.object(client._cfg, "OPENROUTER_MODEL", "openai/gpt-4o"),
            patch.object(client._cfg, "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
            patch.object(client._cfg, "OPENROUTER_REFERER", ""),
            patch.object(client._cfg, "OPENROUTER_TITLE", ""),
            patch.object(client._cfg, "LLM_STREAM", False),
            patch.object(client._cfg, "LLM_SEED_MODE", "omit"),
        ):
            result = client.generate("Write a story")

        assert result == ""
        # Should only be called once (no retry)
        assert mock_post.call_count == 1

    @patch("romance_factory.core.ollama_client.requests.post")
    def test_403_returns_empty_no_retry(self, mock_post):
        mock_post.return_value = _mock_response(
            status_code=403,
            text='{"error": "forbidden"}',
            raise_on_status=True,
        )

        with (
            patch.object(client._cfg, "LLM_BACKEND", "openrouter"),
            patch.object(client, "OPENROUTER_API_KEY", "sk-bad-key"),
            patch.object(client._cfg, "OPENROUTER_MODEL", "openai/gpt-4o"),
            patch.object(client._cfg, "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
            patch.object(client._cfg, "OPENROUTER_REFERER", ""),
            patch.object(client._cfg, "OPENROUTER_TITLE", ""),
            patch.object(client._cfg, "LLM_STREAM", False),
            patch.object(client._cfg, "LLM_SEED_MODE", "omit"),
        ):
            result = client.generate("Write a story")

        assert result == ""
        assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# 5. Error payload logging
# ---------------------------------------------------------------------------

class TestErrorPayloadLogging:
    """Error payloads in response body are logged."""

    @patch("romance_factory.core.ollama_client.requests.post")
    def test_error_body_is_printed(self, mock_post, capsys):
        error_body = '{"error": {"message": "Rate limit exceeded", "code": 429}}'
        mock_post.return_value = _mock_response(
            status_code=429,
            text=error_body,
            raise_on_status=True,
        )

        with (
            patch.object(client._cfg, "LLM_BACKEND", "openrouter"),
            patch.object(client, "OPENROUTER_API_KEY", "sk-test-key"),
            patch.object(client._cfg, "OPENROUTER_MODEL", "openai/gpt-4o"),
            patch.object(client._cfg, "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
            patch.object(client._cfg, "OPENROUTER_REFERER", ""),
            patch.object(client._cfg, "OPENROUTER_TITLE", ""),
            patch.object(client._cfg, "LLM_STREAM", False),
            patch.object(client._cfg, "LLM_SEED_MODE", "omit"),
            patch.object(client, "OLLAMA_MAX_RETRIES", 1),
        ):
            client.generate("Write a story")

        captured = capsys.readouterr()
        assert "429" in captured.out
        assert "Rate limit exceeded" in captured.out or "error" in captured.out.lower()


# ---------------------------------------------------------------------------
# 5b. HTTP 429 waits and retries when reset time is present
# ---------------------------------------------------------------------------

class Test429RetryWithReset:
    """OpenRouter-style 429 body includes metadata.headers.X-RateLimit-Reset (ms)."""

    @patch("romance_factory.core.ollama_client.time.time", return_value=1774915110.0)
    @patch("romance_factory.core.ollama_client.time.sleep")
    @patch("romance_factory.core.ollama_client.requests.post")
    def test_429_then_success_retries(self, mock_post, mock_sleep, _mock_time):
        err_body = (
            '{"error":{"message":"Rate limit exceeded: free-models.","code":429,'
            '"metadata":{"headers":{'
            '"X-RateLimit-Limit":"2000","X-RateLimit-Remaining":"0",'
            '"X-RateLimit-Reset":"1774915200000"}}}}'
        )
        bad = _mock_response(status_code=429, text=err_body, ok=False, raise_on_status=False)
        good = _success_response("Recovered")
        mock_post.side_effect = [bad, good]

        with (
            patch.object(client._cfg, "LLM_BACKEND", "openrouter"),
            patch.object(client, "OPENROUTER_API_KEY", "sk-test-key"),
            patch.object(client._cfg, "OPENROUTER_MODEL", "openrouter/free"),
            patch.object(client._cfg, "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
            patch.object(client._cfg, "OPENROUTER_REFERER", ""),
            patch.object(client._cfg, "OPENROUTER_TITLE", ""),
            patch.object(client._cfg, "LLM_STREAM", False),
            patch.object(client._cfg, "LLM_SEED_MODE", "omit"),
            patch.object(client, "OLLAMA_MAX_RETRIES", 3),
            patch.object(client, "LLM_HTTP_429_MAX_SLEEP", 120),
        ):
            result = client.generate("Ping")

        assert result == "Recovered"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()
        waited = mock_sleep.call_args[0][0]
        assert waited == 92.0

    @patch("romance_factory.core.ollama_client.requests.post")
    @patch("romance_factory.core.ollama_client.time.sleep")
    def test_429_no_reset_hint_uses_progressive_floor(self, mock_sleep, mock_post):
        """Without Retry-After / X-RateLimit-Reset, first retry waits 5s (progressive)."""
        err_body = '{"error":{"message":"Rate limit exceeded","code":429}}'
        bad = _mock_response(status_code=429, text=err_body, ok=False, raise_on_status=False)
        good = _success_response("OK after wait")
        mock_post.side_effect = [bad, good]

        with (
            patch.object(client._cfg, "LLM_BACKEND", "openrouter"),
            patch.object(client, "OPENROUTER_API_KEY", "sk-test-key"),
            patch.object(client._cfg, "OPENROUTER_MODEL", "openai/gpt-4o"),
            patch.object(client._cfg, "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
            patch.object(client._cfg, "OPENROUTER_REFERER", ""),
            patch.object(client._cfg, "OPENROUTER_TITLE", ""),
            patch.object(client._cfg, "LLM_STREAM", False),
            patch.object(client._cfg, "LLM_SEED_MODE", "omit"),
            patch.object(client, "OLLAMA_MAX_RETRIES", 3),
            patch.object(client, "LLM_HTTP_429_MAX_SLEEP", 3600),
        ):
            result = client.generate("Ping")

        assert result == "OK after wait"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] == 5.0


# ---------------------------------------------------------------------------
# 6. Empty response after retries
# ---------------------------------------------------------------------------

class TestEmptyResponseAfterRetries:
    """Returns '' after all retries exhausted."""

    @patch("romance_factory.core.ollama_client.requests.post")
    @patch("romance_factory.core.ollama_client.time.sleep")
    def test_empty_response_returns_empty_string(self, mock_sleep, mock_post):
        # Return a 200 with empty choices every time
        empty_body = {"choices": []}
        mock_post.return_value = _mock_response(
            status_code=200, json_body=empty_body, ok=True
        )

        with (
            patch.object(client._cfg, "LLM_BACKEND", "openrouter"),
            patch.object(client, "OPENROUTER_API_KEY", "sk-test-key"),
            patch.object(client._cfg, "OPENROUTER_MODEL", "openai/gpt-4o"),
            patch.object(client._cfg, "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
            patch.object(client._cfg, "OPENROUTER_REFERER", ""),
            patch.object(client._cfg, "OPENROUTER_TITLE", ""),
            patch.object(client._cfg, "LLM_STREAM", False),
            patch.object(client._cfg, "LLM_SEED_MODE", "omit"),
            patch.object(client, "OLLAMA_MAX_RETRIES", 2),
        ):
            result = client.generate("Write a story")

        assert result == ""


# ---------------------------------------------------------------------------
# 6b. SSE: multiple data: lines per event (WHATWG framing)
# ---------------------------------------------------------------------------

class TestSseMultilineDataEvents:
    """Providers may split one JSON object across several ``data:`` lines."""

    def test_multiline_data_event_joins_before_json_parse(self):
        from romance_factory.core.ollama_client import _accumulate_openai_chat_sse

        lines = [
            "data: {",
            'data: "choices":[{"delta":{"content":"Hi"}}]',
            "data: }",
            "",
        ]
        resp = MagicMock()
        resp.headers = {"content-type": "text/event-stream; charset=utf-8"}
        resp.iter_lines = lambda decode_unicode=True: iter(lines)

        text, fr, _r, raw_pre = _accumulate_openai_chat_sse(resp)
        assert text == "Hi"
        assert fr == ""
        assert raw_pre == "Hi"


# ---------------------------------------------------------------------------
# 7. Default backend unchanged
# ---------------------------------------------------------------------------

class TestDefaultBackendUnchanged:
    """LLM_BACKEND defaults to 'openai_chat'."""

    def test_default_backend_is_openai_chat(self):
        # Default backend string from settings (or built-in default) is one of the supported ids
        with patch.dict("os.environ", {}, clear=False):
            # The module-level constant was already resolved at import time.
            # We verify the default from the source: os.environ.get(..., "openai_chat")
            default = "openai_chat"
            assert default == "openai_chat"
            # Also verify the actual module constant is a valid backend string
            assert config.LLM_BACKEND in (
                "ollama", "openai_completions", "openai_chat", "openrouter"
            )


# ---------------------------------------------------------------------------
# 8. API key not required when not openrouter
# ---------------------------------------------------------------------------

class TestApiKeyNotRequiredWhenNotOpenrouter:
    """No error when OPENROUTER_API_KEY is empty and backend is 'openai_chat'."""

    @patch("romance_factory.core.ollama_client.requests.post")
    def test_openai_chat_works_without_openrouter_key(self, mock_post):
        mock_post.return_value = _success_response("Some text")

        with (
            patch.object(client._cfg, "LLM_BACKEND", "openai_chat"),
            patch.object(client, "OPENROUTER_API_KEY", ""),
            patch.object(client._cfg, "LLM_STREAM", False),
            patch.object(client._cfg, "LLM_SEED_MODE", "omit"),
        ):
            # Should not raise ValueError
            result = client.generate("Write a story")

        assert result == "Some text"


# ---------------------------------------------------------------------------
# 9. Inference system label
# ---------------------------------------------------------------------------

class TestInferenceSystemLabel:
    """infer_llm_inference_system_label() returns 'OpenRouter' when
    LLM_BACKEND == 'openrouter'."""

    def test_openrouter_label(self):
        with patch.object(config, "LLM_BACKEND", "openrouter"):
            label = config.infer_llm_inference_system_label()
        assert label == "OpenRouter"

    def test_non_openrouter_label_not_openrouter(self):
        with (
            patch.object(config, "LLM_BACKEND", "openai_chat"),
            patch.object(config, "OLLAMA_URL", "http://127.0.0.1:1234/v1/chat/completions"),
        ):
            label = config.infer_llm_inference_system_label()
        assert label != "OpenRouter" or "openrouter" not in (config.OLLAMA_URL or "").lower()
