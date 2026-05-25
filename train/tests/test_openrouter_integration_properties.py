"""Property-based tests for OpenRouter integration.

Uses hypothesis to verify universal correctness properties of the OpenRouter
backend across arbitrary inputs: config resolution precedence, request
construction, attribution headers, streaming flag propagation, retry logic,
backward compatibility, and inference profile correctness.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

import romance_factory.core.config as config
import romance_factory.core.ollama_client as client


# ── Strategies ──────────────────────────────────────────────────────────────

safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=1,
    max_size=100,
).filter(lambda s: s.strip())

safe_text_or_empty = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=0,
    max_size=100,
)

model_name = st.from_regex(r"[a-z]{2,10}/[a-z0-9\-]{2,20}", fullmatch=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _mock_response(status_code=200, json_body=None, text="", ok=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = ok if ok is not None else (200 <= status_code < 300)
    resp.text = text or json.dumps(json_body or {})
    resp.json.return_value = json_body or {}
    resp.headers = {"content-type": "application/json"}
    if not resp.ok:
        from requests.exceptions import HTTPError
        resp.raise_for_status.side_effect = HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def _success_response(content="Hello"):
    body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    return _mock_response(status_code=200, json_body=body, ok=True)



# ── Property 1: Config resolution precedence ───────────────────────────────
# Feature: openrouter-integration, Property 1: Config resolution precedence
# **Validates: Requirements 2.1, 3.1, 4.1, 4.2**


class TestConfigResolutionPrecedence:
    """Property 1: Non-secret keys use yaml > default; secrets use env OR yaml."""

    @given(
        env_val=safe_text,
        yaml_val=safe_text,
    )
    @settings(max_examples=100, deadline=10000)
    def test_openrouter_api_key_env_or_yaml(self, env_val: str, yaml_val: str) -> None:
        """**Validates: Requirements 2.1, 3.1, 4.1, 4.2**

        OPENROUTER_API_KEY: non-empty env wins; else yaml via _y_str.
        """
        with patch.dict(config._Y, {"openrouter_api_key": yaml_val}):
            result = config._y_str("openrouter_api_key", "default_val")
            expected = yaml_val.strip() if yaml_val.strip() else "default_val"
            assert result == expected

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": env_val}):
            resolved = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
            assert resolved == env_val.strip()

    @given(yaml_val=safe_text)
    @settings(max_examples=100, deadline=10000)
    def test_yaml_wins_over_default_when_env_unset(self, yaml_val: str) -> None:
        """**Validates: Requirements 2.1, 3.1, 4.1, 4.2**

        When env var is unset but yaml is set, yaml wins.
        _y_str strips whitespace, so compare against stripped value.
        """
        with patch.dict(config._Y, {"openrouter_model": yaml_val}, clear=False):
            result = config._y_str("openrouter_model", "openai/gpt-4o")
            expected = yaml_val.strip() if yaml_val.strip() else "openai/gpt-4o"
            assert result == expected

    @given(default_val=safe_text)
    @settings(max_examples=100, deadline=10000)
    def test_default_wins_when_both_unset(self, default_val: str) -> None:
        """**Validates: Requirements 2.1, 3.1, 4.1, 4.2**

        When both env var and yaml are unset, default wins.
        """
        clean_y = {k: v for k, v in config._Y.items() if k != "test_key_xyz"}
        with patch.dict(config._Y, clean_y, clear=True):
            result = config._y_str("test_key_xyz", default_val)
            assert result == default_val

    @given(
        env_val=safe_text,
        yaml_val=safe_text,
        default_val=safe_text,
    )
    @settings(max_examples=100, deadline=10000)
    def test_full_precedence_chain(self, env_val: str, yaml_val: str, default_val: str) -> None:
        """**Validates: Requirements 2.1, 3.1, 4.1, 4.2**

        Secret-style chain: non-empty env > yaml > default (OPENROUTER_API_KEY pattern).
        """
        # Case 1: env set -> env wins
        with patch.dict(os.environ, {"TEST_PRECEDENCE_VAR": env_val}):
            with patch.dict(config._Y, {"test_prec_key": yaml_val}, clear=False):
                resolved = (os.environ.get("TEST_PRECEDENCE_VAR") or "").strip() or config._y_str("test_prec_key", default_val)
                if env_val.strip():
                    assert resolved == env_val.strip()
                else:
                    expected_yaml = yaml_val.strip() if yaml_val.strip() else default_val
                    assert resolved == expected_yaml

        # Case 2: env unset -> yaml wins over default
        env_clean = {k: v for k, v in os.environ.items() if k != "TEST_PRECEDENCE_VAR"}
        with patch.dict(os.environ, env_clean, clear=True):
            with patch.dict(config._Y, {"test_prec_key": yaml_val}, clear=False):
                resolved = (os.environ.get("TEST_PRECEDENCE_VAR") or "").strip() or config._y_str("test_prec_key", default_val)
                expected_yaml = yaml_val.strip() if yaml_val.strip() else default_val
                assert resolved == expected_yaml



# ── Property 2: OpenRouter request construction ────────────────────────────
# Feature: openrouter-integration, Property 2: OpenRouter request construction
# **Validates: Requirements 1.2, 1.3, 2.2, 2.4, 3.3, 3.4**


class TestOpenRouterRequestConstruction:
    """Property 2: For any prompt/model, request targets correct URL with correct payload and auth."""

    @given(
        prompt=safe_text,
        system_prompt=st.one_of(st.none(), safe_text),
        model=model_name,
        api_key=safe_text,
    )
    @settings(max_examples=100, deadline=10000)
    def test_request_targets_openrouter_url_with_correct_payload(
        self, prompt: str, system_prompt: str | None, model: str, api_key: str,
    ) -> None:
        """**Validates: Requirements 1.2, 1.3, 2.2, 2.4, 3.3, 3.4**

        HTTP request targets openrouter URL, payload has correct model/messages,
        and Authorization uses OPENROUTER_API_KEY (not OPENAI_API_KEY).
        """
        with patch("romance_factory.core.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _success_response("ok")
            with (
                patch.object(client._cfg, "LLM_BACKEND", "openrouter"),
                patch.object(client, "OPENROUTER_API_KEY", api_key),
                patch.object(client._cfg, "OPENROUTER_MODEL", model),
                patch.object(client._cfg, "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
                patch.object(client._cfg, "OPENROUTER_REFERER", ""),
                patch.object(client._cfg, "OPENROUTER_TITLE", ""),
                patch.object(client._cfg, "LLM_STREAM", False),
                patch.object(client._cfg, "LLM_SEED_MODE", "omit"),
                patch.object(client, "OPENAI_API_KEY", "sk-should-not-appear"),
            ):
                client.generate(prompt, system_prompt=system_prompt)

            assert mock_post.called
            call_url = mock_post.call_args[0][0]
            call_payload = mock_post.call_args[1]["json"]
            call_headers = mock_post.call_args[1]["headers"]

            # URL check
            assert call_url == "https://openrouter.ai/api/v1/chat/completions"

            # Payload checks
            assert call_payload["model"] == model
            assert "messages" in call_payload
            assert "max_tokens" in call_payload
            assert "temperature" in call_payload
            assert "top_p" in call_payload

            # Messages format
            msgs = call_payload["messages"]
            if system_prompt:
                assert msgs[0]["role"] == "system"
                assert msgs[0]["content"] == system_prompt
                assert msgs[-1]["role"] == "user"
                assert msgs[-1]["content"] == prompt
            else:
                assert msgs[0]["role"] == "user"
                assert msgs[0]["content"] == prompt

            # Auth header uses OPENROUTER_API_KEY, not OPENAI_API_KEY
            assert call_headers["Authorization"] == f"Bearer {api_key.strip()}"
            assert "sk-should-not-appear" not in call_headers["Authorization"]



# ── Property 3: Attribution headers reflect configuration ──────────────────
# Feature: openrouter-integration, Property 3: Attribution headers reflect configuration
# **Validates: Requirements 4.3, 4.4, 4.5**


class TestAttributionHeaders:
    """Property 3: For any referer/title combination, headers present when non-empty, absent when empty."""

    @given(
        referer=safe_text_or_empty,
        title=safe_text_or_empty,
        api_key=safe_text,
    )
    @settings(max_examples=100, deadline=10000)
    def test_attribution_headers_reflect_config(
        self, referer: str, title: str, api_key: str,
    ) -> None:
        """**Validates: Requirements 4.3, 4.4, 4.5**

        If referer is non-empty, HTTP-Referer is present; if title is non-empty,
        X-OpenRouter-Title is present; if either is empty, corresponding header is absent.
        """
        with (
            patch.object(client, "OPENROUTER_API_KEY", api_key),
            patch.object(client, "OPENROUTER_REFERER", referer),
            patch.object(client, "OPENROUTER_TITLE", title),
        ):
            headers = client._openrouter_headers()

        # Auth always present
        assert headers["Authorization"] == f"Bearer {api_key.strip()}"

        # Referer
        if referer:
            assert headers.get("HTTP-Referer") == referer
        else:
            assert "HTTP-Referer" not in headers

        # Title
        if title:
            assert headers.get("X-OpenRouter-Title") == title
        else:
            assert "X-OpenRouter-Title" not in headers



# ── Property 4: Streaming flag propagation ─────────────────────────────────
# Feature: openrouter-integration, Property 4: Streaming flag propagation
# **Validates: Requirements 5.1, 5.2**


class TestStreamingFlagPropagation:
    """Property 4: For any boolean stream flag, payload stream field matches."""

    @given(stream_flag=st.booleans())
    @settings(max_examples=100, deadline=10000)
    def test_stream_flag_propagated_to_payload(self, stream_flag: bool) -> None:
        """**Validates: Requirements 5.1, 5.2**

        The request payload's stream field matches the flag passed to generate().
        """
        with patch("romance_factory.core.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _success_response("streamed")
            # Make mock work as context manager for streaming path
            mock_post.return_value.__enter__ = lambda s: s
            mock_post.return_value.__exit__ = MagicMock(return_value=False)
            mock_post.return_value.iter_lines = MagicMock(return_value=iter([]))

            with (
                patch.object(client._cfg, "LLM_BACKEND", "openrouter"),
                patch.object(client, "OPENROUTER_API_KEY", "sk-test"),
                patch.object(client._cfg, "OPENROUTER_MODEL", "openai/gpt-4o"),
                patch.object(client._cfg, "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
                patch.object(client._cfg, "OPENROUTER_REFERER", ""),
                patch.object(client._cfg, "OPENROUTER_TITLE", ""),
                patch.object(client._cfg, "LLM_STREAM", False),
                patch.object(client._cfg, "LLM_SEED_MODE", "omit"),
            ):
                client.generate("test prompt", stream=stream_flag)

            assert mock_post.called
            call_payload = mock_post.call_args[1]["json"]

            # When stream_flag is True, _generate_openai_style sends stream=True
            # via the stream_chat path. When False, the initial payload has stream=False.
            if stream_flag:
                assert call_payload.get("stream") is True
            else:
                assert call_payload.get("stream") is False



# ── Property 5: Retryable errors trigger retries ──────────────────────────
# Feature: openrouter-integration, Property 5: Retryable errors trigger retries
# **Validates: Requirements 6.1**


class TestRetryableErrorsTriggerRetries:
    """Property 5: For any sequence of 5xx then 200, client recovers."""

    @given(
        num_failures=st.integers(min_value=1, max_value=4),
        success_text=safe_text,
    )
    @settings(max_examples=100, deadline=10000)
    def test_retries_recover_after_5xx_errors(
        self, num_failures: int, success_text: str,
    ) -> None:
        """**Validates: Requirements 6.1**

        For N consecutive 5xx responses followed by a 200, the client retries
        and returns the successful response text.
        """
        max_retries = num_failures + 1  # enough retries to reach success

        error_responses = [
            _mock_response(status_code=500, text="server error")
            for _ in range(num_failures)
        ]
        success = _success_response(success_text)
        all_responses = error_responses + [success]

        with patch("romance_factory.core.ollama_client.requests.post") as mock_post:
            mock_post.side_effect = all_responses
            with patch("romance_factory.core.ollama_client.time.sleep"):
                with (
                    patch.object(client._cfg, "LLM_BACKEND", "openrouter"),
                    patch.object(client, "OPENROUTER_API_KEY", "sk-test"),
                    patch.object(client._cfg, "OPENROUTER_MODEL", "openai/gpt-4o"),
                    patch.object(client._cfg, "OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
                    patch.object(client._cfg, "OPENROUTER_REFERER", ""),
                    patch.object(client._cfg, "OPENROUTER_TITLE", ""),
                    patch.object(client._cfg, "LLM_STREAM", False),
                    patch.object(client._cfg, "LLM_SEED_MODE", "omit"),
                    patch.object(client, "OLLAMA_MAX_RETRIES", max_retries),
                ):
                    result = client.generate("test prompt")

        # generate() runs mojibake / smart-quote repair on visible assistant text.
        expected = client._fix_mojibake_text(success_text.strip()).strip()
        assert result == expected
        assert mock_post.call_count == num_failures + 1



# ── Property 6: Backward compatibility ─────────────────────────────────────
# Feature: openrouter-integration, Property 6: Backward compatibility
# **Validates: Requirements 1.4, 8.1, 8.4**


class TestBackwardCompatibility:
    """Property 6: For any existing backend and prompt, request structure unchanged."""

    @given(
        backend=st.sampled_from(["openai_completions", "openai_chat"]),
        prompt=safe_text,
    )
    @settings(max_examples=100, deadline=10000)
    def test_existing_backends_use_ollama_url_and_openai_key(
        self, backend: str, prompt: str,
    ) -> None:
        """**Validates: Requirements 1.4, 8.1, 8.4**

        For existing backends, generate() uses OLLAMA_URL and OPENAI_API_KEY
        (not OpenRouter URL/key).
        """
        test_url = "http://127.0.0.1:1234/v1/chat/completions"
        test_openai_key = "sk-openai-test-key"

        with patch("romance_factory.core.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _success_response("result")
            with (
                patch.object(client._cfg, "LLM_BACKEND", backend),
                patch.object(client._cfg, "OLLAMA_URL", test_url),
                patch.object(client, "OPENAI_API_KEY", test_openai_key),
                patch.object(client._cfg, "MODEL_NAME", "test-model"),
                patch.object(client._cfg, "LLM_STREAM", False),
                patch.object(client._cfg, "LLM_SEED_MODE", "omit"),
            ):
                client.generate(prompt)

            assert mock_post.called
            call_url = mock_post.call_args[0][0]
            call_headers = mock_post.call_args[1]["headers"]
            call_payload = mock_post.call_args[1]["json"]

            # URL should be OLLAMA_URL, not OpenRouter
            assert call_url == test_url
            assert "openrouter.ai" not in call_url

            # Auth should use OPENAI_API_KEY
            if test_openai_key:
                assert call_headers.get("Authorization") == f"Bearer {test_openai_key}"

            # Payload structure
            assert call_payload["model"] == "test-model"
            if backend == "openai_chat":
                assert "messages" in call_payload
            elif backend == "openai_completions":
                assert "prompt" in call_payload



# ── Property 7: Inference profile correctness ──────────────────────────────
# Feature: openrouter-integration, Property 7: Inference profile correctness
# **Validates: Requirements 7.1, 7.2, 7.3**


class TestInferenceProfileCorrectness:
    """Property 7: For any model name, profile dict has correct fields."""

    @given(model=model_name)
    @settings(max_examples=100, deadline=10000)
    def test_inference_profile_fields_for_openrouter(self, model: str) -> None:
        """**Validates: Requirements 7.1, 7.2, 7.3**

        When backend is openrouter, get_llm_inference_profile() returns dict
        with llm_backend == 'openrouter', llm_model == configured model,
        and llm_inference_system == 'OpenRouter'.
        """
        with (
            patch.object(config, "LLM_BACKEND", "openrouter"),
            patch.object(config, "OPENROUTER_MODEL", model),
            patch.object(config, "LLM_INFERENCE_SYSTEM", ""),
        ):
            profile = config.get_llm_inference_profile()

        assert profile["llm_backend"] == "openrouter"
        assert profile["llm_model"] == model
        assert profile["llm_inference_system"] == "OpenRouter"
