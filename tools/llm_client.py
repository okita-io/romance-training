"""
Minimal OpenAI-compatible LLM client.

Works with:
  - LM Studio   http://localhost:1234/v1        (default)
  - Ollama      http://localhost:11434/v1
  - Any OpenAI-compatible endpoint

Configure via env vars or pass explicitly:
  LLM_BASE_URL   e.g. http://localhost:1234/v1
  LLM_API_KEY    e.g. lm-studio  (LM Studio ignores this; include anyway)
  LLM_MODEL      model name as shown in LM Studio / Ollama
  LLM_VISION_MODEL  optional override for vision calls (defaults to LLM_MODEL)

Usage:
    from tools.llm_client import complete, DEFAULT_BASE_URL

    text = complete("Summarise this passage.", system="You are a literary critic.")
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "local-model")
DEFAULT_VISION_MODEL = os.environ.get("LLM_VISION_MODEL", DEFAULT_MODEL)


class LLMError(RuntimeError):
    pass


def _chat_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    stop: list[str] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if stop:
        payload["stop"] = stop
    if extra_body:
        payload.update(extra_body)

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise LLMError(f"HTTP {exc.code} from {url}: {exc.read().decode()[:400]}") from exc
    except OSError as exc:
        raise LLMError(
            f"Cannot reach LLM at {url}.\n"
            "Start LM Studio (enable local server) or Ollama, then retry.\n"
            f"Original error: {exc}"
        ) from exc

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected response shape: {body}") from exc


def image_to_data_url(path: Path) -> str:
    """Encode a local image file as a data: URL for vision APIs."""
    mime, _ = mimetypes.guess_type(path.name)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def complete(
    user_prompt: str,
    *,
    system: str = "You are a helpful assistant.",
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    temperature: float = 0.05,
    max_tokens: int = 2048,
    timeout: int = 180,
    stop: list[str] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """
    Send a chat completion request and return the assistant message as a string.
    Raises LLMError on connection failure or non-200 response.
    """
    return _chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        stop=stop,
        extra_body=extra_body,
    )


def complete_with_images(
    user_prompt: str,
    image_paths: list[Path | str],
    *,
    system: str = "You are a helpful assistant.",
    model: str = DEFAULT_VISION_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    temperature: float = 0.05,
    max_tokens: int = 4096,
    timeout: int = 300,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """
    Send a vision chat completion with one or more local image files.
    Uses the OpenAI-compatible image_url content format.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for raw_path in image_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(path)},
            }
        )

    return _chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body=extra_body,
    )


def check_connection(base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY) -> list[str]:
    """
    Return a list of available model IDs, or raise LLMError if unreachable.
    """
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            return [m["id"] for m in body.get("data", [])]
    except OSError as exc:
        raise LLMError(f"Cannot reach {url}: {exc}") from exc
