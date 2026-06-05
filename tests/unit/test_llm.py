"""Unit tests for the LLM client abstraction.

Covers:

* Factory selection per ``LLM_BACKEND`` (null / ollama / openai_compatible)
* Factory rejection of unknown backends
* ``NullLLMClient`` raising ``LLMUnavailableError`` on every generative call
* HTTP-backed clients exercising success paths via ``httpx.MockTransport``
* HTTP-backed clients propagating transport / HTTP / parse failures as
  ``LLMUnavailableError`` (with stable ``code`` "LLM_UNAVAILABLE")
* ``probe_llm`` returning ``skipped`` when the LLM is intentionally inert
  and ``ok`` / ``error`` when actually probing a backend.

No real HTTP traffic — every test uses ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from memory_mcp.config import Settings
from memory_mcp.errors import LLMUnavailableError
from memory_mcp.llm import (
    LLMClient,
    Message,
    build_llm_client,
    probe_llm,
)
from memory_mcp.llm.null import NullLLMClient
from memory_mcp.llm.ollama import OllamaLLMClient
from memory_mcp.llm.openai_compatible import OpenAICompatibleLLMClient


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_backend": "null",
        "llm_model_id": "test-model",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_returns_null_client_by_default() -> None:
    client = build_llm_client(_settings())
    assert isinstance(client, NullLLMClient)
    assert isinstance(client, LLMClient)
    assert client.backend_name == "null"
    assert client.model_id == "test-model"


def test_factory_returns_ollama_client() -> None:
    client = build_llm_client(_settings(llm_backend="ollama"))
    assert isinstance(client, OllamaLLMClient)
    assert client.backend_name == "ollama"


def test_factory_returns_openai_compatible_client() -> None:
    client = build_llm_client(
        _settings(
            llm_backend="openai_compatible",
            llm_base_url="http://vllm.test/v1",
            llm_api_key="sk-test",
        )
    )
    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.backend_name == "openai_compatible"


def test_factory_rejects_unknown_backend() -> None:
    s = _settings()
    object.__setattr__(s, "llm_backend", "made-up")  # bypass validator
    with pytest.raises(ValueError, match="unknown llm backend"):
        build_llm_client(s)


def test_openai_compatible_requires_base_url() -> None:
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        build_llm_client(_settings(llm_backend="openai_compatible"))


# ---------------------------------------------------------------------------
# Null backend — every generative call MUST raise LLM_UNAVAILABLE.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_null_summarize_raises_llm_unavailable() -> None:
    client = NullLLMClient(_settings())
    with pytest.raises(LLMUnavailableError) as exc_info:
        await client.summarize("anything", max_tokens=10)
    assert exc_info.value.code == "LLM_UNAVAILABLE"
    assert exc_info.value.details["backend"] == "null"


@pytest.mark.asyncio
async def test_null_chat_raises_llm_unavailable() -> None:
    client = NullLLMClient(_settings())
    with pytest.raises(LLMUnavailableError):
        await client.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_null_probe_returns_skipped() -> None:
    result = await NullLLMClient(_settings()).probe()
    assert result["status"] == "skipped"
    assert result["backend"] == "null"


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------


def _ollama_handler_factory(
    *,
    chat_response: dict[str, Any] | None = None,
    chat_status: int = 200,
    chat_body_text: str | None = None,
    tags_response: dict[str, Any] | None = None,
    tags_status: int = 200,
    raise_transport: Exception | None = None,
):
    """Build a MockTransport handler with controllable responses."""
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if raise_transport is not None:
            raise raise_transport
        if request.url.path == "/api/chat":
            if chat_body_text is not None:
                return httpx.Response(chat_status, content=chat_body_text)
            return httpx.Response(
                chat_status,
                json=chat_response
                or {
                    "model": "test-model",
                    "message": {"role": "assistant", "content": "default-response"},
                    "done": True,
                },
            )
        if request.url.path == "/api/tags":
            return httpx.Response(
                tags_status,
                json=tags_response or {"models": [{"name": "test-model"}]},
            )
        return httpx.Response(404, json={"error": f"unexpected path {request.url.path}"})

    return handler, captured_requests


def _make_ollama(
    settings_overrides: dict[str, Any] | None = None,
    **handler_kwargs: Any,
) -> tuple[OllamaLLMClient, list[httpx.Request]]:
    handler, captured = _ollama_handler_factory(**handler_kwargs)
    transport = httpx.MockTransport(handler)
    s = _settings(llm_backend="ollama", **(settings_overrides or {}))
    client = OllamaLLMClient(s, _transport=transport)
    return client, captured


@pytest.mark.asyncio
async def test_ollama_summarize_round_trip() -> None:
    client, captured = _make_ollama(
        chat_response={
            "model": "test-model",
            "message": {"role": "assistant", "content": "summarized text"},
            "done": True,
        }
    )
    try:
        result = await client.summarize("hello", max_tokens=64, temperature=0.5)
    finally:
        await client.aclose()
    assert result == "summarized text"
    assert len(captured) == 1
    req = captured[0]
    assert req.url.path == "/api/chat"
    payload = json.loads(req.content)
    assert payload["model"] == "test-model"
    assert payload["stream"] is False
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["options"]["num_predict"] == 64
    assert payload["options"]["temperature"] == 0.5


@pytest.mark.asyncio
async def test_ollama_chat_passes_messages_through() -> None:
    client, captured = _make_ollama()
    msgs: list[Message] = [
        {"role": "system", "content": "You summarize."},
        {"role": "user", "content": "Summarize this."},
    ]
    try:
        await client.chat(msgs)
    finally:
        await client.aclose()
    payload = json.loads(captured[0].content)
    assert payload["messages"] == list(msgs)


@pytest.mark.asyncio
async def test_ollama_uses_settings_defaults_for_max_tokens_and_temperature() -> None:
    client, captured = _make_ollama(settings_overrides={"llm_max_tokens": 123, "llm_temperature": 0.9})
    try:
        await client.summarize("x")
    finally:
        await client.aclose()
    payload = json.loads(captured[0].content)
    assert payload["options"]["num_predict"] == 123
    assert payload["options"]["temperature"] == 0.9


@pytest.mark.asyncio
async def test_ollama_propagates_transport_error_as_llm_unavailable() -> None:
    client, _ = _make_ollama(raise_transport=httpx.ConnectError("connection refused"))
    try:
        with pytest.raises(LLMUnavailableError) as exc_info:
            await client.summarize("x")
    finally:
        await client.aclose()
    assert exc_info.value.code == "LLM_UNAVAILABLE"
    assert exc_info.value.details["backend"] == "ollama"


@pytest.mark.asyncio
async def test_ollama_propagates_http_5xx_as_llm_unavailable() -> None:
    client, _ = _make_ollama(
        chat_status=500,
        chat_body_text="internal server error",
    )
    try:
        with pytest.raises(LLMUnavailableError) as exc_info:
            await client.summarize("x")
    finally:
        await client.aclose()
    assert exc_info.value.code == "LLM_UNAVAILABLE"
    assert exc_info.value.details["status_code"] == 500


@pytest.mark.asyncio
async def test_ollama_handles_non_json_response() -> None:
    client, _ = _make_ollama(chat_body_text="not-json")
    try:
        with pytest.raises(LLMUnavailableError, match="non-JSON"):
            await client.summarize("x")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_ollama_handles_unexpected_response_shape() -> None:
    # No "message" key — simulates an upstream bug or version skew.
    client, _ = _make_ollama(chat_response={"model": "x", "done": True})
    try:
        with pytest.raises(LLMUnavailableError, match="no message dict"):
            await client.summarize("x")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_ollama_probe_reports_ok_when_model_present() -> None:
    client, _ = _make_ollama(tags_response={"models": [{"name": "test-model"}, {"name": "other"}]})
    try:
        result = await client.probe()
    finally:
        await client.aclose()
    assert result["status"] == "ok"
    assert result["model_id"] == "test-model"
    assert result["model_present"] is True


@pytest.mark.asyncio
async def test_ollama_probe_reports_error_on_5xx() -> None:
    client, _ = _make_ollama(tags_status=503)
    try:
        result = await client.probe()
    finally:
        await client.aclose()
    assert result["status"] == "error"
    assert "503" in result["error"]


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------


def _make_openai(
    *,
    chat_response: dict[str, Any] | None = None,
    chat_status: int = 200,
    chat_body_text: str | None = None,
    models_status: int = 200,
    api_key: str | None = "sk-test",
    raise_transport: Exception | None = None,
) -> tuple[OpenAICompatibleLLMClient, list[httpx.Request]]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if raise_transport is not None:
            raise raise_transport
        if request.url.path.endswith("/chat/completions"):
            if chat_body_text is not None:
                return httpx.Response(chat_status, content=chat_body_text)
            return httpx.Response(
                chat_status,
                json=chat_response
                or {
                    "id": "x",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        if request.url.path.endswith("/models"):
            return httpx.Response(models_status, json={"data": [{"id": "test-model"}]})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    s = _settings(
        llm_backend="openai_compatible",
        llm_base_url="http://vllm.test/v1",
        llm_api_key=api_key,
    )
    client = OpenAICompatibleLLMClient(s, _transport=transport)
    return client, captured


@pytest.mark.asyncio
async def test_openai_compatible_summarize_round_trip() -> None:
    client, captured = _make_openai(
        chat_response={"choices": [{"message": {"role": "assistant", "content": "hello world"}}]}
    )
    try:
        result = await client.summarize("ping", max_tokens=42, temperature=0.1)
    finally:
        await client.aclose()
    assert result == "hello world"
    req = captured[0]
    assert req.headers["authorization"] == "Bearer sk-test"
    payload = json.loads(req.content)
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == 42
    assert payload["temperature"] == 0.1
    assert payload["stream"] is False
    assert payload["messages"] == [{"role": "user", "content": "ping"}]


@pytest.mark.asyncio
async def test_openai_compatible_omits_authorization_when_no_key() -> None:
    client, captured = _make_openai(api_key=None)
    try:
        await client.summarize("x")
    finally:
        await client.aclose()
    assert "authorization" not in captured[0].headers


@pytest.mark.asyncio
async def test_openai_compatible_propagates_4xx_as_llm_unavailable() -> None:
    client, _ = _make_openai(chat_status=401, chat_body_text='{"error":"bad-key"}')
    try:
        with pytest.raises(LLMUnavailableError) as exc_info:
            await client.summarize("x")
    finally:
        await client.aclose()
    assert exc_info.value.details["status_code"] == 401


@pytest.mark.asyncio
async def test_openai_compatible_handles_unexpected_shape() -> None:
    client, _ = _make_openai(chat_response={"choices": []})
    try:
        with pytest.raises(LLMUnavailableError, match="unexpected response shape"):
            await client.summarize("x")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_handles_non_string_content() -> None:
    client, _ = _make_openai(chat_response={"choices": [{"message": {"role": "assistant", "content": None}}]})
    try:
        with pytest.raises(LLMUnavailableError, match="not a string"):
            await client.summarize("x")
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# probe_llm — readyz integration helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_llm_skipped_when_summarizer_template() -> None:
    s = _settings(llm_backend="ollama", dream_summarizer="template")
    result = await probe_llm(s)
    assert result["status"] == "skipped"
    assert "template" in result["reason"]


@pytest.mark.asyncio
async def test_probe_llm_skipped_when_backend_null() -> None:
    s = _settings(llm_backend="null", dream_summarizer="llm")
    result = await probe_llm(s)
    assert result["status"] == "skipped"
    assert result["backend"] == "null"


@pytest.mark.asyncio
async def test_probe_llm_returns_error_when_misconfigured() -> None:
    # openai_compatible without LLM_BASE_URL → factory raises ValueError →
    # probe_llm catches and returns error (never raises).
    s = _settings(llm_backend="openai_compatible", dream_summarizer="llm")
    result = await probe_llm(s)
    assert result["status"] == "error"
    assert "LLM_BASE_URL" in result["error"]
