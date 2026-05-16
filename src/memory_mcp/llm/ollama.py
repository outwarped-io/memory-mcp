"""Ollama LLM client.

Talks to Ollama's HTTP API (``https://github.com/ollama/ollama/blob/main/docs/api.md``).
Two endpoints are used:

* ``GET  /api/tags`` — probe; lists locally-available models.
* ``POST /api/chat`` — completion. We always route through ``/api/chat`` even
  for single-prompt summarize calls because it gives a uniform response
  shape (one ``message.content`` field) and doesn't lose anything compared
  to ``/api/generate`` for our use.

The client is **non-streaming**: we set ``stream=false`` and read the whole
response. Dream summarization tasks are short and bounded by ``max_tokens``;
streaming would only complicate error handling without latency benefit.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from memory_mcp.config import Settings
from memory_mcp.errors import LLMUnavailableError
from memory_mcp.llm.base import Message

logger = logging.getLogger("memory_mcp.llm.ollama")


_DEFAULT_BASE_URL = "http://ollama:11434"


class OllamaLLMClient:
    """Async Ollama client over httpx.

    A single ``httpx.AsyncClient`` is reused across calls so connection pooling
    benefits multi-call workloads (a single dream pass may emit dozens of
    proposals). ``aclose`` releases the pool; the runner calls it on shutdown.
    """

    backend_name: str = "ollama"

    def __init__(
        self,
        settings: Settings,
        *,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        base_url = settings.llm_base_url or _DEFAULT_BASE_URL
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=5.0),
            transport=_transport,
        )

    @property
    def model_id(self) -> str:
        return self._settings.llm_model_id

    async def summarize(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        return await self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": list(messages),
            "stream": False,
            "options": {
                # ``num_predict`` is Ollama's name for "max tokens to generate".
                "num_predict": (
                    max_tokens if max_tokens is not None else self._settings.llm_max_tokens
                ),
                "temperature": (
                    temperature if temperature is not None else self._settings.llm_temperature
                ),
            },
        }
        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.RequestError as exc:
            raise LLMUnavailableError(
                f"ollama: transport error contacting {self._base_url}: {exc!s}",
                backend="ollama",
                base_url=self._base_url,
            ) from exc

        if response.status_code >= 400:
            body_excerpt = response.text[:200] if response.text else "<empty>"
            raise LLMUnavailableError(
                f"ollama: HTTP {response.status_code} from /api/chat: {body_excerpt}",
                backend="ollama",
                status_code=response.status_code,
                base_url=self._base_url,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMUnavailableError(
                f"ollama: non-JSON response from /api/chat: {response.text[:200]!r}",
                backend="ollama",
            ) from exc

        message = data.get("message")
        if not isinstance(message, dict):
            raise LLMUnavailableError(
                f"ollama: unexpected response shape (no message dict): {data!r}",
                backend="ollama",
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMUnavailableError(
                f"ollama: unexpected response shape (no string content): {data!r}",
                backend="ollama",
            )
        return content

    async def probe(self) -> dict[str, Any]:
        try:
            response = await self._client.get("/api/tags", timeout=2.0)
        except httpx.RequestError as exc:
            return {
                "status": "error",
                "error": f"transport: {exc!s}"[:200],
                "base_url": self._base_url,
            }
        if response.status_code >= 400:
            return {
                "status": "error",
                "error": f"HTTP {response.status_code}: {response.text[:120]}",
                "base_url": self._base_url,
            }
        # Optionally surface whether the configured model is locally present
        # — useful for the operator but non-fatal if missing (Ollama will
        # pull on first use).
        model_present: bool | None = None
        try:
            data = response.json()
            tags = data.get("models") or []
            names = {m.get("name") for m in tags if isinstance(m, dict)}
            model_present = self.model_id in names
        except Exception:  # noqa: BLE001 — best-effort enrichment only
            model_present = None
        return {
            "status": "ok",
            "base_url": self._base_url,
            "model_id": self.model_id,
            **({"model_present": model_present} if model_present is not None else {}),
        }

    async def aclose(self) -> None:
        await self._client.aclose()
