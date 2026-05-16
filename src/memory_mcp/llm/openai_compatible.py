"""OpenAI-compatible LLM client.

Works with any service that implements OpenAI's
``POST /chat/completions`` shape (relative to the configured ``LLM_BASE_URL``):

* OpenAI itself (``LLM_BASE_URL=https://api.openai.com/v1``)
* vLLM (``LLM_BASE_URL=http://vllm:8000/v1``)
* llama.cpp ``--api`` (``LLM_BASE_URL=http://llamacpp:8080/v1``)
* OpenRouter, Together.ai, Groq, …

Bearer auth via ``LLM_API_KEY``. ``LLM_BASE_URL`` MUST be set — there is no
sane default for "openai-compatible".

Azure OpenAI note
-----------------
Azure's URL is ``{endpoint}/openai/deployments/{deployment}/chat/completions``
with an ``api-version`` query param. v1 keeps the path ``/chat/completions``
hardcoded; full Azure-aware URL synthesis lives in the v1.5 ``[azure]`` extra.
Operators wanting Azure today should front it with a small URL-rewriting
reverse proxy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from memory_mcp.config import Settings
from memory_mcp.errors import LLMUnavailableError
from memory_mcp.llm.base import Message

logger = logging.getLogger("memory_mcp.llm.openai_compatible")


class OpenAICompatibleLLMClient:
    """Async client speaking the OpenAI ``/chat/completions`` shape."""

    backend_name: str = "openai_compatible"

    def __init__(
        self,
        settings: Settings,
        *,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        if not settings.llm_base_url:
            raise ValueError(
                "openai_compatible backend requires LLM_BASE_URL to be set "
                "(e.g. https://api.openai.com/v1). No default exists."
            )
        self._base_url = settings.llm_base_url.rstrip("/")

        headers = {"content-type": "application/json"}
        if settings.llm_api_key:
            headers["authorization"] = f"Bearer {settings.llm_api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
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
            "max_tokens": (
                max_tokens if max_tokens is not None else self._settings.llm_max_tokens
            ),
            "temperature": (
                temperature if temperature is not None else self._settings.llm_temperature
            ),
            "stream": False,
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.RequestError as exc:
            raise LLMUnavailableError(
                f"openai_compatible: transport error contacting {self._base_url}: {exc!s}",
                backend="openai_compatible",
                base_url=self._base_url,
            ) from exc

        if response.status_code >= 400:
            body_excerpt = response.text[:200] if response.text else "<empty>"
            raise LLMUnavailableError(
                f"openai_compatible: HTTP {response.status_code} from /chat/completions: {body_excerpt}",
                backend="openai_compatible",
                status_code=response.status_code,
                base_url=self._base_url,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMUnavailableError(
                f"openai_compatible: non-JSON response: {response.text[:200]!r}",
                backend="openai_compatible",
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(
                f"openai_compatible: unexpected response shape: {data!r}",
                backend="openai_compatible",
            ) from exc
        if not isinstance(content, str):
            raise LLMUnavailableError(
                f"openai_compatible: choices[0].message.content is not a string: {content!r}",
                backend="openai_compatible",
            )
        return content

    async def probe(self) -> dict[str, Any]:
        # ``/models`` (relative to base) is the canonical reachability probe
        # for OpenAI-shaped endpoints. vLLM and llama.cpp also implement it.
        # Some private deployments expose only ``/chat/completions``; in
        # that case the probe gracefully reports the 404 as "error" — the
        # operator can still flip ``DREAM_SUMMARIZER`` to template if the
        # server can't be probed.
        try:
            response = await self._client.get("/models", timeout=2.0)
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
        return {
            "status": "ok",
            "base_url": self._base_url,
            "model_id": self.model_id,
        }

    async def aclose(self) -> None:
        await self._client.aclose()
