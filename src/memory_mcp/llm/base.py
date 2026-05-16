"""LLM client protocol + factory.

A ``LLMClient`` is the minimal surface dream-mode summarizers need. It is
intentionally narrower than what an LLM SDK exposes: we want exactly two
shapes — ``summarize`` (single-prompt completion) and ``chat`` (multi-turn
messages) — both returning plain strings.

Each backend manages its own HTTP client lifecycle. The factory caches a
singleton per ``Settings`` instance via ``functools.lru_cache``-equivalent
behaviour: callers are expected to use :func:`get_llm_client`. Tests that
need fresh state should call :func:`build_llm_client` directly.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from memory_mcp.config import Settings


class Message(TypedDict):
    """One element of a chat-completion conversation.

    ``role`` matches the OpenAI / Ollama vocabulary. We don't model tool
    calls in v1 — dream summarization needs only ``system`` and ``user``.
    """

    role: Literal["system", "user", "assistant"]
    content: str


@runtime_checkable
class LLMClient(Protocol):
    """Stable contract every backend implements.

    Methods are async because the canonical implementations (ollama,
    openai_compatible) are HTTP-backed. ``NullLLMClient`` provides
    matching async signatures so callers don't branch on backend type.
    """

    @property
    def model_id(self) -> str:
        """Identifier the upstream service understands (e.g. ``llama3.2:3b``)."""

    @property
    def backend_name(self) -> str:
        """One of ``"ollama"``, ``"openai_compatible"``, ``"null"``.

        Recorded in proposal payloads alongside ``model_id`` so reviewers can
        tell at a glance which backend produced a given proposal.
        """

    async def summarize(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return a completion for a single user prompt.

        Implementations MAY internally route through the chat endpoint with a
        single ``user`` message — there's no semantic difference at this layer.
        Raises ``LLMUnavailableError`` on transport / parse failures.
        """

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return the assistant message for a conversation.

        Raises ``LLMUnavailableError`` on transport / parse failures.
        """

    async def probe(self) -> dict[str, Any]:
        """Cheap reachability check used by ``/readyz``.

        MUST NOT raise — return ``{"status": "ok"}`` on success or
        ``{"status": "error", "error": "..."}`` on failure. The probe budget
        is the caller's; implementations should respect a 2-second wall-clock
        cap by default.
        """

    async def aclose(self) -> None:
        """Release any HTTP connections / pools. Safe to call repeatedly."""


def build_llm_client(settings: Settings) -> LLMClient:
    """Construct a fresh client per call. Used by tests and by the runner.

    Lazy-imports backend modules so a deployment with ``llm_backend=null``
    never imports ``httpx`` (the LLM clients' transport).
    """
    backend = settings.llm_backend

    if backend == "null":
        from memory_mcp.llm.null import NullLLMClient

        return NullLLMClient(settings)

    if backend == "ollama":
        from memory_mcp.llm.ollama import OllamaLLMClient

        return OllamaLLMClient(settings)

    if backend == "openai_compatible":
        from memory_mcp.llm.openai_compatible import OpenAICompatibleLLMClient

        return OpenAICompatibleLLMClient(settings)

    raise ValueError(f"unknown llm backend: {backend!r}")


_singleton_lock = asyncio.Lock()
_singleton: LLMClient | None = None
_singleton_settings_id: int | None = None


async def get_llm_client(settings: Settings) -> LLMClient:
    """Process-wide singleton accessor.

    Caches one client per ``Settings`` instance. Calling with a different
    ``Settings`` (e.g. tests building a fresh one) closes the prior client
    and constructs a new one — keeps unit tests simple while making
    production behaviour single-instance / connection-pool-reusing.
    """
    global _singleton, _singleton_settings_id  # noqa: PLW0603 — module-level cache

    settings_id = id(settings)
    async with _singleton_lock:
        if _singleton is not None and _singleton_settings_id == settings_id:
            return _singleton
        if _singleton is not None:
            with contextlib.suppress(Exception):  # best-effort close
                await _singleton.aclose()
        _singleton = build_llm_client(settings)
        _singleton_settings_id = settings_id
        return _singleton


async def reset_llm_client() -> None:
    """Drop the cached singleton — for tests and graceful shutdown."""
    global _singleton, _singleton_settings_id  # noqa: PLW0603

    async with _singleton_lock:
        if _singleton is not None:
            with contextlib.suppress(Exception):
                await _singleton.aclose()
        _singleton = None
        _singleton_settings_id = None


async def probe_llm(settings: Settings) -> dict[str, Any]:
    """``/readyz``-friendly LLM dependency probe.

    Returns ``{"status": "skipped", "reason": "..."}`` when the LLM subsystem
    is intentionally inert (``backend=null`` OR ``dream_summarizer=template``).
    Otherwise constructs (but does not cache) a fresh client and runs its
    backend-specific probe with a 2-second wall-clock budget. Never raises.
    """
    if settings.dream_summarizer == "template":
        return {
            "status": "skipped",
            "reason": "dream_summarizer='template' — LLM not used",
            "backend": settings.llm_backend,
        }
    if settings.llm_backend == "null":
        return {
            "status": "skipped",
            "reason": "llm_backend='null' — LLM disabled",
            "backend": "null",
        }

    try:
        client = build_llm_client(settings)
    except Exception as exc:  # noqa: BLE001 — config errors caught here
        return {
            "status": "error",
            "backend": settings.llm_backend,
            "error": str(exc)[:200],
        }

    try:
        try:
            result = await asyncio.wait_for(client.probe(), timeout=2.0)
        except TimeoutError:
            return {
                "status": "error",
                "backend": settings.llm_backend,
                "error": "probe timed out after 2.0s",
            }
        # Stamp the backend name so monitoring knows which client answered.
        result.setdefault("backend", client.backend_name)
        result.setdefault("model_id", client.model_id)
        return result
    finally:
        with contextlib.suppress(Exception):  # best-effort close
            await client.aclose()


__all__ = [
    "LLMClient",
    "Message",
    "build_llm_client",
    "get_llm_client",
    "probe_llm",
    "reset_llm_client",
]
