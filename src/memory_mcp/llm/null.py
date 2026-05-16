"""Null LLM client.

Used by:

* Unit tests that want to exercise summarizer code paths without HTTP.
* Deployments running ``DREAM_SUMMARIZER=template`` that still want a
  consistent client object on the runner (e.g. for diagnostics).
* Operators who explicitly disable the LLM subsystem.

Every call raises :class:`LLMUnavailableError`. ``probe`` returns
``{"status": "skipped"}`` so ``/readyz`` doesn't flag a degraded state when
the operator has chosen to disable the LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from memory_mcp.config import Settings
from memory_mcp.errors import LLMUnavailableError
from memory_mcp.llm.base import Message


class NullLLMClient:
    """No-op client. All generative methods raise."""

    backend_name: str = "null"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

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
        raise LLMUnavailableError(
            "LLM_UNAVAILABLE: backend='null' — set LLM_BACKEND to 'ollama' or "
            "'openai_compatible', or switch DREAM_SUMMARIZER='template'.",
            backend="null",
        )

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        raise LLMUnavailableError(
            "LLM_UNAVAILABLE: backend='null' — set LLM_BACKEND to 'ollama' or "
            "'openai_compatible', or switch DREAM_SUMMARIZER='template'.",
            backend="null",
        )

    async def probe(self) -> dict[str, Any]:
        return {
            "status": "skipped",
            "reason": "llm_backend='null' — LLM disabled",
            "backend": "null",
        }

    async def aclose(self) -> None:
        return None
