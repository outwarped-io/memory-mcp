"""LLM client abstraction (Phase 2.2).

Three backends ship in v1:

* ``ollama.OllamaLLMClient`` — talks HTTP to a sidecar ``ollama`` container.
  Recommended local-only default; runs entirely offline once the model is
  pulled. Compose enables it via ``--profile llm``.
* ``openai_compatible.OpenAICompatibleLLMClient`` — any OpenAI-shaped chat
  completion endpoint: OpenAI, Azure OpenAI, vLLM, llama.cpp ``--api``,
  OpenRouter. Bearer auth via ``LLM_API_KEY``.
* ``null.NullLLMClient`` — raises ``LLMUnavailableError`` on every call.
  Used by tests and by deployments that route all dream summarization
  through ``TemplateSummarizer``.

Selection happens via :func:`get_llm_client` driven by ``Settings.llm_backend``.
The factory is **lazy-import per backend** so a ``DREAM_SUMMARIZER=template``
deployment never pays the import cost of an HTTP-backed client it will never
construct.
"""

from memory_mcp.llm.base import (
    LLMClient,
    Message,
    build_llm_client,
    get_llm_client,
    probe_llm,
    reset_llm_client,
)

__all__ = [
    "LLMClient",
    "Message",
    "build_llm_client",
    "get_llm_client",
    "probe_llm",
    "reset_llm_client",
]
