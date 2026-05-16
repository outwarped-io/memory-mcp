"""Embedder protocol + factory.

Two providers ship in v1:

* ``local.LocalEmbedder`` — sentence-transformers, runs in-process; the default.
* ``azure_openai.AzureOpenAIEmbedder`` — stub, requires the ``[azure]`` extra
  and Managed Identity / DefaultAzureCredential at runtime.

Each environment owns a ``default_embedding_model_id`` (see ``environments``
table). Search and projection paths must verify that the configured embedder
emits the same model id; mismatch returns ``EMBEDDING_MODEL_MISMATCH``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from memory_mcp.config import Settings
from memory_mcp.errors import EmbeddingModelMismatchError


@runtime_checkable
class Embedder(Protocol):
    """Stable surface for embedders.

    Implementations must be **safe to call from async code** (either natively
    async, or by ensuring blocking work happens off the event loop). The MCP
    server wraps blocking implementations with ``run_in_executor``.
    """

    @property
    def model_id(self) -> str:
        """Identifier persisted alongside vectors, e.g. ``all-MiniLM-L6-v2``."""

    @property
    def dimension(self) -> int:
        """Embedding output dimension; must match Qdrant collection size."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one ``list[float]`` per input text. Empty input → empty output."""


# Backwards-compatible alias — the canonical class lives in ``memory_mcp.errors``.
EmbeddingModelMismatch = EmbeddingModelMismatchError


def get_embedder(settings: Settings) -> Embedder:
    """Factory — returns a singleton-style instance based on settings.

    Local provider lazy-loads the model on first ``embed_texts`` call to keep
    process startup fast (model load is ~1s for MiniLM, ~5s for mpnet).
    """
    if settings.embedder == "local":
        from memory_mcp.embeddings.local import LocalEmbedder

        return LocalEmbedder(settings.embedding_model_id)

    if settings.embedder == "azure_openai":
        from memory_mcp.embeddings.azure_openai import AzureOpenAIEmbedder

        return AzureOpenAIEmbedder(settings)

    raise ValueError(f"unknown embedder provider: {settings.embedder!r}")


__all__ = [
    "Embedder",
    "EmbeddingModelMismatch",
    "EmbeddingModelMismatchError",
    "get_embedder",
]
