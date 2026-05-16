"""Azure OpenAI embedder (stub for v1).

Requires the optional ``[azure]`` extra: ``pip install memory-mcp[azure]``.
Authentication uses :class:`azure.identity.DefaultAzureCredential` so it works
unchanged in local dev (Azure CLI), CI (workload identity), and AKS
(managed identity).

Implementation is intentionally deferred to Phase 1.5 — the local embedder
is the documented default for v1 and covers all in-tree tests.
"""

from __future__ import annotations

from collections.abc import Sequence

from memory_mcp.config import Settings


class AzureOpenAIEmbedder:
    """Placeholder; raises on first use to surface the missing wiring loudly."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def model_id(self) -> str:
        return self._settings.embedding_model_id

    @property
    def dimension(self) -> int:
        # Common Azure OpenAI dims: text-embedding-3-small=1536, 3-large=3072.
        # Caller must align Qdrant collection size before flipping the switch.
        raise NotImplementedError(
            "AzureOpenAIEmbedder is not implemented yet. Use embedder='local' "
            "in Phase 1, or vendor in the v1.5 release."
        )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError(
            "AzureOpenAIEmbedder is not implemented yet. Use embedder='local' "
            "in Phase 1, or vendor in the v1.5 release."
        )
