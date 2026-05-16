"""Embedding providers.

* ``local.LocalEmbedder`` — sentence-transformers, default
* ``azure_openai.AzureOpenAIEmbedder`` — managed-identity auth via DefaultAzureCredential
"""

from memory_mcp.embeddings.base import (
    Embedder,
    EmbeddingModelMismatch,
    EmbeddingModelMismatchError,
    get_embedder,
)

__all__ = [
    "Embedder",
    "EmbeddingModelMismatch",
    "EmbeddingModelMismatchError",
    "get_embedder",
]

