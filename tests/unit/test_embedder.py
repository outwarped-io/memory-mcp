"""Unit tests for the embedder factory + protocol.

Avoids loading the actual sentence-transformers model — full E2E coverage
lives in ``tests/integration/`` once Phase 1 wires the search pipeline.
"""

from __future__ import annotations

import pytest

from memory_mcp.config import Settings
from memory_mcp.embeddings import Embedder, EmbeddingModelMismatch, get_embedder
from memory_mcp.embeddings.azure_openai import AzureOpenAIEmbedder
from memory_mcp.embeddings.local import LocalEmbedder


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "embedder": "local",
        "embedding_model_id": "all-MiniLM-L6-v2",
    }
    base.update(overrides)
    # Pydantic-settings reads env by default; pass via constructor instead.
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def test_factory_returns_local_embedder() -> None:
    e = get_embedder(_settings(embedder="local"))
    assert isinstance(e, LocalEmbedder)
    assert e.model_id == "all-MiniLM-L6-v2"
    assert isinstance(e, Embedder)  # runtime_checkable protocol


def test_factory_returns_azure_stub() -> None:
    e = get_embedder(_settings(embedder="azure_openai", embedding_model_id="text-embedding-3-small"))
    assert isinstance(e, AzureOpenAIEmbedder)
    assert e.model_id == "text-embedding-3-small"

    # Azure stub must fail loud on use, not silently return zeros.
    with pytest.raises(NotImplementedError):
        e.embed_texts(["x"])


def test_factory_rejects_unknown_provider() -> None:
    s = _settings()
    object.__setattr__(s, "embedder", "made-up")  # bypass validation for this test
    with pytest.raises(ValueError, match="unknown embedder"):
        get_embedder(s)


def test_local_embedder_empty_input_returns_empty() -> None:
    e = LocalEmbedder("all-MiniLM-L6-v2")
    # Empty input must NOT trigger a model load.
    assert e.embed_texts([]) == []
    assert e._model is None  # type: ignore[attr-defined]


def test_embedding_model_mismatch_carries_both_ids() -> None:
    err = EmbeddingModelMismatch(expected="env-model", actual="config-model")
    assert err.expected == "env-model"
    assert err.actual == "config-model"
    assert "env-model" in str(err) and "config-model" in str(err)
