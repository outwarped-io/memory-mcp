"""Tests for env ops re-embedding adapter."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from memory_mcp.env_ops import _embed


class FakeEmbedder:
    """Deterministic embedder for adapter tests."""

    model_id = "target-model"
    dimension = 1024

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1] * 1024 for _ in texts]


@pytest.mark.asyncio
async def test_maybe_re_embed_returns_empty_when_models_match() -> None:
    memory = SimpleNamespace(id=uuid4(), body="body")

    vectors = await _embed.maybe_re_embed([memory], "same-model", "same-model", session=None)  # type: ignore[arg-type]

    assert vectors == {}


@pytest.mark.asyncio
async def test_maybe_re_embed_embeds_each_memory_when_models_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeEmbedder()
    monkeypatch.setattr(_embed, "get_embedder", lambda settings: fake)
    monkeypatch.setattr(_embed, "get_settings", lambda: object())
    memory_a = SimpleNamespace(id=uuid4(), body="alpha")
    memory_b = SimpleNamespace(id=uuid4(), body="bravo")

    vectors = await _embed.maybe_re_embed(
        [memory_a, memory_b],
        "source-model",
        "target-model",
        session=None,  # type: ignore[arg-type]
    )

    assert vectors == {memory_a.id: [0.1] * 1024, memory_b.id: [0.1] * 1024}
    assert fake.calls == [["alpha"], ["bravo"]]
