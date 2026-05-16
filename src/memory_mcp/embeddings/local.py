"""Local sentence-transformers embedder.

Lazy-loads the underlying model on first call to keep process startup snappy.
Thread-safe loading via a per-instance lock; subsequent calls reuse the cached
model.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any


class LocalEmbedder:
    """Wraps :class:`sentence_transformers.SentenceTransformer`.

    Default model is ``all-MiniLM-L6-v2`` (384-d, ~80MB, fast). Operators may
    point at any HuggingFace model the library accepts; the dimension is
    discovered after first load (we cannot know it without loading).
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model: Any | None = None
        self._dimension: int | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_loaded()
            assert self._dimension is not None  # set by _ensure_loaded
        return self._dimension

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._model_id)
            dim = int(model.get_sentence_embedding_dimension() or 0)
            if dim <= 0:
                raise RuntimeError(
                    f"sentence-transformers model {self._model_id!r} reported "
                    f"non-positive dimension {dim}"
                )
            self._model = model
            self._dimension = dim

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        assert self._model is not None
        # ``encode`` returns numpy by default; flatten to plain Python lists for
        # safe JSON / asyncpg transport.
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in vectors]
