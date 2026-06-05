"""Unit tests for config + lifecycle helpers."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from memory_mcp.config import Settings, get_settings
from memory_mcp.db.types import (
    MemoryStatus,
    is_valid_transition,
)


def test_settings_defaults_match_env_example() -> None:
    """The defaults shipped in code should be safe even with no .env file."""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.embedder == "local"
    assert s.embedding_model_id == "all-MiniLM-L6-v2"
    assert s.vector_backend == "qdrant"
    assert s.graph_backend == "neo4j"
    assert s.mcp_http_port == 8080


def test_settings_env_override() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "EMBEDDER": "azure_openai",
            "EMBEDDING_MODEL_ID": "text-embedding-3-small",
            "MCP_HTTP_PORT": "9090",
            "VECTOR_BACKEND": "pgvector",
        },
        clear=False,
    ):
        s = Settings(_env_file=None)  # type: ignore[call-arg]

    assert s.embedder == "azure_openai"
    assert s.embedding_model_id == "text-embedding-3-small"
    assert s.mcp_http_port == 9090
    assert s.vector_backend == "pgvector"


def test_settings_rejects_invalid_embedder() -> None:
    with (
        mock.patch.dict(os.environ, {"EMBEDDER": "totally_made_up"}, clear=False),
        pytest.raises(ValueError),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_rejects_negative_dream_salience_weights() -> None:
    """``dream_salience_*`` weights have ``ge=0`` constraints to prevent
    operator-misconfigurations that would invert semantics (e.g. a
    negative ``w_negative`` would *reward* negative feedback)."""
    for var in (
        "DREAM_SALIENCE_W_ACCESS",
        "DREAM_SALIENCE_W_NEGATIVE",
        "DREAM_SALIENCE_PINNED_BONUS",
        "DREAM_SALIENCE_VERIFIED_BONUS",
    ):
        with (
            mock.patch.dict(os.environ, {var: "-0.1"}, clear=False),
            pytest.raises(ValueError),
        ):
            Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_rejects_zero_recency_tau() -> None:
    """``recency_tau`` and ``verified_tau`` must be strictly positive
    (used as denominators in ``exp(-Δt / τ)``)."""
    for var in (
        "DREAM_SALIENCE_RECENCY_TAU_SECONDS",
        "DREAM_SALIENCE_VERIFIED_TAU_SECONDS",
    ):
        with (
            mock.patch.dict(os.environ, {var: "0"}, clear=False),
            pytest.raises(ValueError),
        ):
            Settings(_env_file=None)  # type: ignore[call-arg]


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Lifecycle transition matrix — must mirror design.md / migration CHECKs.
# ---------------------------------------------------------------------------

# Each (src, dst, expected) triple. Idempotent (src==dst) always True.
TRANSITIONS = [
    # proposed → ...
    (MemoryStatus.proposed, MemoryStatus.active, True),
    (MemoryStatus.proposed, MemoryStatus.archived, False),
    (MemoryStatus.proposed, MemoryStatus.retired, True),
    (MemoryStatus.proposed, MemoryStatus.stale, False),
    (MemoryStatus.proposed, MemoryStatus.superseded, False),
    # active → ...
    (MemoryStatus.active, MemoryStatus.stale, True),
    (MemoryStatus.active, MemoryStatus.archived, True),
    (MemoryStatus.active, MemoryStatus.superseded, True),
    (MemoryStatus.active, MemoryStatus.retired, True),
    (MemoryStatus.active, MemoryStatus.proposed, False),
    # stale → ...
    (MemoryStatus.stale, MemoryStatus.active, True),
    (MemoryStatus.stale, MemoryStatus.archived, True),
    (MemoryStatus.stale, MemoryStatus.superseded, True),
    (MemoryStatus.stale, MemoryStatus.retired, True),
    (MemoryStatus.stale, MemoryStatus.proposed, False),
    # archived → ...
    (MemoryStatus.archived, MemoryStatus.active, True),
    (MemoryStatus.archived, MemoryStatus.superseded, True),
    (MemoryStatus.archived, MemoryStatus.retired, True),
    (MemoryStatus.archived, MemoryStatus.stale, False),
    # superseded → ...
    (MemoryStatus.superseded, MemoryStatus.retired, True),
    (MemoryStatus.superseded, MemoryStatus.active, False),
    (MemoryStatus.superseded, MemoryStatus.stale, False),
    (MemoryStatus.superseded, MemoryStatus.archived, False),
    # retired is terminal
    (MemoryStatus.retired, MemoryStatus.active, False),
    (MemoryStatus.retired, MemoryStatus.archived, False),
    (MemoryStatus.retired, MemoryStatus.proposed, False),
    (MemoryStatus.retired, MemoryStatus.stale, False),
    (MemoryStatus.retired, MemoryStatus.superseded, False),
]


@pytest.mark.parametrize("src,dst,allowed", TRANSITIONS)
def test_lifecycle_transition_matrix(src: MemoryStatus, dst: MemoryStatus, allowed: bool) -> None:
    assert is_valid_transition(src, dst) is allowed


@pytest.mark.parametrize("status", list(MemoryStatus))
def test_lifecycle_idempotent(status: MemoryStatus) -> None:
    assert is_valid_transition(status, status) is True
