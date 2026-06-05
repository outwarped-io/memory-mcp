"""memory-mcp Python client SDK.

Async-only, typed, namespaced wrapper around the 45 MCP tools exposed by
the memory-mcp server. Designed to be used by any Python agent that needs
shared cross-session memory.

Quick start::

    from memory_mcp_client import MemoryClient

    async with MemoryClient("http://127.0.0.1:8080/mcp") as client:
        envs = await client.envs.list_()
        result = await client.memories.search(
            query="transport on AKS",
            env_ids=[envs[0].id],
            top_k=10,
        )
        for hit in result.hits:
            print(hit.memory.title, hit.score)

The SDK is async-native; there is no sync facade in v1.
"""

from __future__ import annotations

from memory_mcp_client._batch import BatchFailure, BatchResult
from memory_mcp_client._retry import RetryPolicy
from memory_mcp_client.client import MemoryClient
from memory_mcp_client.errors import (
    AlreadyExistsError,
    AuthError,
    ConflictError,
    CycleDetectedError,
    EmbeddingModelMismatchError,
    EnvAmbiguousError,
    EnvNotAttachedError,
    ForbiddenEnvError,
    GraphBackendUnavailableError,
    InternalError,
    InvalidCursorError,
    InvalidInputError,
    InvalidTransitionError,
    LLMUnavailableError,
    MemoryMCPError,
    NotFoundError,
    RateLimitedError,
    RetryExhaustedError,
    SessionRequiredError,
    UnauthorizedError,
    UnknownError,
    ValidationError,
    ValidationFailedError,
    VersionConflictError,
)

__version__ = "0.2.0"

__all__ = [
    "MemoryClient",
    "BatchFailure",
    "BatchResult",
    "MemoryMCPError",
    "RetryPolicy",
    "RetryExhaustedError",
    "VersionConflictError",
    "InvalidTransitionError",
    "NotFoundError",
    "AlreadyExistsError",
    "EnvAmbiguousError",
    "EnvNotAttachedError",
    "SessionRequiredError",
    "EmbeddingModelMismatchError",
    "InvalidCursorError",
    "InvalidInputError",
    "ValidationError",
    "ValidationFailedError",
    "CycleDetectedError",
    "GraphBackendUnavailableError",
    "LLMUnavailableError",
    "UnauthorizedError",
    "ForbiddenEnvError",
    "AuthError",
    "ConflictError",
    "RateLimitedError",
    "InternalError",
    "UnknownError",
]
