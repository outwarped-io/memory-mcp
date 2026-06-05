"""Test fixtures + fake MCP session for memory-mcp-client tests.

The client's ``MemoryClient`` accepts a ``session_factory`` hook that
bypasses the real Streamable-HTTP plumbing and returns a fake
``ClientSession``-shaped object. Tests use :class:`FakeClientSession`
(below) to record dispatched tool calls and replay scripted responses,
exercising the dispatcher / typed-error / namespace plumbing without
spinning up a real server.

A ``FakeClientSession`` is async-context-manageable so the client's
``AsyncExitStack`` can manage its lifetime exactly like a real session.

Scripting model
---------------
Tests configure responses two ways:

1. ``set_response(name, payload)`` — next call to ``name`` returns
   ``payload`` (wrapped in a ``FakeCallToolResult`` with
   ``structuredContent``).
2. ``set_error(name, message)`` — next call to ``name`` raises with
    ``message`` (which the dispatcher will translate via
    ``parse_error``).
3. ``set_raw_result(name, result)`` — next call to ``name`` returns a
   caller-provided ``FakeCallToolResult`` as-is.

Both are stacks: scripted responses are consumed in FIFO order.

Inspecting traffic
------------------
``session.calls`` is a list of ``(tool_name, payload)`` tuples in the
order they were dispatched.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

from memory_mcp_client import MemoryClient


@dataclass
class FakeCallToolResult:
    """Mimics ``mcp.types.CallToolResult`` (the subset our dispatcher reads)."""

    structuredContent: Any | None = None
    content: list[Any] = field(default_factory=list)
    isError: bool = False


@dataclass
class FakeTextBlock:
    """Mimics ``mcp.types.TextContent`` for the ``isError`` error path."""

    text: str
    type: str = "text"


class FakeClientSession:
    """In-memory ClientSession replacement.

    Args:
        responses: optional mapping of tool name → list of payloads to
            return on successive calls.
        errors: optional mapping of tool name → list of error messages to
            raise on successive calls.

    Methods:
        ``set_response(name, payload)`` queues a successful response.
        ``set_error(name, message)`` queues a server-style error.
        ``set_iserror(name, message)`` queues an ``isError=True`` result
            (the MCP-SDK soft-error path, distinct from a raised
            exception).
    """

    def __init__(
        self,
        responses: dict[str, list[Any]] | None = None,
        errors: dict[str, list[str]] | None = None,
    ) -> None:
        self._responses: dict[str, deque[tuple[str, Any]]] = defaultdict(deque)
        if responses:
            for name, payloads in responses.items():
                for p in payloads:
                    self._responses[name].append(("ok", p))
        if errors:
            for name, messages in errors.items():
                for m in messages:
                    self._responses[name].append(("raise", m))

        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._initialized = False

    # Scripting -------------------------------------------------------

    def set_response(self, name: str, payload: Any) -> None:
        """Queue a successful structured response for ``name``."""
        self._responses[name].append(("ok", payload))

    def set_error(self, name: str, message: str) -> None:
        """Queue a raised ToolError-shaped exception for ``name``."""
        self._responses[name].append(("raise", message))

    def set_iserror(self, name: str, message: str) -> None:
        """Queue an ``isError=True`` result for ``name`` (no exception)."""
        self._responses[name].append(("iserror", message))

    def set_raw_result(self, name: str, result: FakeCallToolResult) -> None:
        """Queue an exact CallToolResult-shaped object for ``name``."""
        self._responses[name].append(("raw", result))

    # ClientSession surface ------------------------------------------

    async def initialize(self) -> None:  # pragma: no cover - trivial
        self._initialized = True
        await asyncio.sleep(0)

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> FakeCallToolResult:
        self.calls.append((name, dict(arguments or {})))
        if not self._responses[name]:
            raise AssertionError(
                f"No scripted response for tool {name!r}; calls so far: {self.calls}"
            )
        kind, value = self._responses[name].popleft()
        if kind == "ok":
            return FakeCallToolResult(structuredContent=value)
        if kind == "raw":
            return value
        if kind == "iserror":
            return FakeCallToolResult(content=[FakeTextBlock(text=value)], isError=True)
        # kind == "raise"
        raise RuntimeError(value)


# ----------------------------------------------------------------------
# Pytest fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def fake_session() -> FakeClientSession:
    """A fresh FakeClientSession per test."""
    return FakeClientSession()


@pytest.fixture
def session_factory(fake_session: FakeClientSession):
    """Returns a ``session_factory`` callable for ``MemoryClient``.

    The MemoryClient invokes ``factory(client)`` inside its
    ``AsyncExitStack``; we wrap the fake in an async-context-manager
    that yields it.
    """

    @asynccontextmanager
    async def factory(_client: MemoryClient) -> AsyncIterator[FakeClientSession]:
        await fake_session.initialize()
        yield fake_session

    return factory


@pytest.fixture
async def client(session_factory) -> AsyncIterator[MemoryClient]:
    """An opened MemoryClient backed by ``fake_session``."""
    c = MemoryClient(
        "http://fake.local/mcp",
        session_factory=session_factory,
    )
    async with c:
        yield c


@pytest.fixture
async def client_with_identity(session_factory) -> AsyncIterator[MemoryClient]:
    """A MemoryClient with default agent_id + env_ids preconfigured."""
    c = MemoryClient(
        "http://fake.local/mcp",
        session_factory=session_factory,
        agent_id="00000000-0000-0000-0000-000000000001",
        default_env_ids=["00000000-0000-0000-0000-000000000002"],
    )
    async with c:
        yield c


# ----------------------------------------------------------------------
# Payload factories — minimal valid response shapes per domain.
# Keep these in lockstep with the schemas; if a Pydantic field becomes
# required and a factory drifts, the matching test will fail loudly.
# ----------------------------------------------------------------------


def make_memory_payload(**overrides: Any) -> dict[str, Any]:
    """A fully-populated mem_get/mem_write response payload."""
    base: dict[str, Any] = {
        "id": "00000000-0000-0000-0000-00000000beef",
        "env_id": "00000000-0000-0000-0000-0000000000e0",
        "kind": "fact",
        "status": "active",
        "title": "fake-memory",
        "body": "fake body",
        "tags": [],
        "metadata": {},
        "salience": 0.5,
        "confidence": 0.9,
        "pinned": False,
        "access_count": 0,
        "last_accessed_at": None,
        "negative_feedback_count": 0,
        "verified_at": None,
        "expires_at": None,
        "superseded_by": None,
        "version": 1,
        "created_at": "2026-05-13T00:00:00Z",
        "updated_at": "2026-05-13T00:00:00Z",
    }
    base.update(overrides)
    return base


def make_env_payload(**overrides: Any) -> dict[str, Any]:
    """A minimal valid EnvResponse payload."""
    base: dict[str, Any] = {
        "id": "00000000-0000-0000-0000-0000000000e0",
        "name": "fake-env",
        "kind": None,
        "retention_policy": {},
        "default_embedding_model_id": "all-MiniLM-L6-v2",
        "description": None,
        "created_at": "2026-05-13T00:00:00Z",
        "updated_at": "2026-05-13T00:00:00Z",
    }
    base.update(overrides)
    return base


def make_task_payload(**overrides: Any) -> dict[str, Any]:
    """A minimal valid TaskResponse payload."""
    base: dict[str, Any] = {
        "id": "00000000-0000-0000-0000-0000000000a0",
        "env_id": "00000000-0000-0000-0000-0000000000e0",
        "title": "fake-task",
        "description": None,
        "status": "pending",
        "priority": 50,
        "playbook_id": None,
        "version": 1,
        "created_at": "2026-05-13T00:00:00Z",
        "updated_at": "2026-05-13T00:00:00Z",
    }
    base.update(overrides)
    return base
