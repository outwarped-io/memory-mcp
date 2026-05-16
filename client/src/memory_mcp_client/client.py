"""Async memory-mcp client core.

The public surface is :class:`MemoryClient`, an async context manager
that opens a single Streamable-HTTP MCP session for its lifetime and
exposes namespaced sub-APIs (``.memories``, ``.tasks``, ``.envs``, …)
that wrap the 45 server tools with typed Pydantic models.

Example::

    async with MemoryClient("http://127.0.0.1:8080/mcp") as client:
        envs = await client.envs.list_()
        out = await client.memories.search(query="...", env_ids=[envs[0].id])

Use :meth:`MemoryClient.health` / :meth:`MemoryClient.ready` for
non-MCP probes that hit ``/healthz`` and ``/readyz`` directly via httpx.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any
from urllib.parse import urlparse, urlunparse
from uuid import UUID

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from memory_mcp_client import _session as _session_mod
from memory_mcp_client._retry import RetryPolicy, run_with_retry

log = logging.getLogger(__name__)


class MemoryClient:
    """Async client for the memory-mcp Streamable-HTTP MCP server.

    Args:
        url: Full ``/mcp`` endpoint URL, e.g. ``http://127.0.0.1:8080/mcp``.
        auth_token: Optional bearer token (v1.5+). When set, adds
            ``Authorization: Bearer <token>`` to every request.
        agent_id: Optional client-level default agent UUID. Merged into
            every tool call unless the per-call payload sets its own.
        default_env_ids: Optional client-level default env-id list.
            Merged into every tool call's ``attached_env_ids`` field
            unless overridden per call.
        default_env_names: Optional client-level default env-name list.
            Merged into every tool call's ``attached_env_names`` field
            unless overridden per call. Mutually exclusive with
            ``default_env_ids``.
        attached_env_names: Alias for ``default_env_names`` matching the
            server-side per-call field name.
        headers: Extra static HTTP headers to attach to every request.
        timeout: Per-request HTTP timeout (seconds) for the httpx
            transport used for ``/healthz``/``/readyz`` probes. The MCP
            session uses the SDK's own timeout default.
        session_factory: Internal hook overridable by tests so the
            namespace APIs can run against a fake ``ClientSession``
            without opening a real Streamable-HTTP connection.

    The client must be entered with ``async with`` (or via
    :meth:`aopen` + :meth:`aclose`) before any tool call. Calling a
    namespace method on an un-opened client raises ``RuntimeError``.
    """

    def __init__(
        self,
        url: str,
        *,
        auth_token: str | None = None,
        agent_id: UUID | str | None = None,
        default_env_ids: list[UUID | str] | None = None,
        default_env_names: list[str] | None = None,
        attached_env_names: list[str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        session_factory: Any | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.url = url
        self.auth_token = auth_token
        if default_env_names and attached_env_names:
            raise ValueError("default_env_names and attached_env_names are aliases; provide only one")
        effective_env_names = default_env_names or attached_env_names
        if default_env_ids and effective_env_names:
            raise ValueError("default_env_ids and attached_env_names are mutually exclusive")
        self.agent_id = agent_id
        self.default_env_ids = list(default_env_ids) if default_env_ids else None
        self.default_env_names = list(effective_env_names) if effective_env_names else None
        self.timeout = timeout
        self._extra_headers = dict(headers or {})
        self._session_factory = session_factory
        self.retry_policy = retry_policy or RetryPolicy()

        self._stack: AsyncExitStack | None = None
        self._session: Any | None = None
        self._opened = False

        # Lazy-import namespaces here so the module graph stays acyclic.
        from memory_mcp_client.api import (
            DecisionsAPI,
            DreamAPI,
            EntitiesAPI,
            EnvOpsAPI,
            EnvsAPI,
            MemoriesAPI,
            PlaybooksAPI,
            RelationsAPI,
            TasksAPI,
        )

        self.memories = MemoriesAPI(self)
        self.tasks = TasksAPI(self)
        self.envs = EnvsAPI(self)
        self.env_ops = EnvOpsAPI(self)
        self.entities = EntitiesAPI(self)
        self.relations = RelationsAPI(self)
        self.playbooks = PlaybooksAPI(self)
        self.decisions = DecisionsAPI(self)
        self.dream = DreamAPI(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _request_headers(self) -> dict[str, str]:
        headers = dict(self._extra_headers)
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    async def __aenter__(self) -> MemoryClient:
        await self.aopen()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aopen(self) -> None:
        """Open the underlying MCP session. Idempotent."""
        if self._opened:
            return
        self._stack = AsyncExitStack()

        if self._session_factory is not None:
            # Test hook: bypass the real Streamable-HTTP plumbing.
            self._session = await self._stack.enter_async_context(
                self._session_factory(self),
            )
        else:
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(self.url, headers=self._request_headers())
            )
            session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self._session = session

        self._opened = True

    async def aclose(self) -> None:
        """Close the underlying MCP session. Idempotent."""
        if not self._opened:
            return
        stack = self._stack
        self._stack = None
        self._session = None
        self._opened = False
        if stack is not None:
            await stack.aclose()

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    async def _call(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        model: Any | None = None,
    ) -> Any:
        if not self._opened or self._session is None:
            raise RuntimeError(
                "MemoryClient must be opened with `async with` (or aopen()) "
                "before tool calls."
            )
        full = _session_mod._build_payload(
            payload,
            agent_id_default=self.agent_id,
            env_ids_default=self.default_env_ids,
            env_names_default=self.default_env_names,
        )
        has_idempotency_key = self._has_idempotency_key(full)

        async def _do_call() -> Any:
            return await _session_mod.call_tool(self._session, name, full, model=model)

        return await run_with_retry(
            _do_call,
            tool_name=name,
            policy=self.retry_policy,
            has_idempotency_key=has_idempotency_key,
        )

    @staticmethod
    def _has_idempotency_key(payload: dict[str, Any]) -> bool:
        """Detect an idempotency-key field anywhere in the payload.

        We look at the top-level *and* inside a ``request`` envelope —
        the v0.10/0.11 server dispatcher wraps every payload as
        ``{"request": {...}}`` so the caller's idempotency key lives one
        level down.
        """

        if not payload:
            return False
        if payload.get("idempotency_key"):
            return True
        req = payload.get("request")
        if isinstance(req, dict) and req.get("idempotency_key"):
            return True
        return False

    # ------------------------------------------------------------------
    # Health probes (non-MCP)
    # ------------------------------------------------------------------

    def _http_base(self) -> str:
        parts = urlparse(self.url)
        return urlunparse((parts.scheme, parts.netloc, "", "", "", ""))

    async def health(self) -> dict[str, Any]:
        """GET ``/healthz`` and return the parsed JSON body."""
        return await self._http_probe("/healthz")

    async def ready(self) -> dict[str, Any]:
        """GET ``/readyz`` and return the parsed JSON body."""
        return await self._http_probe("/readyz")

    async def _http_probe(self, path: str) -> dict[str, Any]:
        base = self._http_base()
        headers = self._request_headers()
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            resp = await http.get(f"{base}{path}", headers=headers)
            resp.raise_for_status()
            return resp.json()
