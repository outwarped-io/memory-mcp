"""MCP-wire coverage for mem_search friendly env names."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from memory_mcp_schemas.search import MemorySearchRequest, MemorySearchResponse

from memory_mcp.errors import (
    EnvNotAttachedError,
    EnvNotFoundError,
    EnvRefAmbiguousError,
)
from memory_mcp.identity import AgentContext


async def _call_mem_search(
    monkeypatch: pytest.MonkeyPatch,
    request: dict[str, Any],
    *,
    envs: dict[str, UUID] | None = None,
    lookup_error: Exception | None = None,
    domain_error: Exception | None = None,
) -> tuple[Any, list[MemorySearchRequest]]:
    from memory_mcp import env_resolve, mcp_app, server

    seen: list[MemorySearchRequest] = []
    envs = envs or {}

    async def fake_lookup(name: str, *, include_deleted: bool = False):
        if lookup_error is not None:
            raise lookup_error
        return SimpleNamespace(id=envs[name.lower()])

    async def fake_resolve_ctx(
        *,
        agent_id: UUID | None,
        attached_env_ids: list[UUID] | None,
        attached_env_names: list[str] | None = None,
        settings=None,
    ) -> AgentContext:
        return AgentContext(
            agent_id=agent_id or uuid4(),
            agent_name="test",
            attached_env_ids=list(attached_env_ids or []),
        )

    async def fake_memory_search(req: MemorySearchRequest, *, ctx: AgentContext) -> MemorySearchResponse:
        seen.append(req)
        if domain_error is not None:
            raise domain_error
        return MemorySearchResponse(
            hits=[],
            mode=req.mode,
            effective_mode="lex",
            consistency_used=req.consistency,
            projection_status=[],
        )

    async def noop_async(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(env_resolve, "get_env_by_name_ci", fake_lookup)
    monkeypatch.setattr(mcp_app, "_resolve_ctx", fake_resolve_ctx)
    monkeypatch.setattr(mcp_app, "memory_search", fake_memory_search)
    monkeypatch.setattr(server, "init_engine", lambda _settings: None)
    monkeypatch.setattr(server, "dispose_engine", noop_async)
    monkeypatch.setattr(server, "_close_default_graph_store", noop_async)

    app = server.build_app()

    def httpx_client_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8080",
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    async with (
        app.router.lifespan_context(app),
        streamablehttp_client(
            "http://127.0.0.1:8080/mcp/",
            httpx_client_factory=httpx_client_factory,
        ) as (read_stream, write_stream, _get_session_id),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        result = await session.call_tool(
            "mem_search",
            {"request": request, "agent_id": str(uuid4())},
        )
    return result, seen


def _result_text(result: Any) -> str:
    return "\n".join(getattr(part, "text", str(part)) for part in result.content)


@pytest.mark.asyncio
async def test_mem_search_with_env_names(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()

    result, seen = await _call_mem_search(
        monkeypatch,
        {"query": "x", "env_names": ["cdp"], "mode": "lex"},
        envs={"cdp": env_id},
    )

    assert result.isError is False
    assert seen[0].env_ids == [env_id]
    assert seen[0].env_names is None


@pytest.mark.asyncio
async def test_mem_search_with_env_ids_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()

    result, seen = await _call_mem_search(
        monkeypatch,
        {"query": "x", "env_ids": [str(env_id)], "mode": "lex"},
    )

    assert result.isError is False
    assert seen[0].env_ids == [env_id]
    assert seen[0].env_names is None


@pytest.mark.asyncio
async def test_mem_search_both_provided_raises_both_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()

    result, seen = await _call_mem_search(
        monkeypatch,
        {"query": "x", "env_ids": [str(env_id)], "env_names": ["cdp"], "mode": "lex"},
        envs={"cdp": env_id},
    )

    assert result.isError is True
    assert "ENV_REF_BOTH_PROVIDED" in _result_text(result)
    assert seen == []


@pytest.mark.asyncio
async def test_mem_search_unknown_env_name_raises_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, seen = await _call_mem_search(
        monkeypatch,
        {"query": "x", "env_names": ["typo"], "mode": "lex"},
        lookup_error=EnvNotFoundError(name="typo"),
        domain_error=EnvNotAttachedError("should not win"),
    )

    text = _result_text(result)
    assert result.isError is True
    assert "ENV_NOT_FOUND" in text
    assert "ENV_NOT_ATTACHED" not in text
    assert seen == []


@pytest.mark.asyncio
async def test_error_order_both_provided_precedes_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()
    result, seen = await _call_mem_search(
        monkeypatch,
        {"query": "x", "env_ids": [str(env_id)], "env_names": ["cdp"], "mode": "lex"},
        lookup_error=EnvRefAmbiguousError(name="cdp", candidate_ids=[uuid4(), uuid4()]),
    )

    text = _result_text(result)
    assert "ENV_REF_BOTH_PROVIDED" in text
    assert "ENV_REF_AMBIGUOUS" not in text
    assert seen == []


@pytest.mark.asyncio
async def test_error_order_ambiguous_precedes_not_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, seen = await _call_mem_search(
        monkeypatch,
        {"query": "x", "env_names": ["cdp"], "mode": "lex"},
        lookup_error=EnvRefAmbiguousError(name="cdp", candidate_ids=[uuid4(), uuid4()]),
        domain_error=EnvNotAttachedError("should not win"),
    )

    text = _result_text(result)
    assert "ENV_REF_AMBIGUOUS" in text
    assert "ENV_NOT_ATTACHED" not in text
    assert seen == []


@pytest.mark.asyncio
async def test_error_order_not_found_precedes_not_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, seen = await _call_mem_search(
        monkeypatch,
        {"query": "x", "env_names": ["missing"], "mode": "lex"},
        lookup_error=EnvNotFoundError(name="missing"),
        domain_error=EnvNotAttachedError("should not win"),
    )

    text = _result_text(result)
    assert "ENV_NOT_FOUND" in text
    assert "ENV_NOT_ATTACHED" not in text
    assert seen == []
