"""MCP-wire coverage for graph fallback/min-score request fields."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from memory_mcp.identity import AgentContext
from memory_mcp_schemas.graph import (
    MemNeighborsRequest,
    MemNeighborsResponse,
    MemRelatedRequest,
    MemRelatedResponse,
)


async def _call_graph_tool(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tool_name: str,
    request: dict[str, Any],
    response_model: MemNeighborsResponse | MemRelatedResponse,
    handler_attr: str,
) -> tuple[Any, list[MemNeighborsRequest | MemRelatedRequest]]:
    from memory_mcp import mcp_app, server

    seen: list[MemNeighborsRequest | MemRelatedRequest] = []

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

    async def fake_handler(req, *, ctx: AgentContext):
        seen.append(req)
        return response_model

    async def noop_async(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(mcp_app, "_resolve_ctx", fake_resolve_ctx)
    monkeypatch.setattr(mcp_app, handler_attr, fake_handler)
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

    async with app.router.lifespan_context(app):
        async with streamablehttp_client(
            "http://127.0.0.1:8080/mcp/",
            httpx_client_factory=httpx_client_factory,
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    tool_name,
                    {"request": request, "agent_id": str(uuid4())},
                )
    return result, seen


def _payload(result: Any) -> dict[str, Any]:
    text = "\n".join(getattr(part, "text", str(part)) for part in result.content)
    return json.loads(text)


@pytest.mark.asyncio
async def test_mem_neighbors_mcp_accepts_fallback_and_returns_fallback_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, seen = await _call_graph_tool(
        monkeypatch,
        tool_name="mem_neighbors",
        request={"memory_id": str(uuid4()), "fallback": True},
        response_model=MemNeighborsResponse(
            hits=[],
            next_cursor=None,
            fallback_used=["include_retired"],
        ),
        handler_attr="memory_neighbors",
    )

    assert result.isError is False
    assert seen[0].fallback is True
    assert _payload(result)["fallback_used"] == ["include_retired"]


@pytest.mark.asyncio
async def test_mem_related_mcp_accepts_min_score_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, seen = await _call_graph_tool(
        monkeypatch,
        tool_name="mem_related",
        request={
            "memory_id": str(uuid4()),
            "relation": "semantic",
            "min_score": 0.5,
            "fallback": True,
        },
        response_model=MemRelatedResponse(
            hits=[],
            next_cursor=None,
            note="ok",
            fallback_used=["include_retired"],
        ),
        handler_attr="memory_related",
    )

    assert result.isError is False
    assert seen[0].relation == "semantic"
    assert seen[0].min_score == 0.5
    assert seen[0].fallback is True
    assert _payload(result)["fallback_used"] == ["include_retired"]
