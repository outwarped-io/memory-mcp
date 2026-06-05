"""MCP-wire coverage for mem_stats."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from memory_mcp_schemas.stats import MemoriesStats, MemStatsRequest, MemStatsResponse

from memory_mcp.identity import AgentContext


async def _call_mem_stats(
    monkeypatch: pytest.MonkeyPatch,
    request: dict[str, Any],
    *,
    envs: dict[str, UUID] | None = None,
):
    from memory_mcp import env_resolve, mcp_app, server

    seen: list[MemStatsRequest] = []
    envs = envs or {}

    async def fake_lookup(name: str, *, include_deleted: bool = False):
        return SimpleNamespace(id=envs[name.lower()])

    async def fake_resolve_ctx(
        *,
        agent_id: UUID | None,
        attached_env_ids: list[UUID] | None,
        attached_env_names: list[str] | None = None,
        settings=None,
    ):
        return AgentContext(
            agent_id=agent_id or uuid4(), agent_name="test", attached_env_ids=list(attached_env_ids or [])
        )

    async def fake_compute(req: MemStatsRequest, *, ctx: AgentContext):
        seen.append(req)
        return MemStatsResponse(memories=MemoriesStats(total=7, by_kind={"fact": 7}))

    async def noop_async(*_args, **_kwargs):
        return None

    monkeypatch.setattr(env_resolve, "get_env_by_name_ci", fake_lookup)
    monkeypatch.setattr(mcp_app, "_resolve_ctx", fake_resolve_ctx)
    monkeypatch.setattr(mcp_app, "compute_mem_stats", fake_compute)
    monkeypatch.setattr(server, "init_engine", lambda _settings: None)
    monkeypatch.setattr(server, "dispose_engine", noop_async)
    monkeypatch.setattr(server, "_close_default_graph_store", noop_async)

    app = server.build_app()

    def httpx_client_factory(headers=None, timeout=None, auth=None):
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
        result = await session.call_tool("mem_stats", {"request": request, "agent_id": str(uuid4())})
    return result, seen


def _result_text(result: Any) -> str:
    return "\n".join(getattr(part, "text", str(part)) for part in result.content)


@pytest.mark.asyncio
async def test_mem_stats_with_env_names(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()

    result, seen = await _call_mem_stats(monkeypatch, {"env_names": ["cdp"]}, envs={"cdp": env_id})

    assert result.isError is False
    assert seen[0].env_ids == [env_id]
    assert seen[0].env_names is None
    assert '"total": 7' in _result_text(result)


@pytest.mark.asyncio
async def test_mem_stats_env_id_and_name_conflict_fails_before_compute(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()

    result, seen = await _call_mem_stats(
        monkeypatch,
        {"env_ids": [str(env_id)], "env_names": ["cdp"]},
        envs={"cdp": env_id},
    )

    assert result.isError is True
    assert "ENV_REF_BOTH_PROVIDED" in _result_text(result)
    assert seen == []
