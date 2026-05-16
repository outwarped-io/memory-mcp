"""Smoke tests for the Phase 1 server skeleton.

v1 = local-only flags must be surfaced on every health endpoint so clients
and monitoring can detect they're talking to an unauthenticated build.
The MCP Streamable HTTP transport must mount at ``/mcp`` and the tool
registry must include all v1 tools.

Each test builds a fresh app via :func:`build_app` because FastMCP's
``StreamableHTTPSessionManager.run()`` may only be entered ONCE per
instance — module-level ``app`` is reserved for ``uvicorn``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from memory_mcp.server import build_app


def _assert_local_only_flags(body: dict[str, object]) -> None:
    """Required local-only flags. Schema may add fields; these must be present."""
    assert body["local_only"] is True
    assert body["auth"] == "disabled"
    assert body["bind_host"] == "127.0.0.1"
    assert body["unsafe_remote_bind"] is False
    assert body["transport"]["mcp"]["path"] == "/mcp"  # type: ignore[index]


def test_healthz_returns_local_only_flags() -> None:
    with TestClient(build_app()) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    _assert_local_only_flags(body)


def test_readyz_returns_local_only_flags_and_dependency_probe() -> None:
    with TestClient(build_app()) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    # Without Postgres available in unit tests the dependency probe will
    # report degraded — that's expected; we just verify the structure.
    _assert_local_only_flags(body)
    assert body["status"] in {"ok", "degraded"}
    assert "dependencies" in body


def test_unsafe_remote_bind_flag_when_non_loopback(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Operators overriding to a non-loopback host must surface the warning flag."""
    from memory_mcp import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("MCP_HTTP_HOST", "0.0.0.0")
    try:
        with TestClient(build_app()) as client:
            response = client.get("/healthz")
        body = response.json()
        assert body["unsafe_remote_bind"] is True
        assert body["bind_host"] == "0.0.0.0"
    finally:
        config.get_settings.cache_clear()


def test_request_id_header_round_trips_through_app() -> None:
    """Sanity: the RequestIdMiddleware is wired into the real app."""
    with TestClient(build_app()) as client:
        response = client.get("/healthz", headers={"x-request-id": "test-rid"})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "test-rid"


def test_metrics_endpoint_serves_prometheus_exposition() -> None:
    """``/metrics`` must serve text exposition with our registered series."""
    with TestClient(build_app()) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "mcp_tool_calls_total" in body
    assert "mcp_tool_latency_seconds" in body


def test_mcp_tool_registry_contains_v1_tools() -> None:
    """Every Phase-1 tool must be registered with the FastMCP server."""
    import asyncio

    from memory_mcp.mcp_app import build_mcp_server

    mcp = build_mcp_server()
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    expected = {
        "mem_write", "mem_get", "mem_get_many", "mem_update",
        "mem_archive", "mem_retire", "mem_supersede",
        "mem_journal", "mem_search",
        "ent_upsert", "ent_resolve", "ent_merge",
        "rel_link",
        "env_create_", "env_list_", "env_get_",
        "env_attach_", "env_detach_",
    }
    missing = expected - names
    assert not missing, f"missing tools: {sorted(missing)} (got {sorted(names)})"

