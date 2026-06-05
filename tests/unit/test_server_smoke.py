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
        "mem_write",
        "mem_get",
        "mem_get_many",
        "mem_update",
        "mem_archive",
        "mem_retire",
        "mem_supersede",
        "mem_journal",
        "mem_search",
        "ent_upsert",
        "ent_resolve",
        "ent_merge",
        "rel_link",
        "env_create_",
        "env_list_",
        "env_get_",
        "env_attach_",
        "env_detach_",
    }
    missing = expected - names
    assert not missing, f"missing tools: {sorted(missing)} (got {sorted(names)})"


def test_mcp_server_info_reports_package_version_not_sdk() -> None:
    """``initialize.serverInfo.version`` must be the memory-mcp package version,
    not the MCP SDK version fallback (``pkg_version('mcp')``).

    Regression guard for the FastMCP wrapper not exposing ``version=`` as a
    constructor kwarg — fixed by overriding ``_mcp_server.version`` after
    construction. See ``src/memory_mcp/mcp_app.py:build_mcp_server``.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    from memory_mcp.mcp_app import build_mcp_server

    try:
        expected = pkg_version("memory-mcp")
    except PackageNotFoundError:
        # Editable / dev install where the package isn't pip-visible.
        # In that case the override is a no-op; the SDK default stays.
        # Still assert that the server is constructible and carries a version.
        mcp = build_mcp_server()
        assert mcp._mcp_server.version  # SDK default is non-empty
        return

    mcp = build_mcp_server()
    actual = mcp._mcp_server.version
    assert actual == expected, (
        f"serverInfo.version should report memory-mcp package version "
        f"({expected!r}); got {actual!r} (SDK fallback would be the value "
        f"of pkg_version('mcp'))."
    )
    # Defensive: SDK version is also installed in this venv, so make sure
    # we're not accidentally aligned with it.
    sdk_version = pkg_version("mcp")
    assert actual != sdk_version or expected == sdk_version, (
        f"serverInfo.version {actual!r} matches the MCP SDK version {sdk_version!r} — the override likely isn't wired."
    )


def test_healthz_includes_package_version() -> None:
    """``/healthz`` payload must include ``version`` so ops can probe the
    deployed memory-mcp release without speaking MCP. Best-effort: skipped
    cleanly on editable / dev installs where the package isn't pip-visible."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        expected = pkg_version("memory-mcp")
    except PackageNotFoundError:
        # No package metadata available — the helper returns None and the
        # field is omitted. Verify the omission rather than assert presence.
        with TestClient(build_app()) as client:
            body = client.get("/healthz").json()
        assert "version" not in body
        return

    with TestClient(build_app()) as client:
        body = client.get("/healthz").json()
    assert body.get("version") == expected
