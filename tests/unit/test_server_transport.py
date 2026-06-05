"""Transport-selection tests for the memory-mcp server entrypoint."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from memory_mcp import config, server
from memory_mcp.config import Settings


@pytest.fixture(autouse=True)
def _clear_settings_cache(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    config.get_settings.cache_clear()
    try:
        yield
    finally:
        config.get_settings.cache_clear()


def test_main_defaults_to_http(monkeypatch: pytest.MonkeyPatch) -> None:
    uvicorn_calls: list[dict[str, object]] = []
    stdio_calls: list[Settings] = []

    def fake_uvicorn_run(*args: object, **kwargs: object) -> None:
        uvicorn_calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
    monkeypatch.setattr(server, "run_stdio", lambda settings: stdio_calls.append(settings))

    server.main()

    assert len(uvicorn_calls) == 1
    assert uvicorn_calls[0]["args"] == ("memory_mcp.server:app",)
    assert stdio_calls == []


def test_main_uses_stdio_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    uvicorn_calls: list[dict[str, object]] = []
    stdio_calls: list[Settings] = []

    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    config.get_settings.cache_clear()
    monkeypatch.setattr(
        "uvicorn.run",
        lambda *args, **kwargs: uvicorn_calls.append({"args": args, "kwargs": kwargs}),
    )
    monkeypatch.setattr(server, "run_stdio", lambda settings: stdio_calls.append(settings))

    server.main()

    assert uvicorn_calls == []
    assert len(stdio_calls) == 1
    assert stdio_calls[0].mcp_transport == "stdio"


def test_run_stdio_async_manages_service_lifecycle_in_order() -> None:
    settings = Settings(mcp_transport="stdio")
    calls: list[str] = []
    mcp_server = MagicMock()
    mcp_server.list_tools = AsyncMock(return_value=[])

    async def fake_run_stdio_async() -> None:
        calls.append("run_stdio_async")

    async def fake_close_graph_store() -> None:
        calls.append("_close_default_graph_store")

    async def fake_dispose_engine() -> None:
        calls.append("dispose_engine")

    mcp_server.run_stdio_async = AsyncMock(side_effect=fake_run_stdio_async)

    with (
        patch.object(server, "configure_logging"),
        patch.object(server, "warn_if_otlp_configured"),
        patch.object(server, "build_mcp_server", return_value=mcp_server),
        patch.object(server, "init_engine", side_effect=lambda _settings: calls.append("init_engine")),
        patch.object(server, "_close_default_graph_store", side_effect=fake_close_graph_store),
        patch.object(server, "dispose_engine", side_effect=fake_dispose_engine),
    ):
        asyncio.run(server._run_stdio_async(settings))

    assert calls == [
        "init_engine",
        "run_stdio_async",
        "_close_default_graph_store",
        "dispose_engine",
    ]
    mcp_server.run_stdio_async.assert_awaited_once_with()


def test_invalid_mcp_transport_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_TRANSPORT", "bogus")

    with pytest.raises(ValidationError):
        Settings()
