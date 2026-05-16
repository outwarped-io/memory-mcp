"""MemoryClient lifecycle, header, and health-probe tests."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from memory_mcp_client import MemoryClient
from tests.conftest import FakeClientSession


def counting_session_factory() -> tuple[Any, dict[str, int]]:
    counts = {"opens": 0, "closes": 0}

    @asynccontextmanager
    async def factory(_client: MemoryClient):
        counts["opens"] += 1
        try:
            yield FakeClientSession()
        finally:
            counts["closes"] += 1

    return factory, counts


@pytest.mark.asyncio
async def test_aopen_aclose_idempotent() -> None:
    factory, counts = counting_session_factory()
    c = MemoryClient("http://fake.local/mcp", session_factory=factory)

    await c.aopen()
    await c.aopen()
    assert c._opened is True
    assert counts["opens"] == 1

    await c.aclose()
    await c.aclose()
    assert c._opened is False
    assert c._session is None
    assert counts["closes"] == 1


@pytest.mark.asyncio
async def test_async_with_opens_and_closes() -> None:
    factory, _ = counting_session_factory()
    c = MemoryClient("http://fake.local/mcp", session_factory=factory)

    async with c:
        assert c._opened is True

    assert c._opened is False


@pytest.mark.asyncio
async def test_double_aopen_is_noop() -> None:
    factory, counts = counting_session_factory()
    c = MemoryClient("http://fake.local/mcp", session_factory=factory)

    await c.aopen()
    first_session = c._session
    await c.aopen()

    assert c._session is first_session
    assert counts["opens"] == 1
    await c.aclose()


@pytest.mark.asyncio
async def test_aclose_without_open_is_noop() -> None:
    c = MemoryClient("http://fake.local/mcp")

    await c.aclose()

    assert c._opened is False
    assert c._session is None


@pytest.mark.asyncio
async def test_tool_call_after_close_raises_runtime_error() -> None:
    factory, _ = counting_session_factory()
    c = MemoryClient("http://fake.local/mcp", session_factory=factory)
    await c.aopen()
    await c.aclose()

    with pytest.raises(RuntimeError, match="must be opened"):
        await c.memories.get(uuid4())


def test_default_headers_include_auth_token() -> None:
    c = MemoryClient("http://fake.local/mcp", auth_token="abc")

    assert c._request_headers()["Authorization"] == "Bearer abc"


def test_extra_headers_passed_through() -> None:
    c = MemoryClient("http://fake.local/mcp", headers={"X-Trace-Id": "t1"})

    assert c._request_headers()["X-Trace-Id"] == "t1"


def test_extra_headers_dont_overwrite_auth() -> None:
    c = MemoryClient(
        "http://fake.local/mcp",
        auth_token="abc",
        headers={"Authorization": "Bearer wrong"},
    )

    assert c._request_headers()["Authorization"] == "Bearer abc"


class StubResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.mark.asyncio
async def test_health_returns_parsed_json() -> None:
    c = MemoryClient("http://fake.local/mcp")
    get = AsyncMock(return_value=StubResponse({"status": "ok"}))

    with patch("httpx.AsyncClient.get", get):
        out = await c.health()

    assert out == {"status": "ok"}
    get.assert_awaited_once_with("http://fake.local/healthz", headers={})


@pytest.mark.asyncio
async def test_ready_returns_parsed_json() -> None:
    c = MemoryClient("http://fake.local/mcp")
    get = AsyncMock(return_value=StubResponse({"status": "ok"}))

    with patch("httpx.AsyncClient.get", get):
        out = await c.ready()

    assert out == {"status": "ok"}
    get.assert_awaited_once_with("http://fake.local/readyz", headers={})


def test_http_base_strips_mcp_path() -> None:
    c = MemoryClient("http://host:1234/mcp")

    assert c._http_base() == "http://host:1234"
