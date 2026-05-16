"""Unit tests for the v0.2 SDK retry policy.

The policy lives in :mod:`memory_mcp_client._retry` and wires into
``MemoryClient._call`` so every tool call passes through. These tests
exercise both the standalone policy primitives and the end-to-end
client integration via the existing ``FakeClientSession``.

Coverage
--------
* Read tools retry on transient transport errors and succeed on a later
  attempt.
* Write tools do **not** retry without ``retry_writes=True`` or an
  ``idempotency_key`` in the payload.
* Write tools **do** retry when ``idempotency_key`` is present.
* Non-retryable typed errors (e.g. ``VersionConflict``) skip retries and
  surface immediately.
* When all attempts fail, the caller sees :class:`RetryExhaustedError`
  with per-attempt metadata.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import pytest

from memory_mcp_client import MemoryClient, RetryExhaustedError, RetryPolicy
from memory_mcp_client._retry import (
    DEFAULT_READ_TOOLS,
    _is_retryable_exception,
    run_with_retry,
)
from memory_mcp_client.errors import (
    GraphBackendUnavailableError,
    RateLimitedError,
    VersionConflictError,
)

from tests.conftest import FakeClientSession, make_env_payload, make_memory_payload


# --- RetryPolicy primitives -------------------------------------------------


def test_default_policy_values():
    pol = RetryPolicy()
    assert pol.max_attempts == 3
    assert pol.base_delay_seconds == 0.25
    assert pol.retry_writes is False
    assert "mem_search" in pol.read_tools
    assert "mem_get" in pol.read_tools
    assert "mem_write" not in pol.read_tools


def test_policy_rejects_invalid_values():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay_seconds=-1)
    with pytest.raises(ValueError):
        RetryPolicy(max_delay_seconds=0.1, base_delay_seconds=1.0)
    with pytest.raises(ValueError):
        RetryPolicy(exponential_base=0.5)
    with pytest.raises(ValueError):
        RetryPolicy(jitter=1.5)


def test_policy_is_retryable_tool_for_reads():
    pol = RetryPolicy()
    assert pol.is_retryable_tool("mem_search", has_idempotency_key=False)
    assert pol.is_retryable_tool("env_list_", has_idempotency_key=False)


def test_policy_is_retryable_tool_for_writes_only_with_optin():
    pol = RetryPolicy()
    assert not pol.is_retryable_tool("mem_write", has_idempotency_key=False)
    assert pol.is_retryable_tool("mem_write", has_idempotency_key=True)

    pol_writes = RetryPolicy(retry_writes=True)
    assert pol_writes.is_retryable_tool("mem_write", has_idempotency_key=False)


def test_sleep_for_attempt_growth():
    pol = RetryPolicy(base_delay_seconds=0.1, exponential_base=2.0, jitter=0)
    assert pol.sleep_for_attempt(0) == 0.0
    assert pol.sleep_for_attempt(1) == pytest.approx(0.1)
    assert pol.sleep_for_attempt(2) == pytest.approx(0.2)
    assert pol.sleep_for_attempt(3) == pytest.approx(0.4)


def test_sleep_caps_at_max_delay():
    pol = RetryPolicy(
        base_delay_seconds=0.5,
        exponential_base=4.0,
        max_delay_seconds=1.0,
        jitter=0,
    )
    # 0.5 * 4 ** 5 = 512 → must cap at 1.0
    assert pol.sleep_for_attempt(5) == pytest.approx(1.0)


def test_is_retryable_exception_classifies_correctly():
    assert _is_retryable_exception(RateLimitedError("rate"))
    assert _is_retryable_exception(GraphBackendUnavailableError("neo4j"))
    assert _is_retryable_exception(ConnectionError("refused"))
    assert _is_retryable_exception(asyncio.TimeoutError())
    assert not _is_retryable_exception(VersionConflictError("v"))
    assert not _is_retryable_exception(ValueError("oops"))


# --- run_with_retry primitive ----------------------------------------------


async def _make_fn(*results: Any):
    """Build an async fn that yields successive results/exceptions."""

    it = iter(results)

    async def fn() -> Any:
        value = next(it)
        if isinstance(value, BaseException):
            raise value
        return value

    return fn


@pytest.mark.asyncio
async def test_run_with_retry_succeeds_after_one_failure(monkeypatch):
    # Make sleeps instant.
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    pol = RetryPolicy(max_attempts=3, base_delay_seconds=0, jitter=0)
    fn = await _make_fn(RateLimitedError("slow down"), {"ok": True})

    out = await run_with_retry(fn, tool_name="mem_search", policy=pol)
    assert out == {"ok": True}


@pytest.mark.asyncio
async def test_run_with_retry_non_retryable_passes_through():
    pol = RetryPolicy()
    fn = await _make_fn(VersionConflictError("conflict"))

    with pytest.raises(VersionConflictError):
        await run_with_retry(fn, tool_name="mem_search", policy=pol)


@pytest.mark.asyncio
async def test_run_with_retry_writes_do_not_retry_by_default():
    pol = RetryPolicy()
    fn = await _make_fn(RateLimitedError("slow"), {"unreached": True})

    with pytest.raises(RateLimitedError):
        await run_with_retry(fn, tool_name="mem_write", policy=pol)


@pytest.mark.asyncio
async def test_run_with_retry_writes_retry_with_idempotency_key(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    pol = RetryPolicy(max_attempts=3, base_delay_seconds=0, jitter=0)
    fn = await _make_fn(RateLimitedError("slow"), {"reached": True})

    out = await run_with_retry(
        fn,
        tool_name="mem_write",
        policy=pol,
        has_idempotency_key=True,
    )
    assert out == {"reached": True}


@pytest.mark.asyncio
async def test_run_with_retry_exhausts_and_raises(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    pol = RetryPolicy(max_attempts=2, base_delay_seconds=0, jitter=0)
    fn = await _make_fn(
        RateLimitedError("slow-1"),
        RateLimitedError("slow-2"),
    )

    with pytest.raises(RetryExhaustedError) as excinfo:
        await run_with_retry(fn, tool_name="mem_search", policy=pol)

    err = excinfo.value
    assert err.details["tool"] == "mem_search"
    assert len(err.details["attempts"]) == 2
    assert isinstance(err.__cause__, RateLimitedError)


# --- MemoryClient integration ----------------------------------------------


async def _noop_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
def fast_policy() -> RetryPolicy:
    return RetryPolicy(max_attempts=3, base_delay_seconds=0, jitter=0)


@pytest.fixture
async def client_with_retries(
    fake_session: FakeClientSession, fast_policy: RetryPolicy, monkeypatch
) -> AsyncIterator[MemoryClient]:
    monkeypatch.setattr("memory_mcp_client._retry.asyncio.sleep", _noop_sleep)

    @asynccontextmanager
    async def factory(_client: MemoryClient) -> AsyncIterator[FakeClientSession]:
        await fake_session.initialize()
        yield fake_session

    c = MemoryClient(
        "http://fake.local/mcp",
        session_factory=factory,
        retry_policy=fast_policy,
    )
    async with c:
        yield c


@pytest.mark.asyncio
async def test_client_read_retries_on_rate_limited(
    fake_session: FakeClientSession, client_with_retries: MemoryClient
):
    # First call raises a server-style RATE_LIMITED error; second succeeds.
    fake_session.set_error("env_list_", "[RATE_LIMITED] slow down")
    fake_session.set_response("env_list_", [make_env_payload()])

    envs = await client_with_retries.envs.list_()
    assert len(envs) == 1
    assert envs[0].name == "fake-env"
    # Both attempts recorded.
    assert [c[0] for c in fake_session.calls] == ["env_list_", "env_list_"]


@pytest.mark.asyncio
async def test_client_write_does_not_retry_without_idempotency_key(
    fake_session: FakeClientSession, client_with_retries: MemoryClient
):
    # Two queued errors — if retry kicked in, the second would be consumed.
    fake_session.set_error("mem_write", "[RATE_LIMITED] backpressure")
    fake_session.set_response("mem_write", make_memory_payload())

    with pytest.raises(RateLimitedError):
        await client_with_retries.memories.write(
            env_id="00000000-0000-0000-0000-0000000000e0",
            kind="fact",
            title="t",
            body="b",
        )
    # Only one attempt should have happened.
    assert len([c for c in fake_session.calls if c[0] == "mem_write"]) == 1


@pytest.mark.asyncio
async def test_client_exhausts_and_raises_retry_exhausted(
    fake_session: FakeClientSession, client_with_retries: MemoryClient
):
    for _ in range(3):
        fake_session.set_error("env_list_", "[RATE_LIMITED] slow")

    with pytest.raises(RetryExhaustedError):
        await client_with_retries.envs.list_()

    assert len([c for c in fake_session.calls if c[0] == "env_list_"]) == 3


@pytest.mark.asyncio
async def test_client_non_retryable_passes_through(
    fake_session: FakeClientSession, client_with_retries: MemoryClient
):
    fake_session.set_error(
        "env_list_",
        "[VERSION_CONFLICT] stale write :: {\"expected\": 1, \"actual\": 2}",
    )

    with pytest.raises(VersionConflictError):
        await client_with_retries.envs.list_()
    # Only one attempt — typed non-retryable error short-circuits.
    assert len([c for c in fake_session.calls if c[0] == "env_list_"]) == 1


def test_read_tools_allowlist_well_known():
    """Smoke check the allowlist covers the common reads."""
    expected = {
        "mem_search",
        "mem_get",
        "mem_browse",
        "env_list_",
        "ent_browse",
        "rel_browse",
        "task_list",
    }
    missing = expected - DEFAULT_READ_TOOLS
    assert missing == set(), f"missing reads: {missing}"
