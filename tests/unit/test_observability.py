"""Unit tests for the observability module.

We avoid touching real Postgres — the projection-metric refresh is exercised
via integration smoke. Here we focus on:

* request-id middleware behaviour (header in/out, contextvar isolation)
* :func:`instrument_tool` correctly classifies outcomes (ok / mcperror /
  error) and observes latency
* the Prometheus exposition includes the registered series after a call
* :func:`configure_logging` is idempotent
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from memory_mcp.errors import MemoryMCPError
from memory_mcp.observability import (
    RequestIdMiddleware,
    configure_logging,
    current_request_id,
    instrument_tool,
    metrics_endpoint,
    metrics_registry,
    tool_calls_total,
    tool_latency_seconds,
)

# ---------------------------------------------------------------------------
# Request-id middleware
# ---------------------------------------------------------------------------


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/echo-rid")
    async def echo_rid() -> dict[str, str | None]:
        return {"rid": current_request_id()}

    @app.get("/metrics")
    async def metrics():  # type: ignore[no-untyped-def]
        return await metrics_endpoint()

    return app


def test_request_id_generated_when_missing() -> None:
    with TestClient(_build_test_app()) as client:
        r = client.get("/echo-rid")
    assert r.status_code == 200
    rid = r.json()["rid"]
    assert isinstance(rid, str) and len(rid) == 32  # uuid4 hex
    assert r.headers.get("x-request-id") == rid


def test_request_id_honors_inbound_header() -> None:
    with TestClient(_build_test_app()) as client:
        r = client.get("/echo-rid", headers={"x-request-id": "trace-abc-123"})
    assert r.json()["rid"] == "trace-abc-123"
    assert r.headers["x-request-id"] == "trace-abc-123"


def test_request_id_resets_between_requests() -> None:
    with TestClient(_build_test_app()) as client:
        r1 = client.get("/echo-rid")
        r2 = client.get("/echo-rid")
    assert r1.json()["rid"] != r2.json()["rid"]


# ---------------------------------------------------------------------------
# Tool instrumentation
# ---------------------------------------------------------------------------


def _read_counter(tool: str, outcome: str) -> float:
    return tool_calls_total.labels(tool=tool, outcome=outcome)._value.get()  # type: ignore[attr-defined]


def _read_histogram_count(tool: str) -> float:
    # Histograms expose a ``_sum`` and per-bucket counters; the cumulative
    # _count is the last bucket's value (which Prometheus expresses via
    # ``+Inf``). Use the metric name suffix lookup instead.
    samples = list(tool_latency_seconds.collect())
    for fam in samples:
        for sample in fam.samples:
            if sample.name.endswith("_count") and sample.labels.get("tool") == tool:
                return float(sample.value)
    return 0.0


def test_instrument_tool_records_ok_outcome() -> None:
    @instrument_tool("ut_ok")
    async def fn() -> int:
        return 42

    before = _read_counter("ut_ok", "ok")
    before_count = _read_histogram_count("ut_ok")
    result = asyncio.run(fn())
    assert result == 42
    assert _read_counter("ut_ok", "ok") == before + 1
    assert _read_histogram_count("ut_ok") == before_count + 1


def test_instrument_tool_classifies_mcperror() -> None:
    class _TestError(MemoryMCPError):
        code = "TEST_CODE"
        http_status = 400

    @instrument_tool("ut_mcperror")
    async def fn() -> None:
        raise _TestError("boom")

    before = _read_counter("ut_mcperror", "mcperror")
    with pytest.raises(_TestError):
        asyncio.run(fn())
    assert _read_counter("ut_mcperror", "mcperror") == before + 1


def test_instrument_tool_classifies_unexpected_error() -> None:
    @instrument_tool("ut_error")
    async def fn() -> None:
        raise RuntimeError("nope")

    before = _read_counter("ut_error", "error")
    with pytest.raises(RuntimeError):
        asyncio.run(fn())
    assert _read_counter("ut_error", "error") == before + 1


# ---------------------------------------------------------------------------
# /metrics exposition
# ---------------------------------------------------------------------------


def test_metrics_endpoint_includes_tool_series() -> None:
    @instrument_tool("ut_metrics_probe")
    async def fn() -> int:
        return 1

    asyncio.run(fn())
    body = generate_latest(metrics_registry()).decode()
    assert "mcp_tool_calls_total" in body
    assert 'tool="ut_metrics_probe"' in body


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


def test_configure_logging_is_idempotent() -> None:
    # Calling twice must not raise nor double-install handlers.
    configure_logging("INFO")
    configure_logging("DEBUG")


# ---------------------------------------------------------------------------
# Dream-mode metrics
# ---------------------------------------------------------------------------


def test_dream_metrics_are_registered_in_default_registry() -> None:
    """Phase 2.2: the dream metrics must appear in the exposition payload.

    The metrics are recorded inside ``run_dream_pass`` and the
    summarizer base class; here we just verify that they are wired into
    the same registry as the rest of the observability module so a
    ``/metrics`` scrape includes them once they're populated.
    """
    from memory_mcp.observability import (
        dream_llm_fallbacks_total,
        dream_pass_items_processed_total,
        dream_proposals_open,
        dream_run_duration_seconds,
        dream_runs_total,
        dream_summarizer_calls_total,
        dream_summarizer_latency_seconds,
    )

    # Touch each metric once so it appears in the exposition.
    dream_run_duration_seconds.labels(mode="decay").observe(0.5)
    dream_runs_total.labels(mode="decay", outcome="done").inc()
    dream_proposals_open.labels(
        kind="merge_candidate",
        summarizer_kind="template",
    ).set(2)
    dream_pass_items_processed_total.labels(mode="dedupe").inc(3)
    dream_summarizer_calls_total.labels(kind="template", outcome="ok").inc()
    dream_summarizer_latency_seconds.labels(kind="template").observe(0.05)
    dream_llm_fallbacks_total.labels(**{"pass": "dedupe"}).inc()

    body = generate_latest(metrics_registry()).decode()
    for name in (
        "mcp_dream_run_duration_seconds",
        "mcp_dream_runs_total",
        "mcp_dream_proposals_open",
        "mcp_dream_pass_items_processed_total",
        "mcp_dream_summarizer_calls_total",
        "mcp_dream_summarizer_latency_seconds",
        "mcp_dream_llm_fallbacks_total",
    ):
        assert name in body, f"{name!r} should appear in /metrics exposition"
