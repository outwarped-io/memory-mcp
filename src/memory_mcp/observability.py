"""Observability scaffolding — structured logging, request-id, metrics.

The module is intentionally self-contained so other layers don't need to
know which library is doing the work. Three responsibilities:

* **Structured logging** via :mod:`structlog` with JSON output. Standard
  library loggers funnel through the same pipeline so third-party libs
  (uvicorn, sqlalchemy) emit consistent records.
* **Request correlation** via a ``request_id`` :class:`ContextVar`
  populated by :class:`RequestIdMiddleware`. The id is propagated to log
  records and returned via the ``X-Request-Id`` response header.
* **Prometheus metrics** scraped at ``/metrics``. v1 ships a small but
  high-signal set:

  - ``mcp_tool_calls_total{tool, outcome}`` — counter; outcome is
    ``ok`` / ``error`` / ``mcperror``.
  - ``mcp_tool_latency_seconds{tool}`` — histogram; observes successful
    AND failed calls (the ``outcome`` label on the counter distinguishes
    them).
  - ``mcp_projection_lag_seconds{sink, env_id}`` — gauge backed by
    ``projection_state.lag_seconds``; refreshed on each ``/metrics``
    scrape.
  - ``mcp_projection_event_id{sink, env_id}`` — gauge tracking the last
    delivered ``outbox.event_id`` per sink.
  - ``mcp_outbox_pending_total{sink}`` / ``mcp_outbox_dead_total{sink}``
    — gauges over ``outbox_delivery``.

OTLP exporters are intentionally NOT wired in v1; the
``OTEL_EXPORTER_OTLP_ENDPOINT`` setting is reserved and surfaces a log
warning when set so operators know it's a v1.5 feature.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any

import structlog
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from memory_mcp.config import Settings

# ---------------------------------------------------------------------------
# Request-id correlation
# ---------------------------------------------------------------------------


request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def current_request_id() -> str | None:
    """Return the request id bound to the current async context, if any."""
    return request_id_var.get()


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Bind a request id to the contextvar; echo it back as a response header.

    Honors an inbound ``X-Request-Id`` header for trace propagation; otherwise
    generates a fresh UUID4.
    """

    HEADER = "x-request-id"

    async def dispatch(  # type: ignore[override]
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get(self.HEADER) or uuid.uuid4().hex
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[self.HEADER] = rid
        return response


# ---------------------------------------------------------------------------
# structlog setup
# ---------------------------------------------------------------------------


def _add_request_id(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    rid = request_id_var.get()
    if rid is not None:
        event_dict.setdefault("request_id", rid)
    return event_dict


_LOGGING_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Install a structlog + std-lib pipeline with JSON output.

    Idempotent — repeated calls are safe but only the first installs
    the configuration. Subsequent calls update the level.
    """
    global _LOGGING_CONFIGURED

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(message)s")
    logging.getLogger().setLevel(log_level)

    if _LOGGING_CONFIGURED:
        return

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_request_id,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        cache_logger_on_first_use=True,
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger; call after :func:`configure_logging`."""
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


_REGISTRY = CollectorRegistry(auto_describe=True)


tool_calls_total: Counter = Counter(
    "mcp_tool_calls_total",
    "Total number of MCP tool invocations.",
    labelnames=("tool", "outcome"),
    registry=_REGISTRY,
)

tool_latency_seconds: Histogram = Histogram(
    "mcp_tool_latency_seconds",
    "MCP tool call latency in seconds (records both successes and failures).",
    labelnames=("tool",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=_REGISTRY,
)

projection_lag_seconds: Gauge = Gauge(
    "mcp_projection_lag_seconds",
    "Projection lag from canonical Postgres in seconds (per sink, per env).",
    labelnames=("sink", "env_id"),
    registry=_REGISTRY,
)

projection_event_id: Gauge = Gauge(
    "mcp_projection_event_id",
    "Last outbox event_id delivered to the sink (per env).",
    labelnames=("sink", "env_id"),
    registry=_REGISTRY,
)

outbox_pending_total: Gauge = Gauge(
    "mcp_outbox_pending_total",
    "Outbox deliveries currently in 'pending' or 'in_flight' status.",
    labelnames=("sink",),
    registry=_REGISTRY,
)

outbox_dead_total: Gauge = Gauge(
    "mcp_outbox_dead_total",
    "Outbox deliveries permanently failed (status='dead').",
    labelnames=("sink",),
    registry=_REGISTRY,
)

memories_total: Gauge = Gauge(
    "mcp_memories_total",
    "Canonical memory rows by environment, kind, and status.",
    labelnames=("env", "kind", "status"),
    registry=_REGISTRY,
)

memories_pinned_total: Gauge = Gauge(
    "mcp_memories_pinned_total",
    "Pinned memories by environment.",
    labelnames=("env",),
    registry=_REGISTRY,
)

memories_body_bytes_total: Gauge = Gauge(
    "mcp_memories_body_bytes_total",
    "Sum of memory body bytes by environment (body column only; excludes indexes/embeddings).",
    labelnames=("env",),
    registry=_REGISTRY,
)

memory_chain_depth: Histogram = Histogram(
    "mcp_memory_chain_depth",
    "Observed supersession chain depth samples from canonical memories.",
    buckets=(1, 2, 3, 5, 8, 13, 21),
    registry=_REGISTRY,
)

memory_age_seconds: Histogram = Histogram(
    "mcp_memory_age_seconds",
    "Observed active-memory age samples in seconds.",
    buckets=(60, 3600, 86400, 604800, 2592000, 31536000),
    registry=_REGISTRY,
)

memory_body_length_bytes: Histogram = Histogram(
    "mcp_memory_body_length_bytes",
    "Observed memory body-length samples in bytes.",
    buckets=(64, 256, 1024, 4096, 16384, 65536, 262144),
    registry=_REGISTRY,
)

memory_salience: Histogram = Histogram(
    "mcp_memory_salience",
    "Observed memory salience samples.",
    buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
    registry=_REGISTRY,
)

memory_access_count: Histogram = Histogram(
    "mcp_memory_access_count",
    "Observed memory access-count samples.",
    buckets=(0, 1, 5, 20, 100),
    registry=_REGISTRY,
)

stats_tasks_total: Gauge = Gauge(
    "mcp_tasks_total",
    "Task rows by status.",
    labelnames=("status",),
    registry=_REGISTRY,
)

stats_playbooks_total: Gauge = Gauge(
    "mcp_playbooks_total",
    "Playbook memory rows by status.",
    labelnames=("status",),
    registry=_REGISTRY,
)

stats_decisions_total: Gauge = Gauge(
    "mcp_decisions_total",
    "Decision memory rows by status.",
    labelnames=("status",),
    registry=_REGISTRY,
)

process_resident_memory_bytes: Gauge = Gauge(
    "process_resident_memory_bytes",
    "Current process resident set size in bytes from /proc/self/statm (Linux only).",
    registry=_REGISTRY,
)

_STATS_REFRESH_LOCK = asyncio.Lock()
_STATS_LAST_REFRESH = 0.0


# ---------------------------------------------------------------------------
# Dream-mode metrics (Phase 2.2)
# ---------------------------------------------------------------------------


dream_run_duration_seconds: Histogram = Histogram(
    "mcp_dream_run_duration_seconds",
    "Wall-clock duration of a single dream pass (per env, per mode).",
    labelnames=("mode",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0),
    registry=_REGISTRY,
)

dream_runs_total: Counter = Counter(
    "mcp_dream_runs_total",
    "Total dream pass executions.",
    labelnames=("mode", "outcome"),
    registry=_REGISTRY,
)

dream_proposals_open: Gauge = Gauge(
    "mcp_dream_proposals_open",
    "Open dream proposals by kind and summarizer (refreshed by /metrics scrape).",
    labelnames=("kind", "summarizer_kind"),
    registry=_REGISTRY,
)

dream_pass_items_processed_total: Counter = Counter(
    "mcp_dream_pass_items_processed_total",
    "Items inspected during a dream pass (clusters, candidates, transitions).",
    labelnames=("mode",),
    registry=_REGISTRY,
)

dream_summarizer_calls_total: Counter = Counter(
    "mcp_dream_summarizer_calls_total",
    "Summarizer invocations (per kind, per outcome).",
    labelnames=("kind", "outcome"),
    registry=_REGISTRY,
)

dream_summarizer_latency_seconds: Histogram = Histogram(
    "mcp_dream_summarizer_latency_seconds",
    "Summarizer call latency in seconds (per kind).",
    labelnames=("kind",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=_REGISTRY,
)

dream_llm_fallbacks_total: Counter = Counter(
    "mcp_dream_llm_fallbacks_total",
    (
        "Times the LLMSummarizer fell back to template-style content "
        "due to a per-call failure (timeout, parse error, network error)."
    ),
    labelnames=("pass",),
    registry=_REGISTRY,
)


def metrics_registry() -> CollectorRegistry:
    """Expose the registry — primarily for tests."""
    return _REGISTRY


# ---------------------------------------------------------------------------
# Tool instrumentation decorator
# ---------------------------------------------------------------------------


def instrument_tool[T](
    name: str,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Wrap an async tool callable with metrics + per-call structured log.

    The wrapper:
    * Observes the call latency on :data:`tool_latency_seconds`.
    * Increments :data:`tool_calls_total` with ``outcome`` ∈
      {``ok``, ``mcperror``, ``error``}.
    * Emits a structured log line on completion (``mcp.tool.complete``).

    ``mcperror`` is reserved for ``MemoryMCPError`` (caller-correctable);
    ``error`` covers unexpected exceptions.
    """
    log = get_logger("memory_mcp.tool")

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(fn)
        async def inner(*args: Any, **kwargs: Any) -> T:
            from memory_mcp.errors import MemoryMCPError  # local import — avoids cycle

            start = time.perf_counter()
            outcome = "ok"
            try:
                return await fn(*args, **kwargs)
            except MemoryMCPError:
                outcome = "mcperror"
                raise
            except Exception:
                outcome = "error"
                raise
            finally:
                elapsed = time.perf_counter() - start
                tool_latency_seconds.labels(tool=name).observe(elapsed)
                tool_calls_total.labels(tool=name, outcome=outcome).inc()
                log.info(
                    "mcp.tool.complete",
                    tool=name,
                    outcome=outcome,
                    latency_seconds=round(elapsed, 6),
                )

        return inner

    return decorator


# ---------------------------------------------------------------------------
# Projection metric refresh (lazy — invoked by /metrics)
# ---------------------------------------------------------------------------


async def refresh_projection_metrics() -> None:
    """Read projection_state + outbox_delivery and update gauges.

    Best-effort — exceptions are swallowed and logged so a transient DB hiccup
    cannot break ``/metrics`` scraping.
    """
    log = get_logger("memory_mcp.observability")
    try:
        from sqlalchemy import func, select

        from memory_mcp.db.models import OutboxDelivery, ProjectionState
        from memory_mcp.db.postgres import session_scope

        async with session_scope() as s:
            # projection_state → per-(sink, env) gauges
            ps_rows = (await s.execute(select(ProjectionState))).scalars().all()
            for ps in ps_rows:
                env_label = str(ps.env_id) if ps.env_id else "_all"
                if ps.lag_seconds is not None:
                    projection_lag_seconds.labels(
                        sink=ps.sink,
                        env_id=env_label,
                    ).set(float(ps.lag_seconds))
                if ps.last_event_id is not None:
                    projection_event_id.labels(
                        sink=ps.sink,
                        env_id=env_label,
                    ).set(float(ps.last_event_id))

            # outbox_delivery aggregates → per-sink gauges
            agg = await s.execute(
                select(
                    OutboxDelivery.sink,
                    OutboxDelivery.status,
                    func.count(),
                ).group_by(OutboxDelivery.sink, OutboxDelivery.status)
            )
            pending: dict[str, int] = {}
            dead: dict[str, int] = {}
            for sink, status, count in agg.all():
                if status in ("pending", "in_flight"):
                    pending[sink] = pending.get(sink, 0) + int(count)
                elif status == "dead":
                    dead[sink] = dead.get(sink, 0) + int(count)
            for sink, count in pending.items():
                outbox_pending_total.labels(sink=sink).set(count)
            for sink, count in dead.items():
                outbox_dead_total.labels(sink=sink).set(count)
    except Exception as exc:  # noqa: BLE001 — best-effort metric refresh
        log.warning("observability.refresh_failed", error=str(exc)[:200])


def _stats_refresh_interval_seconds() -> float:
    raw = os.environ.get("MCP_METRICS_REFRESH_INTERVAL_SECONDS")
    if raw is None:
        return 60.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 60.0


async def refresh_metrics_on_scrape_v10(*, force: bool = False) -> None:
    """Refresh v0.10 stats gauges/histograms, TTL-gated for expensive distributions."""
    global _STATS_LAST_REFRESH
    log = get_logger("memory_mcp.observability")
    try:
        from sqlalchemy import text

        from memory_mcp.db.postgres import session_scope
        from memory_mcp.stats import read_process_rss

        rss = read_process_rss()
        if rss.rss_bytes is not None:
            process_resident_memory_bytes.set(float(rss.rss_bytes))

        now = time.monotonic()
        if not force and now - _STATS_LAST_REFRESH < _stats_refresh_interval_seconds():
            return
        async with _STATS_REFRESH_LOCK:
            now = time.monotonic()
            if not force and now - _STATS_LAST_REFRESH < _stats_refresh_interval_seconds():
                return
            async with session_scope() as s:
                mem_rows = await s.execute(
                    text("""
                    SELECT COALESCE(e.name, m.env_id::text) AS env, m.kind, m.status, COUNT(*) AS count
                    FROM memories m
                    LEFT JOIN environments e ON e.id = m.env_id
                    GROUP BY env, m.kind, m.status
                """)
                )
                for env, kind, status, count in mem_rows.all():
                    memories_total.labels(env=str(env), kind=str(kind), status=str(status)).set(int(count))

                env_rows = await s.execute(
                    text("""
                    SELECT COALESCE(e.name, m.env_id::text) AS env,
                           COUNT(*) FILTER (WHERE m.pinned) AS pinned,
                           SUM(octet_length(m.body)) AS body_bytes
                    FROM memories m
                    LEFT JOIN environments e ON e.id = m.env_id
                    GROUP BY env
                """)
                )
                for env, pinned, body_bytes in env_rows.all():
                    memories_pinned_total.labels(env=str(env)).set(int(pinned or 0))
                    memories_body_bytes_total.labels(env=str(env)).set(int(body_bytes or 0))

                for metric, sql in (
                    (stats_tasks_total, "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"),
                    (
                        stats_playbooks_total,
                        "SELECT status, COUNT(*) AS count FROM memories WHERE kind = 'playbook' GROUP BY status",
                    ),
                    (
                        stats_decisions_total,
                        "SELECT status, COUNT(*) AS count FROM memories WHERE kind = 'decision' GROUP BY status",
                    ),
                ):
                    rows = await s.execute(text(sql))
                    for status, count in rows.all():
                        metric.labels(status=str(status)).set(int(count))

                chain_rows = await s.execute(
                    text("""
                    WITH RECURSIVE chain(root_id, id, env_id, depth) AS (
                        SELECT id, id, env_id, 1 FROM memories WHERE superseded_by IS NULL
                        UNION ALL
                        SELECT c.root_id, m.id, m.env_id, c.depth + 1
                        FROM chain c
                        JOIN memories m ON m.superseded_by = c.id AND m.env_id = c.env_id
                        WHERE c.depth < 1000
                    ), depths AS (
                        SELECT root_id, MAX(depth) AS depth FROM chain GROUP BY root_id
                    )
                    SELECT depth FROM depths
                """)
                )
                for (depth,) in chain_rows.all():
                    memory_chain_depth.observe(float(depth))

                sample_rows = await s.execute(
                    text("""
                    SELECT octet_length(body) AS body_length,
                           CASE WHEN status = 'active' THEN EXTRACT(EPOCH FROM (now() - created_at)) ELSE NULL END AS age_seconds,
                           salience::float AS salience,
                           access_count AS access_count
                    FROM memories
                """)
                )
                for body_length, age_seconds, salience, access_count in sample_rows.all():
                    memory_body_length_bytes.observe(float(body_length))
                    if age_seconds is not None:
                        memory_age_seconds.observe(float(age_seconds))
                    memory_salience.observe(float(salience))
                    memory_access_count.observe(float(access_count))
            _STATS_LAST_REFRESH = time.monotonic()
    except Exception as exc:  # noqa: BLE001 — best-effort metric refresh
        log.warning("observability.stats_refresh_failed", error=str(exc)[:200])


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------


async def metrics_endpoint() -> Response:
    """Prometheus exposition. Refreshes projection/outbox and v0.10 stats gauges before rendering."""
    await refresh_projection_metrics()
    await refresh_metrics_on_scrape_v10()
    body = generate_latest(_REGISTRY)
    return Response(content=body, media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# OTLP export (deferred to v1.5)
# ---------------------------------------------------------------------------


def warn_if_otlp_configured(settings: Settings) -> None:
    """Log a one-time warning if OTLP endpoint is set; v1.5 will wire it."""
    if settings.otel_exporter_otlp_endpoint:
        log = get_logger("memory_mcp.observability")
        log.warning(
            "otlp.deferred",
            endpoint=settings.otel_exporter_otlp_endpoint,
            note="OTLP export is deferred to v1.5; metrics available at /metrics",
        )
