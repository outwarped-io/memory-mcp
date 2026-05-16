"""memory-mcp HTTP server entrypoint.

Phase 0/1 transitional: minimal FastAPI app exposing ``/healthz`` and
``/readyz`` with **local-only build flags surfaced in the payload**. Phase
1's ``p1-mcp-transport`` todo will mount the MCP Streamable HTTP transport
and register the tool families under ``memory_mcp.tools.*``.

v1 = local-only safety
----------------------

* ``Settings.mcp_http_host`` defaults to ``127.0.0.1`` so the listener
  refuses non-loopback connections out of the box.
* If an operator sets ``MCP_HTTP_HOST`` to a non-loopback address, the
  startup logs a loud warning and ``/healthz`` reports
  ``"unsafe_remote_bind": true`` so monitoring picks it up.
* ``/healthz`` advertises ``"local_only": true`` and ``"auth": "disabled"``
  so clients can detect they're talking to a v1 build before sending any
  data.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

from memory_mcp.config import Settings, get_settings
from memory_mcp.db.postgres import dispose_engine, init_engine, session_scope
from memory_mcp.graph import _close_default_graph_store
from memory_mcp.mcp_app import build_mcp_server
from memory_mcp.observability import (
    RequestIdMiddleware,
    configure_logging,
    metrics_endpoint,
    warn_if_otlp_configured,
)

logger = logging.getLogger("memory_mcp.server")

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS


def _healthz_payload(settings: Settings) -> dict[str, object]:
    unsafe = not _is_loopback(settings.mcp_http_host)
    return {
        "status": "ok",
        "local_only": True,
        "auth": "disabled",
        "bind_host": settings.mcp_http_host,
        "unsafe_remote_bind": unsafe,
        "transport": {
            "mcp": {"path": "/mcp", "protocol": "streamable_http"},
            "rest": {"healthz": "/healthz", "readyz": "/readyz"},
        },
    }


async def _probe_postgres() -> dict[str, Any]:
    import asyncio

    try:
        async def _do() -> None:
            async with session_scope() as s:
                from sqlalchemy import text
                await s.execute(text("SELECT 1"))

        await asyncio.wait_for(_do(), timeout=2.0)
        return {"status": "ok"}
    except TimeoutError:
        return {"status": "error", "error": "probe timed out after 2.0s"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)[:200]}


async def _probe_qdrant(settings: Settings) -> dict[str, Any]:
    import asyncio

    if settings.vector_backend != "qdrant":
        return {"status": "skipped", "reason": f"vector_backend={settings.vector_backend!r}"}
    try:
        from memory_mcp.db.vector.qdrant import QdrantVectorStore
        store = QdrantVectorStore(settings)
        try:
            await asyncio.wait_for(store.client.get_collections(), timeout=2.0)
            return {"status": "ok"}
        finally:
            await store.close()
    except TimeoutError:
        return {"status": "error", "error": "probe timed out after 2.0s"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)[:200]}


async def _probe_neo4j(settings: Settings) -> dict[str, Any]:
    """Forward to the driver-level probe so `/readyz` can call uniformly."""
    from memory_mcp.db.graph.neo4j import probe_neo4j

    return await probe_neo4j(settings)


async def _probe_llm(settings: Settings) -> dict[str, Any]:
    """Forward to ``memory_mcp.llm.probe_llm`` (lazy-imported).

    Lazy-import keeps server start-up free from the LLM module tree when
    no LLM is configured. ``probe_llm`` itself returns a ``skipped`` status
    when the LLM subsystem is intentionally inert (``backend=null`` or
    ``dream_summarizer=template``), so the import only matters when the
    operator has wired an actual backend.
    """
    from memory_mcp.llm import probe_llm

    return await probe_llm(settings)


@asynccontextmanager
async def _service_lifespan(
    settings: Settings,
    mcp_server: FastMCP,
    *,
    run_http_session_manager: bool,
) -> AsyncIterator[None]:
    """Shared service startup/shutdown around the selected MCP transport."""
    init_engine(settings)

    async def _ready() -> None:
        if not _is_loopback(settings.mcp_http_host):
            logger.warning(
                "memory-mcp: binding to non-loopback host %r — v1 has NO "
                "AUTH; do not expose to untrusted networks",
                settings.mcp_http_host,
            )
        logger.info(
            "memory-mcp ready: transport=%s bind=%s:%s mcp_path=/mcp tools=%d",
            settings.mcp_transport,
            settings.mcp_http_host,
            settings.mcp_http_port,
            len(await mcp_server.list_tools()),
        )

    try:
        if run_http_session_manager:
            async with mcp_server.session_manager.run():
                await _ready()
                yield
        else:
            await _ready()
            yield
    finally:
        await _close_default_graph_store()
        await dispose_engine()


def build_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app with MCP transport mounted.

    Returns a fresh app each call so tests can construct isolated instances.
    """
    settings = settings or get_settings()

    # Observability is configured at app build so module-level loggers
    # downstream are guaranteed to flow through structlog/JSON.
    configure_logging(settings.log_level)
    warn_if_otlp_configured(settings)

    mcp_server = build_mcp_server(settings)
    mcp_streamable_app = mcp_server.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with _service_lifespan(
            settings,
            mcp_server,
            run_http_session_manager=True,
        ):
            yield

    app = FastAPI(title="memory-mcp", version="0.0.1", lifespan=lifespan)
    # Request-id correlation: bind a UUID4 to the contextvar so all log
    # records, downstream tool invocations, and error responses share a
    # single id. Honors inbound ``X-Request-Id`` for trace propagation.
    app.add_middleware(RequestIdMiddleware)

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return _healthz_payload(get_settings())

    @app.get("/readyz")
    async def readyz() -> dict[str, object]:
        cur_settings = get_settings()
        payload = _healthz_payload(cur_settings)
        deps = {
            "postgres": await _probe_postgres(),
            "qdrant": await _probe_qdrant(cur_settings),
            "neo4j": await _probe_neo4j(cur_settings),
            "llm": await _probe_llm(cur_settings),
        }
        all_ok = all(
            d.get("status") in ("ok", "skipped") for d in deps.values()
        )
        payload["status"] = "ok" if all_ok else "degraded"
        payload["dependencies"] = deps
        return payload

    @app.get("/metrics")
    async def metrics() -> Any:  # type: ignore[misc] — Response-typed
        return await metrics_endpoint()

    app.mount("/mcp", mcp_streamable_app)

    return app


def _build_module_app() -> FastAPI | None:
    settings = get_settings()
    if settings.mcp_transport == "stdio":
        return None
    return build_app(settings)


# Module-level app for ``uvicorn memory_mcp.server:app`` in HTTP mode.
app = _build_module_app()


async def _run_stdio_async(settings: Settings) -> None:
    configure_logging(settings.log_level)
    warn_if_otlp_configured(settings)
    mcp_server = build_mcp_server(settings)
    async with _service_lifespan(
        settings,
        mcp_server,
        run_http_session_manager=False,
    ):
        await mcp_server.run_stdio_async()


def run_stdio(settings: Settings) -> None:
    asyncio.run(_run_stdio_async(settings))


def main() -> None:
    """Console-script entrypoint.

    Run uvicorn programmatically so the same binary can be used inside Docker
    and locally. Bind host comes from ``Settings.mcp_http_host`` (defaults to
    ``127.0.0.1``); operators must override deliberately to expose remotely.
    """
    import uvicorn

    settings = get_settings()
    if settings.mcp_transport == "stdio":
        run_stdio(settings)
        return

    if not _is_loopback(settings.mcp_http_host):
        logger.warning(
            "memory-mcp: binding to non-loopback host %r — v1 has NO AUTH; "
            "do not expose to untrusted networks",
            settings.mcp_http_host,
        )

    uvicorn.run(
        "memory_mcp.server:app",
        host=settings.mcp_http_host,
        port=settings.mcp_http_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
