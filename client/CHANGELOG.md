# memory-mcp-client SDK changelog

## [0.4.0] — 2026-05-15 — Batch helpers · v0.13 schema alignment

### Added
- **Batch helpers** — `MemoryClient.memories.write_many()`,
  `entities.upsert_many()`, and `tasks.create_many()` issue a list of
  per-item requests concurrently and return a typed `BatchResult[T]`
  envelope with parallel `successes`, `failures`, and originating-index
  tracking. Concurrency is bounded by `max_concurrency` (default 8).
  Use these for bulk-seed flows (canonical-entity rollouts, bootstrap
  re-mirrors, dataset imports) where serial round-trips dominate
  latency. Failures are surfaced typed (no partial exceptions thrown).
- **`BatchResult` model** — exported from `memory_mcp_client`.
  Fields: `successes: list[BatchSuccess[T]]`, `failures:
  list[BatchFailure]`, helpers `.has_failures`, `.total`.

### Changed
- **Pinned `memory-mcp-schemas>=0.13,<0.14`** — picks up the v0.13
  schema surface: `MemoryHardDeleteRequest.cascade` + `dry_run` +
  `max_cascade_depth` + `max_cascade_count`; `MemorySearchRequest.expansion`
  preset enum; `MemNeighborsRequest` / `MemRelatedRequest` fallback
  cascade fields; 18 schemas gain `env_name` resolver alongside `env_id`.

### Notes
- No `0.3.1` release — the version skipped straight from `0.3.0` to
  `0.4.0` to align with the v0.13 server schema bump.

## [0.2.0] — 2026-05-14 — Typed responses · Retry policy · Live integration tests

### Added
- **Typed mutation responses** — `supersede()` now returns a
  `MemorySupersedeResponse(old, new, superseded_at, projection_status?)`
  Pydantic model (was: `dict[str, Any]`). `journal()` returns a
  dedicated `JournalResponse` (subclass of `MemoryResponse`). Callers
  get full IDE / mypy support on the response shape.
- **`hard_delete()` SDK method** — wires the new v0.11 server tool
  `mem_hard_delete`. Returns a typed `MemoryHardDeleteResponse` with
  `deleted_id`, `deleted_at`, `tombstone_id`, and a
  `projection_eviction` state envelope.
- **Retry policy** — every tool call passes through
  :class:`memory_mcp_client.RetryPolicy`. Reads (`mem_search`,
  `mem_get`, `env_list_`, `task_list`, …) retry by default on transient
  transport errors (`ConnectionError`, `httpx.TransportError`,
  `asyncio.TimeoutError`) and rate-limit signals
  (`RATE_LIMITED`, `GRAPH_BACKEND_UNAVAILABLE`). Writes (`mem_write`,
  `env_*`, …) opt in either via `retry_writes=True` on the client or
  per-call `idempotency_key`. Defaults: 3 attempts · 250 ms base ·
  exponential cap 4 s · ±50 % jitter. Configurable.
- **`RetryExhaustedError`** — distinct typed error raised when a
  retryable call burns through all attempts; the original error is
  preserved on `__cause__`.
- **`ValidationFailedError`** — typed error for the new v0.11 server
  `VALIDATION_FAILED` code; `details["hints"]` carries did-you-mean
  field suggestions.
- **Live integration tests** — `client/tests/integration/` runs one
  happy-path per namespace against a real server (gated on
  `MEMORY_MCP_INTEGRATION_URL`). Hermetic by default — pytest filters
  via the `integration` marker. Run via `make integration`.
- **`Makefile`** — `make test`, `make integration`, `make lint`,
  `make typecheck`, `make all`.

### Changed
- `MemoryClient.__init__` accepts a new `retry_policy: RetryPolicy`
  parameter. Existing call sites keep their previous behavior because
  the default policy retries reads only and writes pass through.
- `supersede()` is a **breaking API change** — callers that destructure
  the result (`out["old"]`, `out["new"]`) must switch to attribute
  access (`out.old`, `out.new`).
- `__version__` bumped from `0.7.1` (last-tagged SDK) to `0.2.0`. SDK
  versioning now decouples from the server's; the SDK starts a fresh
  pre-1.0 sequence to make API stability commitments unambiguous.

### Compatibility
- Requires memory-mcp server **v0.11.0+** for `hard_delete()`,
  `JournalResponse`, and the structured `VALIDATION_FAILED` hint
  payload. The retry policy and typed `supersede()` work against v0.10+
  servers because the server-side response shape is unchanged — only
  the SDK now deserializes it into a typed model.

## Older

The SDK shipped inside the `memory-mcp-schemas` repo's release notes
prior to this changelog. See the server changelog
(`repos/memory-mcp/CHANGELOG.md`) for v0.5–v0.10 SDK additions
(namespaces, identity headers, env_ops, dream, etc.).
