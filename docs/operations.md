# Operator runbook

Day-to-day operational workflows for memory-mcp.

## Operator runbook

### Connection-string hardening
The compose stack defaults are **dev credentials** (`memory:memory`, `neo4j:memorymemory`). Rotate before any persistent deployment.

### Bind host
- Default: `MEMORY_MCP_HOST=127.0.0.1` (loopback).
- For trusted-network deployments (e.g., a private bridge with known peers), set `MEMORY_MCP_HOST=0.0.0.0` AND firewall the port externally. The server logs a `WARN` at startup confirming the non-loopback bind.

### Observability
- `GET /healthz` — process liveness + posture flags.
- `GET /readyz` — dependency probes (Postgres always; Qdrant best-effort with 2s timeout).
- `GET /metrics` — Prometheus exposition. Notable series:
  - `mcp_tool_calls_total{tool, outcome}` — outcomes are `ok` / `mcperror` / `error`.
  - `mcp_tool_latency_seconds{tool}` — histogram.
  - `mcp_projection_lag_seconds{sink, env_id}` — outbox→sink lag (refreshed per scrape).
  - `mcp_outbox_pending_total{sink}` / `mcp_outbox_dead_total{sink}` — backlog & dead-letter counters.
- Logs are JSON via structlog; every log record carries `request_id` (from the inbound `X-Request-Id` header or generated). Echo the `X-Request-Id` response header to correlate with client traces.

### OpenTelemetry (deferred to v1.5)
Setting `OTEL_EXPORTER_OTLP_ENDPOINT` emits a one-time `WARN` and is otherwise ignored in v1. Wire OTLP push in v1.5.

### Backup
**Postgres is the only required backup.** Qdrant and Neo4j are caches, rebuildable from canonical Postgres via Phase-2 admin tools. Standard `pg_dump` / PITR.

### Projection-worker outage
If the projection-worker is down, writes still succeed (outbox grows). `mem_search` with `consistency=fresh` will time out; clients should fall back to `consistency=canonical` (Postgres-only, lex + ID match).

### Dream mode (Phase 2.2)

The **dream-worker** is a separate background process that periodically re-shapes the knowledge graph: it decays salience over time, clusters near-duplicates, promotes recurring observations into facts, and emits **proposals** for human/agent review. Mutating the canonical store always goes through `dream_review_(action="accept")` — the worker never auto-applies.

#### Three passes
- **decay** — walks `active → stale → archived` based on the recomputed `salience` score. `pinned` memories never archive; `retired` / `superseded` are skipped.
- **dedupe** — for each `active` memory in the dedupe window, queries Qdrant for top-K neighbors and clusters by cosine ≥ threshold (default `0.92`). Each cluster ≥ 2 members emits a `merge_candidate` proposal.
- **promote** — clusters recent journal observations referencing the same entity; ≥ N observations emit a `promotion_candidate` proposal that, when accepted, becomes a `proposed` `fact` memory.

A fourth APScheduler job refreshes the `mcp_dream_proposals_open{kind, summarizer_kind}` Prometheus gauge from SQL.

#### Two configuration profiles

| | **Light profile** (default) | **LLM profile** |
|---|---|---|
| `DREAM_SUMMARIZER` | `template` | `llm` |
| Compute | None (pure Python) | Ollama or OpenAI-compatible endpoint |
| Proposal `suggested_*` quality | Longest-member title/body, structured templates | Natural-language summary |
| Failure mode | None — deterministic | LLM unreachable → per-call fallback to template content with `llm_failed=true` |
| `/readyz` LLM probe | Skipped | Best-effort, 2s timeout |
| Recommended for | Air-gapped, low-resource, CI | Reviewer ergonomics, larger deployments |

To run with Ollama locally:
```bash
docker compose --profile llm up -d            # also brings up the `ollama` sidecar
docker compose exec ollama ollama pull llama3.2:3b
DREAM_SUMMARIZER=llm LLM_BACKEND=ollama \
  LLM_BASE_URL=http://ollama:<ollama-port> LLM_MODEL_ID=llama3.2:3b \
  docker compose up -d server dream-worker
```

To target a remote OpenAI-compatible endpoint instead:
```bash
DREAM_SUMMARIZER=llm LLM_BACKEND=openai_compatible \
  LLM_BASE_URL=https://api.openai.com/v1 LLM_API_KEY=sk-... \
  LLM_MODEL_ID=gpt-4o-mini \
  docker compose up -d server dream-worker
```

#### Knobs

| Variable | Default | Notes |
|---|---|---|
| `DREAM_ENABLED` | `false` | Master switch for the worker scheduler. `dream_run_` MCP tool always works regardless. |
| `DREAM_SUMMARIZER` | `llm` | `llm` or `template`. Falls back to template content per-call when the LLM is unreachable. |
| `DREAM_DECAY_INACTIVE_DAYS` | `30` | Skip rows accessed within this window. |
| `DREAM_DECAY_STALE_THRESHOLD` | `0.30` | `salience` below this → `stale`. |
| `DREAM_DECAY_ARCHIVE_THRESHOLD` | `0.10` | `salience` below this → `archived`. |
| `DREAM_DEDUPE_THRESHOLD` | `0.92` | Cosine threshold for clustering. |
| `MEMORY_MCP_DECISION_CONFLICT_COSINE_THRESHOLD` | `0.85` | Cosine cutoff for `DreamMode.decision_conflicts`; accepted decision pairs at or above this score emit `decision_conflict_candidate` proposals. |
| `DREAM_DEDUPE_TOP_K` | `10` | Per-row Qdrant neighbor query size. |
| `DREAM_DEDUPE_WINDOW_DAYS` | `7` | Look-back for changed-or-new memories. |
| `DREAM_PROMOTE_MIN_CLUSTER_SIZE` | `3` | Observations per entity required to emit a promotion. |
| `DREAM_PROMOTE_WINDOW_DAYS` | `14` | Look-back for journal observations. |
| `DREAM_DECAY_CADENCE_SECONDS` | `3600` | APScheduler interval (worker-only). |
| `DREAM_DEDUPE_CADENCE_SECONDS` | `1800` | APScheduler interval. |
| `DREAM_PROMOTE_CADENCE_SECONDS` | `3600` | APScheduler interval. |
| `DREAM_METRICS_REFRESH_SECONDS` | `60` | `0` disables the open-proposal gauge refresher. |
| `DREAM_PASS_TIMEOUT_SECONDS` | `600` | Per-pass wall-clock cap; over-time runs land as `failed`. |
| `LLM_BACKEND` | `null` | `ollama` / `openai_compatible` / `null`. `null` is the test/CI default — calling it raises `LLMUnavailableError`. |
| `LLM_BASE_URL` | _(empty)_ | URL for the chosen backend. |
| `LLM_MODEL_ID` | `llama3.2:3b` | Default Ollama model; override per backend. |
| `LLM_API_KEY` | _(empty)_ | Used by `openai_compatible`. |
| `LLM_TIMEOUT_SECONDS` | `60.0` | Per-call cap. |

#### Manual trigger from the CLI

```jsonc
// dream_run_  → kick a single dedupe pass synchronously
{ "request": { "env_id": "<env-uuid>", "modes": ["dedupe"], "wait": true } }

// dream_proposals_list_  → browse open proposals
{ "request": { "env_id": "<env-uuid>", "kind": "merge_candidate", "status": "open" } }

// dream_review_  → accept (merges by kind, supersedes the duplicates)
{ "request": { "proposal_id": "<proposal-uuid>", "action": "accept" } }

// dream_status_  → recent runs + open-proposal counts + LLM probe
{ "request": { "env_id": "<env-uuid>", "runs_per_mode": 5 } }
```

Sample `dream_run_(dedupe, wait=true)` response:
```jsonc
{
  "scheduled": [{ "env_id": "...", "mode": "dedupe" }],
  "reports": [{
    "env_id": "...",
    "mode": "dedupe",
    "outcome": "done",
    "dream_run_id": "...",
    "summary": { "clusters_emitted": 1, "proposals_inserted": 1 },
    "duration_seconds": 0.42
  }]
}
```

Sample `merge_candidate` payload (template summarizer):
```jsonc
{
  "primary_id": "...",
  "candidate_ids": ["...", "..."],
  "cosine_scores": [1.0, 0.97, 0.94],
  "suggested_merged_title": "DreamSmokeDuplicate v0",
  "suggested_merged_body": "memory-mcp dream pass clusters near-duplicate facts together",
  "summarizer_kind": "template",
  "llm_failed": false,
  "llm_model_id": null
}
```

To **disable dream mode** entirely, leave `DREAM_ENABLED=false` (the default in compose) — the worker stays in heartbeat-only idle. The MCP `dream_run_` tool still triggers passes on demand.

#### Smoke

The end-to-end smoke (see step 5 above) defaults to `DREAM_SUMMARIZER=template` for hermetic runs. Switch the server env to `DREAM_SUMMARIZER=llm` and bring up the `llm` profile to exercise the LLM path.

---

