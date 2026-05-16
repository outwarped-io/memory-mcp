# memory-mcp

A shared, multi-agent **Memory MCP server** exposing 58 MCP tools. Stores facts, procedures, playbooks, tasks, events, decisions, preferences, observations, and snippets across sessions and across agents. Backed by **Postgres (truth) + Qdrant (vectors) + Neo4j (graph)**, exposed over **MCP Streamable HTTP** or **stdio**.

> **Status:** v0.13 — core CRUD, journal, entities, relations, search (lex + sem + hybrid), active recall, procedures/plans, environment operations, hard-delete cascade, search/graph relax presets, and a full stats surface (`mem_stats` MCP tool + Prometheus expansion + Grafana dashboard). Backed by outbox-driven vector/graph projections and the dream worker.

---

## Install (Copilot CLI plugin)

The fastest path:

```text
copilot
> /plugin install outwarped-io/memory-mcp
> /plugin enable memory-mcp
```

Then start the backing services (Postgres + Qdrant + Neo4j + server +
projection-worker):

```bash
cd ~/.copilot/installed-plugins/memory-mcp     # exact path may differ — check ${CLAUDE_PLUGIN_ROOT}
cp .env.example .env                            # edit secrets if needed
docker compose up -d
```

That's it. The plugin manifest registers a stdio MCP server that runs
`docker compose exec server env MCP_TRANSPORT=stdio python -m memory_mcp.server`
against the running stack; Copilot CLI spawns it on demand.

To uninstall: `/plugin disable memory-mcp` → `docker compose down -v` →
`/plugin uninstall memory-mcp`.

### Install without the Copilot CLI plugin command

If you'd rather skip `/plugin install`, the manual path is:

```bash
git clone https://github.com/outwarped-io/memory-mcp.git
cd memory-mcp
cp .env.example .env
docker compose up -d
```

Then drop the appropriate snippet into your MCP client config:

- **Copilot CLI** (stdio against the running container):
  ```json
  {
    "mcpServers": {
      "memory-mcp": {
        "command": "docker",
        "args": ["exec", "-i", "memory-mcp-server-1",
                 "env", "MCP_TRANSPORT=stdio",
                 "python", "-m", "memory_mcp.server"]
      }
    }
  }
  ```
- **Any HTTP-capable MCP client** (Streamable HTTP):
  ```json
  {
    "mcpServers": {
      "memory-mcp": { "type": "http", "url": "http://127.0.0.1:8080/mcp" }
    }
  }
  ```

The container name `memory-mcp-server-1` comes from the compose project name
`memory-mcp` in `docker-compose.yml`.

---

## ⚠️ v1 is LOCAL-ONLY — do not expose

This release **has no authentication** and **no RBAC enforcement**. Any caller that can reach the HTTP endpoint can read and write every memory in every environment.

The server, when run via `docker-compose`, **publishes its port to host loopback only** (`ports: 127.0.0.1:8080:8080`). The container itself binds to `0.0.0.0` internally so Docker can route requests, but the host-side port is gated to `127.0.0.1`. To expose remotely, set `MCP_HTTP_BIND=0.0.0.0` in `.env` AND understand that v1 has no auth.

For a non-Docker direct `uvicorn memory_mcp.server:app` run, set `MCP_HTTP_HOST=127.0.0.1`.

`/healthz` advertises this status:

```json
{ "local_only": true, "auth": "disabled", "transport": { "mcp": { "path": "/mcp" } } }
```

A `WARN` is logged at startup if the bind host is not loopback. Multi-tenant auth (hashed-bearer tokens, `env_grants`, OAuth) is forward-compatible with the v1 schema and lands in v1.5.

---

## What & why

Agents start every session cold. memory-mcp gives them a shared, queryable, time-aware long-term memory that:

- Is **read/write by many agents at once** (HTTP/SSE transport).
- Is **partitioned by environment / profile** (`work`, `private`, `school`, `work_project1`, …); a session can attach multiple envs at once via the `X-Env-Ids` header.
- Combines **lexical + semantic + graph** search (Postgres FTS, embeddings, entity expansion).
- Has a **lifecycle** (`proposed → active → stale → archived → superseded/retired`) and a **dream worker** (Phase 2) that decays, deduplicates, and promotes records in the background.
- Is **installable as a Copilot CLI plugin** — see [Install](#install-copilot-cli-plugin) above.

### Active Recall (v0.6.0)

**F1 — trigger-conditioned auto-context:** `mem_write` accepts `trigger_description`, a short phrase describing when the memory should apply. `mem_auto_context(task_desc, env_id, top_k=8)` searches the separate Qdrant `trigger` named vector and returns matching memories, while memories without triggers remain available through normal `mem_search`.

**F5 — session digest + resume:** `mem_digest(env_id, since_ts?)` persists a six-section `session_digest` (`brief`, `active_context`, `system_patterns`, `tech_context`, `progress`, `open_questions`) using the configured dream LLM backend, with deterministic template fallback when unavailable. `mem_resume(env_id, journal_tail=20)` returns the latest digest plus recent journal entries.

**F7 — compound context pack:** `mem_context_pack(task_desc, env_id, token_budget=4000, include_journal=true)` builds a token-budgeted startup bundle from the latest digest, F1 trigger matches, recent journal, and salience-ranked archival memories. v0.7.0 upgrades it to F7 v2 with tasks, accepted decisions, and matching playbooks in the canonical pack order.

### Procedures & Plans (v0.7.0)

Procedures & Plans adds a durable execution substrate for agents. Playbooks store reusable procedures as `kind="playbook"` memories with `steps` and a case-insensitive `macro`, then `playbook_invoke(macro, env_id)` retrieves the procedure and resolves referenced memories.

The task graph records multi-session plans as first-class `Task` nodes with dependency edges, status transitions, `task_next(env_id)` for the next unblocked pending task, and read-only `task_tree(task_id, max_depth=10, max_nodes=200)` for subtree review. ADR-lite decisions add structured `decision_meta`, including optional `consequences`, and `adr_export(memory_id)` so accepted choices, rationale, constraints, consequences, and supersession chains remain inspectable.

F7 v2 expands `mem_context_pack` so startup packs automatically surface relevant tasks, accepted decisions, and matching playbooks alongside digest, trigger-matched, journal, and archival sections. Migrations 0009/0010/0011 are additive and reversible when upgrading from v0.6.0. v0.7.1 hardening adds real-Postgres race tests, first-class ADR consequences, decision-conflict proposals (`MEMORY_MCP_DECISION_CONFLICT_COSINE_THRESHOLD`, default `0.85`), `{{task:<uuid>}}` playbook placeholders, and the `task_tree` MCP tool.

## Environment Operations (v0.8)

memory-mcp 0.8 introduces full environment lifecycle management. Treat an environment as a portable unit you can export, import, diff, merge, clone, migrate, snapshot, and rename — and per-memory copy/move operations between envs.

### Core capabilities

| Tool | Purpose |
|---|---|
| `env_export` | Dump env to archive (`.tar.gz`) or directory |
| `env_import` | Load archive into new/existing env with mode (fail/skip/overwrite/merge) |
| `env_diff` | Compare two envs at 4 granularity levels |
| `env_clone` | Duplicate an env (full or filtered with closure expansion) |
| `env_merge` | Pairwise merge two envs (entity_merge invoked on canonical_key collisions) |
| `env_migrate` | Bulk filtered memory migration between envs |
| `env_snapshot` | Labeled archive retained on local disk |
| `env_restore` | Restore in-place (preserves UUIDs) or to a new env |
| `env_delete` | Soft-delete env + cascade hard-delete its rows |
| `env_rename` | Update env name / embedding model / retention policy |
| `mem_copy` | Copy a single memory across envs |
| `mem_move` | Move a single memory across envs (source becomes superseded) |

### Quick examples

```bash
# Export an env to an archive (CLI)
memory-mcp-admin env export --env-id 0123abcd-... --target /backup/prod-export --format archive

# Import into a new env (always dry-run by default; pass --no-dry-run to apply)
memory-mcp-admin env import --source /backup/prod-export.tar.gz --target-env-name staging

# Snapshot before a risky experiment
memory-mcp-admin env snapshot --env-id 0123abcd-... --label "before-experiment-1"

# Restore in place if the experiment goes wrong (preserves all UUIDs, so external lineage refs stay valid)
memory-mcp-admin env restore --snapshot-id <id> --mode replace_env_in_place --confirm
```

Python SDK:
```python
from memory_mcp_client import Client
from memory_mcp_schemas.env_ops import EnvExportRequest

async with Client(...) as client:
    report = await client.env_ops.export(EnvExportRequest(env_id=..., target_path=..., format="archive"))
```

### Key invariants

- **UUIDs are remapped**, never reused, across cross-env operations (export/import, clone, migrate). The destination always gets fresh UUIDs.
- **Lineage edges can cross envs** intentionally — they're the audit trail of cross-env moves and merges. Relations stay strictly env-local.
- **Environments are soft-deleted**, never hard-deleted. The env UUID remains valid forever as a lineage anchor.
- **Embedding model mismatch is explicit** — every cross-env tool requires `re_embed_if_model_mismatch=True` to proceed when models differ, and bulk re-embed (>10k memories) requires `allow_bulk_reembed=True`.
- **Destructive operations** (env_delete, env_restore in-place, env_import overwrite mode) require explicit `confirm_destroy=True`.

### Auth model

`memory-mcp-admin` connects over Streamable HTTP and respects server-side RBAC. Identity comes from (in priority order):
1. `--token` flag
2. `MEMORY_MCP_TOKEN` env var
3. `~/.memory-mcp/config.toml` (`token = "..."`, `endpoint = "..."`)

The CLI never bypasses RBAC.

For full per-tool reference including all flags and request/response models, see `docs/env_ops.md`.

### Friendly env names in `mem_search` (v0.9 wave 1)

`mem_search` accepts `env_names` as a friendly alternative to `env_ids`, resolved server-side case-insensitively. Provide either `env_ids` (UUIDs) or `env_names` (strings), not both. The MCP tool docstring includes the canonical top-level `{"request": {"query": "...", "env_names": ["cdp"]}}` example.

## Stats & Observability (v0.10)

### `mem_stats` MCP tool

A read-only operational snapshot in a single round-trip. Counts (total / active / superseded / retired / pinned, `by_env` with friendly names, `by_kind`, `by_status`, `top_tags`), distributions (chain depth, body length, age, salience, access count, tags-per-memory — with p50/p90/p99), v0.7-table snapshots (tasks / playbooks / decisions by status), per-(sink, env) projection lag, outbox pending/dead, and process RSS (Linux). Substrate counts (`pg_database_size`, Qdrant points, Neo4j nodes/relationships) are opt-in via `include_substrates: true`. RBAC-aware; `global: true` requires admin.

```jsonc
// minimal call
{ "request": { "include_distributions": true } }

// with friendly env filter (v0.9 env_names)
{ "request": { "env_names": ["cdp"], "include_substrates": true } }
```

`total_body_bytes` is `SUM(octet_length(body))` — text content only. Statement-timeout (1500ms) safe: on cancel the response carries `total_body_bytes: null` with `total_body_bytes_approximate: true`. For physical disk usage use `include_substrates: true` and read `substrate.postgres.db_size_bytes`.

### Prometheus + Grafana dashboard

Prometheus metrics are exposed at `/metrics` (already wired in v0.x; v0.10 adds 12 new memory-shape + RSS gauges/histograms with cardinality caps). A Grafana dashboard (`dashboards/memory-mcp.json`) ships with 12 panels in 3 rows: operational (tool QPS/latency), substrate (projection lag, outbox, dream pass, RSS), memory shape (counts by kind, chain-depth heatmap, age heatmap, body-length percentiles, tasks/playbooks/decisions).

Bring up the dashboard with the `observability` compose profile (default footprint unchanged otherwise):

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml --profile observability up
# Grafana: http://localhost:3000   (admin/admin)
# Prometheus: http://localhost:9090
```

Tunables: `MCP_METRICS_REFRESH_INTERVAL_SECONDS` (default 60) controls how often the expensive distribution refresh runs — independent of Prometheus scrape interval.

## Architecture

```
agents ──HTTP/SSE──▶ memory-mcp (FastAPI + mcp SDK)
                          │
              ┌───────────┼─────────────┐
              ▼           ▼             ▼
          Postgres     Qdrant         Neo4j
          (truth +     (vectors)      (graph, Phase 2)
           outbox)
              │
   projection-worker  ──drains outbox──▶ Qdrant + Neo4j
   dream-worker       ──scheduled──────▶ decay / duplicates / promotions  (Phase 2)
```

Postgres is the source of truth; Qdrant and Neo4j are projections kept in sync via an **outbox + per-sink delivery** pattern. Both are rebuildable from Postgres alone (`memory_admin_rebuild_qdrant`, `memory_admin_rebuild_neo4j` — Phase 2 admin tooling).

---

## Quick start (Phase 1)

### 1. Bring up the stack

```bash
cp .env.example .env
docker compose up -d
docker compose ps         # postgres / qdrant / neo4j / server / projection-worker should be healthy
```

The `server` image build pulls torch + sentence-transformers and takes ~5 min on first run. Subsequent `up` calls are cached. Built images are tagged `memory-mcp/<role>:dev` by default.

### 2. Verify health

```bash
curl -s localhost:8080/healthz | jq
# { "local_only": true, "auth": "disabled", "transport": {...}, ... }

curl -s localhost:8080/readyz | jq
# { "status": "ok", "dependencies": { "postgres": {"status":"ok"}, "qdrant": {...} }, ... }

curl -s localhost:8080/metrics | head -20
# Prometheus exposition: mcp_tool_calls_total, mcp_tool_latency_seconds,
# mcp_projection_lag_seconds, mcp_outbox_pending_total, ...
```

> Host-side port defaults to **8080** (`MEMORY_MCP_HOST_PORT` in `.env.example`).
> The container itself always listens on 8080 internally; only the host-side
> mapping is configurable. Override `MEMORY_MCP_HOST_PORT=8090` (or any free
> port) when running two memory-mcp stacks side by side.

### 3. Connect from Copilot CLI

If you installed via `/plugin install outwarped-io/memory-mcp`, you're done —
the plugin manifest already registered the MCP server. For a manual install,
the minimal HTTP config is:

```json
{
  "mcpServers": {
    "memory-mcp": {
      "type": "http",
      "url": "http://localhost:8080/mcp",
      "headers": {
        "X-Agent-Id": "${env:MEMORY_MCP_AGENT_ID}",
        "X-Agent-Name": "${env:MEMORY_MCP_AGENT_NAME}"
      }
    }
  }
}
```

`X-Agent-Id` is **any stable UUIDv4** — pick one per agent install. If omitted, the server uses a default agent persisted to `${LOCAL_DEFAULT_AGENT_FILE}` (writes from different agents will then collide on attribution).

`X-Env-Ids` is an optional comma-separated list of env UUIDs the session should "attach" — currently a UX convenience; security boundary lands with auth in v1.5.

### 4. First write & search

Once connected, the agent can call:

```jsonc
// 1. create an env
env_create_({ "request": { "name": "work", "default_embedding_model_id": "all-MiniLM-L6-v2" } })

// 2. write a memory
mem_write({ "request": { "kind": "fact", "body": "memory-mcp binds to 127.0.0.1 in v1", "env_id": "<uuid>" } })

// 3. search (use consistency=fresh to wait for projection drain)
mem_search({ "request": { "query": "where does the server bind", "env_ids": ["<uuid>"], "consistency": "fresh" } })
```

### 4b. Graph-mode search & entity neighbors (Phase 2.1)

Once entities and relations are written, hybrid/graph search and neighborhood
traversal become available. Both the `mem_search(mode=graph)` leg and
`ent_neighbors` accept `consistency: "fresh"` — the server snapshots the
outbox watermark and waits (up to `SEARCH_FRESH_MAX_WAIT_SECONDS`, default 2s)
for the Neo4j projection to catch up before reading.

```jsonc
// upsert an entity chain A → B → C
ent_upsert({ "request": { "env_id": "<uuid>", "kind": "service", "canonical_name": "ServiceA" } })   // → idA
ent_upsert({ "request": { "env_id": "<uuid>", "kind": "service", "canonical_name": "ServiceB" } })   // → idB
ent_upsert({ "request": { "env_id": "<uuid>", "kind": "service", "canonical_name": "ServiceC" } })   // → idC
rel_link({  "request": { "env_id": "<uuid>", "src": {"kind": "entity", "id": "<idA>"}, "dst": {"kind": "entity", "id": "<idB>"}, "type": "depends_on" } })
rel_link({  "request": { "env_id": "<uuid>", "src": {"kind": "entity", "id": "<idB>"}, "dst": {"kind": "entity", "id": "<idC>"}, "type": "depends_on" } })

// 2-hop neighbors from A — reaches both B and C
ent_neighbors({ "request": { "env_id": "<uuid>", "entity_id": "<idA>", "hops": 2, "kind": "entity", "consistency": "fresh" } })

// graph-only search: entity-mention NER → graph expansion → memory scoring
mem_search({ "request": { "query": "what does ServiceB depend on", "env_ids": ["<uuid>"], "mode": "graph", "consistency": "fresh" } })

// hybrid: lex + sem + graph fused via reciprocal rank
mem_search({ "request": { "query": "what does ServiceB depend on", "env_ids": ["<uuid>"], "mode": "hybrid", "consistency": "fresh" } })
```

The graph leg is best-effort: if spaCy NER returns no entities, the leg
contributes nothing and `mode=hybrid` falls back to lex + sem.

### 5. Run the end-to-end smoke

```bash
docker run --rm --network memory-mcp_default \
  -v "$PWD":/app -w /app \
  -e MEMORY_MCP_BASE_URL=http://memory-mcp-server-1:8080 \
  python:3.12-slim bash -lc \
  'pip install --quiet -e ".[test]" && python .tmp/mcp_transport_smoke.py'
```

Exits 0 on full pass; 1 with a per-case detail dump on any failure.

### 6. Tear down

```bash
docker compose down -v        # ``-v`` removes named volumes; drop if you want to keep data
rm -f .env
```

---

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

## Python client

The async [`memory-mcp-client`](./client/) package wraps the Streamable-HTTP MCP tools with typed namespaces and shared schemas. It is path-installed next to the server while the package remains unpublished.

```python
async with MemoryClient("http://127.0.0.1:8080/mcp") as client:
    envs = await client.envs.list_()
```

---

## Configuration

See `.env.example`. Key vars:

| Variable | Default | Notes |
|---|---|---|
| `POSTGRES_URL` | (set in compose) | `postgresql+asyncpg://...` |
| `QDRANT_URL` | (set in compose) | |
| `NEO4J_URL` | (set in compose) | |
| `EMBEDDER` | `local` | `local` (sentence-transformers) or `azure_openai` |
| `EMBEDDING_MODEL_ID` | `all-MiniLM-L6-v2` | Per-env override stored in `environments.default_embedding_model_id` |
| `VECTOR_BACKEND` | `qdrant` | `qdrant` or `pgvector` |
| `GRAPH_BACKEND` | `neo4j` | `neo4j` or `postgres` (recursive-CTE fallback) |
| `MCP_HTTP_HOST` | `0.0.0.0` (compose) | Bind address inside the server process. Compose pins it so Docker port-publishing works; loopback enforcement happens on the host side via `MCP_HTTP_BIND`. |
| `MCP_HTTP_BIND` | `127.0.0.1` | Compose-only — the host interface that publishes the server port. **DO NOT change in v1 unless you understand the local-only contract.** |
| `MEMORY_MCP_HOST_PORT` | `8080` | Host-side port the server is published on. Override to any free port when running two memory-mcp stacks on the same host. Container always binds 8080 internally. |
| `MEMORY_MCP_IMAGE_TAG` | `dev` | Tag applied by `docker compose build` to the three local images. Override to a version (e.g. `v0.13.0`) when cutting a release image. |
| `LOG_LEVEL` | `INFO` | structlog level |
| `LOCAL_DEFAULT_AGENT_FILE` | `/var/lib/memory-mcp/default-agent.json` | Server-default agent persisted on first run (used when no `X-Agent-Id` is provided) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | (unset) | Reserved; v1.5 |
| `MEMORY_MCP_DECISION_CONFLICT_COSINE_THRESHOLD` | `0.85` | Cosine cutoff for dream-mode decision conflict proposals. |

---

## Repo layout

```
memory-mcp/
├── docker/                    # one Dockerfile per process role
│   ├── Dockerfile.server
│   ├── Dockerfile.projection-worker
│   └── Dockerfile.dream-worker
├── docker-compose.yml         # postgres + qdrant + neo4j + server + projection-worker (+ dream-worker stub)
├── docker-compose.dev.yml     # dev overrides (mounts, hot-reload)
├── migrations/                # alembic
├── src/
│   ├── memory_mcp/            # MCP server
│   │   ├── server.py          # FastAPI lifespan + /healthz, /readyz, /metrics + /mcp mount
│   │   ├── mcp_app.py         # MCP tool registrations (45 tools in v0.7.1)
│   │   ├── tools/             # one module per tool family
│   │   ├── db/                # postgres + outbox + vector + graph backends
│   │   ├── embeddings/        # local + azure_openai
│   │   ├── search/            # lex / sem / hybrid / ranking (graph in Phase 2)
│   │   ├── identity.py        # AgentContext + no-op rbac.require()
│   │   ├── envs.py
│   │   └── observability.py   # structlog + Prometheus + RequestIdMiddleware
│   ├── projection_worker/     # drains outbox to Qdrant (+ Neo4j in Phase 2)
│   └── dream_worker/          # Phase 2 stub
├── tests/
│   ├── unit/                  # 270+ tests
│   ├── integration/           # MCP transport smoke (env-gated)
│   └── invariants/            # lifecycle matrix, schema invariants
└── examples/
    ├── client.py
    ├── connect-copilot-cli.json
    └── connect-claude-desktop.json
```

---

## Tools (v1)

For ready-to-paste agent instructions, see the [system prompt cookbook](./docs/system-prompts.md).

Memory: `mem_write`, `mem_get`, `mem_get_many`, `adr_export` *(v0.7.0)*, `mem_update`, `mem_archive`, `mem_retire`, `mem_supersede`, `mem_journal`, `mem_digest` *(v0.6.0)*, `mem_resume` *(v0.6.0)*, `mem_search`, `mem_auto_context` *(v0.6.0)*, `mem_neighbors` *(Sprint B)*, `mem_related` *(Sprint B)*, `mem_lineage` *(Sprint B)*, `mem_sources_browse` *(Sprint B)*, `mem_browse` *(Sprint A)*, `mem_facets` *(Sprint A)*, `mem_context_pack` *(v0.7.0 F7 v2)*, `playbook_invoke` *(v0.7.0)*.
Entities: `ent_upsert`, `ent_resolve`, `ent_merge`, `ent_neighbors` *(Phase 2.1)*, `ent_browse` *(Sprint A)*.
Relations: `rel_link`, `rel_browse` *(Sprint A)*.
Environments: `env_create_`, `env_list_`, `env_get_`, `env_attach_`, `env_detach_`.
Tasks *(v0.7.0)*: `task_create`, `task_substep`, `task_dep_link`, `task_status_set`, `task_list`, `task_next`, `task_tree`, `task_link_memory`.
Dream *(Phase 2.2)*: `dream_run_`, `dream_status_`, `dream_proposals_list_`, `dream_review_`.

### Search modes

`mem_search` supports:

- `lex` — Postgres full-text search for exact terms, IDs, and error codes.
- `sem` — Qdrant vector search for paraphrases and conceptual similarity.
- `graph` *(Phase 2.1)* — Neo4j entity expansion for entity-anchored questions.
- `hybrid` *(default in v0.6.0)* — reciprocal-rank fusion across lexical, semantic, and graph legs.
- `id` — direct Postgres lookup by partial UUID.
- `auto` — opt-in v0.6.0 resolver: UUID-prefix-shaped queries dispatch to `id`; all other non-empty queries dispatch to `hybrid`.

### Stdio transport

Set `MCP_TRANSPORT=stdio` to run the same MCP server over standard input/output instead of Streamable HTTP; backing services are still required. The Copilot-CLI plugin manifest uses this transport mode internally (via `docker compose exec`), so plugin users do not need to configure stdio manually.

Most tools accept a single `request` Pydantic-model argument; Active Recall tools (`mem_digest`, `mem_resume`, `mem_auto_context`, `mem_context_pack`), `playbook_invoke`, `task_next`, `task_tree`, and `adr_export` use direct parameters as registered in `mcp_app.py`. Full schemas are returned by MCP `list_tools`.

### Importing from `@modelcontextprotocol/server-memory`

Use `scripts/import_from_server_memory.py` to migrate an upstream JSONL file into a target env. Dry-run first:

```bash
PYTHONPATH=src python -m scripts.import_from_server_memory \
  --input ~/.config/Claude/memory.jsonl \
  --base-url http://127.0.0.1:8080/mcp \
  --env-id <UUID> \
  --dry-run
```

The importer upserts entities, writes observations with `source_type="import"`, links relations, and skips exact observation duplicates on rerun.

### Browse & explore (Sprint A)

Four read-only tools for open-ended discovery (no relevance ranking — deterministic listing + facets).

* `mem_browse({env_ids?, kinds?, tags?, statuses?, created_after?, updated_after?, order_by?, descending?, limit?, cursor?})`
  Keyset-paginated listing of memories. Filter parity with `mem_search` (same field names, same **OR** semantics for `tags` — a memory matches when *any* listed tag is present). Default visibility is `[proposed, active]`; opt into `stale|archived|superseded|retired` via `statuses`. `order_by` is `updated_at` (default) or `created_at`.

* `mem_facets({env_ids?, facets?, tag_limit?, statuses?, max_rows?})`
  Distinct-value + count aggregation. Default facets `["kind","status","tag"]`; `"month"` is opt-in. When the filtered population exceeds `max_rows` (default 100_000) the response returns totals only with `approximate=true` and an empty `facets` map — caller is expected to narrow filters (time window / kinds) and retry. Per-facet GROUP BYs run under `facet_query_timeout_seconds` (default 2s); on timeout the response sets `approximate=true` with whatever facets completed.

* `ent_browse({env_ids?, kinds?, name_prefix?, order_by?, descending?, limit?, cursor?})`
  Keyset-paginated entity listing. `name_prefix` is case/punctuation-normalized and matches against either the entity's normalized canonical name or any normalized alias (LIKE `prefix%` backed by `text_pattern_ops` indexes; `_` and `%` in the prefix are escaped).

* `rel_browse({env_ids?, types?, src_kind?, dst_kind?, src_id?, dst_id?, created_after?, descending?, limit?, cursor?})`
  Keyset-paginated relation listing. `types` capped at 20 distinct values. `src_id` / `dst_id` pin endpoints to a specific canonical record id (entity.id or memory.id — NOT graph_node id); pinning an id **requires** the matching `src_kind` / `dst_kind` so the lookup is unambiguous.

Cursors are opaque base64-encoded JSON bound to the request's filter fingerprint; changing a filter mid-pagination raises `[INVALID_CURSOR]`. See `migrations/versions/0004_explore_api_sprint_a.py` for the supporting indexes.


### Graph & provenance (Sprint B)

Four read-only tools for memory-rooted traversal, similarity, lineage, and source inspection.

* `mem_neighbors({memory_id, hops?, edge_types?, direction?, kind?, limit?, cursor?, env_id?, consistency?})`
  Starts from a memory and returns neighbor entities (`linked_to` edges) plus neighbor memories one graph hop away. Keyset cursor orders by distance / created_at; RBAC is env-scoped.

* `mem_related({memory_id, relation?, limit?, cursor?, env_id?})`
  Finds related memories. `relation="shared_entity"` ranks by linked-entity overlap count; `relation="semantic"` uses Qdrant cosine similarity from the stored embedding via `VectorStore.get_vector` (never re-embeds) and returns `note="no_embedding"` or `note="vector_store_unavailable"` on degraded paths. Cursor shape is mode-specific; RBAC is env-scoped.

* `mem_lineage({memory_id, direction?, relations?, max_depth?, env_id?})`
  Walks parent / child rows in `memory_lineage` (for example `supersedes`, `derived_from`) and returns ancestors / descendants up to `max_depth`. Uses a Postgres 14+ recursive CTE with native `CYCLE` safety. Lineage is forensic: it bypasses default visibility so superseded / archived nodes appear with status badges.

* `mem_sources_browse({env_ids?, source_type?, agent_id?, memory_id?, created_after?, created_before?, hydrate_memories?, limit?, cursor?})`
  Keyset-paginates `MemorySource` rows by source type, agent, memory, or time window. `hydrate_memories=true` returns linked memories and respects default visibility (browse semantics, not forensic).


---

## License

MIT.
