# memory-mcp

A shared, multi-agent **Memory MCP server** exposing 65 MCP tools. Stores facts, procedures, playbooks, tasks, events, decisions, preferences, observations, and snippets across sessions and across agents. Backed by **Postgres (truth) + Qdrant (vectors) + Neo4j (graph)**, exposed over **MCP Streamable HTTP** or **stdio**.

> **Status:** v0.17 — core CRUD, journal, entities, relations, search (lex + sem + hybrid), active recall, procedures/plans, environment operations, hard-delete cascade, compose/decompose with auto-wire, **inbox/drop-box for inter-agent messaging**, and a full stats surface (`mem_stats` MCP tool + Prometheus expansion + Grafana dashboard). Backed by outbox-driven vector/graph projections and the dream worker.

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
cd ~/.copilot/installed-plugins/_direct/outwarped-io--memory-mcp
cp .env.example .env                            # edit secrets if needed
docker compose up -d
```

That's it. The plugin manifest registers a **Streamable HTTP** MCP server
at `http://127.0.0.1:8080/mcp`; Copilot CLI connects on demand once the
stack is up.

To uninstall: `/plugin disable memory-mcp` → `docker compose down -v` →
`/plugin uninstall memory-mcp`.

### Non-default port

The plugin manifest's `url` is fixed at `http://127.0.0.1:8080/mcp`. If you
override `MEMORY_MCP_HOST_PORT` in `.env` (for example, to run two
memory-mcp stacks on the same host), edit the installed manifest at
`~/.copilot/installed-plugins/_direct/outwarped-io--memory-mcp/.claude-plugin/plugin.json`
to match. Copilot CLI does not env-expand the `url` field.

### Install without the Copilot CLI plugin command

If you'd rather skip `/plugin install`, the manual path is:

```bash
git clone https://github.com/outwarped-io/memory-mcp.git
cd memory-mcp
cp .env.example .env
docker compose up -d
```

Then drop this snippet into your MCP client config (`~/.copilot/settings.json`
for Copilot CLI, or the equivalent for your client):

```json
{
  "mcpServers": {
    "memory-mcp": { "type": "http", "url": "http://127.0.0.1:8080/mcp" }
  }
}
```

This is the same connection target the plugin manifest registers — the
`/plugin install` path and the manual path are now equivalent.

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

---

## Documentation

Deep-dive topics live under [`docs/`](docs/):

- **[Tools reference](docs/tools.md)** — full MCP tool surface (65 tools across CRUD, search, entities, relations, chain-of-thought, plans, decisions, dream worker, compose/decompose, inbox, env ops).
- **[Features](docs/features.md)** — Compose / Decompose / Inbox (drop-box) walkthroughs.
- **[Environment operations](docs/env_ops.md)** — env export / import / snapshot / restore.
- **[Operator runbook](docs/operations.md)** — backup, restore, lifecycle, troubleshooting.
- **[Configuration](docs/configuration.md)** — environment variables.
- **[Observability](docs/observability.md)** — Prometheus / Grafana stack.
- **[Python client](docs/python-client.md)** — packaged sync/async wrapper.
- **[Repo layout](docs/repo-layout.md)** — directory map.
- **[Exploration tour](docs/exploration-tour.md)** — guided walk-through.
- **[System prompts](docs/system-prompts.md)** — recommended agent prompts.

---

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

## License

MIT.
