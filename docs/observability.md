# Observability

## Stats & Observability (v0.10)

### `mem_stats` MCP tool

A read-only operational snapshot in a single round-trip. Counts (total / active / superseded / retired / pinned, `by_env` with friendly names, `by_kind`, `by_status`, `top_tags`), distributions (chain depth, body length, age, salience, access count, tags-per-memory — with p50/p90/p99), v0.7-table snapshots (tasks / playbooks / decisions by status), per-(sink, env) projection lag, outbox pending/dead, and process RSS (Linux). Substrate counts (`pg_database_size`, Qdrant points, Neo4j nodes/relationships) are opt-in via `include_substrates: true`. RBAC-aware; `global: true` requires admin.

```jsonc
// minimal call
{ "request": { "include_distributions": true } }

// with friendly env filter (v0.9 env_names)
{ "request": { "env_names": ["project-a"], "include_substrates": true } }
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

