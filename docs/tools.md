# Tools reference

Full MCP tool surface for memory-mcp v0.17. Tools are surfaced over MCP Streamable HTTP (`http://127.0.0.1:8080/mcp/`) and stdio.

## Tools (v1)

For ready-to-paste agent instructions, see the [system prompt cookbook](./docs/system-prompts.md).

Memory: `mem_write`, `mem_get`, `mem_get_many`, `adr_export` *(v0.7.0)*, `mem_update`, `mem_archive`, `mem_retire`, `mem_supersede`, `mem_compose` *(v0.15.0)*, `mem_decompose` *(v0.15.0)*, `mem_journal`, `mem_digest` *(v0.6.0)*, `mem_resume` *(v0.6.0)*, `mem_search`, `mem_auto_context` *(v0.6.0)*, `mem_neighbors` *(Sprint B)*, `mem_related` *(Sprint B)*, `mem_lineage` *(Sprint B)*, `mem_sources_browse` *(Sprint B)*, `mem_browse` *(Sprint A)*, `mem_facets` *(Sprint A)*, `mem_context_pack` *(v0.7.0 F7 v2)*, `playbook_invoke` *(v0.7.0)*.
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

