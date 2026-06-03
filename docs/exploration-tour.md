# Memory-MCP — Exploration Tour

You've just been pointed at a memory-mcp environment you've never used before. This guide shows the call sequence to **discover what's in it, navigate the graph, and find your bearings** with no prior knowledge of the contents.

The tour exercises the eight exploration tools shipped through v0.10:

| Tool | Purpose | When to reach for it |
|---|---|---|
| [`env_list_`](#1-env_list_) | What envs exist? | Always. First call when bootstrapping. |
| [`mem_facets`](#2-mem_facets) | What kinds / tags / sources are in this env? | Cold-start sniffing. Cheap. |
| [`mem_browse`](#3-mem_browse) | Empty-query, paginated row listing | Browsing by recency or by filter without a search term. |
| [`mem_lineage`](#4-mem_lineage) | "What was this derived from / what derived from this?" | Tracing supersession, summarization, promotion across substrates. |
| [`mem_sources_browse`](#5-mem_sources_browse) | Provenance — where did this memory come from? | Auditing or reconstructing the canonical source. |
| [`mem_neighbors`](#6-mem_neighbors) | Memory-rooted graph walk | "What's structurally related to this row?" |
| [`mem_related`](#7-mem_related) | Semantic similarity by stored embedding | "More like this one." Cheap (no fresh embed). |
| [`ent_browse`](#8-ent_browse) / [`rel_browse`](#9-rel_browse) | List entities / list relations | When you want to inspect the entity/graph layer directly. |

Every tool below is also exposed in the Python SDK (`memory_mcp_client`) — method signatures shown alongside the raw MCP call.

---

## Setup

```python
from memory_mcp_client import MemoryClient

client = MemoryClient(
    endpoint="http://127.0.0.1:8080/mcp/",
    attached_env_names=["scratch"],  # or attached_env_ids=[UUID(...)]
)
```

All exploration tools accept `env_ids` (or `env_names` on v0.9+); calls fall back to the client's attached envs if you omit them.

## The tour

### 1. `env_list_`

What environments exist on this server?

```python
envs = await client.envs.list_()
for e in envs:
    print(e.id, e.name, e.kind, e.status)
```

→ `[{id: ..., name: 'project-a', ...}, {id: ..., name: 'scratch', ...}, {id: ..., name: 'personal', ...}]`

Pick the env you care about. Everything below scopes to that env.

### 2. `mem_facets`

Before browsing rows, ask **what's in this env** in aggregate.

```python
facets = await client.memories.facets(
    env_names=["scratch"],
    dimensions=["kind", "lifecycle", "tag", "source"],
)
print(facets.kind)       # {"fact": 412, "decision": 9, "procedure": 14, ...}
print(facets.lifecycle)  # {"active": 380, "stale": 21, "retired": 7, ...}
print(facets.tag)        # {"task:26q2-260513-...": 17, "topic:docker": 9, ...}
```

This is cheap (Postgres counts only, no Qdrant) and unblocks every "what's this env about?" probe. **Low-count buckets are suppressed in multi-tenant envs** to prevent fingerprinting.

### 3. `mem_browse`

Now look at actual rows. `mem_browse` paginates by `(created_at | updated_at, id)` keyset cursors:

```python
page = await client.memories.browse(
    env_names=["scratch"],
    filter={"kinds": ["fact"], "tags_any": ["topic:docker"]},
    order_by="created_at",
    limit=20,
)
for row in page.rows:
    print(row.id, row.title, row.tags)

# Next page:
next_page = await client.memories.browse(
    env_names=["scratch"],
    filter=page.filter,  # same filter, or pass cursor only
    cursor=page.next_cursor,
)
```

Default visibility matches `mem_search`: proposed + active only. Opt-in via `statuses=["retired", "superseded", ...]` when you really want them.

### 4. `mem_lineage`

You found a memory and want to know what it was derived from (or what built on it):

```python
lineage = await client.memories.lineage(
    memory_id=row.id,
    direction="both",   # "upstream" | "downstream" | "both"
    depth=3,
)
for edge in lineage.edges:
    print(edge.relation, edge.src_id, "→", edge.dst_id)
```

Walks the `memory_lineage` table — covers `supersedes`, `summarized_from`, `promoted_from`, `copied_from`, `replaced_by` (cross-env). Cheap; pure Postgres.

### 5. `mem_sources_browse`

Where did a memory come from? (Was it written by hand? Imported? Promoted from a journal entry?)

```python
sources = await client.memories.sources_browse(
    memory_id=row.id,
)
for src in sources.rows:
    print(src.kind, src.uri, src.recorded_at)
```

→ `[{kind: "learnings_md", uri: "tasks/.../learnings.md#some-heading", recorded_at: ...}, ...]`

This is the provenance back-pointer that the bootstrap script also embeds in every memory body. `mem_sources_browse` queries the structured `memory_sources` table.

### 6. `mem_neighbors`

The graph view — what's structurally adjacent to this memory?

```python
neighbors = await client.memories.neighbors(
    memory_id=row.id,
    direction="both",
    relation_types=["evidence_for", "derives_from"],
    depth=2,
)
for n in neighbors.rows:
    print(n.relation, n.peer_id, n.peer.title)
```

Reuses the same `GraphStore` walker `ent_neighbors` is built on. Pass `consistency="fresh"` if you need the latest projection writes; default is `default` (canonical-when-not-degraded).

### 7. `mem_related`

"More like this one" using the memory's stored embedding — no fresh embed, no LLM round trip:

```python
similar = await client.memories.related(
    memory_id=row.id,
    limit=10,
    env_names=["scratch"],  # optional; cross-env semantic search
)
for s in similar.rows:
    print(f"{s.score:.3f}  {s.title}")
```

Useful follow-on to `mem_lineage` when the lineage trail goes cold — semantic similarity often surfaces sibling thinking that wasn't explicitly linked.

### 8. `ent_browse`

If the env has entities (people, repos, services, projects), list them:

```python
ents = await client.entities.browse(
    env_names=["scratch"],
    kinds=["repo", "service"],
    limit=20,
)
for e in ents.rows:
    print(e.kind, e.canonical_name, e.aliases)
```

Pair with `ent_neighbors(entity_id=...)` to walk the entity graph.

### 9. `rel_browse`

List relations directly. Useful for "who points at this entity?" or "what `evidence_for` edges exist in this env?":

```python
rels = await client.relations.browse(
    env_names=["scratch"],
    relation_types=["evidence_for"],
    limit=50,
)
for r in rels.rows:
    print(r.src_id, "—", r.type, "→", r.dst_id)
```

---

## Suggested onboarding flow

A first-time-in-this-env agent flow that takes ~5 calls:

1. `env_list_` → pick env.
2. `mem_facets(dimensions=["kind", "lifecycle", "tag", "source"])` → understand shape.
3. `mem_browse(order_by="updated_at", limit=20)` → see the freshest 20 memories.
4. Pick a few interesting rows. For each:
   - `mem_lineage(direction="both", depth=2)` to see derivation history.
   - `mem_related(limit=5)` to surface sibling thinking.
5. If entities are in play: `ent_browse(limit=20)` → spot the canonical names; `ent_neighbors` from any node of interest.

This is the same pattern the [`mem_auto_context`](https://github.com/memory-mcp) and [`mem_context_pack`](https://github.com/memory-mcp) helpers automate — when you need a higher-level prebuilt context blob, prefer those.

## See also

- [`docs/system-prompts.md`](system-prompts.md) — best practices for system prompts that lean on this surface.
- [`docs/env_ops.md`](env_ops.md) — environment-level operations (copy/move/migrate/snapshot/export/import).
