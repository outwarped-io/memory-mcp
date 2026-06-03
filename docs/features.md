# Features — Compose, Decompose, Inbox

Three composable primitives on top of the core memory store.

## Compose (v0.15.0 — Phase 2)

`mem_compose` is a **caller-driven N→1 aggregation** tool: pick 2–20 sibling memories in one env, fold them into a single new memory, and write a lineage trail linking the new row back to its sources. It pairs with the dream-worker's automatic dedup pipeline — that pipeline proposes; `mem_compose` lets agents and humans **act directly** with no proposal envelope.

### Two modes

| Mode | Source state after | Lineage relation | Tag-policy default |
|---|---|---|---|
| `promote` *(default — non-destructive)* | sources stay `active` | `promoted_from` | `target` (target tags only) |
| `merge` *(destructive)* | sources go to `superseded` | `supersedes` | `target_plus_union` (target ∪ union of source tags) |

`promote` is the safe default — sources remain queryable; the new memory points back at them. `merge` is irreversible (sources are tombstoned in the outbox so Qdrant drops their vectors); use it when sources are genuine duplicates or pre-merge drafts.

### Quick example

```python
from memory_mcp_schemas.compose import MemComposeRequest, MemComposeTarget
from memory_mcp.db.types import MemoryKind

resp = await client.compose(MemComposeRequest(
    source_ids=[src1_id, src2_id, src3_id],
    target=MemComposeTarget(
        kind=MemoryKind.fact,
        title="What we learned about Cosmos throttling",
        body="Three observations from the throttling investigation …",
        tags=["topic:cosmos", "component:management"],
    ),
    mode="promote",
))

print(resp.memory.id)              # UUID of the new memory
print(resp.idempotency_replay)     # False on first call; True on replay
print(resp.tag_policy_applied)     # "target"
```

### Idempotency contract

`mem_compose` derives a deterministic **dedupe key** from `{schema_version, operation, env_id, mode, sorted(source_ids), target_kind, sha256(title+body), sorted(target_tags)}`. The key is stored in a partial-unique index on `memories(env_id, compose_dedupe_key)`. A second call with the same envelope returns the original memory with `idempotency_replay=True` — **no new rows, no audit entries, no popularity bumps**.

- Callers can override the derived key with an explicit `idempotency_key` (≤128 chars) — useful when the same logical compose must dedupe across different source orderings.
- Changing **any** request field (mode, source set, target body, tags, kind) changes the dedupe key. Reusing a caller-supplied `idempotency_key` with different sources or a different mode raises `InvalidInputError` rather than silently echoing a stale response.

### Lineage traversal

Walk the lineage with `mem_lineage(memory_id=resp.memory.id, direction='back')`. The graph is Postgres-only — Neo4j does not project lineage edges in v1. For an entity-anchored view of the merged memory, use `mem_neighbors`.

### Popularity caveat

The merged memory **starts at `reference_count=0`**. Compose does *not* transfer the sources' inbound citations (`rel_link` rows, embedded `{{memory:<uuid>}}` references in playbooks, task citations). Sources retain their full popularity profile; the new memory accrues its own from this point forward. Lineage edges are intentionally excluded from `reference_count_*` accounting — they describe how the memory got here, not who depends on it.

Citation rewrite / lazy resolution is **deferred to v1.5**. If the merged memory needs to inherit citations, the recommended pattern is to `mem_archive` or `mem_retire` the sources after compose so consumers naturally migrate.

### Audit shape

Each compose call writes:

- On the merged memory: one `op='create'` row + one `op='mem_compose:{mode}'` row.
- On each source (only when `mode='merge'`): one `op='supersede'` row.
- No audit rows for `mode='promote'` sources (they stay active and untouched in the body).

Outbox shape mirrors the audit:

- `mode='promote'` → 1 `upsert` for the merged memory; 0 events for sources.
- `mode='merge'` → 1 `upsert` for the merged memory; N `tombstone` events for sources (one per source).
- Lineage rows never produce outbox events — they're Postgres-local.

### Validation contract

- **2 ≤ source_ids ≤ 20** (schema-enforced; duplicates rejected).
- All sources must live in the **same env** (cross-env compose is rejected with `InvalidInputError`).
- Sources must be **visible to the caller's attached envs** (raw-UUID access from outside the attached set raises `NotFoundError`).
- Sources must be in `active` or `stale` status — `retired` / `superseded` / `proposed` / `archived` sources raise `InvalidTransitionError`.
- `mode='merge'` requires **all sources to share `kind`**, and the target's `kind` must match (`InvalidInputError` otherwise).
- `mode='promote'` allows mixed-kind sources — the target's `kind` is free.
- `expected_versions` (optional) is a per-source optimistic-lock check; mismatch raises `VersionConflictError`.

See `tests/integration/test_compose_transaction.py` for the full behavioral matrix (25 cases — smoke, accounting, validation, tag-policy, race).

### Auto-wire (Phase 4, OFF by default)

When `autowire_enabled=True`, every successful `mem_compose` call **may** insert up to K `related_to_popular` edges from the newly-composed memory to the most-relevant popular neighbors in the same env. The pass runs in two stages: a read-only **Stage A** (top-by-salience PG fetch → lineage-ancestor exclusion → off-thread body embed → Qdrant similarity → combined `salience × sim` ranking) executes before the compose transaction opens; a small in-transaction **Stage B** inserts the edges with `ON CONFLICT (src_node_id, dst_node_id, type) DO NOTHING`, emits one audit row per edge, and enqueues the projection outbox events. The new edges appear in `auto_wired: list[UUID]` on `MemComposeResponse`.

Knobs (all under `Settings`):

| Knob | Default | Range | Purpose |
|---|---|---|---|
| `autowire_enabled` | `False` | bool | Master switch. OFF in v0.15.0. |
| `autowire_top_k` | `3` | `1..10` | Max edges emitted per compose. |
| `autowire_sim_threshold` | `0.70` | `0.0..1.0` | Provisional — calibration pending. |
| `autowire_candidate_limit` | `20` | `>= top_k`, `1..200` | Postgres pre-pull size for top-by-salience. |

Important semantics:

- **`related_to_popular` is excluded from popularity-counter triggers** (migrations 0017 + 0021). Adding the edge does NOT bump the dst's `reference_count_rel_link`. The same predicate is excluded from `mem_top` velocity windows + the dream recount pass. Auto-wire is a navigation aid, never a popularity vote.
- **Skip filter** — auto-wire skips memories with `kind=playbook`, any tag starting with `directive:active`, or empty/whitespace body. Applied both pre-compute (Stage A) and defensively at insert (Stage B).
- **Replay is state-current, not operation-exact** — a second identical compose call replays via dedupe-key and reconstructs `auto_wired` from the live `relations` table. If a `rel_link` call between the same nodes manually added a `related_to_popular` edge later, replay will surface it too.
- **All failures degrade silently** — embedder failure, vector-store failure, graph-node resolution failure, insert race: the compose call still succeeds; auto-wire returns `[]`.
- **Decompose auto-wire shipped in v0.16** via the additive `auto_wired_by_child: dict[UUID, list[UUID]] | None` field on `MemDecomposeResponse` and the per-tool `autowire_decompose_enabled` knob. See `## Decompose ### Auto-wire (v0.16, OFF by default)` below.

See `tests/integration/test_autowire_compose.py` for the 6 end-to-end cases (OFF regression, Stage B direct insert, ON CONFLICT semantics, replay reconstruction, compose-hook ON path, replay returns state-current).

## Decompose (v0.15.0 — Phase 3)

`mem_decompose` is the **caller-driven 1→N counterpart** to `mem_compose`: pick one source memory in one env, fan it out into 2–20 children, and write a lineage trail linking each child back to the source. It's the tool for splitting a long observation into atomic facts, deriving sub-procedures from a runbook, or breaking a decision into the smaller decisions it implies.

### Two modes

| Mode | Source state after | Lineage relation | Popularity bump |
|---|---|---|---|
| `derive` *(default — non-destructive)* | source stays `active` | `derived_from` | source's `reference_count_lineage` += N (whitelisted) |
| `split` *(destructive)* | source goes to `retired` (`version += 1`) | `split_from` | source's `reference_count_lineage` unchanged (whitelist **excludes** `split_from`) |

`derive` is the safe default — the source remains queryable and the agent can keep citing it. `split` is irreversible (source is tombstoned in the outbox so Qdrant drops its vector); use it when the source genuinely *becomes* its children — you don't want both queryable side-by-side.

The whitelist asymmetry is deliberate: with `derive`, the source is the conceptual originator of N atomic derivatives and the popularity bump reflects that. With `split`, the source is being retired — bumping a retired memory's counter is purely forensic, pollutes analytics on accidental reactivation, and creates a feedback loop with `mem_top` if the source ever returns to `active`. See `tests/integration/test_decompose_transaction.py::test_split_lineage_does_not_bump_reference_count_lineage` for the regression test.

### Quick example

```python
from memory_mcp_schemas.decompose import MemDecomposeChild, MemDecomposeRequest
from memory_mcp.db.types import MemoryKind

resp = await client.decompose(MemDecomposeRequest(
    source_id=runbook_id,
    children=[
        MemDecomposeChild(kind=MemoryKind.procedure, title="Detect", body="Watch for 429s in …"),
        MemDecomposeChild(kind=MemoryKind.procedure, title="Mitigate", body="Failover to …"),
        MemDecomposeChild(kind=MemoryKind.procedure, title="Recover", body="Replay events …"),
    ],
    mode="derive",
))

print([c.id for c in resp.children])    # 3 new memory UUIDs
print(resp.source.status)               # 'active' — derive doesn't retire
print(resp.operation_id)                # UUID of the decompose_operations row
print(resp.idempotency_replay)          # False on first call; True on replay
```

### Idempotency contract

Decompose uses a dedicated `decompose_operations` table (not a column on the source) so a single source can be decomposed multiple times without collisions. The dedupe key is derived from `{schema_version, operation, env_id, mode, source_id, sorted(canonical_json(children))}` and stored alongside a stricter **request fingerprint** that includes every field — including `expected_version`, per-child `trigger_description`, per-child `expires_at`.

The unique index `(env_id, dedupe_key)` is the race winner-loser arbiter. Concurrent identical calls serialize via the source's `FOR UPDATE` lock — the loser sees the persisted operation row, validates the request fingerprint matches, and returns the replay path with `idempotency_replay=True`.

- Callers can override the derived key with an explicit `idempotency_key` (≤128 chars). The fingerprint is *always* canonical, so reusing a caller-supplied key with a different source / mode / children-set raises `InvalidInputError("idempotency_key reused with different scope")` rather than silently echoing a stale response.
- Replay returns the *original* children's UUIDs even if any child has since been `mem_update`d or retired. Idempotency is about the transactional outcome (which child ids were created), not the living content.

### Lineage traversal

Walk forward from the source with `mem_lineage(memory_id=source_id, direction='forward')` to see all children. Walk back from a child with `mem_lineage(memory_id=child_id, direction='back')` to find the source. The lineage graph is Postgres-only — Neo4j does not project lineage edges in v1.

### Provenance

Each child carries a `MemorySource` row with `source_type='agent'` and `source_ref=str(operation_id)`. The `MemorySourceType` enum does not currently include a `mem_decompose` value, so the operation-id is used as the back-pointer. To surface "this memory came from decompose op X" in a UI, query `decompose_operations` by id directly.

### Audit shape

Each decompose call writes:

- On each child: one `op='create'` row with `extra_after={decompose_mode, decompose_source, decompose_operation_id}`.
- On the source: one `op='mem_decompose:{mode}'` aggregate row with `extra_after={child_ids, dedupe_key, operation_id, decompose_mode}`.
- On the source (only when `mode='split'`): an additional `op='retire'` row.

Outbox shape:

- `mode='derive'` → N `upsert` events (one per child); 0 events for the source.
- `mode='split'` → N `upsert` events for children + 1 `tombstone` for the source.
- Lineage rows and the `decompose_operations` row never produce outbox events — they're Postgres-local.

### Validation contract

- **2 ≤ children ≤ 20** (schema-enforced).
- Each child's content must be **unique** by canonical-JSON hash (`kind + title + body + tags + metadata + decision_meta + confidence + salience + pinned`); duplicates raise `InvalidInputError`.
- **`kind=playbook` rejected** per child (playbook needs a `steps` field that `MemDecomposeChild` does not expose). Schema-layer 422.
- `decision_meta` valid **only** on `kind=decision` children; the deep validation (against env policy via `validate_decision_meta`) runs once the session is open.
- **Mixed-kind children are allowed** in either mode (D.5 confirmed) — decompose is heterogeneous by nature.
- Source must exist, be **visible to the caller's attached envs** (raw-UUID access from outside raises `NotFoundError`), and be in `active` or `stale` status on first write (replay survives later retirement per the dedupe-before-state-validation rule).
- Source must **not** be `kind=playbook` (playbook sources carry `steps` that children can't represent).
- `expected_version` (optional) is an optimistic-lock check on the source; mismatch raises `VersionConflictError`.

See `tests/integration/test_decompose_transaction.py` for the full behavioral matrix (21 cases — smoke, validation, RBAC, race, whitelist, audit).

### Auto-wire (v0.16, OFF by default)

When **both** `autowire_enabled=True` **and** `autowire_decompose_enabled=True`, each successful `mem_decompose` call **may** wire up to `autowire_decompose_per_child_top_k` `related_to_popular` edges **per child** to the most-relevant popular neighbors in the same env. The decompose auto-wire flow mirrors the v0.15.0 compose auto-wire but runs **per child**: one shared PG candidate pull + one shared lineage-ancestor CTE seeded with `[source_id]`, **one batched embedder call** for all N children's bodies, **N parallel Qdrant searches** via `asyncio.gather`, per-child top-K + a global total-cap downsample, then per-child Stage B inserts wrapped in savepoints so one child's failure does not leak partial side effects to a sibling.

The per-child mapping lives in a new additive response field:

```python
class MemDecomposeResponse:
    auto_wired: list[UUID]                              # flat union (deduped, ordered by child insertion + per-child order)
    auto_wired_by_child: dict[UUID, list[UUID]] | None  # v0.16+ per-child mapping
```

Knobs (under `Settings`):

| Knob | Default | Range | Purpose |
|---|---|---|---|
| `autowire_decompose_enabled` | `False` | bool | Per-tool switch. **Requires `autowire_enabled=True`** (cross-knob invariant). |
| `autowire_decompose_per_child_top_k` | `3` | `1..10` (`<= autowire_candidate_limit`) | Max edges emitted per child. |
| `autowire_decompose_total_cap` | `30` | `1..100` (`>= per_child_top_k`) | Global ceiling. When `sum(per-child results) > total_cap`, results are flattened by `combined_score`, sorted desc, top-N kept, regrouped per-child. |

Important semantics:

- **Three-state `auto_wired_by_child`**:
  - `None` → feature OFF on **first write** (master or per-decompose switch disabled). **Never** returned on replay — replay always reflects current state.
  - `{child_id: []}` for every child → feature was ON but no edges resulted (empty candidates, Stage-A failure, or Stage-B savepoint rollback). Also the shape returned on replay of an operation originally written with the feature OFF.
  - Populated mapping → wired edges. Each child id maps to its `list[UUID]` of dst memory ids.
- **State-current replay** — `mem_decompose` replays via the `decompose_operations` dedupe key; the auto-wired edge set is reconstructed from the live `relations` table via `reconstruct_auto_wired_by_child`. Same caveat as compose: a manual `rel_link(type='related_to_popular')` between a child and another memory after the original decompose call will surface in replay.
- **Sibling exclusion is NOT performed** — children's candidate pool is queried pre-txn from the existing memory set, so just-inserted siblings cannot appear in their pool. A separate auto-wire edge between siblings, if it ever shows up, is a separate operation (state-current replay would surface it).
- **Per-child skip rules apply independently** — kind=playbook, tag `directive:active`, empty/whitespace body cause that one child's slot to be `[]`. The other children proceed normally.
- **Lineage exclusion is shared** — the recursive ancestor CTE rooted at `[source_id]` is computed once and applied to every child's candidate set. Ancestors of the source are not auto-wire-able for any child.
- **Stage-A failure → batch-empty, but feature-ON shape** — embedder or vector-store outage causes all children to receive `[]` (not `None`). The decompose call still commits.
- **Stage-B failure is per-child** — each child's relation INSERTs run inside `async with s.begin_nested()`. One child raising rolls back only that child's relations; the sibling's edges commit cleanly. The failing child gets `[]`.
- **Flat `auto_wired` is ordered-unique** — iterate children in insertion order, then each child's dst list, adding unseen UUIDs only. Duplicates across children (two children → same popular dst) appear exactly once in the flat list.
- **`related_to_popular` is still excluded from popularity counters** (migrations 0017 + 0021). Per-child fan-out does not multiply the popularity-counter feedback loop because the predicate is excluded from the trigger whitelist.
- **Outbox ordering** — for any child that produced wired edges, the child-memory `upsert` outbox row has a strictly smaller `event_id` than its relation outbox rows (`Outbox.event_id` is a monotonic BigInteger PK).

See `tests/integration/test_autowire_decompose.py` for the 8 end-to-end cases (OFF baseline, per-decompose-off, happy path, ordered-unique flat list, replay-state-current, replay-of-off, per-child Stage-B failure isolation, outbox ordering).

## Inbox / Drop-Box (v0.17.0 — Phase 5)

Three new tools turn memory-mcp into a **user-orchestrated** inter-agent message-passing substrate. Agents don't subscribe or listen autonomously — the user copy-pastes a short reference between agents to route messages. The reference is the central UX primitive.

### Reference format

Every channel has a stable copy-pasteable URL:

```
mem-inbox://<env-name>/<channel-slug>
```

Example: `mem-inbox://personal/quiet-otter`. The slug is the channel entity's `canonical_name` (kebab-case, ≤64 chars). Auto-generated slugs use a curated `<adjective>-<noun>` wordlist for pronounceability when the caller omits `name`. References are **server-formatted**; clients pass response strings back verbatim.

### Three user-orchestrated flows

**UC1 — recipient opens an inbox.** User asks Agent A to *"open an inbox"*. Agent A calls `mem_inbox_open(env_name="personal")` → response carries `reference="mem-inbox://personal/quiet-otter"`. User copies the reference to Agent B with *"send me notes here"*. Agent B calls `mem_inbox_send(to="mem-inbox://personal/quiet-otter", body=...)`. Later, user asks Agent A to *"check the inbox"* → Agent A calls `mem_inbox(to=<ref>)`.

**UC2 — sender shares an instruction.** User asks Agent A to *"share this with another agent: …"*. Agent A calls `mem_inbox_open` then `mem_inbox_send`, echoes the reference back. User pastes reference into Agent B's chat *"read this"*. Agent B calls `mem_inbox(to=<ref>)`.

**UC3 — established channel.** Both agents and the user already know the reference. User says *"pass to mem-inbox://personal/quiet-otter: please update the build pipeline"*. Agent A calls `mem_inbox_send(to=<ref>, body=…)` directly. User to Agent B: *"check mem-inbox://personal/quiet-otter"* → Agent B calls `mem_inbox(to=<ref>)`.

### Tool surface

| Tool | Wraps | Purpose |
|---|---|---|
| `mem_inbox_open(env_id/env_name, *, name?, title?, idempotent=False)` | `entity_upsert(kind="channel")` | Creates a channel entity. Returns the formatted reference. Auto-generates slug when `name` omitted. |
| `mem_inbox_send(to, body, *, env_id/env_name?, title?, expires_at?, display_from?)` | `memory_write(kind=message)` | Sends a message to the channel. Default 7-day TTL (cap 90d). **Rejects** non-existent slugs — explicit `mem_inbox_open` required first (prevents typo-driven channel proliferation). |
| `mem_inbox(to, *, env_id/env_name?, cursor?, limit=20, include_expired=False, order="desc")` | internal SQL | Reads messages from the channel. Internal SQL (not `mem_browse` — its `tags` filter is OR semantics; inbox needs AND between `kind='message'` and the entity link). Returns newest-first by default. |

### Reference parsing

Both `mem_inbox_send` and `mem_inbox` accept either form on `to`:

* URL form (`mem-inbox://<env>/<slug>`) — env resolved from URL. If caller also passes `env_id` or `env_name`, both must match or the call raises `InvalidInputError` (UC2 invariant — prevents silent cross-env writes).
* Bare slug — `env_id` or `env_name` arg required.

### Schema additions

* **`MemoryKind.message`** — distinct from `event` so messages don't pollute timeline reads or factual digests.
* **`EntityKind.channel`** — distinct from `agent` so channels are visibly multi-reader endpoints, not identities.
* **`inbox` fast-filter tag** — single fixed tag; recipient is in `entity_links`, not the tag.
* **`display_from` metadata** (display-only) — server always records `created_by_agent_id`; `display_from` is for human-readable attribution and is never anonymization.

### Cross-cutting `expires_at` filter

v0.17 also tightens the default-read contract: **expired memories are now excluded from all default reads** (`mem_search`, `mem_browse`, `mem_resume`, `mem_context_pack`, `mem_auto_context`, `mem_facets`, `mem_top`, `mem_inbox`). Backward-compatible — memories without `expires_at` are unaffected. Pass `include_expired=True` on any of these read paths to surface expired rows for audit / forensic flows.

### ARC vs memory-mcp inbox

memory-mcp inbox and `agent-relay-chat` (ARC) are **not interchangeable**. Pick by intent:

| Use **ARC** for | Use **memory-mcp inbox** for |
|---|---|
| Conversation, questions, replies | Durable handoff notes / instructions |
| Real-time-ish coordination | Searchable operational memory with TTL |
| Channel chat history | Drop-box updates that survive ARC's per-session UUID gap |
| Explicit ack semantics | Recipient-addressed memory artifacts with lineage |

memory-mcp inbox does NOT carry reply threading, ack substrate, channel admin, listen/subscribe, or real-time push — those live in ARC.

See `tests/integration/test_inbox.py` (6 integration cases covering the three UC flows end-to-end + the `_resolve_env_refs` await regression) and `tests/integration/test_expires_at_filter.py` (8 cross-cutting filter cases).

