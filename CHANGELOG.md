# Changelog

## [Unreleased]

## [0.15.1] — 2026-05-29

### Fixed

- **`initialize.serverInfo.version` now reports memory-mcp's package version** instead of the MCP SDK fallback (`pkg_version("mcp")`, which was surfacing `"1.27.1"`). `FastMCP.__init__` does not expose `version=` as a kwarg, so the underlying `Server` falls through to `pkg_version("mcp")` when none is passed. Worked around by overriding `mcp._mcp_server.version = pkg_version("memory-mcp")` directly after `FastMCP(...)` construction in `src/memory_mcp/mcp_app.py`, guarded with `try/except PackageNotFoundError` for editable / dev installs. Lets MCP clients discover the deployed server version through the standard `initialize` handshake without a memory-mcp-specific tool. Upstream cleanup deferred — see roadmap (`memory-mcp-fastmcp-version-kwarg`).

### Added

- **`/healthz` payload now includes `"version"`** (memory-mcp package version) so ops who hit HTTP directly — without speaking MCP — can confirm the deployed release. Field is omitted on editable / dev installs where the package isn't pip-visible.
- **Tests** — `test_mcp_server_info_reports_package_version_not_sdk` and `test_healthz_includes_package_version` in `tests/unit/test_server_smoke.py`. Both guard against the SDK-fallback regression.

## [0.15.0] — 2026-05-29

Major release adding three new memory primitives — `mem_compose` (N→1 manual aggregation, Phase 2), `mem_decompose` (1→N manual fan-out, Phase 3), and `mem_compose` auto-wire to popular neighbors via the new `related_to_popular` predicate (Phase 4, OFF by default). Builds on v0.14.x popularity foundations (`reference_count`, `reference_velocity`, `mem_top`, decay reference-floor). Schemas package bumped to 0.15.0.

### Added — Phase 4 auto-wire (compose-only, OFF by default)

- **`related_to_popular` predicate** — new auto-wire edge type emitted by `mem_compose` when `autowire_enabled=True`. Snapshot-in-time directional edge from the newly-composed memory to the top-K most-relevant popular neighbors in the same env. Bounded fan-out (K cap), one-way (no reciprocal popular→new edges to prevent link-farm feedback), and **excluded from popularity-counter triggers** (migrations 0017 + 0021 already skip the predicate in `memories_bump_on_relation_change` INSERT + DELETE branches). Same exclusion applies to `mem_top` velocity windows and the dream recount pass — auto-wire is a navigation aid, never a popularity vote.
- **Two-stage execution** in `src/memory_mcp/autowire.py` keeps the compose lock-window minimal. **Stage A** (read-only, pre-transaction): skip filter → PG top-by-salience candidates → recursive-CTE lineage-ancestor exclusion (depth 20) → off-thread body embed via `loop.run_in_executor` → Qdrant similarity → combined `salience × sim_score` ranking with deterministic `(combined DESC, id DESC)` tie-break. **Stage B** (in-transaction, step 13.5 of `_compose_in_session`): graph-node resolution + raw-SQL INSERT with `ON CONFLICT (src_node_id, dst_node_id, type) DO NOTHING` for race / replay safety + one audit row per edge (`op=auto_wire:related_to_popular`) + outbox relation events.
- **Skip filter** rejects targets whose `kind=playbook` OR whose tags include any `directive:active` prefix OR whose body is empty/whitespace. Applied both pre-compute (Stage A) and defensively at insert (Stage B) so a feature-flag flip mid-flight can't smuggle in a skip-listed target.
- **Replay reconstruction is state-current, not operation-exact.** Both compose replay branches (dedupe-hit and savepoint race-loss) call `reconstruct_auto_wired` to re-query live `related_to_popular` edges via `relations → graph_nodes` join. A manually-added edge of the same type between the same memories will surface in replay output — documented in the module docstring and the `MemComposeResponse.auto_wired` field docstring.
- **Failures degrade silently.** Embedder failure, vector-store failure, graph-node resolution failure, insert race — each is logged at WARN and short-circuits auto-wire to `[]`. The compose call always succeeds; auto-wire never blocks the transaction.
- **Settings knobs** (`src/memory_mcp/config.py`): `autowire_enabled` (default `False`), `autowire_top_k` (`3`, range `1..10`), `autowire_sim_threshold` (`0.70`, provisional — calibration pending and documented), `autowire_candidate_limit` (`20`, must be `>= autowire_top_k`, range `1..200`). Validator enforces the invariant.
- **Decompose auto-wire deferred to v0.16.** `MemDecomposeResponse.auto_wired: list[UUID]` is already on the schema but always `[]` in v0.15.0 — the flat shape can't disambiguate per-child mapping for N children × K edges. v0.16 candidate: additive optional `auto_wired_by_child: dict[UUID, list[UUID]] | None`.
- **README `## Compose` → `### Auto-wire (Phase 4, OFF by default)`** subsection covers two-stage design, knobs table, replay-state-current semantics, skip-filter rules, popularity exclusion guard, decompose deferral.

### Tests (Phase 4)

- Unit: 23 cases across `tests/unit/test_autowire.py` — predicate constant lock-in (1), skip filter (10 — empty body, whitespace, playbook kind, playbook string, `directive:active` tag prefix exact + suffix, normal pass-through, `directive:retired` pass-through, empty/None tags), Stage A (12 — feature OFF, skip filter, no PG candidates, embedder failure, vector store failure, threshold cutoff, combined ranking with deterministic tie-break, source-id exclusion, lineage-ancestor exclusion, invalid Qdrant id silently skipped, empty Qdrant response, embedder returns empty vectors).
- Integration: 6 cases across `tests/integration/test_autowire_compose.py` — OFF regression (no rows, no extra audits), Stage B direct insert (relations + audit + popularity-trigger guard), Stage B `ON CONFLICT DO NOTHING` semantics, replay reconstruction round-trip, compose-hook ON path with fake embedder + vector store (1 edge emitted + response populated), compose-replay-reconstructs (`idempotency_replay=True` + matching `auto_wired`).
- Full suites green at Phase 4 close: **1244 unit + 52 integration** (compose 25 + decompose 21 + auto-wire 6).

### Internal (Phase 4)

- `src/memory_mcp/autowire.py` (~450 LOC) — new module home. Three public functions (`autowire_fetch_candidates` / `autowire_compose_target` / `reconstruct_auto_wired`) plus internal `_should_skip_target` and `_collect_lineage_ancestors` helpers. Mirrors `composers.py` failure-handling style (BLE001 noqa around all external dependencies). Imports `_ensure_graph_node` + `_record_relation_audit` from `relations.py`, `enqueue_event` from `db/outbox.py`, `get_embedder` from `embeddings/base.py`, `RelationEndpoint` from `memory_mcp_schemas.relations`.
- `src/memory_mcp/composers.py` — `_build_response` now accepts `auto_wired: list[UUID] | None = None`; new `_autowire_stage_a` helper opens a short read-only session pre-transaction; `_compose_in_session` accepts the pre-computed candidates kwarg; both replay branches reconstruct via `reconstruct_auto_wired`; Step 13.5 calls `autowire_compose_target` between outbox enqueue and final response builder.
- Two bug fixes in `autowire.py` discovered during D6b integration runs and shipped with the same commit: raw-SQL INSERT now wraps UUID binds in `CAST(:name AS uuid)` (asyncpg was inferring `varchar`); `ON CONFLICT` now targets the columns `(src_node_id, dst_node_id, type)` instead of the (non-existent) named constraint `relations_src_dst_type_uniq` — migration 0001 created the UNIQUE constraint inline without an explicit name. `reconstruct_auto_wired` carries the same UUID cast.

### Added — Phase 3 `mem_decompose` (1→N, manual)

- **`mem_decompose` MCP tool** — caller-driven 1→N decomposition. 1 source → 2–20 children in one env, two modes (`derive` non-destructive default; `split` destructive). Atomic transaction: lock source → cheap envelope validation → dedupe-key lookup (operation table) → source-state validation → insert children → write lineage rows → audit + outbox events → optional source retirement (split only). Mirrors `composers.py` structurally — same `_lock_memories` primitive, same dedupe-key envelope shape, same replay path via the operations table.
- **`MemDecomposeChild` / `MemDecomposeRequest` / `MemDecomposeResponse` / `DecomposeLineageRow`** schemas in `memory_mcp_schemas.decompose`. Strict envelope: `children: list[MemDecomposeChild]` with `min_length=2`, `max_length=20`, per-child validator rejects `kind=playbook` (children can't carry `steps`). `expected_version` for optimistic-lock on the source. `idempotency_key` ≤128 chars override of the derived hash.
- **Migration `0021_decompose_operations`** — adds the `decompose_operations` table (`id`, `env_id`, `source_id`, `mode`, `dedupe_key`, `request_fingerprint`, `child_ids[]`, `created_at`, `created_by_agent_id`) with unique index `ix_decompose_operations_dedupe (env_id, dedupe_key)` (the race arbiter) and `ix_decompose_operations_source (source_id)`. Widens `memory_lineage_relation_check` CHECK to admit `split_from` + `derived_from` (the migration discovered the column is TEXT with an inline CHECK, NOT a Postgres ENUM type, so the lookup matches the column-name token via regex rather than `ALTER TYPE`). Re-issues the popularity-trigger function `memories_lineage_increment` / `_decrement` with `split_from` **removed** from the whitelist (E.11 — split retires the source; bumping a retired memory's counter is forensic pollution and risks feedback loops on reactivation). Forward-only (alembic head: `0021`). Companion edits to `recount.py` `_LINEAGE_WHITELIST` and `top.py` `_LINEAGE_VELOCITY_WHITELIST` so the runtime whitelists stay in lock-step with the trigger.
- **Idempotency contract** — dedicated `decompose_operations` table (NOT a column on the source — a source can be decomposed multiple times without collisions). Dedupe key derived from `{schema_version, operation, env_id, mode, source_id, sorted(canonical_json(children))}` → SHA-256 → 32 hex chars. Stored alongside a stricter **request fingerprint** that includes every field — including `expected_version`, per-child `trigger_description`, per-child `expires_at`. Replay returns the original children's UUIDs with `idempotency_replay=True` (no new rows, no audit entries, no popularity bumps). Caller-supplied `idempotency_key` (≤128 chars) overrides the derived hash; the fingerprint is always canonical, so reusing a caller key with a different source / mode / child-set raises `InvalidInputError("idempotency_key reused with different scope")` rather than silently echoing a stale response.
- **Concurrency** — the source's `FOR UPDATE` lock plus the `(env_id, dedupe_key)` unique constraint on `decompose_operations` form the race winner-loser arbiter. Concurrent identical decomposes serialize via the source lock; the loser sees the persisted operation row, validates the request fingerprint matches, and returns the replay path. Both succeed; one carries `idempotency_replay=True`. Pattern mirrors compose's `_is_compose_dedupe_error` classifier with a decompose-specific `_is_decompose_dedupe_error` checking `ix_decompose_operations_dedupe` across `orig.constraint_name` / `orig.diag.constraint_name` / substring fallback.
- **Lineage rows** — `mode='derive'` emits `derived_from` (parent=source, child=new); `mode='split'` emits `split_from`. With the revised whitelist, `derived_from` bumps `source.reference_count_lineage += N` per call (whitelisted), `split_from` does NOT (excluded — the source is retired). Lineage rows never produce outbox events (Postgres-only invariant preserved, matching compose / dream).
- **Audit shape** — each child: `op='create'` with `extra_after={decompose_mode, decompose_source, decompose_operation_id}`. Source: one `op='mem_decompose:{mode}'` aggregate row with `extra_after={child_ids, dedupe_key, operation_id, decompose_mode}`. Source on `mode='split'`: an additional `op='retire'` row. Filterable on `op LIKE 'mem_decompose:%'`.
- **Outbox shape** — `mode='derive'`: N `upsert` events (one per child); 0 events for the source. `mode='split'`: N `upsert` events + 1 `tombstone` for the source. Lineage and `decompose_operations` rows produce zero outbox events.
- **Provenance** — each child carries a `MemorySource` row with `source_type='agent'` and `source_ref=str(operation_id)` (the `MemorySourceType` enum does not currently include a `mem_decompose` value; the operation-id is the back-pointer for analytics / lineage traversal).
- **README `## Decompose (v0.15.0 — Phase 3)`** — new section covering two-mode model, popularity whitelist asymmetry, idempotency contract, provenance convention, audit / outbox shape, validation contract. Tools (v1) line updated to include `mem_decompose`.

### Validation matrix (Phase 3)

`InvalidInputError` raised for: duplicate children by canonical-JSON hash, `decision_meta` on non-`decision` child, caller-key reuse with different scope, source `kind=playbook`. `InvalidTransitionError(src=status, dst='decomposed')` for retired / superseded / proposed / archived sources on first write (replay survives later transitions per the dedupe-before-state-validation rule). `VersionConflictError` for stale `expected_version`. `NotFoundError` for source not visible in caller's `attached_env_ids`. Schema-layer Pydantic rejects `len(children)<2 / >20`, `idempotency_key>128`, per-child `kind=playbook`. Mixed-kind children allowed in both modes (D.5 confirmed — decompose is heterogeneous by nature).

### Popularity / citation caveat (Phase 3)

`split_from` lineage edges **never** bump `reference_count_lineage` (whitelist excludes them). The asymmetry is deliberate — split retires the source, and bumping a retired memory's counter pollutes analytics on accidental reactivation. `derived_from` edges DO bump the source's counter (it remains the conceptual originator of N atomic derivatives). Children always start at `reference_count=0` and accrue their own incoming citations from this point forward. Citation transfer (rewriting incoming `rel_link` / `{{memory:<uuid>}}` references to point at children) is **deferred to v1.5** — same boundary compose adopts.

### Tests (Phase 3)

- Unit: schema + helpers (existing C3 + C4 coverage already in `[Unreleased]`).
- Integration: 21 cases across `tests/integration/test_decompose_transaction.py` (8 smoke + 13 matrix — duplicate, decision_meta on non-decision + on decision, RBAC, retired-source, expected_version mismatch, mixed-kind, concurrent-race, split_from regression, audit shape for split + derive, schema-layer rejections).

### Internal (Phase 3)

- `src/memory_mcp/decomposers.py` (~1040 LOC) — new module home for decompose. Public `memory_decompose()` entry point opens `session_scope()`; `_decompose_in_session()` is the 18-step transaction body. Mirrors `composers.py` (lock → dedupe → validate → mutate → audit → outbox → build response). Latent C6 bugfix discovered during C7: split-mode source UPDATE referenced `Memory.retired_at`, which does not exist (Memory carries `updated_at` + `status` only). Removed the unconsumed value; compose's parity pattern (status + version + updated_at) is what split uses too.

### Added — Phase 2 `mem_compose` (N→1, manual)

- **`mem_compose` MCP tool** — caller-driven N→1 aggregation. 2–20 sources in one env, two modes (`promote` non-destructive default; `merge` destructive). Atomic transaction: lock → dedupe-key check → validate → insert merged memory → write lineage rows → audit + outbox events → optional source retirement (merge only). Mirrors the dream-worker's `_accept_merge` / `_accept_promotion` lineage shape but bypasses the proposal envelope so agents and humans can compose directly.
- **`MemComposeRequest` / `MemComposeTarget` / `MemComposeResponse`** schemas in `memory_mcp_schemas.compose`. Strict envelope: `source_ids: list[UUID]` with `min_length=2`, `max_length=20`, dedupe + subset validation on `expected_versions`. Per-mode tag-policy defaults (`promote→target`, `merge→target_plus_union`); explicit override via `tag_policy ∈ {"target", "union", "target_plus_union"}`. `None` means mode-default; `[]` means "no target tags but policy still folds sources".
- **Migration `0020_compose_dedupe_key`** — adds `memories.compose_dedupe_key TEXT NULL` + partial unique index `memories_compose_dedupe_unique (env_id, compose_dedupe_key) WHERE compose_dedupe_key IS NOT NULL`. Forward-only (alembic head: `0020`).
- **Idempotency contract** — deterministic dedupe key derived from `{schema_version, operation, env_id, mode, sorted(source_ids), target.kind, sha256(title+body), sorted(target.tags)}` → SHA-256 → 32 hex chars. Stored on `memories.compose_dedupe_key`. Replay returns the original memory with `idempotency_replay=True`; no new rows, no audit entries, no popularity bumps. Caller-supplied `idempotency_key` (≤128 chars) overrides the derived hash. Reusing a caller key with a different mode / source set raises `InvalidInputError`.
- **Lineage rows** — `mode='promote'` emits `promoted_from` (child=merged, parent=source); `mode='merge'` emits `supersedes` (whitelist-excluded from `reference_count_lineage` — sources do NOT gain inbound-lineage credit on the merge path; matches dream-worker semantics).
- **Audit shape** — merged memory: `op='create'` + `op='mem_compose:{mode}'`. Sources on `mode='merge'`: one `op='supersede'` each. Sources on `mode='promote'`: untouched.
- **Outbox shape** — merged memory: 1 `upsert`. Sources on `mode='merge'`: N `tombstone`. Sources on `mode='promote'`: 0 events. Lineage rows: 0 outbox events (Postgres-only invariant preserved).
- **README `## Compose (v0.15.0 — Phase 2)`** — new section covering two-mode model, idempotency contract, popularity caveat, audit / outbox shape, validation contract. Tools (v1) line updated to include `mem_compose`.

### Validation matrix

`InvalidInputError` raised for: cross-env sources, env-mismatch on `request.env_id`, mode=merge with mixed source kinds, mode=merge with `target.kind != source.kind`, caller-key reuse with different scope. `InvalidTransitionError(src=status, dst='composed')` for retired / superseded / proposed / archived sources. `VersionConflictError` for stale `expected_versions`. `NotFoundError` for sources not visible in caller's `attached_env_ids`. Schema-layer Pydantic rejects `len<2 / >20`, duplicate IDs, `expected_versions ⊄ source_ids`, both `env_id`+`env_name`, `idempotency_key>128`.

### Popularity / citation caveat

The merged memory **starts at `reference_count=0`**. Compose does **not** transfer the sources' inbound citations (`rel_link`, embedded `{{memory:<uuid>}}` references, task citations). Sources retain their full popularity profile. Lineage edges are intentionally excluded from `reference_count_*` (they describe how the memory got here, not who depends on it). Citation rewrite / lazy resolution deferred to v1.5; recommended pattern: `mem_archive` or `mem_retire` sources after compose so consumers naturally migrate.

### Tests

- Unit: 17 dedupe-key + schema cases.
- Integration: 25 cases across `tests/integration/test_compose_transaction.py` (8 smoke + 6 accounting + 11 validation/tag-policy/race matrix).

### Internal

- `_lock_memories` extracted from `dream/api` to `memories` (B3a). Shared by compose and dream-accept paths.
- `composers.py` is the new home for compose logic (~700 LOC). `dream/api._accept_merge` / `_accept_promotion` to be refactored on top of `composers._compose_in_session` in a follow-up.
- Fixed latent bug in `composers.py`: `InvalidTransitionError` was called with single positional message arg, but the class signature is `(src, dst)`. Smoke tests never exercised the 4 envelope-validation paths so the bug was latent until B-finish-2 wrote them. Fixed: envelope checks now raise `InvalidInputError`; status check uses proper `InvalidTransitionError(src=status, dst='composed')`.

## [0.14.1] — 2026-05-19 — Popularity Phase 1e (authority weighting)

### Added
- **`reference_authority` signal** — each `memories` row gains four per-kind float columns (`ref_authority_rel_link`, `ref_authority_lineage`, `ref_authority_task`, `ref_authority_playbook`) plus a stored `reference_authority NUMERIC GENERATED ALWAYS AS (sum) STORED`. Surfaces the **weighted** citation footprint (`Σ source.salience` over inbound citations) — complementing the **counted** footprint (`reference_count_*`) added in v0.14.0.
- **Migration 0018 (`authority_columns`)** — adds the four authority columns + GENERATED total + partial covering index `memories_reference_authority_idx (env_id, status, reference_authority DESC, created_at DESC, id DESC) WHERE reference_authority > 0` (matches `mem_top by="reference_authority"` access pattern).
- **Migration 0019 (`salience_formula_version`)** — adds `memories.salience_formula_version INTEGER NOT NULL DEFAULT 0`. Used by the recount pass to detect rows on a stale salience formula and re-stamp them; future-proofs the salience math against silent drift after operators upgrade.
- **Recount authority leg** — `dream.passes.recount` extended (R-B3 / R-S8 / R-S9) to compute `Σ source.salience` from inbound `relations` (rel_link + task), `memory_lineage` (lineage), and embedded `{{memory:<uuid>}}` macros in `playbook.steps` (per-occurrence). Writes the four per-kind columns; the total is GENERATED. Reconciles drift (canonical writer pattern from v0.14.0). Mirrors the integer-counter exclusions: chain-ancestry, retired-citer, `related_to_popular` predicate, cross-env playbook macros, self-citation. Task-sourced edges contribute `0` to `ref_authority_task` (tasks have no salience column).
- **Recount salience-recompute step** — counter-changed and authority-changed rows feed into a salience recompute leg that re-evaluates `compute_salience()` and stamps `salience_formula_version` to the current setting. Persists via `MemoryUpdatePatch` so outbox / Qdrant payload stay consistent. Bounded per cycle via `dream_recount_salience_recompute_cap=500` (configurable; `0` = unbounded).
- **Formula-version backfill** — first post-deploy recount cycle picks up all existing rows (default `salience_formula_version=0 < target=1`) and re-stamps them under the cap. `RecountPassResult` exposes `memories_formula_version_restamped` + `memories_formula_version_pending` so operators can monitor the drain.
- **`compute_salience` authority term** — `w_authority · clamp01(log1p(reference_authority) / log1p(authority_window))`. Pure function — knob-gated via the weight: `salience_weights_from_settings` returns `w_authority=0.0` when `dream_popularity_authority_weighted=False`. Default OFF — no behaviour change on existing envs until opt-in.
- **New Settings**:
  - `dream_popularity_authority_weighted: bool = False` — master knob.
  - `dream_salience_w_authority: float = 0.10` — weight of the authority term in salience.
  - `dream_salience_authority_window: float = 25.0` — log1p normalization saturation point (~50 citers at avg salience 0.5).
  - `dream_popularity_authority_damping: float = 1.0` — recurrence damping; reserved (no-op at 1.0).
  - `dream_salience_formula_version: int = 1` — current formula version stamp.
  - `dream_recount_salience_recompute_cap: int = 500` — per-cycle cap on formula-version backfill rows.
- **`mem_top by="reference_authority"`** — new ranking metric. Returns the highest-authority memories with stable tie-breaker `(reference_authority DESC, created_at DESC, id DESC)`. Mirrors `reference_velocity` semantics: zero-authority rows are excluded from `items` but counted by `total_examined`.
- **`MemoryResponse.reference_authority: float = 0.0`** — additive response field. Defaults to 0.0 so old clients are tolerated.
- **`AUTHORITY_DISABLED` error code** — raised by `mem_top by="reference_authority"` when `dream_popularity_authority_weighted=False`. Fires before env / RBAC / DB so callers get a clean "metric unavailable" signal at no cost.

### Changed
- **`w_negative` default `0.40 → 0.46`** — absorbs the new authority term (saturated at `w_authority=0.10`) so the dominance invariant holds at the narrowed scope (`confidence=0, pinned=False, verified_at=None`). New margin `-0.0242` (was `-0.1242` in v0.14.0); still negative-clamped to 0 → 5 negative events still suppress a memory. Memories with 1 negative event now subtract `~0.319` instead of `~0.277` (~15% bigger hit).
- **`SalienceWeights.w_authority` default `0.10 → 0.0`** — direct-constructor unit-test paths now default to knob-OFF semantics. The `0.10` lives in `Settings.dream_salience_w_authority` only and is bound through `salience_weights_from_settings`. Callers constructing `SalienceWeights(...)` directly must pass `w_authority=...` explicitly to opt in.
- **`SalienceInputs.reference_authority: float = 0.0`** — new dataclass field; all callsites (recount, decay, two `memories.py` access-bump paths) updated to populate it. Access-bump reads the field but does **not** stamp `salience_formula_version` — only recount stamps.

### Notes
- **DB migration required**: alembic head advances to **`0019_salience_formula_version`** (via `0018_authority_columns`). No backfill of authority counters in `upgrade()` — the recount pass is canonical writer.
- **First post-deploy recount cycle**: all existing rows have `salience_formula_version=0`, so the formula-version backfill leg lifts up to `dream_recount_salience_recompute_cap=500` per cycle. Default cap × default cadence (3600s) drains ~12k rows/day. Operators with larger envs can raise the cap or set it to `0` (unbounded — drains in one cycle).
- **Ship-dormant**: `dream_popularity_authority_weighted` defaults to `False`. With the knob OFF, `compute_salience` zeros the authority term, recount writes zeros to the four `ref_authority_*` columns (idempotent), and `mem_top by="reference_authority"` returns `AUTHORITY_DISABLED`. No observable behaviour change on existing envs until an operator opts in.
- **Response wire-format**: `MemoryResponse.reference_authority` is additive and Pydantic-default-safe. Strict typed clients (Pydantic `extra="forbid"` consumers on the *response* side) may need to recompile their schemas to recognize the new field; the JSON wire is byte-compatible for clients that ignore unknown fields.
- **Decay-resistance shift (knob-ON only)**: with `dream_popularity_authority_weighted=True` and `reference_authority > 0`, salience is higher → cited memories decay more slowly than under v0.14.0. Intentional design (cited memories ARE more valuable). Max boost is `w_authority=0.10` at saturation; cannot lift below-threshold rows past the dominance invariant.
- **Future formula bumps**: any change to `compute_salience` math MUST bump `dream_salience_formula_version`. The recount pass picks up the version mismatch and re-stamps existing rows on subsequent cycles. Documented in `salience.py` module docstring + `config.py` field comment.

## [0.14.0] — 2026-05-19 — Popularity Phase 1 (graph-citation reference counters)

### Added
- **Four per-kind `reference_count_*` columns on `memories`** — `reference_count_rel_link`, `reference_count_lineage`, `reference_count_task`, `reference_count_playbook` — plus a stored `reference_count INTEGER GENERATED ALWAYS AS (sum) STORED` for ordered-scan indexes. Surface the structural graph-citation signal that lived only on the read axis (`access_count` / `last_accessed_at`) before now.
- **Three Postgres triggers (Migration 0017)** keep counters at transactional truth in the hot path:
  - `memories_bump_on_relation_change` — `AFTER INSERT OR DELETE ON relations`. Branches on `src.node_type` (task → `task`; else → `rel_link`), skips Phase 4's reserved `related_to_popular` predicate, no-ops on UPDATE.
  - `memories_bump_on_lineage_change` — `AFTER INSERT OR DELETE ON memory_lineage`. Counts only the load-bearing whitelist `{summarized_from, promoted_from, derives_from, split_from, derived_from}`; **excludes `supersedes`** so version-chain bookkeeping does not inflate parent authority. `split_from` / `derived_from` are forward-listed for Phase 3.
  - `memories_status_flip_decrement` — `AFTER UPDATE OF status ON memories WHEN OLD.status IS DISTINCT FROM NEW.status`. Walks outgoing edges from the flipping memory on `active ↔ (retired | superseded)`; symmetric (re-increments on un-retire).
- **Bounded sync backfill in `upgrade()`** — single-pass aggregate when `relations + memory_lineage` together are below 100k edges; above that the migration leaves counters at zero and emits a NOTICE. The dream `recount` pass is canonical truth and fills in either case.
- **`mem_top` MCP tool** — `mem_top(env_id?, by, kinds?, tags?, tag_match='any'|'all', status?, velocity_window_days?, limit=10)`. Returns memories ordered by `salience | access_count | reference_count | reference_velocity` with stable tie-breaker `ORDER BY metric DESC, created_at DESC, id DESC`. `tag_match='all'` enables AND semantics (vs `mem_browse`'s OR default).
- **`MemoryResponse` carries `reference_count`, `reference_breakdown: {rel_link, lineage, task, playbook}`, and `reference_velocity`** — additive; old clients tolerate the new fields.
- **Salience extension** — `SalienceInputs` and `SalienceWeights` gain reference-aware terms. Per-kind counts are normalized independently via `log1p(N) / log1p(window)`, then weighted-averaged and scaled by `w_references` (default `0.15`). Per-kind weights: `w_rl=1.0`, `w_ln=1.5`, `w_tk=1.2`, `w_pb=2.0`; per-kind windows: `50 / 5 / 20 / 10`. The dominance invariant (negative term swallows max positives at 5 negative events) is preserved by bumping `w_negative` from `0.30` to `0.40`.
- **Decay-resistance gate** — `dream_decay_reference_floor: int = 3` (settable). The decay pass skips the `active → stale` transition when `reference_count >= floor`. Structural decay (stale → archived) is not gated.
- **New dream pass `recount`** (`memory_mcp.dream.passes.recount`) — canonical writer of the four counters. Reconciles drift, performs supersede-chain ancestry exclusion (S6 — triggers cannot afford it on the hot path), and text-scans active playbook `steps[]` for `{{memory:<uuid>}}` macros (the playbook counter has no edge row backing it). Idempotent.
- **Scheduler wiring for the new pass** — `JOB_ID_RECOUNT = "dream-recount"`; default `dream_recount_cadence_seconds = 3600`; `_MODE_LOCK_KEY` slot 5; `DreamMode.recount` registered in both `schemas/enums.py` and `schemas/dream.py`; dispatch branch in `dream_worker.jobs._dispatch_pass`.
- **New indexes** — `memories_reference_count_idx (env_id, status, reference_count DESC, created_at DESC, id DESC)` for stable-order `mem_top by=reference_count`; `relations_velocity_idx (env_id, created_at DESC, dst_node_id)` and `memory_lineage_velocity_idx (created_at DESC, parent_memory_id)` cover the velocity CTEs.

### Changed
- **`w_negative` default raised `0.30 → 0.40`** to absorb the new positive `references` term while preserving the dominance invariant. Memories with 1 negative event now subtract `~0.277` instead of `~0.208` (~33% bigger hit). Pre-existing decay tests updated.
- **`dream_runs.mode` CHECK extended** to admit `'recount'` alongside the existing `decay / dedupe / promote / retention / decision_conflicts`. Migration 0017 drops and recreates the constraint following the v0.12 pattern.
- **Default velocity window raised `14 → 30` days** (`mem_reference_velocity_window_days: int = 30`). Per-request override via `mem_top.velocity_window_days`. Playbook citations are excluded from velocity (no per-edge timestamp).

### Notes
- DB migration required: alembic head advances to **`0017_popularity_counters`**. The fast-path backfill is bounded to 100k edges; envs above that rely on the recount pass for first-fill.
- The salience formula change is **observable to callers**: memories with substantial graph-citation footprint will see their `salience` rise; the dominance invariant is preserved at the negative-event tail. Decay candidates change accordingly.
- **Phase 1 deferred (`mpc-phase1-authority`)** — authority weighting via `Σ source.salience` is fractional and cannot live in integer counter columns; the design needs separate float columns and a chicken-and-egg resolution for circular citations. Re-planned for a v0.14.x follow-up.
- **Intra-supersede-chain rel_link exclusion is recount-only.** Triggers may transiently over-count when a `rel_link` between two memories in the same supersede chain is inserted; the next recount-pass run converges. Documented limitation S6.
- **Backwards compatible.** `mem_top` is opt-in; the new `MemoryResponse` fields are additive; the salience change preserves the invariant; the decay-resistance gate defaults to `floor=3` (set `floor=0` to disable).

## [0.13.1] — 2026-05-16 — Plugin manifest switched to Streamable HTTP

### Changed
- **`.claude-plugin/plugin.json`** — `mcpServers.memory-mcp` switched from `stdio`-via-`docker compose exec` to **`streamable-http`** pointing at `http://127.0.0.1:8080/mcp`. The new entry shape:
  ```jsonc
  "memory-mcp": { "type": "streamable-http", "url": "http://127.0.0.1:8080/mcp" }
  ```
  Eliminates the per-call `docker exec` round-trip plus the per-call `python -m memory_mcp.server` cold-start (torch + sentence-transformers reload). Reduces per-call dispatch from ~2-3 s to sub-millisecond. Unifies the plugin install path with the README's manual-install HTTP snippet — both shapes are now the same connection target.
- **Failure mode when the stack is not running** changed from cryptic `service "server" is not running` (compose-exec error) to actionable `connection refused at 127.0.0.1:8080` (HTTP).
- **README "Install (Copilot CLI plugin)" section** rewritten to describe the HTTP transport and note that non-default `MEMORY_MCP_HOST_PORT` setups require editing the installed manifest at `~/.copilot/installed-plugins/_direct/outwarped-io--memory-mcp/.claude-plugin/plugin.json` to match.

### Notes
- **The stack still needs `docker compose up -d` after `/plugin install`** — HTTP doesn't fix the bootstrap gap, just the failure UX. The remaining auto-start gap is tracked as `memory-mcp-v014-plugin-stack-bootstrap` in the workspace's `.github/TODO.md`.
- **No database migration.** Alembic head stays at `0016_cascade_root` (from v0.13.0).
- **No server-side code changes.** memory-mcp has served Streamable HTTP at `/mcp` since v0.1; this release flips a single line in the plugin manifest.
- **Per-session identity headers** (`X-Agent-Id` / `X-Agent-Name`) are **not** included in the plugin manifest. Plugin-manifest `url` values are not env-templated by Copilot CLI, so the headers would be fixed strings — useless for per-session identity. Users who want per-session identity should keep using `~/.copilot/settings.json` with `${env:...}` header values. The broader per-session-identity gap remains open as `arc-per-session-identity-gap` in the workspace's `.github/TODO.md`.

## [0.13.0] — 2026-05-15 — Hard-delete cascade + search/graph relax + env-name wave 2

### Added
- **`MemoryHardDeleteRequest.cascade: bool = False`**, **`max_cascade_depth: int = 5`**, **`max_cascade_count: int = 20`**, and **`dry_run: bool = False`** — opt-in cascade mode on `mem_hard_delete`. When `cascade=True`, the server walks forward lineage via `mem_lineage(direction=forward)`, topo-orders the affected subtree leaves-first / root-last, `SELECT FOR UPDATE`s every row, and hard-deletes the entire set in a single transaction. `dry_run=True` returns the plan without mutating rows.
- **`MemoryHardDeleteResponse.cascade_root: UUID | None`** and **`affected: list[MemoryHardDeleteAffected]`** — caller-visible blast-radius report for cascade mode. `affected` is ordered leaves first, root last; one `memory_tombstones` row is written per destroyed row and all share the same `cascade_root` correlation id.
- **`BlastRadiusExceededError`** — safety guard for cascade deletes. Raised when either cap is exceeded; names which cap fired (`depth` or `count`) and returns the partial `affected` list gathered so far so the caller can review before retrying with wider limits.
- **`MemorySearchRequest.expansion: ExpansionPreset | None = None`** (`narrow | default | broad`) plus **`MemorySearchResponse.expansion_resolved: dict[str, Any] | None`** — preset bundles on top of the v0.12 raw recall knobs. `narrow` resolves to `{min_score=0.035, fallback=False, follow_superseded=False}`; `default` aliases current behavior; `broad` resolves to `{min_score=None, fallback=True, follow_superseded=True, include_stale=True, include_archived=True}` while still excluding retired rows by default.
- **Graph-traversal relax knobs on `mem_neighbors` and `mem_related`** — `MemNeighborsRequest.fallback: bool = False`, `MemRelatedRequest.min_score: float | None = None`, `MemRelatedRequest.fallback: bool = False`, and `fallback_used: list[str]` on both responses. Fallback cascade steps fire in order: `widen_hops` (cap 3) → `drop_predicate` → `include_retired`.
- **Wave-2 friendly env-name twins on 18 P1 request schemas** — `env_name: str | None` / `env_names: list[str] | None` alongside optional `env_id` / `env_ids` on `DigestRequest`, `ResumeRequest`, `DreamRunRequest`, `DreamStatusRequest`, `DreamProposalsListRequest`, `EntityBrowseRequest`, `EnvDeleteRequest`, `EnvExportRequest`, `EnvRenameRequest`, `EnvSnapshotRequest`, `EntityNeighborsRequest`, `MemNeighborsRequest`, `MemRelatedRequest`, `MemLineageRequest`, `MemSourcesBrowseRequest`, `RelationBrowseRequest`, `TaskCreateRequest`, and `TaskListRequest`.

### Changed
- **`MemoryHardDeleteResponse.deleted_at`**, **`projection_eviction`**, and **`tombstone_id`** are now nullable so `dry_run=True` can return a truthful non-mutating response without fake timestamps or tombstones.
- **`mcp_app.py` wave-2 wrapper wiring** — all 18 affected tool wrappers now resolve env-name siblings through `_resolve_env_refs`; forensic/recovery wrappers `env_delete_`, `env_rename_`, `env_export_`, `env_snapshot_`, and `mem_lineage` pass `allow_deleted=True` so deleted-env forensics still work.
- **Validators tightened around the new relax presets** — `MemorySearchRequest` rejects `expansion` combined with explicit overrides for the six fields it owns (`min_score`, `fallback`, `follow_superseded`, `include_stale`, `include_archived`, `include_retired`); `MemRelatedRequest.min_score` is rejected for non-semantic relations because the threshold is post-fusion semantic-only.

### Notes
- DB migration required: alembic head advances to **`0016_cascade_root`**, adding nullable `memory_tombstones.cascade_root UUID` plus an index for cascade-correlation lookups.
- The SDK (`memory-mcp-client`) stays at **0.3.0** in v0.13. The new request fields flow through the existing generic `**kwargs` plumbing; SDK **0.4.0** ships separately with v0.14 batch helpers.
- v0.13 is backwards compatible: every new field defaults to the pre-v0.13 no-op behavior (`cascade=False`, `dry_run=False`, `expansion=None`, graph `fallback=False`, graph `min_score=None`, and env-name twins omitted).
- Deferred follow-up: 20 response/support schemas still surface env identity as UUIDs only; tracked in `.github/TODO.md` as `memory-mcp-v013-response-schema-env-names`.

## [0.12.0] — 2026-05-14 — Search relax/tighten knobs

### Added
- **`MemorySearchRequest.min_score: float | None`** — post-fusion score
  threshold (the *tighten* lever). Hits with `score < min_score` are
  dropped before truncation; applied after the salience boost so the
  threshold reflects the caller-visible final `score`. Empirical
  reference points on the default RRF + salience scale: ~0.016 at the
  50th percentile, ~0.035 at the 90th percentile. Combines with
  `fallback` — if the threshold empties the result set, the fallback
  ladder treats it as 0 hits and continues broadening.
- **`MemorySearchRequest.fallback: bool`** — auto-broaden cascade on
  empty results (the *loosen* lever). When True and the initial query
  returns 0 hits, the server re-runs the query with progressively
  broader scope and returns the first non-empty pass. Steps in order:
  (1) `mode=lex` → `hybrid`, (2) drop `kinds` / `tags` / time bounds,
  (3) widen lifecycle (`include_stale` + `include_archived`),
  (4) drop `follow_superseded` and boost `limit` to `min(limit*5, 100)`.
  Each step is gated on the prior pass returning 0 hits and is skipped
  when it would be a no-op. `mode=id` does not participate.
- **`MemorySearchResponse.fallback_used: list[str]`** — names of the
  cascade steps that actually fired (`mode->hybrid`, `drop_filters`,
  `widen_lifecycle`, `boost_limit`). Empty when no fallback ran or the
  original query already returned hits — additive response field; older
  clients can safely ignore it.

### Notes
- Both new request fields default to no-op (`min_score=None`,
  `fallback=False`). v0.12 is fully backwards compatible — no migration
  required, no behavior change for existing callers.
- The SDK (`memory-mcp-client`) wires the new kwargs through its
  generic `**kwargs` plumbing; no API-shape change beyond the schema
  bump.
- Deferred to a future release pending real-traffic telemetry:
  a preset `expansion: narrow | default | broad` enum bundle and a
  loosen knob on graph traversal (`mem_neighbors` / `mem_related`).

## [0.11.0] — 2026-05-14 — Hard delete · Wave-2 env naming · did-you-mean hints

### Added
- **`mem_hard_delete` MCP tool** — permanently destroys a memory's canonical row,
  body, embedding payload, lineage edges (via FK CASCADE), and projections (via
  `OutboxOp.tombstone` to Qdrant + Neo4j). Required for the sensitive-write
  recovery protocol in `memory-mcp.instructions.md §14`. Refs-guarded (V1
  rejects when other rows cite the target — caller must `mem_retire` /
  `mem_supersede` dependents first; tracked as `memory-mcp-hard-delete-cascade`
  follow-up). Optimistic-locked via `expected_version`. Requires
  `confirm_destroy=true` and a non-empty `reason`.
- **`memory_tombstones` table** (migration `0015_memory_tombstones`) — audit
  rows recording (`env_id`, `deleted_by_agent_id`, `reason`, `original_kind`,
  `original_status`, `deleted_at`). No body, no embeddings, no PII — just
  enough audit footprint to investigate leak-recovery after the fact.
- **v0.9 wave-2 env-name twins on 7 high-value request schemas** —
  `MemoryWriteRequest`, `MemoryJournalRequest`, `MemBrowseRequest`,
  `MemFacetsRequest`, `EntityUpsertRequest`, `EntityResolveRequest`, and
  `RelationLinkRequest` now accept `env_name` / `env_names` alongside the
  canonical `env_id` / `env_ids`. Same case-insensitive resolution path as
  wave 1; same mutual-exclusion model_validator; same
  `ENV_REF_BOTH_PROVIDED` / `ENV_REF_AMBIGUOUS` / `ENV_NOT_FOUND` errors.
- **`AgentContext.attached_env_names`** — friendly-name twin to
  `attached_env_ids` resolved by `_resolve_ctx` at every tool call. SDK
  exposes `default_env_names` and `attached_env_names` parameters.
- **Did-you-mean hints on `VALIDATION_FAILED` errors** — when a tool call
  fails Pydantic validation, the error surfaces as
  `[VALIDATION_FAILED] :: {hints, errors}` with friendly hints (alias map +
  Levenshtein ≤ 2 + SequenceMatcher ≥ 0.7) for `req`→`request`,
  `env`→`env_ids`, `q`→`query`, etc. Safe payload — never echoes
  `input_value`. Implemented as a `_tool_manager.call_tool` interception
  because FastMCP's `Tool` is a frozen Pydantic model.
- **`Example:` blocks on every MCP tool docstring** — paste-able JSON
  request bodies so agents see the canonical shape via `tools/list`.
- **Exploration-API tour** (`docs/exploration-tour.md`) — close-out of
  v0.10 exploration surface; documents the 4-step exploration workflow.
- **Bootstrap v2** (`memory-mcp-bootstrap.ps1` v2) — per-file env hint via
  `learnings.md` HTML comment, `mem-env: <name>` summary preamble, or
  `<task-folder>.env` sidecar (resolved lazily per file); second pass
  applies `rel_link` edges from intra-file `[See: <heading>]` and
  `learnings.md#<slug>` markers. Idempotent on re-run via per-section
  `mem-id` / `mem-sha256` comments. Multi-env aware.

### Changed
- `mcp_app.py`: every tool now resolves a `request` whose `env_names` /
  `env_ids` pair is validated via `_resolve_env_refs`; `_resolve_ctx`
  honors `attached_env_names`.
- `memory-mcp-schemas` bumped to `0.11.0` to match the server. The SDK
  (`memory-mcp-client`) stays at `0.10.0` — its v0.2 release ships in
  Phase 2.

### Notes
- Hard-delete on a memory whose body has already been replicated to a
  cold-storage backup is a workflow concern, not a substrate concern —
  the substrate has no way to reach into your backups. Operators that
  back up Postgres dumps should treat tombstone events as triggers to
  prune the affected memory id from the next dump cycle.
- MCP tool count: **58 → 59**.

## [0.10.0] — 2026-05-13 — Stats surface (API + Prometheus + Grafana dashboard)

### Added
- **`mem_stats` MCP tool** — read-only operational snapshot. Returns counts (total / active / superseded / retired / pinned, `by_env` with friendly env names, `by_kind`, `by_status`, `top_tags`), distributions (chain depth, body length, age, salience, access count, tags-per-memory — all with p50/p90/p99), v0.7-table snapshots (tasks / playbooks / decisions by status), opt-in substrate stats (`pg_database_size`, Qdrant points, Neo4j nodes/relationships), per-(sink, env) projection lag with friendly env names, outbox pending/dead aggregates, and process RSS (Linux). RBAC-aware (default scopes to caller's `attached_env_ids`; `global=true` admin-only). Honors v0.9 `env_names`/`env_ids` ergonomics.
- **Prometheus metric expansion** (12 new metrics): `mcp_memories_total{env,kind,status}`, `mcp_memories_pinned_total{env}`, `mcp_memories_body_bytes_total{env}`, histograms `mcp_memory_chain_depth`, `mcp_memory_age_seconds`, `mcp_memory_body_length_bytes`, `mcp_memory_salience`, `mcp_memory_access_count`, gauges `mcp_tasks_total{status}`, `mcp_playbooks_total{status}`, `mcp_decisions_total{status}`, and `process_resident_memory_bytes`. Cardinality-capped (~200 series for the `env × kind × status` matrix).
- **TTL-cached scrape refresh** — expensive distribution refresh gated by `MCP_METRICS_REFRESH_INTERVAL_SECONDS` (default 60s) so they do not recompute on every 15s Prometheus scrape.
- **Grafana dashboard** (`dashboards/memory-mcp.json`) — 12 panels in 3 rows: operational (tool QPS/latency), substrate (projection lag heatmap, outbox, dream pass, RSS), memory shape (counts by kind, chain-depth heatmap, age heatmap, body-length percentiles, tasks/playbooks/decisions).
- **Prometheus + Grafana compose profile** — `docker-compose.observability.yml` with `--profile observability` so default footprint is unchanged. Bring up with `docker compose --profile observability up`; open http://localhost:3000 (admin/admin).
- **Friendly env names on response sub-schemas** — `ProjectionLagEntry.env_name` added next to `env_id` (response-side ergonomics, parallel to the v0.9 wave-1 request-side work).

### Notes
- Statement-timeout-bounded queries (`SET LOCAL statement_timeout = '1500ms'`) protect the `SUM(octet_length(body))` and recursive chain-depth CTE on large memory stores; on timeout the response carries `total_body_bytes: null` with `total_body_bytes_approximate: true` (no fake partial sums — Postgres cancels the query).
- Substrate snapshots (`pg_database_size`, Qdrant, Neo4j) are **opt-in** via `include_substrates: true` to keep the default `mem_stats` call fast. Each substrate query is wrapped in per-sink try/except so one degraded backend never breaks the response.
- Process RSS is sourced from `/proc/self/statm` (Linux only). Non-Linux callers see `rss_bytes: null` with `rss_reason: "unsupported_os"`. No `resource.getrusage` fallback (different OSes report different units/semantics).
- MCP tool count: **57 → 58**.

## [0.9.0] — 2026-05-13 — Ergonomic env naming (wave 1)

### Added
- `MemorySearchRequest.env_names: list[str] | None` as a friendly alternative to `env_ids` for `mem_search`.
- Central `_resolve_env_refs` request resolver and case-insensitive `get_env_by_name_ci` helper for future wave-2 rollout.
- New env-reference error codes: `ENV_REF_BOTH_PROVIDED`, `ENV_REF_AMBIGUOUS`, and `ENV_NOT_FOUND`.

### Changed
- `mem_search` now resolves `env_names` server-side case-insensitively, then passes only UUIDs to downstream search logic.
- `env_ids` and `env_names` are mutually exclusive; providing both raises `ENV_REF_BOTH_PROVIDED` before authorization/attachment checks.
- MCP discovery for `mem_search` now shows a concrete `{"request": {"env_names": ["cdp"]}}` example. Wave 2 will apply the same pattern to the remaining env-referencing schemas.

### Added
- **`memory-mcp-client` Python package** — async Streamable-HTTP SDK wrapping the MCP tool surface with typed namespaces.
- **`memory-mcp-schemas` Python package** — shared Pydantic DTO/enums package extracted from the server for client/server schema parity.

### Changed
- Server modules now re-export DTOs from `memory_mcp_schemas`; no behavior changes. Server tests stay green (898 passing).

### Migration
- Internal-only package split; no breaking API changes.

## [0.8.0] — 2026-05-13 — Environment Operations

### Added
- 12 new MCP tools for env-as-a-unit operations: `env_export`, `env_import`, `env_diff`, `env_clone`, `env_merge`, `env_migrate`, `env_snapshot`, `env_restore`, `env_delete`, `env_rename`, `mem_copy`, `mem_move`.
- `memory-mcp-admin` CLI with 12 subcommands across `env` and `mem` groups. Connects to a running memory-mcp instance over Streamable HTTP using token-based auth (CLI flag, env var, or `~/.memory-mcp/config.toml`).
- Python SDK extension: `client.env_ops.*` namespace + `client.memories.copy()` / `.move()` methods.
- `snapshots` table (alembic migration 0014) for tracking labeled snapshot archives at `<data_root>/snapshots/<env_id>/<snapshot_id>.memarchive.tar.gz`.
- Soft-delete columns on `environments` (migration 0013): `status` enum (`active`|`deleted`), `deleted_at` timestamp. The env UUID remains valid forever as a lineage anchor.
- `RemapTable` schema (`memory_mcp_schemas.env_ops`) for tracking UUID remapping across cross-env operations.
- `MemoryVectorRecord` schema for portable per-memory embedding records keyed by (`memory_id`, `vector_name`) with `memory_version` for staleness detection.

### Changed
- `env_get` now accepts `include_deleted: bool = False` to surface soft-deleted environments.
- `MemoryUpdatePatch` deliberately still omits `env_id` (cross-env moves go through `mem_move`, not `mem_update`).
- `mem_supersede`'s `CROSS_ENV_SUPERSEDE_BLOCKED` guard remains in place; `mem_move` bypasses it via direct UPDATE inside its own transaction.
- `entity_merge` now atomically repoints relations and deletes orphan graph nodes when both merged entities have graph nodes (P3.5 hardening). Critical precondition for `env_merge` with entity collisions.

### Notes
- Default behaviors are conservative: `dry_run=True` for imports, `include_grants=False`, `include_dream_history=False`, `allow_bulk_reembed=False`.
- v0.8 is tagged experimental; archive format `schema_version` is `0.8.0`. Future versions reject unsafe imports unless explicitly forced.
- Snapshots are NOT auto-pruned; server warns when `data_root/snapshots/` exceeds 10 GB.
- Cross-env lineage edges entering a deleted env are dropped (not tombstoned) — see §13.1 for revisit in v0.9.

## [0.7.1] — 2026-05-12

### Added
- **`task_tree` MCP tool** — DFS pre-order indented view of a task subtree. Parameters: `task_id`, `max_depth=10`, `max_nodes=200`. Tool count: 44 → 45.
- **ADR `consequences` first-class field** on `DecisionMeta`. `adr_export` now renders a dedicated `## Consequences` section from this field (v0.7.0 placeholder cross-referencing Constraints removed). Back-compat: v0.7.0 decisions without the field still export correctly with `_(none recorded)_` placeholder.
- **Playbook `{{task:<uuid>}}` placeholders** — `playbook_invoke` now resolves task references inline (`"[task <short8>] <status>: <desc>"`). Unknown/cross-env refs leave the literal token in place and append to a new `missing_task_refs: list[UUID]` response field. UX aligned with existing `{{memory:<uuid>}}` behavior.
- **Dream-worker decision-conflict detector** — new `DreamMode.decision_conflicts` + new `DreamProposalKind.decision_conflict_candidate`. Scans accepted decisions per env; pairs whose body-vector cosine ≥ 0.85 surface in `dream_review`. Threshold env-overridable via `MEMORY_MCP_DECISION_CONFLICT_COSINE_THRESHOLD`. Pair-cap: 500 decisions per env per pass. Wired into scheduler and manual `dream_run`. New batch `QdrantVectorStore.get_vectors()` method.

### Hardening
- **Real-Postgres concurrency tests** — `tests/integration/` now exercises the `task_dep_link` cycle race (advisory lock) and `playbook` macro race (partial-unique-index) against testcontainers-postgres. Tests use a real `asyncio.Event` barrier monkeypatched into `_acquire_dep_lock` and `_ensure_macro_available` to guarantee simultaneous-critical-section execution. Iteration count env-overridable via `MEMORY_MCP_RACE_ITERATIONS` (default 20).
- **Manual `dream_run` dispatch fixed** — `_ALL_MODES` now includes `decision_conflicts`; `vector_store` correctly passed for both `dedupe` and `decision_conflicts`.

### Migrations
- **0012_dream_decision_conflicts** — extends `dream_runs.mode` and `dream_proposals.kind` CHECK constraints to allow new values. Downgrade includes cleanup of new-enum-value rows.

### Tests
- **858 unit tests** passing (+24 over v0.7.0's 834).
- **6 integration tests** passing (testcontainers-postgres, marker-gated).

## [0.7.0] — 2026-05-12

### Added — Procedures & Plans
- **Playbooks (F6):** new `MemoryKind.playbook` with `steps` and `macro` fields; per-env case-insensitive macro uniqueness. New `playbook_invoke` MCP tool for procedure execution.
- **Tasks (B1):** new `tasks` table + `:Task` graph nodes with cycle-safe `depends_on` edges (Postgres advisory lock + DFS). 7 new MCP tools: `task_create`, `task_substep`, `task_dep_link`, `task_status_set`, `task_link_memory`, `task_list`, `task_next`.
- **Decisions (B2):** new `decision_meta` JSONB on `MemoryKind.decision`. New `adr_export` MCP tool renders strict markdown ADR template (Title, Status, Context, Decision, Constraints, Consequences, Superseded By).
- **Context Pack F7 v2:** `mem_context_pack` now surfaces playbooks, tasks, and decisions alongside existing primitives. Canonical 7-section order. Token-based + Qdrant semantic playbook matching; in_progress→unblocked task ordering.
- **Tool count:** 35 → 44.

### Migrations
- 0009: `MemoryKind.playbook` enum + `steps`/`macro` columns + partial-unique index `ix_memories_macro_per_env`.
- 0010: `tasks` table + `graph_nodes.task_id` column + rebuilt `graph_nodes_exactly_one_target_chk` constraint.
- 0011: `decision_meta` JSONB column.

### Hardening
- Generic `relation_link` now delegates task→task `depends_on` to advisory-lock + cycle-check path; cannot bypass `task_dep_link` safety.
- 5 new tools (`task_substep`, `task_dep_link`, `task_status_set`, `task_link_memory`, `adr_export`) enforce strict `_require_env_attached` RBAC at the MCP wrapper layer.
- IntegrityError translation for macro uniqueness narrowed to the `ix_memories_macro_per_env` constraint; unrelated DB errors propagate cleanly.
- `task_link_memory` now idempotent via `INSERT ... ON CONFLICT DO NOTHING RETURNING`.
- `adr_export` wraps malformed `decision_meta` with a clean `InvalidInputError`.
- F7 v2 playbook matching extended to 2-char tokens and full-task-desc substring fallback.

### Known limitations (deferred to v0.7.1)
- Real-Postgres concurrency tests for advisory-lock and macro-unique races (currently mocked).
- ADR `Consequences` section renders a `_(see Constraints — separate consequences field planned for v0.7.1)_` placeholder until a dedicated `consequences` field is added to `DecisionMeta`.
- Dream-worker decision-conflict hook (deferred from v0.7 scope).

### Tests
- 834 unit tests passing (+99 over v0.6.0's 735).

## [0.6.0] — 2026-05-12

### Added
- Added `mem_auto_context(task_desc, env_id, top_k=8)` to surface task-relevant memories without explicit search, backed by the new `trigger_description` field and a per-env `trigger` named Qdrant vector.
- Added optional `trigger_description: str | None` on memories to describe when a memory should apply; memories without one are skipped by `mem_auto_context` but remain searchable through `mem_search`.
- Added `mem_digest(env_id, since_ts?)` to synthesize and store a six-section digest (`brief`, `active_context`, `system_patterns`, `tech_context`, `progress`, `open_questions`) via the dream worker LLM backend, with deterministic `source_type="digest-template"` fallback when the LLM is unavailable.
- Added `mem_resume(env_id, journal_tail=20)` to return the latest digest plus recent journal entries as the single session-start tool call.
- Added `mem_context_pack(task_desc, env_id, token_budget=4000, include_journal=True)` to compose digest, trigger-matched, recent journal, and salience-ranked archival context within a token budget, with section caps of 25% / 40% / 20% / 15% and proportional redistribution for missing sections.
- Added MCP-resource-style `kind="session_digest"` memories with structured six-section bodies.
- Added the `has_trigger_description` Qdrant payload index for efficient trigger-filter scans.

### Changed
- Changed per-env Qdrant collections to named vectors (`body` + `trigger`); `ensure_env_collection` now detects legacy v0.5 single-vector collections, recreates them with named vectors, and backfills body embeddings from Postgres with no operator action required.

### Fixed
- Fixed RBAC enforcement for `mem_auto_context`; it now resolves agent context and calls `rbac.require("read", env_id)` before searching.
- Fixed `mem_context_pack` to apply the same default visibility policy as `mem_search`, preventing archived or retired memory leakage.
- Fixed `mem_context_pack` trigger-search failures so the trigger section is skipped and the remaining budget is redistributed instead of crashing the pack.
- Fixed the F5 digest loader to SQL-limit candidates to the top 200 by salience/recency instead of materializing the entire environment.
- Fixed F5 LLM-output validation so empty `brief` or `active_context` sections fall back to the template summarizer.
- Fixed migration `0008` downgrade to remove only values introduced by that revision.

## [0.5.0] — 2026-05-11

### Added
- Added opt-in `mode="auto"` for `mem_search`; UUID-shaped queries dispatch to id lookup, other queries dispatch to hybrid, and `effective_mode` reports the dispatched mode. Default remains `hybrid` in v0.5.
- Added `MCP_TRANSPORT=stdio|http` in `config.py`; stdio uses `FastMCP.run_stdio_async()` while still requiring Postgres, Qdrant, and Neo4j backing services.
- Added stdio JSON client-config snippets for Claude Desktop and Copilot CLI under `client-config-snippets/`.

### Changed
- Changed `docker-compose.dev.yml` to use the `memory-mcp-server` entrypoint instead of hardcoded `uvicorn`, so `MCP_TRANSPORT` works consistently across HTTP and stdio in dev.

### Fixed
- Fixed `mode="auto"` resolving to id lookup so it now correctly populates the id-lookup request path and returns the matching memory.

## [0.4.0] — 2026-05-11

### Added
- Added a 3-prompt system-prompt cookbook (`SYSTEM_PROMPTS.md`) covering personal memory, multi-agent collaboration, and project memory patterns.
- Added an upstream `@modelcontextprotocol/server-memory` JSONL importer (`scripts/import_from_server_memory.py`) with idempotency via lex-search plus content-hash, backed by companion migration `0006_import_source_type`.
- Added optional `source_type`, `source_ref`, and `evidence_span` fields on `mem_write` for provenance tracking when the source is known.
- Added a migration guide section in `USAGE.md` for users coming from upstream `server-memory`.
