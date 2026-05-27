"""Recount pass — canonical writer of per-memory reference counters
and (Phase 1e, gated) per-kind authority weights.

The four ``reference_count_*`` columns added in Migration 0017 are
maintained at *transactional* truth by three Postgres triggers
(``memories_bump_on_relation_change``,
``memories_bump_on_lineage_change``, ``memories_status_flip_decrement``).
Phase 1e (Migration 0018) adds four parallel ``ref_authority_*``
NUMERIC(18,6) columns whose canonical truth is ``Σ source.salience``
per inbound citation kind. Triggers do NOT maintain these — citer
salience changes continuously (decay pass, access bumps, recount
itself) and any trigger would either fan out updates O(N edges) per
salience write or carry stale values. Authority lives entirely in
the recount pass, gated on
``Settings.dream_popularity_authority_weighted``.

Triggers are cheap, narrow, and exact for the cases they cover — but by
design they cannot handle three things at transaction time:

1. **Playbook macro citations.** A playbook's ``steps[]`` array embeds
   ``{{memory:<uuid>}}`` macros that resolve to memory bodies at
   ``playbook_invoke`` time. There is no edge row to fire on — the
   reference lives in free text. Triggers can't text-scan inside an
   ``UPDATE`` cheaply, and even if they could the cost would land on
   every step-array edit. Recount does the scan once per cadence.

2. **Supersede-chain ancestry exclusion.** A ``rel_link`` from a memory
   ``M'`` (which superseded ``M``) back to ``M`` (or vice versa) is
   structurally bookkeeping, not authority signal. Triggers would have
   to walk an unbounded ancestor chain on every edge insert — too
   expensive in the hot path. Recount walks the chain index once per
   env and applies the exclusion in a single sweep.

3. **Drift reconciliation.** Trigger bugs, partial replication of edges
   from an external loader, manual ``DELETE`` outside the trigger path,
   migration backfill skipped above the 100k-edge fast-path threshold —
   anything that lands counters out of sync with the canonical edge
   tables gets reconciled here. Recount is **idempotent**: re-running
   produces the same canonical values.

Output
------

For each env, the pass produces fresh totals for all four counters per
memory and issues atomic per-memory ``UPDATE`` statements. The
``RecountPassResult`` payload exposes how many memories were examined,
how many were adjusted (drift was non-zero), and the per-counter drift
totals so observability can flag environments where the triggers are
losing track of changes.

When ``Settings.dream_popularity_authority_weighted`` is true, the
pass additionally:

1. Recomputes ``ref_authority_*`` per kind = ``Σ source.salience`` of
   inbound citations.
2. Blends new values with current under
   ``dream_popularity_authority_damping`` (default 1.0 = no-op).
3. Stamps ``authority_last_recount_at`` on every adjusted target.

Authority writes use ``Decimal`` math and ``_q()``-quantize at the
NUMERIC(18,6) precision boundary; idempotency requires that.

Salience-recompute (R-B3 staleness fix) runs once per pass for **any**
memory whose integer counters OR authority columns drifted, gated only
on there being targets to recompute. This is intentional: the
references term in the salience formula (Phase 1) consumes the integer
counters, so a counter drift correction must be paired with a salience
write or stored salience goes stale immediately. Salience writes go
through the ``memory_update`` API so the standard outbox event fires
and the Qdrant payload (which embeds ``salience``) stays consistent.

Concurrency contract
--------------------

The pass is single-writer for the env (dispatcher holds a per-(mode,
env) advisory lock; see Scope below). Inside one invocation:

* **Integer counters** — single read, single write per drifted row.
  Triggers fire on other transactions during the pass; their effects
  show up in *the next* recount cycle. Within-pass consistency is
  guaranteed only against the read snapshot.
* **Authority leg** — citer salience is read **once** at the top of
  the leg via :func:`_load_active_memory_salience`. A citer whose
  salience the *same* pass recomputes via R-B3 will have its new
  salience visible to authority sums only on the **next** pass. This
  is Jacobi-style eventual convergence — NOT within-pass
  consistency — and is deliberate. The alternative (re-reading citer
  salience per target) introduces order-dependent recurrence: target
  A's authority would depend on whether target B was processed first
  this pass.
* **Salience recompute** — runs **after** the authority writes within
  the same pass, on the union of (integer-drifted, authority-drifted)
  rows. Goes through :func:`memory_update` with optimistic-lock
  ``expected_version``; ``VersionConflictError`` is counted on
  ``salience_version_conflicts`` and skipped — the next pass will
  pick the row up.

The contract favors safe progress over perfect simultaneity. In a
healthy env both legs converge in O(1) passes; in a degenerate env
(many concurrent writers) the eventual-convergence model still
terminates because reads at the top of each pass break dependency
cycles.

Scope
-----

Per call: one env. The dispatcher (:mod:`dream_worker.jobs`) holds the
per-(mode, env) advisory lock so concurrent recount + decay /
recount + dedupe on the *same env* are serialized for safety. Different
envs run in parallel.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

from sqlalchemy import bindparam, text

from memory_mcp.config import Settings
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import LineageRelation
from memory_mcp.dream.salience import (
    SalienceInputs,
    compute_salience,
    salience_weights_from_settings,
)
from memory_mcp.errors import VersionConflictError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryUpdatePatch, memory_update

log = logging.getLogger(__name__)


# Mirror of ``memory_mcp.playbooks.api._PLACEHOLDER_RE`` — duplicated
# here so the recount pass does not import a tool-facing module (clean
# layering: dream → schemas/db; tools → dream is fine).
_PLAYBOOK_MACRO_RE = re.compile(r"\{\{memory:([0-9a-f-]{36})\}\}", re.IGNORECASE)


# Lineage relation whitelist — the same set Migration 0017's triggers
# and backfill enforce. ``supersedes`` is intentionally absent so
# version-chain bookkeeping does not inflate parent authority. Phase 3
# (decompose) values ``split_from`` / ``derived_from`` are listed even
# though no rows carry them yet — listing them now means Phase 3 needs
# only a CHECK-constraint widening, not a recount rewrite.
_LINEAGE_WHITELIST: frozenset[str] = frozenset(
    {
        LineageRelation.summarized_from.value,
        LineageRelation.promoted_from.value,
        # Forward-listed; ``derives_from`` is a 0017-era speculative entry
        # not yet in the CHECK constraint or any concrete operation.
        # ``derived_from`` is Phase 3's derive-mode relation and bumps
        # legitimately. ``split_from`` (Phase 3 split-mode) was previously
        # forward-listed here but Migration 0021 removed it from the
        # load-bearing whitelist (rows of that relation must not bump
        # ``reference_count_lineage`` — the parent is retired).
        "derives_from",
        "derived_from",
    }
)


# Phase 4's auto-wire predicate — recount must skip it for the same
# reason the triggers do (popularity must not feed back into itself).
_AUTO_WIRE_PREDICATE = "related_to_popular"


# Phase 1e — Decimal precision boundary for authority math. Matches
# ``ref_authority_*`` columns (Migration 0018 NUMERIC(18,6)). All
# Decimal values flowing through aggregation, blend, and equality
# comparisons are quantized here so float-representation surprises do
# not produce spurious drift between passes (and idempotency-at-α=1.0
# holds exactly).
_AUTHORITY_QUANT: Decimal = Decimal("0.000001")


def _q(value) -> Decimal:
    """Quantize a numeric value to NUMERIC(18,6) precision.

    Uses ``Decimal(str(value))`` (not ``Decimal(value)``) for floats so
    the IEEE-754 binary representation does not leak into the quantized
    result. ``None`` is treated as zero — convenient for reading
    nullable columns.
    """

    if value is None:
        return Decimal("0").quantize(_AUTHORITY_QUANT)
    if isinstance(value, Decimal):
        return value.quantize(_AUTHORITY_QUANT)
    return Decimal(str(value)).quantize(_AUTHORITY_QUANT)


@dataclass(frozen=True)
class RecountPassResult:
    """Per-pass observability payload.

    The drift counters report **net adjustments** applied across all
    memories in the env. A healthy env with the triggers in place will
    report drift near zero — non-zero drift is signal of (a) a freshly
    landed migration that skipped the fast-path backfill, (b) a
    playbook that was edited without the recount pass running yet, or
    (c) a bug in one of the trigger functions.

    Phase 1e fields are zero-valued when
    ``Settings.dream_popularity_authority_weighted`` is false (the
    authority leg short-circuits and no salience-recompute is forced
    by authority drift). ``memories_salience_recomputed`` still
    increments when integer-counter drift forces salience updates
    (R-B3) — that happens regardless of the authority knob.
    """

    env_id: UUID
    memories_examined: int = 0
    memories_adjusted: int = 0
    drift_rel_link: int = 0
    drift_lineage: int = 0
    drift_task: int = 0
    drift_playbook: int = 0
    playbooks_scanned: int = 0
    duration_seconds: float = 0.0
    # ---- Phase 1e (v0.14.1) — authority leg observability ----------
    memories_authority_adjusted: int = 0
    memories_salience_recomputed: int = 0
    salience_version_conflicts: int = 0
    drift_authority_rel_link: Decimal = Decimal("0")
    drift_authority_lineage: Decimal = Decimal("0")
    drift_authority_task: Decimal = Decimal("0")
    drift_authority_playbook: Decimal = Decimal("0")
    # ---- Phase 1e-d (v0.14.1) — formula-version backfill observability ----
    # Active rows re-stamped this cycle because their stored
    # ``salience_formula_version`` was behind
    # ``Settings.dream_salience_formula_version``. After a formula bump
    # (any change to ``compute_salience`` math), expect this to be
    # non-zero for several cycles until the backlog drains.
    memories_formula_version_restamped: int = 0
    # Active rows still behind ``target_version`` after this cycle —
    # bounded by ``Settings.dream_recount_salience_recompute_cap``. A
    # non-zero value here means the next cycle will pick up more work.
    memories_formula_version_pending: int = 0


@dataclass(frozen=True)
class _CountBundle:
    """Per-memory canonical counter tuple."""

    rel_link: int = 0
    lineage: int = 0
    task: int = 0
    playbook: int = 0

    def with_rel_link(self, n: int) -> "_CountBundle":
        return _CountBundle(n, self.lineage, self.task, self.playbook)

    def with_lineage(self, n: int) -> "_CountBundle":
        return _CountBundle(self.rel_link, n, self.task, self.playbook)

    def with_task(self, n: int) -> "_CountBundle":
        return _CountBundle(self.rel_link, self.lineage, n, self.playbook)

    def with_playbook(self, n: int) -> "_CountBundle":
        return _CountBundle(self.rel_link, self.lineage, self.task, n)


@dataclass(frozen=True)
class _AuthorityBundle:
    """Per-memory canonical authority tuple (Phase 1e).

    Values are ``Decimal``-typed throughout to preserve NUMERIC(18,6)
    precision. ``task`` is fixed at zero by design (D1 — task sources
    have no salience column; the structural ``reference_count_task``
    integer counter carries that signal).
    """

    rel_link: Decimal = Decimal("0")
    lineage: Decimal = Decimal("0")
    task: Decimal = Decimal("0")
    playbook: Decimal = Decimal("0")

    def with_rel_link(self, v: Decimal) -> "_AuthorityBundle":
        return _AuthorityBundle(v, self.lineage, self.task, self.playbook)

    def with_lineage(self, v: Decimal) -> "_AuthorityBundle":
        return _AuthorityBundle(self.rel_link, v, self.task, self.playbook)

    def with_task(self, v: Decimal) -> "_AuthorityBundle":
        return _AuthorityBundle(self.rel_link, self.lineage, v, self.playbook)

    def with_playbook(self, v: Decimal) -> "_AuthorityBundle":
        return _AuthorityBundle(self.rel_link, self.lineage, self.task, v)


def _quantized_eq(a: _AuthorityBundle, b: _AuthorityBundle) -> bool:
    """Equality at NUMERIC(18,6) precision (rounds away float noise)."""

    return (
        _q(a.rel_link) == _q(b.rel_link)
        and _q(a.lineage) == _q(b.lineage)
        and _q(a.task) == _q(b.task)
        and _q(a.playbook) == _q(b.playbook)
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def run_recount(
    env_id: UUID,
    *,
    actor_ctx: AgentContext,
    settings: Settings,
    now: dt.datetime | None = None,
) -> RecountPassResult:
    """Reconcile per-memory reference counters against canonical edge tables.

    Reads ``relations``, ``memory_lineage``, and the ``steps`` array of
    every active playbook in ``env_id``. Builds the supersede-chain
    membership index from ``memories.superseded_by``. Issues one
    ``UPDATE`` per memory whose canonical totals differ from its
    current stored values.

    When ``settings.dream_popularity_authority_weighted`` is true,
    additionally recomputes ``ref_authority_*`` per kind (Phase 1e).

    Always runs the R-B3 salience-recompute leg for any row whose
    integer counters OR authority columns drifted — keeps stored
    salience consistent with the canonical counters that drive the
    references / authority terms in the formula.
    """

    started = time.perf_counter()
    # D9: scope ``now_ts`` once at the top so both legs and the
    # salience-recompute step use the same wall clock — independent of
    # which legs end up running.
    now_ts = now or dt.datetime.now(dt.UTC)

    authority_enabled = bool(settings.dream_popularity_authority_weighted)
    damping = _q(settings.dream_popularity_authority_damping)

    # Set of memory ids whose stored salience needs to be recomputed.
    # Populated by (a) the integer-counter loop when it adjusts a row,
    # and (b) the authority loop (when knob is on) when it adjusts a
    # row. The R-B3 step at the end of the pass processes the union.
    salience_targets: set[UUID] = set()

    authority_adjusted = 0
    drift_auth_rl = Decimal("0")
    drift_auth_ln = Decimal("0")
    drift_auth_tk = Decimal("0")
    drift_auth_pb = Decimal("0")

    async with session_scope() as s:
        chain_of = await _build_chain_index(s, env_id)
        # Pre-load active-status set so rel_link can skip retired-src
        # edges without an extra round trip per edge.
        active_memory_ids = await _load_active_memory_ids(s, env_id)

        canonical: dict[UUID, _CountBundle] = defaultdict(_CountBundle)

        rl_counts, task_counts = await _count_relations(
            s,
            env_id=env_id,
            chain_of=chain_of,
            active_memory_ids=active_memory_ids,
        )
        for mid, n in rl_counts.items():
            canonical[mid] = canonical[mid].with_rel_link(n)
        for mid, n in task_counts.items():
            canonical[mid] = canonical[mid].with_task(n)

        ln_counts = await _count_lineage(s, env_id=env_id)
        for mid, n in ln_counts.items():
            canonical[mid] = canonical[mid].with_lineage(n)

        pb_counts, playbooks_scanned = await _count_playbook_macros(
            s, env_id=env_id, env_memory_ids=set(active_memory_ids),
        )
        for mid, n in pb_counts.items():
            canonical[mid] = canonical[mid].with_playbook(n)

        current = await _load_current_counters(s, env_id)

        # Union of "any side has a value" — memories absent from
        # ``canonical`` but present in ``current`` need a zero-reset, and
        # vice versa.
        all_ids = set(canonical.keys()) | set(current.keys())

        adjusted = 0
        drift_rl = drift_ln = drift_tk = drift_pb = 0
        for mid in all_ids:
            want = canonical.get(mid, _CountBundle())
            have = current.get(mid, _CountBundle())
            if want == have:
                continue
            drift_rl += want.rel_link - have.rel_link
            drift_ln += want.lineage - have.lineage
            drift_tk += want.task - have.task
            drift_pb += want.playbook - have.playbook
            await _apply_counters(s, mid, want)
            adjusted += 1
            # Counter change feeds the references term in the salience
            # formula; mark this row for the R-B3 recompute step.
            salience_targets.add(mid)

        # ---- Phase 1e (v0.14.1) — authority leg (knob-gated) -------
        if authority_enabled:
            # D7 — read citer salience ONCE at the top of the leg.
            # A citer whose salience this same pass recomputes via
            # R-B3 will have its new value reflected on the NEXT
            # pass. Re-reading per target would create
            # order-dependent recurrence.
            citer_salience = await _load_active_memory_salience(s, env_id)
            canonical_auth = await _recompute_authority(
                s,
                env_id=env_id,
                chain_of=chain_of,
                active_memory_ids=active_memory_ids,
                citer_salience=citer_salience,
            )
            current_auth = await _load_current_authority(s, env_id)

            all_auth_ids = set(canonical_auth.keys()) | set(current_auth.keys())
            for mid in all_auth_ids:
                want_canon = canonical_auth.get(mid, _AuthorityBundle())
                have_auth = current_auth.get(mid, _AuthorityBundle())
                want_blended = _blend(have_auth, want_canon, damping=damping)
                if _quantized_eq(want_blended, have_auth):
                    continue
                drift_auth_rl += _q(want_blended.rel_link - have_auth.rel_link)
                drift_auth_ln += _q(want_blended.lineage - have_auth.lineage)
                drift_auth_tk += _q(want_blended.task - have_auth.task)
                drift_auth_pb += _q(want_blended.playbook - have_auth.playbook)
                await _apply_authority(s, mid, want_blended, now_ts=now_ts)
                authority_adjusted += 1
                salience_targets.add(mid)

    # ---- Phase 1e-d formula-version backfill ------------------------
    # Pull active rows whose stored ``salience_formula_version`` lags
    # ``Settings.dream_salience_formula_version`` (bumped on every
    # ``compute_salience`` math change). Union into ``salience_targets``
    # so the next loop re-stamps both ``salience`` AND
    # ``salience_formula_version`` atomically through ``MemoryUpdatePatch``.
    # Bounded by ``Settings.dream_recount_salience_recompute_cap`` so a
    # single cycle doesn't pump unbounded audit / outbox rows on first
    # post-bump deploy.
    target_formula_version = settings.dream_salience_formula_version
    formula_cap = settings.dream_recount_salience_recompute_cap
    mismatched_ids = await _load_formula_version_mismatched(
        env_id=env_id,
        target_version=target_formula_version,
        cap=formula_cap,
    )
    salience_targets = salience_targets | mismatched_ids

    # ---- R-B3 salience recompute (always runs) -----------------------
    # Runs OUTSIDE the outer session so memory_update can open its own
    # session_scope and the integer-counter / authority writes from the
    # outer transaction are already committed. D6: also runs when the
    # authority knob is OFF — integer counter drift still feeds the
    # references term in the salience formula and stored salience must
    # not lag.
    salience_recomputed, salience_conflicts = await _recompute_salience_for(
        salience_targets,
        env_id=env_id,
        actor_ctx=actor_ctx,
        settings=settings,
        now=now_ts,
    )

    # Phase 1e-d — after the recompute leg, query the remaining backlog
    # for observability. Pending > 0 means the cap was hit (or version
    # conflicts left some rows behind); next cycle continues from where
    # this one stopped.
    formula_version_pending = await _count_formula_version_pending(
        env_id=env_id,
        target_version=target_formula_version,
    )

    result = RecountPassResult(
        env_id=env_id,
        memories_examined=len(all_ids),
        memories_adjusted=adjusted,
        drift_rel_link=drift_rl,
        drift_lineage=drift_ln,
        drift_task=drift_tk,
        drift_playbook=drift_pb,
        playbooks_scanned=playbooks_scanned,
        duration_seconds=time.perf_counter() - started,
        memories_authority_adjusted=authority_adjusted,
        memories_salience_recomputed=salience_recomputed,
        salience_version_conflicts=salience_conflicts,
        drift_authority_rel_link=drift_auth_rl,
        drift_authority_lineage=drift_auth_ln,
        drift_authority_task=drift_auth_tk,
        drift_authority_playbook=drift_auth_pb,
        memories_formula_version_restamped=len(mismatched_ids),
        memories_formula_version_pending=formula_version_pending,
    )

    log.info(
        "recount: env=%s examined=%d adjusted=%d drift(rl/ln/tk/pb)=%d/%d/%d/%d "
        "auth_adjusted=%d salience_recomputed=%d salience_conflicts=%d "
        "drift_auth(rl/ln/tk/pb)=%s/%s/%s/%s "
        "formula_version(target=%d restamped=%d pending=%d)",
        env_id,
        result.memories_examined,
        result.memories_adjusted,
        drift_rl, drift_ln, drift_tk, drift_pb,
        authority_adjusted, salience_recomputed, salience_conflicts,
        drift_auth_rl, drift_auth_ln, drift_auth_tk, drift_auth_pb,
        target_formula_version, len(mismatched_ids), formula_version_pending,
    )
    return result


# ---------------------------------------------------------------------------
# Supersede-chain index
# ---------------------------------------------------------------------------


async def _build_chain_index(s, env_id: UUID) -> dict[UUID, frozenset[UUID]]:
    """Map each memory_id → frozen set of memories in its supersede chain.

    Uses union-find over the ``superseded_by`` edges. A memory not
    involved in any supersede relation is absent from the returned
    dict (callers should default to ``frozenset()``). This keeps the
    common case — no supersede involvement — cheap on the lookup
    side and avoids one entry per memory in the index.
    """

    rows = (await s.execute(
        text(
            "SELECT id, superseded_by FROM memories "
            "WHERE env_id = :env_id AND superseded_by IS NOT NULL"
        ),
        {"env_id": env_id},
    )).all()

    if not rows:
        return {}

    parent: dict[UUID, UUID] = {}

    def find(x: UUID) -> UUID:
        root = x
        while parent.setdefault(root, root) != root:
            root = parent[root]
        # Path compression — keeps subsequent finds O(α(n)).
        cur = x
        while parent[cur] != root:
            parent[cur], cur = root, parent[cur]
        return root

    def union(a: UUID, b: UUID) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in rows:
        union(r.id, r.superseded_by)

    groups: dict[UUID, set[UUID]] = defaultdict(set)
    for m in list(parent.keys()):
        groups[find(m)].add(m)

    chain_of: dict[UUID, frozenset[UUID]] = {}
    for members in groups.values():
        frozen = frozenset(members)
        for m in members:
            chain_of[m] = frozen
    return chain_of


async def _load_active_memory_ids(s, env_id: UUID) -> set[UUID]:
    """Snapshot the env's active-memory id set.

    Used to enforce the status-flip-equivalent rule on rel_link edges:
    edges sourced from a retired/superseded memory should not count
    toward their target's authority. Mirrors the 0017 backfill's
    ``sm.status = 'active'`` filter.
    """

    rows = await s.execute(
        text(
            "SELECT id FROM memories "
            "WHERE env_id = :env_id AND status = 'active'"
        ),
        {"env_id": env_id},
    )
    return {r.id for r in rows}


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


async def _count_relations(
    s,
    *,
    env_id: UUID,
    chain_of: dict[UUID, frozenset[UUID]],
    active_memory_ids: set[UUID],
) -> tuple[dict[UUID, int], dict[UUID, int]]:
    """Return (rel_link_counts, task_counts) per dst memory.

    Filters mirror Migration 0017's backfill plus the supersede-chain
    ancestry exclusion (S6) which the triggers cannot afford. Edges
    from a retired/superseded src memory are skipped — the status-flip
    trigger would have decremented them, so canonical truth excludes
    them.
    """

    # ``UNION ALL`` keeps the executor honest: we need a row per edge
    # so the Python loop can apply the chain-exclusion check. A
    # group-by in SQL would force us to push the exclusion into SQL
    # too (a recursive CTE), which is harder to test and only pays
    # off above ~50k edges per env — beyond Phase 1's target scale.
    stmt = text(
        """
        SELECT
            r.id              AS rel_id,
            r.type            AS rel_type,
            gn_src.node_type  AS src_node_type,
            gn_src.memory_id  AS src_memory_id,
            gn_dst.memory_id  AS dst_memory_id
        FROM relations r
        JOIN graph_nodes gn_src ON gn_src.id = r.src_node_id
        JOIN graph_nodes gn_dst ON gn_dst.id = r.dst_node_id
        WHERE r.env_id = :env_id
          AND gn_dst.memory_id IS NOT NULL
        """
    )
    rows = (await s.execute(stmt, {"env_id": env_id})).all()

    rl: dict[UUID, int] = defaultdict(int)
    tk: dict[UUID, int] = defaultdict(int)

    for r in rows:
        if r.rel_type == _AUTO_WIRE_PREDICATE:
            continue
        dst = r.dst_memory_id
        # Task source → task counter (no ancestry check; tasks don't
        # supersede).
        if r.src_node_type == "task":
            tk[dst] += 1
            continue
        # Memory or entity source → rel_link counter.
        if r.src_node_type == "memory":
            # Skip if src memory is retired / superseded — its outgoing
            # citations were decremented by the status-flip trigger and
            # must remain decremented in canonical truth.
            if r.src_memory_id not in active_memory_ids:
                continue
            # Ancestry exclusion: if src and dst are in the same
            # supersede chain, this edge is bookkeeping, not authority.
            dst_chain = chain_of.get(dst)
            if dst_chain is not None and r.src_memory_id in dst_chain:
                continue
        rl[dst] += 1

    return rl, tk


async def _count_lineage(s, *, env_id: UUID) -> dict[UUID, int]:
    """Count per-parent lineage citations restricted to the whitelist.

    Mirrors Migration 0017's lineage backfill: child must be ``active``
    and ``relation`` must be in the load-bearing whitelist.
    """

    # ``IN :whitelist`` requires an expanding bindparam — set on the
    # statement so SQLAlchemy emits a proper IN clause.
    stmt = text(
        """
        SELECT ml.parent_memory_id AS parent_id, count(*) AS n
          FROM memory_lineage ml
          JOIN memories child ON child.id = ml.child_memory_id
         WHERE child.env_id = :env_id
           AND child.status = 'active'
           AND ml.relation IN :whitelist
         GROUP BY ml.parent_memory_id
        """
    ).bindparams(bindparam("whitelist", expanding=True))

    rows = (await s.execute(
        stmt,
        {"env_id": env_id, "whitelist": list(_LINEAGE_WHITELIST)},
    )).all()
    return {r.parent_id: int(r.n) for r in rows}


async def _count_playbook_macros(
    s, *, env_id: UUID, env_memory_ids: set[UUID],
) -> tuple[dict[UUID, int], int]:
    """Scan active playbook ``steps[]`` arrays for ``{{memory:<uuid>}}`` macros.

    Each occurrence in any step of any active playbook increments the
    target's playbook counter — duplicates across steps within the same
    playbook all count (a playbook that cites the same memory three
    times signals stronger structural use than one that cites it once).

    Cross-env citations are silently dropped: the playbook's env owns
    the count, and a macro that resolves to a memory in another env
    appears in ``missing_refs`` at invocation time, not as
    authority signal.
    """

    rows = (await s.execute(
        text(
            "SELECT id, steps FROM memories "
            "WHERE env_id = :env_id "
            "  AND kind = 'playbook' "
            "  AND status = 'active' "
            "  AND steps IS NOT NULL"
        ),
        {"env_id": env_id},
    )).all()

    counts: dict[UUID, int] = defaultdict(int)
    playbooks = 0
    for r in rows:
        playbooks += 1
        for step in r.steps or []:
            for match in _PLAYBOOK_MACRO_RE.finditer(step):
                try:
                    target = UUID(match.group(1))
                except (ValueError, TypeError):
                    continue
                # Cross-env macros are caller-error at invocation time;
                # exclude them from popularity so a cross-env macro
                # cannot inflate a memory in another env. The triggers
                # don't run here anyway; this is recount-only logic.
                if target not in env_memory_ids:
                    continue
                counts[target] += 1
    return counts, playbooks


# ---------------------------------------------------------------------------
# Drift application
# ---------------------------------------------------------------------------


async def _load_current_counters(
    s, env_id: UUID,
) -> dict[UUID, _CountBundle]:
    """Snapshot current ``reference_count_*`` columns for env.

    The bundle includes only the four base counters — the computed
    ``reference_count`` sum is recalculated automatically by Postgres.
    """

    rows = (await s.execute(
        text(
            "SELECT id, reference_count_rel_link, reference_count_lineage, "
            "       reference_count_task, reference_count_playbook "
            "FROM memories "
            "WHERE env_id = :env_id"
        ),
        {"env_id": env_id},
    )).all()
    return {
        r.id: _CountBundle(
            rel_link=int(r.reference_count_rel_link or 0),
            lineage=int(r.reference_count_lineage or 0),
            task=int(r.reference_count_task or 0),
            playbook=int(r.reference_count_playbook or 0),
        )
        for r in rows
    }


async def _apply_counters(
    s, memory_id: UUID, want: _CountBundle,
) -> None:
    """Atomically overwrite a memory's four reference counters.

    The computed ``reference_count`` column updates automatically.
    Bumps to counters do NOT trigger the status-flip trigger (which
    guards on ``UPDATE OF status``), so this UPDATE is safe to issue
    without re-entrancy concerns.
    """

    await s.execute(
        text(
            "UPDATE memories SET "
            "  reference_count_rel_link = :rl, "
            "  reference_count_lineage  = :ln, "
            "  reference_count_task     = :tk, "
            "  reference_count_playbook = :pb "
            "WHERE id = :mid"
        ),
        {
            "rl": want.rel_link,
            "ln": want.lineage,
            "tk": want.task,
            "pb": want.playbook,
            "mid": memory_id,
        },
    )


# ---------------------------------------------------------------------------
# Phase 1e — authority leg
# ---------------------------------------------------------------------------


async def _load_active_memory_salience(s, env_id: UUID) -> dict[UUID, Decimal]:
    """Snapshot ``(id -> salience)`` for every active memory in ``env_id``.

    Used as the citer-salience lookup for :func:`_recompute_authority`.
    Read ONCE at the top of the authority leg (D7) to avoid
    order-dependent recurrence: a target's authority is computed from
    the salience values that existed at pass start, not the values
    that the same pass is in the middle of recomputing.

    Retired / superseded rows are excluded — they cannot contribute
    citer salience, mirroring the status-flip exclusion that the
    integer-counter triggers apply.
    """

    rows = await s.execute(
        text(
            "SELECT id, salience FROM memories "
            "WHERE env_id = :env_id AND status = 'active'"
        ),
        {"env_id": env_id},
    )
    return {r.id: _q(r.salience) for r in rows}


async def _recompute_authority(
    s,
    *,
    env_id: UUID,
    chain_of: dict[UUID, frozenset[UUID]],
    active_memory_ids: set[UUID],
    citer_salience: dict[UUID, Decimal],
) -> dict[UUID, _AuthorityBundle]:
    """Aggregate ``Σ source.salience`` per inbound citation kind.

    Walks the same edge sets as :func:`_count_relations`,
    :func:`_count_lineage`, and :func:`_count_playbook_macros`. The
    integer-counter rules carry over to authority unchanged with one
    documented asymmetry (D8: self-citation).

    Exclusion rules — citations dropped from the authority sum:

    1. **Auto-wire predicate** (``related_to_popular``). Phase 4
       wire-up must not feed back into the popularity signal that
       produced it.
    2. **Retired / superseded source memory.** Mirrors the
       integer-counter's ``active_memory_ids`` filter — a citer that
       isn't visible in projections can't contribute authority.
    3. **Supersede-chain ancestry.** A ``rel_link`` from a memory in
       the dst's supersede chain is bookkeeping, not authority.
    4. **Self-citation (D8 — asymmetric with integer counter).**
       ``src_memory_id == dst_memory_id`` excluded. Once slice 1e-d
       wires authority into the salience formula, allowing self-cites
       would create a fixed-point feedback loop: target salience
       depends on its own authority depends on its own salience. The
       integer counter may include self-cites in its raw total
       (incidentally caught only by the chain-ancestry sweep in
       limited cases); the Phase 1 contract for the integer counter
       is not changed.
    5. **Cross-env playbook macro target.** A macro that resolves to
       a memory in a different env appears in ``missing_refs`` at
       invocation time and never contributes authority.
    6. **Task source** (D1). Tasks have no ``salience`` column. The
       structural ``reference_count_task`` integer counter already
       carries the task-source signal; ``ref_authority_task`` stays
       fixed at zero. Future work: revisit if tasks gain salience.

    All values are quantized to NUMERIC(18,6) precision via :func:`_q`
    so subsequent equality comparisons and idempotency checks are
    exact.
    """

    bundles: dict[UUID, _AuthorityBundle] = defaultdict(_AuthorityBundle)

    # --- rel_link authority ---------------------------------------------
    rl_stmt = text(
        """
        SELECT
            r.type            AS rel_type,
            gn_src.node_type  AS src_node_type,
            gn_src.memory_id  AS src_memory_id,
            gn_dst.memory_id  AS dst_memory_id
        FROM relations r
        JOIN graph_nodes gn_src ON gn_src.id = r.src_node_id
        JOIN graph_nodes gn_dst ON gn_dst.id = r.dst_node_id
        WHERE r.env_id = :env_id
          AND gn_dst.memory_id IS NOT NULL
        """
    )
    rows = (await s.execute(rl_stmt, {"env_id": env_id})).all()

    rl_sum: dict[UUID, Decimal] = defaultdict(lambda: Decimal("0"))
    for r in rows:
        # (1) auto-wire predicate
        if r.rel_type == _AUTO_WIRE_PREDICATE:
            continue
        # (6) task source → contributes 0 to authority (D1)
        if r.src_node_type == "task":
            continue
        # Entity sources are also skipped — only memory-sourced citations
        # carry salience signal.
        if r.src_node_type != "memory":
            continue
        src = r.src_memory_id
        dst = r.dst_memory_id
        # (2) retired / superseded source
        if src not in active_memory_ids:
            continue
        # (3) supersede-chain ancestry
        dst_chain = chain_of.get(dst)
        if dst_chain is not None and src in dst_chain:
            continue
        # (4) self-citation (D8)
        if src == dst:
            continue
        rl_sum[dst] = rl_sum[dst] + citer_salience.get(src, Decimal("0"))

    for mid, total in rl_sum.items():
        bundles[mid] = bundles[mid].with_rel_link(_q(total))

    # --- lineage authority ----------------------------------------------
    # Child salience contributes to parent. Mirror integer counter:
    # child.status='active' filter is enforced via citer_salience
    # (which only contains active rows); the WHERE clause keeps
    # parity for the query plan.
    ln_stmt = text(
        """
        SELECT ml.parent_memory_id AS parent_id,
               ml.child_memory_id  AS child_id
          FROM memory_lineage ml
          JOIN memories child ON child.id = ml.child_memory_id
         WHERE child.env_id = :env_id
           AND child.status = 'active'
           AND ml.relation IN :whitelist
        """
    ).bindparams(bindparam("whitelist", expanding=True))
    rows = (await s.execute(
        ln_stmt,
        {"env_id": env_id, "whitelist": list(_LINEAGE_WHITELIST)},
    )).all()

    ln_sum: dict[UUID, Decimal] = defaultdict(lambda: Decimal("0"))
    for r in rows:
        # (4) self-citation guard — a memory listed as both child and
        # parent of itself is pathological; the integer counter would
        # technically count it but we exclude here for D8 consistency.
        if r.child_id == r.parent_id:
            continue
        ln_sum[r.parent_id] = ln_sum[r.parent_id] + citer_salience.get(
            r.child_id, Decimal("0"),
        )

    for mid, total in ln_sum.items():
        bundles[mid] = bundles[mid].with_lineage(_q(total))

    # --- playbook authority ---------------------------------------------
    # Each macro occurrence in a playbook's steps contributes the
    # playbook's salience to the target's playbook authority.
    # Duplicates within one playbook all count (mirror integer counter:
    # repeated structural use IS a stronger signal).
    pb_stmt = text(
        "SELECT id, steps FROM memories "
        "WHERE env_id = :env_id "
        "  AND kind = 'playbook' "
        "  AND status = 'active' "
        "  AND steps IS NOT NULL"
    )
    rows = (await s.execute(pb_stmt, {"env_id": env_id})).all()

    pb_sum: dict[UUID, Decimal] = defaultdict(lambda: Decimal("0"))
    for r in rows:
        pb_id = r.id
        pb_sal = citer_salience.get(pb_id, Decimal("0"))
        for step in r.steps or []:
            for match in _PLAYBOOK_MACRO_RE.finditer(step):
                try:
                    target = UUID(match.group(1))
                except (ValueError, TypeError):
                    continue
                # (5) cross-env playbook macro
                if target not in active_memory_ids:
                    continue
                # (4) self-citation (D8) — a playbook citing itself
                if target == pb_id:
                    continue
                pb_sum[target] = pb_sum[target] + pb_sal

    for mid, total in pb_sum.items():
        bundles[mid] = bundles[mid].with_playbook(_q(total))

    # task remains 0 by default — D1.
    return bundles


async def _load_current_authority(
    s, env_id: UUID,
) -> dict[UUID, _AuthorityBundle]:
    """Snapshot current ``ref_authority_*`` columns for env.

    The generated ``reference_authority`` total is recomputed by
    Postgres on every UPDATE; we never read it directly.
    """

    rows = (await s.execute(
        text(
            "SELECT id, ref_authority_rel_link, ref_authority_lineage, "
            "       ref_authority_task, ref_authority_playbook "
            "FROM memories "
            "WHERE env_id = :env_id"
        ),
        {"env_id": env_id},
    )).all()
    return {
        r.id: _AuthorityBundle(
            rel_link=_q(r.ref_authority_rel_link),
            lineage=_q(r.ref_authority_lineage),
            task=_q(r.ref_authority_task),
            playbook=_q(r.ref_authority_playbook),
        )
        for r in rows
    }


def _blend(
    current: _AuthorityBundle,
    canonical: _AuthorityBundle,
    *,
    damping: Decimal,
) -> _AuthorityBundle:
    """Damped blend: ``new = (1 - α) · current + α · canonical``.

    With α = 1.0 (default per ``Settings.dream_popularity_authority_damping``)
    the blend reduces to the canonical value — no-op damping. At α < 1.0
    the value approaches canonical monotonically over successive passes;
    the function is *convergent* but no longer *idempotent* at a single
    pass — tests must use α = 1.0 to assert idempotency (D4).

    All arithmetic happens in Decimal; the result is quantized to
    NUMERIC(18,6).
    """

    if damping >= Decimal("1.0"):
        return _AuthorityBundle(
            rel_link=_q(canonical.rel_link),
            lineage=_q(canonical.lineage),
            task=_q(canonical.task),
            playbook=_q(canonical.playbook),
        )
    keep = Decimal("1") - damping
    return _AuthorityBundle(
        rel_link=_q(keep * current.rel_link + damping * canonical.rel_link),
        lineage=_q(keep * current.lineage + damping * canonical.lineage),
        task=_q(keep * current.task + damping * canonical.task),
        playbook=_q(keep * current.playbook + damping * canonical.playbook),
    )


async def _apply_authority(
    s,
    memory_id: UUID,
    want: _AuthorityBundle,
    *,
    now_ts: dt.datetime,
) -> None:
    """Atomically overwrite the four ``ref_authority_*`` columns and
    bump ``authority_last_recount_at``.

    The generated ``reference_authority`` total is auto-updated by
    Postgres. Bumps to authority columns do NOT trigger the
    status-flip trigger (which guards on ``UPDATE OF status``), so
    this UPDATE is safe to issue without re-entrancy concerns.
    """

    await s.execute(
        text(
            "UPDATE memories SET "
            "  ref_authority_rel_link    = :rl, "
            "  ref_authority_lineage     = :ln, "
            "  ref_authority_task        = :tk, "
            "  ref_authority_playbook    = :pb, "
            "  authority_last_recount_at = :now_ts "
            "WHERE id = :mid"
        ),
        {
            "rl": want.rel_link,
            "ln": want.lineage,
            "tk": want.task,
            "pb": want.playbook,
            "now_ts": now_ts,
            "mid": memory_id,
        },
    )


async def _recompute_salience_for(
    memory_ids: set[UUID],
    *,
    env_id: UUID,
    actor_ctx: AgentContext,
    settings: Settings,
    now: dt.datetime,
) -> tuple[int, int]:
    """Recompute and persist salience for a set of memories (R-B3 fix).

    For each id, loads the latest inputs from the DB, calls
    :func:`compute_salience` with the current settings-derived
    weights, and patches the row through :func:`memory_update`. Going
    through the API (rather than direct ``UPDATE memories SET salience``)
    enqueues an outbox event so the Qdrant payload — which embeds
    ``salience`` (see ``db.vector.qdrant``) — stays consistent.

    Only **active** rows are recomputed; retired / archived rows are
    skipped because their salience does not surface in recall or
    decay decisions, and a write would generate wasted outbox traffic.

    Returns ``(recomputed_count, version_conflict_count)``. Conflicts
    are logged at DEBUG and silently skipped (mirrors decay.py:387) —
    the next recount pass will pick the row up if drift persists.
    """

    if not memory_ids:
        return 0, 0

    weights = salience_weights_from_settings(settings)
    target_formula_version = settings.dream_salience_formula_version
    recomputed = 0
    conflicts = 0

    async with session_scope() as s:
        rows = (await s.execute(
            text(
                "SELECT id, version, access_count, last_accessed_at, "
                "       confidence, pinned, negative_feedback_count, "
                "       verified_at, created_at, "
                "       reference_count_rel_link, reference_count_lineage, "
                "       reference_count_task, reference_count_playbook, "
                "       reference_authority "
                "FROM memories "
                "WHERE id = ANY(:ids) "
                "  AND env_id = :env_id "
                "  AND status = 'active'"
            ),
            {"ids": list(memory_ids), "env_id": env_id},
        )).all()

    for row in rows:
        inputs = SalienceInputs(
            access_count=int(row.access_count or 0),
            last_accessed_at=row.last_accessed_at,
            confidence=float(row.confidence or 0.0),
            pinned=bool(row.pinned),
            negative_feedback_count=int(row.negative_feedback_count or 0),
            verified_at=row.verified_at,
            created_at=row.created_at,
            reference_count_rel_link=int(row.reference_count_rel_link or 0),
            reference_count_lineage=int(row.reference_count_lineage or 0),
            reference_count_task=int(row.reference_count_task or 0),
            reference_count_playbook=int(row.reference_count_playbook or 0),
            # Phase 1e-d — formula now reads ``reference_authority``. When
            # the knob is OFF, ``weights.w_authority`` is 0 (set by
            # ``salience_weights_from_settings``) so the term zeros out;
            # the field still flows for symmetry / future-proofing.
            reference_authority=float(row.reference_authority or 0),
        )
        new_salience = compute_salience(inputs, now=now, weights=weights)
        try:
            await memory_update(
                row.id,
                MemoryUpdatePatch(
                    expected_version=row.version,
                    salience=new_salience,
                    # Phase 1e-d — stamp the formula version atomically
                    # with the salience write. Recount is the sole stamper
                    # (access-bump + decay leave the version unchanged on
                    # purpose). On the next recount cycle, a row with a
                    # stamped current version is no longer "behind" so
                    # the formula-version mismatch query won't pull it in.
                    salience_formula_version=target_formula_version,
                ),
                ctx=actor_ctx,
                settings=settings,
            )
        except VersionConflictError:
            conflicts += 1
            log.debug(
                "recount: salience version conflict on memory %s (env %s); skipping",
                row.id, env_id,
            )
            continue
        recomputed += 1

    return recomputed, conflicts


async def _load_formula_version_mismatched(
    *,
    env_id: UUID,
    target_version: int,
    cap: int,
) -> set[UUID]:
    """Phase 1e-d — load active memory IDs whose stored salience formula version
    is behind ``target_version``.

    Bounded by ``cap`` (``Settings.dream_recount_salience_recompute_cap``)
    to keep the first-cycle audit/outbox spike from blowing past sane
    limits when an operator bumps ``dream_salience_formula_version``.
    ``cap=0`` means unbounded (test-only).

    Order is deterministic — ``(created_at ASC, id ASC)`` — so multiple
    consecutive recount cycles make forward progress through the
    backlog without thrashing the same rows. Newer memories naturally
    inherit ``salience_formula_version`` equal to the current version
    on creation (server default ``0`` is bumped to current on first
    salience write); the backlog is the set of pre-bump rows whose
    formula-version stamp was set under an earlier setting.
    """

    if target_version <= 0:
        # version 0 = "any version is fine", short-circuit so we never
        # treat an unstamped pre-1e-d env as a mismatched backlog.
        return set()

    async with session_scope() as s:
        if cap == 0:
            stmt = text(
                "SELECT id FROM memories "
                "WHERE env_id = :env_id "
                "  AND status = 'active' "
                "  AND salience_formula_version < :v "
                "ORDER BY created_at ASC, id ASC"
            )
            params = {"env_id": env_id, "v": target_version}
        else:
            stmt = text(
                "SELECT id FROM memories "
                "WHERE env_id = :env_id "
                "  AND status = 'active' "
                "  AND salience_formula_version < :v "
                "ORDER BY created_at ASC, id ASC "
                "LIMIT :cap"
            )
            params = {"env_id": env_id, "v": target_version, "cap": cap}
        rows = (await s.execute(stmt, params)).all()

    return {row.id for row in rows}


async def _count_formula_version_pending(
    *,
    env_id: UUID,
    target_version: int,
) -> int:
    """Phase 1e-d — total active rows still behind ``target_version``.

    Used to populate ``RecountPassResult.memories_formula_version_pending``
    so operators have visibility into how many rows still need re-stamping
    after the cap-limited cycle finishes.
    """

    if target_version <= 0:
        return 0

    async with session_scope() as s:
        result = (await s.execute(
            text(
                "SELECT COUNT(*) AS n FROM memories "
                "WHERE env_id = :env_id "
                "  AND status = 'active' "
                "  AND salience_formula_version < :v"
            ),
            {"env_id": env_id, "v": target_version},
        )).scalar_one()

    return int(result or 0)


__all__ = [
    "RecountPassResult",
    "run_recount",
]
