"""Recount pass — canonical writer of per-memory reference counters.

The four ``reference_count_*`` columns added in Migration 0017 are
maintained at *transactional* truth by three Postgres triggers
(``memories_bump_on_relation_change``,
``memories_bump_on_lineage_change``, ``memories_status_flip_decrement``).
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
from uuid import UUID

from sqlalchemy import bindparam, text

from memory_mcp.config import Settings
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import LineageRelation
from memory_mcp.identity import AgentContext

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
        # Forward-listed for Phase 3; not yet in the enum / CHECK.
        "derives_from",
        "split_from",
        "derived_from",
    }
)


# Phase 4's auto-wire predicate — recount must skip it for the same
# reason the triggers do (popularity must not feed back into itself).
_AUTO_WIRE_PREDICATE = "related_to_popular"


@dataclass(frozen=True)
class RecountPassResult:
    """Per-pass observability payload.

    The drift counters report **net adjustments** applied across all
    memories in the env. A healthy env with the triggers in place will
    report drift near zero — non-zero drift is signal of (a) a freshly
    landed migration that skipped the fast-path backfill, (b) a
    playbook that was edited without the recount pass running yet, or
    (c) a bug in one of the trigger functions.
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


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def run_recount(
    env_id: UUID,
    *,
    actor_ctx: AgentContext,  # noqa: ARG001 — kept for dispatcher symmetry
    settings: Settings,  # noqa: ARG001 — no tunables today; reserved for future caps
    now: dt.datetime,  # noqa: ARG001 — reserved for future time-windowed recount
) -> RecountPassResult:
    """Reconcile per-memory reference counters against canonical edge tables.

    Reads ``relations``, ``memory_lineage``, and the ``steps`` array of
    every active playbook in ``env_id``. Builds the supersede-chain
    membership index from ``memories.superseded_by``. Issues one
    ``UPDATE`` per memory whose canonical totals differ from its
    current stored values.
    """

    started = time.perf_counter()
    result = RecountPassResult(env_id=env_id)

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
    )

    log.info(
        "recount: env=%s examined=%d adjusted=%d drift(rl/ln/tk/pb)=%d/%d/%d/%d",
        env_id,
        result.memories_examined,
        result.memories_adjusted,
        drift_rl, drift_ln, drift_tk, drift_pb,
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


__all__ = [
    "RecountPassResult",
    "run_recount",
]
