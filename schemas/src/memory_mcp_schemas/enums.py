"""Domain types: Enums + Pydantic v2 schemas for tool I/O.

The Postgres schema uses ``text + CHECK`` rather than native ``ENUM`` types
(see migration ``0001_v1_initial`` for rationale). Python-side we map them to
``StrEnum`` for type safety and tool-surface validation.
"""

from __future__ import annotations

from enum import StrEnum

# ---------------------------------------------------------------------------
# Enums (mirror the migration CHECK constraints)
# ---------------------------------------------------------------------------

class MemoryKind(StrEnum):
    fact = "fact"
    procedure = "procedure"
    playbook = "playbook"
    event = "event"
    decision = "decision"
    preference = "preference"
    observation = "observation"
    journal_entry = "journal_entry"
    session_digest = "session_digest"
    snippet = "snippet"


class MemoryStatus(StrEnum):
    proposed = "proposed"
    active = "active"
    stale = "stale"
    archived = "archived"
    superseded = "superseded"
    retired = "retired"


class DecisionStatus(StrEnum):
    proposed = "proposed"
    accepted = "accepted"
    deprecated = "deprecated"
    superseded = "superseded"


class GrantRole(StrEnum):
    read = "read"
    write = "write"
    admin = "admin"


class GraphNodeType(StrEnum):
    entity = "entity"
    memory = "memory"
    task = "task"


class OutboxAggregateType(StrEnum):
    memory = "memory"
    entity = "entity"
    relation = "relation"
    env = "env"
    task = "task"


class OutboxOp(StrEnum):
    upsert = "upsert"
    tombstone = "tombstone"
    update = "update"


class OutboxSink(StrEnum):
    qdrant = "qdrant"
    neo4j = "neo4j"
    pgvector = "pgvector"


class OutboxDeliveryStatus(StrEnum):
    pending = "pending"
    in_flight = "in_flight"
    done = "done"
    dead = "dead"


class ProjectionStatus(StrEnum):
    healthy = "healthy"
    degraded = "degraded"
    down = "down"
    rebuilding = "rebuilding"


class MemorySourceType(StrEnum):
    session = "session"
    file = "file"
    import_ = "import"
    url = "url"
    llm = "llm"
    dream = "dream"
    digest = "digest"
    digest_template = "digest-template"
    user = "user"
    agent = "agent"
    other = "other"


class LineageRelation(StrEnum):
    promoted_from = "promoted_from"
    summarized_from = "summarized_from"
    copied_from = "copied_from"
    moved_from = "moved_from"
    supersedes = "supersedes"


class TaskStatus(StrEnum):
    pending = "pending"
    in_progress = "in_progress"
    blocked = "blocked"
    done = "done"
    cancelled = "cancelled"


class TaskRelationKind(StrEnum):
    depends_on = "depends_on"
    motivated_by = "motivated_by"
    produces = "produces"
    references = "references"


# ---------------------------------------------------------------------------
# Lifecycle transition table
# ---------------------------------------------------------------------------
# Encodes the matrix from design.md → Lifecycle. Used by the lifecycle helper
# in the canonical writer and by tests.

_LIFECYCLE_TRANSITIONS: dict[MemoryStatus, frozenset[MemoryStatus]] = {
    # ``proposed`` only resolves via dream_review: accept→active, reject→retired.
    # Archiving a proposal would muddle the dream-review workflow, so it's
    # disallowed (callers archive after acceptance, or simply reject).
    MemoryStatus.proposed: frozenset({MemoryStatus.active, MemoryStatus.retired}),
    MemoryStatus.active: frozenset({
        MemoryStatus.stale,
        MemoryStatus.archived,
        MemoryStatus.superseded,
        MemoryStatus.retired,
    }),
    MemoryStatus.stale: frozenset({
        MemoryStatus.active,
        MemoryStatus.archived,
        MemoryStatus.superseded,
        MemoryStatus.retired,
    }),
    # An archived memory may still be superseded — corrects the rubber-duck
    # gate-3 finding.  Reactivation goes through admin tools.
    MemoryStatus.archived: frozenset({
        MemoryStatus.active,
        MemoryStatus.superseded,
        MemoryStatus.retired,
    }),
    MemoryStatus.superseded: frozenset({MemoryStatus.retired}),
    MemoryStatus.retired: frozenset(),
}


def is_valid_transition(src: MemoryStatus, dst: MemoryStatus) -> bool:
    """Return True if ``src → dst`` is a permitted lifecycle move."""
    if src == dst:
        return True  # idempotent re-application of current status
    return dst in _LIFECYCLE_TRANSITIONS.get(src, frozenset())


_TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.pending: frozenset({
        TaskStatus.in_progress,
        TaskStatus.blocked,
        TaskStatus.cancelled,
    }),
    TaskStatus.in_progress: frozenset({
        TaskStatus.blocked,
        TaskStatus.done,
        TaskStatus.cancelled,
        TaskStatus.pending,
    }),
    TaskStatus.blocked: frozenset({
        TaskStatus.pending,
        TaskStatus.in_progress,
        TaskStatus.cancelled,
    }),
    TaskStatus.done: frozenset(),
    TaskStatus.cancelled: frozenset(),
}


def is_valid_task_transition(src: TaskStatus, dst: TaskStatus) -> bool:
    """Return True if ``src → dst`` is a permitted task lifecycle move."""
    if src == dst:
        return True
    return dst in _TASK_TRANSITIONS.get(src, frozenset())


# ---------------------------------------------------------------------------
# Dream worker enums (Phase 2.2)
# ---------------------------------------------------------------------------

class DreamMode(StrEnum):
    decay = "decay"
    dedupe = "dedupe"
    promote = "promote"
    decision_conflicts = "decision_conflicts"
    retention = "retention"


class DreamRunStatus(StrEnum):
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class DreamRunTrigger(StrEnum):
    scheduler = "scheduler"
    tool = "tool"
    test = "test"


class DreamProposalKind(StrEnum):
    merge_candidate = "merge_candidate"
    promotion_candidate = "promotion_candidate"
    decay_candidate = "decay_candidate"
    decision_conflict_candidate = "decision_conflict_candidate"


class DreamProposalStatus(StrEnum):
    open = "open"
    accepted = "accepted"
    rejected = "rejected"
    amended = "amended"
    deferred = "deferred"
    expired = "expired"


class DreamReviewAction(StrEnum):
    accept = "accept"
    reject = "reject"
    amend = "amend"
    defer = "defer"


class SummarizerKind(StrEnum):
    llm = "llm"
    template = "template"
