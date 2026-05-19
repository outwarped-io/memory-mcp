"""Canonical error classes for memory-mcp tools.

Each error has a stable ``code`` attribute that maps 1:1 to the wire-format
``error.code`` field documented in the plan's "Error Model" table. The MCP
transport layer translates these into JSON-RPC error responses; HTTP clients
hitting the REST surface get them as 4xx/5xx with a structured body.

The ``code`` is the **public contract** — error class names are an
implementation detail and may change. Tests assert on ``code``, never on
``isinstance``-of-class-name.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID


class MemoryMCPError(Exception):
    """Base class. Subclasses set a class-level ``code`` attribute."""

    code: str = "INTERNAL"

    def __init__(self, message: str | None = None, **details: object) -> None:
        super().__init__(message or self.__class__.__name__)
        self.details: dict[str, object] = dict(details)


# ---------------------------------------------------------------------------
# Reserved for v1.5 — DEFINED but never raised in v1 (local-only build).
# Kept here so tool code paths can have ``except FORBIDDEN_ENV`` shaped
# handling already wired without conditional imports.
# ---------------------------------------------------------------------------

class UnauthorizedError(MemoryMCPError):
    """Bad / missing / expired token. Reserved; v1 never raises."""

    code = "UNAUTHORIZED"


class ForbiddenEnvError(MemoryMCPError):
    """Caller has no grant on the requested env. Reserved; v1 never raises."""

    code = "FORBIDDEN_ENV"


class EnvNotAttachedError(MemoryMCPError):
    """Caller requested an env that is not attached to the current session."""

    code = "ENV_NOT_ATTACHED"


class EnvRefBothProvidedError(MemoryMCPError):
    """Caller provided both an env id field and its friendly-name twin."""

    code = "ENV_REF_BOTH_PROVIDED"

    def __init__(self, field: str) -> None:
        super().__init__(
            f"ENV_REF_BOTH_PROVIDED: both id and name provided for {field!r}",
            field=field,
        )
        self.field = field


class EnvRefAmbiguousError(MemoryMCPError):
    """Case-insensitive env-name lookup matched multiple rows."""

    code = "ENV_REF_AMBIGUOUS"

    def __init__(self, name: str, candidate_ids: list[UUID]) -> None:
        super().__init__(
            f"ENV_REF_AMBIGUOUS: {name!r} matched multiple environments",
            name=name,
            candidate_ids=[str(candidate_id) for candidate_id in candidate_ids],
        )
        self.name = name
        self.candidate_ids = list(candidate_ids)


class EnvNotFoundError(MemoryMCPError):
    """Friendly env name did not resolve to an environment."""

    code = "ENV_NOT_FOUND"

    def __init__(self, name: str) -> None:
        super().__init__(f"ENV_NOT_FOUND: environment name {name!r} not found", name=name)
        self.name = name


# Back-compat / ergonomic aliases used by helper modules and tests.
EnvRefBothProvided = EnvRefBothProvidedError
EnvRefAmbiguous = EnvRefAmbiguousError
EnvNotFound = EnvNotFoundError


# ---------------------------------------------------------------------------
# Live in v1.
# ---------------------------------------------------------------------------

class EnvAmbiguousError(MemoryMCPError):
    """Caller has >1 writable env attached and didn't specify which to use."""

    code = "ENV_AMBIGUOUS"


class VersionConflictError(MemoryMCPError):
    """Optimistic-lock collision on update / supersede.

    Carries ``expected`` and ``actual`` version numbers so the caller can
    re-fetch and merge.
    """

    code = "VERSION_CONFLICT"

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"VERSION_CONFLICT: expected={expected}, actual={actual}",
            expected=expected,
            actual=actual,
        )
        self.expected = expected
        self.actual = actual


class InvalidTransitionError(MemoryMCPError):
    """Lifecycle transition not allowed by the canonical state machine."""

    code = "INVALID_TRANSITION"

    def __init__(self, src: str, dst: str) -> None:
        super().__init__(
            f"INVALID_TRANSITION: {src!r} → {dst!r}",
            src=src,
            dst=dst,
        )
        self.src = src
        self.dst = dst


class BlastRadiusExceededError(MemoryMCPError):
    """Cascade hard-delete would exceed the caller's blast-radius cap."""

    code = "BLAST_RADIUS_EXCEEDED"

    def __init__(
        self,
        *,
        cap_hit: Literal["depth", "count"],
        limit: int,
        would_affect: list[UUID],
        message: str,
        **details: object,
    ) -> None:
        serialized = [str(memory_id) for memory_id in would_affect]
        super().__init__(
            message,
            cap_hit=cap_hit,
            limit=limit,
            would_affect=serialized,
            **details,
        )
        self.cap_hit = cap_hit
        self.limit = limit
        self.would_affect = list(would_affect)


class NotFoundError(MemoryMCPError):
    """Record id missing within the addressable scope."""

    code = "NOT_FOUND"


class AlreadyExistsError(MemoryMCPError):
    """Insert collided with an existing row on a unique key (e.g. env name)."""

    code = "ALREADY_EXISTS"


class SessionRequiredError(MemoryMCPError):
    """Tool needs a session id (``X-Session-Id`` header) but none was provided.

    Used by ``env_attach`` / ``env_detach`` whose state is per-session. v1
    keeps this state in-memory; the header is the only stable key.
    """

    code = "SESSION_REQUIRED"


class EmbeddingModelMismatchError(MemoryMCPError):
    """Env's default embedding model id does not match the configured embedder."""

    code = "EMBEDDING_MODEL_MISMATCH"

    def __init__(self, expected: str, actual: str) -> None:
        super().__init__(
            f"EMBEDDING_MODEL_MISMATCH: env expects {expected!r}, embedder is {actual!r}",
            expected=expected,
            actual=actual,
        )
        self.expected = expected
        self.actual = actual


class InvalidCursorError(MemoryMCPError):
    """Pagination cursor is malformed or doesn't match the current query shape.

    Raised when a backend rejects a cursor (e.g. ``GraphStore.neighbors``
    detects that the embedded query shape mismatches the new call). This
    is a caller-correctable condition — callers should drop the cursor
    and re-page from the start.
    """

    code = "INVALID_CURSOR"


class InvalidInputError(MemoryMCPError):
    """Caller-supplied input is malformed or violates a structural invariant.

    Used for:

    * Cypher / SQL identifier allowlist rejections (e.g. relation type
      contains a backtick or unsafe character).
    * Proposal payloads referenced by ``dream_review`` that are missing
      required keys, contain unparseable UUIDs, or self-reference
      (e.g. ``primary_id`` appearing in ``candidate_ids``).
    * Reserved-but-unimplemented action surfaces (``dream_review
      action='amend'``).

    Distinct from :class:`InvalidTransitionError` which is reserved for
    *valid input that violates the lifecycle state machine*. Callers
    should not retry without first correcting the request.
    """

    code = "INVALID_INPUT"


class CycleDetectedError(MemoryMCPError):
    """Adding a graph edge would create a cycle."""

    code = "CYCLE_DETECTED"


class GraphBackendUnavailableError(MemoryMCPError):
    """The configured graph store is unreachable / degraded.

    Raised when a ``mode=graph`` request must propagate a backend failure
    so the caller can distinguish "no graph hits" from "graph subsystem
    down". ``mode=hybrid`` swallows the backend error and degrades
    silently to lex+sem; ``mode=graph`` surfaces this code.
    """

    code = "GRAPH_BACKEND_UNAVAILABLE"


class LLMUnavailableError(MemoryMCPError):
    """The configured LLM backend is unreachable, mis-configured, or disabled.

    Raised by:
    * ``NullLLMClient`` on every call (wired when ``LLM_BACKEND=null``)
    * HTTP-backed clients when the upstream is unreachable / times out / 5xxs
    * The factory when settings are inconsistent (e.g. missing API key)

    Dream-worker callers MUST catch this and either fall back to template
    output (``LLMSummarizer``'s default behaviour) or surface a structured
    proposal-generation failure. Tool-surface code paths that do not have
    a meaningful fallback should propagate it as ``LLM_UNAVAILABLE``.
    """

    code = "LLM_UNAVAILABLE"


class AuthorityDisabledError(MemoryMCPError):
    """The ``reference_authority`` signal is dormant — caller asked for it anyway.

    Raised by ``mem_top(by="reference_authority")`` when
    ``Settings.dream_popularity_authority_weighted`` is ``False`` (default).

    When the knob is OFF, the recount pass does not maintain the four
    ``ref_authority_*`` columns, so they stay at 0 across the env and
    the metric would return only zero-ranked rows. Rather than surface
    a meaningless ranking, the tool fails fast — callers must flip the
    knob in settings and let at least one recount cycle complete before
    this metric is meaningful.

    The check fires before env resolution / RBAC / DB so callers see a
    clean "metric unavailable" signal at no cost.
    """

    code = "AUTHORITY_DISABLED"


class ValidationFailedError(MemoryMCPError):
    """Request did not satisfy the tool's input schema.

    Raised when Pydantic's ``model_validate`` rejects the caller's
    arguments. The :attr:`details` dict carries:

    * ``errors`` — the raw ``pydantic.ValidationError.errors()`` payload
      (re-shaped so ``loc`` is a list of strings/ints, never tuples).
    * ``hints`` — a list of ``{loc, offered, suggested, confidence}``
      objects, present **only when** the resolver is reasonably sure the
      caller meant a different field (allowlist match OR fuzzy match
      above the configured threshold).

    The wire-format message includes a parenthesised ``did you mean
    ``<field>``?`` hint when exactly one high-confidence suggestion
    exists; multiple or low-confidence hints stay in ``details`` only.

    ``input_value`` from Pydantic's error payload is **never** echoed —
    request bodies may carry secrets.
    """

    code = "VALIDATION_FAILED"


__all__ = [
    "AlreadyExistsError",
    "AuthorityDisabledError",
    "BlastRadiusExceededError",
    "CycleDetectedError",
    "EmbeddingModelMismatchError",
    "EnvAmbiguousError",
    "EnvNotAttachedError",
    "EnvNotFound",
    "EnvNotFoundError",
    "EnvRefAmbiguous",
    "EnvRefAmbiguousError",
    "EnvRefBothProvided",
    "EnvRefBothProvidedError",
    "ForbiddenEnvError",
    "GraphBackendUnavailableError",
    "InvalidCursorError",
    "InvalidInputError",
    "InvalidTransitionError",
    "LLMUnavailableError",
    "MemoryMCPError",
    "NotFoundError",
    "SessionRequiredError",
    "UnauthorizedError",
    "ValidationFailedError",
    "VersionConflictError",
]
