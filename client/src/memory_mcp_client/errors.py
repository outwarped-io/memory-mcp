"""Typed exception hierarchy for memory-mcp client SDK.

The server raises :class:`mcp.server.fastmcp.exceptions.ToolError` whose
message embeds a stable error code in the form ``[CODE] message...``
(see :mod:`memory_mcp.errors` server-side for the canonical list). The
client parses these back into a typed exception hierarchy so callers can
``except VersionConflictError`` rather than string-matching error text.

Every server error code maps to exactly one client exception class. The
``code`` is the public contract; class names are an implementation
detail. Unknown codes fall back to :class:`UnknownError`.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

_CODE_RE = re.compile(r"^\s*\[([A-Z][A-Z0-9_]*)\]\s*(.*)$", re.DOTALL)


class MemoryMCPError(Exception):
    """Base class for every memory-mcp protocol-level error."""

    code: ClassVar[str] = "INTERNAL"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = details or {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# --- Live in v1 -------------------------------------------------------------


class VersionConflictError(MemoryMCPError):
    """Optimistic-lock collision on update / supersede.

    ``details`` carries ``expected`` and ``actual`` version numbers so the
    caller can re-fetch and merge.
    """

    code = "VERSION_CONFLICT"


class InvalidTransitionError(MemoryMCPError):
    """Lifecycle transition not allowed by the state machine."""

    code = "INVALID_TRANSITION"


class NotFoundError(MemoryMCPError):
    """Record id missing within the addressable scope."""

    code = "NOT_FOUND"


class AlreadyExistsError(MemoryMCPError):
    """Insert collided with an existing row on a unique key."""

    code = "ALREADY_EXISTS"


class EnvAmbiguousError(MemoryMCPError):
    """Caller has >1 writable env attached and didn't specify which."""

    code = "ENV_AMBIGUOUS"


class EnvNotAttachedError(MemoryMCPError):
    """Caller requested an env not bound to the current session."""

    code = "ENV_NOT_ATTACHED"


class SessionRequiredError(MemoryMCPError):
    """Tool needs an X-Session-Id header but none was provided."""

    code = "SESSION_REQUIRED"


class EmbeddingModelMismatchError(MemoryMCPError):
    """Env's default embedder mismatches the configured embedding model."""

    code = "EMBEDDING_MODEL_MISMATCH"


class InvalidCursorError(MemoryMCPError):
    """Pagination cursor malformed or stale relative to current query shape."""

    code = "INVALID_CURSOR"


class InvalidInputError(MemoryMCPError):
    """Caller-supplied input is malformed or violates a structural invariant."""

    code = "INVALID_INPUT"


class ValidationError(MemoryMCPError):
    """Alias for client-side Pydantic-style validation errors raised by the SDK."""

    code = "VALIDATION"


class ValidationFailedError(ValidationError):
    """v0.11 — Pydantic validation failure with did-you-mean hints.

    Same root cause as :class:`ValidationError` but the server includes
    a ``hints`` block in ``details`` that surfaces likely field-name
    typos (``req`` → ``request``, ``env`` → ``env_ids``, etc.) so
    callers can correct payloads without staring at raw Pydantic loc
    paths.
    """

    code = "VALIDATION_FAILED"


class RetryExhaustedError(MemoryMCPError):
    """SDK-side: all retry attempts failed.

    Raised by the v0.2 retry policy when a retryable error fired through
    every attempt without succeeding. The wrapped underlying error is
    surfaced as ``__cause__`` (chain) and ``details["attempts"]`` records
    the per-attempt error codes.
    """

    code = "RETRY_EXHAUSTED"


class CycleDetectedError(MemoryMCPError):
    """Adding a graph edge would create a cycle."""

    code = "CYCLE_DETECTED"


class GraphBackendUnavailableError(MemoryMCPError):
    """Configured graph store is unreachable / degraded."""

    code = "GRAPH_BACKEND_UNAVAILABLE"


class LLMUnavailableError(MemoryMCPError):
    """Configured LLM backend is unreachable / mis-configured / disabled."""

    code = "LLM_UNAVAILABLE"


# --- Reserved for v1.5 (forward-compat) ------------------------------------


class UnauthorizedError(MemoryMCPError):
    """Bad / missing / expired token. Reserved; v1 never raises."""

    code = "UNAUTHORIZED"


class ForbiddenEnvError(MemoryMCPError):
    """Caller has no grant on the requested env. Reserved."""

    code = "FORBIDDEN_ENV"


# --- Client-side convenience aliases ---------------------------------------


class AuthError(UnauthorizedError):
    """Convenience alias collapsing UNAUTHORIZED + FORBIDDEN_ENV families."""


class ConflictError(AlreadyExistsError):
    """Convenience alias for ALREADY_EXISTS-style unique-violation errors."""


class RateLimitedError(MemoryMCPError):
    """Caller was rate-limited (v1.5+ surface)."""

    code = "RATE_LIMITED"


class InternalError(MemoryMCPError):
    """Server wrapped an unexpected exception into [INTERNAL]."""

    code = "INTERNAL"


class UnknownError(MemoryMCPError):
    """Fallback for error codes the client doesn't recognize.

    The raw server-side code is preserved on ``self.code`` so debugging
    output still surfaces it.
    """

    code = "UNKNOWN"


_CODE_TO_CLASS: dict[str, type[MemoryMCPError]] = {
    "VERSION_CONFLICT": VersionConflictError,
    "INVALID_TRANSITION": InvalidTransitionError,
    "NOT_FOUND": NotFoundError,
    "ALREADY_EXISTS": AlreadyExistsError,
    "ENV_AMBIGUOUS": EnvAmbiguousError,
    "ENV_NOT_ATTACHED": EnvNotAttachedError,
    "SESSION_REQUIRED": SessionRequiredError,
    "EMBEDDING_MODEL_MISMATCH": EmbeddingModelMismatchError,
    "INVALID_CURSOR": InvalidCursorError,
    "INVALID_INPUT": InvalidInputError,
    "VALIDATION": ValidationError,
    "VALIDATION_ERROR": ValidationError,
    "VALIDATION_FAILED": ValidationFailedError,
    "RETRY_EXHAUSTED": RetryExhaustedError,
    "CYCLE_DETECTED": CycleDetectedError,
    "GRAPH_BACKEND_UNAVAILABLE": GraphBackendUnavailableError,
    "LLM_UNAVAILABLE": LLMUnavailableError,
    "UNAUTHORIZED": UnauthorizedError,
    "FORBIDDEN_ENV": ForbiddenEnvError,
    "RATE_LIMITED": RateLimitedError,
    "INTERNAL": InternalError,
    "INTERNAL_ERROR": InternalError,
}


def parse_error(message: str) -> MemoryMCPError:
    """Parse a ``ToolError`` message into a typed client exception.

    Expected server format: ``[CODE] message :: {json_details}`` produced
    by :func:`memory_mcp.mcp_app._format_tool_error`. Older or external
    callers may send only ``[CODE] message`` without the separator, or
    even a bare message — all three shapes are handled. Unrecognized
    codes fall back to :class:`UnknownError` (with the raw code
    preserved). Messages without a ``[CODE]`` prefix yield the bare
    :class:`MemoryMCPError`.
    """

    match = _CODE_RE.match(message)
    if match is None:
        return MemoryMCPError(message)

    raw_code = match.group(1)
    rest = match.group(2).strip()

    body, details = _split_details(rest)
    exc_cls = _CODE_TO_CLASS.get(raw_code, UnknownError)
    return exc_cls(body or rest, code=raw_code, details=details)


def _split_details(text: str) -> tuple[str, dict[str, Any] | None]:
    """Return ``(body, details_dict_or_None)`` for a server-format message.

    Tries the canonical ``message :: {json}`` shape first; falls back to
    extracting a trailing JSON object anywhere at the tail.
    """

    sep = text.rfind(" :: ")
    if sep != -1:
        body = text[:sep].strip()
        tail = text[sep + 4 :].strip()
        details = _try_parse_object(tail)
        if details is not None:
            return body, details

    details, body = _split_trailing_json(text)
    if details is not None:
        return body, details
    return text, None


def _try_parse_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _split_trailing_json(text: str) -> tuple[dict[str, Any] | None, str]:
    """Pull a trailing JSON object off the end of ``text``, if any."""

    stripped = text.rstrip()
    if not stripped.endswith("}"):
        return None, text

    depth = 0
    start = -1
    for i in range(len(stripped) - 1, -1, -1):
        ch = stripped[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                start = i
                break
    if start < 0:
        return None, text

    candidate = stripped[start:]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None, text
    if not isinstance(parsed, dict):
        return None, text

    body = stripped[:start].rstrip(" \t\n:|—-")
    return parsed, body
