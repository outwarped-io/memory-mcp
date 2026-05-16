"""SDK retry policy (v0.2, Phase 2.2).

Read calls are retried by default; write calls only when the caller opts
in explicitly (``retry_writes=True`` on the MemoryClient or per-call
``idempotency_key``). The policy is intentionally narrow so it cannot
double-write a memory or twice-bump a version counter unless the caller
has acknowledged the duplicate-prevention contract.

Retry triggers
--------------

The policy attempts another call when:

* ``ConnectionError`` from the transport (refused, reset, DNS).
* ``TimeoutError`` from the transport.
* ``httpx.TransportError`` (read timeout, write timeout, network).
* :class:`memory_mcp_client.errors.RateLimitedError` (server pushed back).
* :class:`memory_mcp_client.errors.GraphBackendUnavailableError` for read
  tools that have a non-graph fallback (caller must opt in via tool
  name allowlist — currently only ``mem_search`` qualifies because the
  server degrades hybrid → lex+sem automatically).

It does **not** retry on:

* :class:`MemoryMCPError` subclasses for validation, conflicts, not-found,
  forbidden, version mismatches — these are not transient.
* Any non-retryable transport error not in the allowlist above.

Configuration
-------------

Defaults (3 attempts, 250 ms base, 4× exponential cap, ±50 % jitter)
match the plan §2.2. Override via :class:`RetryPolicy` at client
construction.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

import httpx

from memory_mcp_client.errors import (
    GraphBackendUnavailableError,
    MemoryMCPError,
    RateLimitedError,
    RetryExhaustedError,
)

log = logging.getLogger(__name__)


# Tools that are safe to retry without an idempotency-key. These are pure
# reads — no state change on the server. Keep this list **conservative**;
# adding a tool here is a contract change. Listing tools count as reads.
DEFAULT_READ_TOOLS: frozenset[str] = frozenset(
    {
        # Memories
        "mem_get",
        "mem_get_many",
        "mem_search",
        "mem_browse",
        "mem_facets",
        "mem_neighbors",
        "mem_related",
        "mem_lineage",
        "mem_sources_browse",
        # Envs
        "env_list_",
        "env_export",
        "env_diff",
        "env_snapshot_browse",
        # Entities / relations / graph
        "ent_browse",
        "ent_neighbors",
        "ent_resolve",
        "rel_browse",
        # Tasks
        "task_list",
        "task_browse",
        "task_neighbors",
        # Playbooks / decisions / digest
        "playbook_browse",
        "adr_browse",
        "mem_digest",
        "mem_resume",
        "mem_context_pack",
        # Dream
        "dream_status_",
        "dream_proposals_list",
        # Health / stats / facets
        "mem_stats",
        # Provenance / planning helpers
        "adr_export",
    }
)


@dataclass(frozen=True)
class RetryPolicy:
    """Per-client retry configuration.

    Args:
        max_attempts: Total attempts, including the first call. ``3``
            means "try, retry, retry, then give up". Set to ``1`` to
            disable retries.
        base_delay_seconds: Initial back-off after the first failure.
        max_delay_seconds: Upper bound on each individual sleep before
            jitter.
        exponential_base: Multiplier applied between attempts; the
            policy sleeps ``base_delay_seconds * exponential_base ** n``
            where ``n`` is the zero-indexed failed-attempt count.
        jitter: ``0..1`` randomization range applied to each sleep.
            ``0.5`` means each sleep is multiplied by ``[0.5, 1.5]``.
        retry_writes: When ``False`` (default), only reads are retried.
            Writes opt in by passing ``idempotency_key`` per call, or by
            setting this flag globally.
        read_tools: Override the default read-tools allowlist if needed.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 4.0
    exponential_base: float = 2.0
    jitter: float = 0.5
    retry_writes: bool = False
    read_tools: frozenset[str] = field(default_factory=lambda: DEFAULT_READ_TOOLS)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be >= 0")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if self.exponential_base < 1.0:
            raise ValueError("exponential_base must be >= 1.0")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be in [0, 1]")

    def is_retryable_tool(self, tool_name: str, *, has_idempotency_key: bool) -> bool:
        """Return True when this tool's calls may be retried.

        Reads (in ``read_tools``) always qualify. Writes qualify only
        when the caller passed an ``idempotency_key`` or
        ``retry_writes=True`` is set globally.
        """

        if tool_name in self.read_tools:
            return True
        if has_idempotency_key:
            return True
        return self.retry_writes

    def sleep_for_attempt(self, attempt_index: int) -> float:
        """Compute the sleep (seconds) before attempt index ``i`` (>=1)."""
        if attempt_index < 1:
            return 0.0
        raw = self.base_delay_seconds * (self.exponential_base ** (attempt_index - 1))
        capped = min(raw, self.max_delay_seconds)
        if self.jitter > 0:
            scale = 1.0 + random.uniform(-self.jitter, self.jitter)
            capped *= max(scale, 0.0)
        return capped


_RETRYABLE_MEMORY_MCP_ERRORS: tuple[type[MemoryMCPError], ...] = (
    RateLimitedError,
    GraphBackendUnavailableError,
)


def _is_retryable_exception(exc: BaseException) -> bool:
    """Return True if ``exc`` is transient enough to warrant a retry."""

    if isinstance(exc, _RETRYABLE_MEMORY_MCP_ERRORS):
        return True
    # Other MemoryMCPError subclasses are non-transient (validation,
    # version_conflict, not_found, forbidden, etc.) — don't retry.
    if isinstance(exc, MemoryMCPError):
        return False
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    return False


async def run_with_retry(
    fn: Callable[[], Awaitable[Any]],
    *,
    tool_name: str,
    policy: RetryPolicy,
    has_idempotency_key: bool = False,
) -> Any:
    """Invoke ``fn`` repeatedly per ``policy`` until success or exhaustion.

    Args:
        fn: Zero-arg async producer to retry. Must be safe to call more
            than once for the policy to apply correctly.
        tool_name: Name of the MCP tool being called. Used to decide
            read vs. write retry eligibility.
        policy: The retry policy.
        has_idempotency_key: Whether the per-call payload included an
            ``idempotency_key`` field; if so, writes become retry-eligible.

    Returns:
        The result of the first successful attempt.

    Raises:
        The underlying exception when the tool is not retry-eligible.
        :class:`RetryExhaustedError` when all attempts failed; the last
        underlying exception is preserved on ``__cause__``.
    """

    retry_eligible = policy.is_retryable_tool(
        tool_name, has_idempotency_key=has_idempotency_key
    )

    attempts_meta: list[dict[str, Any]] = []
    last_exc: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            attempts_meta.append(
                {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error_code": getattr(exc, "code", None),
                    "message": str(exc),
                }
            )

            if not retry_eligible:
                raise

            if attempt >= policy.max_attempts:
                break

            if not _is_retryable_exception(exc):
                raise

            sleep_for = policy.sleep_for_attempt(attempt)
            log.warning(
                "memory-mcp retry: tool=%s attempt=%d/%d sleeping=%.3fs error=%s: %s",
                tool_name,
                attempt,
                policy.max_attempts,
                sleep_for,
                type(exc).__name__,
                exc,
            )
            await asyncio.sleep(sleep_for)

    # All attempts exhausted.
    raise RetryExhaustedError(
        f"tool={tool_name} exhausted after {policy.max_attempts} attempts",
        details={"attempts": attempts_meta, "tool": tool_name},
    ) from last_exc


__all__ = [
    "DEFAULT_READ_TOOLS",
    "RetryPolicy",
    "run_with_retry",
]
