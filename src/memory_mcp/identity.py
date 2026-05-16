"""Agent identity for v1 (local-only build).

Surface
-------

* :class:`AgentContext` — request-scoped dataclass carrying ``agent_id``,
  optional friendly ``agent_name``, optional ``session_id``, and the list of
  ``attached_env_ids`` (populated by ``env_attach`` in :mod:`memory_mcp.tools.envs`).

* :class:`IdentityResolver` — lookup-or-create logic for the ``agents``
  table, with a stable server-default agent persisted to a file on first
  run. The resolver is a singleton per process; instantiate once via
  :func:`get_identity_resolver` (lru-cached on settings).

Header contract (v1)
--------------------

* ``X-Agent-Id`` — UUID. Caller-managed identity. If present, the resolver
  looks the row up in ``agents`` and inserts on conflict-do-nothing.
* ``X-Agent-Name`` — optional friendly name; persisted on first create only
  (we don't update existing rows).
* ``X-Session-Id`` — optional UUID; clients may pass to keep ``env_attach``
  state stable across calls. If absent, the server treats each request as
  its own ephemeral session.

Default-agent file
------------------

When the ``X-Agent-Id`` header is missing, requests run under the
server-default agent. Its UUID is persisted to
``settings.local_default_agent_file`` on first start. Format::

    {
      "agent_id": "<uuid>",
      "agent_name": "default-local-agent",
      "created_at": "<iso8601>"
    }

File mode is set to ``0600``. **Deleting the file orphans memories created
under the old default agent** — the audit trail still references the old
UUID, but no future requests will be attributed to it. This is a known
local-only trade-off; see the rubber-duck gate #2 review notes.

v1.5 forward-compat
-------------------

When auth lands, header parsing is replaced by token verification, the
``AgentContext`` schema gets a non-empty ``scopes`` field, and
``IdentityResolver`` becomes one of two strategies behind a Protocol. None
of the calling tool code changes.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from memory_mcp.config import Settings
from memory_mcp.db.models import Agent
from memory_mcp.db.postgres import session_scope

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentContext:
    """Request-scoped identity + attached envs.

    Mutable in v1 only via the ``env_attach`` / ``env_detach`` tools, which
    update ``attached_env_ids`` on the caller's session-state container (in
    :mod:`memory_mcp.tools.envs`). Tools should treat this as effectively
    immutable for the duration of a single tool call.
    """

    agent_id: UUID
    agent_name: str | None = None
    session_id: UUID | None = None
    attached_env_ids: list[UUID] = field(default_factory=list)
    attached_env_names: list[str] = field(default_factory=list)
    is_default_agent: bool = False


class IdentityResolver:
    """Looks up / creates ``agents`` rows; manages the default-agent file.

    Thread- and asyncio-safe via an ``asyncio.Lock`` around the default-agent
    bootstrap. After bootstrap, lookups are pure ``SELECT``s with no shared
    state.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._default_agent_id: UUID | None = None
        self._default_agent_name: str = settings.local_default_agent_name
        self._bootstrap_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve(
        self,
        *,
        agent_id_header: str | None,
        agent_name_header: str | None,
        session_id_header: str | None,
    ) -> AgentContext:
        """Build an ``AgentContext`` for the current request.

        Raises ``ValueError`` on malformed UUID headers. Tools should map
        that to a 400-class error in the transport layer.
        """
        # 1. Session id — pure parsing, no DB hit.
        session_id: UUID | None = _parse_uuid(session_id_header) if session_id_header else None

        # 2. Agent id — header path or default.
        if agent_id_header:
            requested_id = _parse_uuid(agent_id_header)
            await self._ensure_agent_row(requested_id, agent_name_header)
            return AgentContext(
                agent_id=requested_id,
                agent_name=agent_name_header,
                session_id=session_id,
                is_default_agent=False,
            )

        # 3. Default agent — bootstrap once, then reuse.
        default_id = await self._ensure_default_agent()
        return AgentContext(
            agent_id=default_id,
            agent_name=self._default_agent_name,
            session_id=session_id,
            is_default_agent=True,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _ensure_agent_row(self, agent_id: UUID, agent_name: str | None) -> None:
        """Idempotent upsert: insert on first sight, refresh ``last_seen_at`` on conflict.

        Two concurrent requests with the same ``agent_id`` will see exactly
        one row created; subsequent calls just bump ``last_seen_at``. Names
        are NOT updated on conflict — first-create wins.
        """
        name = agent_name or "agent-" + agent_id.hex[:8]
        async with session_scope() as s:
            stmt = (
                pg_insert(Agent.__table__)
                .values(id=agent_id, name=name, last_seen_at=func.now())
                .on_conflict_do_update(
                    index_elements=[Agent.__table__.c.id],
                    set_={"last_seen_at": func.now()},
                )
            )
            await s.execute(stmt)

    async def _ensure_default_agent(self) -> UUID:
        """Read or create the server-default agent.

        Multi-process safe: races on the default-agent file are resolved via
        an ``O_CREAT | O_EXCL`` atomic create. The first writer wins; any
        loser reads the file the winner produced. The file mode is set to
        ``0600``.

        Corruption-aware: a file that exists but is unreadable / malformed
        raises :class:`RuntimeError` to fail fast — silently rotating the
        default identity would orphan more memories than the operator
        expects. Move the bad file aside manually to recover.
        """
        if self._default_agent_id is not None:
            return self._default_agent_id

        async with self._bootstrap_lock:
            if self._default_agent_id is not None:  # double-check
                return self._default_agent_id

            file_path = Path(self._settings.local_default_agent_file)
            agent_id = self._read_or_create_default_file(file_path)
            await self._ensure_agent_row(agent_id, self._default_agent_name)
            self._default_agent_id = agent_id
            return agent_id

    def _read_or_create_default_file(self, path: Path) -> UUID:
        """Return the persisted default-agent UUID, creating the file if absent.

        Loop semantics: read → if missing, try atomic create → if a peer won
        the race, read again. At most two iterations.
        """
        last_attempt = 0
        for attempt in (1, 2):
            last_attempt = attempt
            payload = self._read_default_file(path)
            if payload is not None:
                self._default_agent_name = str(
                    payload.get("agent_name") or self._default_agent_name
                )
                return _parse_uuid(payload["agent_id"])

            new_id = uuid4()
            new_payload: dict[str, object] = {
                "agent_id": str(new_id),
                "agent_name": self._default_agent_name,
                "created_at": dt.datetime.now(dt.UTC).isoformat(),
            }
            if self._atomic_create_default_file(path, new_payload):
                log.info("identity: created default agent %s at %s", new_id, path)
                return new_id

            # Lost the race — peer process created the file. Loop to read it.
            log.info("identity: bootstrap race lost; reading peer-created %s", path)

        raise RuntimeError(
            f"identity: bootstrap loop exhausted at {path} (attempt {last_attempt})",
        )

    @staticmethod
    def _read_default_file(path: Path) -> dict[str, object] | None:
        """Return the parsed file or ``None`` if it doesn't exist.

        Raises ``RuntimeError`` if the file exists but is unreadable or
        malformed — silent rotation would orphan memories. Operators should
        move the bad file aside manually to recover.
        """
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"identity: default-agent file {path} exists but is unreadable: {exc}. "
                "Move it aside manually to bootstrap a new default agent.",
            ) from exc
        if not isinstance(data, dict) or "agent_id" not in data:
            raise RuntimeError(
                f"identity: default-agent file {path} is missing required keys "
                "(expected at least 'agent_id'). Move it aside manually.",
            )
        return data

    @staticmethod
    def _atomic_create_default_file(path: Path, payload: dict[str, object]) -> bool:
        """Try to create the file exclusively. Returns True if we created it.

        Uses ``O_CREAT | O_EXCL`` so two processes cannot both win. The
        winner writes the JSON payload; losers see ``FileExistsError`` and
        return ``False`` so the caller re-reads.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        try:
            data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            os.write(fd, data)
        finally:
            os.close(fd)
        # On Windows ``os.open`` with mode 0o600 may not stick — best-effort fix-up.
        try:
            os.chmod(path, 0o600)
        except OSError:
            log.warning("identity: could not chmod 0600 on %s", path)
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"invalid UUID header value: {value!r}") from exc


_resolver_cache: tuple[Settings, IdentityResolver] | None = None


def get_identity_resolver(settings: Settings) -> IdentityResolver:
    """Process-wide cached resolver.

    Cannot use :func:`functools.lru_cache` because :class:`Settings` (a
    pydantic-settings model) is not hashable. We keep a single (settings,
    resolver) pair and rebuild if the settings instance changes (test
    isolation).
    """
    global _resolver_cache
    if _resolver_cache is not None and _resolver_cache[0] is settings:
        return _resolver_cache[1]
    resolver = IdentityResolver(settings)
    _resolver_cache = (settings, resolver)
    return resolver


def _reset_identity_resolver_cache() -> None:
    """Test helper — drop the cached resolver so the next call rebuilds."""
    global _resolver_cache
    _resolver_cache = None


# Test-friendly alias matching the lru_cache pattern callers use elsewhere.
get_identity_resolver.cache_clear = _reset_identity_resolver_cache  # type: ignore[attr-defined]


__all__ = ["AgentContext", "IdentityResolver", "get_identity_resolver"]
