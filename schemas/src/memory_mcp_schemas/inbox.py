"""v0.17 inbox / drop-box request and response models.

Three tools — ``mem_inbox_open``, ``mem_inbox_send``, ``mem_inbox`` — wrap
existing entity and memory primitives to provide user-orchestrated
message passing between agents. Channels are entities of kind
``"channel"``; messages are memories of kind :class:`MemoryKind.message`
linked to the channel entity via ``entity_links``.

References use a copy-pasteable URL form ``mem-inbox://<env-name>/<slug>``.
The server is the only producer of well-formed references; clients pass
them back verbatim. Bare slugs are also accepted on ``to`` fields when
an explicit ``env_id`` or ``env_name`` arg is provided.

See :mod:`memory_mcp.inbox` for the operational layer and the workspace
overlay ``mem-inbox.instructions.md`` for the three user-orchestrated
flows this surface supports.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_mcp_schemas._env_refs import validate_optional_env_ref_pair


# Reference URL scheme + cross-cutting constants. Mirrored in
# memory_mcp.inbox so the server and clients stay aligned.
REFERENCE_SCHEME: str = "mem-inbox"
INBOX_TAG: str = "inbox"
DEFAULT_TTL_DAYS: int = 7
MAX_TTL_DAYS: int = 90


class MemInboxOpenRequest(BaseModel):
    """Open (or look up) an inbox channel.

    The result carries a copy-pasteable :data:`REFERENCE_SCHEME` URL
    that another agent can pass to :class:`MemInboxSendRequest.to` or
    :class:`MemInboxRequest.to` without further parsing.

    ``name`` is the channel slug. When omitted, the server generates a
    pronounceable ``<adjective>-<noun>`` slug; when provided, it must
    be kebab-case (1–64 chars).

    ``idempotent=True`` lets the caller re-open an existing channel
    without raising; ``False`` raises if a channel with that slug
    already exists in the env (still safe to retry via
    ``ent_resolve``).
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=400)
    env_id: UUID | None = None
    env_name: str | None = None
    idempotent: bool = False

    @model_validator(mode="after")
    def _validate_env_mutex(self) -> "MemInboxOpenRequest":
        return validate_optional_env_ref_pair(self)


class MemInboxOpenResponse(BaseModel):
    """Result of :class:`MemInboxOpenRequest`.

    ``reference`` is the server-formatted URL — clients pass it through
    to :class:`MemInboxSendRequest.to` / :class:`MemInboxRequest.to`
    verbatim. ``created`` distinguishes a freshly-opened channel from
    an idempotent re-open of an existing one.
    """

    reference: str
    entity_id: UUID
    canonical_name: str
    env_id: UUID
    env_name: str
    created: bool


class MemInboxSendRequest(BaseModel):
    """Drop a message into a channel.

    ``to`` accepts either the URL form ``mem-inbox://<env>/<slug>`` or
    a bare slug. When the URL form is used and ``env_id`` / ``env_name``
    is *also* provided, they must agree — disagreement raises rather
    than silently writing cross-env.

    No auto-create on send: a non-existent slug raises. Use
    :class:`MemInboxOpenRequest` first.

    ``expires_at`` defaults to ``now() + DEFAULT_TTL_DAYS`` and is
    capped at ``MAX_TTL_DAYS``.
    """

    model_config = ConfigDict(extra="forbid")

    to: str = Field(min_length=1, max_length=512)
    body: str = Field(min_length=1)
    env_id: UUID | None = None
    env_name: str | None = None
    title: str | None = Field(default=None, max_length=400)
    expires_at: dt.datetime | None = None
    display_from: str | None = Field(default=None, max_length=200)
    source_ref: str | None = Field(default=None, max_length=2000)
    tags: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_env_mutex(self) -> "MemInboxSendRequest":
        return validate_optional_env_ref_pair(self)


class MemInboxSendResponse(BaseModel):
    """Result of :class:`MemInboxSendRequest`.

    ``reference`` echoes the canonical URL so the agent can confirm the
    destination to the user. ``id`` / ``version`` identify the new
    message memory; ``recipient_entity_id`` identifies the channel.
    """

    id: UUID
    version: int
    reference: str
    recipient_entity_id: UUID
    expires_at: dt.datetime | None


class MemInboxRequest(BaseModel):
    """List messages from a channel.

    See :class:`MemInboxSendRequest.to` for the ``to`` field shape.

    ``cursor`` is an opaque keyset cursor returned by a prior call;
    pass it back verbatim to continue paging. ``order='desc'`` returns
    newest first (the default for the user-orchestrated "check inbox"
    pattern).

    ``include_expired=True`` returns messages past their TTL —
    primarily for debugging.
    """

    model_config = ConfigDict(extra="forbid")

    to: str = Field(min_length=1, max_length=512)
    env_id: UUID | None = None
    env_name: str | None = None
    cursor: str | None = Field(default=None, max_length=256)
    limit: int = Field(default=20, ge=1, le=100)
    include_expired: bool = False
    order: Literal["desc", "asc"] = "desc"

    @model_validator(mode="after")
    def _validate_env_mutex(self) -> "MemInboxRequest":
        return validate_optional_env_ref_pair(self)


class MemInboxItem(BaseModel):
    """One message returned by :class:`MemInboxResponse`.

    Slim projection of the underlying memory; callers wanting the full
    memory shape (lineage, sources, tags) should follow up with
    ``mem_get(id)``.
    """

    id: UUID
    title: str | None
    body: str
    expires_at: dt.datetime | None
    created_at: dt.datetime
    display_from: str | None
    source_ref: str | None
    sender_agent_id: UUID | None
    tags: list[str] = Field(default_factory=list)


class MemInboxResponse(BaseModel):
    """Result of :class:`MemInboxRequest`.

    ``next_cursor`` is non-None iff ``has_more`` is True. Cursors are
    opaque base64-url-encoded ``(created_at, id)`` keyset pairs;
    clients pass them through unmodified.
    """

    items: list[MemInboxItem]
    next_cursor: str | None
    has_more: bool
    count: int
    reference: str


__all__ = (
    "REFERENCE_SCHEME",
    "INBOX_TAG",
    "DEFAULT_TTL_DAYS",
    "MAX_TTL_DAYS",
    "MemInboxOpenRequest",
    "MemInboxOpenResponse",
    "MemInboxSendRequest",
    "MemInboxSendResponse",
    "MemInboxRequest",
    "MemInboxResponse",
    "MemInboxItem",
)
