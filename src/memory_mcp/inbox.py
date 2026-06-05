"""v0.17 inbox / drop-box implementation layer.

Three tools wrap existing primitives:

* :func:`mem_inbox_open` wraps :func:`memory_mcp.entities.entity_upsert`
  with ``kind="channel"`` (see :data:`CHANNEL_ENTITY_KIND`). Generates a
  pronounceable slug when ``name`` is omitted. Returns a
  copy-pasteable ``mem-inbox://<env-name>/<slug>`` reference.

* :func:`mem_inbox_send` wraps :func:`memory_mcp.memories.memory_write`
  with ``kind=MemoryKind.message``, the channel entity in
  ``entity_links``, the ``inbox`` tag, and a default 7-day TTL.
  **Rejects** non-existent slugs — explicit
  :func:`mem_inbox_open` required first.

* :func:`mem_inbox` is internal SQL — graph-joined query against
  ``relations`` (``type='mentions'``) to find every message anchored to
  the channel entity. ``mem_browse`` would not work: its ``tags``
  filter is OR semantics and we need ``AND`` between
  ``kind='message'`` and the entity link.

Reference URL format ``mem-inbox://<env-name>/<slug>``:

* Slug is the entity's ``canonical_name`` (kebab-case, 1–64 chars).
* Server is the only producer of well-formed references; clients pass
  the response string back verbatim.
* Bare slugs accepted on ``to`` fields when ``env_id`` or
  ``env_name`` arg is provided.

**UC2 invariant** — when ``to`` carries the URL form AND the caller
also passes ``env_id``/``env_name``, both must resolve to the same
env; mismatch raises :class:`InvalidInputError` rather than silently
cross-env-writing.

The agent surface and the three user-orchestrated flows are documented
in the workspace overlay ``mem-inbox.instructions.md`` and in the
v0.17 plan in this repo's ``plan.md``.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import logging
import random
import re
import string
from typing import Any
from uuid import UUID

from memory_mcp_schemas.entities import EntityUpsertRequest, _normalize_name
from memory_mcp_schemas.enums import MemoryKind, MemorySourceType
from memory_mcp_schemas.inbox import (
    DEFAULT_TTL_DAYS,
    INBOX_TAG,
    MAX_TTL_DAYS,
    REFERENCE_SCHEME,
    MemInboxItem,
    MemInboxOpenRequest,
    MemInboxOpenResponse,
    MemInboxRequest,
    MemInboxResponse,
    MemInboxSendRequest,
    MemInboxSendResponse,
)
from memory_mcp_schemas.memories import MemoryWriteRequest
from sqlalchemy import select

from memory_mcp._filters import exclude_expired_clause
from memory_mcp.db.models import Entity, GraphNode, Memory, MemoryTag, Relation, Tag
from memory_mcp.db.postgres import session_scope
from memory_mcp.entities import (
    CHANNEL_ENTITY_KIND,
    _resolve_env_id,
    entity_upsert,
)
from memory_mcp.env_resolve import _resolve_env_refs
from memory_mcp.envs import get_env_by_id, get_env_by_name_ci
from memory_mcp.errors import (
    AlreadyExistsError,
    InvalidCursorError,
    InvalidInputError,
    NotFoundError,
)
from memory_mcp.identity import AgentContext
from memory_mcp.memories import memory_write

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference parsing / formatting
# ---------------------------------------------------------------------------

# Kebab-case slug: lowercase ASCII letters + digits + internal hyphens, no
# leading/trailing hyphen, 1..64 chars. Mirrors the wordlist composition.
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SLUG_MAX_LEN = 64

# Reference URL: ``mem-inbox://<env-name>/<slug>``. Env names follow the
# same rules as envs.py case-insensitive matching; here we just split on
# the first ``/`` so any printable env name lands.
_REFERENCE_PREFIX = f"{REFERENCE_SCHEME}://"


def _validate_slug(slug: str) -> str:
    """Reject anything that's not kebab-case 1..64 chars."""
    if not slug or len(slug) > _SLUG_MAX_LEN or not _SLUG_RE.match(slug):
        raise InvalidInputError(
            "INVALID_INBOX_SLUG: slug must be kebab-case (lowercase letters/"
            f"digits with single hyphens), 1..{_SLUG_MAX_LEN} chars",
            slug=slug,
        )
    return slug


def parse_reference(value: str) -> tuple[str | None, str]:
    """Parse a ``to`` field into ``(env_name | None, slug)``.

    Accepts two forms:

    * ``mem-inbox://<env-name>/<slug>`` — env name lifted from URL.
    * ``<slug>`` — bare slug; env must come from request arg.

    Raises :class:`InvalidInputError` for malformed strings.
    """
    if not value or not isinstance(value, str):
        raise InvalidInputError("INVALID_INBOX_REFERENCE: empty 'to'")
    if value.startswith(_REFERENCE_PREFIX):
        remainder = value[len(_REFERENCE_PREFIX) :]
        if "/" not in remainder:
            raise InvalidInputError(
                f"INVALID_INBOX_REFERENCE: URL form requires '{REFERENCE_SCHEME}://<env-name>/<slug>'",
                reference=value,
            )
        env_name, _, slug = remainder.partition("/")
        if not env_name:
            raise InvalidInputError(
                "INVALID_INBOX_REFERENCE: empty env-name in URL",
                reference=value,
            )
        # An accidental trailing '/' or a second segment is a typo, not
        # a feature.
        if "/" in slug:
            raise InvalidInputError(
                "INVALID_INBOX_REFERENCE: only one slug segment allowed",
                reference=value,
            )
        _validate_slug(slug)
        return env_name, slug
    # Bare slug path.
    _validate_slug(value)
    return None, value


def format_reference(env_name: str, slug: str) -> str:
    """Server-side canonical URL emission.

    Clients never compose this themselves — they receive it from the
    open/send/list responses and pass it back verbatim.
    """
    return f"{_REFERENCE_PREFIX}{env_name}/{slug}"


# ---------------------------------------------------------------------------
# Slug generation — curated adjective/noun wordlist
# ---------------------------------------------------------------------------

# Pronounceable, dictation-friendly. ~100 × ~100 = ~10k unique combos
# before re-roll. Kept inline to avoid a data file.
_ADJECTIVES: tuple[str, ...] = (
    "able",
    "agile",
    "alert",
    "amber",
    "ample",
    "arctic",
    "azure",
    "blue",
    "bold",
    "brave",
    "brisk",
    "bright",
    "calm",
    "candid",
    "cheerful",
    "civic",
    "clear",
    "clever",
    "cool",
    "cosmic",
    "cozy",
    "crisp",
    "curious",
    "daring",
    "deep",
    "dewy",
    "early",
    "eager",
    "easy",
    "elfin",
    "ember",
    "emerald",
    "epic",
    "even",
    "fancy",
    "fast",
    "fierce",
    "fine",
    "firm",
    "fluffy",
    "free",
    "fresh",
    "frosty",
    "gentle",
    "giant",
    "glad",
    "glass",
    "glossy",
    "golden",
    "grand",
    "grave",
    "happy",
    "humble",
    "icy",
    "ivory",
    "jade",
    "jolly",
    "keen",
    "kind",
    "lively",
    "lofty",
    "lone",
    "loyal",
    "lucent",
    "lucky",
    "mellow",
    "merry",
    "mighty",
    "mild",
    "misty",
    "mossy",
    "neat",
    "nimble",
    "noble",
    "olive",
    "open",
    "patient",
    "peach",
    "pearl",
    "perky",
    "plain",
    "polar",
    "proud",
    "quick",
    "quiet",
    "ready",
    "regal",
    "ripe",
    "rosy",
    "royal",
    "rustic",
    "sage",
    "scarlet",
    "shy",
    "silent",
    "silver",
    "sleek",
    "smart",
    "smooth",
    "snappy",
    "snug",
    "solar",
    "spry",
    "still",
    "sturdy",
    "subtle",
    "sunny",
    "swift",
    "tame",
    "tender",
    "tidy",
    "topaz",
    "tough",
    "true",
    "warm",
    "wild",
    "wise",
    "witty",
    "young",
    "zealous",
)

_NOUNS: tuple[str, ...] = (
    "ant",
    "ape",
    "badger",
    "bass",
    "bat",
    "bear",
    "bee",
    "beetle",
    "bird",
    "bison",
    "boar",
    "buck",
    "bunny",
    "calf",
    "camel",
    "carp",
    "cat",
    "cattle",
    "cheetah",
    "chick",
    "clam",
    "cobra",
    "colt",
    "coyote",
    "crab",
    "crane",
    "crow",
    "cub",
    "deer",
    "dingo",
    "dog",
    "donkey",
    "dove",
    "drake",
    "duck",
    "eagle",
    "eel",
    "egret",
    "elk",
    "ermine",
    "falcon",
    "fawn",
    "ferret",
    "finch",
    "fish",
    "fox",
    "frog",
    "gecko",
    "gnu",
    "goose",
    "gull",
    "hare",
    "hawk",
    "hen",
    "heron",
    "horse",
    "hound",
    "ibex",
    "jay",
    "kestrel",
    "kitten",
    "lamb",
    "lark",
    "lemur",
    "leopard",
    "lion",
    "lizard",
    "llama",
    "lynx",
    "magpie",
    "mare",
    "marten",
    "mink",
    "mole",
    "moose",
    "moth",
    "mouse",
    "newt",
    "ocelot",
    "otter",
    "owl",
    "ox",
    "panda",
    "panther",
    "parrot",
    "peacock",
    "pelican",
    "pony",
    "pug",
    "puma",
    "quail",
    "rabbit",
    "raccoon",
    "ram",
    "raven",
    "robin",
    "salmon",
    "seal",
    "shark",
    "sheep",
    "shrew",
    "skunk",
    "snake",
    "sparrow",
    "spider",
    "squid",
    "stag",
    "stoat",
    "stork",
    "swan",
    "tiger",
    "toad",
    "trout",
    "turtle",
    "viper",
    "vole",
    "weasel",
    "whale",
    "wolf",
    "wren",
    "yak",
    "zebra",
)


def generate_slug(rng: random.Random | None = None) -> str:
    """Return an ``<adjective>-<noun>`` pronounceable slug.

    Collision-checking is the caller's responsibility (re-roll on
    detected dupe). ``rng`` is injected only for deterministic tests.
    """
    rng = rng or random.SystemRandom()
    return f"{rng.choice(_ADJECTIVES)}-{rng.choice(_NOUNS)}"


# ---------------------------------------------------------------------------
# Cursor encode / decode — opaque base64-url over (created_at, id)
# ---------------------------------------------------------------------------


def encode_cursor(created_at: dt.datetime, memory_id: UUID) -> str:
    """Encode a keyset cursor as a base64-url-safe JSON blob.

    Pairs ``(created_at, id)`` deterministically so a paginating
    client never sees the same item twice (id breaks ties on equal
    timestamps).
    """
    payload = {"t": created_at.isoformat(), "i": str(memory_id)}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[dt.datetime, UUID]:
    """Decode a cursor produced by :func:`encode_cursor`.

    Malformed cursors raise :class:`InvalidCursorError`.
    """
    try:
        # Pad back to a multiple of 4 — encode() strips the '='.
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
        created_at = dt.datetime.fromisoformat(payload["t"])
        memory_id = UUID(payload["i"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError(
            "INVALID_CURSOR: inbox cursor is malformed",
            cursor=cursor,
        ) from exc
    return created_at, memory_id


# ---------------------------------------------------------------------------
# Env resolution — UC2 invariant
# ---------------------------------------------------------------------------


async def _resolve_inbox_env(
    *,
    to_env_name: str | None,
    request_env_id: UUID | None,
    request_env_name: str | None,
    ctx: AgentContext,
) -> tuple[UUID, str]:
    """Resolve ``(env_id, env_name)`` honoring the UC2 invariant.

    Priority:

    1. If ``request_env_id`` set, that wins (must match ``to_env_name``
       if both present).
    2. Else if ``request_env_name`` set, resolve case-insensitively
       (must match ``to_env_name`` if both present).
    3. Else if ``to_env_name`` set (URL form), resolve it.
    4. Else fall back to ``_resolve_env_id(explicit=None, ctx=ctx)`` —
       sole-attached env or :class:`EnvAmbiguousError`.

    Returns ``(env_id, canonical_env_name)``. Callers use the canonical
    name to format the response reference so it's always normalized.
    """
    resolved_id: UUID
    resolved_name: str

    if request_env_id is not None:
        env = await get_env_by_id(request_env_id)
        if env is None:
            raise NotFoundError(
                "ENV_NOT_FOUND: env_id does not exist",
                env_id=str(request_env_id),
            )
        resolved_id = env.id
        resolved_name = env.name
    elif request_env_name is not None:
        env = await get_env_by_name_ci(request_env_name)
        resolved_id = env.id
        resolved_name = env.name
    elif to_env_name is not None:
        env = await get_env_by_name_ci(to_env_name)
        resolved_id = env.id
        resolved_name = env.name
    else:
        # Sole-attached fall-through. _resolve_env_id raises EnvAmbiguousError
        # if there isn't exactly one attached env.
        resolved_id = _resolve_env_id(explicit=None, ctx=ctx)
        env = await get_env_by_id(resolved_id)
        if env is None:
            # Should be unreachable — attached envs are validated at
            # session-setup time — but guard for the FK race.
            raise NotFoundError(
                "ENV_NOT_FOUND: resolved env id no longer exists",
                env_id=str(resolved_id),
            )
        resolved_name = env.name

    # UC2 invariant: URL env must match request env when both provided.
    if to_env_name is not None and not _env_names_equal(to_env_name, resolved_name):
        raise InvalidInputError(
            "INBOX_ENV_MISMATCH: 'to' URL env-name does not match the explicit env_id/env_name argument",
            to_env_name=to_env_name,
            resolved_env_name=resolved_name,
        )

    return resolved_id, resolved_name


def _env_names_equal(a: str, b: str) -> bool:
    """Case-insensitive env-name equality (matches :func:`get_env_by_name_ci`)."""
    return a.casefold() == b.casefold()


# ---------------------------------------------------------------------------
# Channel entity lookup
# ---------------------------------------------------------------------------


async def _find_channel_entity(
    *,
    s: Any,
    env_id: UUID,
    slug: str,
) -> Entity | None:
    """Look up the channel entity by ``(env_id, normalized_name, kind)``.

    Returns ``None`` when the channel doesn't exist. Callers either
    auto-open (open tool, idempotent) or reject (send/list tools).
    """
    normalized = _normalize_name(slug)
    stmt = select(Entity).where(
        Entity.env_id == env_id,
        Entity.normalized_name == normalized,
        Entity.kind == CHANNEL_ENTITY_KIND,
    )
    return (await s.execute(stmt)).scalar_one_or_none()


# ---------------------------------------------------------------------------
# Tool 1 — mem_inbox_open
# ---------------------------------------------------------------------------


async def mem_inbox_open(
    request: MemInboxOpenRequest,
    *,
    ctx: AgentContext,
    settings: Any | None = None,
) -> MemInboxOpenResponse:
    """Open (or look up) an inbox channel.

    Wraps :func:`memory_mcp.entities.entity_upsert` with
    ``kind=CHANNEL_ENTITY_KIND``. Generates a pronounceable slug when
    ``request.name`` is omitted; retries on collision (bounded).
    """
    # Resolve env first so we can collision-check generated slugs.
    request = await _resolve_env_refs(request)
    env_id, env_name = await _resolve_inbox_env(
        to_env_name=None,
        request_env_id=request.env_id,
        request_env_name=None,
        ctx=ctx,
    )

    if request.name is not None:
        slug = _validate_slug(request.name)
    else:
        # Generate + collision-check. With ~100×~100 wordlist a single
        # collision is rare for any reasonable env size; cap retries.
        slug = await _generate_unique_slug(env_id=env_id)

    # Detect pre-existing channel with this slug. We do this BEFORE
    # entity_upsert so we can honor idempotent=False semantics — the
    # upsert layer is "create-or-no-op-update" by design.
    async with session_scope() as s:
        existing = await _find_channel_entity(s=s, env_id=env_id, slug=slug)

    if existing is not None:
        if not request.idempotent:
            raise AlreadyExistsError(
                "INBOX_CHANNEL_EXISTS: a channel with this slug already exists; "
                "pass idempotent=True to reuse it or pick a different name",
                slug=slug,
                env_name=env_name,
                entity_id=str(existing.id),
            )
        return MemInboxOpenResponse(
            reference=format_reference(env_name, existing.canonical_name),
            entity_id=existing.id,
            canonical_name=existing.canonical_name,
            env_id=env_id,
            env_name=env_name,
            created=False,
        )

    # Create. entity_upsert manages its own session_scope and commits.
    metadata: dict[str, Any] = {}
    if request.title:
        metadata["title"] = request.title
    upsert_req = EntityUpsertRequest(
        kind=CHANNEL_ENTITY_KIND,
        canonical_name=slug,
        env_id=env_id,
        metadata=metadata,
    )
    entity = await entity_upsert(upsert_req, ctx=ctx, settings=settings)

    return MemInboxOpenResponse(
        reference=format_reference(env_name, entity.canonical_name),
        entity_id=entity.id,
        canonical_name=entity.canonical_name,
        env_id=env_id,
        env_name=env_name,
        created=True,
    )


async def _generate_unique_slug(*, env_id: UUID, max_retries: int = 8) -> str:
    """Generate a slug not already in use within ``env_id``.

    Picks a fresh ``<adjective>-<noun>`` pair, checks against the
    channel-entity table. With ~10k combinations a couple of retries
    handle any practical density.
    """
    rng = random.SystemRandom()
    async with session_scope() as s:
        for _ in range(max_retries):
            candidate = generate_slug(rng)
            existing = await _find_channel_entity(s=s, env_id=env_id, slug=candidate)
            if existing is None:
                return candidate
    # Extremely unlikely; fall back to suffixing with random digits.
    suffix = "".join(rng.choices(string.digits, k=4))
    return f"{generate_slug(rng)}-{suffix}"


# ---------------------------------------------------------------------------
# Tool 2 — mem_inbox_send
# ---------------------------------------------------------------------------


async def mem_inbox_send(
    request: MemInboxSendRequest,
    *,
    ctx: AgentContext,
    settings: Any | None = None,
) -> MemInboxSendResponse:
    """Drop a message into a channel.

    Wraps :func:`memory_mcp.memories.memory_write` with
    ``kind=MemoryKind.message``, channel entity in ``entity_links``,
    the ``inbox`` tag, and a TTL.

    Rejects non-existent slugs — explicit :func:`mem_inbox_open` is
    required first. This prevents typo-driven channel proliferation.
    """
    request = await _resolve_env_refs(request)
    to_env_name, slug = parse_reference(request.to)
    env_id, env_name = await _resolve_inbox_env(
        to_env_name=to_env_name,
        request_env_id=request.env_id,
        request_env_name=None,
        ctx=ctx,
    )

    async with session_scope() as s:
        channel = await _find_channel_entity(s=s, env_id=env_id, slug=slug)
    if channel is None:
        raise InvalidInputError(
            "INBOX_CHANNEL_NOT_FOUND: no channel with this slug exists; call mem_inbox_open first",
            slug=slug,
            env_name=env_name,
        )

    # TTL handling — default 7d, cap at 90d.
    now = dt.datetime.now(dt.UTC)
    if request.expires_at is None:
        expires_at: dt.datetime | None = now + dt.timedelta(days=DEFAULT_TTL_DAYS)
    else:
        expires_at = request.expires_at
        if expires_at.tzinfo is None:
            # Treat naive timestamps as UTC; otherwise comparison below blows up.
            expires_at = expires_at.replace(tzinfo=dt.UTC)
        max_allowed = now + dt.timedelta(days=MAX_TTL_DAYS)
        if expires_at > max_allowed:
            raise InvalidInputError(
                f"INBOX_TTL_TOO_LARGE: expires_at exceeds the {MAX_TTL_DAYS}-day cap",
                expires_at=expires_at.isoformat(),
                max_allowed=max_allowed.isoformat(),
            )
        if expires_at <= now:
            raise InvalidInputError(
                "INBOX_TTL_IN_PAST: expires_at must be in the future",
                expires_at=expires_at.isoformat(),
            )

    # Merge tags: always include 'inbox' marker plus any caller-supplied tags.
    caller_tags = list(request.tags or [])
    merged_tags = [INBOX_TAG] + [t for t in caller_tags if t != INBOX_TAG]

    # Merge metadata: caller's metadata wins for explicit keys; display_from
    # / source_ref get split out into dedicated MemInboxItem fields on read,
    # but persisted on the memory metadata blob so the message memory carries
    # the same shape regardless of how it was written.
    metadata: dict[str, Any] = dict(request.metadata)
    if request.display_from is not None:
        metadata["display_from"] = request.display_from

    write_req = MemoryWriteRequest(
        kind=MemoryKind.message,
        title=request.title,
        body=request.body,
        env_id=env_id,
        tags=merged_tags,
        entity_links=[channel.id],
        expires_at=expires_at,
        metadata=metadata,
        source_type=MemorySourceType.agent,
        source_ref=request.source_ref,
    )
    response = await memory_write(write_req, ctx=ctx, settings=settings)

    return MemInboxSendResponse(
        id=response.id,
        version=response.version,
        reference=format_reference(env_name, channel.canonical_name),
        recipient_entity_id=channel.id,
        expires_at=expires_at,
    )


# ---------------------------------------------------------------------------
# Tool 3 — mem_inbox
# ---------------------------------------------------------------------------


async def mem_inbox(
    request: MemInboxRequest,
    *,
    ctx: AgentContext,
    settings: Any | None = None,
) -> MemInboxResponse:
    """List messages in a channel.

    Internal SQL query — joins through ``graph_nodes`` + ``relations``
    (``type='mentions'``) to find every message anchored to the channel
    entity. Newest first by default; opaque keyset cursor on
    ``(created_at, id)``.
    """
    request = await _resolve_env_refs(request)
    to_env_name, slug = parse_reference(request.to)
    env_id, env_name = await _resolve_inbox_env(
        to_env_name=to_env_name,
        request_env_id=request.env_id,
        request_env_name=None,
        ctx=ctx,
    )

    async with session_scope() as s:
        channel = await _find_channel_entity(s=s, env_id=env_id, slug=slug)
        if channel is None:
            raise InvalidInputError(
                "INBOX_CHANNEL_NOT_FOUND: no channel with this slug exists; call mem_inbox_open first",
                slug=slug,
                env_name=env_name,
            )

        # Decode cursor up front so we can compose the keyset predicate.
        cursor_created_at: dt.datetime | None = None
        cursor_id: UUID | None = None
        if request.cursor is not None:
            cursor_created_at, cursor_id = decode_cursor(request.cursor)

        # Build the inbox query. We need to join through graph_nodes twice
        # (src=memory, dst=channel-entity) per the entity_links → Relation
        # mechanism in memories.py. The keyset predicate uses (created_at,
        # id) so tied timestamps don't lose rows.
        gn_src = GraphNode.__table__.alias("gn_src")
        gn_dst = GraphNode.__table__.alias("gn_dst")
        rel = Relation.__table__.alias("rel")

        order_desc = request.order == "desc"

        # Base SELECT — Memory row, restricted to the channel via the
        # relations graph.
        stmt = (
            select(Memory)
            .join(gn_src, gn_src.c.memory_id == Memory.id)
            .join(rel, rel.c.src_node_id == gn_src.c.id)
            .join(gn_dst, gn_dst.c.id == rel.c.dst_node_id)
            .where(
                Memory.env_id == env_id,
                Memory.kind == MemoryKind.message.value,
                Memory.status.in_(("proposed", "active")),
                rel.c.type == "mentions",
                gn_dst.c.entity_id == channel.id,
            )
        )

        # TTL filter — hide expired by default. ``NULL`` expires_at = never.
        if not request.include_expired:
            stmt = stmt.where(exclude_expired_clause())

        # Keyset cursor predicate.
        if cursor_created_at is not None and cursor_id is not None:
            if order_desc:
                stmt = stmt.where(
                    (Memory.created_at < cursor_created_at)
                    | ((Memory.created_at == cursor_created_at) & (Memory.id < cursor_id))
                )
            else:
                stmt = stmt.where(
                    (Memory.created_at > cursor_created_at)
                    | ((Memory.created_at == cursor_created_at) & (Memory.id > cursor_id))
                )

        # Stable ordering — tie-break on id so cursor pagination is
        # deterministic.
        if order_desc:
            stmt = stmt.order_by(Memory.created_at.desc(), Memory.id.desc())
        else:
            stmt = stmt.order_by(Memory.created_at.asc(), Memory.id.asc())

        # Over-fetch by 1 so we can compute ``has_more`` without a COUNT.
        stmt = stmt.limit(request.limit + 1)

        rows = (await s.execute(stmt)).scalars().all()

        has_more = len(rows) > request.limit
        page_rows = list(rows[: request.limit])

        # Load tags for the returned memories in one query.
        tags_by_memory: dict[UUID, list[str]] = {}
        if page_rows:
            page_ids = [row.id for row in page_rows]
            tag_stmt = (
                select(MemoryTag.memory_id, Tag.name)
                .join(Tag, Tag.id == MemoryTag.tag_id)
                .where(MemoryTag.memory_id.in_(page_ids))
            )
            for memory_id, tag_name in (await s.execute(tag_stmt)).all():
                tags_by_memory.setdefault(memory_id, []).append(tag_name)

    items: list[MemInboxItem] = []
    for row in page_rows:
        metadata = row.metadata_ or {}
        items.append(
            MemInboxItem(
                id=row.id,
                title=row.title,
                body=row.body,
                expires_at=row.expires_at,
                created_at=row.created_at,
                display_from=metadata.get("display_from"),
                source_ref=None,  # source_ref lives on memory_sources, not metadata
                sender_agent_id=None,  # MVP: see plan.md - authorship via audit log
                tags=sorted(tags_by_memory.get(row.id, [])),
            )
        )

    next_cursor: str | None = None
    if has_more and page_rows:
        last = page_rows[-1]
        next_cursor = encode_cursor(last.created_at, last.id)

    return MemInboxResponse(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
        count=len(items),
        reference=format_reference(env_name, channel.canonical_name),
    )


__all__ = (
    "REFERENCE_SCHEME",
    "INBOX_TAG",
    "DEFAULT_TTL_DAYS",
    "MAX_TTL_DAYS",
    "parse_reference",
    "format_reference",
    "generate_slug",
    "encode_cursor",
    "decode_cursor",
    "mem_inbox_open",
    "mem_inbox_send",
    "mem_inbox",
)
