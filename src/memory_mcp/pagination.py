"""Generic keyset-cursor pagination for browse-style tools.

Sprint A introduces four browse/facet tools (``mem_browse``,
``mem_facets``, ``ent_browse``, ``rel_browse``). All of them share the
same pagination contract: keyset cursors over ``(order_value, id)``
pairs, with a filter fingerprint baked into the cursor so callers cannot
silently change the filter set mid-pagination.

Public surface:

* :func:`compute_filter_fingerprint` — stable 16-char hash of a
  filter dict. Identical inputs → identical fingerprints across process
  boundaries (sorted JSON, no environment-dependent ordering).
* :func:`encode_cursor` — pack ``(filter_fingerprint, order_field,
  order_value, id, direction)`` into an opaque url-safe base64 string.
* :func:`decode_cursor` — round-trip; verifies the embedded fingerprint
  matches the *current* query's filter set, raising
  :class:`InvalidCursorError` on mismatch (corruption, schema drift,
  filter change).

Why not offset/limit: offset pagination is O(n) on Postgres and gets
worse as pages grow; keyset is O(log n) using the supporting index.
:class:`InvalidCursorError` callers should drop the cursor and re-page
from the start.

Why fingerprints: changing filters mid-pagination would silently return
a mixed-filter page (rows that no longer match). Embedding a fingerprint
turns that into a fast error rather than a subtle correctness bug.

Cursor wire format (``schema_version=1``)::

    {
        "sv": 1,                # schema version
        "fp": "0123456789abcdef",  # 16-char filter fingerprint
        "ob": "updated_at",     # order field name
        "ov": "2026-05-10T...", # order value as ISO-8601 (or string)
        "id": "uuid-string",    # tiebreak record id (uuid as str)
        "d":  "desc"            # "asc" | "desc"
    }

Encoded as compact JSON → utf-8 → urlsafe base64 (no padding).
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel

from memory_mcp.errors import InvalidCursorError

__all__ = [
    "Direction",
    "KeysetCursor",
    "compute_filter_fingerprint",
    "encode_cursor",
    "decode_cursor",
]


Direction = Literal["asc", "desc"]
SCHEMA_VERSION = 1


class KeysetCursor(BaseModel):
    """Decoded cursor payload returned by :func:`decode_cursor`."""

    filter_fingerprint: str
    order_field: str
    order_value: str  # ISO-8601 for datetimes; raw str for text orders
    tiebreak_id: UUID
    direction: Direction


def _normalize_for_fingerprint(value: Any) -> Any:
    """Recursively coerce a filter value into a JSON-stable shape.

    UUIDs → str; datetimes → ISO-8601 in UTC; lists are sorted when the
    elements are sortable (so ``[A, B]`` and ``[B, A]`` produce the same
    fingerprint — order does not affect filter semantics for any
    Sprint A filter). Dict keys are always sorted by :func:`json.dumps`
    with ``sort_keys=True``.
    """
    if value is None:
        return None
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        return value.astimezone(dt.UTC).isoformat()
    if isinstance(value, (list, tuple)):
        normalized = [_normalize_for_fingerprint(v) for v in value]
        try:
            return sorted(normalized, key=lambda x: (x is None, json.dumps(x, sort_keys=True)))
        except TypeError:
            return normalized
    if isinstance(value, dict):
        return {k: _normalize_for_fingerprint(v) for k, v in value.items()}
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    # Fall back to repr for anything exotic; keeps fingerprint stable
    # within a process and surfaces the issue if someone passes an
    # unexpected type.
    return repr(value)


def compute_filter_fingerprint(filter_dict: dict[str, Any]) -> str:
    """Return a deterministic 16-char fingerprint of ``filter_dict``.

    Equal inputs (modulo JSON-stable normalisation) produce equal
    fingerprints across processes. Different inputs are extremely
    unlikely to collide (16 hex chars = 64 bits of SHA-256).
    """
    normalized = _normalize_for_fingerprint(filter_dict)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def _order_value_to_str(value: Any) -> str:
    """Coerce an ``order_value`` to its cursor wire form."""
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        return value.astimezone(dt.UTC).isoformat()
    if isinstance(value, UUID):
        return str(value)
    return str(value)


def encode_cursor(
    *,
    filter_fingerprint: str,
    order_field: str,
    order_value: Any,
    tiebreak_id: UUID,
    direction: Direction,
) -> str:
    """Pack a keyset cursor into an opaque urlsafe-base64 string."""
    payload = {
        "sv": SCHEMA_VERSION,
        "fp": filter_fingerprint,
        "ob": order_field,
        "ov": _order_value_to_str(order_value),
        "id": str(tiebreak_id),
        "d": direction,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_cursor(
    cursor: str,
    *,
    expected_fingerprint: str,
    expected_order_field: str | None = None,
    expected_direction: Direction | None = None,
) -> KeysetCursor:
    """Decode + validate a cursor against the current query shape.

    Raises :class:`InvalidCursorError` on:

    * Malformed base64 / JSON.
    * Schema-version mismatch (forward-incompat cursor).
    * Filter-fingerprint mismatch (caller changed filters mid-page).
    * Order-field or direction mismatch (caller changed ``order_by``).
    * Missing required keys.
    """
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(cursor.encode("ascii") + padding.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidCursorError(f"INVALID_CURSOR: malformed cursor: {exc}") from exc

    if not isinstance(payload, dict):
        raise InvalidCursorError("INVALID_CURSOR: cursor payload must be an object")

    sv = payload.get("sv")
    if sv != SCHEMA_VERSION:
        raise InvalidCursorError(
            f"INVALID_CURSOR: cursor schema_version {sv!r} != server {SCHEMA_VERSION}",
        )

    missing = [k for k in ("fp", "ob", "ov", "id", "d") if k not in payload]
    if missing:
        raise InvalidCursorError(f"INVALID_CURSOR: missing keys: {missing}")

    if payload["fp"] != expected_fingerprint:
        raise InvalidCursorError(
            "INVALID_CURSOR: filter fingerprint mismatch — caller changed filters mid-pagination",
        )

    if expected_order_field is not None and payload["ob"] != expected_order_field:
        raise InvalidCursorError(
            f"INVALID_CURSOR: order_field mismatch (cursor={payload['ob']!r}, "
            f"request={expected_order_field!r})",
        )

    if expected_direction is not None and payload["d"] != expected_direction:
        raise InvalidCursorError(
            f"INVALID_CURSOR: direction mismatch (cursor={payload['d']!r}, "
            f"request={expected_direction!r})",
        )

    try:
        tiebreak = UUID(payload["id"])
    except (ValueError, TypeError) as exc:
        raise InvalidCursorError(f"INVALID_CURSOR: bad tiebreak id: {exc}") from exc

    return KeysetCursor(
        filter_fingerprint=payload["fp"],
        order_field=payload["ob"],
        order_value=str(payload["ov"]),
        tiebreak_id=tiebreak,
        direction=payload["d"],
    )
