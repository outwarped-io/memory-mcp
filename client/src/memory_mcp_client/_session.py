"""MCP session wrapper and tool-call dispatcher for the memory-mcp client SDK.

This module owns the actual ``mcp.client.streamable_http`` plumbing so
the public :class:`memory_mcp_client.client.MemoryClient` can stay
declarative.

The dispatcher (``_call_tool``) handles three responsibilities:

1. Merge client-level identity defaults (``agent_id`` /
   ``attached_env_ids`` / ``attached_env_names``) into the per-call payload unless the caller
   supplied their own.
2. Invoke ``session.call_tool(name, payload)`` and translate any error
   into a typed :class:`memory_mcp_client.errors.MemoryMCPError`.
3. Extract the structured response — preferring ``structuredContent``
   when the server emits it (Pydantic ``model_dump`` is enabled), falling
   back to JSON-parsing the first ``TextContent`` block. If a Pydantic
   ``model`` is supplied, validate the response into that class;
   otherwise return the raw value.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel

from memory_mcp_client.errors import MemoryMCPError, parse_error

T = TypeVar("T", bound=BaseModel)


class _ClientSessionProtocol(Protocol):
    """Subset of ``mcp.ClientSession`` we actually use."""

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any: ...  # pragma: no cover


def _coerce_identity(value: Any) -> Any:
    """JSON-serialize UUIDs that may live in identity defaults."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, list):
        return [_coerce_identity(v) for v in value]
    return value


def _build_payload(
    base: dict[str, Any] | None,
    *,
    agent_id_default: UUID | str | None,
    env_ids_default: list[UUID | str] | None,
    env_names_default: list[str] | None = None,
) -> dict[str, Any]:
    """Merge identity defaults into ``base`` without clobbering explicit keys."""
    payload: dict[str, Any] = dict(base or {})
    if "agent_id" not in payload and agent_id_default is not None:
        payload["agent_id"] = _coerce_identity(agent_id_default)
    if "attached_env_ids" not in payload and "attached_env_names" not in payload:
        if env_ids_default:
            payload["attached_env_ids"] = _coerce_identity(env_ids_default)
        elif env_names_default:
            payload["attached_env_names"] = _coerce_identity(env_names_default)
    return payload


def _extract_structured(result: Any) -> Any:
    """Pull the structured response out of an ``mcp`` ``CallToolResult``.

    Order of preference:

    1. ``result.structuredContent`` (server-side ``model_dump`` path —
       this is what memory-mcp's ``_dump`` helper produces for every
       tool). Newer mcp SDK shapes return ``structuredContent`` as a
       dict; some return it under a ``"result"`` key wrapper —
       unwrap that.
    2. The first ``content`` block's ``text`` parsed as JSON.
    3. The first ``content`` block's ``text`` returned as-is (rare —
       happens for tools that return a plain string).
    """

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured

    content = getattr(result, "content", None) or []
    if not content:
        return None
    first = content[0]
    text = getattr(first, "text", None)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _error_text(result: Any) -> str:
    """Pull a human-readable error message from a failing call result."""
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            return text
    return repr(result)


async def call_tool[T: BaseModel](
    session: _ClientSessionProtocol,
    name: str,
    payload: dict[str, Any],
    *,
    model: type[T] | None = None,
) -> Any:
    """Invoke an MCP tool and return its parsed result.

    Raises:
        MemoryMCPError (or a subclass) when the server returned an error
        or when the MCP layer surfaced an ``isError`` result.
    """

    try:
        result = await session.call_tool(name, payload)
    except MemoryMCPError:
        raise
    except Exception as exc:  # noqa: BLE001 — translate everything
        # ``ToolError`` from the MCP SDK shows up here when the server
        # raises an error inside the tool body. Translate it into our
        # typed hierarchy by parsing the embedded ``[CODE]`` prefix.
        message = str(exc) or repr(exc)
        raise parse_error(message) from exc

    if getattr(result, "isError", False):
        raise parse_error(_error_text(result))

    payload_out = _extract_structured(result)
    if model is None:
        return payload_out
    if payload_out is None:
        # A few tools (e.g. memory_archive returning `{}`) intentionally
        # produce empty payloads. Materialize an empty model instance so
        # callers always get the declared return type.
        return model.model_validate({})
    if isinstance(payload_out, list):
        # Some tools return lists (e.g. ``ent_resolve``); when the model
        # describes a single object, wrap the list under ``{"items": ...}``
        # if the model has an ``items`` field, otherwise return as-is.
        return [model.model_validate(item) for item in payload_out]
    return model.model_validate(payload_out)
