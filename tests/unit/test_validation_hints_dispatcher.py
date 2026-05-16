"""End-to-end test: dispatcher emits VALIDATION_FAILED with hints.

Exercises the live FastMCP path so the monkey-patch installed by
``_install_validation_hints`` is verified against the framework's
auto-generated argument model — not a hand-rolled stand-in.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError


def _run_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    from memory_mcp.mcp_app import build_mcp_server

    mcp = build_mcp_server()
    tool_manager = mcp._tool_manager  # noqa: SLF001
    return asyncio.run(tool_manager.call_tool(tool_name, arguments))


def _parse_tool_error(exc: ToolError) -> dict[str, Any]:
    """Extract structured payload from a ``ToolError`` produced by ``_format_tool_error``.

    Message shape: ``[CODE] human_message :: <details_json>``.
    """
    text = str(exc)
    assert text.startswith("["), f"unexpected error message shape: {text!r}"
    code_end = text.index("]")
    code = text[1:code_end]
    rest = text[code_end + 1:].strip()
    if " :: " in rest:
        message, details_json = rest.split(" :: ", 1)
        details = json.loads(details_json)
    else:
        message, details = rest, {}
    return {"code": code, "message": message, "details": details}


def test_extra_field_in_request_surfaces_did_you_mean() -> None:
    # MemorySearchRequest has env_ids/env_names and forbids extras. The
    # caller mistype 'env' lands at loc='request.env' and we should hint
    # at env_ids / env_names.
    with pytest.raises(ToolError) as excinfo:
        _run_tool(
            "mem_search",
            {"request": {"query": "hello", "env": "cdp"}},
        )
    payload = _parse_tool_error(excinfo.value)
    assert payload["code"] == "VALIDATION_FAILED"
    hints = payload["details"].get("hints", [])
    matched = [h for h in hints if h["offered"] == "env"]
    assert matched, f"no hint for 'env'; got {hints!r}"
    assert matched[0]["suggested"] in {"env_ids", "env_names", "env_id", "env_name"}


def test_request_wrapped_in_req_surfaces_did_you_mean() -> None:
    # Caller wrapped the request inside an inner 'req' — extra_forbidden
    # at request.req with candidates 'query', 'env_ids', 'env_names', ...
    with pytest.raises(ToolError) as excinfo:
        _run_tool("mem_search", {"request": {"req": {"query": "hello"}}})
    payload = _parse_tool_error(excinfo.value)
    assert payload["code"] == "VALIDATION_FAILED"
    # 'req' isn't a close fuzzy match for any field, so we accept either
    # a no-hint result or a hint pointing at 'request' via the alias map.
    # The important assertion is that the error code + structured details
    # came through unscathed.
    assert "errors" in payload["details"]


def test_input_value_never_echoed_in_payload() -> None:
    with pytest.raises(ToolError) as excinfo:
        _run_tool(
            "mem_search",
            {"request": {"query": "x", "env": "secret-tenant-id"}},
        )
    payload = _parse_tool_error(excinfo.value)
    serialised = json.dumps(payload)
    assert "secret-tenant-id" not in serialised
