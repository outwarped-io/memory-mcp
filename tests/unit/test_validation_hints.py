"""Unit tests for the did-you-mean hint engine."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from memory_mcp.validation_hints import (
    build_hints,
    format_message,
    safe_error_payload,
)


class InnerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    limit: int = 10
    env_id: str | None = None
    env_name: str | None = None


class OuterModel(BaseModel):
    """Mirrors FastMCP's auto-generated argument model shape."""

    model_config = ConfigDict(extra="forbid")
    request: InnerRequest
    agent_id: str | None = None


def _validate(payload: dict) -> ValidationError:
    try:
        OuterModel.model_validate(payload)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected ValidationError")


def test_alias_env_to_env_id() -> None:
    err = _validate({"request": {"query": "x", "env": "cdp"}})
    hints = build_hints(OuterModel, err)
    suggestions = {h["offered"]: (h["suggested"], h["source"]) for h in hints}
    assert "env" in suggestions
    suggested, source = suggestions["env"]
    assert suggested in {"env_id", "env_name"}
    assert source == "alias"


def test_alias_req_top_level() -> None:
    # caller wraps request in `req:` (the bug class memory-mcp errored on)
    err = _validate({"req": {"query": "x"}, "request": {"query": "x"}})
    hints = build_hints(OuterModel, err)
    sugg = next((h for h in hints if h["offered"] == "req"), None)
    assert sugg is not None
    assert sugg["suggested"] == "request"
    assert sugg["source"] == "alias"


def test_fuzzy_typo_within_levenshtein() -> None:
    err = _validate({"request": {"query": "x", "limt": 5}})
    hints = build_hints(OuterModel, err)
    sugg = next((h for h in hints if h["offered"] == "limt"), None)
    assert sugg is not None
    assert sugg["suggested"] == "limit"
    assert sugg["source"] == "fuzzy"
    assert sugg["confidence"] >= 0.7


def test_no_hint_for_far_offered() -> None:
    err = _validate({"request": {"query": "x", "completely_unrelated_key": 1}})
    hints = build_hints(OuterModel, err)
    assert hints == []


def test_loc_path_includes_full_nesting() -> None:
    err = _validate({"request": {"query": "x", "env": "cdp"}})
    hints = build_hints(OuterModel, err)
    sugg = next((h for h in hints if h["offered"] == "env"), None)
    assert sugg is not None
    assert sugg["loc"] == ["request", "env"]


def test_safe_payload_strips_input_value() -> None:
    err = _validate({"request": {"query": "x", "env": "secret-tenant-id"}})
    payload = safe_error_payload(err)
    for entry in payload:
        for key in entry:
            assert key != "input"
            assert key != "input_value"
        if "ctx" in entry:
            assert "input_value" not in entry["ctx"]


def test_format_message_includes_single_high_confidence_hint() -> None:
    err = _validate({"request": {"query": "x", "env": "cdp"}})
    errors = safe_error_payload(err)
    hints = build_hints(OuterModel, err)
    msg = format_message("mem_search", errors, hints)
    assert "VALIDATION_FAILED" in msg
    assert "did you mean" in msg


def test_format_message_silent_for_zero_hints() -> None:
    err = _validate({"request": {"query": "x", "completely_unrelated_key": 1}})
    errors = safe_error_payload(err)
    hints = build_hints(OuterModel, err)
    msg = format_message("mem_search", errors, hints)
    assert "did you mean" not in msg


def test_missing_required_field_no_spurious_hint() -> None:
    # 'request' is required but missing — no fuzzy hint should fire
    # because there's nothing to spell-check.
    err = _validate({})
    hints = build_hints(OuterModel, err)
    # Either zero hints or no hint whose offered == 'request'
    assert not any(h["offered"] == "request" for h in hints)
