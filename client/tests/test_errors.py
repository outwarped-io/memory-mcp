"""Typed error parsing and dispatcher translation coverage."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from memory_mcp_client._session import call_tool
from memory_mcp_client.errors import (
    AlreadyExistsError,
    AuthError,
    CycleDetectedError,
    EmbeddingModelMismatchError,
    EnvAmbiguousError,
    EnvNotAttachedError,
    ForbiddenEnvError,
    GraphBackendUnavailableError,
    InternalError,
    InvalidCursorError,
    InvalidInputError,
    InvalidTransitionError,
    LLMUnavailableError,
    MemoryMCPError,
    NotFoundError,
    SessionRequiredError,
    UnauthorizedError,
    VersionConflictError,
    parse_error,
)
from tests.conftest import FakeCallToolResult, FakeTextBlock, make_memory_payload


SERVER_ERROR_CASES = [
    ("VERSION_CONFLICT", VersionConflictError),
    ("INVALID_TRANSITION", InvalidTransitionError),
    ("NOT_FOUND", NotFoundError),
    ("ALREADY_EXISTS", AlreadyExistsError),
    ("ENV_AMBIGUOUS", EnvAmbiguousError),
    ("ENV_NOT_ATTACHED", EnvNotAttachedError),
    ("SESSION_REQUIRED", SessionRequiredError),
    ("EMBEDDING_MODEL_MISMATCH", EmbeddingModelMismatchError),
    ("INVALID_CURSOR", InvalidCursorError),
    ("INVALID_INPUT", InvalidInputError),
    ("CYCLE_DETECTED", CycleDetectedError),
    ("GRAPH_BACKEND_UNAVAILABLE", GraphBackendUnavailableError),
    ("LLM_UNAVAILABLE", LLMUnavailableError),
    ("UNAUTHORIZED", UnauthorizedError),
    ("FORBIDDEN_ENV", ForbiddenEnvError),
    ("INTERNAL", InternalError),
]


@pytest.mark.parametrize(("code", "expected_class"), SERVER_ERROR_CASES)
def test_parse_error_maps_each_code_to_its_class(
    code: str,
    expected_class: type[MemoryMCPError],
) -> None:
    details = {"code": code, "retryable": False}

    err = parse_error(f"[{code}] msg :: {json.dumps(details)}")

    assert type(err) is expected_class
    assert err.code == code
    assert err.details == details


def test_parse_error_without_json_details() -> None:
    err = parse_error("[NOT_FOUND] msg")

    assert type(err) is NotFoundError
    assert err.details == {}


def test_parse_error_with_malformed_json() -> None:
    err = parse_error("[NOT_FOUND] msg :: {not json}")

    assert type(err) is NotFoundError
    assert err.details == {}


def test_parse_error_with_unknown_code() -> None:
    err = parse_error("[WAT_NEW_CODE] msg")

    assert isinstance(err, MemoryMCPError)
    assert err.code == "WAT_NEW_CODE"


def test_parse_error_plain_message() -> None:
    err = parse_error("plain failure")

    assert type(err) is MemoryMCPError
    assert err.code == "INTERNAL"
    assert err.details == {}


def test_convenience_aliases_resolve() -> None:
    assert issubclass(AuthError, UnauthorizedError)
    assert issubclass(AuthError, MemoryMCPError)


@pytest.mark.asyncio
async def test_call_tool_translates_raised_tool_error(client, fake_session) -> None:
    fake_session.set_error("mem_get", "[NOT_FOUND] no such memory")

    with pytest.raises(NotFoundError):
        await client.memories.get(uuid4())


@pytest.mark.asyncio
async def test_call_tool_translates_iserror_result(client, fake_session) -> None:
    fake_session.set_iserror("mem_get", "[VERSION_CONFLICT] mismatch")

    with pytest.raises(VersionConflictError):
        await client.memories.get(uuid4())


@pytest.mark.asyncio
async def test_call_tool_preserves_error_subclass_on_re_raise() -> None:
    expected = NotFoundError("already parsed")

    class TypedErrorSession:
        async def call_tool(self, name, arguments=None):
            raise expected

    with pytest.raises(NotFoundError) as exc_info:
        await call_tool(TypedErrorSession(), "mem_get", {"memory_id": str(uuid4())})

    assert exc_info.value is expected


@pytest.mark.asyncio
async def test_call_tool_with_no_structured_content(client, fake_session) -> None:
    memory_id = str(uuid4())
    fake_session.set_raw_result(
        "mem_get",
        FakeCallToolResult(
            content=[FakeTextBlock(text=json.dumps(make_memory_payload(id=memory_id)))],
            structuredContent=None,
        ),
    )

    out = await client.memories.get(memory_id)

    assert str(out.id) == memory_id
    assert out.title == "fake-memory"
