"""Schema coverage for the v0.10 mem_stats surface."""

from __future__ import annotations

from uuid import uuid4

import pytest
from memory_mcp_schemas.stats import MemStatsRequest, MemStatsResponse
from pydantic import ValidationError

from memory_mcp.env_resolve import _resolve_env_refs
from memory_mcp.errors import EnvRefBothProvidedError


def test_request_defaults() -> None:
    req = MemStatsRequest()

    assert req.env_ids is None
    assert req.env_names is None
    assert req.global_ is False
    assert req.include_substrates is False
    assert req.include_body_bytes is True
    assert req.include_distributions is True
    assert req.tag_top_k == 20


def test_global_alias_round_trip() -> None:
    req = MemStatsRequest.model_validate({"global": True})

    assert req.global_ is True
    assert req.model_dump(by_alias=True)["global"] is True


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        MemStatsRequest(watch=True)  # type: ignore[call-arg]


def test_tag_top_k_bounds() -> None:
    MemStatsRequest(tag_top_k=0)
    MemStatsRequest(tag_top_k=500)
    with pytest.raises(ValidationError):
        MemStatsRequest(tag_top_k=-1)
    with pytest.raises(ValidationError):
        MemStatsRequest(tag_top_k=501)


@pytest.mark.asyncio
async def test_env_ids_env_names_mutual_exclusion_resolved_before_tool_body() -> None:
    with pytest.raises(EnvRefBothProvidedError) as exc_info:
        await _resolve_env_refs(MemStatsRequest(env_ids=[uuid4()], env_names=["cdp"]))

    assert exc_info.value.code == "ENV_REF_BOTH_PROVIDED"


def test_response_defaults_are_stable() -> None:
    out = MemStatsResponse()

    assert out.schema_version == 1
    assert out.memories.total == 0
    assert out.envs.total == 0
    assert out.degraded_substrates == []
    assert out.degraded_sections == []
