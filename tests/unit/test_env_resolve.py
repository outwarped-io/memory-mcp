"""Unit tests for friendly env-name resolution."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from memory_mcp import env_resolve
from memory_mcp.env_resolve import _resolve_env_refs
from memory_mcp.errors import (
    EnvNotFoundError,
    EnvRefAmbiguousError,
    EnvRefBothProvidedError,
)
from memory_mcp_schemas.search import MemorySearchRequest


class SingleEnvRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_id: UUID | None = None
    env_name: str | None = None


async def test_resolve_passes_through_when_only_ids() -> None:
    env_id = uuid4()
    request = MemorySearchRequest(env_ids=[env_id])

    resolved = await _resolve_env_refs(request)

    assert resolved is request
    assert resolved.env_ids == [env_id]
    assert resolved.env_names is None


async def test_resolve_translates_names_to_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()

    async def fake_lookup(name: str, *, include_deleted: bool = False):
        assert name == "cdp"
        assert include_deleted is False
        return SimpleNamespace(id=env_id)

    monkeypatch.setattr(env_resolve, "get_env_by_name_ci", fake_lookup)

    resolved = await _resolve_env_refs(MemorySearchRequest(env_names=["cdp"]))

    assert resolved.env_ids == [env_id]
    assert resolved.env_names is None


async def test_resolve_rejects_both_provided_list() -> None:
    with pytest.raises(EnvRefBothProvidedError) as exc_info:
        await _resolve_env_refs(MemorySearchRequest(env_ids=[uuid4()], env_names=["cdp"]))

    assert exc_info.value.code == "ENV_REF_BOTH_PROVIDED"
    assert exc_info.value.details == {"field": "env_list"}


async def test_resolve_rejects_both_provided_single() -> None:
    with pytest.raises(EnvRefBothProvidedError) as exc_info:
        await _resolve_env_refs(SingleEnvRequest(env_id=uuid4(), env_name="cdp"))

    assert exc_info.value.code == "ENV_REF_BOTH_PROVIDED"
    assert exc_info.value.details == {"field": "env"}


async def test_resolve_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()

    async def fake_lookup(name: str, *, include_deleted: bool = False):
        assert name == "CDP"
        assert name.lower() == "cdp"
        assert include_deleted is False
        return SimpleNamespace(id=env_id)

    monkeypatch.setattr(env_resolve, "get_env_by_name_ci", fake_lookup)

    resolved = await _resolve_env_refs(MemorySearchRequest(env_names=["CDP"]))

    assert resolved.env_ids == [env_id]
    assert resolved.env_names is None


async def test_resolve_unknown_name_raises_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_lookup(name: str, *, include_deleted: bool = False):
        raise EnvNotFoundError(name=name)

    monkeypatch.setattr(env_resolve, "get_env_by_name_ci", fake_lookup)

    with pytest.raises(EnvNotFoundError) as exc_info:
        await _resolve_env_refs(MemorySearchRequest(env_names=["does-not-exist"]))

    assert exc_info.value.code == "ENV_NOT_FOUND"
    assert exc_info.value.details == {"name": "does-not-exist"}


async def test_resolve_deleted_excluded_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_lookup(name: str, *, include_deleted: bool = False):
        assert name == "deleted-env"
        assert include_deleted is False
        raise EnvNotFoundError(name=name)

    monkeypatch.setattr(env_resolve, "get_env_by_name_ci", fake_lookup)

    with pytest.raises(EnvNotFoundError):
        await _resolve_env_refs(MemorySearchRequest(env_names=["deleted-env"]))


async def test_resolve_deleted_included_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()

    async def fake_lookup(name: str, *, include_deleted: bool = False):
        assert name == "deleted-env"
        assert include_deleted is True
        return SimpleNamespace(id=env_id)

    monkeypatch.setattr(env_resolve, "get_env_by_name_ci", fake_lookup)

    resolved = await _resolve_env_refs(
        MemorySearchRequest(env_names=["deleted-env"]),
        allow_deleted=True,
    )

    assert resolved.env_ids == [env_id]
    assert resolved.env_names is None


async def test_resolve_ambiguous_when_deleted_and_active_share_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_ids = [uuid4(), uuid4()]

    async def fake_lookup(name: str, *, include_deleted: bool = False):
        assert name == "cdp"
        assert include_deleted is True
        raise EnvRefAmbiguousError(name=name, candidate_ids=candidate_ids)

    monkeypatch.setattr(env_resolve, "get_env_by_name_ci", fake_lookup)

    with pytest.raises(EnvRefAmbiguousError) as exc_info:
        await _resolve_env_refs(MemorySearchRequest(env_names=["cdp"]), allow_deleted=True)

    assert exc_info.value.code == "ENV_REF_AMBIGUOUS"
    assert exc_info.value.details == {
        "name": "cdp",
        "candidate_ids": [str(candidate_id) for candidate_id in candidate_ids],
    }


async def test_resolve_preserves_order_in_list(monkeypatch: pytest.MonkeyPatch) -> None:
    env_ids = {"a": uuid4(), "b": uuid4()}
    seen: list[str] = []

    async def fake_lookup(name: str, *, include_deleted: bool = False):
        seen.append(name)
        return SimpleNamespace(id=env_ids[name])

    monkeypatch.setattr(env_resolve, "get_env_by_name_ci", fake_lookup)

    resolved = await _resolve_env_refs(MemorySearchRequest(env_names=["b", "a"]))

    assert seen == ["b", "a"]
    assert resolved.env_ids == [env_ids["b"], env_ids["a"]]
    assert resolved.env_names is None
