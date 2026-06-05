"""Unit tests for ``memory_mcp.memories`` — pure-Python coverage.

DB-touching paths (SELECT FOR UPDATE, optimistic-lock disambiguation,
outbox payload routing, supersede atomicity) are covered by the
integration smoke against real Postgres. These tests cover the parts
that are testable without I/O: schema validation, tag normalization,
env resolution, helper functions, and lifecycle wrappers' delegation.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from memory_mcp.db.types import MemoryKind, MemoryStatus, OutboxOp
from memory_mcp.errors import EnvAmbiguousError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import (
    MemoryUpdatePatch,
    MemoryWriteRequest,
    _audit_snapshot,
    _hash_body,
    _is_macro_integrity_error,
    _normalize_tags,
    _outbox_op_for,
    _resolve_env_id,
    memory_write,
)

# ---------------------------------------------------------------------------
# Tag normalization
# ---------------------------------------------------------------------------


class TestNormalizeTags:
    def test_strips_whitespace(self) -> None:
        assert _normalize_tags(["  foo  ", "bar"]) == ["foo", "bar"]

    def test_drops_empty_strings(self) -> None:
        assert _normalize_tags(["foo", "", "  ", "bar"]) == ["foo", "bar"]

    def test_dedupes_preserving_first_seen_order(self) -> None:
        assert _normalize_tags(["foo", "bar", "foo", "baz"]) == ["foo", "bar", "baz"]

    def test_dedupe_after_strip(self) -> None:
        assert _normalize_tags(["foo", "  foo  ", " foo"]) == ["foo"]

    def test_case_preserving(self) -> None:
        # Tags are case-sensitive by design — different case = different tag.
        assert _normalize_tags(["Foo", "foo", "FOO"]) == ["Foo", "foo", "FOO"]

    def test_rejects_overly_long_tag(self) -> None:
        with pytest.raises(ValueError, match="tag too long"):
            _normalize_tags(["x" * 201])

    def test_empty_input(self) -> None:
        assert _normalize_tags([]) == []


# ---------------------------------------------------------------------------
# Env resolution (covers ENV_AMBIGUOUS contract)
# ---------------------------------------------------------------------------


def _ctx(*envs: UUID) -> AgentContext:
    return AgentContext(agent_id=uuid4(), attached_env_ids=list(envs))


class TestResolveEnvId:
    def test_explicit_wins_over_attached(self) -> None:
        a, b, c = uuid4(), uuid4(), uuid4()
        assert _resolve_env_id(explicit=a, ctx=_ctx(b, c)) == a

    def test_sole_attached_used_when_no_explicit(self) -> None:
        a = uuid4()
        assert _resolve_env_id(explicit=None, ctx=_ctx(a)) == a

    def test_no_envs_raises_ambiguous(self) -> None:
        with pytest.raises(EnvAmbiguousError) as exc:
            _resolve_env_id(explicit=None, ctx=_ctx())
        assert exc.value.code == "ENV_AMBIGUOUS"
        assert exc.value.details["attached"] == []

    def test_multiple_envs_raises_ambiguous(self) -> None:
        a, b = uuid4(), uuid4()
        with pytest.raises(EnvAmbiguousError) as exc:
            _resolve_env_id(explicit=None, ctx=_ctx(a, b))
        assert exc.value.code == "ENV_AMBIGUOUS"
        assert set(exc.value.details["attached"]) == {str(a), str(b)}

    def test_dedupes_attached_before_counting(self) -> None:
        # The same env id appearing twice in attached should still count as one.
        a = uuid4()
        ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[a, a])
        assert _resolve_env_id(explicit=None, ctx=ctx) == a


# ---------------------------------------------------------------------------
# Outbox op routing for lifecycle states
# ---------------------------------------------------------------------------


class TestOutboxOpRouting:
    @pytest.mark.parametrize(
        "status",
        [
            MemoryStatus.proposed,
            MemoryStatus.active,
            MemoryStatus.stale,
        ],
    )
    def test_visible_status_create(self, status: MemoryStatus) -> None:
        assert _outbox_op_for(status, is_create=True) == OutboxOp.upsert

    @pytest.mark.parametrize(
        "status",
        [
            MemoryStatus.proposed,
            MemoryStatus.active,
            MemoryStatus.stale,
        ],
    )
    def test_visible_status_update(self, status: MemoryStatus) -> None:
        assert _outbox_op_for(status, is_create=False) == OutboxOp.update

    @pytest.mark.parametrize(
        "status",
        [
            MemoryStatus.archived,
            MemoryStatus.superseded,
            MemoryStatus.retired,
        ],
    )
    def test_hidden_status_always_tombstone(self, status: MemoryStatus) -> None:
        # Critical: archived must tombstone too — caught by rubber-duck gate-3 review.
        assert _outbox_op_for(status, is_create=True) == OutboxOp.tombstone
        assert _outbox_op_for(status, is_create=False) == OutboxOp.tombstone


# ---------------------------------------------------------------------------
# Audit snapshot — GDPR-aware (no body content stored)
# ---------------------------------------------------------------------------


class _StubMemory:
    """Lightweight stand-in for a Memory ORM row in pure-Python tests."""

    def __init__(self, **kwargs: object) -> None:
        defaults: dict[str, object] = {
            "id": uuid4(),
            "env_id": uuid4(),
            "kind": "fact",
            "status": "active",
            "title": "demo",
            "body": "the body",
            "salience": 0.5,
            "confidence": 0.5,
            "pinned": False,
            "version": 1,
            "superseded_by": None,
            "expires_at": None,
            "verified_at": None,
            "metadata_": {"foo": "bar"},
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class TestAuditSnapshot:
    def test_body_stored_as_hash_only(self) -> None:
        m = _StubMemory(body="secret PII content")
        snap = _audit_snapshot(m)  # type: ignore[arg-type]
        assert "body" not in snap
        assert snap["body_hash"] == _hash_body("secret PII content")
        assert snap["body_length"] == len("secret PII content")

    def test_metadata_stored_as_keys_only(self) -> None:
        m = _StubMemory(metadata_={"a": "secret value", "b": 42})
        snap = _audit_snapshot(m)  # type: ignore[arg-type]
        assert snap["metadata_keys"] == ["a", "b"]
        # metadata values are NOT in the snapshot
        assert "metadata" not in snap
        assert "secret value" not in str(snap)

    def test_empty_metadata_omits_keys_field(self) -> None:
        m = _StubMemory(metadata_={})
        snap = _audit_snapshot(m)  # type: ignore[arg-type]
        assert "metadata_keys" not in snap

    def test_title_kept_as_is(self) -> None:
        # Title is conventionally short, low-PII — kept for human-readable audit.
        m = _StubMemory(title="My memory")
        snap = _audit_snapshot(m)  # type: ignore[arg-type]
        assert snap["title"] == "My memory"

    def test_tags_included_when_passed(self) -> None:
        m = _StubMemory()
        snap = _audit_snapshot(m, tag_names=["a", "b"])  # type: ignore[arg-type]
        assert snap["tags"] == ["a", "b"]

    def test_tags_omitted_when_not_passed(self) -> None:
        m = _StubMemory()
        snap = _audit_snapshot(m)  # type: ignore[arg-type]
        assert "tags" not in snap


# ---------------------------------------------------------------------------
# Pydantic schemas — patch semantics (absent != cleared)
# ---------------------------------------------------------------------------


class TestMemoryUpdatePatchSemantics:
    def test_fields_set_distinguishes_absent_from_null(self) -> None:
        # Absence: not in fields_set
        p1 = MemoryUpdatePatch(expected_version=1)
        assert "title" not in p1.model_fields_set
        assert "expires_at" not in p1.model_fields_set

        # Explicit None: IS in fields_set (means "clear me")
        p2 = MemoryUpdatePatch(expected_version=1, title=None, expires_at=None)
        assert "title" in p2.model_fields_set
        assert "expires_at" in p2.model_fields_set

        # Explicit value: IS in fields_set
        p3 = MemoryUpdatePatch(expected_version=1, title="new title")
        assert "title" in p3.model_fields_set

    def test_expected_version_required(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
            MemoryUpdatePatch()  # type: ignore[call-arg]

    def test_expected_version_must_be_positive(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            MemoryUpdatePatch(expected_version=0)

    def test_status_validates_against_enum(self) -> None:
        # Invalid status is rejected at the schema layer.
        with pytest.raises(Exception):  # noqa: B017
            MemoryUpdatePatch(expected_version=1, status="bogus")  # type: ignore[arg-type]

    def test_salience_bounded(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            MemoryUpdatePatch(expected_version=1, salience=1.5)


class TestMemoryWriteRequest:
    def test_body_required(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            MemoryWriteRequest(kind=MemoryKind.fact)  # type: ignore[call-arg]

    def test_body_min_length_one(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            MemoryWriteRequest(kind=MemoryKind.fact, body="")

    def test_env_id_optional(self) -> None:
        # env_id can be omitted; tool layer infers from attached envs.
        req = MemoryWriteRequest(kind=MemoryKind.fact, body="hi")
        assert req.env_id is None

    def test_default_tags_metadata(self) -> None:
        req = MemoryWriteRequest(kind=MemoryKind.fact, body="hi")
        assert req.tags == []
        assert req.metadata == {}

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            MemoryWriteRequest(  # type: ignore[call-arg]
                kind=MemoryKind.fact,
                body="hi",
                bogus_field="x",  # type: ignore[arg-type]
            )


class _OrigWithConstraint:
    def __init__(self, constraint_name: str) -> None:
        self.constraint_name = constraint_name


def test_macro_integrity_helper_rejects_non_matching_constraint_name() -> None:
    exc = IntegrityError("INSERT", {}, _OrigWithConstraint("memories_env_id_fkey"))
    assert _is_macro_integrity_error(exc) is False


@pytest.mark.asyncio
async def test_memory_write_macro_integrity_non_macro_constraint_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memory_mcp import memories

    env_id = uuid4()
    original = IntegrityError("INSERT", {}, _OrigWithConstraint("memories_env_id_fkey"))

    async def fake_write(*_args, **_kwargs):
        raise original

    monkeypatch.setattr(memories, "_memory_write_in_session", fake_write)

    @asynccontextmanager
    async def fake_scope():
        yield object()

    monkeypatch.setattr(memories, "session_scope", fake_scope)

    with pytest.raises(IntegrityError) as exc_info:
        await memory_write(
            MemoryWriteRequest(
                env_id=env_id,
                kind=MemoryKind.playbook,
                body="body",
                steps=["step"],
                macro="ci",
            ),
            ctx=_ctx(env_id),
        )

    assert exc_info.value is original


# ---------------------------------------------------------------------------
# Lifecycle wrappers delegate to memory_update with the expected patch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLifecycleWrappers:
    async def test_memory_archive_delegates(self) -> None:
        from memory_mcp import memories

        ctx = _ctx()
        memory_id = uuid4()
        with patch.object(memories, "memory_update") as mock_update:
            mock_update.return_value = "<sentinel>"
            result = await memories.memory_archive(memory_id, expected_version=3, ctx=ctx)
            assert result == "<sentinel>"
            args, kwargs = mock_update.call_args
            assert args[0] == memory_id
            patch_arg: MemoryUpdatePatch = args[1]
            assert patch_arg.expected_version == 3
            assert patch_arg.status == MemoryStatus.archived
            assert kwargs["ctx"] is ctx

    async def test_memory_retire_delegates_with_reason(self) -> None:
        from memory_mcp import memories

        ctx = _ctx()
        memory_id = uuid4()
        with patch.object(memories, "memory_update") as mock_update:
            mock_update.return_value = "<sentinel>"
            await memories.memory_retire(memory_id, expected_version=5, reason="obsolete", ctx=ctx)
            args, kwargs = mock_update.call_args
            patch_arg: MemoryUpdatePatch = args[1]
            assert patch_arg.expected_version == 5
            assert patch_arg.status == MemoryStatus.retired
            assert kwargs["_audit_extra"] == {"retire_reason": "obsolete"}

    async def test_memory_retire_rejects_blank_reason(self) -> None:
        from memory_mcp import memories

        ctx = _ctx()
        memory_id = uuid4()
        with pytest.raises(ValueError, match="reason is required"):
            await memories.memory_retire(memory_id, expected_version=1, reason="   ", ctx=ctx)
        with pytest.raises(ValueError, match="reason is required"):
            await memories.memory_retire(memory_id, expected_version=1, reason="", ctx=ctx)
