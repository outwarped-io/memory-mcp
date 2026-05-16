"""Unit tests for :mod:`memory_mcp.journal`."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from memory_mcp.db.types import MemoryKind
from memory_mcp.identity import AgentContext
from memory_mcp.journal import JournalRequest, memory_journal


def _ctx(*envs):
    return AgentContext(agent_id=uuid4(), attached_env_ids=list(envs))


class TestJournalRequest:
    def test_content_required(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            JournalRequest()  # type: ignore[call-arg]

    def test_content_min_length_one(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            JournalRequest(content="")

    def test_default_tags_metadata(self) -> None:
        req = JournalRequest(content="hi")
        assert req.tags == []
        assert req.metadata == {}
        assert req.env_id is None
        assert req.salience is None

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            JournalRequest(  # type: ignore[call-arg]
                content="hi",
                title="not allowed",  # type: ignore[arg-type]
            )


@pytest.mark.asyncio
class TestMemoryJournalDelegation:
    async def test_delegates_to_memory_write_with_observation_kind(self) -> None:
        from memory_mcp import journal as journal_mod

        env_id = uuid4()
        ctx = _ctx(env_id)
        with patch.object(journal_mod, "memory_write", new_callable=AsyncMock) as mock_write:
            mock_write.return_value = "<sentinel>"
            result = await memory_journal(
                JournalRequest(
                    content="something happened",
                    env_id=env_id,
                    tags=["seen"],
                    metadata={"k": "v"},
                    salience=0.7,
                ),
                ctx=ctx,
            )
            assert result == "<sentinel>"
            args, kwargs = mock_write.call_args
            write_req = args[0]
            assert write_req.kind == MemoryKind.observation
            assert write_req.body == "something happened"
            assert write_req.title is None  # journal forces no title
            assert write_req.env_id == env_id
            assert write_req.tags == ["seen"]
            assert write_req.metadata == {"k": "v"}
            assert write_req.salience == 0.7
            assert kwargs["ctx"] is ctx

    async def test_omits_settings_when_not_provided(self) -> None:
        from memory_mcp import journal as journal_mod

        ctx = _ctx(uuid4())
        with patch.object(journal_mod, "memory_write", new_callable=AsyncMock) as mock_write:
            mock_write.return_value = "<sentinel>"
            await memory_journal(JournalRequest(content="x"), ctx=ctx)
            _, kwargs = mock_write.call_args
            assert kwargs["settings"] is None
