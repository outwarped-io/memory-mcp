"""Live integration coverage for SDK batch helpers."""

from __future__ import annotations

from uuid import uuid4

import pytest

from memory_mcp_client import MemoryClient
from memory_mcp_schemas.memories import MemoryWriteRequest

pytestmark = pytest.mark.asyncio


async def test_memories_write_many_live(integration_url: str) -> None:
    async with MemoryClient(integration_url) as client:
        try:
            await client.ready()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"server at {integration_url} not ready: {exc}")

        env = await client.envs.create(name=f"sdk-it-batch-{uuid4().hex[:8]}")
        try:
            items = [
                MemoryWriteRequest(
                    env_id=env.id,
                    kind="fact",
                    title=f"batch-live-{index}-{uuid4().hex[:6]}",
                    body=f"payload-{index}",
                    tags=["sdk-batch-it"],
                )
                for index in range(10)
            ]

            out = await client.memories.write_many(items, max_concurrency=4)

            assert out.failure_count == 0
            assert out.success_count == 10
            assert out.is_partial is False
            ids = [response.id for response in out.successes]
            assert len(set(ids)) == 10
            assert all(response.env_id == env.id for response in out.successes)
        finally:
            try:
                await client.envs.delete(env_id=env.id, confirm_destroy=True)
            except Exception:  # noqa: BLE001
                pass
