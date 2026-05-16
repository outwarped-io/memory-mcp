"""Runnable memory-mcp-client example.

Usage:
    PYTHONPATH=client/src:schemas/src:src .venv/bin/python examples/client.py

Defaults to the local memory-mcp container at http://127.0.0.1:8080/mcp.
Override with MEMORY_MCP_URL, MEMORY_MCP_AGENT_ID, MEMORY_MCP_ENV_NAME, or
MEMORY_MCP_DEFAULT_ENV_IDS (comma-separated UUIDs).
"""

from __future__ import annotations

import argparse
import asyncio
import os
from uuid import UUID

from memory_mcp_client import MemoryClient, NotFoundError
from memory_mcp_schemas.enums import MemoryKind
from memory_mcp_schemas.graph import MemRelatedRequest
from memory_mcp_schemas.search import MemorySearchRequest


def _uuid_list(raw: str | None) -> list[UUID] | None:
    if not raw:
        return None
    return [UUID(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a memory-mcp client SDK smoke example.")
    parser.add_argument(
        "--url",
        default=os.getenv("MEMORY_MCP_URL", "http://127.0.0.1:8080/mcp"),
        help="memory-mcp Streamable HTTP endpoint",
    )
    parser.add_argument(
        "--agent-id",
        default=os.getenv("MEMORY_MCP_AGENT_ID"),
        help="optional client-level agent UUID default",
    )
    parser.add_argument(
        "--default-env-ids",
        default=os.getenv("MEMORY_MCP_DEFAULT_ENV_IDS"),
        help="optional comma-separated env UUID defaults",
    )
    parser.add_argument(
        "--env-name",
        default=os.getenv("MEMORY_MCP_ENV_NAME", "client-sdk-example"),
        help="env to get or create before writing",
    )
    return parser.parse_args()


async def get_or_create_env(client: MemoryClient, name: str):
    try:
        env = await client.envs.get(name=name)
        print(f"Using existing env {env.name}: {env.id}")
        return env
    except NotFoundError:
        env = await client.envs.create(name=name)
        print(f"Created env {env.name}: {env.id}")
        return env


async def run() -> None:
    args = parse_args()
    default_env_ids = _uuid_list(args.default_env_ids)

    async with MemoryClient(
        args.url,
        agent_id=args.agent_id,
        default_env_ids=default_env_ids,
    ) as client:
        envs = await client.envs.list_()
        print(f"Server has {len(envs)} env(s)")

        env = await get_or_create_env(client, args.env_name)

        memory = await client.memories.write(
            kind=MemoryKind.fact,
            title="memory-mcp-client example",
            body="The Python SDK wraps memory-mcp tools over Streamable HTTP.",
            env_id=env.id,
            tags=["example", "client-sdk"],
            attached_env_ids=[env.id],
        )
        print(f"Wrote memory {memory.id}")

        search = await client.memories.search(
            MemorySearchRequest(
                query="Python SDK Streamable HTTP",
                env_ids=[env.id],
                consistency="fresh",
                limit=5,
            ),
            attached_env_ids=[env.id],
        )
        print("Search hits:")
        for hit in search.hits:
            print(f"- {hit.score:.3f} {hit.memory.id} {hit.memory.title!r}")

        related = await client.memories.related(
            MemRelatedRequest(
                memory_id=memory.id,
                relation="semantic",
                env_id=env.id,
                limit=5,
            ),
            attached_env_ids=[env.id],
        )
        print(f"Related ({related.note}): {[hit.memory_id for hit in related.hits]}")

        try:
            await client.memories.get(
                "00000000-0000-0000-0000-000000000000",
                attached_env_ids=[env.id],
            )
        except NotFoundError:
            print("NotFoundError: missing memory handled cleanly")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
