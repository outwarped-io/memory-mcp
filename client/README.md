# memory-mcp-client

## What it is

Async Python client SDK for `memory-mcp` v0.11+ over MCP Streamable HTTP. It wraps the 59 MCP tools with typed Pydantic request/response models from [`memory-mcp-schemas`](../schemas/) and exposes them through `MemoryClient` namespaces.

Version `0.2.0` adds:

- **Typed mutation responses** — `supersede()`, `journal()`, `hard_delete()`.
- **Retry policy** — reads retry by default; writes opt in via `retry_writes=True` or per-call `idempotency_key`. See `RetryPolicy` for tuning.
- **Live integration tests** under `tests/integration/` (gated on `MEMORY_MCP_INTEGRATION_URL`); run with `make integration`.
- **Typed `RetryExhaustedError` and `ValidationFailedError`** for the new v0.11 server error codes.

## Installation

The package is path-installed alongside the server today; it is not published to PyPI yet.

```bash
# From another uv-managed project next to this repo:
uv add ../client

# Or from the memory-mcp repo root for editable local work:
pip install -e ./client
```

`memory-mcp-client` depends on the sibling `memory-mcp-schemas` package; `uv` resolves it from `../schemas` via `client/pyproject.toml`.

## Quickstart

```python
import asyncio

from memory_mcp_client import MemoryClient
from memory_mcp_schemas.enums import MemoryKind
from memory_mcp_schemas.search import MemorySearchRequest

async def main() -> None:
    async with MemoryClient("http://127.0.0.1:8080/mcp") as client:
        envs = await client.envs.list_()
        env = envs[0] if envs else await client.envs.create(name="scratch")

        memory = await client.memories.write(
            kind=MemoryKind.fact,
            title="SDK quickstart",
            body="memory-mcp-client writes memories over Streamable HTTP.",
            env_id=env.id,
            tags=["sdk", "quickstart"],
            attached_env_ids=[env.id],
        )

        results = await client.memories.search(
            MemorySearchRequest(
                query="Streamable HTTP client",
                env_ids=[env.id],
                consistency="fresh",
                limit=5,
            ),
            attached_env_ids=[env.id],
        )
        print(memory.id, [hit.memory.title for hit in results.hits])

asyncio.run(main())
```

## Identity defaults vs. per-call overrides

Pass `agent_id` and `default_env_ids` to `MemoryClient` to inject identity into every tool call. Per-call `agent_id=` or `attached_env_ids=` wins when supplied, so shared clients can still target a different agent/env for a single request.

```python
async with MemoryClient(
    "http://127.0.0.1:8080/mcp",
    agent_id="00000000-0000-0000-0000-000000000001",
    default_env_ids=["00000000-0000-0000-0000-0000000000e0"],
) as client:
    await client.memories.search(query="uses defaults")
    await client.memories.search(query="override", attached_env_ids=[other_env_id])
```

## Namespaces

- `client.memories` — 19 methods.
- `client.tasks` — 8 methods.
- `client.envs` — 5 methods.
- `client.entities` — 5 methods.
- `client.relations` — 2 methods.
- `client.playbooks` — 1 method.
- `client.decisions` — 1 method.
- `client.dream` — 4 methods.

Use `dir(client.memories)` for the callable surface, or read the module docstrings under [`src/memory_mcp_client/api/`](src/memory_mcp_client/api/).

## Error handling

All protocol-level errors derive from `MemoryMCPError`. Server messages in `[CODE] msg :: {json}` format are parsed into typed exceptions such as `NotFoundError` and `VersionConflictError`; see [`errors.py`](src/memory_mcp_client/errors.py) for the full hierarchy.

```python
from memory_mcp_client import MemoryMCPError, NotFoundError, VersionConflictError

try:
    memory = await client.memories.get(memory_id)
    await client.memories.archive(memory.id, expected_version=memory.version)
except NotFoundError:
    print("memory does not exist in the attached envs")
except VersionConflictError as exc:
    print("refetch and retry", exc.details)
except MemoryMCPError as exc:
    print(exc.code, exc.message)
```

## Health probes

`health()` and `ready()` call `/healthz` and `/readyz` directly with `httpx`; they do not require an opened MCP session.

```python
client = MemoryClient("http://127.0.0.1:8080/mcp")
print(await client.health())
print(await client.ready())
```

## Testing against the client

For user tests that need a fake MCP server, copy the pattern from [`client/tests/conftest.py`](tests/conftest.py). `FakeClientSession` records calls and scripts structured responses or server-style errors, and `MemoryClient(session_factory=...)` runs namespace methods against it without opening a network connection.

## Compatibility

- Python >= 3.12.
- Async-only; no sync facade.
- Requires a running `memory-mcp` server exposing Streamable HTTP at `/mcp`.
- Intended for `memory-mcp` v0.7.1+ with the matching `memory-mcp-schemas` package.
