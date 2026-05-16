# memory-mcp-schemas

Pydantic v2 request/response models and enums shared by the
[`memory-mcp`](../) server and the [`memory-mcp-client`](../client/)
Python SDK. Pure DTOs — no SQLAlchemy, no I/O, no transport.

This package exists so the server and client cannot drift apart: every
tool's `*Request` / `*Response` lives here, and both packages depend on
it as a path / editable install.

## Layout

```
memory_mcp_schemas/
├── enums.py            # MemoryKind, MemoryStatus, TaskStatus, …
├── memories.py         # MemoryWriteRequest, MemoryResponse, …
├── tasks.py            # TaskCreateRequest, TaskTreeResponse, …
├── envs.py
├── entities.py
├── relations.py
├── graph.py
├── search.py
├── browse.py
├── journal.py
├── digest.py
├── provenance.py
├── decisions.py
├── playbooks.py
├── context_pack.py
└── dream.py
```

Each module mirrors the analogous server submodule. The server modules
re-export by name so internal call sites remain stable; the SDK imports
from `memory_mcp_schemas.<domain>` directly.
