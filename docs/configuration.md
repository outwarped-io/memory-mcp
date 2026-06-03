# Configuration

Environment variables consumed by memory-mcp at startup.

## Configuration

See `.env.example`. Key vars:

| Variable | Default | Notes |
|---|---|---|
| `POSTGRES_URL` | (set in compose) | `postgresql+asyncpg://...` |
| `QDRANT_URL` | (set in compose) | |
| `NEO4J_URL` | (set in compose) | |
| `EMBEDDER` | `local` | `local` (sentence-transformers) or `azure_openai` |
| `EMBEDDING_MODEL_ID` | `all-MiniLM-L6-v2` | Per-env override stored in `environments.default_embedding_model_id` |
| `VECTOR_BACKEND` | `qdrant` | `qdrant` or `pgvector` |
| `GRAPH_BACKEND` | `neo4j` | `neo4j` or `postgres` (recursive-CTE fallback) |
| `MCP_HTTP_HOST` | `0.0.0.0` (compose) | Bind address inside the server process. Compose pins it so Docker port-publishing works; loopback enforcement happens on the host side via `MCP_HTTP_BIND`. |
| `MCP_HTTP_BIND` | `127.0.0.1` | Compose-only — the host interface that publishes the server port. **DO NOT change in v1 unless you understand the local-only contract.** |
| `MEMORY_MCP_HOST_PORT` | `8080` | Host-side port the server is published on. Override to any free port when running two memory-mcp stacks on the same host. Container always binds 8080 internally. |
| `MEMORY_MCP_IMAGE_TAG` | `dev` | Tag applied by `docker compose build` to the three local images. Override to a version (e.g. `v0.17.0`) when cutting a release image. |
| `LOG_LEVEL` | `INFO` | structlog level |
| `LOCAL_DEFAULT_AGENT_FILE` | `/var/lib/memory-mcp/default-agent.json` | Server-default agent persisted on first run (used when no `X-Agent-Id` is provided) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | (unset) | Reserved; v1.5 |
| `MEMORY_MCP_DECISION_CONFLICT_COSINE_THRESHOLD` | `0.85` | Cosine cutoff for dream-mode decision conflict proposals. |

---

