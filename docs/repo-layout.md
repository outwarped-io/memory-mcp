# Repository layout

## Repo layout

```
memory-mcp/
├── docker/                    # one Dockerfile per process role
│   ├── Dockerfile.server
│   ├── Dockerfile.projection-worker
│   └── Dockerfile.dream-worker
├── docker-compose.yml         # postgres + qdrant + neo4j + server + projection-worker (+ dream-worker stub)
├── docker-compose.dev.yml     # dev overrides (mounts, hot-reload)
├── migrations/                # alembic
├── src/
│   ├── memory_mcp/            # MCP server
│   │   ├── server.py          # FastAPI lifespan + /healthz, /readyz, /metrics + /mcp mount
│   │   ├── mcp_app.py         # MCP tool registrations (45 tools in v0.7.1)
│   │   ├── tools/             # one module per tool family
│   │   ├── db/                # postgres + outbox + vector + graph backends
│   │   ├── embeddings/        # local + azure_openai
│   │   ├── search/            # lex / sem / hybrid / ranking (graph in Phase 2)
│   │   ├── identity.py        # AgentContext + no-op rbac.require()
│   │   ├── envs.py
│   │   └── observability.py   # structlog + Prometheus + RequestIdMiddleware
│   ├── projection_worker/     # drains outbox to Qdrant (+ Neo4j in Phase 2)
│   └── dream_worker/          # Phase 2 stub
├── tests/
│   ├── unit/                  # 270+ tests
│   ├── integration/           # MCP transport smoke (env-gated)
│   └── invariants/            # lifecycle matrix, schema invariants
└── examples/
    ├── client.py
    ├── connect-copilot-cli.json
    └── connect-claude-desktop.json
```

---

