"""Shared Pydantic schemas + enums for memory-mcp server and client SDK.

This package is the single source of truth for every MCP tool's request
and response model. Both the server (``memory_mcp``) and the Python
client SDK (``memory_mcp_client``) depend on it so the two cannot drift.

Public modules:

* :mod:`memory_mcp_schemas.enums`        — string enums + lifecycle helpers
* :mod:`memory_mcp_schemas.memories`     — memory CRUD + lifecycle
* :mod:`memory_mcp_schemas.tasks`        — task tree
* :mod:`memory_mcp_schemas.envs`         — environments
* :mod:`memory_mcp_schemas.entities`     — entity upsert/resolve/merge/browse
* :mod:`memory_mcp_schemas.relations`    — relation link/browse
* :mod:`memory_mcp_schemas.graph`        — neighbor traversal
* :mod:`memory_mcp_schemas.search`       — lex/sem/hybrid search
* :mod:`memory_mcp_schemas.browse`       — mem_browse + mem_facets
* :mod:`memory_mcp_schemas.journal`      — journal entry
* :mod:`memory_mcp_schemas.digest`       — session digest + resume
* :mod:`memory_mcp_schemas.provenance`   — lineage + sources
* :mod:`memory_mcp_schemas.decisions`    — ADR-lite
* :mod:`memory_mcp_schemas.playbooks`    — playbook invoke
* :mod:`memory_mcp_schemas.context_pack` — F7 startup pack
* :mod:`memory_mcp_schemas.dream`        — dream worker proposals
* :mod:`memory_mcp_schemas.stats`        — v0.10 stats snapshot
* :mod:`memory_mcp_schemas.compose`      — v0.15 mem_compose (N→1)
* :mod:`memory_mcp_schemas.decompose`    — v0.15 mem_decompose (1→N)
"""

from __future__ import annotations

__version__ = "0.15.0a2"
