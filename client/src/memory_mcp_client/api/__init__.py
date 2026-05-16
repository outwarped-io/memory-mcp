"""Per-domain API namespaces for :class:`memory_mcp_client.MemoryClient`.

Each namespace class wraps a related set of MCP tools and is exposed as
an attribute on the client (``client.memories``, ``client.tasks``, …).
Methods accept either a Pydantic request model from
:mod:`memory_mcp_schemas` or matching kwargs (which build the model
internally), and return a typed response model when the server tool has
one.
"""

from __future__ import annotations

from memory_mcp_client.api.decisions import DecisionsAPI
from memory_mcp_client.api.dream import DreamAPI
from memory_mcp_client.api.entities import EntitiesAPI
from memory_mcp_client.api.env_ops import EnvOpsAPI
from memory_mcp_client.api.envs import EnvsAPI
from memory_mcp_client.api.memories import MemoriesAPI
from memory_mcp_client.api.playbooks import PlaybooksAPI
from memory_mcp_client.api.relations import RelationsAPI
from memory_mcp_client.api.tasks import TasksAPI

__all__ = [
    "DecisionsAPI",
    "DreamAPI",
    "EntitiesAPI",
    "EnvOpsAPI",
    "EnvsAPI",
    "MemoriesAPI",
    "PlaybooksAPI",
    "RelationsAPI",
    "TasksAPI",
]
