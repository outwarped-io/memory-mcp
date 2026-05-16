"""MCP tool families.

Each module registers a group of tools with the MCP server during Phase 1:

* ``memory`` — write/get/update/search/supersede/archive/retire/journal/promote
* ``entity`` — upsert/resolve/neighbors/merge
* ``relation`` — link
* ``env`` — list/create/attach/detach/grant/copy
* ``dream`` — run/status/review
* ``admin`` — rebuild_qdrant/rebuild_neo4j/projection_status/restore/delete_hard/export
"""
