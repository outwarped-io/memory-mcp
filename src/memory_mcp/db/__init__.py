"""Database layer: canonical Postgres + outbox + projection backends."""

from memory_mcp.db.models import Base

__all__ = ["Base"]
