"""Projection worker — drains the Postgres ``outbox`` to Qdrant + Neo4j.

Phase 0: an idle loop with structured logging so the container can run.
Phase 1 will implement the actual outbox drain + per-sink delivery handling.
"""

__version__ = "0.0.1"
