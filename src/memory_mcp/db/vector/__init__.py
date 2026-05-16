"""Vector backends.

Two implementations behind a common ``VectorStore`` interface (Phase 1):

* ``qdrant.QdrantVectorStore`` — default, high-performance vector DB
* ``pgvector.PgvectorVectorStore`` — Postgres-only fallback for small deployments
"""
