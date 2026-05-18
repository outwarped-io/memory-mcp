"""Configuration / dependency-injection scaffolding.

Single ``Settings`` object (pydantic-settings) drives every adapter:

* Postgres async engine + session factory (``db.postgres``)
* Qdrant async client (``db.vector.qdrant``)
* Neo4j async driver — lazy: Phase 1 only constructs it if ``GRAPH_BACKEND=neo4j``
* Embedder factory (``embeddings``)

Settings load from environment variables (matching ``.env.example``) and from
an optional ``.env`` file in the working directory. Names are case-insensitive.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_local_agent_path() -> Path:
    """Resolve a user-local default for the default-agent file.

    Honours ``XDG_DATA_HOME`` if set, otherwise falls back to
    ``~/.local/share`` on POSIX. On Windows ``Path.home()`` resolves to the
    user profile, so the same expression works without per-OS branching.
    """
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "memory-mcp" / "default-agent.json"


class Settings(BaseSettings):
    """memory-mcp runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Postgres ----------------------------------------------------------
    postgres_url: str = Field(
        default="postgresql+asyncpg://memory:memory@postgres:5432/memory",
        description="SQLAlchemy async URL for Postgres (must use asyncpg driver).",
    )
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 5
    postgres_statement_timeout_ms: int = 15_000

    # ---- Qdrant ------------------------------------------------------------
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None

    # ---- Neo4j -------------------------------------------------------------
    neo4j_url: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "memorymemory"

    # ---- Embedder ----------------------------------------------------------
    embedder: Literal["local", "azure_openai"] = "local"
    embedding_model_id: str = "all-MiniLM-L6-v2"
    embedding_dim: int | None = None  # If None, derived from model on first use.

    # ---- Backends ----------------------------------------------------------
    vector_backend: Literal["qdrant", "pgvector"] = "qdrant"
    graph_backend: Literal["neo4j", "postgres"] = "neo4j"

    # ---- HTTP server -------------------------------------------------------
    mcp_transport: Literal["http", "stdio"] = Field(
        default="http",
        description="MCP transport mode",
    )
    mcp_http_port: int = 8080
    # v1 = local-only: bind to loopback by default. Operators running this on
    # a trusted private network can override to ``0.0.0.0`` *deliberately*.
    mcp_http_host: str = "127.0.0.1"
    log_level: str = "INFO"

    # ---- Identity (v1 = local-only, no auth) -------------------------------
    # Path where the server-default agent UUID is persisted on first run.
    # Subsequent restarts read the file so the "default agent" stays stable
    # across restarts. Deleting the file orphans memories created under the
    # previous default agent — see ADR / journal.
    #
    # Default targets a user-local XDG path so first-run UX works for plain
    # ``python -m memory_mcp.server`` invocations. Docker Compose overrides
    # this via the ``LOCAL_DEFAULT_AGENT_FILE`` env var to point at a mounted
    # volume under ``/var/lib/memory-mcp/`` in the container.
    local_default_agent_file: str = Field(
        default_factory=lambda: str(_default_local_agent_path()),
    )
    local_default_agent_name: str = "default-local-agent"

    # ---- Local data ---------------------------------------------------------
    data_root: Path = Field(
        default_factory=lambda: Path.home() / ".local" / "share" / "memory-mcp",
        description="Root for local retained artifacts such as env snapshots.",
    )

    # ---- Auth (RESERVED — v1 = local-only; v1.5 will wire these) -----------
    bootstrap_admin_token_file: str = "/run/secrets/bootstrap_admin_token"
    auth_token_pepper: str | None = Field(
        default=None,
        description=(
            "Optional pepper added to the argon2 hashing context. "
            "Rotate by issuing new tokens; old hashes verify against the previous pepper. "
            "Reserved for v1.5 — unused in v1 (local-only)."
        ),
    )

    # ---- Observability -----------------------------------------------------
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "memory-mcp"

    # ---- Worker behaviour --------------------------------------------------
    projection_lease_seconds: int = 60
    projection_max_attempts: int = 8
    projection_batch_size: int = 64
    projection_idle_sleep_seconds: float = 1.0

    # ---- Search -----------------------------------------------------------
    # Max time ``memory_search(consistency="fresh")`` waits for the
    # ``qdrant`` projection to catch up before degrading to ``canonical``.
    search_fresh_max_wait_seconds: float = 2.0
    # Per-leg recall (lex / sem) before fusion. Final ``limit`` is the
    # number of results returned to the caller. Recall is capped at
    # ``max(2 * limit, search_min_per_leg)``.
    search_min_per_leg: int = 50

    # ---- Browse / facets (mem_browse, mem_facets) -------------------------
    # Postgres ``statement_timeout`` (seconds) applied via ``SET LOCAL``
    # around each facet aggregation. On timeout the response degrades
    # to ``approximate=True`` with whatever facets completed.
    facet_query_timeout_seconds: float = 2.0

    # ---- Graph search (mode=graph + graph leg of mode=hybrid) -------------
    # spaCy NER pipeline; if not installed at runtime the graph leg
    # falls back to a regex-based identifier extractor.
    ner_model: str = "en_core_web_sm"
    # Direct entity-memory traversal only in v1.
    graph_search_hops: int = 1
    # Hard caps on graph fan-out to bound Neo4j load and prevent the
    # graph leg from flooding the fused candidate pool with low-signal
    # hits resolved from a high-degree entity.
    graph_search_max_mentions: int = 8
    graph_search_max_resolved_entities_per_env: int = 8
    graph_search_max_resolved_entities_total: int = 16
    graph_search_max_concurrent_neighbors: int = 4
    # The raw query is treated as a synthetic mention only when it has
    # at most this many whitespace-separated tokens (catches "ServiceA"
    # without flooding multi-word natural-language queries).
    graph_search_raw_query_max_tokens: int = 3

    # ---- LLM (Phase 2.2 — used by DreamSummarizer when summarizer=llm) -----
    # Pluggable backend: ``ollama`` (default; sidecar via compose --profile llm),
    # ``openai_compatible`` (any URL + key — OpenAI, Azure OpenAI, vLLM,
    # OpenRouter, llama.cpp --api), or ``null`` (raises LLMUnavailableError —
    # the safe default for unit tests and template-only deployments).
    llm_backend: Literal["ollama", "openai_compatible", "null"] = "null"
    # Backend-specific endpoint. Empty string means "use the backend's default".
    # ollama default → http://ollama:11434  ;  openai_compatible has no default.
    llm_base_url: str = ""
    # Model id understood by the backend. ollama: ``llama3.2:3b`` (recommended
    # default for 8GB compose stacks). openai_compatible: ``gpt-4o-mini`` etc.
    llm_model_id: str = "llama3.2:3b"
    # Optional bearer token. Required for ``openai_compatible``; ollama ignores it.
    llm_api_key: str | None = None
    # Per-call HTTP timeout. Generation can be slow; cap defends the worker.
    llm_timeout_seconds: float = 60.0
    # Max tokens generated per call. Caller may override per-prompt.
    llm_max_tokens: int = 512
    # Default sampling temperature. Caller may override per-prompt.
    llm_temperature: float = 0.2

    # ---- Dream worker (Phase 2.2 — values applied by p2.2-runner) ---------
    # Selects the summarizer used by dream passes. ``llm`` (default) routes
    # through ``LLMSummarizer`` and requires ``LLM_BACKEND`` to be set;
    # ``template`` routes through ``TemplateSummarizer`` (pure-Python, no
    # external deps). Both impls are first-class — see Phase 2.2 plan.
    dream_summarizer: Literal["llm", "template"] = "llm"

    # ---- Dream salience weights (Phase 2.2) -------------------------------
    # All weights are configurable; defaults are tuned so that on the
    # default profile a memory with low confidence and rising negative
    # feedback decays toward 0 even if accessed often (negatives dominate).
    # Salience is clamped to [0, 1]. ``ge=0`` constraints prevent
    # operator-misconfiguration from inverting semantics (e.g. negative
    # ``w_negative`` would *reward* negative feedback).
    dream_salience_w_access: float = Field(default=0.30, ge=0.0)
    dream_salience_w_recency: float = Field(default=0.25, ge=0.0)
    dream_salience_w_confidence: float = Field(default=0.30, ge=0.0)
    # 0.40 (raised from 0.30 in Phase 1 v0.14) — tuned so the dominance
    # invariant (negatives outweigh positives at saturation) survives the
    # addition of the references term below. See salience.py module docstring.
    dream_salience_w_negative: float = Field(default=0.40, ge=0.0)
    dream_salience_pinned_bonus: float = Field(default=0.30, ge=0.0)
    dream_salience_verified_bonus: float = Field(default=0.10, ge=0.0)
    # Access count at which the access term saturates: at
    # ``access_count == access_window`` the access term hits the weight
    # ceiling exactly; counts above ``access_window`` are clamped at the
    # ceiling (no continued growth).
    dream_salience_access_window: int = Field(default=100, ge=1)
    # Time horizon (seconds) for the recency term.
    # exp(-Δt / τ) ≈ 0.37 after τ; ≈ 0.05 after 3τ.
    dream_salience_recency_tau_seconds: int = Field(default=7 * 24 * 3600, gt=0)
    # Recency boost for ``verified_at`` is capped at this many seconds —
    # past which the bonus is fully decayed. Default: 30 days.
    dream_salience_verified_tau_seconds: int = Field(default=30 * 24 * 3600, gt=0)

    # ---- Phase 1 (v0.14): graph-citation references term ------------------
    # The references term is the maximum salience contribution that
    # graph-citation signal (rel_link / lineage / task / playbook) can make.
    # Default 0.15 leaves head-room for the dominance invariant while still
    # noticeably preferring well-cited memories in default-sort.
    dream_salience_w_references: float = Field(default=0.15, ge=0.0)
    # Per-kind sub-weights — relative magnitudes inside the references-term
    # envelope. Set higher for kinds that carry stronger signal-per-edge:
    # playbook embeds are the strongest, followed by lineage parents (
    # load-bearing structural derivations), then task references, then
    # ad-hoc rel_link mentions.
    dream_salience_w_references_rl: float = Field(default=1.0, ge=0.0)
    dream_salience_w_references_ln: float = Field(default=1.5, ge=0.0)
    dream_salience_w_references_tk: float = Field(default=1.2, ge=0.0)
    dream_salience_w_references_pb: float = Field(default=2.0, ge=0.0)
    # Per-kind saturation windows — N_k at which the kind's per-kind term
    # hits 1.0. Smaller windows saturate faster (more sensitive to a few
    # citations). rel_link is cheap so window is large (50); lineage and
    # playbook are rare so windows are small (5, 10).
    dream_salience_window_rl: int = Field(default=50, ge=1)
    dream_salience_window_ln: int = Field(default=5, ge=1)
    dream_salience_window_tk: int = Field(default=20, ge=1)
    dream_salience_window_pb: int = Field(default=10, ge=1)

    # ---- Dream decay pass (Phase 2.2) -------------------------------------
    # Days since ``last_accessed_at`` after which an ``active`` memory is
    # eligible to be considered for staling. Memories accessed inside the
    # window are skipped wholesale (no salience recompute, no UPDATE) so
    # the pass is cheap on a healthy environment.
    dream_decay_inactive_days: int = Field(default=30, ge=1)
    # Salience threshold below which an ``active`` memory transitions to
    # ``stale``. Tuned against the salience formula's mid-range output
    # (~0.40 for a typical fresh memory) so that a memory must show
    # multiple decay signals (low recency AND low confidence OR negative
    # feedback) before it stales.
    dream_decay_stale_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    # Salience threshold below which a ``stale`` memory transitions to
    # ``archived``. Tighter than the stale threshold — a memory must be
    # *thoroughly* decayed before we hide it from default search.
    dream_decay_archive_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    # Per-pass row cap (per env, per leg). Bounds wall-clock time and
    # outbox pressure on environments with very large memory tables.
    # Tunable so operators can speed up backfill on a quiet weekend.
    dream_decay_batch_cap: int = Field(default=500, ge=1)
    # Phase 1 (v0.14) — graph-citation popularity gate. When > 0, an
    # ``active`` memory whose ``reference_count`` (sum of per-kind
    # reference counters maintained by Migration 0017's triggers) is
    # at or above this floor is held back from staling regardless of
    # salience. Protects structurally load-bearing memories from
    # archival just because nobody read them recently. Set to ``0``
    # to disable the gate entirely.
    dream_decay_reference_floor: int = Field(default=3, ge=0)
    # Phase 1 (v0.14) — default window for ``mem_top by=reference_velocity``
    # when the caller does not supply ``velocity_window_days``. 30 days
    # matches the typical task-cadence sweep window.
    mem_reference_velocity_window_days: int = Field(default=30, ge=1)

    # ---- Dream dedupe pass (Phase 2.2) ------------------------------------
    # Only consider ``active`` memories updated in the last N days as
    # cluster seeds. A memory that hasn't changed in months is unlikely
    # to grow new neighbors that the previous run wouldn't have already
    # surfaced, so capping the seed window keeps re-runs cheap.
    dream_dedupe_window_days: int = Field(default=7, ge=1)
    # Cosine similarity threshold above which two memories are considered
    # near-duplicates. Tuned conservatively — false positives (two
    # genuinely distinct memories accidentally clustered) are far more
    # costly than false negatives (a real duplicate slipping through
    # until the next pass), because every cluster surfaces as a
    # human-reviewed proposal.
    dream_dedupe_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    # Qdrant ``limit`` per seed query. 10 is generous — most clusters in
    # practice are pairs or triples; any source that finds 10 neighbors
    # above 0.92 is almost certainly a structural problem (mass-import
    # duplicate, agent in a loop) that a reviewer needs to see.
    dream_dedupe_top_k: int = Field(default=10, ge=2)
    # Per-run cap on the number of *new* proposals emitted. Bounds LLM
    # call volume when the summarizer runs in ``llm`` mode and bounds
    # reviewer cognitive load on a busy env.
    dream_dedupe_batch_cap: int = Field(default=200, ge=1)

    # ---- Dream promote pass (Phase 2.2) -----------------------------------
    # Only consider observation memories created within the last N days
    # as candidates for promotion. Older observations have presumably
    # been seen by past runs; if they were promotable, a proposal would
    # already exist (and the cross-run dedupe key would short-circuit).
    dream_promote_window_days: int = Field(default=14, ge=1)
    # Minimum number of observations (referencing the same entity) that
    # must cluster together before a ``promotion_candidate`` proposal is
    # emitted. Tuned high enough that a single chatty session can't
    # promote half its journal: a fact must be observed by at least
    # this many independent journal entries.
    dream_promote_min_cluster_size: int = Field(default=3, ge=2)
    # Per-run cap on the number of *new* proposals emitted. Bounds LLM
    # call volume and reviewer load. A large value is safe for the
    # template summarizer; lower it when running ``llm`` mode against
    # an expensive backend.
    dream_promote_batch_cap: int = Field(default=100, ge=1)
    # Maximum number of observations passed to the summarizer per
    # cluster. If a cluster contains more, the most-recent N are kept;
    # the proposal payload still records the full set so reviewers can
    # see how broad the evidence is, but the summarizer prompt stays
    # bounded.
    dream_promote_observations_per_cluster: int = Field(default=20, ge=2)
    decision_conflict_cosine_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        alias="MEMORY_MCP_DECISION_CONFLICT_COSINE_THRESHOLD",
    )

    # ---- Dream worker runner (Phase 2.2) ----------------------------------
    # Master switch — when False the dream worker idles (heartbeat only).
    # Default OFF in tests/dev so suites don't race the worker; the
    # production compose file flips it on.
    dream_enabled: bool = Field(default=False)
    # Cadence in seconds for each pass. Defaults are conservative;
    # operators can speed them up on larger envs. Each tick of a job
    # iterates all envs and acquires a per-(env, mode) advisory lock
    # so a slow env doesn't block others.
    dream_decay_cadence_seconds: int = Field(default=3600, ge=60)
    dream_dedupe_cadence_seconds: int = Field(default=1800, ge=60)
    dream_promote_cadence_seconds: int = Field(default=3600, ge=60)
    dream_decision_conflicts_cadence_seconds: int = Field(default=3600, ge=60)
    # Phase 1 (v0.14) — recount pass cadence. Heavier than decay
    # because it walks all rel_link / lineage / playbook macros per env.
    # Hourly default mirrors decay; tune up to 86400 (daily) for envs
    # with low edge churn, or down to 600 for envs with high churn and
    # active reviewer feedback loops.
    dream_recount_cadence_seconds: int = Field(default=3600, ge=60)
    # Per-pass wall-clock budget. If a pass exceeds this, the runner
    # logs a warning and the dream_run row is marked ``failed`` with
    # ``last_error="timeout"``. The pass itself isn't cancelled
    # mid-flight (cancellation is handled at SIGTERM via scheduler
    # shutdown); the timeout is observability only.
    dream_pass_timeout_seconds: int = Field(default=600, ge=10)
    # APScheduler max instances per job — ``1`` ensures a slow tick
    # never overlaps with the next, which would race over the same
    # advisory lock and waste cycles.
    dream_scheduler_max_instances: int = Field(default=1, ge=1)
    # Cadence for the proposals-open gauge refresh job. Cheap SQL
    # (one COUNT GROUP BY across env_id, kind, summarizer_kind) so
    # 60s is fine. Set to 0 to disable the refresher entirely.
    dream_metrics_refresh_seconds: int = Field(default=60, ge=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor.

    Tests can clear the cache via ``get_settings.cache_clear()``.
    """
    return Settings()
