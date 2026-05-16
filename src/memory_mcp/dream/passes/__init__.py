"""Dream-mode passes — structural and proposal-emitting workloads.

Each pass is a self-contained async function the runner schedules per
``(env_id, mode)``. Passes never construct their own dependencies (LLM
client, summarizer, session) — the runner injects them so unit tests can
drop in mocks without touching environment-bound state.

Three passes ship in v1:

* :mod:`memory_mcp.dream.passes.decay` — structural-only lifecycle
  transitions (``active → stale → archived``). No summarizer.
* :mod:`memory_mcp.dream.passes.dedupe` — vector-cluster similar memories
  and emit ``merge_candidate`` proposals.
* :mod:`memory_mcp.dream.passes.promote` — cluster journal observations
  around an entity and emit ``promotion_candidate`` proposals.

Passes are idempotent by construction: re-running over an unchanged
dataset must converge to the same final state. The runner relies on this
so a crash mid-pass is recovered by re-running the next scheduled tick.
"""
