"""Dream-mode subsystem (Phase 2.2).

Background workloads that compact, decay, and propose changes to the
canonical memory store. Three passes ship in v1:

* ``passes.decay``  — salience decay & lifecycle transitions
  (``active`` → ``stale`` → ``archived``).
* ``passes.dedupe`` — vector-cluster similar memories and emit
  ``merge_candidate`` proposals.
* ``passes.promote`` — cluster journal observations and emit
  ``promotion_candidate`` proposals to upgrade observations to first-class
  ``fact`` memories.

Proposals are reviewed via the ``dream_review`` MCP tool — the dream worker
never mutates canonical state directly.

The ``salience`` module is the lowest-level building block: a pure function
``compute_salience(row, …) -> float`` consumed by both the on-read access
boost path (in ``memories.py``) and the decay pass.
"""

from memory_mcp.dream.salience import (
    SalienceInputs,
    SalienceWeights,
    compute_salience,
    salience_weights_from_settings,
)
from memory_mcp.dream.summarizer import (
    DreamSummarizer,
    LLMSummarizer,
    MergeCluster,
    MergeClusterMember,
    MergeSummary,
    PromotionCluster,
    PromotionClusterObservation,
    PromotionSummary,
    TemplateSummarizer,
    build_summarizer,
)

__all__ = [
    "DreamSummarizer",
    "LLMSummarizer",
    "MergeCluster",
    "MergeClusterMember",
    "MergeSummary",
    "PromotionCluster",
    "PromotionClusterObservation",
    "PromotionSummary",
    "SalienceInputs",
    "SalienceWeights",
    "TemplateSummarizer",
    "build_summarizer",
    "compute_salience",
    "salience_weights_from_settings",
]
