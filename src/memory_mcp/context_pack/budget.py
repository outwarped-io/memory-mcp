"""Token-budget helpers for context-pack assembly.

No tokenizer is currently shared by memory-mcp. These helpers use a simple
word-count approximation: one token is roughly 0.75 words, so tokens are
estimated as ``ceil(words / 0.75)``. This intentionally over-allocates for
plain English and should be replaced if a model-specific tokenizer is added.
"""

from __future__ import annotations

import math
import re

from memory_mcp.context_pack.models import ContextPackSectionName


MAX_TOKEN_BUDGET = 100_000
MIN_TOKEN_BUDGET = 200

SECTION_WEIGHTS: dict[ContextPackSectionName, float] = {
    "digest": 0.18,
    "trigger_matched": 0.25,
    "recent_journal": 0.20,
    "tasks": 0.10,
    "decisions": 0.10,
    "playbooks": 0.07,
    "archival": 0.10,
}

_WORD_RE = re.compile(r"\S+")


def clamp_token_budget(token_budget: int) -> int:
    return min(token_budget, MAX_TOKEN_BUDGET)


def estimate_tokens(text: str | None) -> int:
    """Estimate token count using a word-based approximation."""
    if not text:
        return 0
    word_count = len(_WORD_RE.findall(text))
    if word_count == 0:
        return 0
    return max(1, math.ceil(word_count / 0.75))


def truncate_to_token_budget(text: str, cap_tokens: int) -> tuple[str, bool, int]:
    """Return ``(possibly_truncated_text, was_truncated, estimated_tokens)``."""
    if cap_tokens <= 0:
        return "", bool(text), 0
    current = estimate_tokens(text)
    if current <= cap_tokens:
        return text, False, current

    words = _WORD_RE.findall(text)
    keep_words = max(1, math.floor(cap_tokens * 0.75))
    candidate = " ".join(words[:keep_words])
    while candidate and estimate_tokens(candidate) > cap_tokens:
        keep_words -= 1
        candidate = " ".join(words[:keep_words])
    return candidate, True, estimate_tokens(candidate)


def calculate_section_caps(
    token_budget: int,
    *,
    include_journal: bool = True,
    available_sections: set[ContextPackSectionName] | None = None,
) -> dict[ContextPackSectionName, int]:
    """Calculate per-section caps, redistributing skipped-section budget.

    If ``available_sections`` is provided, only those sections receive budget.
    This reclaims missing digest/trigger/journal/archival allocations and
    proportionally distributes them across remaining sections.
    """
    budget = clamp_token_budget(token_budget)
    weights = dict(SECTION_WEIGHTS)
    if not include_journal:
        weights.pop("recent_journal", None)
    if available_sections is not None:
        weights = {k: v for k, v in weights.items() if k in available_sections}
    if not weights:
        return {}

    total_weight = sum(weights.values())
    raw = {name: budget * weight / total_weight for name, weight in weights.items()}
    caps = {name: math.floor(value) for name, value in raw.items()}
    remainder = budget - sum(caps.values())
    for name, _ in sorted(
        raw.items(),
        key=lambda item: (item[1] - math.floor(item[1]), SECTION_WEIGHTS[item[0]]),
        reverse=True,
    )[:remainder]:
        caps[name] += 1
    return caps
