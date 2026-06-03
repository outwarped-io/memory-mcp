"""Query-time entity extraction for the graph leg of ``mem_search``.

v1 strategy — conservative, deterministic, no surprises:

1. **spaCy NER** (``ents`` only — *not* ``noun_chunks``). Lazily loaded
   per process; cached behind an ``asyncio.Lock``. Missing model is
   tolerated: we mark the model unavailable and fall through to the
   regex fallback.
2. **Regex identifier extractor** as a fallback. Catches mixed-case
   identifiers (``ServiceA``), acronyms (``API``, ``SRE``), and
   hyphen/dot/slash-separated names (``service-a``, ``foo.bar``,
   ``org/project``). This guarantees graph search works even when the
   spaCy model is not installed.
3. **Raw-query fallback** — if and only if the query has at most
   ``settings.graph_search_raw_query_max_tokens`` whitespace tokens,
   we treat the *whole* query as a synthetic mention. This lets a
   bare ``"ServiceA"`` query alias-match an entity even when neither
   NER nor the regex picks it up.

Returned mentions are **deduplicated by their normalized form** — the
same string the entity / alias tables index on.

Why no ``noun_chunks``? RRF gives the graph leg one vote per memory
regardless of raw-score scale. Generic chunks like "deployment",
"service", "incident" routinely alias-match high-degree entities and
flood the fused candidate pool with low-signal hits. Adding chunks
without aggressive filtering does more harm than good in v1; we
revisit when downstream signals (entity-degree, IDF) are available.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from memory_mcp.config import Settings
from memory_mcp.entities import _normalize_name

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# spaCy pipeline cache (per model name)
# ---------------------------------------------------------------------------

# We key on the model name so a test that swaps `settings.ner_model` from
# under us doesn't get a cached "unavailable" sentinel that was learned
# under a different model.
_NLP_CACHE: dict[str, Any | None] = {}
_NLP_CACHE_LOCK = asyncio.Lock()


# Whitelist of spaCy entity labels that are likely to map onto things in
# the entities table. ``DATE`` / ``TIME`` / ``PERCENT`` / ``CARDINAL`` /
# ``ORDINAL`` / ``MONEY`` / ``QUANTITY`` are deliberately excluded —
# they almost never identify a domain entity.
_USEFUL_ENT_LABELS: frozenset[str] = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT",
    "WORK_OF_ART", "EVENT", "FAC", "NORP", "LAW", "LANGUAGE",
})


# Regex identifier fallback. We match tokens that "look like a name":
#   - CamelCase with internal capital (ServiceA, FooBarBaz) — at least
#     one capital that is NOT at position 0; this excludes plain
#     capitalized words like "Deploy", "The".
#   - camelCase with internal capital (serviceA).
#   - all-uppercase 2+ letters (API, SRE) — the 2+ guard avoids "I", "A".
#   - hyphen / dot / slash / underscore separated alphanumerics
#     (service-a, foo.bar, org/project, my_service).
_IDENTIFIER_RE = re.compile(
    r"""
    \b(
        # CamelCase: leading uppercase, then lowercase, then another upper
        [A-Z][a-z]+[A-Z][A-Za-z0-9]*
        |
        # camelCase: leading lowercase, then internal upper
        [a-z]+[A-Z][A-Za-z0-9]*
        |
        # all-caps acronym, length >= 2
        [A-Z]{2,}[A-Z0-9]*
        |
        # separated identifier — at least one separator, at least 2 segments
        [A-Za-z0-9]+(?:[-./_][A-Za-z0-9]+)+
    )\b
    """,
    re.VERBOSE,
)


async def _get_pipeline(model_name: str) -> Any | None:
    """Return the cached spaCy ``Language`` pipeline, or ``None`` if missing.

    The first caller for each ``model_name`` performs the import + load;
    subsequent callers reuse the cached object. A model that fails to
    load is cached as ``None`` so we never retry the slow path mid-request.
    """
    if model_name in _NLP_CACHE:
        return _NLP_CACHE[model_name]
    async with _NLP_CACHE_LOCK:
        if model_name in _NLP_CACHE:
            return _NLP_CACHE[model_name]
        try:
            import spacy  # type: ignore[import-untyped]

            nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
        except (OSError, ImportError) as exc:
            log.warning(
                "graph_search: spaCy model %r unavailable (%s); "
                "falling back to regex identifier extraction",
                model_name,
                exc,
            )
            _NLP_CACHE[model_name] = None
            return None
        _NLP_CACHE[model_name] = nlp
        return nlp


def _reset_pipeline_cache_for_tests() -> None:
    """Test hook — clears the spaCy pipeline cache."""
    _NLP_CACHE.clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_query_mentions(
    query: str,
    *,
    settings: Settings,
) -> list[str]:
    """Return a deduplicated list of normalized mention strings.

    Output is suitable for direct lookup against
    ``entities.normalized_name`` / ``entity_aliases.normalized_alias``.
    Empty / whitespace-only queries return ``[]``.
    """
    q = (query or "").strip()
    if not q:
        return []

    raw: list[str] = []

    # 1. spaCy NER — best signal when available.
    nlp = await _get_pipeline(settings.ner_model)
    if nlp is not None:
        try:
            # spaCy is sync; offload to a thread so we never block the
            # event loop on tokenization.
            doc = await asyncio.get_running_loop().run_in_executor(
                None, nlp, q
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning(
                "graph_search: spaCy NER raised %s; falling back to regex",
                exc,
            )
        else:
            for ent in doc.ents:
                if ent.label_ in _USEFUL_ENT_LABELS:
                    text = ent.text.strip()
                    if text:
                        raw.append(text)

    # 2. Regex identifier fallback — independent of spaCy availability.
    #    Catches things like "ServiceA", "API", "foo.bar".
    raw.extend(m.group(1) for m in _IDENTIFIER_RE.finditer(q))

    # 3. Raw-query fallback for short queries.
    tokens = q.split()
    if 0 < len(tokens) <= settings.graph_search_raw_query_max_tokens:
        raw.append(q)

    # Dedupe by normalized form, preserving insertion order. This is the
    # same key the entity tables index on.
    seen: set[str] = set()
    out: list[str] = []
    for s in raw:
        norm = _normalize_name(s)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)

    # Hard cap so a query containing dozens of identifier-like tokens
    # can't blow up downstream fan-out. Order is deterministic.
    return out[: settings.graph_search_max_mentions]


__all__ = [
    "extract_query_mentions",
]
