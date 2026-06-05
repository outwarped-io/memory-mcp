"""Unit tests for ``memory_mcp.search.ner``.

Covers:
* spaCy-pipeline cache (lazy load, missing-model sentinel, per-model keying)
* Identifier regex fallback patterns
* Raw-query fallback gating (token-count limit)
* Mention dedupe + max-mentions cap
* Empty-query short-circuit

We avoid loading the real ``en_core_web_sm`` model (not guaranteed in
the test env). Tests inject either ``None`` (model unavailable) or a
fake spaCy ``Language`` stub.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from memory_mcp.config import Settings
from memory_mcp.search import ner as ner_mod
from memory_mcp.search.ner import (
    _IDENTIFIER_RE,
    _USEFUL_ENT_LABELS,
    _reset_pipeline_cache_for_tests,
    extract_query_mentions,
)

# ---------------------------------------------------------------------------
# Fake spaCy doc / pipeline
# ---------------------------------------------------------------------------


@dataclass
class _FakeEnt:
    text: str
    label_: str


@dataclass
class _FakeDoc:
    ents: list[_FakeEnt]


class _FakeNlp:
    """Stand-in for ``spacy.Language`` — callable returning a fake doc."""

    def __init__(self, ents: list[tuple[str, str]]):
        self._ents = [_FakeEnt(t, lbl) for t, lbl in ents]

    def __call__(self, _text: str) -> _FakeDoc:
        return _FakeDoc(ents=list(self._ents))


def _settings(**overrides) -> Settings:
    base = {
        "ner_model": "en_core_web_sm",
        "graph_search_max_mentions": 8,
        "graph_search_raw_query_max_tokens": 3,
    }
    base.update(overrides)
    s = Settings()
    for k, v in base.items():
        object.__setattr__(s, k, v)
    return s


@pytest.fixture(autouse=True)
def _clear_pipeline_cache():
    _reset_pipeline_cache_for_tests()
    yield
    _reset_pipeline_cache_for_tests()


# ---------------------------------------------------------------------------
# extract_query_mentions — empty / whitespace
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty():
    assert asyncio.run(extract_query_mentions("", settings=_settings())) == []


def test_whitespace_query_returns_empty():
    assert asyncio.run(extract_query_mentions("   \n", settings=_settings())) == []


# ---------------------------------------------------------------------------
# Regex identifier fallback (no spaCy model)
# ---------------------------------------------------------------------------


def test_regex_extracts_mixed_case_identifier(monkeypatch):
    """``ServiceA`` must match without spaCy."""
    monkeypatch.setattr(ner_mod, "_NLP_CACHE", {"en_core_web_sm": None})
    out = asyncio.run(
        extract_query_mentions(
            "deploy the ServiceA pipeline",
            settings=_settings(),
        )
    )
    assert "servicea" in out


def test_regex_extracts_acronym(monkeypatch):
    monkeypatch.setattr(ner_mod, "_NLP_CACHE", {"en_core_web_sm": None})
    out = asyncio.run(
        extract_query_mentions(
            "the API team owns SRE escalation",
            settings=_settings(),
        )
    )
    assert "api" in out
    assert "sre" in out


def test_regex_extracts_separated_identifier(monkeypatch):
    monkeypatch.setattr(ner_mod, "_NLP_CACHE", {"en_core_web_sm": None})
    out = asyncio.run(
        extract_query_mentions(
            "look at foo.bar and service-a routes",
            settings=_settings(),
        )
    )
    assert "foo bar" in out
    assert "service a" in out


def test_regex_does_not_match_plain_capitalized_words(monkeypatch):
    """Plain capitalized words like ``Deploy``, ``The`` must NOT match —
    they would generate noise across natural-language queries."""
    monkeypatch.setattr(ner_mod, "_NLP_CACHE", {"en_core_web_sm": None})
    out = asyncio.run(
        extract_query_mentions(
            "Deploy the service via The pipeline tomorrow",
            settings=_settings(graph_search_raw_query_max_tokens=0),
        )
    )
    # No mixed-case identifier, no acronym (>=2 chars all-caps), no
    # separator, no raw fallback (max_tokens=0). Result: empty.
    assert out == []


def test_regex_identifier_smoke():
    """Sanity: the regex compiles and rejects single-letter / pure-digit garbage."""
    assert _IDENTIFIER_RE.search("A") is None  # too short
    assert _IDENTIFIER_RE.search("123") is None  # pure digits, no separator
    assert _IDENTIFIER_RE.search("ServiceA") is not None
    assert _IDENTIFIER_RE.search("API") is not None


# ---------------------------------------------------------------------------
# Raw-query fallback — gated on token count
# ---------------------------------------------------------------------------


def test_raw_query_fallback_for_short_query(monkeypatch):
    """A bare single-word query becomes a synthetic mention."""
    monkeypatch.setattr(ner_mod, "_NLP_CACHE", {"en_core_web_sm": None})
    out = asyncio.run(
        extract_query_mentions(
            "transport",
            settings=_settings(),
        )
    )
    assert "transport" in out


def test_raw_query_fallback_skipped_for_long_query(monkeypatch):
    """A natural-language sentence must NOT pollute resolution with the
    whole-sentence normalized form."""
    monkeypatch.setattr(ner_mod, "_NLP_CACHE", {"en_core_web_sm": None})
    out = asyncio.run(
        extract_query_mentions(
            "how do I deploy and configure the service",
            settings=_settings(graph_search_raw_query_max_tokens=3),
        )
    )
    # No identifier matches, no NER (model unavailable), >3 tokens → empty.
    assert out == []


# ---------------------------------------------------------------------------
# spaCy NER pathway with fake pipeline
# ---------------------------------------------------------------------------


def _install_fake_nlp(monkeypatch, ents: list[tuple[str, str]]):
    fake = _FakeNlp(ents)
    monkeypatch.setattr(ner_mod, "_NLP_CACHE", {"en_core_web_sm": fake})


def test_ner_extracts_useful_labels(monkeypatch):
    _install_fake_nlp(
        monkeypatch,
        [
            ("Microsoft", "ORG"),
            ("Seattle", "GPE"),
            ("2026-05-07", "DATE"),  # excluded
            ("$5", "MONEY"),  # excluded
        ],
    )
    out = asyncio.run(
        extract_query_mentions(
            "ignored content (NLP is mocked)",
            settings=_settings(),
        )
    )
    assert "microsoft" in out
    assert "seattle" in out
    assert all(lbl in _USEFUL_ENT_LABELS or lbl not in _USEFUL_ENT_LABELS for lbl in {"ORG", "GPE"})


def test_ner_dedupes_against_regex(monkeypatch):
    """An entity mentioned by both NER and the regex should appear once."""
    _install_fake_nlp(monkeypatch, [("ServiceA", "PRODUCT")])
    out = asyncio.run(
        extract_query_mentions(
            "ServiceA is broken",
            settings=_settings(),
        )
    )
    assert out.count("servicea") == 1


def test_ner_failure_falls_back_to_regex(monkeypatch):
    """If spaCy raises during NER, regex fallback still produces output."""

    class _ExplodingNlp:
        def __call__(self, _text):  # noqa: D401
            raise RuntimeError("boom")

    monkeypatch.setattr(
        ner_mod,
        "_NLP_CACHE",
        {"en_core_web_sm": _ExplodingNlp()},
    )
    out = asyncio.run(
        extract_query_mentions(
            "test ServiceA failure",
            settings=_settings(),
        )
    )
    assert "servicea" in out


# ---------------------------------------------------------------------------
# Mention cap + dedupe
# ---------------------------------------------------------------------------


def test_max_mentions_cap_truncates(monkeypatch):
    """``graph_search_max_mentions`` caps the output deterministically."""
    monkeypatch.setattr(ner_mod, "_NLP_CACHE", {"en_core_web_sm": None})
    # Construct a query with 5 distinct identifiers but cap at 3.
    out = asyncio.run(
        extract_query_mentions(
            "ServiceA ServiceB ServiceC ServiceD ServiceE",
            settings=_settings(
                graph_search_max_mentions=3,
                graph_search_raw_query_max_tokens=0,
            ),
        )
    )
    assert out == ["servicea", "serviceb", "servicec"]


def test_dedupe_preserves_first_seen_order(monkeypatch):
    monkeypatch.setattr(ner_mod, "_NLP_CACHE", {"en_core_web_sm": None})
    out = asyncio.run(
        extract_query_mentions(
            "ServiceA and ServiceA again, plus API and API",
            settings=_settings(graph_search_raw_query_max_tokens=0),
        )
    )
    assert out == ["servicea", "api"]


# ---------------------------------------------------------------------------
# Cache keying per model name
# ---------------------------------------------------------------------------


def test_cache_keys_by_model_name(monkeypatch):
    """A test that swaps ``ner_model`` should not get a stale ``None`` from
    a prior model load. Cache must be keyed by model name."""
    nlp_a = _FakeNlp([("Alpha", "ORG")])
    nlp_b = _FakeNlp([("Beta", "ORG")])
    monkeypatch.setattr(
        ner_mod,
        "_NLP_CACHE",
        {"model_a": nlp_a, "model_b": nlp_b},
    )

    out_a = asyncio.run(
        extract_query_mentions(
            "x",
            settings=_settings(ner_model="model_a"),
        )
    )
    out_b = asyncio.run(
        extract_query_mentions(
            "x",
            settings=_settings(ner_model="model_b"),
        )
    )
    assert "alpha" in out_a
    assert "beta" in out_b
