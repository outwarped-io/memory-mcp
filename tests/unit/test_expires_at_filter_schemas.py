"""Schema-level tests for the ``include_expired`` opt-out flag.

v0.17 cross-cutting filter rule: expired memories (``expires_at`` in the
past) are hidden by default on **every** read path. Callers opt back in
via ``include_expired=True`` on four request schemas:

* ``MemorySearchRequest`` (``mem_search``)
* ``MemBrowseRequest`` (``mem_browse``)
* ``MemFacetsRequest`` (``mem_facets``)
* ``MemTopRequest`` (``mem_top``)

Convenience surfaces (``mem_resume``, ``mem_context_pack``,
``mem_auto_context``, digest) apply the filter unconditionally and
expose no opt-out.

This module verifies the schema surface only — that the field exists,
defaults to ``False``, and accepts ``True``. The runtime enforcement
(WHERE-clauses, ``_passes_post_filters`` helper, browse cursor
fingerprint) lives in the integration suite.
"""

from __future__ import annotations

from memory_mcp_schemas.browse import MemBrowseRequest, MemFacetsRequest
from memory_mcp_schemas.search import MemorySearchRequest
from memory_mcp_schemas.top import MemTopRequest


class TestSearchRequestIncludeExpired:
    def test_default_false(self) -> None:
        req = MemorySearchRequest(query="anything")
        assert req.include_expired is False

    def test_accepts_true(self) -> None:
        req = MemorySearchRequest(query="anything", include_expired=True)
        assert req.include_expired is True


class TestBrowseRequestIncludeExpired:
    def test_default_false(self) -> None:
        req = MemBrowseRequest()
        assert req.include_expired is False

    def test_accepts_true(self) -> None:
        req = MemBrowseRequest(include_expired=True)
        assert req.include_expired is True


class TestFacetsRequestIncludeExpired:
    def test_default_false(self) -> None:
        req = MemFacetsRequest()
        assert req.include_expired is False

    def test_accepts_true(self) -> None:
        req = MemFacetsRequest(include_expired=True)
        assert req.include_expired is True


class TestTopRequestIncludeExpired:
    def test_default_false(self) -> None:
        req = MemTopRequest()
        assert req.include_expired is False

    def test_accepts_true(self) -> None:
        req = MemTopRequest(include_expired=True)
        assert req.include_expired is True
