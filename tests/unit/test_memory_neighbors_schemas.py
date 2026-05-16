"""Pure-Python schema tests for the Sprint B memory-neighbors tool."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from memory_mcp.graph import MemNeighborsRequest, MemNeighborsResponse


# ---------------------------------------------------------------------------
# MemNeighborsRequest schema
# ---------------------------------------------------------------------------


class TestMemNeighborsRequest:
    def test_defaults(self) -> None:
        req = MemNeighborsRequest(memory_id=uuid4())
        assert req.hops == 1
        assert req.direction == "both"
        assert req.kind == "both"
        assert req.limit == 20
        assert req.consistency == "default"
        assert req.cursor is None

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            MemNeighborsRequest(memory_id=uuid4(), bogus=1)  # type: ignore[call-arg]

    def test_hops_bounds(self) -> None:
        MemNeighborsRequest(memory_id=uuid4(), hops=1)
        MemNeighborsRequest(memory_id=uuid4(), hops=3)
        with pytest.raises(ValidationError):
            MemNeighborsRequest(memory_id=uuid4(), hops=0)
        with pytest.raises(ValidationError):
            MemNeighborsRequest(memory_id=uuid4(), hops=4)

    def test_limit_bounds(self) -> None:
        MemNeighborsRequest(memory_id=uuid4(), limit=1)
        MemNeighborsRequest(memory_id=uuid4(), limit=100)
        with pytest.raises(ValidationError):
            MemNeighborsRequest(memory_id=uuid4(), limit=0)
        with pytest.raises(ValidationError):
            MemNeighborsRequest(memory_id=uuid4(), limit=101)

    def test_edge_types_validation(self) -> None:
        assert MemNeighborsRequest(memory_id=uuid4(), edge_types=None).edge_types is None
        assert MemNeighborsRequest(memory_id=uuid4(), edge_types=[" mentions "]).edge_types == ["mentions"]
        with pytest.raises(ValidationError):
            MemNeighborsRequest(memory_id=uuid4(), edge_types=[""])
        with pytest.raises(ValidationError):
            MemNeighborsRequest(memory_id=uuid4(), edge_types=["x" * 201])
        with pytest.raises(ValidationError):
            MemNeighborsRequest(memory_id=uuid4(), edge_types=[f"type_{i}" for i in range(21)])

    def test_aliases(self) -> None:
        memory_id = uuid4()
        req = MemNeighborsRequest(id=memory_id, types=["x"])  # type: ignore[call-arg]
        assert req.memory_id == memory_id
        assert req.edge_types == ["x"]

    def test_kind_enum(self) -> None:
        MemNeighborsRequest(memory_id=uuid4(), kind="entity")
        MemNeighborsRequest(memory_id=uuid4(), kind="memory")
        MemNeighborsRequest(memory_id=uuid4(), kind="both")
        with pytest.raises(ValidationError):
            MemNeighborsRequest(memory_id=uuid4(), kind="environment")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MemNeighborsResponse schema
# ---------------------------------------------------------------------------


class TestMemNeighborsResponse:
    def test_response_shape(self) -> None:
        r = MemNeighborsResponse(hits=[], next_cursor=None)
        assert r.hits == []
        assert r.next_cursor is None
        with pytest.raises(ValidationError):
            MemNeighborsResponse(hits="nope")  # type: ignore[arg-type]
