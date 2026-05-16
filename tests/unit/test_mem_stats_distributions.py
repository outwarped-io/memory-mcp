"""Pure distribution helper tests for mem_stats."""

from __future__ import annotations

from memory_mcp import stats


def test_chain_depth_bucketing_and_percentiles() -> None:
    out = stats._chain_depth_stats([1, 1, 2, 3, 4, 7])

    assert out.buckets == {"1": 2, "2": 1, "3": 1, "4+": 2}
    assert out.p50 == 2
    assert out.p90 == 7
    assert out.p99 == 7
    assert out.max == 7


def test_salience_buckets() -> None:
    out = stats._salience_stats([0.0, 0.19, 0.2, 0.49, 0.5, 0.79, 0.8, 1.0])

    assert out.buckets == {
        "0.0-0.2": 2,
        "0.2-0.5": 2,
        "0.5-0.8": 2,
        "0.8-1.0": 2,
    }


def test_access_buckets_and_percentiles() -> None:
    out = stats._access_stats([0, 1, 5, 6, 50, 51, 100])

    assert out.buckets == {"never": 1, "1-5": 2, "6-50": 2, "51+": 2}
    assert out.p50 == 6
    assert out.p90 == 100
    assert out.p99 == 100


def test_tags_per_memory_counts_untagged() -> None:
    out = stats._tags_per_memory_stats([1, 3, 5], total_memories=5)

    assert out.untagged == 2
    assert out.p50 == 1
    assert out.p90 == 5
    assert out.max == 5


def test_empty_percentiles_return_nulls() -> None:
    assert stats._percentiles([]) == {"p50": None, "p90": None, "p99": None, "max": None}
