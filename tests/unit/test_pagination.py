"""Unit tests for :mod:`memory_mcp.pagination`."""

from __future__ import annotations

import base64
import datetime as dt
import json
from uuid import UUID, uuid4

import pytest

from memory_mcp.errors import InvalidCursorError
from memory_mcp.pagination import (
    SCHEMA_VERSION,
    compute_filter_fingerprint,
    decode_cursor,
    encode_cursor,
)


# ---------------------------------------------------------------------------
# compute_filter_fingerprint
# ---------------------------------------------------------------------------


class TestComputeFilterFingerprint:
    def test_returns_16_hex_chars(self) -> None:
        fp = compute_filter_fingerprint({"env_ids": [str(uuid4())]})
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_stable_across_calls(self) -> None:
        f = {"env_ids": [uuid4()], "kinds": ["fact", "observation"], "limit": 50}
        assert compute_filter_fingerprint(f) == compute_filter_fingerprint(f)

    def test_list_order_insensitive(self) -> None:
        # Filter semantics are set-based; order should not matter
        a = compute_filter_fingerprint({"kinds": ["fact", "observation"]})
        b = compute_filter_fingerprint({"kinds": ["observation", "fact"]})
        assert a == b

    def test_dict_key_order_insensitive(self) -> None:
        a = compute_filter_fingerprint({"kinds": ["fact"], "limit": 10})
        b = compute_filter_fingerprint({"limit": 10, "kinds": ["fact"]})
        assert a == b

    def test_uuid_str_and_obj_equivalent(self) -> None:
        u = uuid4()
        assert compute_filter_fingerprint({"id": u}) == compute_filter_fingerprint({"id": str(u)})

    def test_datetime_tz_normalised_to_utc(self) -> None:
        naive = dt.datetime(2026, 5, 10, 12, 0, 0)
        aware_utc = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)
        # Naive is assumed UTC by the normaliser; equivalent fingerprints
        assert compute_filter_fingerprint({"t": naive}) == compute_filter_fingerprint({"t": aware_utc})

    def test_different_filters_different_fingerprints(self) -> None:
        a = compute_filter_fingerprint({"kinds": ["fact"]})
        b = compute_filter_fingerprint({"kinds": ["observation"]})
        assert a != b

    def test_empty_dict_fingerprint(self) -> None:
        fp = compute_filter_fingerprint({})
        assert len(fp) == 16  # still deterministic

    def test_none_values_distinct_from_missing(self) -> None:
        a = compute_filter_fingerprint({})
        b = compute_filter_fingerprint({"created_after": None})
        # Documented: None values participate in the fingerprint
        assert a != b

    def test_nested_dicts(self) -> None:
        a = compute_filter_fingerprint({"x": {"a": 1, "b": 2}})
        b = compute_filter_fingerprint({"x": {"b": 2, "a": 1}})
        assert a == b

    def test_bool_distinct_from_int(self) -> None:
        # Pydantic occasionally coerces; we keep bool separate from 0/1
        # for filters like ``descending``.
        a = compute_filter_fingerprint({"flag": True})
        b = compute_filter_fingerprint({"flag": 1})
        # JSON serialization treats True and 1 differently — verify.
        assert a != b


# ---------------------------------------------------------------------------
# encode_cursor / decode_cursor — round-trip
# ---------------------------------------------------------------------------


class TestCursorRoundTrip:
    def test_round_trip_datetime(self) -> None:
        fp = "0123456789abcdef"
        oid = uuid4()
        at = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)
        cur = encode_cursor(
            filter_fingerprint=fp,
            order_field="updated_at",
            order_value=at,
            tiebreak_id=oid,
            direction="desc",
        )
        decoded = decode_cursor(cur, expected_fingerprint=fp,
                                expected_order_field="updated_at",
                                expected_direction="desc")
        assert decoded.filter_fingerprint == fp
        assert decoded.order_field == "updated_at"
        assert decoded.tiebreak_id == oid
        assert decoded.direction == "desc"
        # order_value stored as ISO string
        assert decoded.order_value == at.isoformat()

    def test_round_trip_string_order(self) -> None:
        """Entity browse uses canonical_name (string) as order key."""
        fp = compute_filter_fingerprint({})
        oid = uuid4()
        cur = encode_cursor(
            filter_fingerprint=fp,
            order_field="canonical_name",
            order_value="apple",
            tiebreak_id=oid,
            direction="asc",
        )
        decoded = decode_cursor(cur, expected_fingerprint=fp)
        assert decoded.order_value == "apple"
        assert decoded.direction == "asc"

    def test_cursor_is_url_safe(self) -> None:
        cur = encode_cursor(
            filter_fingerprint="x" * 16,
            order_field="updated_at",
            order_value=dt.datetime(2026, 5, 10, tzinfo=dt.UTC),
            tiebreak_id=uuid4(),
            direction="desc",
        )
        # urlsafe_b64encode produces only [A-Za-z0-9-_]; no padding from rstrip
        assert all(c.isalnum() or c in "-_" for c in cur), cur

    def test_cursor_is_compact(self) -> None:
        # A typical cursor should be < 200 chars (well under our 4096 limit).
        cur = encode_cursor(
            filter_fingerprint="0123456789abcdef",
            order_field="updated_at",
            order_value=dt.datetime(2026, 5, 10, tzinfo=dt.UTC),
            tiebreak_id=uuid4(),
            direction="desc",
        )
        assert len(cur) < 200


# ---------------------------------------------------------------------------
# decode_cursor — error paths
# ---------------------------------------------------------------------------


class TestDecodeErrors:
    def _make_cursor(self, **overrides: object) -> str:
        defaults = {
            "filter_fingerprint": "0123456789abcdef",
            "order_field": "updated_at",
            "order_value": dt.datetime(2026, 5, 10, tzinfo=dt.UTC),
            "tiebreak_id": uuid4(),
            "direction": "desc",
        }
        defaults.update(overrides)  # type: ignore[arg-type]
        return encode_cursor(**defaults)  # type: ignore[arg-type]

    def test_fingerprint_mismatch_raises(self) -> None:
        cur = self._make_cursor(filter_fingerprint="aaaa" * 4)
        with pytest.raises(InvalidCursorError, match="fingerprint mismatch"):
            decode_cursor(cur, expected_fingerprint="bbbb" * 4)

    def test_order_field_mismatch_raises(self) -> None:
        cur = self._make_cursor(order_field="updated_at")
        with pytest.raises(InvalidCursorError, match="order_field mismatch"):
            decode_cursor(
                cur,
                expected_fingerprint="0123456789abcdef",
                expected_order_field="created_at",
            )

    def test_direction_mismatch_raises(self) -> None:
        cur = self._make_cursor(direction="desc")
        with pytest.raises(InvalidCursorError, match="direction mismatch"):
            decode_cursor(
                cur,
                expected_fingerprint="0123456789abcdef",
                expected_direction="asc",
            )

    def test_garbage_string_raises(self) -> None:
        with pytest.raises(InvalidCursorError, match="malformed"):
            decode_cursor("this-is-not-a-cursor!!!", expected_fingerprint="x")

    def test_base64_but_not_json_raises(self) -> None:
        bad = base64.urlsafe_b64encode(b"not json").decode("ascii").rstrip("=")
        with pytest.raises(InvalidCursorError, match="malformed"):
            decode_cursor(bad, expected_fingerprint="x")

    def test_json_but_wrong_shape_raises(self) -> None:
        bad = base64.urlsafe_b64encode(b'["a","b"]').decode("ascii").rstrip("=")
        with pytest.raises(InvalidCursorError, match="must be an object"):
            decode_cursor(bad, expected_fingerprint="x")

    def test_schema_version_mismatch_raises(self) -> None:
        # Hand-craft a cursor with a bogus sv.
        payload = {
            "sv": SCHEMA_VERSION + 99,
            "fp": "0123456789abcdef",
            "ob": "updated_at",
            "ov": "2026-05-10T00:00:00+00:00",
            "id": str(uuid4()),
            "d": "desc",
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        cur = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        with pytest.raises(InvalidCursorError, match="schema_version"):
            decode_cursor(cur, expected_fingerprint="0123456789abcdef")

    def test_missing_required_keys_raises(self) -> None:
        payload = {"sv": SCHEMA_VERSION, "fp": "x" * 16, "ob": "updated_at"}
        raw = json.dumps(payload).encode()
        cur = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        with pytest.raises(InvalidCursorError, match="missing keys"):
            decode_cursor(cur, expected_fingerprint="x" * 16)

    def test_bad_uuid_raises(self) -> None:
        payload = {
            "sv": SCHEMA_VERSION,
            "fp": "0123456789abcdef",
            "ob": "updated_at",
            "ov": "2026-05-10T00:00:00+00:00",
            "id": "not-a-uuid",
            "d": "desc",
        }
        raw = json.dumps(payload).encode()
        cur = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        with pytest.raises(InvalidCursorError, match="tiebreak"):
            decode_cursor(cur, expected_fingerprint="0123456789abcdef")


# ---------------------------------------------------------------------------
# Tiebreak stability under mutable order key
# ---------------------------------------------------------------------------


class TestTiebreakStability:
    def test_ties_resolved_by_id(self) -> None:
        """Two rows with identical ``updated_at`` must be deterministically
        ordered by ``id`` so a cursor at one row uniquely identifies its
        position regardless of which other row shares the timestamp.
        """
        fp = "0123456789abcdef"
        at = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)
        # Two different cursors with same order_value differ by tiebreak_id
        a = encode_cursor(
            filter_fingerprint=fp, order_field="updated_at",
            order_value=at, tiebreak_id=UUID(int=1), direction="desc",
        )
        b = encode_cursor(
            filter_fingerprint=fp, order_field="updated_at",
            order_value=at, tiebreak_id=UUID(int=2), direction="desc",
        )
        assert a != b
        da = decode_cursor(a, expected_fingerprint=fp)
        db = decode_cursor(b, expected_fingerprint=fp)
        assert da.tiebreak_id != db.tiebreak_id
        assert da.order_value == db.order_value  # same timestamp


# ---------------------------------------------------------------------------
# Naive datetime gets normalised to UTC
# ---------------------------------------------------------------------------


class TestDatetimeNormalisation:
    def test_naive_datetime_treated_as_utc(self) -> None:
        fp = "0123456789abcdef"
        naive = dt.datetime(2026, 5, 10, 12, 0, 0)
        cur = encode_cursor(
            filter_fingerprint=fp, order_field="updated_at",
            order_value=naive, tiebreak_id=uuid4(), direction="desc",
        )
        decoded = decode_cursor(cur, expected_fingerprint=fp)
        # Order value should carry +00:00 tz on the wire
        assert decoded.order_value.endswith("+00:00")
