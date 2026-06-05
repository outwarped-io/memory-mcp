"""Unit tests for ``memory_mcp.inbox`` pure helpers (Phase C-4).

Pure-function tests for the reference parser/formatter, slug generator,
cursor encode/decode, and validation predicates that the three inbox
tools consume. Async tool entry points (``mem_inbox_open``,
``mem_inbox_send``, ``mem_inbox``) — including the
``await _resolve_env_refs`` regression guard for the bug fixed at
commit ``f77bf6e`` — are covered end-to-end in
``tests/integration/test_inbox.py`` because they require a live
Postgres + Qdrant testcontainer stack (env resolution + entity table
lookup are not unit-mockable in any useful way).
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import random
from uuid import uuid4

import pytest
from memory_mcp_schemas.inbox import (
    DEFAULT_TTL_DAYS,
    INBOX_TAG,
    MAX_TTL_DAYS,
    REFERENCE_SCHEME,
)

from memory_mcp.errors import InvalidCursorError, InvalidInputError
from memory_mcp.inbox import (
    _ADJECTIVES,
    _NOUNS,
    _SLUG_MAX_LEN,
    _env_names_equal,
    _validate_slug,
    decode_cursor,
    encode_cursor,
    format_reference,
    generate_slug,
    parse_reference,
)

# ---------------------------------------------------------------------------
# Reference parsing
# ---------------------------------------------------------------------------


class TestParseReference:
    def test_url_form_round_trip(self) -> None:
        env_name, slug = parse_reference("mem-inbox://workspace/quiet-otter")
        assert env_name == "workspace"
        assert slug == "quiet-otter"

    def test_url_form_with_digits_in_slug(self) -> None:
        env_name, slug = parse_reference("mem-inbox://cdp/incident-123-handoff")
        assert env_name == "cdp"
        assert slug == "incident-123-handoff"

    def test_url_form_with_kebab_env_name(self) -> None:
        # Env names can have hyphens too (e.g., 'cdp-workspace').
        env_name, slug = parse_reference("mem-inbox://cdp-workspace/brave-falcon")
        assert env_name == "cdp-workspace"
        assert slug == "brave-falcon"

    def test_bare_slug_returns_none_env(self) -> None:
        env_name, slug = parse_reference("quiet-otter")
        assert env_name is None
        assert slug == "quiet-otter"

    def test_bare_single_word_slug(self) -> None:
        env_name, slug = parse_reference("scratch")
        assert env_name is None
        assert slug == "scratch"

    def test_url_form_missing_slug_rejected(self) -> None:
        # 'mem-inbox://workspace' has no '/' after the env, so the URL
        # form is malformed.
        with pytest.raises(InvalidInputError, match="URL form requires"):
            parse_reference("mem-inbox://workspace")

    def test_url_form_empty_env_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="empty env-name"):
            parse_reference("mem-inbox:///quiet-otter")

    def test_url_form_extra_segment_rejected(self) -> None:
        # Trailing '/' or a second slug segment indicates a typo, not a
        # nested-path feature.
        with pytest.raises(InvalidInputError, match="only one slug segment"):
            parse_reference("mem-inbox://workspace/foo/bar")

    def test_url_form_invalid_slug_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="INVALID_INBOX_SLUG"):
            parse_reference("mem-inbox://workspace/UPPERCASE")

    def test_url_form_slug_with_underscore_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="INVALID_INBOX_SLUG"):
            parse_reference("mem-inbox://workspace/quiet_otter")

    def test_bare_slug_invalid_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="INVALID_INBOX_SLUG"):
            parse_reference("Quiet-Otter")

    def test_bare_slug_with_leading_hyphen_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="INVALID_INBOX_SLUG"):
            parse_reference("-quiet-otter")

    def test_bare_slug_with_trailing_hyphen_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="INVALID_INBOX_SLUG"):
            parse_reference("quiet-otter-")

    def test_bare_slug_double_hyphen_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="INVALID_INBOX_SLUG"):
            parse_reference("quiet--otter")

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="empty 'to'"):
            parse_reference("")

    def test_non_string_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="empty 'to'"):
            parse_reference(None)  # type: ignore[arg-type]

    def test_slug_at_max_length_accepted(self) -> None:
        slug = "a" + "-b" * 31 + "c"  # exactly 64 chars
        assert len(slug) == _SLUG_MAX_LEN
        env_name, parsed = parse_reference(f"mem-inbox://workspace/{slug}")
        assert env_name == "workspace"
        assert parsed == slug

    def test_slug_over_max_length_rejected(self) -> None:
        slug = "a" * (_SLUG_MAX_LEN + 1)
        with pytest.raises(InvalidInputError, match="INVALID_INBOX_SLUG"):
            parse_reference(f"mem-inbox://workspace/{slug}")


# ---------------------------------------------------------------------------
# Reference formatting (round-trip with parser)
# ---------------------------------------------------------------------------


class TestFormatReference:
    def test_basic(self) -> None:
        assert format_reference("workspace", "quiet-otter") == "mem-inbox://workspace/quiet-otter"

    def test_round_trip(self) -> None:
        ref = format_reference("cdp", "incident-handoff")
        env_name, slug = parse_reference(ref)
        assert env_name == "cdp"
        assert slug == "incident-handoff"

    def test_scheme_constant_alignment(self) -> None:
        # The format function MUST use the shared REFERENCE_SCHEME so
        # the schema package and operational layer agree on the prefix.
        ref = format_reference("e", "s")
        assert ref.startswith(f"{REFERENCE_SCHEME}://")

    def test_round_trip_with_kebab_env(self) -> None:
        ref = format_reference("cdp-workspace", "brave-falcon")
        env_name, slug = parse_reference(ref)
        assert env_name == "cdp-workspace"
        assert slug == "brave-falcon"


# ---------------------------------------------------------------------------
# Slug validation
# ---------------------------------------------------------------------------


class TestValidateSlug:
    @pytest.mark.parametrize(
        "slug",
        [
            "a",
            "quiet-otter",
            "incident-123-handoff",
            "scratch1",
            "a-b-c-d-e-f",
        ],
    )
    def test_valid(self, slug: str) -> None:
        assert _validate_slug(slug) == slug

    @pytest.mark.parametrize(
        "slug",
        [
            "",
            "UPPER",
            "with_underscore",
            "-leading",
            "trailing-",
            "double--hyphen",
            "with space",
            "with.dot",
            "with/slash",
            "a" * (_SLUG_MAX_LEN + 1),
        ],
    )
    def test_invalid(self, slug: str) -> None:
        with pytest.raises(InvalidInputError, match="INVALID_INBOX_SLUG"):
            _validate_slug(slug)


# ---------------------------------------------------------------------------
# Slug generation + wordlist invariants
# ---------------------------------------------------------------------------


class TestGenerateSlug:
    def test_shape_adjective_dash_noun(self) -> None:
        slug = generate_slug()
        head, _, tail = slug.partition("-")
        assert head in _ADJECTIVES
        assert tail in _NOUNS

    def test_passes_validator(self) -> None:
        # Every wordlist combination must satisfy the slug regex,
        # otherwise generate_slug can produce values _validate_slug
        # rejects.
        for adj in _ADJECTIVES:
            for noun in _NOUNS:
                candidate = f"{adj}-{noun}"
                # Must not raise.
                _validate_slug(candidate)

    def test_wordlist_entries_kebab_safe(self) -> None:
        # No word in either list contains hyphens (otherwise the
        # generated form would have a double-hyphen and fail validation).
        for word in (*_ADJECTIVES, *_NOUNS):
            assert "-" not in word, word
            assert "_" not in word, word
            assert word == word.lower(), word

    def test_deterministic_with_injected_rng(self) -> None:
        rng = random.Random(42)
        first = generate_slug(rng)
        rng = random.Random(42)
        second = generate_slug(rng)
        assert first == second

    def test_different_seeds_produce_different_slugs(self) -> None:
        # Statistical: with two distinct seeds the most likely outcome
        # is a different slug. This guards against a frozen RNG.
        a = generate_slug(random.Random(1))
        b = generate_slug(random.Random(2))
        # If both happen to collide, re-roll with a third pair to
        # confirm at least one of the three pairs differs.
        c = generate_slug(random.Random(3))
        assert {a, b, c} != {a}, f"all three slugs identical: {a}"


# ---------------------------------------------------------------------------
# Cursor encode / decode round-trip
# ---------------------------------------------------------------------------


class TestCursorRoundTrip:
    def test_encode_returns_url_safe_ascii(self) -> None:
        now = dt.datetime(2026, 6, 3, 14, 30, 0, tzinfo=dt.UTC)
        cursor = encode_cursor(now, uuid4())
        # Padding stripped.
        assert "=" not in cursor
        # URL-safe alphabet only.
        assert all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in cursor)

    def test_round_trip_preserves_timestamp_and_id(self) -> None:
        when = dt.datetime(2026, 6, 3, 14, 30, 45, 123456, tzinfo=dt.UTC)
        mem_id = uuid4()
        decoded_when, decoded_id = decode_cursor(encode_cursor(when, mem_id))
        assert decoded_when == when
        assert decoded_id == mem_id

    def test_round_trip_naive_timestamp(self) -> None:
        # encode_cursor uses .isoformat() which preserves naive
        # timestamps verbatim; decode_cursor reads them back as naive.
        # The query path normalizes; the cursor primitive should not.
        when = dt.datetime(2026, 6, 3, 14, 30, 45)
        mem_id = uuid4()
        decoded_when, decoded_id = decode_cursor(encode_cursor(when, mem_id))
        assert decoded_when == when
        assert decoded_when.tzinfo is None
        assert decoded_id == mem_id

    def test_round_trip_microsecond_precision(self) -> None:
        # The default isoformat preserves microseconds; we rely on this
        # for accurate keyset pagination.
        when = dt.datetime(2026, 6, 3, 14, 30, 45, 999999, tzinfo=dt.UTC)
        mem_id = uuid4()
        decoded_when, _ = decode_cursor(encode_cursor(when, mem_id))
        assert decoded_when.microsecond == 999999

    def test_round_trip_many_random_pairs(self) -> None:
        rng = random.Random(0)
        for _ in range(50):
            when = dt.datetime(
                2020 + rng.randint(0, 9),
                rng.randint(1, 12),
                rng.randint(1, 28),
                rng.randint(0, 23),
                rng.randint(0, 59),
                rng.randint(0, 59),
                rng.randint(0, 999999),
                tzinfo=dt.UTC,
            )
            mem_id = uuid4()
            decoded_when, decoded_id = decode_cursor(encode_cursor(when, mem_id))
            assert decoded_when == when
            assert decoded_id == mem_id


class TestCursorErrors:
    def test_garbage_string_rejected(self) -> None:
        with pytest.raises(InvalidCursorError, match="INVALID_CURSOR"):
            decode_cursor("this is not a cursor!!!")

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(InvalidCursorError, match="INVALID_CURSOR"):
            decode_cursor("")

    def test_valid_base64_but_not_json_rejected(self) -> None:
        bad = base64.urlsafe_b64encode(b"not json").decode("ascii").rstrip("=")
        with pytest.raises(InvalidCursorError, match="INVALID_CURSOR"):
            decode_cursor(bad)

    def test_json_missing_keys_rejected(self) -> None:
        payload = json.dumps({"t": "2026-06-03T00:00:00+00:00"}).encode("utf-8")
        bad = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        with pytest.raises(InvalidCursorError, match="INVALID_CURSOR"):
            decode_cursor(bad)

    def test_json_bad_uuid_rejected(self) -> None:
        payload = json.dumps({"t": "2026-06-03T00:00:00+00:00", "i": "not-a-uuid"}).encode("utf-8")
        bad = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        with pytest.raises(InvalidCursorError, match="INVALID_CURSOR"):
            decode_cursor(bad)

    def test_json_bad_timestamp_rejected(self) -> None:
        payload = json.dumps({"t": "nonsense", "i": str(uuid4())}).encode("utf-8")
        bad = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        with pytest.raises(InvalidCursorError, match="INVALID_CURSOR"):
            decode_cursor(bad)


# ---------------------------------------------------------------------------
# Env-name equality
# ---------------------------------------------------------------------------


class TestEnvNamesEqual:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("workspace", "workspace", True),
            ("Workspace", "workspace", True),
            ("WORKSPACE", "workspace", True),
            ("cdp", "CDP", True),
            ("cdp-workspace", "CDP-Workspace", True),
            ("workspace", "private", False),
            ("workspace", "workspaces", False),
            ("", "", True),
        ],
    )
    def test_casefold(self, a: str, b: str, expected: bool) -> None:
        assert _env_names_equal(a, b) is expected


# ---------------------------------------------------------------------------
# Cross-package constant agreement
# ---------------------------------------------------------------------------


class TestSchemaConstantsAlignment:
    """Constants exported by the schemas package must align with the
    operational layer's behavior. If these drift, send/list paths break
    silently.
    """

    def test_reference_scheme(self) -> None:
        # parse_reference accepts only the canonical scheme.
        bad = REFERENCE_SCHEME.replace("-", "_")
        with pytest.raises(InvalidInputError, match="INVALID_INBOX_SLUG"):
            # Treated as a bare slug; underscore is invalid.
            parse_reference(f"{bad}://workspace/foo")

    def test_inbox_tag_is_kebab(self) -> None:
        # Tag policy is workspace-wide kebab-case (memory-mcp policy
        # §7). Misalignment would mean inbox messages are tagged
        # differently from the rest of the substrate.
        assert INBOX_TAG.lower() == INBOX_TAG
        assert " " not in INBOX_TAG
        assert "_" not in INBOX_TAG

    def test_default_ttl_within_max(self) -> None:
        # If DEFAULT > MAX, the send tool's "use default" branch would
        # exceed the cap on first call.
        assert DEFAULT_TTL_DAYS <= MAX_TTL_DAYS
        assert DEFAULT_TTL_DAYS > 0
        assert MAX_TTL_DAYS > 0
