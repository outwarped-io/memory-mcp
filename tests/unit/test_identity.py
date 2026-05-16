"""Unit tests for identity module — pure-Python parts only.

The DB-touching pieces (``IdentityResolver._ensure_agent_row``,
``IdentityResolver.resolve``) live in the integration suite that the
``p1-tests`` todo will set up — they need a real Postgres connection on the
``memory-mcp_default`` network.

What's covered here:

* Default-agent file create/read round-trip
* Permission bits (0600) on Unix
* Malformed file FAILS FAST (no silent identity rotation)
* Atomic create-or-read race recovery (multi-process safety)
* UUID parser rejects bad input with a clear error
* ``AgentContext`` mutability rules
"""

from __future__ import annotations

import json
import os
import platform
import stat
import uuid
from pathlib import Path

import pytest

from memory_mcp.config import Settings
from memory_mcp.identity import (
    AgentContext,
    IdentityResolver,
    _parse_uuid,
)


def _make_settings(tmp_path: Path) -> Settings:
    """Build a Settings whose default-agent file lives under tmp_path."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        local_default_agent_file=str(tmp_path / "default-agent.json"),
    )


def _payload(agent_id: uuid.UUID, name: str = "x") -> dict[str, object]:
    return {"agent_id": str(agent_id), "agent_name": name}


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------

def test_agent_context_defaults() -> None:
    aid = uuid.uuid4()
    ctx = AgentContext(agent_id=aid)
    assert ctx.agent_id == aid
    assert ctx.agent_name is None
    assert ctx.session_id is None
    assert ctx.attached_env_ids == []
    assert ctx.is_default_agent is False


def test_agent_context_attached_envs_independent_per_instance() -> None:
    """Default factory should not leak state between AgentContext instances."""
    a = AgentContext(agent_id=uuid.uuid4())
    b = AgentContext(agent_id=uuid.uuid4())
    a.attached_env_ids.append(uuid.uuid4())
    assert b.attached_env_ids == []


# ---------------------------------------------------------------------------
# Default-agent file management
# ---------------------------------------------------------------------------

def test_default_agent_file_create_when_missing(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    file_path = Path(settings.local_default_agent_file)
    assert not file_path.exists()

    resolver = IdentityResolver(settings)
    payload_before = IdentityResolver._read_default_file(file_path)
    assert payload_before is None  # truly empty

    new_id = uuid.uuid4()
    created = IdentityResolver._atomic_create_default_file(file_path, _payload(new_id))
    assert created is True

    assert file_path.exists()
    payload_after = IdentityResolver._read_default_file(file_path)
    assert payload_after is not None
    assert _parse_uuid(str(payload_after["agent_id"])) == new_id
    # Resolver is just instantiated, no DB hit yet.
    assert resolver._default_agent_id is None  # type: ignore[attr-defined]


@pytest.mark.skipif(platform.system() == "Windows", reason="POSIX permission bits only")
def test_default_agent_file_is_chmod_600(tmp_path: Path) -> None:
    file_path = tmp_path / "default-agent.json"
    IdentityResolver._atomic_create_default_file(file_path, _payload(uuid.uuid4()))
    mode = stat.S_IMODE(os.stat(file_path).st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_default_agent_file_malformed_fails_fast(tmp_path: Path) -> None:
    """Corrupt files must NOT silently rotate identity (gate #2 NB#2)."""
    file_path = tmp_path / "default-agent.json"
    file_path.write_text("not valid json {", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable"):
        IdentityResolver._read_default_file(file_path)


def test_default_agent_file_missing_required_key_fails_fast(tmp_path: Path) -> None:
    file_path = tmp_path / "default-agent.json"
    file_path.write_text(json.dumps({"unrelated": "value"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing required keys"):
        IdentityResolver._read_default_file(file_path)


def test_default_agent_file_round_trip_preserves_uuid(tmp_path: Path) -> None:
    file_path = tmp_path / "default-agent.json"
    expected_id = uuid.uuid4()
    IdentityResolver._atomic_create_default_file(file_path, _payload(expected_id))
    payload = IdentityResolver._read_default_file(file_path)
    assert payload is not None
    assert _parse_uuid(str(payload["agent_id"])) == expected_id


def test_atomic_create_default_file_returns_false_when_file_exists(tmp_path: Path) -> None:
    """Race-loss path: peer process already created the file."""
    file_path = tmp_path / "default-agent.json"
    first_id = uuid.uuid4()
    assert IdentityResolver._atomic_create_default_file(file_path, _payload(first_id)) is True

    # Second attempt from a "racing peer" must return False, leave file unchanged.
    other_id = uuid.uuid4()
    assert IdentityResolver._atomic_create_default_file(file_path, _payload(other_id)) is False
    payload = IdentityResolver._read_default_file(file_path)
    assert payload is not None
    assert _parse_uuid(str(payload["agent_id"])) == first_id  # winner persists


def test_read_or_create_recovers_from_lost_race(tmp_path: Path) -> None:
    """Simulate a peer winning the race between our read() and our create()."""
    file_path = tmp_path / "default-agent.json"
    settings = _make_settings(tmp_path)
    resolver = IdentityResolver(settings)

    peer_id = uuid.uuid4()
    real_create = IdentityResolver._atomic_create_default_file
    call_state = {"first_call": True}

    def racing_create(path: Path, payload: dict[str, object]) -> bool:
        if call_state["first_call"]:
            # Peer wins right before our create — write peer's UUID then refuse ours.
            call_state["first_call"] = False
            real_create(path, _payload(peer_id))
            return False
        return real_create(path, payload)

    # Patch onto the instance class — we want the wrapper, not bound state.
    IdentityResolver._atomic_create_default_file = staticmethod(racing_create)  # type: ignore[assignment]
    try:
        resolved = resolver._read_or_create_default_file(file_path)
    finally:
        IdentityResolver._atomic_create_default_file = staticmethod(real_create)  # type: ignore[assignment]

    assert resolved == peer_id, "must adopt peer's UUID after losing the race"


# ---------------------------------------------------------------------------
# UUID parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["not-a-uuid", "", "1234", "00000000"])
def test_parse_uuid_rejects_bad_input(raw: str) -> None:
    with pytest.raises(ValueError, match="invalid UUID"):
        _parse_uuid(raw)


def test_parse_uuid_accepts_canonical_form() -> None:
    expected = uuid.uuid4()
    assert _parse_uuid(str(expected)) == expected


def test_parse_uuid_accepts_hex_no_dashes() -> None:
    expected = uuid.uuid4()
    assert _parse_uuid(expected.hex) == expected
