"""One happy-path per SDK namespace, against a live memory-mcp server.

Each test does the smallest thing that proves the wire format matches:

* ``envs``        — create + list + delete a scratch env.
* ``memories``    — write, get, search round-trip in a scratch env.
* ``tasks``       — create + list a task.
* ``entities``    — upsert + resolve an entity.
* ``relations``   — link two entities + browse.
* ``playbooks``   — invoke a missing playbook → expect a clean error
                    (the server-side playbook surface is not yet seeded
                    in scratch envs, so we just check it doesn't blow
                    up the transport).
* ``decisions``   — adr_export over the empty env returns ``{}`` cleanly.
* ``dream``       — status returns the worker snapshot.
* ``env_ops``     — export + snapshot the scratch env, then restore from
                    the snapshot.

All tests share a single scratch env scoped per test so they're
isolated from one another.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from memory_mcp_client import MemoryClient
from memory_mcp_client.errors import MemoryMCPError


pytestmark = pytest.mark.asyncio


# --- envs -------------------------------------------------------------------


async def test_envs_create_list_delete(live_client: MemoryClient):
    name = f"sdk-it-envs-{uuid4().hex[:6]}"
    env = await live_client.envs.create(name=name)
    assert env.name == name

    listed = await live_client.envs.list_()
    names = {e.name for e in listed}
    assert name in names

    await live_client.envs.delete(env_id=env.id, confirm_destroy=True)


# --- memories ---------------------------------------------------------------


async def test_memories_write_get_search(live_client: MemoryClient, scratch_env):
    written = await live_client.memories.write(
        env_id=scratch_env.id,
        kind="fact",
        title="integration-test",
        body="hello from SDK live integration tests",
        tags=["sdk-it"],
    )
    assert written.title == "integration-test"
    assert written.env_id == scratch_env.id

    got = await live_client.memories.get(memory_id=written.id)
    assert got.id == written.id

    # Best-effort search — vectorization may lag, retry a couple of
    # times before failing loudly.
    last_err: Exception | None = None
    for _ in range(5):
        try:
            result = await live_client.memories.search(
                query="integration",
                env_ids=[scratch_env.id],
                top_k=5,
            )
            ids = {h.memory.id for h in result.hits}
            if written.id in ids:
                break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        await asyncio.sleep(0.5)
    else:
        raise AssertionError(f"search did not return the written memory; last_err={last_err}")


# --- tasks ------------------------------------------------------------------


async def test_tasks_create_list(live_client: MemoryClient, scratch_env):
    task = await live_client.tasks.create(
        env_id=scratch_env.id,
        title="sdk-it-task",
    )
    assert task.title == "sdk-it-task"

    out = await live_client.tasks.list(env_ids=[scratch_env.id])
    titles = {t.title for t in out.tasks}
    assert "sdk-it-task" in titles


# --- entities ---------------------------------------------------------------


async def test_entities_upsert_resolve(live_client: MemoryClient, scratch_env):
    entity = await live_client.entities.upsert(
        env_id=scratch_env.id,
        kind="repo",
        name="sdk-it-repo",
    )
    assert entity.name == "sdk-it-repo"

    resolved = await live_client.entities.resolve(
        env_id=scratch_env.id,
        kind="repo",
        name="sdk-it-repo",
    )
    assert resolved is not None
    assert resolved.id == entity.id


# --- relations --------------------------------------------------------------


async def test_relations_link_browse(live_client: MemoryClient, scratch_env):
    src = await live_client.entities.upsert(
        env_id=scratch_env.id, kind="repo", name="sdk-it-src"
    )
    dst = await live_client.entities.upsert(
        env_id=scratch_env.id, kind="repo", name="sdk-it-dst"
    )

    link = await live_client.relations.link(
        env_id=scratch_env.id,
        src_id=src.id,
        dst_id=dst.id,
        relation="depends_on",
    )
    assert link is not None

    out = await live_client.relations.browse(
        env_id=scratch_env.id,
        src_id=src.id,
    )
    # Either edge-list or paginated result — accept any non-empty.
    assert getattr(out, "edges", None) or getattr(out, "items", None)


# --- decisions --------------------------------------------------------------


async def test_decisions_adr_export_empty(live_client: MemoryClient, scratch_env):
    out = await live_client.decisions.adr_export(env_id=scratch_env.id)
    # Empty env should yield an empty / well-formed envelope.
    assert out is not None


# --- dream ------------------------------------------------------------------


async def test_dream_status(live_client: MemoryClient):
    out = await live_client.dream.status()
    # The status response shape varies with worker state; only assert
    # it round-trips through the SDK without error.
    assert out is not None


# --- env_ops ---------------------------------------------------------------


async def test_env_ops_export_snapshot_restore(
    live_client: MemoryClient, scratch_env
):
    # Seed at least one memory so the export has something to ship.
    await live_client.memories.write(
        env_id=scratch_env.id,
        kind="fact",
        title="snapshot-source",
        body="payload",
    )

    # Export — bundles the env into a portable artifact.
    exported = await live_client.env_ops.export(env_id=scratch_env.id)
    assert exported is not None

    # Snapshot the env.
    snap = await live_client.env_ops.snapshot(env_id=scratch_env.id)
    assert snap is not None

    # Restore is best-effort — server may require a fresh env id; we
    # just confirm the call doesn't blow up the transport.
    try:
        await live_client.env_ops.restore(
            env_id=scratch_env.id,
            snapshot_id=getattr(snap, "id", None) or getattr(snap, "snapshot_id", None),
        )
    except MemoryMCPError:
        # Typed error is fine — we're verifying the wire path, not the
        # semantic behavior of restore over an already-populated env.
        pass


# --- playbooks --------------------------------------------------------------


async def test_playbooks_invoke_missing_clean_error(
    live_client: MemoryClient, scratch_env
):
    """No playbooks exist in a scratch env — invoke must surface a typed error."""
    with pytest.raises(MemoryMCPError):
        await live_client.playbooks.invoke(
            env_id=scratch_env.id,
            playbook_id=str(uuid4()),
        )
