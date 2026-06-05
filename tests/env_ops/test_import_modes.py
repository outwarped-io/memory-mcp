from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from memory_mcp_schemas.env_ops import EnvImportRequest, ExportFormat, ImportMode
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from test_roundtrip import _create_target_env, _export, postgres_factory, roundtrip_db  # noqa: F401

from memory_mcp import entities as entities_mod
from memory_mcp.db.models import Entity, EntityAlias, Environment, Memory, MemoryTag, Tag
from memory_mcp.env_ops import import_ as importer
from memory_mcp.env_ops.import_ import import_env
from memory_mcp.identity import AgentContext


@pytest.fixture
def ctx() -> AgentContext:
    return AgentContext(agent_id=uuid4(), agent_name="import-modes-agent")


@pytest.fixture
async def import_modes_db(
    roundtrip_db: tuple[AsyncSession, Any, AgentContext],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[AsyncSession, AgentContext]:
    monkeypatch.setattr(entities_mod, "session_scope", importer.session_scope)
    session, _store, ctx = roundtrip_db
    return session, ctx


@pytest.mark.asyncio
async def test_import_overwrite_replaces_tag_with_same_name(
    tmp_path: Path,
    import_modes_db: tuple[AsyncSession, AgentContext],
) -> None:
    session, ctx = import_modes_db
    src_env_id = await _create_tagged_env(session, tag_name="x", memory_count=1)
    src = await _export(src_env_id, tmp_path / "overwrite_tag_src", ExportFormat.directory, ctx)
    target = await _create_target_env("overwrite-tag", ctx)
    old_tag_id = uuid4()
    old_memories = await _add_memories(session, target.id, 2)
    session.add(Tag(id=old_tag_id, env_id=target.id, name="x"))
    await session.flush()
    session.add_all([MemoryTag(memory_id=memory_id, tag_id=old_tag_id, env_id=target.id) for memory_id in old_memories])
    await session.commit()

    await import_env(
        EnvImportRequest(
            source_path=src.output_path, target_env_id=target.id, mode=ImportMode.overwrite, dry_run=False
        ),
        ctx=ctx,
    )

    tags = (await session.execute(select(Tag).where(Tag.env_id == target.id, Tag.name == "x"))).scalars().all()
    assert len(tags) == 1
    assert tags[0].id != old_tag_id
    assert (
        await _count(session, select(MemoryTag).where(MemoryTag.env_id == target.id, MemoryTag.tag_id == old_tag_id))
        == 0
    )
    assert (
        await _count(session, select(MemoryTag).where(MemoryTag.env_id == target.id, MemoryTag.tag_id == tags[0].id))
        == 1
    )


@pytest.mark.asyncio
async def test_import_overwrite_requires_admin(
    monkeypatch: pytest.MonkeyPatch,
    ctx: AgentContext,
) -> None:
    def deny_admin(role: str, **_kwargs: object) -> None:
        if role == "admin":
            raise PermissionError("import mode=overwrite requires admin")

    monkeypatch.setattr(importer.rbac, "require", deny_admin)
    with pytest.raises(PermissionError, match="overwrite requires admin"):
        await import_env(
            EnvImportRequest(
                source_path="unused",
                target_env_name="blocked-overwrite",
                mode=ImportMode.overwrite,
                dry_run=False,
            ),
            ctx=ctx,
        )


@pytest.mark.asyncio
async def test_import_merge_unions_tags(
    tmp_path: Path,
    import_modes_db: tuple[AsyncSession, AgentContext],
) -> None:
    session, ctx = import_modes_db
    src_env_id = await _create_tagged_env(session, tag_name="x", memory_count=1)
    src = await _export(src_env_id, tmp_path / "merge_tag_src", ExportFormat.directory, ctx)
    target = await _create_target_env("merge-tag", ctx)
    old_tag_id = uuid4()
    session.add(Tag(id=old_tag_id, env_id=target.id, name="x"))
    await session.commit()

    await import_env(
        EnvImportRequest(source_path=src.output_path, target_env_id=target.id, mode=ImportMode.merge, dry_run=False),
        ctx=ctx,
    )

    tags = (await session.execute(select(Tag).where(Tag.env_id == target.id, Tag.name == "x"))).scalars().all()
    assert len(tags) == 1
    assert tags[0].id == old_tag_id
    assert (
        await _count(session, select(MemoryTag).where(MemoryTag.env_id == target.id, MemoryTag.tag_id == old_tag_id))
        == 1
    )


@pytest.mark.asyncio
async def test_import_merge_collapses_entity_collision_via_ent_merge(
    tmp_path: Path,
    import_modes_db: tuple[AsyncSession, AgentContext],
) -> None:
    session, ctx = import_modes_db
    src_env_id = await _create_entity_env(session, canonical="Source E1", normalized="e1", aliases=["source alias"])
    src = await _export(src_env_id, tmp_path / "merge_entity_src", ExportFormat.directory, ctx)
    target = await _create_target_env("merge-entity", ctx)
    keep_id = await _add_entity(session, target.id, canonical="Dest E1", normalized="e1", aliases=["dest alias"])
    await session.commit()

    report = await import_env(
        EnvImportRequest(source_path=src.output_path, target_env_id=target.id, mode=ImportMode.merge, dry_run=False),
        ctx=ctx,
    )

    entities = (await session.execute(select(Entity).where(Entity.env_id == target.id))).scalars().all()
    aliases = (await session.execute(select(EntityAlias).where(EntityAlias.env_id == target.id))).scalars().all()
    assert report.entity_merges_performed >= 1
    assert len([entity for entity in entities if entity.normalized_name == "e1"]) == 1
    assert {alias.normalized_alias for alias in aliases} == {"dest alias", "source alias"}
    assert {alias.entity_id for alias in aliases} == {keep_id}


@pytest.mark.asyncio
async def test_import_overwrite_deletes_dst_entity_and_replaces(
    tmp_path: Path,
    import_modes_db: tuple[AsyncSession, AgentContext],
) -> None:
    session, ctx = import_modes_db
    src_env_id = await _create_entity_env(session, canonical="Source E1", normalized="e1", aliases=["source alias"])
    src = await _export(src_env_id, tmp_path / "overwrite_entity_src", ExportFormat.directory, ctx)
    target = await _create_target_env("overwrite-entity", ctx)
    old_entity_id = await _add_entity(
        session, target.id, canonical="Dest E1", normalized="e1", aliases=["old a", "old b"]
    )
    await session.commit()

    await import_env(
        EnvImportRequest(
            source_path=src.output_path, target_env_id=target.id, mode=ImportMode.overwrite, dry_run=False
        ),
        ctx=ctx,
    )

    entities = (await session.execute(select(Entity).where(Entity.env_id == target.id))).scalars().all()
    aliases = (await session.execute(select(EntityAlias).where(EntityAlias.env_id == target.id))).scalars().all()
    assert len(entities) == 1
    assert entities[0].id != old_entity_id
    assert entities[0].canonical_name == "Source E1"
    assert [alias.normalized_alias for alias in aliases] == ["source alias"]
    assert aliases[0].entity_id == entities[0].id


@pytest.mark.asyncio
async def test_import_overwrite_cascade_on_memory_not_triggered(
    tmp_path: Path,
    import_modes_db: tuple[AsyncSession, AgentContext],
) -> None:
    session, ctx = import_modes_db
    src_env_id = await _create_tagged_env(session, tag_name="source", memory_count=2)
    src = await _export(src_env_id, tmp_path / "overwrite_memories_src", ExportFormat.directory, ctx)
    target = await _create_target_env("overwrite-memories", ctx)
    await _add_memories(session, target.id, 1)
    await session.commit()

    await import_env(
        EnvImportRequest(
            source_path=src.output_path, target_env_id=target.id, mode=ImportMode.overwrite, dry_run=False
        ),
        ctx=ctx,
    )

    assert await _count(session, select(Memory).where(Memory.env_id == target.id)) == 3


@pytest.mark.asyncio
async def test_import_merge_extends_report_with_entity_merges(
    tmp_path: Path,
    import_modes_db: tuple[AsyncSession, AgentContext],
) -> None:
    session, ctx = import_modes_db
    src_env_id = await _create_entity_env(session, canonical="Source E1", normalized="e1", aliases=["source alias"])
    src = await _export(src_env_id, tmp_path / "merge_report_src", ExportFormat.directory, ctx)
    target = await _create_target_env("merge-report", ctx)
    await _add_entity(session, target.id, canonical="Dest E1", normalized="e1", aliases=[])
    await session.commit()

    report = await import_env(
        EnvImportRequest(source_path=src.output_path, target_env_id=target.id, mode=ImportMode.merge, dry_run=False),
        ctx=ctx,
    )

    assert report.entity_merges_performed == 1


async def _create_tagged_env(session: AsyncSession, *, tag_name: str, memory_count: int) -> UUID:
    env_id = uuid4()
    session.add(
        Environment(
            id=env_id,
            name=f"mode-src-{uuid4().hex[:8]}",
            kind="test",
            retention_policy={},
            default_embedding_model_id="test-model",
        )
    )
    await session.flush()
    tag_id = uuid4()
    session.add(Tag(id=tag_id, env_id=env_id, name=tag_name))
    await session.flush()
    memory_ids = await _add_memories(session, env_id, memory_count)
    session.add_all([MemoryTag(memory_id=memory_id, tag_id=tag_id, env_id=env_id) for memory_id in memory_ids])
    await session.commit()
    return env_id


async def _create_entity_env(session: AsyncSession, *, canonical: str, normalized: str, aliases: list[str]) -> UUID:
    env_id = uuid4()
    session.add(
        Environment(
            id=env_id,
            name=f"mode-entity-src-{uuid4().hex[:8]}",
            kind="test",
            retention_policy={},
            default_embedding_model_id="test-model",
        )
    )
    await session.flush()
    await _add_entity(session, env_id, canonical=canonical, normalized=normalized, aliases=aliases)
    await session.commit()
    return env_id


async def _add_memories(session: AsyncSession, env_id: UUID, count: int) -> list[UUID]:
    memory_ids = [uuid4() for _ in range(count)]
    session.add_all(
        [
            Memory(
                id=memory_id,
                env_id=env_id,
                kind="fact",
                status="active",
                title=f"memory {index}",
                body=f"body {index} {uuid4()}",
                metadata_={},
                version=1,
            )
            for index, memory_id in enumerate(memory_ids)
        ]
    )
    await session.flush()
    return memory_ids


async def _add_entity(
    session: AsyncSession,
    env_id: UUID,
    *,
    canonical: str,
    normalized: str,
    aliases: list[str],
) -> UUID:
    entity_id = uuid4()
    session.add(
        Entity(
            id=entity_id,
            env_id=env_id,
            kind="service",
            canonical_name=canonical,
            normalized_name=normalized,
            metadata_={},
            version=1,
        )
    )
    await session.flush()
    session.add_all(
        [EntityAlias(entity_id=entity_id, env_id=env_id, alias=alias, normalized_alias=alias) for alias in aliases]
    )
    await session.flush()
    return entity_id


async def _count(session: AsyncSession, stmt: Any) -> int:
    value = await session.scalar(select(func.count()).select_from(stmt.subquery()))
    return int(value or 0)
