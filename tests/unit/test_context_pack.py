"""Unit tests for ``mem_context_pack`` orchestration."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from memory_mcp.context_pack import api
from memory_mcp.errors import InvalidInputError


@dataclass
class FakeMemory:
    id: UUID
    kind: str
    body: str
    title: str | None = None
    salience: float = 0.5
    created_at: dt.datetime = dt.datetime(2026, 5, 12, tzinfo=dt.UTC)
    updated_at: dt.datetime = dt.datetime(2026, 5, 12, tzinfo=dt.UTC)
    status: str = "active"
    macro: str | None = None
    decision_meta: dict | None = None


@dataclass
class FakeTask:
    id: UUID
    title: str
    status: str
    priority: int
    updated_at: dt.datetime = dt.datetime(2026, 5, 12, tzinfo=dt.UTC)
    blocked: bool = False


def mem(
    kind: str,
    body: str,
    *,
    title: str | None = None,
    salience: float = 0.5,
    status: str = "active",
    macro: str | None = None,
    decision_meta: dict | None = None,
) -> FakeMemory:
    return FakeMemory(
        id=uuid4(),
        kind=kind,
        title=title,
        body=body,
        salience=salience,
        status=status,
        macro=macro,
        decision_meta=decision_meta,
    )


def task(title: str, status: str = "pending", *, priority: int = 50, blocked: bool = False) -> FakeTask:
    return FakeTask(id=uuid4(), title=title, status=status, priority=priority, blocked=blocked)


def patch_fetches(
    monkeypatch: pytest.MonkeyPatch,
    *,
    digest: FakeMemory | None = None,
    triggers: list[FakeMemory] | None = None,
    journal: list[FakeMemory] | None = None,
    tasks: list[FakeTask] | None = None,
    decisions: list[FakeMemory] | None = None,
    playbooks: list[FakeMemory] | None = None,
    archival: list[FakeMemory] | None = None,
) -> None:
    async def fetch_digest(env_id: UUID) -> FakeMemory | None:
        return digest

    async def fetch_triggers(task_desc: str, env_id: UUID, *, top_k: int) -> list[FakeMemory]:
        return triggers or []

    async def fetch_journal(env_id: UUID, *, limit: int = 50) -> list[FakeMemory]:
        return journal or []

    async def fetch_tasks(env_id: UUID, *, top_k: int = 5) -> list[FakeTask]:
        return tasks or []

    async def fetch_decisions(
        env_id: UUID,
        *,
        top_k: int = 5,
        exclude_ids: set[UUID] | None = None,
    ) -> list[FakeMemory]:
        excluded = exclude_ids or set()
        return [m for m in (decisions or []) if m.id not in excluded]

    async def fetch_playbooks(
        task_desc: str,
        env_id: UUID,
        *,
        top_k: int = 3,
        exclude_ids: set[UUID] | None = None,
    ) -> list[FakeMemory]:
        excluded = exclude_ids or set()
        return [m for m in (playbooks or []) if m.id not in excluded]

    async def fetch_archival(env_id: UUID, *, exclude_ids: set[UUID], limit: int = 100) -> list[FakeMemory]:
        return [m for m in (archival or []) if m.id not in exclude_ids]

    monkeypatch.setattr(api, "_fetch_latest_digest", fetch_digest)
    monkeypatch.setattr(api, "_fetch_trigger_matches", fetch_triggers)
    monkeypatch.setattr(api, "_fetch_recent_journal", fetch_journal)
    monkeypatch.setattr(api, "_fetch_tasks", fetch_tasks)
    monkeypatch.setattr(api, "_fetch_decisions", fetch_decisions)
    monkeypatch.setattr(api, "_fetch_playbooks", fetch_playbooks)
    monkeypatch.setattr(api, "_fetch_archival", fetch_archival)


@pytest.mark.asyncio
async def test_empty_env_skips_all_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fetches(monkeypatch)

    out = await api.pack("implement feature", uuid4())

    assert out.sections == []
    assert out.sections_skipped == [
        "digest",
        "trigger_matched",
        "recent_journal",
        "tasks",
        "decisions",
        "playbooks",
        "archival",
    ]
    assert out.total_tokens == 0


@pytest.mark.asyncio
async def test_under_budget_includes_all_sections_without_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fetches(
        monkeypatch,
        digest=mem("session_digest", "## brief\nsmall\n## active_context\ncontext", title="digest"),
        triggers=[mem("fact", "trigger body", title="trigger", salience=0.8)],
        journal=[mem("journal_entry", "journal body", title="journal")],
        archival=[mem("decision", "archival body", title="archive", salience=0.7)],
    )

    out = await api.pack("implement feature", uuid4(), token_budget=1000)

    assert [section.name for section in out.sections] == [
        "digest",
        "trigger_matched",
        "recent_journal",
        "archival",
    ]
    assert out.sections_skipped == ["tasks", "decisions", "playbooks"]
    assert out.total_tokens < 1000
    assert sum(section.truncation_count for section in out.sections) == 0


@pytest.mark.asyncio
async def test_over_budget_truncates_proportionally(monkeypatch: pytest.MonkeyPatch) -> None:
    large = " ".join(f"word{i}" for i in range(1000))
    patch_fetches(
        monkeypatch,
        digest=mem(
            "session_digest",
            f"# Full\n{large}\n## brief\n{large}\n## active_context\n{large}",
            title="digest",
        ),
        triggers=[mem("fact", large, title="trigger")],
        journal=[mem("journal_entry", large, title="journal")],
        archival=[mem("procedure", large, title="archive")],
    )

    out = await api.pack("large task", uuid4(), token_budget=200)

    assert out.total_tokens <= 200
    assert sum(section.truncation_count for section in out.sections) >= 4
    assert all(section.tokens_used <= section.cap_tokens for section in out.sections)


@pytest.mark.asyncio
async def test_missing_digest_skips_digest_and_reallocates_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fetches(
        monkeypatch,
        triggers=[mem("fact", "trigger body", title="trigger")],
        journal=[mem("journal_entry", "journal body", title="journal")],
        archival=[mem("procedure", "archival body", title="archive")],
    )

    out = await api.pack("implement feature", uuid4(), token_budget=1000)

    assert "digest" in out.sections_skipped
    trigger_section = next(section for section in out.sections if section.name == "trigger_matched")
    assert trigger_section.cap_tokens > 400


@pytest.mark.asyncio
async def test_missing_triggers_skips_trigger_section(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fetches(
        monkeypatch,
        digest=mem("session_digest", "digest body", title="digest"),
        triggers=[],
        journal=[mem("journal_entry", "journal body", title="journal")],
        archival=[mem("fact", "archival body", title="archive")],
    )

    out = await api.pack("no trigger matches", uuid4())

    assert "trigger_matched" in out.sections_skipped
    assert "trigger_matched" not in [section.name for section in out.sections]


@pytest.mark.asyncio
async def test_include_journal_false_absent_and_reallocated(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_fetch_journal(env_id: UUID, *, limit: int = 50) -> list[FakeMemory]:
        raise AssertionError("journal fetch should not be called")

    patch_fetches(
        monkeypatch,
        digest=mem("session_digest", "digest body", title="digest"),
        triggers=[mem("fact", "trigger body", title="trigger")],
        journal=[mem("journal_entry", "journal body", title="journal")],
        archival=[mem("procedure", "archival body", title="archive")],
    )
    monkeypatch.setattr(api, "_fetch_recent_journal", fail_fetch_journal)

    out = await api.pack("implement feature", uuid4(), token_budget=1000, include_journal=False)

    assert "recent_journal" not in [section.name for section in out.sections]
    digest_section = next(section for section in out.sections if section.name == "digest")
    assert digest_section.cap_tokens > 250


@pytest.mark.asyncio
async def test_empty_task_desc_raises_invalid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fetches(monkeypatch)

    with pytest.raises(InvalidInputError):
        await api.pack("   ", uuid4())


@pytest.mark.asyncio
async def test_trigger_search_failure_skips_trigger_section_and_reallocates(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = mem("session_digest", "digest body", title="digest")
    journal = [mem("journal_entry", "journal body", title="journal")]
    archival = [mem("fact", "archival body", title="archive")]

    async def fetch_digest(env_id: UUID) -> FakeMemory | None:
        return digest

    async def fetch_journal(env_id: UUID, *, limit: int = 50) -> list[FakeMemory]:
        return journal

    async def fetch_archival(env_id: UUID, *, exclude_ids: set[UUID], limit: int = 100) -> list[FakeMemory]:
        return [m for m in archival if m.id not in exclude_ids]

    async def fetch_tasks(env_id: UUID, *, top_k: int = 5) -> list[FakeTask]:
        return []

    async def fetch_decisions(
        env_id: UUID,
        *,
        top_k: int = 5,
        exclude_ids: set[UUID] | None = None,
    ) -> list[FakeMemory]:
        return []

    async def fetch_playbooks(
        task_desc: str,
        env_id: UUID,
        *,
        top_k: int = 3,
        exclude_ids: set[UUID] | None = None,
    ) -> list[FakeMemory]:
        return []

    def fail_trigger(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("qdrant unavailable")

    from memory_mcp.search import api as search_api

    monkeypatch.setattr(api, "_fetch_latest_digest", fetch_digest)
    monkeypatch.setattr(api, "_fetch_recent_journal", fetch_journal)
    monkeypatch.setattr(api, "_fetch_tasks", fetch_tasks)
    monkeypatch.setattr(api, "_fetch_decisions", fetch_decisions)
    monkeypatch.setattr(api, "_fetch_playbooks", fetch_playbooks)
    monkeypatch.setattr(api, "_fetch_archival", fetch_archival)
    monkeypatch.setattr(search_api, "_search_by_trigger", fail_trigger)

    out = await api.pack("implement feature", uuid4(), token_budget=1000)

    assert "trigger_matched" in out.sections_skipped
    assert "trigger_matched" not in [section.name for section in out.sections]
    digest_section = next(section for section in out.sections if section.name == "digest")
    assert digest_section.cap_tokens > 250


@pytest.mark.asyncio
async def test_playbook_section_appears_for_macro_token_match(monkeypatch: pytest.MonkeyPatch) -> None:
    pb = mem("playbook", "Run deploy checklist", title="deploy pb", macro="deploy-service")

    async def fetch_playbooks(task_desc: str, env_id: UUID, *, top_k: int = 3, exclude_ids=None):
        tokens = api._task_desc_tokens(task_desc)  # noqa: SLF001
        return [pb] if any(token in (pb.macro or "").lower() for token in tokens) else []

    patch_fetches(monkeypatch, digest=mem("session_digest", "digest", title="digest"))
    monkeypatch.setattr(api, "_fetch_playbooks", fetch_playbooks)

    out = await api.pack("Please deploy the service", uuid4(), token_budget=1000)

    section = next(section for section in out.sections if section.name == "playbooks")
    assert section.items[0].memory_id == pb.id


@pytest.mark.asyncio
async def test_playbook_section_appears_for_two_character_macro_token(monkeypatch: pytest.MonkeyPatch) -> None:
    pb = mem("playbook", "Debug CI failure", title="ci pb", macro="ci")

    async def fetch_playbooks(task_desc: str, env_id: UUID, *, top_k: int = 3, exclude_ids=None):
        tokens = api._task_desc_tokens(task_desc)  # noqa: SLF001
        return [pb] if any(token in (pb.macro or "").lower() for token in tokens) else []

    patch_fetches(monkeypatch, digest=mem("session_digest", "digest", title="digest"))
    monkeypatch.setattr(api, "_fetch_playbooks", fetch_playbooks)

    out = await api.pack("debug ci failure", uuid4(), token_budget=1000)

    section = next(section for section in out.sections if section.name == "playbooks")
    assert section.items[0].memory_id == pb.id


@pytest.mark.asyncio
async def test_playbook_semantic_match_supplements_macro_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    macro = mem("playbook", "Macro playbook", title="macro", macro="deploy")
    semantic = mem("playbook", "Semantic playbook", title="semantic")

    async def fetch_playbooks(task_desc: str, env_id: UUID, *, top_k: int = 3, exclude_ids=None):
        return [macro, semantic]

    patch_fetches(monkeypatch, digest=mem("session_digest", "digest", title="digest"))
    monkeypatch.setattr(api, "_fetch_playbooks", fetch_playbooks)

    out = await api.pack("deploy release", uuid4(), token_budget=1000)
    items = next(section.items for section in out.sections if section.name == "playbooks")

    assert [item.title for item in items] == ["macro", "semantic"]


@pytest.mark.asyncio
async def test_playbook_qdrant_failure_degrades_and_section_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fetch_playbooks(task_desc: str, env_id: UUID, *, top_k: int = 3, exclude_ids=None):
        raise RuntimeError("qdrant unavailable")

    patch_fetches(
        monkeypatch,
        digest=mem("session_digest", "digest", title="digest"),
        triggers=[mem("fact", "trigger body", title="trigger")],
    )
    monkeypatch.setattr(api, "_fetch_playbooks", fetch_playbooks)

    out = await api.pack("deploy release", uuid4(), token_budget=1000)

    assert "playbooks" in out.sections_skipped
    assert [section.name for section in out.sections] == ["digest", "trigger_matched"]


@pytest.mark.asyncio
async def test_playbook_section_empty_when_no_playbooks_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fetches(monkeypatch, digest=mem("session_digest", "digest", title="digest"), playbooks=[])

    out = await api.pack("deploy release", uuid4(), token_budget=1000)

    assert "playbooks" in out.sections_skipped
    assert "playbooks" not in [section.name for section in out.sections]


@pytest.mark.asyncio
async def test_tasks_section_orders_in_progress_first_then_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [
        task("in progress high", "in_progress", priority=20),
        task("in progress low", "in_progress", priority=50),
        task("pending top", "pending", priority=10),
        task("pending later", "pending", priority=30),
    ]
    blocked = task("pending blocked", "pending", priority=1, blocked=True)

    async def fetch_tasks(env_id: UUID, *, top_k: int = 5):
        in_progress = sorted([t for t in tasks if t.status == "in_progress"], key=lambda t: t.priority)[:3]
        pending = sorted(
            [t for t in [*tasks, blocked] if t.status == "pending" and not t.blocked],
            key=lambda t: t.priority,
        )
        return [*in_progress, *pending][:top_k]

    patch_fetches(monkeypatch, digest=mem("session_digest", "digest", title="digest"))
    monkeypatch.setattr(api, "_fetch_tasks", fetch_tasks)

    out = await api.pack("continue work", uuid4(), token_budget=1000)
    titles = [item.title for item in next(section.items for section in out.sections if section.name == "tasks")]

    assert titles == ["in progress high", "in progress low", "pending top", "pending later"]


@pytest.mark.asyncio
async def test_tasks_section_skips_done_and_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    visible = task("visible", "pending")
    all_tasks = [visible, task("done", "done"), task("cancelled", "cancelled")]

    async def fetch_tasks(env_id: UUID, *, top_k: int = 5):
        return [t for t in all_tasks if t.status not in {"done", "cancelled"}]

    patch_fetches(monkeypatch)
    monkeypatch.setattr(api, "_fetch_tasks", fetch_tasks)

    out = await api.pack("continue work", uuid4(), token_budget=1000)

    items = next(section.items for section in out.sections if section.name == "tasks")
    assert [item.title for item in items] == ["visible"]


@pytest.mark.asyncio
async def test_tasks_unblocked_pending_respects_dependency_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    unblocked = task("unblocked pending", "pending")
    blocked = task("blocked pending", "pending", blocked=True)

    async def fetch_tasks(env_id: UUID, *, top_k: int = 5):
        return [t for t in [unblocked, blocked] if not t.blocked]

    patch_fetches(monkeypatch)
    monkeypatch.setattr(api, "_fetch_tasks", fetch_tasks)

    out = await api.pack("continue work", uuid4(), token_budget=1000)
    titles = [item.title for item in next(section.items for section in out.sections if section.name == "tasks")]

    assert titles == ["unblocked pending"]


@pytest.mark.asyncio
async def test_decisions_section_filters_accepted_decision_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    accepted = mem("decision", "Use Postgres", title="accepted", decision_meta={"status": "accepted"})
    proposed = mem("decision", "Maybe Redis", title="proposed", decision_meta={"status": "proposed"})

    async def fetch_decisions(env_id: UUID, *, top_k: int = 5, exclude_ids=None):
        return [m for m in [accepted, proposed] if (m.decision_meta or {}).get("status") == "accepted"]

    patch_fetches(monkeypatch)
    monkeypatch.setattr(api, "_fetch_decisions", fetch_decisions)

    out = await api.pack("decide storage", uuid4(), token_budget=1000)

    items = next(section.items for section in out.sections if section.name == "decisions")
    assert [item.title for item in items] == ["accepted"]


@pytest.mark.asyncio
async def test_decisions_section_ignores_null_decision_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    accepted = mem("decision", "Accepted", title="accepted", decision_meta={"status": "accepted"})
    null_meta = mem("decision", "Null", title="null", decision_meta=None)

    async def fetch_decisions(env_id: UUID, *, top_k: int = 5, exclude_ids=None):
        return [m for m in [accepted, null_meta] if m.decision_meta is not None]

    patch_fetches(monkeypatch)
    monkeypatch.setattr(api, "_fetch_decisions", fetch_decisions)

    out = await api.pack("decide storage", uuid4(), token_budget=1000)

    items = next(section.items for section in out.sections if section.name == "decisions")
    assert [item.title for item in items] == ["accepted"]


@pytest.mark.asyncio
async def test_decisions_section_excludes_archived_memories(monkeypatch: pytest.MonkeyPatch) -> None:
    active = mem("decision", "Active", title="active", status="active", decision_meta={"status": "accepted"})
    archived = mem("decision", "Archived", title="archived", status="archived", decision_meta={"status": "accepted"})

    async def fetch_decisions(env_id: UUID, *, top_k: int = 5, exclude_ids=None):
        return [m for m in [active, archived] if m.status in {"proposed", "active"}]

    patch_fetches(monkeypatch)
    monkeypatch.setattr(api, "_fetch_decisions", fetch_decisions)

    out = await api.pack("decide storage", uuid4(), token_budget=1000)

    items = next(section.items for section in out.sections if section.name == "decisions")
    assert [item.title for item in items] == ["active"]


@pytest.mark.asyncio
async def test_full_pack_all_seven_sections_respects_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fetches(
        monkeypatch,
        digest=mem("session_digest", "digest body", title="digest"),
        triggers=[mem("fact", "trigger body", title="trigger")],
        journal=[mem("journal_entry", "journal body", title="journal")],
        tasks=[task("task", "in_progress", priority=1)],
        decisions=[mem("decision", "decision body", title="decision", decision_meta={"status": "accepted"})],
        playbooks=[mem("playbook", "playbook body", title="playbook", macro="deploy")],
        archival=[mem("fact", "archival body", title="archival")],
    )

    out = await api.pack("deploy feature", uuid4(), token_budget=300)

    assert [section.name for section in out.sections] == list(api._SECTION_ORDER)  # noqa: SLF001
    assert sum(section.tokens_used for section in out.sections) <= 300


@pytest.mark.asyncio
async def test_section_order_matches_canonical_order(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fetches(
        monkeypatch,
        digest=mem("session_digest", "digest body", title="digest"),
        triggers=[mem("fact", "trigger body", title="trigger")],
        journal=[mem("journal_entry", "journal body", title="journal")],
        tasks=[task("task", "in_progress")],
        decisions=[mem("decision", "decision body", title="decision", decision_meta={"status": "accepted"})],
        playbooks=[mem("playbook", "playbook body", title="playbook")],
        archival=[mem("fact", "archival body", title="archival")],
    )

    out = await api.pack("deploy feature", uuid4(), token_budget=1000)

    assert [section.name for section in out.sections] == [
        "digest",
        "trigger_matched",
        "recent_journal",
        "tasks",
        "decisions",
        "playbooks",
        "archival",
    ]


@pytest.mark.asyncio
async def test_sections_skipped_when_new_section_data_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fetches(monkeypatch, triggers=[mem("fact", "trigger body", title="trigger")])

    out = await api.pack("implement feature", uuid4(), token_budget=1000)

    assert out.sections[0].name == "trigger_matched"
    assert {"digest", "tasks", "decisions", "playbooks"}.issubset(set(out.sections_skipped))


@pytest.mark.asyncio
async def test_archival_section_still_excludes_archived_status_memories(monkeypatch: pytest.MonkeyPatch) -> None:
    active = mem("fact", "active archival", title="active", status="active")
    archived = mem("fact", "archived archival", title="archived", status="archived")

    async def fetch_archival(env_id: UUID, *, exclude_ids: set[UUID], limit: int = 100):
        return [m for m in [active, archived] if m.status in {"proposed", "active"} and m.id not in exclude_ids]

    patch_fetches(monkeypatch)
    monkeypatch.setattr(api, "_fetch_archival", fetch_archival)

    out = await api.pack("implement feature", uuid4(), token_budget=1000)

    items = next(section.items for section in out.sections if section.name == "archival")
    assert [item.title for item in items] == ["active"]


@pytest.mark.asyncio
async def test_recent_journal_appears_before_tasks_decisions_playbooks(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fetches(
        monkeypatch,
        journal=[mem("journal_entry", "journal body", title="journal")],
        tasks=[task("task", "in_progress")],
        decisions=[mem("decision", "decision body", title="decision", decision_meta={"status": "accepted"})],
        playbooks=[mem("playbook", "playbook body", title="playbook")],
    )

    out = await api.pack("continue work", uuid4(), token_budget=1000)
    names = [section.name for section in out.sections]

    assert names.index("recent_journal") < names.index("tasks") < names.index("decisions") < names.index("playbooks")
