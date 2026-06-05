"""Schema coverage guard for friendly env-name fields."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from pydantic import BaseModel

SKIP = {
    # Response/support schemas that still expose canonical env UUIDs only.
    "DigestMemoryEntry",
    "DreamHeartbeatEntry",
    "DreamProposalEntry",
    "DreamRunReport",
    "DreamRunScheduledItem",
    "DreamRunSummaryEntry",
    "EntityResponse",
    "EnvDeleteReport",
    "EnvDeleteResponse",
    "EnvRenameResponse",
    "EnvSnapshotResponse",
    "JournalResponse",
    "MemSourceHit",
    "MemoryResponse",
    "NeighborNodeResponse",
    "ProjectionStatusEntry",
    "RelationBrowseHit",
    "RelationResponse",
    "SnapshotResponse",
    "TaskResponse",
}


def walk_pydantic_models(package_name: str) -> list[type[BaseModel]]:
    package = importlib.import_module(package_name)
    models: list[type[BaseModel]] = []
    for modinfo in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(modinfo.name)
        for _name, candidate in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(candidate, BaseModel)
                and candidate is not BaseModel
                and candidate.__module__ == module.__name__
            ):
                models.append(candidate)
    return models


def test_every_env_id_field_has_matching_env_name() -> None:
    missing: set[str] = set()
    for model in walk_pydantic_models("memory_mcp_schemas"):
        fields = model.model_fields
        if "env_id" in fields and "env_name" not in fields:
            missing.add(model.__name__)
        if "env_ids" in fields and "env_names" not in fields:
            missing.add(model.__name__)

    assert missing == SKIP

    for model in walk_pydantic_models("memory_mcp_schemas"):
        if model.__name__ in SKIP:
            continue
        fields = model.model_fields
        if "env_id" in fields:
            assert "env_name" in fields, f"{model.__name__}: env_id without env_name"
        if "env_ids" in fields:
            assert "env_names" in fields, f"{model.__name__}: env_ids without env_names"
