"""Shared env-id/env-name validator helpers for request schemas."""

from __future__ import annotations

from typing import Any


def validate_optional_env_ref_pair(
    model: Any,
    *,
    id_field: str = "env_id",
    name_field: str = "env_name",
) -> Any:
    env_id = getattr(model, id_field)
    env_name = getattr(model, name_field)
    if env_id is not None and env_name is not None:
        raise ValueError(f"{id_field} and {name_field} are mutually exclusive")
    return model


def validate_required_env_ref_pair(
    model: Any,
    *,
    id_field: str = "env_id",
    name_field: str = "env_name",
) -> Any:
    validate_optional_env_ref_pair(model, id_field=id_field, name_field=name_field)
    if getattr(model, id_field) is None and getattr(model, name_field) is None:
        raise ValueError(f"exactly one of {id_field} or {name_field} must be set")
    return model


def validate_optional_env_ref_list_pair(
    model: Any,
    *,
    ids_field: str = "env_ids",
    names_field: str = "env_names",
) -> Any:
    env_ids = getattr(model, ids_field)
    env_names = getattr(model, names_field)
    if env_ids is not None and env_names is not None:
        raise ValueError(f"{ids_field} and {names_field} are mutually exclusive")
    return model


def validate_required_env_ref_list_pair(
    model: Any,
    *,
    ids_field: str = "env_ids",
    names_field: str = "env_names",
) -> Any:
    validate_optional_env_ref_list_pair(model, ids_field=ids_field, names_field=names_field)
    if getattr(model, ids_field) is None and getattr(model, names_field) is None:
        raise ValueError(f"exactly one of {ids_field} or {names_field} must be set")
    return model
