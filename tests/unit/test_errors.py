"""Unit tests for the canonical error classes."""

from __future__ import annotations

import pytest

from memory_mcp.errors import (
    EmbeddingModelMismatchError,
    EnvAmbiguousError,
    EnvNotFoundError,
    EnvRefAmbiguousError,
    EnvRefBothProvidedError,
    ForbiddenEnvError,
    InvalidTransitionError,
    MemoryMCPError,
    NotFoundError,
    UnauthorizedError,
    VersionConflictError,
)


@pytest.mark.parametrize(
    "cls,expected_code",
    [
        (UnauthorizedError, "UNAUTHORIZED"),
        (ForbiddenEnvError, "FORBIDDEN_ENV"),
        (EnvAmbiguousError, "ENV_AMBIGUOUS"),
        (EnvRefBothProvidedError, "ENV_REF_BOTH_PROVIDED"),
        (EnvRefAmbiguousError, "ENV_REF_AMBIGUOUS"),
        (EnvNotFoundError, "ENV_NOT_FOUND"),
        (VersionConflictError, "VERSION_CONFLICT"),
        (InvalidTransitionError, "INVALID_TRANSITION"),
        (NotFoundError, "NOT_FOUND"),
        (EmbeddingModelMismatchError, "EMBEDDING_MODEL_MISMATCH"),
    ],
)
def test_error_codes_are_stable(cls: type[MemoryMCPError], expected_code: str) -> None:
    """The wire-format ``code`` is the public contract."""
    assert cls.code == expected_code
    # Class attr also accessible on instances of error classes that take no args.
    if cls in {UnauthorizedError, ForbiddenEnvError, EnvAmbiguousError, NotFoundError}:
        instance = cls("msg")
        assert instance.code == expected_code
    elif cls is EnvRefBothProvidedError:
        assert cls(field="env").code == expected_code
    elif cls is EnvRefAmbiguousError:
        assert cls(name="env", candidate_ids=[]).code == expected_code
    elif cls is EnvNotFoundError:
        assert cls(name="env").code == expected_code


def test_version_conflict_carries_versions() -> None:
    err = VersionConflictError(expected=3, actual=5)
    assert err.expected == 3
    assert err.actual == 5
    assert "expected=3" in str(err) and "actual=5" in str(err)
    assert err.details == {"expected": 3, "actual": 5}


def test_invalid_transition_carries_states() -> None:
    err = InvalidTransitionError(src="retired", dst="active")
    assert err.src == "retired"
    assert err.dst == "active"
    assert "retired" in str(err) and "active" in str(err)


def test_embedding_mismatch_carries_ids() -> None:
    err = EmbeddingModelMismatchError(expected="env-model", actual="cfg-model")
    assert err.expected == "env-model"
    assert err.actual == "cfg-model"
    assert err.code == "EMBEDDING_MODEL_MISMATCH"


def test_all_errors_inherit_from_base() -> None:
    for cls in (
        UnauthorizedError,
        ForbiddenEnvError,
        EnvAmbiguousError,
        EnvRefBothProvidedError,
        EnvRefAmbiguousError,
        EnvNotFoundError,
        VersionConflictError,
        InvalidTransitionError,
        NotFoundError,
        EmbeddingModelMismatchError,
    ):
        assert issubclass(cls, MemoryMCPError)
