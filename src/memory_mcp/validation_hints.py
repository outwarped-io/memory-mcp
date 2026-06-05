"""Did-you-mean hint engine for Pydantic ValidationError responses.

This module translates a raw :class:`pydantic.ValidationError` raised by
FastMCP's argument validator into a structured ``ValidationFailedError``
carrying an actionable hint. Hints fire on two error types:

* ``extra_forbidden`` — caller sent a field name that the model rejects.
  We look for a near-miss field on the relevant submodel.
* ``missing`` — caller omitted a required field. Less actionable for
  did-you-mean (no input to spell-check), so we surface the field name
  but no suggestion unless an allowlisted alias was sent as a sibling.

Safety rails (per plan §1.3):

* Allowlist of common aliases (``env`` → ``env_id``/``env_name``;
  ``req`` → ``request``; ``id`` → ``memory_id``/``entity_id`` for
  request-scoped contexts; etc.).
* Levenshtein distance ≤ 2 **or** SequenceMatcher ratio ≥ 0.7 — both
  matching the configured threshold yields a suggestion.
* Nested ``loc`` paths are honored — we descend into nested Pydantic
  models so candidate-field lists are scoped to the offending
  submodel, not the root.
* ``input_value`` from Pydantic's payload is **never** echoed — caller
  payloads may carry secrets.
* Distinct ``loc`` errors are processed independently; one error doesn't
  poison another.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError

# Common aliases the agent knows callers reach for. Keys are the
# offered (rejected) field; values are the canonical field names to
# look for on the offending submodel — first hit wins.
_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "env": ("env_id", "env_name", "env_ids", "env_names"),
    "envs": ("env_ids", "env_names", "env_id", "env_name"),
    "env_name": ("env_id", "env_names"),
    "env_id": ("env_name", "env_ids"),
    "env_names": ("env_ids", "env_name"),
    "env_ids": ("env_names", "env_id"),
    "req": ("request",),
    "request_": ("request",),
    "args": ("request",),
    "params": ("request",),
    "body": ("request",),
    "id": ("memory_id", "entity_id", "task_id"),
    "uuid": ("memory_id", "entity_id", "task_id"),
    "q": ("query", "title"),
    "text": ("query", "title", "body"),
    "limit": ("limit",),
}

# Floor for fuzzy match acceptance. Combine Levenshtein with sequence-ratio.
_LEVENSHTEIN_MAX = 2
_RATIO_FLOOR = 0.7


def _levenshtein(a: str, b: str) -> int:
    """Iterative Wagner–Fischer; small inputs keep this trivially fast."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if abs(len(a) - len(b)) > _LEVENSHTEIN_MAX + 4:
        return _LEVENSHTEIN_MAX + 5  # short-circuit hopeless pairs
    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _resolve_submodel(root_model: type[BaseModel], loc_path: tuple[Any, ...]) -> type[BaseModel] | None:
    """Walk ``loc_path`` (excluding the offending leaf) to the submodel.

    Returns ``None`` if any segment can't be resolved — we degrade
    silently rather than misattribute a suggestion to the wrong submodel.

    ``loc`` segments may be ints (list indices) or strings (field
    names). Ints are skipped; the model type itself doesn't change when
    descending into a list of submodels of the same type.
    """
    if not loc_path:
        return root_model
    current: type[BaseModel] | None = root_model
    for seg in loc_path:
        if current is None:
            return None
        if isinstance(seg, int):
            continue
        if not isinstance(seg, str):
            return None
        field = current.model_fields.get(seg)
        if field is None:
            return None
        annotation = field.annotation
        # Unwrap Optional[X], list[X], dict[K, X], etc.
        candidates = [annotation, *get_args(annotation)]
        next_model: type[BaseModel] | None = None
        for cand in candidates:
            if isinstance(cand, type) and issubclass(cand, BaseModel):
                next_model = cand
                break
            # Handle e.g. list[Foo] where get_origin(annotation) is list
            origin = get_origin(cand)
            if origin is not None:
                inner = get_args(cand)
                for i in inner:
                    if isinstance(i, type) and issubclass(i, BaseModel):
                        next_model = i
                        break
        current = next_model
    return current


def _allowed_fields(model_cls: type[BaseModel]) -> list[str]:
    return list(model_cls.model_fields.keys())


def _best_match(offered: str, candidates: list[str]) -> tuple[str | None, float]:
    """Return ``(best_field, confidence)`` or ``(None, 0.0)``."""
    best: str | None = None
    best_score = 0.0
    offered_lc = offered.lower()
    for c in candidates:
        c_lc = c.lower()
        dist = _levenshtein(offered_lc, c_lc)
        ratio = _ratio(offered_lc, c_lc)
        if dist <= _LEVENSHTEIN_MAX or ratio >= _RATIO_FLOOR:
            score = max(ratio, 1.0 - dist / max(len(offered), len(c), 1))
            if score > best_score:
                best = c
                best_score = score
    return best, best_score


def build_hints(root_model: type[BaseModel], err: ValidationError) -> list[dict[str, Any]]:
    """Return a list of structured hint dicts (possibly empty).

    Every entry has the shape::

        {
            "loc": ["request", "env_name"],
            "offered": "env",
            "suggested": "env_name",
            "confidence": 0.83,
            "source": "alias" | "fuzzy",
        }
    """
    hints: list[dict[str, Any]] = []
    for e in err.errors():
        etype = e.get("type")
        if etype not in ("extra_forbidden", "missing"):
            continue
        loc = tuple(e.get("loc", ()))
        if not loc:
            continue
        offered = str(loc[-1])
        parent_loc = loc[:-1]
        target = _resolve_submodel(root_model, parent_loc)
        if target is None:
            continue
        candidates = _allowed_fields(target)
        if etype == "missing" and offered in candidates:
            # The framework lists every missing required field — we have
            # nothing to suggest beyond "you must provide this". Skip.
            continue
        suggestion: str | None = None
        confidence: float = 0.0
        source: str = ""
        if offered in _ALIAS_MAP:
            for alias in _ALIAS_MAP[offered]:
                if alias in candidates:
                    suggestion = alias
                    confidence = 1.0
                    source = "alias"
                    break
        if suggestion is None and etype == "extra_forbidden":
            best, score = _best_match(offered, candidates)
            if best is not None:
                suggestion = best
                confidence = round(score, 3)
                source = "fuzzy"
        if suggestion:
            hints.append(
                {
                    "loc": [str(s) if isinstance(s, str) else s for s in loc],
                    "offered": offered,
                    "suggested": suggestion,
                    "confidence": confidence,
                    "source": source,
                }
            )
    return hints


def safe_error_payload(err: ValidationError) -> list[dict[str, Any]]:
    """Return ``ValidationError.errors()`` with sensitive fields stripped.

    Pydantic includes ``input`` (the offending value) by default —
    that may carry secrets, so we drop it. ``url`` is also pruned (it
    points at docs and adds noise on the wire).
    """
    safe: list[dict[str, Any]] = []
    for e in err.errors():
        clean = {
            "loc": list(e.get("loc", ())),
            "type": e.get("type"),
            "msg": e.get("msg"),
        }
        # Surface ctx for `extra_forbidden` (contains nothing sensitive) but
        # drop everything else by default.
        if e.get("type") == "extra_forbidden" and "ctx" in e:
            ctx = e["ctx"]
            if isinstance(ctx, dict):
                clean["ctx"] = {k: v for k, v in ctx.items() if k != "input_value"}
        safe.append(clean)
    return safe


def format_message(tool_name: str, errors_payload: list[dict[str, Any]], hints: list[dict[str, Any]]) -> str:
    """Compose a human-readable one-liner for the wire-format ``message``.

    Includes a parenthesised ``did you mean '<field>'?`` only when
    exactly one alias-class hint exists OR exactly one fuzzy hint above
    a high confidence threshold. Multiple hints stay in ``details``
    only — we don't want callers chasing a wrong guess.
    """
    err_count = len(errors_payload)
    first = errors_payload[0] if errors_payload else {}
    base_loc = ".".join(str(p) for p in first.get("loc", ()))
    msg = (
        f"VALIDATION_FAILED: {err_count} validation error(s) for {tool_name}"
        f" (first: {first.get('type', 'unknown')} at {base_loc!r})"
    )
    high_conf = [h for h in hints if h["source"] == "alias" or h["confidence"] >= 0.85]
    if len(high_conf) == 1:
        h = high_conf[0]
        msg += f" — did you mean {h['suggested']!r}?"
    return msg


__all__ = [
    "build_hints",
    "format_message",
    "safe_error_payload",
]
