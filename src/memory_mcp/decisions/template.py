"""Markdown rendering for ADR-lite decision memories."""

from __future__ import annotations

from typing import Any

from memory_mcp.decisions.models import DecisionMeta


def _memory_title(memory: Any) -> str:
    return str(memory.title or f"Decision {memory.id}")


def render_adr(memory: Any, meta: DecisionMeta | None) -> str:
    """Render a memory as strict ADR-style markdown."""
    title = _memory_title(memory)
    status = meta.status.value if meta else "(unset)"
    rationale = meta.rationale if meta else "_(no rationale captured)_"
    if meta and meta.constraints:
        constraints = "\n".join(f"- {item}" for item in meta.constraints)
    else:
        constraints = "_(none captured)_"
    if meta and meta.consequences:
        consequences = "\n".join(f"- {item}" for item in meta.consequences)
    else:
        consequences = "_(none recorded)_"
    superseded_by = str(meta.superseded_by) if meta and meta.superseded_by else "_(none)_"

    parts: list[str] = []
    if meta is None:
        parts.append(
            "> **Note:** This decision has no structured metadata. Edit via mem_update to add decision_meta.\n"
        )
    parts.append(
        f"# {title}\n\n"
        f"**Status:** {status}\n\n"
        "## Context\n\n"
        f"{memory.body}\n\n"
        "## Decision\n\n"
        f"{rationale}\n\n"
        "## Consequences\n\n"
        f"{consequences}\n\n"
        "## Constraints\n\n"
        f"{constraints}\n\n"
        "## Superseded By\n\n"
        f"{superseded_by}"
    )
    return "".join(parts)
