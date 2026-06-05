"""Validate the Grafana dashboard metric references.

Slice B ships before/alongside Slice A. Once Slice A has landed, the authoritative
metric inventory comes from ``memory_mcp.observability.metrics_registry()``. Until
then, this test keeps the v0.10 contract's expected metric names as a fallback so
Dashboard-only work can validate without depending on unmerged gauge code.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

try:
    from memory_mcp.observability import metrics_registry

    OBSERVABILITY_IMPORTABLE = True
except Exception:  # pragma: no cover - only for concurrent Slice A import churn
    metrics_registry = None  # type: ignore[assignment]
    OBSERVABILITY_IMPORTABLE = False

pytestmark = pytest.mark.skipif(
    not OBSERVABILITY_IMPORTABLE,
    reason="memory_mcp.observability is not importable while Slice A is in flight",
)

DASHBOARD_PATH = Path(__file__).resolve().parents[2] / "dashboards" / "memory-mcp.json"
METRIC_RE = re.compile(r"\b(?:mcp|process)_[a-zA-Z_:][a-zA-Z0-9_:]*\b")
EXPECTED_V010_METRICS = {
    "mcp_tool_calls_total",
    "mcp_tool_latency_seconds_bucket",
    "mcp_projection_lag_seconds",
    "mcp_outbox_pending_total",
    "mcp_outbox_dead_total",
    "mcp_dream_run_duration_seconds_bucket",
    "mcp_dream_llm_fallbacks_total",
    "mcp_memories_total",
    "mcp_memories_pinned_total",
    "mcp_memories_body_bytes_total",
    "mcp_memory_chain_depth_bucket",
    "mcp_memory_age_seconds_bucket",
    "mcp_memory_body_length_bytes_bucket",
    "mcp_memory_salience_bucket",
    "mcp_memory_access_count_bucket",
    "mcp_tasks_total",
    "mcp_playbooks_total",
    "mcp_decisions_total",
    "process_resident_memory_bytes",
}


def _registry_metric_names() -> set[str]:
    if metrics_registry is None:
        return set()
    names: set[str] = set()
    for family in metrics_registry().collect():
        names.add(family.name)
        names.update(sample.name for sample in family.samples)
    return names


def _iter_exprs(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        expr = value.get("expr")
        if isinstance(expr, str):
            yield expr
        for child in value.values():
            yield from _iter_exprs(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_exprs(item)


def test_dashboard_json_parses() -> None:
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))

    assert dashboard["uid"] == "memory-mcp-overview"
    assert dashboard["title"] == "Memory MCP — Overview"
    assert dashboard["schemaVersion"] >= 38
    assert dashboard["tags"] == ["memory-mcp", "mcp"]
    assert sum(1 for panel in dashboard["panels"] if panel.get("type") != "row") == 13


def _known_metric_names() -> set[str]:
    registry_names = _registry_metric_names()
    if registry_names >= EXPECTED_V010_METRICS:
        return registry_names
    return EXPECTED_V010_METRICS


def test_dashboard_references_known_metrics() -> None:
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    referenced = {metric for expr in _iter_exprs(dashboard) for metric in METRIC_RE.findall(expr)}
    known = _known_metric_names()

    assert referenced
    assert referenced <= known
