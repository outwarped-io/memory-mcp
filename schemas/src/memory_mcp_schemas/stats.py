"""Pydantic schemas for the v0.10 memory statistics surface."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TagCount(BaseModel):
    tag: str
    count: int


class EnvMemoryStats(BaseModel):
    name: str | None = None
    count: int = 0
    body_bytes: int | None = None


class MemoriesStats(BaseModel):
    total: int = 0
    active: int = 0
    superseded: int = 0
    retired: int = 0
    pinned: int = 0
    total_body_bytes: int | None = None
    total_body_bytes_approximate: bool = False
    by_env: dict[UUID, EnvMemoryStats] = Field(default_factory=dict)
    by_kind: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)
    top_tags: list[TagCount] = Field(default_factory=list)


class EnvStats(BaseModel):
    total: int = 0
    active: int = 0
    deleted: int = 0


class PercentileStats(BaseModel):
    p50: float | int | None = None
    p90: float | int | None = None
    p99: float | int | None = None
    max: float | int | None = None


class ChainDepthStats(PercentileStats):
    buckets: dict[str, int] = Field(default_factory=lambda: {"1": 0, "2": 0, "3": 0, "4+": 0})


class AgeStats(BaseModel):
    p50: float | int | None = None
    p90: float | int | None = None
    p99: float | int | None = None
    oldest: float | int | None = None


class BucketStats(BaseModel):
    buckets: dict[str, int] = Field(default_factory=dict)


class AccessCountStats(BucketStats):
    p50: float | int | None = None
    p90: float | int | None = None
    p99: float | int | None = None


class TagsPerMemoryStats(PercentileStats):
    untagged: int = 0


class DistributionStats(BaseModel):
    chain_depth: ChainDepthStats | None = None
    body_length: PercentileStats | None = None
    age_seconds: AgeStats | None = None
    salience: BucketStats | None = None
    access_count: AccessCountStats | None = None
    tags_per_memory: TagsPerMemoryStats | None = None


class SubstrateStats(BaseModel):
    postgres: dict[str, int | str | None] | None = None
    qdrant: dict[str, int | str | None] | None = None
    neo4j: dict[str, int | str | None] | None = None


class ProjectionLagEntry(BaseModel):
    sink: str
    env_id: UUID | None = None
    env_name: str | None = None
    lag_seconds: float | None = None
    last_event_id: int | None = None
    status: str | None = None


class OutboxStats(BaseModel):
    by_sink: dict[str, dict[str, int]] = Field(default_factory=dict)


class ProcessStats(BaseModel):
    rss_bytes: int | None = None
    rss_reason: str | None = None
    uptime_seconds: float | None = None


class MemStatsRequest(BaseModel):
    """Input schema for the read-only memory statistics snapshot."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    env_ids: list[UUID] | None = None
    env_names: list[str] | None = None
    global_: bool = Field(default=False, alias="global")
    include_substrates: bool = False
    include_body_bytes: bool = True
    include_distributions: bool = True
    tag_top_k: int = Field(default=20, ge=0, le=500)


class MemStatsResponse(BaseModel):
    memories: MemoriesStats = Field(default_factory=MemoriesStats)
    envs: EnvStats = Field(default_factory=EnvStats)
    distributions: DistributionStats | None = None
    tasks: dict[str, int] = Field(default_factory=dict)
    playbooks: dict[str, int] = Field(default_factory=dict)
    decisions: dict[str, object] = Field(default_factory=dict)
    substrate: SubstrateStats | None = None
    projection_lag: list[ProjectionLagEntry] = Field(default_factory=list)
    outbox: OutboxStats = Field(default_factory=OutboxStats)
    process: ProcessStats = Field(default_factory=ProcessStats)
    degraded_substrates: list[str] = Field(default_factory=list)
    degraded_sections: list[str] = Field(default_factory=list)
    schema_version: Literal[1] = 1
