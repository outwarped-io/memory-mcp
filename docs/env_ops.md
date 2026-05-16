# Environment Operations (v0.8)

memory-mcp v0.8 adds environment-as-a-unit operations across MCP tools, the `memory-mcp-admin` CLI, and the Python SDK. Canonical request/response models live in `memory_mcp_schemas.env_ops`.

## Shared semantics

- Archive `schema_version` is `0.8.0`; imports reject unsafe future versions unless explicitly forced.
- Embeddings are portable `embeddings/memory_vectors.jsonl` rows (`MemoryVectorRecord`) keyed by `(memory_id, vector_name)` with `memory_version` staleness detection.
- Cross-env operations allocate fresh destination UUIDs. Restore-in-place is the exception: it preserves env and memory UUIDs from the snapshot.
- Lineage may cross envs; relations are strictly env-local.
- Environments are soft-deleted (`status='deleted'`, `deleted_at`); env-scoped child rows are hard-deleted.
- Conservative defaults: imports are dry-run, grants and dream history are excluded, and bulk re-embed (>10k memories) is blocked.

## Archive layout

```text
manifest.json, checksums.sha256, env.json, memories.jsonl, tags.jsonl,
memory_tags.jsonl, entities.jsonl, entity_aliases.jsonl, relations.jsonl,
memory_sources.jsonl, memory_lineage.jsonl, graph_nodes.jsonl, tasks.jsonl,
embeddings/memory_vectors.jsonl
```

Optional files: `grants.jsonl` (`include_grants=true`), `dream_runs.jsonl` and `dream_proposals.jsonl` (`include_dream_history=true`). Decisions and playbooks are memory rows, not separate files.

## CLI authentication

`memory-mcp-admin` connects over Streamable HTTP and respects server RBAC. Token resolution order: `--token`, `MEMORY_MCP_TOKEN`, then `~/.memory-mcp/config.toml` (`token = "..."`, `endpoint = "..."`).

## env_export

Stream an environment to a portable archive or inspectable directory.

### Request
| Field | Type | Default | Description |
|---|---|---|---|
| `env_id` | UUID | required | Source environment. |
| `format` | `archive` \| `directory` | required | Output container type. |
| `target_path` | str | required | Destination path; archive mode emits `.tar.gz`. |
| `include_embeddings` | bool | `True` | Include vector records. |
| `include_provenance` | bool | `True` | Include `MemorySource` rows. |
| `include_grants` | bool | `False` | Include security-sensitive grants. |
| `include_dream_history` | bool | `False` | Include dream run/proposal rows. |
| `chunk_size` | int | `5000` | Streaming read chunk size. |

### Behavior
Exports env-scoped rows, manifest, checksums, and optional embeddings under a repeatable read snapshot. Global agent FKs are nulled or re-attributed on import.

### Errors
`ENV_DELETED`, `EXPORT_TARGET_NOT_EMPTY`, `NOT_FOUND`.

### CLI
```bash
memory-mcp-admin env export --env-id <uuid> --target /backup/prod --format archive
```

### SDK
```python
await client.env_ops.export(EnvExportRequest(env_id=env_id, target_path="/backup/prod", format="archive"))
```

## env_import

Load an archive or directory into a new or existing environment.

### Request
| Field | Type | Default | Description |
|---|---|---|---|
| `source_path` | str | required | Archive or directory to read. |
| `target_env_name` | str \| None | `None` | Name for a new env; exclusive with `target_env_id`. |
| `target_env_id` | UUID \| None | `None` | Existing destination env. |
| `mode` | `fail` \| `skip` \| `overwrite` \| `merge` | `fail` | Conflict policy. |
| `dry_run` | bool | `True` | Report planned changes only. |
| `re_embed_if_model_mismatch` | bool | `True` | Re-embed if models differ. |
| `allow_bulk_reembed` | bool | `False` | Permit >10k-memory re-embed. |

### Behavior
Validates checksums/version, allocates fresh UUIDs through `RemapTable`, inserts rows in FK-safe order, and queues vector rebuild after commit. Dry-run returns counts and conflicts without writes.

### Errors
`IMPORT_ARCHIVE_VERSION`, `IMPORT_BULK_REEMBED_BLOCKED`, `EMBEDDING_MODEL_MISMATCH`, `NOT_FOUND`.

### CLI
```bash
memory-mcp-admin env import --source /backup/prod.memarchive.tar.gz --target-env-name staging --no-dry-run
```

### SDK
```python
await client.env_ops.import_(EnvImportRequest(source_path="/backup/prod.memarchive.tar.gz", target_env_name="staging", dry_run=False))
```

## env_diff

Compare two environments at one of four granularities.

### Request
| Field | Type | Default | Description |
|---|---|---|---|
| `env_a_id` | UUID | required | Left-side env. |
| `env_b_id` | UUID | required | Right-side env. |
| `granularity` | `counts` \| `entity_keys` \| `memory_hashes` \| `full` | `counts` | Detail level. |

### Behavior
`counts` compares per-table totals; `entity_keys` adds canonical key sets; `memory_hashes` adds stable memory content hashes; `full` adds bounded tag/relation/task/graph/lineage samples.

### Errors
`ENV_DELETED`, `NOT_FOUND`.

### CLI
```bash
memory-mcp-admin env diff --env-a <uuid> --env-b <uuid> --granularity full --pretty
```

### SDK
```python
await client.env_ops.diff(EnvDiffRequest(env_a_id=a, env_b_id=b, granularity="memory_hashes"))
```

## env_clone

Duplicate an environment into a fresh environment.

### Request
| Field | Type | Default | Description |
|---|---|---|---|
| `src_env_id` | UUID | required | Source env. |
| `new_name` | str | required | Destination env name. |
| `include_embeddings` | bool | `True` | Copy embeddings when possible. |
| `filter` | `MemBrowseRequest` \| None | `None` | Optional seed-memory filter. |
| `lineage_depth` | int | `1` | Lineage-parent closure depth, max 5. |
| `include_referenced_entities` | bool | `True` | Include referenced entities. |

### Behavior
Creates a new env with fresh UUIDs. Filtered clone expands closure to include supersession targets, lineage parents, referenced entities, and tags, reported in `closure_inclusions`.

### Errors
`ENV_DELETED`, `NOT_FOUND`, `EMBEDDING_MODEL_MISMATCH`.

### CLI
```bash
memory-mcp-admin env clone --src-env-id <uuid> --new-name sandbox
```

### SDK
```python
await client.env_ops.clone(EnvCloneRequest(src_env_id=src, new_name="sandbox"))
```

## env_merge

Merge a source environment into a destination environment.

### Request
| Field | Type | Default | Description |
|---|---|---|---|
| `src_env_id` | UUID | required | Source env. |
| `dst_env_id` | UUID | required | Destination env. |
| `entity_strategy` | `by_canonical_key` \| `by_id` \| `keep_both` | `by_canonical_key` | Entity conflict policy. |
| `tag_strategy` | `union` \| `src_wins` \| `dst_wins` | `union` | Tag conflict policy. |
| `dry_run` | bool | `False` | Report without writing. |
| `delete_src_after` | bool | `True` | Soft-delete source after merge. |
| `allow_embedding_mismatch` | bool | `False` | Permit model mismatch. |
| `allow_external_ref_rewrite` | bool | `False` | Rewrite lineage touching source. |

### Behavior
Copies source rows into destination with fresh UUIDs, unions tags, invokes `entity_merge` on configured collisions, rewrites only env-local relations, and can soft-delete source after lineage rewrite.

### Errors
`ENV_DELETED`, `EXTERNAL_REFS_BLOCKING`, `EMBEDDING_MODEL_MISMATCH`, `NOT_FOUND`.

### CLI
```bash
memory-mcp-admin env merge --src-env-id <uuid> --dst-env-id <uuid> --allow-external-ref-rewrite
```

### SDK
```python
await client.env_ops.merge(EnvMergeRequest(src_env_id=src, dst_env_id=dst, allow_external_ref_rewrite=True))
```

## env_migrate

Bulk-copy or bulk-move selected memories between environments.

### Request
| Field | Type | Default | Description |
|---|---|---|---|
| `src_env_id` | UUID | required | Source env. |
| `dst_env_id` | UUID | required | Destination env. |
| `filter` | `MemBrowseRequest` \| None | `None` | Memory selection filter. |
| `mode` | `copy` \| `move` | `copy` | Per-memory operation. |
| `copy_tags` / `copy_provenance` | bool | `True` | Copy metadata with each memory. |
| `create_lineage_edges` | bool | `True` | Create migration lineage edges. |
| `preserve_timestamps` | bool | `False` | Preserve source timestamps. |
| `re_embed_if_model_mismatch` | bool | `False` | Re-embed into destination model. |
| `preserve_supersession_chain` | bool | `True` | Migrate full supersession chains. |
| `include_superseded` | bool | `False` | Include already-superseded memories. |
| `fail_fast` | bool | `False` | Stop at first failure. |
| `dry_run` | bool | `False` | Preview the batch. |

### Behavior
Enumerates matching memories and calls `mem_copy` or `mem_move` for each. Best-effort by default: partial successes are retained and failures are reported with IDs/codes.

### Errors
`ENV_DELETED`, `EMBEDDING_MODEL_MISMATCH`, `NOT_FOUND`.

### CLI
```bash
memory-mcp-admin env migrate --src-env-id <uuid> --dst-env-id <uuid> --mode move --no-dry-run
```

### SDK
```python
await client.env_ops.migrate(EnvMigrateRequest(src_env_id=src, dst_env_id=dst, mode="move"))
```

## env_snapshot

Persist a labeled local archive for an environment.

### Request
| Field | Type | Default | Description |
|---|---|---|---|
| `env_id` | UUID | required | Source env. |
| `label` | str | required | Snapshot label. |
| `include_embeddings` | bool | `True` | Include vectors. |

### Behavior
Writes `<data_root>/snapshots/<env_id>/<snapshot_id>.memarchive.tar.gz` and a `snapshots` row with path, size, and checksum. Storage is not auto-pruned; server warns above 10 GB.

### Errors
`ENV_DELETED`, `NOT_FOUND`.

### CLI
```bash
memory-mcp-admin env snapshot --env-id <uuid> --label before-experiment
```

### SDK
```python
await client.env_ops.snapshot(EnvSnapshotRequest(env_id=env_id, label="before-experiment"))
```

## env_restore

Restore from a snapshot in place or into a new environment.

### Request
| Field | Type | Default | Description |
|---|---|---|---|
| `snapshot_id` | UUID | required | Snapshot row. |
| `mode` | `replace_env_in_place` \| `restore_to_new_env` | required | Restore behavior. |
| `confirm_destroy` | bool | `False` | Required for in-place restore. |
| `new_env_name` | str \| None | `None` | Required for `restore_to_new_env`. |

### Behavior
In-place restore validates first, then deletes and reloads env rows in one PG transaction while preserving UUIDs. Restore-to-new-env imports with normal UUID remapping.

### Errors
`CONFIRM_DESTROY_REQUIRED`, `ENV_DELETED`, `NOT_FOUND`, `IMPORT_ARCHIVE_VERSION`.

### CLI
```bash
memory-mcp-admin env restore --snapshot-id <uuid> --mode replace_env_in_place --confirm
```

### SDK
```python
await client.env_ops.restore(EnvRestoreRequest(snapshot_id=snapshot_id, mode="replace_env_in_place", confirm_destroy=True))
```

## env_delete

Soft-delete an environment and remove its env-scoped rows.

### Request
| Field | Type | Default | Description |
|---|---|---|---|
| `env_id` | UUID | required | Environment to delete. |
| `confirm_destroy` | bool | required | Must be true. |
| `cascade_external_refs` | bool | `False` | Drop cross-env lineage edges pointing in. |

### Behavior
Deletes env-local rows in dependency order, then marks the environment row deleted. With `cascade_external_refs=False`, inbound lineage from other envs blocks deletion.

### Errors
`CONFIRM_DESTROY_REQUIRED`, `EXTERNAL_REFS_BLOCKING`, `NOT_FOUND`.

### CLI
```bash
memory-mcp-admin env delete --env-id <uuid> --confirm
```

### SDK
```python
await client.env_ops.delete(EnvDeleteRequest(env_id=env_id, confirm_destroy=True))
```

## env_rename

Update mutable environment metadata.

### Request
| Field | Type | Default | Description |
|---|---|---|---|
| `env_id` | UUID | required | Environment to update. |
| `new_name` | str \| None | `None` | New unique name. |
| `new_default_embedding_model_id` | str \| None | `None` | Model for future memories only. |
| `new_retention_policy` | dict \| None | `None` | Replacement retention policy. |

### Behavior
Changes only supplied fields. Model changes do not re-embed existing memories and return a warning.

### Errors
`ENV_DELETED`, `NOT_FOUND`, `INVALID_INPUT`.

### CLI
```bash
memory-mcp-admin env rename --env-id <uuid> --new-name prod-renamed
```

### SDK
```python
await client.env_ops.rename(EnvRenameRequest(env_id=env_id, new_name="prod-renamed"))
```

## mem_copy

Copy one memory across environments.

### Request
| Field | Type | Default | Description |
|---|---|---|---|
| `memory_id` | UUID | required | Source memory. |
| `dst_env_id` | UUID | required | Destination env. |
| `copy_tags` / `copy_provenance` | bool | `True` | Copy tags/provenance. |
| `create_lineage_edge` | bool | `True` | Create migrated-from lineage. |
| `preserve_timestamps` | bool | `False` | Preserve source timestamps. |
| `re_embed_if_model_mismatch` | bool | `False` | Re-embed for destination model. |
| `copy_lineage` | bool | `False` | Copy selected lineage context. |
| `copy_entities` | `never` \| `if_present_in_dst` \| `always_create` | `if_present_in_dst` | Entity handling policy. |

### Behavior
Creates a destination memory with a fresh UUID and optional tags, provenance, entity references, and lineage. Source memory is unchanged.

### Errors
`ENV_DELETED`, `EMBEDDING_MODEL_MISMATCH`, `NOT_FOUND`.

### CLI
```bash
memory-mcp-admin mem copy --memory-id <uuid> --dst-env-id <uuid>
```

### SDK
```python
await client.memories.copy(MemCopyRequest(memory_id=memory_id, dst_env_id=dst))
```

## mem_move

Move one memory across environments.

### Request
| Field | Type | Default | Description |
|---|---|---|---|
| `memory_id` | UUID | required | Source memory. |
| `dst_env_id` | UUID | required | Destination env. |
| `redirect_source` | bool | `True` | Supersede source instead of deleting. |
| `copy_tags` / `copy_provenance` | bool | `True` | Copy tags/provenance. |
| `create_lineage_edge` | bool | `True` | Create migrated-from lineage. |
| `preserve_timestamps` | bool | `False` | Preserve source timestamps. |
| `re_embed_if_model_mismatch` | bool | `False` | Re-embed for destination model. |
| `copy_lineage` | bool | `False` | Copy selected lineage context. |

### Behavior
Runs `mem_copy`, then marks source `superseded` with `superseded_by` pointing at destination, or deletes it when `redirect_source=False` and no rows block deletion. This is the only allowed cross-env supersession path.

### Errors
`ENV_DELETED`, `EMBEDDING_MODEL_MISMATCH`, `NOT_FOUND`, `CROSS_ENV_SUPERSEDE_BLOCKED` for non-`mem_move` callers.

### CLI
```bash
memory-mcp-admin mem move --memory-id <uuid> --dst-env-id <uuid>
```

### SDK
```python
await client.memories.move(MemMoveRequest(memory_id=memory_id, dst_env_id=dst))
```
