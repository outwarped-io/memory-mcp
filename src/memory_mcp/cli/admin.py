"""Admin CLI for memory-mcp environment operations."""

from __future__ import annotations

import asyncio
import json
import os
import pprint
import sys
import tomllib
from collections.abc import Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import typer
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel

from memory_mcp_schemas.env_ops import (
    EnvCloneRequest,
    EnvDeleteRequest,
    EnvDiffRequest,
    EnvExportRequest,
    EnvImportRequest,
    EnvMergeRequest,
    EnvMigrateRequest,
    EnvRenameRequest,
    EnvRestoreRequest,
    EnvSnapshotRequest,
    MemCopyRequest,
    MemMoveRequest,
)

DEFAULT_ENDPOINT = "http://localhost:8000/mcp"

TOOL_SUMMARY = (
    "Tools: env export, env import, env diff, env clone, env merge, env migrate, "
    "env snapshot, env restore, env delete, env rename, mem copy, mem move."
)

app = typer.Typer(
    name="memory-mcp-admin",
    help=f"Admin CLI for memory-mcp environment operations.\n\n{TOOL_SUMMARY}",
)
env_app = typer.Typer(help="Environment-level operations")
mem_app = typer.Typer(help="Per-memory operations")
app.add_typer(env_app, name="env")
app.add_typer(mem_app, name="mem")


def _config_path() -> Path:
    override = os.getenv("MEMORY_MCP_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".memory-mcp" / "config.toml"


def _load_config_file() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        return {}
    nested = data.get("memory_mcp") or data.get("memory-mcp") or {}
    if isinstance(nested, dict):
        return {**data, **nested}
    return data


def _resolve_endpoint(endpoint: str | None) -> str:
    if endpoint:
        return endpoint
    env_endpoint = os.getenv("MEMORY_MCP_ENDPOINT")
    if env_endpoint:
        return env_endpoint
    config_endpoint = _load_config_file().get("endpoint")
    if isinstance(config_endpoint, str) and config_endpoint:
        return config_endpoint
    return DEFAULT_ENDPOINT


def _resolve_token(token: str | None) -> str | None:
    if token:
        return token
    env_token = os.getenv("MEMORY_MCP_TOKEN")
    if env_token:
        return env_token
    config_token = _load_config_file().get("token")
    if isinstance(config_token, str) and config_token:
        return config_token
    return None


def _json_dict(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True)


def _request(model_cls: type[BaseModel], **values: Any) -> BaseModel:
    allowed = set(model_cls.model_fields)
    return model_cls(**{key: value for key, value in values.items() if key in allowed})


def _parse_json_option(value: str | None, option_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{option_name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"{option_name} must decode to a JSON object")
    return parsed


def _extract_call_result(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return result


async def _call_tool_async(endpoint: str, token: str | None, tool_name: str, request: BaseModel) -> Any:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with AsyncExitStack() as stack:
        read, write, _ = await stack.enter_async_context(streamablehttp_client(endpoint, headers=headers))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        result = await session.call_tool(tool_name, {"request": _json_dict(request)})
        if getattr(result, "isError", False):
            raise RuntimeError(_extract_call_result(result))
        return _extract_call_result(result)


def _call_tool_sync(endpoint: str, token: str | None, tool_name: str, request: BaseModel) -> Any:
    return asyncio.run(_call_tool_async(endpoint, token, tool_name, request))


def _format_output(data: Any, *, compact_json: bool = False, pretty: bool = False, diff_table: bool = False) -> None:
    if isinstance(data, BaseModel):
        data = _json_dict(data)
    if pretty:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            if diff_table and isinstance(data, Mapping):
                counts = data.get("counts") or {}
                if isinstance(counts, Mapping):
                    table = Table(title="Environment diff counts")
                    table.add_column("Item")
                    table.add_column("A", justify="right")
                    table.add_column("B", justify="right")
                    for key, value in counts.items():
                        if isinstance(value, Mapping):
                            table.add_row(str(key), str(value.get("a", "")), str(value.get("b", "")))
                    console.print(table)
                    return
            console.print_json(json.dumps(data, default=str))
            return
        except Exception:
            pprint.pp(data)
            return
    indent = None if compact_json else 2
    typer.echo(json.dumps(data, indent=indent, sort_keys=True, default=str))


def _require_confirmation(label: str, confirm: bool, action: str) -> None:
    if confirm:
        return
    try:
        typed = typer.prompt(f"{action}: type '{label}' to confirm", default="", show_default=False)
    except (typer.Abort, EOFError) as exc:
        raise typer.Exit(1) from exc
    if typed != label:
        typer.echo("Confirmation did not match; aborting.", err=True)
        raise typer.Exit(1)


def _run_tool_command(
    *,
    endpoint: str | None,
    token: str | None,
    tool_name: str,
    request: BaseModel,
    compact_json: bool = False,
    pretty: bool = False,
    diff_table: bool = False,
) -> None:
    resolved_endpoint = _resolve_endpoint(endpoint)
    resolved_token = _resolve_token(token)
    try:
        result = _call_tool_sync(resolved_endpoint, resolved_token, tool_name, request)
    except Exception as exc:
        typer.echo(f"memory-mcp-admin: {exc}", err=True)
        raise typer.Exit(1) from exc
    _format_output(result, compact_json=compact_json, pretty=pretty, diff_table=diff_table)


@env_app.command("export")
def env_export_cmd(
    env_id: str = typer.Option(..., "--env-id"),
    target: Path = typer.Option(..., "--target"),
    output_format: str = typer.Option("archive", "--format"),
    no_embeddings: bool = typer.Option(False, "--no-embeddings"),
    include_grants: bool = typer.Option(False, "--include-grants"),
    include_dream_history: bool = typer.Option(False, "--include-dream-history"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="MEMORY_MCP_ENDPOINT"),
    token: str | None = typer.Option(None, "--token", envvar="MEMORY_MCP_TOKEN"),
    json_output: bool = typer.Option(False, "--json"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Export an env to a directory or tar.gz archive."""
    request = _request(
        EnvExportRequest,
        env_id=env_id,
        target_path=str(target),
        format=output_format,
        include_embeddings=not no_embeddings,
        include_grants=include_grants,
        include_dream_history=include_dream_history,
    )
    _run_tool_command(
        endpoint=endpoint,
        token=token,
        tool_name="env_export_",
        request=request,
        compact_json=json_output,
        pretty=pretty,
    )


@env_app.command("import")
def env_import_cmd(
    source: Path = typer.Option(..., "--source"),
    target_env_name: str = typer.Option(..., "--target-env-name"),
    mode: str = typer.Option("fail", "--mode"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
    confirm: bool = typer.Option(False, "--confirm"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="MEMORY_MCP_ENDPOINT"),
    token: str | None = typer.Option(None, "--token", envvar="MEMORY_MCP_TOKEN"),
    json_output: bool = typer.Option(False, "--json"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Import an env archive into a target environment."""
    if mode == "overwrite":
        _require_confirmation(target_env_name, confirm, "Overwrite import")
    request = _request(
        EnvImportRequest,
        source_path=str(source),
        target_env_name=target_env_name,
        mode=mode,
        dry_run=dry_run,
    )
    _run_tool_command(
        endpoint=endpoint,
        token=token,
        tool_name="env_import_",
        request=request,
        compact_json=json_output,
        pretty=pretty,
    )


@env_app.command("diff")
def env_diff_cmd(
    env_a: str = typer.Option(..., "--env-a"),
    env_b: str = typer.Option(..., "--env-b"),
    granularity: str = typer.Option("counts", "--granularity"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="MEMORY_MCP_ENDPOINT"),
    token: str | None = typer.Option(None, "--token", envvar="MEMORY_MCP_TOKEN"),
    json_output: bool = typer.Option(False, "--json"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Diff two environments."""
    request = _request(EnvDiffRequest, env_a_id=env_a, env_b_id=env_b, granularity=granularity)
    _run_tool_command(
        endpoint=endpoint,
        token=token,
        tool_name="env_diff_",
        request=request,
        compact_json=json_output,
        pretty=pretty,
        diff_table=True,
    )


@env_app.command("clone")
def env_clone_cmd(
    src: str = typer.Option(..., "--src"),
    new_name: str = typer.Option(..., "--new-name"),
    lineage_depth: int = typer.Option(1, "--lineage-depth"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="MEMORY_MCP_ENDPOINT"),
    token: str | None = typer.Option(None, "--token", envvar="MEMORY_MCP_TOKEN"),
    json_output: bool = typer.Option(False, "--json"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Clone an environment."""
    request = _request(EnvCloneRequest, src_env_id=src, new_name=new_name, lineage_depth=lineage_depth)
    _run_tool_command(
        endpoint=endpoint,
        token=token,
        tool_name="env_clone_",
        request=request,
        compact_json=json_output,
        pretty=pretty,
    )


@env_app.command("merge")
def env_merge_cmd(
    src: str = typer.Option(..., "--src"),
    dst: str = typer.Option(..., "--dst"),
    allow_external_ref_rewrite: bool = typer.Option(False, "--allow-external-ref-rewrite"),
    allow_embedding_mismatch: bool = typer.Option(False, "--allow-embedding-mismatch"),
    keep_src: bool = typer.Option(False, "--keep-src"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="MEMORY_MCP_ENDPOINT"),
    token: str | None = typer.Option(None, "--token", envvar="MEMORY_MCP_TOKEN"),
    json_output: bool = typer.Option(False, "--json"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Merge src environment into dst."""
    request = _request(
        EnvMergeRequest,
        src_env_id=src,
        dst_env_id=dst,
        allow_external_ref_rewrite=allow_external_ref_rewrite,
        allow_embedding_mismatch=allow_embedding_mismatch,
        delete_src_after=not keep_src,
    )
    _run_tool_command(
        endpoint=endpoint,
        token=token,
        tool_name="env_merge_",
        request=request,
        compact_json=json_output,
        pretty=pretty,
    )


@env_app.command("migrate")
def env_migrate_cmd(
    src: str = typer.Option(..., "--src"),
    dst: str = typer.Option(..., "--dst"),
    mode: str = typer.Option("copy", "--mode"),
    filter_json: str | None = typer.Option(None, "--filter"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="MEMORY_MCP_ENDPOINT"),
    token: str | None = typer.Option(None, "--token", envvar="MEMORY_MCP_TOKEN"),
    json_output: bool = typer.Option(False, "--json"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Bulk copy or move memories between environments."""
    request = _request(
        EnvMigrateRequest,
        src_env_id=src,
        dst_env_id=dst,
        mode=mode,
        filter=_parse_json_option(filter_json, "--filter"),
    )
    _run_tool_command(
        endpoint=endpoint,
        token=token,
        tool_name="env_migrate_",
        request=request,
        compact_json=json_output,
        pretty=pretty,
    )


@env_app.command("snapshot")
def env_snapshot_cmd(
    env_id: str = typer.Option(..., "--env-id"),
    label: str = typer.Option(..., "--label"),
    notes: str | None = typer.Option(None, "--notes"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="MEMORY_MCP_ENDPOINT"),
    token: str | None = typer.Option(None, "--token", envvar="MEMORY_MCP_TOKEN"),
    json_output: bool = typer.Option(False, "--json"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Create an environment snapshot."""
    request = _request(EnvSnapshotRequest, env_id=env_id, label=label, notes=notes)
    _run_tool_command(
        endpoint=endpoint,
        token=token,
        tool_name="env_snapshot_",
        request=request,
        compact_json=json_output,
        pretty=pretty,
    )


@env_app.command("restore")
def env_restore_cmd(
    snapshot_id: str = typer.Option(..., "--snapshot-id"),
    mode: str = typer.Option("replace_env_in_place", "--mode"),
    new_env_name: str | None = typer.Option(None, "--new-env-name"),
    confirm: bool = typer.Option(False, "--confirm"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="MEMORY_MCP_ENDPOINT"),
    token: str | None = typer.Option(None, "--token", envvar="MEMORY_MCP_TOKEN"),
    json_output: bool = typer.Option(False, "--json"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Restore from an environment snapshot."""
    if mode == "replace_env_in_place":
        _require_confirmation(snapshot_id, confirm, "Restore in place")
    request = _request(
        EnvRestoreRequest,
        snapshot_id=snapshot_id,
        mode=mode,
        new_env_name=new_env_name,
        confirm_destroy=confirm,
    )
    _run_tool_command(
        endpoint=endpoint,
        token=token,
        tool_name="env_restore_",
        request=request,
        compact_json=json_output,
        pretty=pretty,
    )


@env_app.command("delete")
def env_delete_cmd(
    env_id: str = typer.Option(..., "--env-id"),
    cascade_external_refs: bool = typer.Option(False, "--cascade-external-refs"),
    confirm: bool = typer.Option(False, "--confirm"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="MEMORY_MCP_ENDPOINT"),
    token: str | None = typer.Option(None, "--token", envvar="MEMORY_MCP_TOKEN"),
    json_output: bool = typer.Option(False, "--json"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Delete an environment after confirmation."""
    _require_confirmation(env_id, confirm, "Delete env")
    request = _request(
        EnvDeleteRequest,
        env_id=env_id,
        cascade_external_refs=cascade_external_refs,
        confirm_destroy=confirm,
    )
    _run_tool_command(
        endpoint=endpoint,
        token=token,
        tool_name="env_delete_",
        request=request,
        compact_json=json_output,
        pretty=pretty,
    )


@env_app.command("rename")
def env_rename_cmd(
    env_id: str = typer.Option(..., "--env-id"),
    new_name: str | None = typer.Option(None, "--new-name"),
    embedding_model: str | None = typer.Option(None, "--embedding-model"),
    retention_policy: str | None = typer.Option(None, "--retention-policy"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="MEMORY_MCP_ENDPOINT"),
    token: str | None = typer.Option(None, "--token", envvar="MEMORY_MCP_TOKEN"),
    json_output: bool = typer.Option(False, "--json"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Rename or update mutable environment metadata."""
    request = _request(
        EnvRenameRequest,
        env_id=env_id,
        new_name=new_name,
        new_default_embedding_model_id=embedding_model,
        new_retention_policy=_parse_json_option(retention_policy, "--retention-policy"),
    )
    _run_tool_command(
        endpoint=endpoint,
        token=token,
        tool_name="env_rename_",
        request=request,
        compact_json=json_output,
        pretty=pretty,
    )


@mem_app.command("copy")
def mem_copy_cmd(
    memory_id: str = typer.Option(..., "--memory-id"),
    dst_env_id: str = typer.Option(..., "--dst-env-id"),
    no_tags: bool = typer.Option(False, "--no-tags"),
    no_provenance: bool = typer.Option(False, "--no-provenance"),
    no_lineage: bool = typer.Option(False, "--no-lineage"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="MEMORY_MCP_ENDPOINT"),
    token: str | None = typer.Option(None, "--token", envvar="MEMORY_MCP_TOKEN"),
    json_output: bool = typer.Option(False, "--json"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Copy a memory to another environment."""
    request = _request(
        MemCopyRequest,
        memory_id=memory_id,
        dst_env_id=dst_env_id,
        copy_tags=not no_tags,
        copy_provenance=not no_provenance,
        create_lineage_edge=not no_lineage,
        copy_lineage=not no_lineage,
    )
    _run_tool_command(
        endpoint=endpoint,
        token=token,
        tool_name="mem_copy_",
        request=request,
        compact_json=json_output,
        pretty=pretty,
    )


@mem_app.command("move")
def mem_move_cmd(
    memory_id: str = typer.Option(..., "--memory-id"),
    dst_env_id: str = typer.Option(..., "--dst-env-id"),
    no_tags: bool = typer.Option(False, "--no-tags"),
    no_provenance: bool = typer.Option(False, "--no-provenance"),
    no_lineage: bool = typer.Option(False, "--no-lineage"),
    hard_delete: bool = typer.Option(False, "--hard-delete"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="MEMORY_MCP_ENDPOINT"),
    token: str | None = typer.Option(None, "--token", envvar="MEMORY_MCP_TOKEN"),
    json_output: bool = typer.Option(False, "--json"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Move a memory to another environment."""
    request = _request(
        MemMoveRequest,
        memory_id=memory_id,
        dst_env_id=dst_env_id,
        copy_tags=not no_tags,
        copy_provenance=not no_provenance,
        create_lineage_edge=not no_lineage,
        copy_lineage=not no_lineage,
        redirect_source=not hard_delete,
    )
    _run_tool_command(
        endpoint=endpoint,
        token=token,
        tool_name="mem_move_",
        request=request,
        compact_json=json_output,
        pretty=pretty,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
