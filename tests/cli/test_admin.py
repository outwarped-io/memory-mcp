from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from memory_mcp.cli import admin

runner = CliRunner()


def _patch_call(monkeypatch):
    calls = []

    def fake_call(endpoint, token, tool_name, request):
        calls.append(
            {
                "endpoint": endpoint,
                "token": token,
                "tool_name": tool_name,
                "request": request,
            }
        )
        return {"ok": True}

    monkeypatch.setattr(admin, "_call_tool_sync", fake_call)
    return calls


def test_admin_help_lists_all_subcommands() -> None:
    result = runner.invoke(admin.app, ["--help"])

    assert result.exit_code == 0
    assert "env" in result.output
    assert "mem" in result.output
    for command in [
        "export",
        "import",
        "diff",
        "clone",
        "merge",
        "migrate",
        "snapshot",
        "restore",
        "delete",
        "rename",
        "copy",
        "move",
    ]:
        assert command in result.output


def test_admin_env_help_lists_10_subcommands() -> None:
    result = runner.invoke(admin.app, ["env", "--help"])

    assert result.exit_code == 0
    for command in [
        "export",
        "import",
        "diff",
        "clone",
        "merge",
        "migrate",
        "snapshot",
        "restore",
        "delete",
        "rename",
    ]:
        assert command in result.output


def test_admin_mem_help_lists_2_subcommands() -> None:
    result = runner.invoke(admin.app, ["mem", "--help"])

    assert result.exit_code == 0
    assert "copy" in result.output
    assert "move" in result.output


def test_admin_env_delete_requires_confirm(monkeypatch) -> None:
    _patch_call(monkeypatch)
    env_id = str(uuid4())

    result = runner.invoke(admin.app, ["env", "delete", "--env-id", env_id], input="n\n")

    assert result.exit_code != 0
    assert "Confirmation did not match" in result.output


def test_admin_env_delete_with_confirm_calls_tool(monkeypatch) -> None:
    calls = _patch_call(monkeypatch)
    env_id = str(uuid4())

    result = runner.invoke(admin.app, ["env", "delete", "--env-id", env_id, "--confirm"])

    assert result.exit_code == 0
    assert calls[0]["tool_name"] == "env_delete_"
    assert calls[0]["request"].confirm_destroy is True
    assert str(calls[0]["request"].env_id) == env_id


def test_admin_env_restore_in_place_requires_confirm(monkeypatch) -> None:
    _patch_call(monkeypatch)
    snapshot_id = str(uuid4())

    result = runner.invoke(admin.app, ["env", "restore", "--snapshot-id", snapshot_id], input="no\n")

    assert result.exit_code != 0
    assert "Confirmation did not match" in result.output


def test_admin_env_export_calls_tool_with_correct_params(monkeypatch) -> None:
    calls = _patch_call(monkeypatch)
    env_id = str(uuid4())

    result = runner.invoke(
        admin.app,
        [
            "env",
            "export",
            "--env-id",
            env_id,
            "--target",
            "backup.tar.gz",
            "--format",
            "directory",
            "--no-embeddings",
            "--include-grants",
            "--include-dream-history",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["tool_name"] == "env_export_"
    request = calls[0]["request"]
    assert str(request.env_id) == env_id
    assert request.target_path == "backup.tar.gz"
    assert request.format == "directory"
    assert request.include_embeddings is False
    assert request.include_grants is True
    assert request.include_dream_history is True


def test_admin_reads_endpoint_from_env_var(monkeypatch) -> None:
    calls = _patch_call(monkeypatch)
    monkeypatch.setenv("MEMORY_MCP_ENDPOINT", "http://x/mcp")

    result = runner.invoke(
        admin.app,
        ["mem", "copy", "--memory-id", str(uuid4()), "--dst-env-id", str(uuid4())],
    )

    assert result.exit_code == 0
    assert calls[0]["endpoint"] == "http://x/mcp"


def test_admin_reads_token_from_config_file(monkeypatch) -> None:
    calls = _patch_call(monkeypatch)
    home = Path(".pytest-cli-admin-home").resolve()
    if home.exists():
        shutil.rmtree(home)
    config_dir = home / ".memory-mcp"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('token = "config-token"\nendpoint = "http://config/mcp"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("MEMORY_MCP_TOKEN", raising=False)
    monkeypatch.delenv("MEMORY_MCP_ENDPOINT", raising=False)

    try:
        result = runner.invoke(
            admin.app,
            ["mem", "copy", "--memory-id", str(uuid4()), "--dst-env-id", str(uuid4())],
        )
    finally:
        shutil.rmtree(home, ignore_errors=True)

    assert result.exit_code == 0
    assert calls[0]["token"] == "config-token"
    assert calls[0]["endpoint"] == "http://config/mcp"


def test_admin_mem_copy_passes_through_request(monkeypatch) -> None:
    calls = _patch_call(monkeypatch)
    memory_id = str(uuid4())
    dst_env_id = str(uuid4())

    result = runner.invoke(
        admin.app,
        [
            "mem",
            "copy",
            "--memory-id",
            memory_id,
            "--dst-env-id",
            dst_env_id,
            "--no-tags",
            "--no-provenance",
            "--no-lineage",
        ],
    )

    assert result.exit_code == 0
    request = calls[0]["request"]
    assert calls[0]["tool_name"] == "mem_copy_"
    assert str(request.memory_id) == memory_id
    assert str(request.dst_env_id) == dst_env_id
    assert request.copy_tags is False
    assert request.copy_provenance is False
    assert request.create_lineage_edge is False
    assert request.copy_lineage is False
