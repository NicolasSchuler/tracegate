"""RCA-02 / D11: MCP tools confine ``out_dir`` to ``$LLMDEBUG_MCP_ROOT``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from llmdebug.mcp_paths import MCP_ROOT_ENV, mcp_root, resolve_mcp_out_dir

mcp_server = pytest.importorskip("llmdebug.mcp_server")

_TOOL_NAMES = (
    "llmdebug_diagnose",
    "llmdebug_show",
    "llmdebug_list",
    "llmdebug_frame",
    "llmdebug_git_context",
    "llmdebug_diff",
    "llmdebug_rca_status",
    "llmdebug_rca_history",
    "llmdebug_rca_advance",
)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv(MCP_ROOT_ENV, str(root))
    return root.resolve()


def test_relative_and_absolute_paths_inside_the_root(root: Path) -> None:
    assert resolve_mcp_out_dir(".llmdebug") == str(root / ".llmdebug")
    assert resolve_mcp_out_dir(str(root / "a" / "b")) == str(root / "a" / "b")
    assert resolve_mcp_out_dir(str(root)) == str(root)


@pytest.mark.parametrize("escape", ["../outside", "/etc", "a/../../outside", "~"])
def test_paths_outside_the_root_are_rejected(root: Path, escape: str) -> None:
    with pytest.raises(ValueError, match="outside the MCP root"):
        resolve_mcp_out_dir(escape)


def test_symlink_out_of_the_root_is_rejected(root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outside the MCP root"):
        resolve_mcp_out_dir("link")


def test_root_defaults_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(MCP_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    assert mcp_root() == tmp_path.resolve()


@pytest.mark.parametrize("tool_name", _TOOL_NAMES)
def test_every_tool_rejects_an_outside_out_dir_without_creating_it(
    root: Path, tmp_path: Path, tool_name: str
) -> None:
    """The handoff repro: handle_rca_status(out_dir=/tmp/x/deep/nested) created the tree."""
    outside = tmp_path / "x" / "deep" / "nested"
    tool = getattr(mcp_server, tool_name)
    kwargs = {"out_dir": str(outside), "response_format": "json"}
    if tool_name == "llmdebug_rca_advance":
        kwargs["action"] = "bind_evidence"

    payload = json.loads(tool(**kwargs))

    assert payload["error"]["code"] == "INVALID_ARGUMENT"
    assert "outside the MCP root" in payload["error"]["message"]
    assert not (tmp_path / "x").exists()


def test_tool_schemas_keep_their_parameters() -> None:
    tools = asyncio.run(mcp_server._mcp.list_tools())
    by_name = {tool.name: tool for tool in tools}

    assert set(_TOOL_NAMES) <= set(by_name)
    for name in _TOOL_NAMES:
        assert "out_dir" in by_name[name].inputSchema["properties"], name
    assert "index" in by_name["llmdebug_frame"].inputSchema["properties"]


_READ_ONLY_CALLS: dict[str, dict[str, object]] = {
    "llmdebug_diagnose": {},
    "llmdebug_show": {},
    "llmdebug_list": {},
    "llmdebug_frame": {"index": 0},
    "llmdebug_git_context": {},
    "llmdebug_diff": {},
    "llmdebug_rca_status": {},
    "llmdebug_rca_history": {},
}


def _tree(path: Path) -> list[str]:
    return sorted(str(item.relative_to(path)) for item in path.rglob("*"))


@pytest.mark.parametrize("tool_name", sorted(_READ_ONLY_CALLS))
def test_read_only_tools_do_not_create_a_missing_out_dir(root: Path, tool_name: str) -> None:
    """D11: read-only tools never create files, even inside the root."""
    getattr(mcp_server, tool_name)(out_dir="missing", **_READ_ONLY_CALLS[tool_name])

    assert not (root / "missing").exists()


@pytest.mark.parametrize("tool_name", sorted(_READ_ONLY_CALLS))
def test_read_only_tools_do_not_create_the_state_database(root: Path, tool_name: str) -> None:
    from llmdebug.config import SnapshotConfig
    from llmdebug.output import write_bundle

    out_dir = root / ".llmdebug"
    cfg = SnapshotConfig(out_dir=str(out_dir), output_format="json")
    for name in ("first", "second"):
        write_bundle({"name": name, "exception": {"type": "E", "message": name}, "frames": []}, cfg)
    for db_file in out_dir.glob("llmdebug_state.sqlite3*"):
        db_file.unlink()
    before = _tree(out_dir)

    getattr(mcp_server, tool_name)(out_dir=".llmdebug", **_READ_ONLY_CALLS[tool_name])

    assert _tree(out_dir) == before
