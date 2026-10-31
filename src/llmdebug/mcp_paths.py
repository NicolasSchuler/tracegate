"""Confine MCP tool paths to one root directory (RCA-02, decision D11).

MCP clients are LLM agents, so every ``out_dir`` they pass is untrusted input. Tools
resolve it against the MCP root and refuse anything outside, which also bounds the
snapshot paths and refs that are validated against ``out_dir``.
"""

from __future__ import annotations

import os
from pathlib import Path

MCP_ROOT_ENV = "LLMDEBUG_MCP_ROOT"


def mcp_root() -> Path:
    """Return the directory the MCP tools may use.

    Returns:
        The resolved ``$LLMDEBUG_MCP_ROOT`` when it is set and non-empty, else the
        server's current working directory.
    """
    raw = os.environ.get(MCP_ROOT_ENV, "").strip()
    return Path(raw or os.getcwd()).expanduser().resolve()


def resolve_mcp_out_dir(out_dir: str) -> str:
    """Resolve *out_dir* (relative to the MCP root) and return it as an absolute path.

    Args:
        out_dir: Directory requested by the MCP client; relative paths are taken
            relative to the MCP root, and ``~`` is expanded.

    Returns:
        The absolute, symlink-resolved directory path.

    Raises:
        ValueError: If the resolved directory, after following symlinks, is outside
            the MCP root.
    """
    root = mcp_root()
    candidate = Path(out_dir).expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"out_dir {out_dir!r} is outside the MCP root {root}; "
            f"set {MCP_ROOT_ENV} to allow another directory"
        )
    return str(resolved)
