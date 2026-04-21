"""Tests for optional-dependency import guards in CLI and MCP server modules."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


class TestCLIImportGuard:
    """Verify cli.py exits cleanly when click is missing."""

    def test_missing_click_raises_system_exit(self) -> None:
        """cli.py should raise SystemExit(1) with a helpful message when click is absent."""
        mod_name = "llmdebug.cli"
        saved = sys.modules.pop(mod_name, None)
        try:
            real_import = (
                __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
            )  # type: ignore[union-attr]

            def _block_click(name: str, *args: object, **kwargs: object) -> object:
                if name == "click":
                    raise ImportError("No module named 'click'")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_block_click):
                with pytest.raises(SystemExit) as exc_info:
                    importlib.import_module(mod_name)
                assert exc_info.value.code == 1
        finally:
            # Restore the module so other tests are unaffected
            if saved is not None:
                sys.modules[mod_name] = saved
            else:
                sys.modules.pop(mod_name, None)


class TestMCPServerImportGuard:
    """Verify mcp_server.py exits cleanly when mcp is missing."""

    def test_missing_mcp_raises_system_exit(self) -> None:
        """mcp_server.py should raise SystemExit(1) when mcp package is absent."""
        mod_name = "llmdebug.mcp_server"
        saved = sys.modules.pop(mod_name, None)
        try:
            real_import = (
                __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
            )  # type: ignore[union-attr]

            def _block_mcp(name: str, *args: object, **kwargs: object) -> object:
                if name == "mcp.server.fastmcp" or name == "mcp":
                    raise ImportError("No module named 'mcp'")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_block_mcp):
                with pytest.raises(SystemExit) as exc_info:
                    importlib.import_module(mod_name)
                assert exc_info.value.code == 1
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved
            else:
                sys.modules.pop(mod_name, None)
