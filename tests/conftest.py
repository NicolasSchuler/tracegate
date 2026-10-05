"""Shared test fixtures and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llmdebug.output import get_latest_snapshot


@pytest.fixture(autouse=True)
def _mcp_root_covers_tmp_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """MCP tools confine ``out_dir`` to ``$LLMDEBUG_MCP_ROOT`` (RCA-02); tests use tmp dirs."""
    monkeypatch.setenv("LLMDEBUG_MCP_ROOT", str(tmp_path_factory.getbasetemp()))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options for live repair integration test overrides."""
    parser.addoption(
        "--live-openai-base-url",
        action="store",
        default="http://127.0.0.1:1234/v1",
        help=(
            "OpenAI-compatible base URL for live repair integration tests "
            "(default: http://127.0.0.1:1234/v1)."
        ),
    )
    parser.addoption(
        "--live-openai-model",
        action="store",
        default="",
        help=(
            "Model name for live repair integration tests. "
            "When omitted, the first model from /v1/models is used."
        ),
    )
    parser.addoption(
        "--live-openai-compat-profile",
        action="store",
        choices=("auto", "qwen_openai_raw", "qwen_dashscope", "lmstudio_model_yaml", "off"),
        default="auto",
        help=("OpenAI compatibility profile for live repair integration tests (default: auto)."),
    )
    parser.addoption(
        "--live-decision-model-url",
        action="store",
        default="",
        help=(
            "Base URL of a running System-1 decision-model sidecar (e.g. "
            "http://127.0.0.1:8090) for the live decision-model integration test. "
            "Falls back to $TRACEGATE_DECISION_MODEL_URL; the test skips when both are unset."
        ),
    )


def read_latest_snapshot(out_dir: Path) -> dict[str, Any]:
    """Read the latest snapshot, auto-expanding compact keys.

    Raises AssertionError if no snapshot exists in the directory.
    """
    snapshot = get_latest_snapshot(str(out_dir))
    assert snapshot is not None, f"No snapshot in {out_dir}"
    return snapshot
