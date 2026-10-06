"""The generated CLI-defaults tables come from the parser and catch drift (X-08)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from evals import doc_sync_cli
from evals.doc_sync_cli import BASELINE_CLI_DEFAULTS, GeneratedBlock, cli_flag_rows
from evals.eval_cli_args import build_parser


def test_rows_use_the_parser_defaults() -> None:
    rows = "\n".join(cli_flag_rows("--baseline"))
    parser = build_parser()
    teacher = parser.get_default("baseline_intervenor_teacher_max_tokens")
    assert (
        f"| `--baseline-intervenor-teacher-max-tokens` | "
        f"`baseline_intervenor_teacher_max_tokens` | `{teacher}` |"
    ) in rows
    # A flag without its own default says so instead of restating the fallback.
    assert "| `--baseline-process-timeout-sec` | `baseline_process_timeout_sec` | unset |" in rows


def test_negations_share_a_row_and_help_drops_the_default_note() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-flag", dest="x_flag", action="store_true", default=True, help="On.")
    parser.add_argument("--no-x-flag", dest="x_flag", action="store_false", help="Off.")
    parser.add_argument("--x-size", type=int, default=3, help="Size | count (default: 3).")
    assert cli_flag_rows("--x", parser) == [
        "| `--x-flag` / `--no-x-flag` | `x_flag` | `true` | On. |",
        "| `--x-size` | `x_size` | `3` | Size \\| count. |",
    ]


def test_block_detects_and_rewrites_stale_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doc_sync_cli, "REPO_ROOT", tmp_path)
    doc = tmp_path / "doc.md"
    block = GeneratedBlock(name="DEMO", docs=(doc,), render_body=lambda: ["| a |"])
    doc.write_text(f"intro\n{block.begin}\nold\n{block.end}\noutro\n", encoding="utf-8")

    assert block.stale_docs() == ["doc.md"]
    assert block.write() == ["doc.md"]
    assert block.stale_docs() == []
    assert block.write() == []
    text = doc.read_text(encoding="utf-8")
    assert text.startswith("intro\n") and text.endswith("outro\n")

    doc.write_text("no markers\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        block.write()


def test_repository_baseline_doc_is_in_sync() -> None:
    assert BASELINE_CLI_DEFAULTS.stale_docs() == []
