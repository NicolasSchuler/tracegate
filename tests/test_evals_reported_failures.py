"""Failures that used to be swallowed are now reported (X-07)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from evals.importers.base import existing_source_ids
from evals.importers.trickybugs_importer import _iter_python_candidates, import_trickybugs
from evals.swebench_pro import score_docker


def test_trickybugs_scan_reports_unparseable_json(tmp_path: Path) -> None:
    root = tmp_path / "trickybugs"
    root.mkdir()
    (root / "broken.json").write_text("{not json", encoding="utf-8")

    unreadable: list[Path] = []
    assert _iter_python_candidates(root, unreadable=unreadable) == []
    assert unreadable == [root / "broken.json"]

    summary = import_trickybugs(
        tmp_path / "cases",
        trickybugs_root=root,
        data_root=tmp_path / "work",
        dry_run=True,
    )
    assert "Ignored 1 unreadable JSON file(s) during the scan." in summary.notes


def test_source_id_scan_warns_about_unreadable_case_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    good = tmp_path / "good"
    good.mkdir()
    (good / "case.json").write_text('{"source_id": "src-1"}', encoding="utf-8")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "case.json").write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="evals.importers.base"):
        assert existing_source_ids(tmp_path) == {"src-1"}
    assert any(str(bad / "case.json") in record.getMessage() for record in caplog.records)


class _StuckContainer:
    short_id = "abc123"

    def stop(self, **_kwargs: Any) -> None:
        raise RuntimeError("daemon gone")

    def remove(self, **_kwargs: Any) -> None:
        raise RuntimeError("daemon gone")


def test_container_cleanup_failures_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="evals.swebench_pro.score_docker"):
        score_docker._cleanup_container(_StuckContainer())
    messages = [record.getMessage() for record in caplog.records]
    assert any("could not stop scoring container abc123" in m for m in messages)
    assert any("could not remove scoring container abc123" in m for m in messages)


def test_log_copy_failure_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "out.log"
    src.write_text("log", encoding="utf-8")
    dst = tmp_path / "missing-dir" / "out.log"
    with caplog.at_level(logging.WARNING, logger="evals.swebench_pro.score_docker"):
        score_docker._copy_log(src, dst)
    assert any("could not copy scoring log" in r.getMessage() for r in caplog.records)
