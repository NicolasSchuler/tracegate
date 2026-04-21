"""Tests for evidence ID assignment and multi-file context in eval runs."""

from __future__ import annotations

from pathlib import Path

from evals import run_eval


def _make_case_dir(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case_src"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text(
        "import helper\n\n\ndef f() -> int:\n    return helper.value()\n",
        encoding="utf-8",
    )
    (case_dir / "helper.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return case_dir


def test_execute_context_requests_deduplicates_evidence_ids(tmp_path) -> None:
    root = _make_case_dir(tmp_path)
    payload = {
        "need_more_context": True,
        "requests": [
            {"type": "file", "path": "buggy.py"},
            {"type": "file", "path": "buggy.py"},
            {"type": "search", "path": "buggy.py", "pattern": "helper.value"},
            {"type": "search", "path": "buggy.py", "pattern": "helper.value"},
        ],
    }

    result = run_eval.execute_context_requests(
        request_payload=payload,
        root=root,
        snapshot=None,
        file_index="buggy.py\nhelper.py\ntest_buggy.py",
        stdout="",
        stderr="",
        max_requests=6,
        max_total_chars=60_000,
    )

    ids = [str(block.get("id") or "") for block in result.blocks]
    assert result.used_fallback is False
    assert len(ids) == len(set(ids))
    assert result.requests_count == len(result.blocks)
    assert result.text.count("--- buggy.py ---") == 1


def test_fallback_context_bundle_evidence_ids_stable(tmp_path) -> None:
    root = _make_case_dir(tmp_path)
    snapshot = {
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 4,
                "function": "f",
                "code": "return helper.value()",
            }
        ]
    }

    result_a = run_eval.build_fallback_context_bundle(
        root=root,
        snapshot=snapshot,
        stdout="tests/test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
        stderr="",
        max_total_chars=60_000,
        reason="test",
    )
    result_b = run_eval.build_fallback_context_bundle(
        root=root,
        snapshot=snapshot,
        stdout="tests/test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
        stderr="",
        max_total_chars=60_000,
        reason="test",
    )

    ids_a = [str(block.get("id") or "") for block in result_a.blocks]
    ids_b = [str(block.get("id") or "") for block in result_b.blocks]
    assert ids_a
    assert ids_a == ids_b


def test_context_bundle_text_annotates_evidence_ids(tmp_path) -> None:
    root = _make_case_dir(tmp_path)
    payload = {
        "need_more_context": True,
        "requests": [{"type": "file", "path": "buggy.py"}],
    }

    result = run_eval.execute_context_requests(
        request_payload=payload,
        root=root,
        snapshot=None,
        file_index="buggy.py\nhelper.py\ntest_buggy.py",
        stdout="",
        stderr="",
        max_requests=6,
        max_total_chars=60_000,
    )

    assert result.blocks
    evidence_id = str(result.blocks[0]["id"])
    assert evidence_id
    assert f"[{evidence_id}]" in result.text
