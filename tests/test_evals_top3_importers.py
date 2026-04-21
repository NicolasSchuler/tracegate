"""Tests for BugsInPy, PyBugHive, and TrickyBugs dataset importers."""

from __future__ import annotations

import zipfile
from pathlib import Path

from evals.importers.bugsinpy_importer import (
    _extract_python_files_from_patch,
    _iter_bugs,
    import_bugsinpy,
)
from evals.importers.native_repro_helpers import normalize_native_pytest_command
from evals.importers.pybughive_importer import _load_issue_specs, import_pybughive
from evals.importers.trickybugs_importer import _iter_python_candidates, import_trickybugs
from llmdebug import _json as json


def test_native_command_normalization_accepts_pytest_and_rejects_wrappers() -> None:
    parsed = normalize_native_pytest_command("pytest -q tests/test_demo.py::test_case\n")
    assert parsed is not None
    assert parsed.python_args == ["-m", "pytest", "-q", "tests/test_demo.py::test_case"]
    assert parsed.test_paths == ["tests/test_demo.py"]

    unsupported = normalize_native_pytest_command("pipenv run pytest -- tests/test_demo.py")
    assert unsupported is None


def test_bugsinpy_parser_reads_project_and_bug_entries(tmp_path) -> None:
    repo = tmp_path / "BugsInPy"
    bug_dir = repo / "projects" / "demo_project" / "bugs" / "1"
    bug_dir.mkdir(parents=True, exist_ok=True)

    (repo / "projects" / "demo_project" / "project.info").write_text(
        'github_url="https://github.com/example/demo-project"\n',
        encoding="utf-8",
    )
    (bug_dir / "bug.info").write_text(
        "\n".join(
            [
                'buggy_commit_id="abc123"',
                'fixed_commit_id="def456"',
                'test_file="tests/test_demo.py"',
                'pythonpath=""',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (bug_dir / "run_test.sh").write_text(
        "pytest -q tests/test_demo.py::test_case\n",
        encoding="utf-8",
    )
    (bug_dir / "bug_patch.txt").write_text(
        "\n".join(
            [
                "diff --git a/src/demo.py b/src/demo.py",
                "--- a/src/demo.py",
                "+++ b/src/demo.py",
                "@@ -1,2 +1,2 @@",
                "-return 1",
                "+return 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    specs = _iter_bugs(repo)
    assert len(specs) == 1
    assert specs[0].project_name == "demo_project"
    assert specs[0].bug_id == "1"
    assert specs[0].buggy_commit_id == "abc123"
    assert specs[0].fixed_commit_id == "def456"
    assert specs[0].test_file == "tests/test_demo.py"
    assert "pytest -q" in specs[0].run_test_cmd

    changed = _extract_python_files_from_patch(specs[0].patch_text)
    assert changed == ["src/demo.py"]


def test_bugsinpy_importer_emits_native_repro_metadata(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "BugsInPy"
    bug_dir = repo / "projects" / "demo_project" / "bugs" / "1"
    bug_dir.mkdir(parents=True, exist_ok=True)

    (repo / "projects" / "demo_project" / "project.info").write_text(
        'github_url="https://github.com/example/demo-project"\n',
        encoding="utf-8",
    )
    (bug_dir / "bug.info").write_text(
        "\n".join(
            [
                'buggy_commit_id="abc123"',
                'fixed_commit_id="def456"',
                'test_file="tests/test_demo.py"',
                'pythonpath=""',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (bug_dir / "run_test.sh").write_text(
        "pytest -q tests/test_demo.py::test_case\n",
        encoding="utf-8",
    )
    (bug_dir / "bug_patch.txt").write_text(
        "\n".join(
            [
                "diff --git a/src/demo.py b/src/demo.py",
                "--- a/src/demo.py",
                "+++ b/src/demo.py",
                "@@ -1 +1 @@",
                "-def add(a, b):\n-    return a-b",
                "+def add(a, b):\n+    return a+b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "evals.importers.bugsinpy_importer.ensure_git_repo", lambda *args, **kwargs: (True, None)
    )

    def _fake_git_show(repo_dir, request):
        del repo_dir
        if request.rel_path == "src/demo.py" and request.commitish == "abc123":
            return "def add(a, b):\n    return a-b\n"
        if request.rel_path == "src/demo.py" and request.commitish == "def456":
            return "def add(a, b):\n    return a+b\n"
        if request.rel_path == "tests/test_demo.py":
            return "from src.demo import add\n\n\ndef test_case():\n    assert add(2, 2) == 5\n"
        return None

    monkeypatch.setattr("evals.importers.bugsinpy_importer.git_show_file", _fake_git_show)

    out_dir = tmp_path / "cases"
    summary = import_bugsinpy(
        out_dir,
        bugsinpy_repo=repo,
        data_root=tmp_path / "work",
        max_cases=0,
        validate=False,
        dry_run=False,
        jobs=1,
    )

    assert summary.imported == 1
    case_dir = out_dir / "bip_demo_project_1"
    assert (case_dir / "src" / "demo.py").exists()
    assert (case_dir / "tests" / "test_demo.py").exists()
    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    assert meta["repro_mode"] == "native_dataset_test"
    assert meta["source_test_command"] == "pytest -q tests/test_demo.py::test_case"
    assert meta["source_test_selector"] == "tests/test_demo.py::test_case"
    assert meta["source_buggy_commit"] == "abc123"
    assert meta["source_fixed_commit"] == "def456"
    assert "marker" not in (case_dir / "test_buggy.py").read_text(encoding="utf-8")


def test_pybughive_parser_maps_issue_metadata(tmp_path) -> None:
    repo = tmp_path / "pybughive"
    dataset_dir = repo / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "username": "example",
            "repository": "project",
            "issues": [
                {
                    "id": 12,
                    "title": "Fix parser edge case",
                    "testSteps": "pytest -q tests/test_parser.py",
                    "testStepsFull": "pytest -q",
                    "commits": [
                        {
                            "hash": "fix123",
                            "parents": "bug123",
                            "stat": {
                                "files": [
                                    {"filename": "src/parser.py"},
                                    {"filename": "README.md"},
                                ]
                            },
                        }
                    ],
                }
            ],
        }
    ]
    (dataset_dir / "pybughive_current.json").write_text(json.dumps(payload), encoding="utf-8")

    specs = _load_issue_specs(repo)
    assert len(specs) == 1
    assert specs[0].username == "example"
    assert specs[0].repository == "project"
    assert specs[0].issue_id == 12
    assert specs[0].fix_commit == "fix123"
    assert specs[0].parent_commit == "bug123"
    assert specs[0].changed_python_files == ["src/parser.py"]


def test_pybughive_importer_uses_offline_archive_authoritatively(tmp_path) -> None:
    repo = tmp_path / "pybughive"
    dataset_dir = repo / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "username": "example",
            "repository": "demo",
            "issues": [
                {
                    "id": 7,
                    "title": "Fix demo",
                    "testSteps": "pytest -q tests/test_demo.py",
                    "commits": [
                        {
                            "hash": "fix123",
                            "parents": "bug123",
                            "stat": {"files": [{"filename": "src/demo.py"}]},
                        }
                    ],
                }
            ],
        }
    ]
    (dataset_dir / "pybughive_current.json").write_text(json.dumps(payload), encoding="utf-8")

    archive = tmp_path / "demo.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("dump/bug123/src/demo.py", "def add(a,b):\n    return a-b\n")
        zf.writestr("dump/fix123/src/demo.py", "def add(a,b):\n    return a+b\n")
        zf.writestr(
            "dump/fix123/tests/test_demo.py",
            "from src.demo import add\n\n\ndef test_demo():\n    assert add(2, 2) == 5\n",
        )

    out_dir = tmp_path / "cases"
    summary = import_pybughive(
        out_dir,
        pybughive_repo=repo,
        offline_archives={"offline_part1_demo_zip": archive},
        data_root=tmp_path / "work",
        max_cases=0,
        validate=False,
        dry_run=False,
        jobs=1,
    )

    assert summary.imported == 1
    case_dir = out_dir / "pbh_demo_7"
    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    assert meta["bug_source"] == "pybughive"
    assert meta["repro_mode"] == "native_dataset_test"
    assert meta["source_test_selector"] == "tests/test_demo.py"
    assert meta["source_buggy_commit"] == "bug123"
    assert meta["source_fixed_commit"] == "fix123"


def test_pybughive_importer_resolves_across_split_archives(tmp_path) -> None:
    repo = tmp_path / "pybughive"
    dataset_dir = repo / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "username": "example",
            "repository": "pandas",
            "issues": [
                {
                    "id": 70,
                    "title": "Fix split dump mapping",
                    "testSteps": "pytest -q tests/test_demo.py",
                    "commits": [
                        {
                            "hash": "fix123",
                            "parents": "bug123",
                            "stat": {"files": [{"filename": "src/demo.py"}]},
                        }
                    ],
                }
            ],
        }
    ]
    (dataset_dir / "pybughive_current.json").write_text(json.dumps(payload), encoding="utf-8")

    archive_1 = tmp_path / "pandas_1.zip"
    archive_2 = tmp_path / "pandas_2.zip"
    with zipfile.ZipFile(archive_1, "w") as zf:
        zf.writestr("dump/bug123/src/demo.py", "def add(a,b):\n    return a-b\n")
    with zipfile.ZipFile(archive_2, "w") as zf:
        zf.writestr("dump/fix123/src/demo.py", "def add(a,b):\n    return a+b\n")
        zf.writestr(
            "dump/fix123/tests/test_demo.py",
            "from src.demo import add\n\n\ndef test_demo():\n    assert add(2, 2) == 5\n",
        )

    summary = import_pybughive(
        tmp_path / "cases",
        pybughive_repo=repo,
        offline_archives={
            "offline_part2_pandas_1_zip": archive_1,
            "offline_part2_pandas_2_zip": archive_2,
        },
        data_root=tmp_path / "work",
        max_cases=0,
        validate=False,
        dry_run=True,
        jobs=1,
    )

    assert summary.attempted == 1
    assert summary.imported == 1
    assert summary.skipped == 0


def test_pybughive_importer_skips_ambiguous_offline_archive_mapping(tmp_path) -> None:
    repo = tmp_path / "pybughive"
    dataset_dir = repo / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "username": "example",
            "repository": "demo",
            "issues": [
                {
                    "id": 8,
                    "title": "Fix demo",
                    "testSteps": "pytest -q tests/test_demo.py",
                    "commits": [
                        {
                            "hash": "fix123",
                            "parents": "bug123",
                            "stat": {"files": [{"filename": "src/demo.py"}]},
                        }
                    ],
                }
            ],
        }
    ]
    (dataset_dir / "pybughive_current.json").write_text(json.dumps(payload), encoding="utf-8")

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    archive_a = dir_a / "demo.zip"
    archive_b = dir_b / "demo.zip"
    with zipfile.ZipFile(archive_a, "w") as zf:
        zf.writestr("dump/fix123/src/demo.py", "def add(a,b):\n    return a+b\n")
        zf.writestr("dump/bug123/src/demo.py", "def add(a,b):\n    return a-b\n")
        zf.writestr("dump/fix123/tests/test_demo.py", "def test_demo():\n    assert False\n")
    with zipfile.ZipFile(archive_b, "w") as zf:
        zf.writestr("dup/fix123/src/demo.py", "def add(a,b):\n    return a+b\n")
        zf.writestr("dup/bug123/src/demo.py", "def add(a,b):\n    return a-b\n")
        zf.writestr("dup/fix123/tests/test_demo.py", "def test_demo():\n    assert False\n")

    summary = import_pybughive(
        tmp_path / "cases",
        pybughive_repo=repo,
        offline_archives={"a": archive_a, "b": archive_b},
        data_root=tmp_path / "work",
        max_cases=0,
        validate=False,
        dry_run=True,
        jobs=1,
    )

    assert summary.attempted == 1
    assert summary.imported == 0
    assert summary.skip_reasons == {"offline_dump_path_ambiguous": 1}


def test_pybughive_importer_offline_missing_counts_attempted_per_issue(tmp_path) -> None:
    repo = tmp_path / "pybughive"
    dataset_dir = repo / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "username": "example",
            "repository": "demo",
            "issues": [
                {
                    "id": 1,
                    "title": "one",
                    "testSteps": "pytest -q tests/test_demo.py",
                    "commits": [
                        {
                            "hash": "fix111",
                            "parents": "bug111",
                            "stat": {"files": [{"filename": "src/demo.py"}]},
                        }
                    ],
                },
                {
                    "id": 2,
                    "title": "two",
                    "testSteps": "pytest -q tests/test_demo.py",
                    "commits": [
                        {
                            "hash": "fix222",
                            "parents": "bug222",
                            "stat": {"files": [{"filename": "src/demo.py"}]},
                        }
                    ],
                },
            ],
        }
    ]
    (dataset_dir / "pybughive_current.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = import_pybughive(
        tmp_path / "cases",
        pybughive_repo=repo,
        offline_archives={},
        data_root=tmp_path / "work",
        max_cases=0,
        validate=False,
        dry_run=True,
        jobs=1,
    )

    assert summary.attempted == 2
    assert summary.imported == 0
    assert summary.skipped == 2
    assert summary.skip_reasons == {"offline_dumps_missing": 2}


def test_trickybugs_candidate_extraction_and_skip_reason(tmp_path) -> None:
    root = tmp_path / "TrickyBugs"
    source_dir = root / "dataset"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "prog_a_buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (source_dir / "prog_a_fixed.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    (source_dir / "test_prog_a.py").write_text(
        "def test_prog_a():\n    assert False\n", encoding="utf-8"
    )

    payload = [
        {
            "pid": "p100",
            "program_id": "prog_a",
            "language": "Python",
            "buggy_file": "dataset/prog_a_buggy.py",
            "fixed_file": "dataset/prog_a_fixed.py",
            "test_command": "pytest -q dataset/test_prog_a.py",
        },
        {
            "pid": "p101",
            "program_id": "prog_b",
            "language": "python",
            "buggy_file": "dataset/prog_a_buggy.py",
            "fixed_file": "dataset/prog_a_fixed.py",
        },
    ]
    (root / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")

    candidates = _iter_python_candidates(root)
    assert len(candidates) == 2

    summary = import_trickybugs(
        tmp_path / "cases",
        trickybugs_root=root,
        data_root=tmp_path / "work",
        max_cases=0,
        validate=False,
        dry_run=True,
        jobs=1,
    )

    assert summary.attempted == 2
    assert summary.imported == 1
    assert summary.skip_reasons["missing_native_test_oracle_mapping"] == 1
