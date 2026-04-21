"""High-yield synthetic tests for the eval importer hotspots.

These tests stay fully local: they use temporary directories, in-memory fake
datasets, and stubbed helper functions so we can exercise the branch-heavy
importer logic without downloading HuggingFace data or touching real repos.
"""

from __future__ import annotations

import sys
import types
import zipfile
from pathlib import Path

from evals.importers import (
    condefects_importer,
    debugbench_importer,
    humanevalpack_importer,
    pyresbugs_importer,
    repair_debugbench_patches,
)


def _install_fake_datasets(monkeypatch, rows: list[dict[str, object]]) -> None:
    module = types.ModuleType("datasets")

    def _load_dataset(*args, **kwargs):
        return rows

    module.load_dataset = _load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


def _write_case(case_dir: Path, *, source_id: str, oracle_patch: str = "old patch\n") -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(
        f'{{"source_id": "{source_id}"}}\n',
        encoding="utf-8",
    )
    (case_dir / "buggy.py").write_text("print('buggy')\n", encoding="utf-8")
    (case_dir / "oracle.patch").write_text(oracle_patch, encoding="utf-8")


def _write_condefects_zip(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for idx, name in enumerate(
            [
                "example_00.txt",
                "sample_00.txt",
                "sample_01.txt",
                "sample_02.txt",
                "sample_03.txt",
                "sample_04.txt",
                "sample_05.txt",
            ]
        ):
            zf.writestr(f"Test/abc221/A/in/{name}", f"{idx}\n")
            zf.writestr(f"Test/abc221/A/out/{name}", f"{idx + 10}\n")
        zf.writestr(
            "Test/abc221/A/in/huge.txt",
            "x" * (condefects_importer.MAX_TEST_INPUT_BYTES + 1),
        )
        zf.writestr("Test/abc221/A/out/huge.txt", "999\n")
        zf.writestr("Test/abc223/Ex/in/example_00.txt", "8\n")
        zf.writestr("Test/abc223/Ex/out/example_00.txt", "18\n")


def test_condefects_helpers_cover_zip_pairing_and_classification(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "difficulty.txt").write_text(
        "abc221_a 350\nabc222_b nope\nabc223 1000\nabc224_c 1400\n",
        encoding="utf-8",
    )

    assert condefects_importer._parse_difficulty_file(repo) == {
        "abc221_a": 350,
        "abc222_b": None,
        "abc223": 1000,
        "abc224_c": 1400,
    }
    assert condefects_importer._difficulty_from_score(None) is None
    assert condefects_importer._difficulty_from_score(400) == "easy"
    assert condefects_importer._difficulty_from_score(1200) == "medium"
    assert condefects_importer._difficulty_from_score(1300) == "hard"
    assert condefects_importer._code_path_to_test_prefix("abc221_a") == (
        "Test/abc221/A",
        "Test/abc221/Ex",
    )
    assert condefects_importer._code_path_to_test_prefix("contest") == (
        "Test/contest",
        "Test/contest/Ex",
    )
    assert (
        condefects_importer._classify_bug_from_patch(
            "if value < limit:\n    return 1\n",
            "if value <= limit:\n    return 1\n",
        )
        == "logic_error"
    )
    assert (
        condefects_importer._classify_bug_from_patch(
            "return int(value)\n",
            "return str(value)\n",
        )
        == "type_coercion"
    )
    assert condefects_importer._estimate_difficulty(1, 10) == "easy"
    assert condefects_importer._estimate_difficulty(4, 80) == "medium"
    assert condefects_importer._estimate_difficulty(5, 81) == "hard"

    zip_path = tmp_path / "tests.zip"
    _write_condefects_zip(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        index = condefects_importer._build_zip_index(zf)
        assert index["Test/abc221/A/in"][0] == "example_00.txt"
        pairs = condefects_importer._extract_test_pairs_from_zip(zf, index, "Test/abc221/A")
        assert len(pairs) == condefects_importer.MAX_TEST_PAIRS
        assert pairs[0] == ("0\n", "10\n")
        assert pairs[1] == ("1\n", "11\n")
        fallback_pairs = condefects_importer._extract_test_pairs_from_zip(
            zf,
            index,
            "Test/abc223/Ex",
        )
        assert fallback_pairs == [("8\n", "18\n")]

    generated = condefects_importer._generate_test([("input\n", "output\n"), ("x\n", "y\n")])
    assert "test_condefects_0" in generated
    assert "test_condefects_1" in generated
    assert "input=" in generated
    assert "expected = 'output\\n'" in generated


def test_condefects_import_dry_run_covers_primary_fallback_and_skips(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    code_dir = repo / "Code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (repo / "difficulty.txt").write_text("abc221_a 350\nabc223 1000\n", encoding="utf-8")

    valid_primary = code_dir / "abc221_a" / "Python" / "1"
    valid_primary.mkdir(parents=True, exist_ok=True)
    (valid_primary / "faultyVersion.py").write_text(
        "def solve():\n    return 1\n", encoding="utf-8"
    )
    (valid_primary / "correctVersion.py").write_text(
        "def solve():\n    return 2\n", encoding="utf-8"
    )

    valid_fallback = code_dir / "abc223_c" / "Python" / "2"
    valid_fallback.mkdir(parents=True, exist_ok=True)
    (valid_fallback / "faultyVersion.py").write_text(
        "def solve():\n    return 3\n", encoding="utf-8"
    )
    (valid_fallback / "correctVersion.py").write_text(
        "def solve():\n    return 4\n", encoding="utf-8"
    )

    no_python = code_dir / "abc222_b" / "Java"
    no_python.mkdir(parents=True, exist_ok=True)
    (no_python / "README.txt").write_text("no python here\n", encoding="utf-8")

    external = code_dir / "abc224_d" / "Python" / "4"
    external.mkdir(parents=True, exist_ok=True)
    (external / "faultyVersion.py").write_text("external dependency\n", encoding="utf-8")
    (external / "correctVersion.py").write_text("external dependency fixed\n", encoding="utf-8")

    nondeterministic = code_dir / "abc225_e" / "Python" / "5"
    nondeterministic.mkdir(parents=True, exist_ok=True)
    (nondeterministic / "faultyVersion.py").write_text("nondet code\n", encoding="utf-8")
    (nondeterministic / "correctVersion.py").write_text("nondet code fixed\n", encoding="utf-8")

    no_patch = code_dir / "abc226_f" / "Python" / "6"
    no_patch.mkdir(parents=True, exist_ok=True)
    (no_patch / "faultyVersion.py").write_text("nopatch code\n", encoding="utf-8")
    (no_patch / "correctVersion.py").write_text("nopatch code fixed\n", encoding="utf-8")

    long_patch = code_dir / "abc227_g" / "Python" / "7"
    long_patch.mkdir(parents=True, exist_ok=True)
    (long_patch / "faultyVersion.py").write_text("longpatch code\n", encoding="utf-8")
    (long_patch / "correctVersion.py").write_text("longpatch code fixed\n", encoding="utf-8")

    no_tests = code_dir / "abc229_i" / "Python" / "9"
    no_tests.mkdir(parents=True, exist_ok=True)
    (no_tests / "faultyVersion.py").write_text("def solve():\n    return 9\n", encoding="utf-8")
    (no_tests / "correctVersion.py").write_text("def solve():\n    return 10\n", encoding="utf-8")

    missing_fixed = code_dir / "abc230_j" / "Python" / "10"
    missing_fixed.mkdir(parents=True, exist_ok=True)
    (missing_fixed / "faultyVersion.py").write_text(
        "def solve():\n    return 11\n", encoding="utf-8"
    )

    _install_fake_datasets(monkeypatch, [])
    _write_condefects_zip(tmp_path / "tests.zip")
    monkeypatch.setattr(
        condefects_importer, "infer_snapshot_dependency", lambda category: f"dep:{category}"
    )
    monkeypatch.setattr(condefects_importer, "ruff_format", lambda code: code)
    monkeypatch.setattr(
        condefects_importer,
        "has_external_deps",
        lambda code: "external" in code,
    )
    monkeypatch.setattr(
        condefects_importer,
        "is_deterministic",
        lambda code: "nondet" not in code,
    )

    def _fake_generate_oracle_patch(buggy_code: str, fixed_code: str) -> str:
        if "nopatch" in buggy_code:
            return ""
        if "longpatch" in buggy_code:
            return "LONGPATCH"
        return "--- a/buggy.py\n+++ b/buggy.py\n@@ -1,2 +1,2 @@\n-    return 1\n+    return 2\n"

    monkeypatch.setattr(condefects_importer, "generate_oracle_patch", _fake_generate_oracle_patch)
    monkeypatch.setattr(
        condefects_importer,
        "count_patch_lines",
        lambda patch: 20 if patch == "LONGPATCH" else 1,
    )

    output_dir = tmp_path / "cases"
    imported = condefects_importer.import_condefects(
        output_dir,
        repo_path=repo,
        test_zip=tmp_path / "tests.zip",
        dry_run=True,
    )

    assert [case["case_id"] for case in imported] == [
        "cd_abc221_a_1_easy",
        "cd_abc223_c_2_medium",
    ]
    assert imported[0]["metadata"]["snapshot_dependency"] == "dep:logic_error"
    assert imported[1]["metadata"]["snapshot_dependency"] == "dep:logic_error"


def test_humanevalpack_helpers_and_dry_run_import(tmp_path: Path, monkeypatch) -> None:
    assert humanevalpack_importer._generate_test(
        "f()",
        "def check(x):\n    assert x == 2\n",
        "",
    ).startswith("from buggy import *")
    assert "check(f())" in humanevalpack_importer._generate_test(
        "f()",
        "def check(x):\n    assert x == 2\n",
        "",
    )
    assert "def test_main():" in humanevalpack_importer._generate_test(
        "f()",
        "assert f() == 2\n",
        "",
    )
    assert humanevalpack_importer._infer_exception("Any", "assertion failed") == "AssertionError"
    assert humanevalpack_importer._infer_exception("Any", "type mismatch") == "TypeError"
    assert humanevalpack_importer._infer_exception("Any", "index missing") == "IndexError"
    assert humanevalpack_importer._infer_exception("Any", "name missing") == "NameError"
    assert humanevalpack_importer._infer_exception("Any", "") == "AssertionError"

    rows = [
        {"task_id": "", "buggy_solution": "", "canonical_solution": "", "test": ""},
        {
            "task_id": "Task/One",
            "buggy_solution": "def f():\n    return 1\n",
            "canonical_solution": "def f():\n    return 2\n",
            "prompt": "",
            "test": "def check(x):\n    assert x == 2\n",
            "bug_type": "Name Bug",
            "entry_point": "f()",
            "failure_symptoms": "name missing",
        },
        {
            "task_id": "Task Two",
            "buggy_solution": "def g():\n    return 1\n",
            "canonical_solution": "def g():\n    return 2\n",
            "prompt": "",
            "test": "assert g() == 2\n",
            "bug_type": "Type Bug",
            "entry_point": "g()",
            "failure_symptoms": "type mismatch",
        },
        {
            "task_id": "Task Long",
            "buggy_solution": "def h():\n    return 1\n",
            "canonical_solution": "def h():\n    return 2\n",
            "prompt": "",
            "test": "",
            "bug_type": "Name Bug",
            "entry_point": "h()",
            "failure_symptoms": "name missing",
        },
        {
            "task_id": "Task Long Patch",
            "buggy_solution": "longpatch\n",
            "canonical_solution": "longpatch fixed\n",
            "prompt": "",
            "test": "assert True\n",
            "bug_type": "Name Bug",
            "entry_point": "h()",
            "failure_symptoms": "name missing",
        },
        {
            "task_id": "Task Many",
            "buggy_solution": "\n".join(["x = 1"] * (humanevalpack_importer.MAX_CODE_LINES + 1)),
            "canonical_solution": "\n".join(
                ["x = 2"] * (humanevalpack_importer.MAX_CODE_LINES + 1)
            ),
            "prompt": "",
            "test": "assert True\n",
            "bug_type": "Type Bug",
            "entry_point": "g()",
            "failure_symptoms": "type mismatch",
        },
    ]
    _install_fake_datasets(monkeypatch, rows)
    monkeypatch.setattr(
        humanevalpack_importer,
        "map_humanevalpack_type",
        lambda bug_type: bug_type.lower().replace(" ", "_") or "logic_error",
    )
    monkeypatch.setattr(
        humanevalpack_importer, "infer_snapshot_dependency", lambda category: f"dep:{category}"
    )
    monkeypatch.setattr(
        humanevalpack_importer,
        "generate_oracle_patch",
        lambda buggy, fixed: (
            "LONGPATCH"
            if "longpatch" in buggy.lower()
            else ("--- a/buggy.py\n+++ b/buggy.py\n-    return 1\n+    return 2\n")
        ),
    )
    monkeypatch.setattr(
        humanevalpack_importer,
        "count_patch_lines",
        lambda patch: 20 if patch == "LONGPATCH" else 1,
    )

    imported = humanevalpack_importer.import_humanevalpack(
        tmp_path / "cases",
        dry_run=True,
        validate=False,
    )

    assert [case["case_id"] for case in imported] == [
        "hep_task_one_name_bug",
        "hep_task_two_type_bug",
    ]
    assert imported[0]["metadata"]["expected_exception"] == "NameError"
    assert imported[1]["metadata"]["expected_exception"] == "TypeError"


def test_repair_debugbench_patch_applies_success_and_failure(tmp_path: Path, monkeypatch) -> None:
    case_dir = tmp_path / "db_ok"
    case_dir.mkdir()
    (case_dir / "oracle.patch").write_text("patch\n", encoding="utf-8")

    def _ok_run(*args, **kwargs):
        return __import__("subprocess").CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.setattr(repair_debugbench_patches.subprocess, "run", _ok_run)
    assert repair_debugbench_patches._patch_applies(case_dir) is True

    def _timeout(*args, **kwargs):
        raise __import__("subprocess").TimeoutExpired(cmd="patch", timeout=5)

    monkeypatch.setattr(repair_debugbench_patches.subprocess, "run", _timeout)
    assert repair_debugbench_patches._patch_applies(case_dir) is False


def test_repair_debugbench_patches_repairs_and_reports_failures(
    tmp_path: Path, monkeypatch
) -> None:
    case_root = tmp_path / "cases"
    ok = case_root / "db_ok"
    bad_json = case_root / "db_bad_json"
    bad_source = case_root / "db_bad_source"
    bad_parse = case_root / "db_bad_parse"
    missing_row = case_root / "db_missing_row"
    repair = case_root / "db_repair"

    _write_case(ok, source_id="debugbench_0")
    _write_case(bad_source, source_id="oops")
    _write_case(bad_parse, source_id="debugbench_x")
    _write_case(missing_row, source_id="debugbench_99")
    _write_case(repair, source_id="debugbench_0")
    bad_json.mkdir(parents=True, exist_ok=True)
    (bad_json / "oracle.patch").write_text("patch\n", encoding="utf-8")

    rows = [
        {
            "language": "python3",
            "buggy_code": "from helper import value\n\nclass Solution:\n    def solve(self):\n        return value()\n",
            "solution": "from helper import value\n\nclass Solution:\n    def solve(self):\n        return value() + 1\n",
        }
    ]
    _install_fake_datasets(monkeypatch, rows)
    monkeypatch.setattr(repair_debugbench_patches, "ruff_format", lambda code: code)
    monkeypatch.setattr(
        repair_debugbench_patches,
        "generate_oracle_patch",
        lambda buggy, fixed: "new patch\n",
    )
    monkeypatch.setattr(repair_debugbench_patches, "count_patch_lines", lambda patch: 1)

    def _fake_patch_applies(case_dir: Path) -> bool:
        if case_dir.name == "db_ok":
            return True
        if case_dir.name == "db_repair":
            return "new patch" in (case_dir / "oracle.patch").read_text(encoding="utf-8")
        return False

    monkeypatch.setattr(repair_debugbench_patches, "_patch_applies", _fake_patch_applies)

    results = repair_debugbench_patches.repair_debugbench_patches(case_root)

    assert results["db_ok"] == "already_ok"
    assert results["db_bad_json"] == "failed: missing case.json"
    assert results["db_bad_source"].startswith("failed: unexpected source_id=")
    assert results["db_bad_parse"].startswith("failed: can't parse row index")
    assert results["db_missing_row"] == "failed: row 99 not in dataset"
    assert results["db_repair"] == "repaired"


def test_pyresbugs_helpers_cover_exception_and_test_generation() -> None:
    assert pyresbugs_importer._unescape_code("a\\\\nb\\t\\'\\\"\\\\") == "a\\nb\t'\"\\"
    assert pyresbugs_importer._infer_difficulty("x = 1\n", "x = 2\n", 1) == "easy"
    assert pyresbugs_importer._infer_difficulty("x = 1\n" * 61, "x = 2\n", 3) == "hard"
    assert pyresbugs_importer._infer_difficulty("x = 1\n" * 10, "x = 2\n", 4) == "medium"
    assert (
        pyresbugs_importer._infer_exception("MIA", "def f():\n    return 1\n", "import")
        == "ImportError"
    )
    assert (
        pyresbugs_importer._infer_exception("Any", "def f():\n    return 1\n", "import")
        == "NameError"
    )
    assert (
        pyresbugs_importer._infer_exception("WPFV", "def f():\n    return 1\n", "call")
        == "NameError"
    )
    assert (
        pyresbugs_importer._infer_exception("Any", "def f():\n    return 1\n", "call")
        == "AssertionError"
    )
    assert pyresbugs_importer._infer_exception("Any", "def broken(", "call") == "SyntaxError"
    assert pyresbugs_importer._find_top_level_functions(
        "def a():\n    pass\n\nasync def b():\n    pass\n"
    ) == [
        "a",
        "b",
    ]
    assert pyresbugs_importer._find_top_level_functions("def broken(") == []
    assert (
        pyresbugs_importer._find_changed_function(
            "def a():\n    return 1\n\ndef b():\n    return 2\n",
            "def a():\n    return 1\n\ndef b():\n    return 3\n",
        )
        == "b"
    )
    assert (
        pyresbugs_importer._find_changed_function(
            "def a():\n    return 1\n",
            "def a():\n    return 2\n",
        )
        == "a"
    )
    assert pyresbugs_importer._get_function_params(
        "def f(self, x, cls, y=1):\n    return x\n", "f"
    ) == [
        "x",
        "y",
    ]
    assert pyresbugs_importer._get_function_params("def broken(", "f") is None
    assert "def test_import()" in pyresbugs_importer._generate_test(
        "def f():\n    return 1\n",
        "def f():\n    return 2\n",
        "MIA",
    )
    assert "def test_call():" in pyresbugs_importer._generate_test(
        "def f():\n    return 1\n",
        "def f():\n    return 2\n",
        "WPFV",
    )
    assert "def test_import()" in pyresbugs_importer._generate_test(
        "def f(x):\n    return x\n",
        "def f(x):\n    return x + 1\n",
        "TYPE",
    )
    assert (
        pyresbugs_importer._ensure_valid_module("    def f():\n        return 1\n")
        == "def f():\n    return 1\n"
    )


def test_pyresbugs_import_dry_run_covers_filters_and_metadata(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {"Faulty Code": "", "Fault Free Code": "", "Fault_Acronym": "MIA"},
        {
            "Faulty Code": "def ping():\\n    return 1\\n",
            "Fault Free Code": "def ping():\\n    return 2\\n",
            "Fault_Acronym": "MIA",
            "High-Level Description": "missing import",
            "Project": "demo",
        },
        {
            "Faulty Code": "def pong():\\n    return 1\\n",
            "Fault Free Code": "def pong():\\n    return 2\\n",
            "Fault_Acronym": "WPFV",
            "High-Level Description": "wrong value",
            "Project": "demo",
        },
        {
            "Faulty Code": "def needs_arg(x):\\n    return x\\n",
            "Fault Free Code": "def needs_arg(x):\\n    return x + 1\\n",
            "Fault_Acronym": "TYPE",
            "High-Level Description": "requires args",
            "Project": "demo",
        },
        {
            "Faulty Code": "external import requests\\n",
            "Fault Free Code": "external import requests fixed\\n",
            "Fault_Acronym": "MIA",
        },
        {
            "Faulty Code": "nondet code\\n",
            "Fault Free Code": "nondet code fixed\\n",
            "Fault_Acronym": "MIA",
        },
        {
            "Faulty Code": "\n".join(["x = 1"] * (pyresbugs_importer.MAX_CODE_LINES + 1)),
            "Fault Free Code": "\n".join(["x = 2"] * (pyresbugs_importer.MAX_CODE_LINES + 1)),
            "Fault_Acronym": "MIA",
        },
    ]
    _install_fake_datasets(monkeypatch, rows)
    monkeypatch.setattr(pyresbugs_importer, "map_pyresbugs_type", lambda acronym: acronym.lower())
    monkeypatch.setattr(
        pyresbugs_importer, "infer_snapshot_dependency", lambda category: f"dep:{category}"
    )
    monkeypatch.setattr(pyresbugs_importer, "ruff_format", lambda code: code)
    monkeypatch.setattr(
        pyresbugs_importer,
        "has_external_deps",
        lambda code: "external" in code,
    )
    monkeypatch.setattr(
        pyresbugs_importer,
        "is_deterministic",
        lambda code: "nondet" not in code,
    )
    monkeypatch.setattr(
        pyresbugs_importer,
        "generate_oracle_patch",
        lambda buggy, fixed: "--- a/buggy.py\n+++ b/buggy.py\n-    return 1\n+    return 2\n",
    )
    monkeypatch.setattr(pyresbugs_importer, "count_patch_lines", lambda patch: 1)

    imported = pyresbugs_importer.import_pyresbugs(tmp_path / "cases", dry_run=True, validate=False)

    assert [case["case_id"] for case in imported] == [
        "prb_0001_easy",
        "prb_0002_easy",
        "prb_0003_easy",
    ]
    assert imported[0]["metadata"]["expected_exception"] == "ImportError"
    assert imported[1]["metadata"]["expected_exception"] == "NameError"
    assert imported[2]["metadata"]["expected_exception"] == "NameError"


def test_debugbench_helpers_cover_example_and_fallback_paths() -> None:
    code = (
        "class Solution:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "    def solve(self, nums, limit=1):\n"
        "        return nums\n"
    )
    assert debugbench_importer._extract_method_name(code) == "solve"
    assert debugbench_importer._extract_method_name("class X:\n    pass\n") is None
    assert debugbench_importer._parse_examples(
        ["Input: nums = [1, 2]\nOutput: 3\n", "ignored"]
    ) == [("nums = [1, 2]", "3")]
    assert debugbench_importer._parse_input_args("nums = [1, 2], target = 3") == [
        ("nums", "[1, 2]"),
        ("target", "3"),
    ]
    assert debugbench_importer._method_accepts_call_without_args(
        "class Solution:\n    def solve(self):\n        return 1\n",
        "solve",
    )
    assert not debugbench_importer._method_accepts_call_without_args(
        "class Solution:\n    def solve(self, nums):\n        return nums\n",
        "solve",
    )
    assert (
        debugbench_importer._infer_exception("syntax error", "indentation", "bad")
        == "IndentationError"
    )
    assert debugbench_importer._infer_exception("reference error", "", "bad") == "NameError"
    assert debugbench_importer._infer_exception("type error", "", "bad") == "TypeError"
    assert debugbench_importer._infer_exception("logic error", "", "bad") == "AssertionError"
    assert debugbench_importer._generate_test(
        code,
        ["Input: nums = [1, 2]\nOutput: 3\n"],
        "logic error",
    ).startswith("from buggy import Solution")
    assert "def test_case_0()" in debugbench_importer._generate_test(
        code,
        ["Input: nums = [1, 2]\nOutput: 3\n"],
        "logic error",
    )
    assert "sol.solve([1, 2])" in debugbench_importer._generate_test(
        code,
        ["Input: nums = [1, 2]\nOutput: 3\n"],
        "logic error",
    )
    assert "def test_smoke()" in debugbench_importer._generate_test(
        "class Solution:\n    def solve(self):\n        return 1\n",
        [],
        "reference error",
    )
    assert (
        debugbench_importer._generate_test(
            "class Solution:\n    def solve(self, nums):\n        return nums\n",
            [],
            "reference error",
        )
        is None
    )
    assert "def test_syntax()" in debugbench_importer._generate_test(
        "def broken(",
        None,
        "syntax error",
    )


def test_debugbench_import_dry_run_covers_main_paths(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {"language": "python2"},
        {
            "language": "python3",
            "buggy_code": "def broken(",
            "solution": "class Solution:\n    def solve(self):\n        return 1\n",
            "category": "syntax bug",
            "subtype": "indentation issue",
            "level": "easy",
            "examples": None,
        },
        {
            "language": "python3",
            "buggy_code": "class Solution:\n    def solve(self):\n        return value\n",
            "solution": "class Solution:\n    def solve(self):\n        return 1\n",
            "category": "reference bug",
            "subtype": "",
            "level": "hard",
            "examples": None,
        },
        {
            "language": "python3",
            "buggy_code": "class Solution:\n    def solve(self, nums):\n        return nums\n",
            "solution": "class Solution:\n    def solve(self, nums):\n        return nums + 1\n",
            "category": "logic bug",
            "subtype": "",
            "level": "invalid",
            "examples": ["Input: nums = [1, 2]\nOutput: 3\n"],
        },
        {
            "language": "python3",
            "buggy_code": "external code\n",
            "solution": "external code fixed\n",
            "category": "logic bug",
            "subtype": "",
            "level": "easy",
            "examples": None,
        },
        {
            "language": "python3",
            "buggy_code": "nondet code\n",
            "solution": "nondet code fixed\n",
            "category": "logic bug",
            "subtype": "",
            "level": "easy",
            "examples": None,
        },
        {
            "language": "python3",
            "buggy_code": "\n".join(
                ["class Solution:"]
                + ["    def solve(self):\n        return 1"]
                * (debugbench_importer.MAX_CODE_LINES + 1)
            ),
            "solution": "class Solution:\n    def solve(self):\n        return 2\n",
            "category": "logic bug",
            "subtype": "",
            "level": "easy",
            "examples": None,
        },
    ]
    _install_fake_datasets(monkeypatch, rows)
    monkeypatch.setattr(
        debugbench_importer,
        "map_debugbench_type",
        lambda bug_type: (
            "syntax_error"
            if "syntax" in bug_type
            else "reference error"
            if "reference" in bug_type
            else "logic_error"
        ),
    )
    monkeypatch.setattr(
        debugbench_importer, "infer_snapshot_dependency", lambda category: f"dep:{category}"
    )
    monkeypatch.setattr(debugbench_importer, "ruff_format", lambda code: code)
    monkeypatch.setattr(
        debugbench_importer,
        "has_external_deps",
        lambda code: "external" in code,
    )
    monkeypatch.setattr(
        debugbench_importer,
        "is_deterministic",
        lambda code: "nondet" not in code,
    )
    monkeypatch.setattr(
        debugbench_importer,
        "generate_oracle_patch",
        lambda buggy, fixed: "--- a/buggy.py\n+++ b/buggy.py\n-    return 1\n+    return 2\n",
    )
    monkeypatch.setattr(debugbench_importer, "count_patch_lines", lambda patch: 1)

    def _fake_generate_test(
        fixed_code: str, examples: list[str] | None, category: str
    ) -> str | None:
        if "syntax" in category:
            return "def test_syntax():\n    import buggy  # noqa: F401\n"
        if "reference" in category:
            return (
                "from buggy import Solution\n\n\n"
                "def test_smoke():\n"
                "    sol = Solution()\n"
                "    sol.solve()\n"
            )
        if examples:
            return (
                "from buggy import Solution\n\n\n"
                "def test_case_0():\n"
                "    sol = Solution()\n"
                "    result = sol.solve([1, 2])\n"
                "    assert result == 3\n"
            )
        return None

    monkeypatch.setattr(debugbench_importer, "_generate_test", _fake_generate_test)

    imported = debugbench_importer.import_debugbench(
        tmp_path / "cases", dry_run=True, validate=False
    )

    assert [case["case_id"] for case in imported] == [
        "db_0001_easy",
        "db_0002_hard",
        "db_0003_medium",
    ]
    assert imported[0]["metadata"]["expected_exception"] == "SyntaxError"
    assert imported[1]["metadata"]["expected_exception"] == "NameError"
    assert imported[2]["metadata"]["expected_exception"] == "AssertionError"
