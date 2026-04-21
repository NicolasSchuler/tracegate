"""Tests for the dataset importers CLI dispatch and argument parsing."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from evals.importers import cli as importers_cli
from evals.importers.top3_bootstrap import MaterializationReport, SourceMaterializationReport


def _install_fake_debugbench_importer(monkeypatch):
    calls: dict[str, object] = {}

    fake_module = types.ModuleType("evals.importers.debugbench_importer")

    def _fake_import_debugbench(output_dir, max_cases, validate, dry_run):
        calls["output_dir"] = output_dir
        calls["max_cases"] = max_cases
        calls["validate"] = validate
        calls["dry_run"] = dry_run
        return []

    fake_module.import_debugbench = _fake_import_debugbench
    monkeypatch.setitem(sys.modules, "evals.importers.debugbench_importer", fake_module)
    return calls


def _install_fake_top3_bootstrap(monkeypatch, *, hard_fail: bool):
    calls: dict[str, object] = {}

    fake_module = types.ModuleType("evals.importers.top3_bootstrap")

    class _Dataset:
        def __init__(self, source: str) -> None:
            self.source = source

    class _Manifest:
        def __init__(self) -> None:
            self.datasets = [_Dataset("bugsinpy"), _Dataset("pybughive"), _Dataset("trickybugs")]

    def _fake_prepare_top3_lock_manifest(manifest_path):
        calls["prepare_manifest_path"] = manifest_path
        return _Manifest()

    def _fake_materialize_top3(
        *,
        manifest_path,
        output_dir,
        data_root,
        download_policy,
        jobs,
        max_cases,
        validate,
        dry_run,
    ):
        calls["materialize_manifest_path"] = manifest_path
        calls["materialize_output_dir"] = output_dir
        calls["materialize_data_root"] = data_root
        calls["download_policy"] = download_policy
        calls["jobs"] = jobs
        calls["max_cases"] = max_cases
        calls["validate"] = validate
        calls["dry_run"] = dry_run
        report = MaterializationReport(
            schema_version=1,
            started_at_utc="2026-01-01T00:00:00+00:00",
            finished_at_utc="2026-01-01T00:00:01+00:00",
            manifest_path=str(manifest_path),
            data_root=str(data_root),
            output_dir=str(output_dir),
            download_policy=download_policy,
            max_cases=max_cases,
            validate=validate,
            dry_run=dry_run,
            jobs=jobs,
            sources=[
                SourceMaterializationReport(
                    source="bugsinpy",
                    attempted=3,
                    imported=2,
                    skipped=1,
                    hard_failed=False,
                    skip_reasons={"example_reason": 1},
                    imported_case_ids=["bip_a", "bip_b"],
                    notes=[],
                )
            ],
        )
        return report, Path("evals/artifacts/import_reports/top3-test.json")

    def _fake_report_has_hard_failures(report) -> bool:
        del report
        return hard_fail

    fake_module.prepare_top3_lock_manifest = _fake_prepare_top3_lock_manifest
    fake_module.materialize_top3 = _fake_materialize_top3
    fake_module.report_has_hard_failures = _fake_report_has_hard_failures
    monkeypatch.setitem(sys.modules, "evals.importers.top3_bootstrap", fake_module)
    return calls


def test_importers_cli_accepts_execution_flags_after_subcommand(monkeypatch) -> None:
    calls = _install_fake_debugbench_importer(monkeypatch)
    exit_code = importers_cli.main(["debugbench", "--executor", "local", "--dry-run"])
    assert exit_code == 0
    assert calls["dry_run"] is True
    assert isinstance(calls["output_dir"], Path)


def test_importers_cli_accepts_execution_flags_before_subcommand(monkeypatch) -> None:
    calls = _install_fake_debugbench_importer(monkeypatch)
    exit_code = importers_cli.main(["--executor", "local", "debugbench", "--dry-run"])
    assert exit_code == 0
    assert calls["dry_run"] is True
    assert isinstance(calls["output_dir"], Path)


def test_cli_registry_parser_supports_materialize_top3() -> None:
    parser = importers_cli._build_parser()
    parsed = parser.parse_args(["--executor", "local", "materialize-top3"])
    assert parsed.source == "materialize-top3"


def test_prepare_top3_writes_manifest_via_bootstrap(monkeypatch, tmp_path) -> None:
    calls = _install_fake_top3_bootstrap(monkeypatch, hard_fail=False)
    manifest_path = tmp_path / "top3.lock.json"
    exit_code = importers_cli.main(["prepare-top3", "--manifest", str(manifest_path)])
    assert exit_code == 0
    assert calls["prepare_manifest_path"] == manifest_path


def test_materialize_top3_parses_arguments_and_executes(monkeypatch, tmp_path) -> None:
    calls = _install_fake_top3_bootstrap(monkeypatch, hard_fail=False)
    monkeypatch.setattr(importers_cli, "check_pytest_available", lambda: (True, "ok"))

    exit_code = importers_cli.main(
        [
            "--executor",
            "local",
            "materialize-top3",
            "--manifest",
            str(tmp_path / "top3.lock.json"),
            "--output-dir",
            str(tmp_path / "cases"),
            "--data-root",
            str(tmp_path / "data"),
            "--download-policy",
            "never",
            "--jobs",
            "3",
            "--max-cases",
            "9",
            "--validate",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert calls["download_policy"] == "never"
    assert calls["jobs"] == 3
    assert calls["max_cases"] == 9
    assert calls["validate"] is True
    assert calls["dry_run"] is True
    assert calls["materialize_output_dir"] == tmp_path / "cases"
    assert calls["materialize_data_root"] == tmp_path / "data"


def test_materialize_top3_returns_nonzero_on_hard_failure(monkeypatch, tmp_path) -> None:
    _install_fake_top3_bootstrap(monkeypatch, hard_fail=True)
    monkeypatch.setattr(importers_cli, "check_pytest_available", lambda: (True, "ok"))

    exit_code = importers_cli.main(
        [
            "--executor",
            "local",
            "materialize-top3",
            "--manifest",
            str(tmp_path / "top3.lock.json"),
            "--output-dir",
            str(tmp_path / "cases"),
            "--data-root",
            str(tmp_path / "data"),
            "--download-policy",
            "never",
        ]
    )

    assert exit_code == 1


# ---------------------------------------------------------------------------
# Coverage gap tests: _print_case_import_summary dry_run preview (lines 37-43)
# ---------------------------------------------------------------------------


class TestPrintCaseImportSummary:
    """Cover the dry_run preview branch of _print_case_import_summary."""

    def test_dry_run_with_results_prints_preview(self, capsys) -> None:
        results = [{"case_id": f"case_{i}", "metadata": {"lang": "python"}} for i in range(3)]
        importers_cli._print_case_import_summary(
            source="test-source",
            dry_run=True,
            results=results,
        )
        captured = capsys.readouterr().out
        assert "3 case(s) previewed" in captured
        assert "Preview:" in captured
        assert "case_0:" in captured
        assert "case_2:" in captured

    def test_dry_run_more_than_five_results_shows_overflow(self, capsys) -> None:
        results = [{"case_id": f"case_{i}", "metadata": {}} for i in range(8)]
        importers_cli._print_case_import_summary(
            source="test-source",
            dry_run=True,
            results=results,
        )
        captured = capsys.readouterr().out
        assert "Preview:" in captured
        # Only the first 5 shown individually
        assert "case_4:" in captured
        assert "case_5:" not in captured
        assert "... and 3 more" in captured

    def test_dry_run_no_results_skips_preview(self, capsys) -> None:
        importers_cli._print_case_import_summary(
            source="src",
            dry_run=True,
            results=[],
        )
        captured = capsys.readouterr().out
        assert "0 case(s) previewed" in captured
        assert "Preview:" not in captured

    def test_non_dry_run_skips_preview(self, capsys) -> None:
        results = [{"case_id": "c1", "metadata": {}}]
        importers_cli._print_case_import_summary(
            source="src",
            dry_run=False,
            results=results,
        )
        captured = capsys.readouterr().out
        assert "1 case(s) imported" in captured
        assert "Preview:" not in captured


# ---------------------------------------------------------------------------
# Coverage gap tests: _handle_materialize_top3 validation (lines 89-93)
# ---------------------------------------------------------------------------


class TestMaterializeTop3ArgValidation:
    """Cover invalid --jobs and --max-cases in _handle_materialize_top3."""

    def test_jobs_less_than_one_returns_2(self, monkeypatch, capsys) -> None:
        _install_fake_top3_bootstrap(monkeypatch, hard_fail=False)
        exit_code = importers_cli.main(
            [
                "materialize-top3",
                "--jobs",
                "0",
            ]
        )
        assert exit_code == 2
        assert "Invalid --jobs" in capsys.readouterr().out

    def test_negative_max_cases_returns_2(self, monkeypatch, capsys) -> None:
        _install_fake_top3_bootstrap(monkeypatch, hard_fail=False)
        exit_code = importers_cli.main(
            [
                "materialize-top3",
                "--max-cases",
                "-1",
            ]
        )
        assert exit_code == 2
        assert "Invalid --max-cases" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Coverage gap tests: main() error paths (lines 340-361)
# ---------------------------------------------------------------------------


class TestMainErrorPaths:
    """Cover error paths in main(): ValueError, no source, unknown source,
    and validate environment check failures."""

    def test_invalid_execution_config_returns_2(self, monkeypatch, capsys) -> None:
        """ValueError from build_execution_config_from_args → exit 2."""
        monkeypatch.setattr(
            importers_cli,
            "build_execution_config_from_args",
            lambda _args: (_ for _ in ()).throw(ValueError("bad --executor")),
        )
        exit_code = importers_cli.main(["debugbench"])
        assert exit_code == 2
        assert "Invalid arguments" in capsys.readouterr().out

    def test_no_source_prints_help_and_returns_2(self, capsys) -> None:
        """No subcommand supplied → print_help + exit 2."""
        exit_code = importers_cli.main([])
        assert exit_code == 2

    def test_unknown_source_returns_2(self, monkeypatch, capsys) -> None:
        """Unknown subcommand name passed via Namespace → exit 2."""
        # Monkeypatch the parser to allow an unknown source through
        original_build = importers_cli._build_parser

        def _patched_build():
            parser = original_build()
            return parser

        monkeypatch.setattr(importers_cli, "_build_parser", _patched_build)
        # We inject source directly by patching parse_args
        import argparse

        class _FakeNamespace(argparse.Namespace):
            pass

        ns = _FakeNamespace(
            source="nonexistent-dataset",
            executor="local",
            container_image="img",
            container_network="bridge",
            container_cpus=1.0,
            container_memory="1g",
            container_pids_limit=100,
            container_user="1000:1000",
            container_tmpfs_size_mb=64,
            container_mount_repo_src=True,
            container_env=[],
        )
        monkeypatch.setattr(
            importers_cli,
            "_build_parser",
            lambda: type(
                "P",
                (),
                {
                    "parse_args": lambda self, argv=None: ns,
                    "print_help": lambda self: None,
                },
            )(),
        )
        exit_code = importers_cli.main(["nonexistent-dataset"])
        assert exit_code == 2
        assert "Unknown source" in capsys.readouterr().out

    def test_validate_local_env_check_fail_returns_2(
        self,
        monkeypatch,
        capsys,
    ) -> None:
        """--validate with local executor but check_pytest_available fails → exit 2."""
        _install_fake_debugbench_importer(monkeypatch)
        monkeypatch.setattr(
            importers_cli,
            "check_pytest_available",
            lambda: (False, "pytest not found"),
        )
        exit_code = importers_cli.main(
            [
                "--executor",
                "local",
                "debugbench",
                "--validate",
            ]
        )
        assert exit_code == 2
        assert "Environment check failed" in capsys.readouterr().out

    def test_validate_testcontainers_env_check_fail_returns_2(
        self,
        monkeypatch,
        capsys,
    ) -> None:
        """--validate with testcontainers but runtime check fails → exit 2."""
        _install_fake_debugbench_importer(monkeypatch)
        monkeypatch.setattr(
            importers_cli,
            "check_testcontainers_runtime_available",
            lambda: (False, "docker not found"),
        )
        exit_code = importers_cli.main(
            [
                "--executor",
                "testcontainers",
                "debugbench",
                "--validate",
            ]
        )
        assert exit_code == 2
        assert "Environment check failed" in capsys.readouterr().out
