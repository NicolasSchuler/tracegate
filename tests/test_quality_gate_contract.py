"""Tests that CI/CD workflows and quality-gate config stay in sync with pyproject.toml."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

from llmdebug import _json as json

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW_PATH = ".github/workflows/ci-cd.yml"
QUALITY_ACTION_PATH = ".github/actions/run-quality-contract/action.yml"
QUALITY_DEPENDENCIES = (
    "check",
    "lint-evals",
    "coverage-no-regression",
    "baseline-boundary-coverage",
    "deps",
    "audit",
    "docs-check",
    "dead-code",
    "complexity",
    "complexity-gate",
    "security",
    "type-coverage-strict",
    "dependency-sync",
)


def _read_repo_text(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _extract_toml_section(text: str, section_name: str) -> str:
    pattern = rf"^\[{re.escape(section_name)}\]\s*$\n(?P<body>.*?)(?=^\[.+\]\s*$|\Z)"
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, f"missing TOML section [{section_name}]"
    return match.group("body")


def _extract_cov_thresholds(text: str) -> set[int]:
    return {int(raw) for raw in re.findall(r"--cov-fail-under=(\d+)", text)}


def _extract_cov_commands(text: str) -> list[str]:
    return re.findall(r"^[^\n]*pytest[^\n]*--cov-fail-under=\d+[^\n]*$", text, flags=re.MULTILINE)


def _extract_xenon_tuples(text: str) -> set[tuple[str, str, str]]:
    tuples: set[tuple[str, str, str]] = set()
    for absolute, modules, average in re.findall(
        r"xenon\s+--max-absolute\s+([A-Z])\s+--max-modules\s+([A-Z])\s+--max-average\s+([A-Z])",
        text,
    ):
        tuples.add((absolute, modules, average))
    return tuples


def _extract_just_dependencies(text: str, recipe: str) -> tuple[str, ...]:
    match = re.search(rf"^{re.escape(recipe)}:(?P<body>[^\n]*)$", text, flags=re.MULTILINE)
    assert match is not None, f"missing just recipe: {recipe}"
    return tuple(match.group("body").split())


def _extract_just_body(text: str, recipe: str) -> str:
    match = re.search(
        rf"^{re.escape(recipe)}:\s*$\n(?P<body>(?:^[ \t]+.*(?:\n|\Z))*)",
        text,
        flags=re.MULTILINE,
    )
    assert match is not None, f"missing just recipe body: {recipe}"
    return match.group("body")


def test_coverage_threshold_and_flags_aligned_between_pyproject_and_justfile() -> None:
    pyproject = _read_repo_text("pyproject.toml")
    coverage_section = _extract_toml_section(pyproject, "tool.coverage.report")
    match = re.search(r"^\s*fail_under\s*=\s*(\d+)\s*$", coverage_section, flags=re.MULTILINE)
    assert match is not None, "missing coverage fail_under in pyproject"
    canonical_threshold = int(match.group(1))

    justfile = _read_repo_text("justfile")
    just_cov_commands = _extract_cov_commands(justfile)
    assert just_cov_commands, "missing pytest coverage commands in justfile"
    default_cov_commands = [
        cmd for cmd in just_cov_commands if "--cov={{ src }}" in cmd or "--cov=src/llmdebug" in cmd
    ]
    seam_cov_commands = [cmd for cmd in just_cov_commands if "--cov=baselines.adapter_base" in cmd]

    assert default_cov_commands, "missing default pytest coverage command"
    assert seam_cov_commands, "missing focused seam coverage command"
    assert _extract_cov_thresholds("\n".join(default_cov_commands)) == {canonical_threshold}
    assert _extract_cov_thresholds("\n".join(seam_cov_commands)) == {0}

    for command in just_cov_commands:
        assert "--cov-branch" in command
        assert "--cov-report=json" in command


def test_diff_cover_threshold_defined_once_in_justfile() -> None:
    justfile = _read_repo_text("justfile")

    just_thresholds = {
        int(raw)
        for raw in re.findall(r"diff-cover[^\n]*--fail-under=(\d+)", justfile, flags=re.MULTILINE)
    }
    assert len(just_thresholds) == 1, "expected exactly one diff-cover threshold in justfile"


def test_xenon_thresholds_defined_once_in_justfile() -> None:
    justfile = _read_repo_text("justfile")

    just_tuples = _extract_xenon_tuples(justfile)
    assert len(just_tuples) == 1, "expected one canonical xenon tuple in justfile"


def test_shared_quality_action_delegates_to_just_contract() -> None:
    shared_action = _read_repo_text(QUALITY_ACTION_PATH)
    assert "just quality" in shared_action
    assert "just diff-coverage" in shared_action
    assert shared_action.count("just test-evals-cov") == 1
    assert shared_action.count("just eval-coverage-no-regression") == 1
    assert "scripts/quality/check_coverage_no_regression.py" not in shared_action


def test_ci_workflow_uses_shared_quality_action() -> None:
    ci_workflow = _read_repo_text(CI_WORKFLOW_PATH)
    assert "uses: ./.github/actions/run-quality-contract" in ci_workflow


def test_release_deployment_workflow_is_not_configured() -> None:
    assert not (REPO_ROOT / ".github/workflows/release.yml").exists()


def test_just_quality_wires_coverage_no_regression_gate() -> None:
    justfile = _read_repo_text("justfile")
    assert "coverage-no-regression" in justfile
    assert "scripts/quality/check_coverage_no_regression.py" in justfile
    assert "--coverage-json coverage.json" in justfile
    assert "--baseline-json quality/coverage_baseline.json" in justfile


def test_just_quality_and_ci_membership_are_exact() -> None:
    justfile = _read_repo_text("justfile")

    assert _extract_just_dependencies(justfile, "quality") == QUALITY_DEPENDENCIES
    assert _extract_just_dependencies(justfile, "ci") == ("quality",)


def test_eval_coverage_measurement_and_ratchet_are_separate() -> None:
    justfile = _read_repo_text("justfile")
    measurement_recipe = _extract_just_body(justfile, "test-evals-cov")
    ratchet_recipe = _extract_just_body(justfile, "eval-coverage-no-regression")

    assert "--cov=evals" in measurement_recipe
    assert "--cov-branch" in measurement_recipe
    assert "--cov-report=json:coverage-evals.json" in measurement_recipe
    assert "check_coverage_no_regression.py" not in measurement_recipe
    assert ratchet_recipe.count("scripts/quality/check_coverage_no_regression.py") == 1
    assert "--coverage-json coverage-evals.json" in ratchet_recipe
    assert "--baseline-json quality/coverage_evals_baseline.json" in ratchet_recipe


def test_just_quality_wires_baseline_boundary_coverage_gate() -> None:
    justfile = _read_repo_text("justfile")
    assert re.search(r"^baseline-boundary-coverage:\s*$", justfile, flags=re.MULTILINE)
    assert "scripts/quality/check_boundary_module_coverage.py" in justfile
    assert "--cov=baselines.adapter_base" in justfile
    assert "--cov=evals.patchers.baseline_patcher" in justfile
    assert "--cov=evals.eval_patches" in justfile
    assert "--cov=evals.attempt_verification" in justfile
    quality_match = re.search(r"^quality:\s+(?P<body>.+)$", justfile, flags=re.MULTILINE)
    assert quality_match is not None
    assert "baseline-boundary-coverage" in quality_match.group("body")


def test_just_quality_wires_eval_lint_gate() -> None:
    justfile = _read_repo_text("justfile")
    assert re.search(r"^lint-evals:\s*$", justfile, flags=re.MULTILINE)
    assert "ruff check evals" in justfile
    assert "ruff format --check evals" in justfile
    quality_match = re.search(r"^quality:\s+(?P<body>.+)$", justfile, flags=re.MULTILINE)
    assert quality_match is not None
    assert "lint-evals" in quality_match.group("body")


def test_audit_recipe_documents_the_only_temporary_ignore() -> None:
    justfile = _read_repo_text("justfile")
    assert re.search(r"^audit:\s*$", justfile, flags=re.MULTILINE)
    assert "pip-audit" in justfile
    assert "--ignore-vuln GHSA-5239-wwwm-4pmq" in justfile
    assert "--ignore-vuln CVE-2026-4539" in justfile


def test_coverage_baseline_file_has_required_schema() -> None:
    baseline_path = REPO_ROOT / "quality" / "coverage_baseline.json"
    assert baseline_path.exists(), "missing quality/coverage_baseline.json"
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "line_percent" in payload
    assert "branch_percent" in payload
    assert "source" in payload
    assert "updated_at_utc" in payload
    assert isinstance(payload["line_percent"], (int, float))
    assert isinstance(payload["branch_percent"], (int, float))
    assert isinstance(payload["source"], str)
    assert isinstance(payload["updated_at_utc"], str)


def test_eval_coverage_baseline_file_has_required_schema() -> None:
    baseline_path = REPO_ROOT / "quality" / "coverage_evals_baseline.json"
    assert baseline_path.exists(), "missing quality/coverage_evals_baseline.json"
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert isinstance(payload.get("line_percent"), (int, float))
    assert isinstance(payload.get("branch_percent"), (int, float))
    assert isinstance(payload.get("source"), str)
    assert isinstance(payload.get("updated_at_utc"), str)


def test_type_coverage_baseline_file_has_required_schema() -> None:
    baseline_path = REPO_ROOT / "quality" / "type_coverage_baseline.json"
    assert baseline_path.exists(), "missing quality/type_coverage_baseline.json"
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "completeness_percent" in payload
    assert "exported_known" in payload
    assert "exported_ambiguous" in payload
    assert "exported_unknown" in payload
    assert "source" in payload
    assert "updated_at_utc" in payload
    assert isinstance(payload["completeness_percent"], (int, float))
    assert isinstance(payload["exported_known"], int)
    assert isinstance(payload["exported_ambiguous"], int)
    assert isinstance(payload["exported_unknown"], int)
    assert isinstance(payload["source"], str)
    assert isinstance(payload["updated_at_utc"], str)


def test_deptry_dep002_ignores_eval_only_analytics_dependencies() -> None:
    pyproject = _read_repo_text("pyproject.toml")
    deptry_ignores_section = _extract_toml_section(pyproject, "tool.deptry.per_rule_ignores")
    match = re.search(
        r"^\s*DEP002\s*=\s*\[(?P<body>.*?)\]\s*$",
        deptry_ignores_section,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "missing DEP002 ignore list in deptry config"
    dep002_ignored_packages = set(re.findall(r'"([^"]+)"', match.group("body")))
    assert {"pandas", "polars", "scikit-learn", "scipy", "seaborn"}.issubset(
        dep002_ignored_packages
    )


def test_evals_extra_declares_pydantic_for_direct_eval_imports() -> None:
    pyproject = _read_repo_text("pyproject.toml")
    optional_deps_section = _extract_toml_section(pyproject, "project.optional-dependencies")
    match = re.search(
        r"^\s*evals\s*=\s*\[(?P<body>.*?)\]\s*$",
        optional_deps_section,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "missing evals extra in pyproject"
    evals_dependencies = re.findall(r'"([^"]+)"', match.group("body"))
    assert any(dep.startswith("pydantic") for dep in evals_dependencies), (
        "evals extra must include pydantic because evals/stage_a_contract.py "
        "and evals/lambda_rlm_backend.py import it directly"
    )


def test_analytics_dependencies_do_not_bloat_the_core_install() -> None:
    payload = tomllib.loads(_read_repo_text("pyproject.toml"))
    core_dependencies = payload["project"]["dependencies"]
    optional_dependencies = payload["project"]["optional-dependencies"]

    for package in ("pandas", "polars", "seaborn"):
        assert not any(dep.startswith(package) for dep in core_dependencies)
    assert any(dep.startswith("pandas") for dep in optional_dependencies["analysis"])
    assert any(dep.startswith("seaborn") for dep in optional_dependencies["analysis"])
    assert any(dep.startswith("polars") for dep in optional_dependencies["evals"])


def test_just_quality_wires_type_coverage_regression_gate() -> None:
    justfile = _read_repo_text("justfile")
    assert "type-coverage-strict" in justfile
    assert "scripts/quality/check_type_coverage.py --strict" in justfile
    quality_match = re.search(r"^quality:\s+(?P<body>.+)$", justfile, flags=re.MULTILINE)
    assert quality_match is not None
    assert "type-coverage-strict" in quality_match.group("body")


def test_dev_dependency_lists_stay_in_sync() -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    optional_deps = payload["project"]["optional-dependencies"]["dev"]
    dependency_group = payload["dependency-groups"]["dev"]
    assert optional_deps == dependency_group


def test_agents_guide_is_canonical_without_claude_redirect() -> None:
    agents = _read_repo_text("AGENTS.md")

    assert "CLAUDE.md" not in agents


def test_issue_template_omits_placeholder_security_contact() -> None:
    issue_config = _read_repo_text(".github/ISSUE_TEMPLATE/config.yml")
    security_policy = _read_repo_text("SECURITY.md")

    assert "example.com" not in issue_config
    assert "Security Vulnerability" not in issue_config
    assert "Please do not open a public issue" in security_policy


def test_bench_regression_uses_distinct_compact_baseline_path() -> None:
    regression_script = _read_repo_text("scripts/quality/check_bench_regression.py")
    scripts_readme = _read_repo_text("scripts/README.md")
    justfile = _read_repo_text("justfile")
    pytest_benchmark_baseline = ".benchmarks/capture-overhead-baseline.json"
    compact_regression_baseline = ".benchmarks/capture-overhead-regression-baseline.json"

    assert f'DEFAULT_BASELINE = Path("{compact_regression_baseline}")' in regression_script
    assert f'DEFAULT_BASELINE = Path("{pytest_benchmark_baseline}")' not in regression_script
    assert f'base="{pytest_benchmark_baseline}"' in justfile
    assert compact_regression_baseline in scripts_readme
    assert pytest_benchmark_baseline in scripts_readme


def test_just_quality_wires_dev_dependency_sync_gate() -> None:
    justfile = _read_repo_text("justfile")
    assert re.search(r"^dependency-sync:\s*$", justfile, flags=re.MULTILINE)
    assert "scripts/quality/check_dev_dependency_sync.py" in justfile
    quality_match = re.search(r"^quality:\s+(?P<body>.+)$", justfile, flags=re.MULTILINE)
    assert quality_match is not None
    assert "dependency-sync" in quality_match.group("body")


def test_justfile_exposes_eval_typing_and_integration_test_recipes() -> None:
    justfile = _read_repo_text("justfile")
    assert re.search(r"^typecheck-evals:\s*$", justfile, flags=re.MULTILINE)
    assert "pyright -p pyrightconfig.evals.json" in justfile
    assert re.search(r"^test-integration:\s*$", justfile, flags=re.MULTILINE)
    assert re.search(r"^test-baseline-runtime:\s*$", justfile, flags=re.MULTILINE)
    assert 'pytest -m "integration and not baseline_runtime" -v' in justfile
    assert 'pytest -m "integration and baseline_runtime" -v' in justfile


def test_pyrightconfig_evals_exists_with_documented_module_set() -> None:
    config_text = _read_repo_text("pyrightconfig.evals.json")
    config = json.loads(config_text)
    assert config["typeCheckingMode"] == "basic"
    assert config["pythonVersion"] == "3.10"
    assert config["reportMissingTypeStubs"] is False
    # X-05: the whole harness is checked; new modules are in by default.
    assert config["include"] == ["evals", "baselines"]
    # Holdouts may only be removed. Each still has basic-mode errors to fix first.
    holdouts = {
        "evals/eval_attempt_flow.py",
        "evals/analysis_summary.py",
        "evals/analysis_adaptive_rollup.py",
        "evals/analysis_ope_estimators.py",
        "evals/analysis_paired_stats.py",
        "baselines/rlm/adapter.py",
        "baselines/lambda_rlm/adapter.py",
    }
    generated_or_external = {
        ".venv",
        "evals/artifacts",
        "evals/cases",
        "evals/results",
        "evals/swebench_runs",
        "evals/swebench_pro_runs",
        "evals/swebench_pro_assets",
        "evals/docker",
        "baselines/envs",
        "baselines/repos",
        "baselines/stubs",
        "tests",
    }
    excluded = set(config["exclude"])
    assert excluded - generated_or_external <= holdouts
    assert generated_or_external <= excluded


def test_default_test_recipes_skip_integration_marker() -> None:
    justfile = _read_repo_text("justfile")
    assert 'pytest -m "not integration"' in justfile
    assert 'pytest -m "integration and not baseline_runtime" -v' in justfile
    assert 'pytest -m "integration and baseline_runtime" -v' in justfile


def test_ci_workflow_surfaces_eval_typecheck_non_blocking() -> None:
    ci_workflow = _read_repo_text(CI_WORKFLOW_PATH)
    assert "Run eval static checks (informational)" in ci_workflow
    assert "continue-on-error: true" in ci_workflow
    assert "just typecheck-evals" in ci_workflow


def test_ci_workflow_runs_integration_lane() -> None:
    ci_workflow = _read_repo_text(CI_WORKFLOW_PATH)
    assert "Provision baseline repos and smoke envs" in ci_workflow
    assert "./baselines/setup_repos.sh" in ci_workflow
    assert 'BASELINE_ENV_MODE="smoke" bash baselines/setup_envs.sh' in ci_workflow
    assert "Run integration test lane" in ci_workflow
    assert 'PYTHON_VERSION="3.13" just test-integration' in ci_workflow
    assert ci_workflow.index("Provision baseline repos and smoke envs") < ci_workflow.index(
        "Run integration test lane"
    )


def test_ci_workflow_runs_baseline_runtime_lane_non_blocking() -> None:
    ci_workflow = _read_repo_text(CI_WORKFLOW_PATH)
    assert "Run baseline runtime integration lane (informational)" in ci_workflow
    assert 'PYTHON_VERSION="3.13" just test-baseline-runtime' in ci_workflow
    baseline_lane_index = ci_workflow.index("Run baseline runtime integration lane (informational)")
    assert "continue-on-error: true" in ci_workflow[baseline_lane_index:]
    assert ci_workflow.index("Run integration test lane") < baseline_lane_index


def test_docs_reference_staged_type_and_integration_commands() -> None:
    contributing = _read_repo_text("CONTRIBUTING.md")
    testing_guide = _read_repo_text("docs/testing-guide.md")
    tests_readme = _read_repo_text("tests/README.md")
    quality_map = _read_repo_text("docs/quality-map.md")
    quality_readme = _read_repo_text("quality/README.md")
    scripts_readme = _read_repo_text("scripts/README.md")

    for text in (contributing, testing_guide, tests_readme, quality_map):
        assert "test-integration" in text
        assert "test-baseline-runtime" in text
        assert "typecheck-evals" in text

    assert "type-coverage-strict" in contributing
    assert "quality/type_coverage_baseline.json" in quality_map
    for text in (contributing, testing_guide, tests_readme, quality_map, quality_readme):
        assert "baseline-boundary-coverage" in text
    assert "check_boundary_module_coverage.py" in scripts_readme


def test_quality_docs_match_committed_coverage_contract() -> None:
    pyproject = _read_repo_text("pyproject.toml")
    coverage_section = _extract_toml_section(pyproject, "tool.coverage.report")
    threshold_match = re.search(
        r"^\s*fail_under\s*=\s*(\d+)\s*$", coverage_section, flags=re.MULTILINE
    )
    assert threshold_match is not None
    threshold = int(threshold_match.group(1))

    package_baseline = json.loads(_read_repo_text("quality/coverage_baseline.json"))
    eval_baseline = json.loads(_read_repo_text("quality/coverage_evals_baseline.json"))
    assert isinstance(package_baseline, dict)
    assert isinstance(eval_baseline, dict)

    contributing = _read_repo_text("CONTRIBUTING.md")
    testing_guide = _read_repo_text("docs/testing-guide.md")
    quality_map = _read_repo_text("docs/quality-map.md")
    quality_readme = _read_repo_text("quality/README.md")

    assert f"line coverage floor `>= {threshold}`" in quality_map
    package_example = (
        f'"line_percent": {package_baseline["line_percent"]},\n'
        f'  "branch_percent": {package_baseline["branch_percent"]}'
    )
    assert package_example in contributing
    assert package_example in testing_guide

    eval_summary = (
        f"{eval_baseline['line_percent']}% line / {eval_baseline['branch_percent']}% branch"
    )
    for text in (contributing, testing_guide, quality_map, quality_readme):
        assert eval_summary in text

    integration_row = re.search(
        r"^\| `just test-integration` \|.*\| (?P<blocking>Yes|No) \|",
        quality_map,
        flags=re.MULTILINE,
    )
    assert integration_row is not None
    assert integration_row.group("blocking") == "Yes"
    assert "check_type_coverage.py --update-baseline" in quality_readme
    assert "check_type_coverage.py --update\n" not in quality_readme


def test_testing_guide_quality_membership_matches_justfile() -> None:
    testing_guide = _read_repo_text("docs/testing-guide.md")
    match = re.search(
        r"`just quality` has these direct recipe dependencies:\s*"
        r"```text\n(?P<body>.*?)\n```",
        testing_guide,
        flags=re.DOTALL,
    )
    assert match is not None, "testing guide must enumerate just quality membership"
    documented = tuple(
        line.removeprefix("just ") for line in match.group("body").splitlines() if line.strip()
    )

    assert documented == QUALITY_DEPENDENCIES
