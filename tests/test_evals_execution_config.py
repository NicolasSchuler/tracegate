"""Tests for eval execution configuration parsing and argument defaults."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import eval_config, eval_main_setup, lambda_rlm_backend, run_eval
from evals.eval_types import ConfigResolutionState
from llmdebug import _json as json


def _make_synthetic_case(repo_root: Path, case_id: str) -> run_eval.CaseSpec:
    case_dir = repo_root / "synthetic_cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return run_eval.CaseSpec(
        case_id=case_id,
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )


def test_execution_config_defaults_to_testcontainers() -> None:
    parser = argparse.ArgumentParser()
    run_eval.add_execution_arguments(parser)
    args = parser.parse_args([])
    config = run_eval.build_execution_config_from_args(args)
    assert config.executor == "testcontainers"
    assert config.container_image


def test_execution_config_rejects_invalid_container_env() -> None:
    parser = argparse.ArgumentParser()
    run_eval.add_execution_arguments(parser)
    args = parser.parse_args(["--container-env", "INVALID"])
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        run_eval.build_execution_config_from_args(args)


def test_run_eval_cli_defaults_match_repo_eval_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "cli_defaults_case")

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="null"))
    monkeypatch.setattr(
        run_eval,
        "run_one_condition",
        lambda *, case, condition, run_id, run_metadata, **_kwargs: [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ],
    )

    run_id = "cli-defaults"
    exit_code = run_eval.main(
        [
            "--run-id",
            run_id,
            "--patcher",
            "null",
            "--max-cases",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    lock_path = repo_root / "evals" / "artifacts" / "configs" / f"{run_id}.resolved.json"
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["k"] == 2
    assert lock_payload["resolved_args"]["jobs"] == 3
    assert lock_payload["resolved_args"]["openai_base_url"] == "http://127.0.0.1:12000/v1"
    assert lock_payload["resolved_args"]["syntax_preflight_fallthrough"] is True


def test_run_eval_main_propagates_syntax_preflight_fallthrough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "cli_syntax_preflight_fallthrough_case")
    observed: dict[str, object] = {}

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="null"))

    def _fake_run_one_condition(
        *,
        case,
        condition,
        run_id,
        run_metadata,
        syntax_preflight_fallthrough,
        **_kwargs,
    ) -> list[dict[str, object]]:
        observed["syntax_preflight_fallthrough"] = syntax_preflight_fallthrough
        observed["patcher_config"] = dict(run_metadata.get("patcher_config") or {})
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _fake_run_one_condition)

    run_id = "cli-no-syntax-preflight-fallthrough"
    exit_code = run_eval.main(
        [
            "--run-id",
            run_id,
            "--patcher",
            "null",
            "--max-cases",
            "1",
            "--jobs",
            "1",
            "--executor",
            "local",
            "--no-syntax-preflight-fallthrough",
        ]
    )

    assert exit_code == 0
    assert observed["syntax_preflight_fallthrough"] is False
    patcher_config = observed["patcher_config"]
    assert isinstance(patcher_config, dict)
    assert patcher_config["syntax_preflight_fallthrough"] is False
    lock_path = repo_root / "evals" / "artifacts" / "configs" / f"{run_id}.resolved.json"
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["syntax_preflight_fallthrough"] is False


def test_run_eval_warns_on_unique_similar_existing_run_id(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    existing_run_id = "intervenor-20260326T205854"
    requested_run_id = "intervenor-20260326T20584"
    results_root = repo_root / "evals" / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / f"{existing_run_id}.jsonl").write_text("", encoding="utf-8")
    startup_warnings: list[str] = []

    eval_main_setup._append_similar_run_id_warning(
        run_id=requested_run_id,
        results_path=results_root / f"{requested_run_id}.jsonl",
        startup_warnings=startup_warnings,
    )

    assert startup_warnings == [
        "No existing results file found for "
        f"--run-id {requested_run_id!r}. Found similar existing run-id {existing_run_id!r}; "
        "if you intended to resume it, use the exact run-id."
    ]


@pytest.mark.parametrize("scope", ["disabled", "adaptive_only", "all"])
def test_run_eval_main_propagates_patch_verifier_scope_and_min_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, f"cli_patch_verifier_scope_{scope}")
    observed: dict[str, object] = {}

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="null"))

    def _fake_run_one_condition(
        *,
        case,
        condition,
        run_id,
        run_metadata,
        patch_verifier_scope,
        patch_verifier_min_score,
        **_kwargs,
    ) -> list[dict[str, object]]:
        observed["patch_verifier_scope"] = patch_verifier_scope
        observed["patch_verifier_min_score"] = patch_verifier_min_score
        observed["patcher_config"] = dict(run_metadata.get("patcher_config") or {})
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _fake_run_one_condition)

    run_id = f"cli-patch-verifier-{scope}"
    exit_code = run_eval.main(
        [
            "--run-id",
            run_id,
            "--patcher",
            "null",
            "--max-cases",
            "1",
            "--jobs",
            "1",
            "--executor",
            "local",
            "--patch-verifier-scope",
            scope,
            "--patch-verifier-min-score",
            "3",
        ]
    )

    assert exit_code == 0
    assert observed["patch_verifier_scope"] == scope
    assert observed["patch_verifier_min_score"] == 3
    patcher_config = observed["patcher_config"]
    assert isinstance(patcher_config, dict)
    assert patcher_config["patch_verifier_scope"] == scope
    assert patcher_config["patch_verifier_min_score"] == 3
    lock_path = repo_root / "evals" / "artifacts" / "configs" / f"{run_id}.resolved.json"
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["patch_verifier_scope"] == scope
    assert lock_payload["resolved_args"]["patch_verifier_min_score"] == 3


def test_run_eval_main_rejects_negative_patch_verifier_min_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "cli_negative_patch_verifier_min_score_case")

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="null"))

    exit_code = run_eval.main(
        [
            "--run-id",
            "cli-negative-patch-verifier-min-score",
            "--patcher",
            "null",
            "--max-cases",
            "1",
            "--executor",
            "local",
            "--patch-verifier-min-score",
            "-1",
        ]
    )

    assert exit_code == 2


def test_resolve_conditions_for_baseline_defaults_to_matching_baseline() -> None:
    args = argparse.Namespace(
        patcher="baseline",
        baseline_name="ldb",
        conditions="base_repair",
    )
    config_state = ConfigResolutionState(value_sources={"conditions": "default"})

    conditions = run_eval._resolve_conditions_for_run(args=args, config_state=config_state)

    assert conditions == [run_eval.Condition.LDB]
    assert args.conditions == "ldb"


def test_resolve_conditions_for_baseline_requires_baseline_name() -> None:
    args = argparse.Namespace(
        patcher="baseline",
        baseline_name="",
        conditions="base_repair",
    )
    config_state = ConfigResolutionState(value_sources={"conditions": "cli"})

    with pytest.raises(ValueError, match="--baseline-name is required"):
        run_eval._resolve_conditions_for_run(args=args, config_state=config_state)


def test_resolve_conditions_for_baseline_rejects_mismatch() -> None:
    args = argparse.Namespace(
        patcher="baseline",
        baseline_name="ldb",
        conditions="intervenor",
    )
    config_state = ConfigResolutionState(value_sources={"conditions": "cli"})

    with pytest.raises(ValueError, match="matching baseline condition or one adaptive condition"):
        run_eval._resolve_conditions_for_run(args=args, config_state=config_state)


def test_resolve_conditions_for_baseline_rejects_multiple_conditions() -> None:
    args = argparse.Namespace(
        patcher="baseline",
        baseline_name="ldb",
        conditions="layered_adaptive_gated,layered_adaptive_llm",
    )
    config_state = ConfigResolutionState(value_sources={"conditions": "cli"})

    with pytest.raises(ValueError, match="requires exactly one condition"):
        run_eval._resolve_conditions_for_run(args=args, config_state=config_state)


@pytest.mark.parametrize("baseline_name", ["intervenor", "rlm"])
def test_resolve_conditions_for_baseline_allows_adaptive_hybrid_condition(
    baseline_name: str,
) -> None:
    args = argparse.Namespace(
        patcher="baseline",
        baseline_name=baseline_name,
        conditions="layered_adaptive_llm",
    )
    config_state = ConfigResolutionState(value_sources={"conditions": "cli"})

    conditions = run_eval._resolve_conditions_for_run(args=args, config_state=config_state)

    assert conditions == [run_eval.Condition.LAYERED_ADAPTIVE_LLM]


def test_resolve_conditions_for_native_cot_baseline_defaults_to_cot_repair() -> None:
    args = argparse.Namespace(
        patcher="openai",
        native_baseline_mode="cot_repair",
        conditions="base_repair",
    )
    config_state = ConfigResolutionState(value_sources={"conditions": "default"})

    conditions = run_eval._resolve_conditions_for_run(args=args, config_state=config_state)

    assert conditions == [run_eval.Condition.COT_REPAIR]
    assert args.conditions == "cot_repair"


@pytest.mark.parametrize(
    ("condition_name", "expected"),
    [
        ("cot_repair", run_eval.Condition.COT_REPAIR),
        ("layered_adaptive_gated", run_eval.Condition.LAYERED_ADAPTIVE_GATED),
        ("layered_adaptive_llm", run_eval.Condition.LAYERED_ADAPTIVE_LLM),
    ],
)
def test_resolve_conditions_for_native_cot_baseline_allows_supported_conditions(
    condition_name: str,
    expected: run_eval.Condition,
) -> None:
    args = argparse.Namespace(
        patcher="openai",
        native_baseline_mode="cot_repair",
        conditions=condition_name,
    )
    config_state = ConfigResolutionState(value_sources={"conditions": "cli"})

    conditions = run_eval._resolve_conditions_for_run(args=args, config_state=config_state)

    assert conditions == [expected]


@pytest.mark.parametrize(
    ("conditions", "match"),
    [
        ("base_repair", "requires one of"),
        ("layered_adaptive_llm,layered_adaptive_gated", "requires exactly one condition"),
    ],
)
def test_resolve_conditions_for_native_cot_baseline_rejects_invalid_conditions(
    conditions: str,
    match: str,
) -> None:
    args = argparse.Namespace(
        patcher="openai",
        native_baseline_mode="cot_repair",
        conditions=conditions,
    )
    config_state = ConfigResolutionState(value_sources={"conditions": "cli"})

    with pytest.raises(ValueError, match=match):
        run_eval._resolve_conditions_for_run(args=args, config_state=config_state)


def test_baseline_patch_inventory_reads_sorted_patch_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    patch_dir = repo_root / "baselines" / "ldb" / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    (patch_dir / "b.patch").write_text("diff b", encoding="utf-8")
    (patch_dir / "a.patch").write_text("diff a", encoding="utf-8")
    (patch_dir / "ignore.txt").write_text("skip", encoding="utf-8")

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    assert run_eval._baseline_patch_inventory("LDB") == ["a.patch", "b.patch"]


def test_autodetect_baseline_python_returns_empty_for_blank_name() -> None:
    assert run_eval._autodetect_baseline_python("   ") == ""


def test_normalize_baseline_selection_rejects_conflicting_patcher() -> None:
    args = argparse.Namespace(
        patcher="openai",
        baseline="ldb",
        baseline_name="",
        baseline_repo_dir="",
        baseline_python="",
    )

    with pytest.raises(ValueError, match="cannot be combined with --patcher"):
        run_eval._normalize_baseline_selection(args=args, config_state=None)


def test_normalize_baseline_selection_rejects_conflicting_baseline_name() -> None:
    args = argparse.Namespace(
        patcher="command",
        baseline="ldb",
        baseline_name="rlm",
        baseline_repo_dir="",
        baseline_python="",
    )

    with pytest.raises(ValueError, match="conflicts with --baseline-name"):
        run_eval._normalize_baseline_selection(args=args, config_state=None)


def test_normalize_baseline_selection_requires_baseline_name_for_baseline_patcher() -> None:
    args = argparse.Namespace(
        patcher="baseline",
        baseline="",
        baseline_name="",
        baseline_repo_dir="",
        baseline_python="",
    )

    with pytest.raises(ValueError, match="--baseline-name is required"):
        run_eval._normalize_baseline_selection(args=args, config_state=None)


def test_normalize_baseline_selection_tracecoder_sets_default_repo_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    args = argparse.Namespace(
        patcher="command",
        baseline="tracecoder",
        baseline_name="",
        baseline_repo_dir="",
        baseline_python="",
    )

    run_eval._normalize_baseline_selection(args=args, config_state=None)

    assert args.patcher == "baseline"
    assert args.baseline_name == "tracecoder"
    assert args.baseline_repo_dir == str(repo_root / "baselines" / "repos" / "TraceCoder")
    assert args.baseline_python == ""


def test_normalize_baseline_selection_rlm_sets_default_repo_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    args = argparse.Namespace(
        patcher="command",
        baseline="rlm",
        baseline_name="",
        baseline_repo_dir="",
        baseline_python="",
    )

    run_eval._normalize_baseline_selection(args=args, config_state=None)

    assert args.patcher == "baseline"
    assert args.baseline_name == "rlm"
    assert args.baseline_repo_dir == str(repo_root / "baselines" / "repos" / "rlm")
    assert args.baseline_python == ""


def test_normalize_adaptive_controller_lambda_rlm_settings_rewrites_stale_repo_local_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_dir = repo_root / "baselines" / "repos" / "lambda-RLM"
    repo_dir.mkdir(parents=True, exist_ok=True)
    python_executable = repo_root / "baselines" / "envs" / "lambda_rlm" / "bin" / "python"
    python_executable.parent.mkdir(parents=True, exist_ok=True)
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)
    stale_root = tmp_path / "stale-checkout"

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    args = argparse.Namespace(
        adaptive_controller_lambda_rlm=True,
        adaptive_controller_lambda_rlm_repo_dir=str(
            stale_root / "baselines" / "repos" / "lambda-RLM"
        ),
        adaptive_controller_lambda_rlm_python=str(
            stale_root / "baselines" / "envs" / "lambda_rlm" / "bin" / "python"
        ),
    )

    run_eval._normalize_adaptive_controller_lambda_rlm_settings(
        args=args,
        config_state=None,
    )

    assert args.adaptive_controller_lambda_rlm_repo_dir == str(repo_dir)
    assert args.adaptive_controller_lambda_rlm_python == str(python_executable)


def test_normalize_baseline_selection_cot_repair_sets_native_openai_mode() -> None:
    args = argparse.Namespace(
        patcher="",
        baseline="cot_repair",
        baseline_name="",
        baseline_repo_dir="",
        baseline_python="",
    )
    config_state = ConfigResolutionState(value_sources={"baseline": "cli"})

    run_eval._normalize_baseline_selection(args=args, config_state=config_state)

    assert args.patcher == "openai"
    assert args.native_baseline_mode == "cot_repair"
    assert args.baseline_name == ""
    assert args.baseline_repo_dir == ""
    assert args.baseline_python == ""
    assert config_state.value_sources["patcher"] == "derived_from_baseline:cli"
    assert config_state.value_sources["native_baseline_mode"] == "derived_from_baseline:cli"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"patcher": "baseline"}, "omit --patcher or use --patcher openai"),
        ({"baseline_name": "ldb"}, "cannot be combined with --baseline-name"),
        ({"baseline_repo_dir": "/tmp/baseline"}, "cannot be combined with --baseline-repo-dir"),
        ({"baseline_python": "/tmp/python"}, "cannot be combined with --baseline-python"),
    ],
)
def test_normalize_baseline_selection_cot_repair_rejects_external_baseline_options(
    kwargs: dict[str, str],
    match: str,
) -> None:
    args_payload = {
        "patcher": "",
        "baseline": "cot_repair",
        "baseline_name": "",
        "baseline_repo_dir": "",
        "baseline_python": "",
        "native_baseline_mode": "",
    }
    args_payload.update(kwargs)
    args = argparse.Namespace(**args_payload)

    with pytest.raises(ValueError, match=match):
        run_eval._normalize_baseline_selection(args=args, config_state=None)


def test_normalize_adaptive_controller_lambda_rlm_settings_autofills_repo_and_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "lambda-RLM"
    repo_dir.mkdir()
    python_executable = tmp_path / "lambda-python"
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)
    args = argparse.Namespace(
        adaptive_controller_lambda_rlm=True,
        adaptive_controller_lambda_rlm_repo_dir="",
        adaptive_controller_lambda_rlm_python="",
    )
    config_state = ConfigResolutionState(
        value_sources={
            "adaptive_controller_lambda_rlm_repo_dir": "default",
            "adaptive_controller_lambda_rlm_python": "default",
        }
    )

    monkeypatch.setattr(
        lambda_rlm_backend,
        "default_lambda_rlm_repo_dir",
        lambda: repo_dir,
    )
    monkeypatch.setattr(
        run_eval,
        "_autodetect_lambda_rlm_python",
        lambda: str(python_executable),
    )

    run_eval._normalize_adaptive_controller_lambda_rlm_settings(
        args=args,
        config_state=config_state,
    )

    assert args.adaptive_controller_lambda_rlm_repo_dir == str(repo_dir)
    assert args.adaptive_controller_lambda_rlm_python == str(python_executable)
    assert (
        config_state.value_sources["adaptive_controller_lambda_rlm_repo_dir"]
        == "derived_from_adaptive_controller_lambda_rlm"
    )
    assert (
        config_state.value_sources["adaptive_controller_lambda_rlm_python"]
        == "derived_from_adaptive_controller_lambda_rlm"
    )


def test_normalize_adaptive_controller_lambda_rlm_settings_uses_manifest_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external_repo_dir = tmp_path / "external-repos" / "lambda-RLM"
    external_repo_dir.mkdir(parents=True)
    external_python = tmp_path / "external-envs" / "lambda_rlm" / "bin" / "python"
    external_python.parent.mkdir(parents=True, exist_ok=True)
    external_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    external_python.chmod(0o755)
    manifest_path = repo_root / "baselines" / "envs" / "runtime_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "lambda_rlm": {
                    "repo_dir": str(external_repo_dir),
                    "repo_dir_repo_relative": "baselines/repos/lambda-RLM",
                    "python_executable": str(external_python),
                    "python_repo_relative": "baselines/envs/lambda_rlm/bin/python",
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        adaptive_controller_lambda_rlm=True,
        adaptive_controller_lambda_rlm_repo_dir="",
        adaptive_controller_lambda_rlm_python="",
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(
        lambda_rlm_backend,
        "default_lambda_rlm_repo_dir",
        lambda: repo_root / "baselines" / "repos" / "lambda-RLM",
    )
    monkeypatch.setattr(run_eval, "_autodetect_lambda_rlm_python", lambda: "")

    run_eval._normalize_adaptive_controller_lambda_rlm_settings(
        args=args,
        config_state=None,
    )

    assert args.adaptive_controller_lambda_rlm_repo_dir == str(external_repo_dir)
    assert args.adaptive_controller_lambda_rlm_python == str(external_python)


@pytest.mark.parametrize(
    ("baseline_name", "timeout_attr"),
    [
        ("lambda_rlm", "baseline_lambda_timeout_sec"),
        ("rlm", "baseline_rlm_timeout_sec"),
    ],
)
def test_baseline_extra_from_args_omits_raw_timeout_when_unset(
    baseline_name: str,
    timeout_attr: str,
) -> None:
    args = argparse.Namespace(
        baseline_openai_compat_profile="auto",
        baseline_openai_allow_reasoning_fallback=False,
        baseline_openai_enable_thinking=False,
        baseline_lambda_context_window_chars=run_eval.DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
        baseline_lambda_accuracy_target=0.80,
        baseline_lambda_a_leaf=0.95,
        baseline_lambda_a_compose=0.90,
        baseline_lambda_verbose=False,
        baseline_lambda_stream=False,
        baseline_lambda_temperature=None,
        baseline_lambda_top_p=None,
        baseline_lambda_max_tokens=None,
        baseline_lambda_timeout_sec=None,
        baseline_rlm_max_depth=run_eval.DEFAULT_BASELINE_RLM_MAX_DEPTH,
        baseline_rlm_max_iterations=run_eval.DEFAULT_BASELINE_RLM_MAX_ITERATIONS,
        baseline_rlm_timeout_sec=None,
        baseline_rlm_max_errors=run_eval.DEFAULT_BASELINE_RLM_MAX_ERRORS,
        baseline_rlm_compaction=False,
        baseline_rlm_compaction_threshold_pct=(
            run_eval.DEFAULT_BASELINE_RLM_COMPACTION_THRESHOLD_PCT
        ),
        baseline_rlm_verbose=False,
        baseline_rlm_temperature=None,
        baseline_rlm_top_p=None,
        baseline_rlm_max_tokens=None,
        baseline_rlm_max_total_tokens=None,
    )

    extra = run_eval._baseline_extra_from_args(args=args, baseline_name=baseline_name)

    assert timeout_attr not in extra
    assert extra["baseline_openai_compat_profile"] == "auto"
    assert extra["baseline_openai_allow_reasoning_fallback"] is False


def test_baseline_extra_from_args_includes_lambda_force_task_type_when_set() -> None:
    args = argparse.Namespace(
        baseline_openai_compat_profile="auto",
        baseline_openai_allow_reasoning_fallback=False,
        baseline_openai_enable_thinking=False,
        baseline_lambda_context_window_chars=run_eval.DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
        baseline_lambda_accuracy_target=0.80,
        baseline_lambda_a_leaf=0.95,
        baseline_lambda_a_compose=0.90,
        baseline_lambda_verbose=False,
        baseline_lambda_stream=False,
        baseline_lambda_temperature=None,
        baseline_lambda_top_p=None,
        baseline_lambda_max_tokens=None,
        baseline_lambda_timeout_sec=None,
        baseline_lambda_force_task_type="qa",
        baseline_lambda_controller_guidance_max_chars=2500,
        baseline_lambda_failure_feedback_max_chars=3500,
        baseline_lambda_failure_feedback_meta_max_chars=2000,
    )

    extra = run_eval._baseline_extra_from_args(args=args, baseline_name="lambda_rlm")

    assert extra["baseline_lambda_force_task_type"] == "qa"


def test_baseline_extra_from_args_thinking_auto_enables_reasoning_fallback() -> None:
    args = argparse.Namespace(
        baseline_openai_compat_profile="lmstudio_model_yaml",
        baseline_openai_allow_reasoning_fallback=False,
        baseline_openai_enable_thinking=True,
        baseline_hybrid_guidance_mode="tuned_v1",
    )

    extra = run_eval._baseline_extra_from_args(args=args, baseline_name="ldb")

    assert extra["baseline_openai_compat_profile"] == "lmstudio_model_yaml"
    assert extra["baseline_openai_enable_thinking"] is True
    assert extra["baseline_openai_allow_reasoning_fallback"] is True
    assert extra["baseline_hybrid_guidance_mode"] == "tuned_v1"


@pytest.mark.parametrize(
    ("baseline_name", "repo_name"),
    [
        ("ldb", "LLMDebugger"),
        ("intervenor", "INTERVENOR"),
    ],
)
def test_run_eval_baseline_shorthand_records_external_baseline_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    baseline_name: str,
    repo_name: str,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "baselines" / "repos" / repo_name).mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / f"{baseline_name}_baseline_shorthand_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    env_python = repo_root / "baselines" / "envs" / baseline_name / "bin" / "python"
    env_python.parent.mkdir(parents=True, exist_ok=True)
    env_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    env_python.chmod(0o755)

    case = run_eval.CaseSpec(
        case_id=f"{baseline_name}_baseline_shorthand_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name=baseline_name))

    captured: dict[str, object] = {}

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        captured["condition"] = condition.value
        captured["run_metadata"] = dict(run_metadata)
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "baseline-shorthand",
            "--baseline",
            baseline_name,
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    assert captured["condition"] == baseline_name
    run_metadata = captured["run_metadata"]
    assert isinstance(run_metadata, dict)
    patcher_config = run_metadata.get("patcher_config")
    assert isinstance(patcher_config, dict)
    assert patcher_config["baseline_name"] == baseline_name
    assert patcher_config["baseline_condition"] == baseline_name
    assert patcher_config["baseline_repo_dir"] == str(repo_root / "baselines" / "repos" / repo_name)
    assert patcher_config["baseline_python"] == str(env_python)
    assert patcher_config["native_baseline_mode"] == ""
    lock_path = repo_root / "evals" / "artifacts" / "configs" / "baseline-shorthand.resolved.json"
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["conditions"] == baseline_name


def test_run_eval_native_cot_baseline_shorthand_records_resolved_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "native_cot_baseline_case")
    captured: dict[str, object] = {}

    monkeypatch.setenv("OPENAI_API_KEY", "test-token")
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="openai"))

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        captured["condition"] = condition.value
        captured["run_metadata"] = dict(run_metadata)
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    run_id = "native-cot-baseline"
    exit_code = run_eval.main(
        [
            "--run-id",
            run_id,
            "--baseline",
            "cot_repair",
            "--openai-base-url",
            "http://127.0.0.1:12000/v1",
            "--openai-model",
            "dummy-model",
            "--max-cases",
            "1",
            "--k",
            "1",
            "--jobs",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    assert captured["condition"] == "cot_repair"
    run_metadata = captured["run_metadata"]
    assert isinstance(run_metadata, dict)
    patcher_config = run_metadata.get("patcher_config")
    assert isinstance(patcher_config, dict)
    assert patcher_config["patcher"] == "openai"
    assert patcher_config["native_baseline_mode"] == "cot_repair"
    assert patcher_config["resolved_condition"] == "cot_repair"
    assert "baseline_name" not in patcher_config

    lock_path = repo_root / "evals" / "artifacts" / "configs" / f"{run_id}.resolved.json"
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["native_baseline_mode"] == "cot_repair"
    assert lock_payload["resolved_args"]["conditions"] == "cot_repair"


def test_run_eval_native_cot_baseline_shorthand_supports_layered_adaptive_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "native_cot_baseline_adaptive_case")
    captured: dict[str, object] = {}

    monkeypatch.setenv("OPENAI_API_KEY", "test-token")
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="openai"))

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        captured["condition"] = condition.value
        captured["run_metadata"] = dict(run_metadata)
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    run_id = "native-cot-baseline-adaptive"
    exit_code = run_eval.main(
        [
            "--run-id",
            run_id,
            "--baseline",
            "cot_repair",
            "--conditions",
            "layered_adaptive_llm",
            "--openai-base-url",
            "http://127.0.0.1:12000/v1",
            "--openai-model",
            "dummy-model",
            "--max-cases",
            "1",
            "--k",
            "1",
            "--jobs",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    assert captured["condition"] == "layered_adaptive_llm"
    run_metadata = captured["run_metadata"]
    assert isinstance(run_metadata, dict)
    patcher_config = run_metadata.get("patcher_config")
    assert isinstance(patcher_config, dict)
    assert patcher_config["patcher"] == "openai"
    assert patcher_config["native_baseline_mode"] == "cot_repair"
    assert patcher_config["resolved_condition"] == "layered_adaptive_llm"

    lock_path = repo_root / "evals" / "artifacts" / "configs" / f"{run_id}.resolved.json"
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["native_baseline_mode"] == "cot_repair"
    assert lock_payload["resolved_args"]["conditions"] == "layered_adaptive_llm"


def test_run_eval_main_parses_lambda_force_task_type_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "baselines" / "repos" / "lambda-RLM").mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "lambda_force_task_type_case")
    env_python = repo_root / "baselines" / "envs" / "lambda_rlm" / "bin" / "python"
    env_python.parent.mkdir(parents=True, exist_ok=True)
    env_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    env_python.chmod(0o755)
    assert env_python.is_file()
    observed: dict[str, object] = {}

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])

    def _make_patcher(args: argparse.Namespace):
        observed["baseline_lambda_force_task_type"] = args.baseline_lambda_force_task_type
        return SimpleNamespace(name="baseline")

    monkeypatch.setattr(run_eval, "make_patcher", _make_patcher)
    monkeypatch.setattr(
        run_eval,
        "run_one_condition",
        lambda *, case, condition, run_id, run_metadata, **_kwargs: [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ],
    )

    exit_code = run_eval.main(
        [
            "--run-id",
            "lambda-force-task-type",
            "--baseline",
            "lambda_rlm",
            "--baseline-lambda-force-task-type",
            "qa",
            "--max-cases",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    assert observed["baseline_lambda_force_task_type"] == "qa"


def test_execute_eval_unit_streams_layered_adaptive_llm_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_synthetic_case(tmp_path, "adaptive_stream_case")
    streamed: list[dict[str, object]] = []

    def _fake_run_one_condition(**kwargs):
        record = {
            "run_id": "stream-run",
            "case_id": case.case_id,
            "condition": kwargs["condition"].value,
            "attempt": 1,
            "success": False,
            "note": "tests_still_failing",
            "patch_meta": {},
            "attempt_metadata": {},
        }
        kwargs["record_sink"](record)
        return [record]

    monkeypatch.setattr(run_eval, "run_one_condition", _fake_run_one_condition)

    outcome = run_eval._execute_eval_unit(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        artifacts_root=tmp_path / "artifacts",
        inputs=run_eval.EvalUnitInputs(
            patcher_args=argparse.Namespace(),
            patcher_override=SimpleNamespace(name="intervenor"),
            keep_workdirs=False,
            progress_verbosity="off",
            progress_heartbeat_sec=0.0,
            run_condition=run_eval.RunConditionInputs(
                patcher=None,
                k=1,
                output_format="diff",
                run_id="stream-run",
                include_files_context=False,
                max_context_chars=200_000,
                case_fail_fast_on_request_budget_exceeded=False,
                force_patch_attempt=False,
                context_protocol="monolithic",
                context_stage_a_max_requests=6,
                context_stage_a_max_chars=120_000,
                retry_prompt_mode="auto",
                retry_context_mode="history",
                retry_history_max_chars=48_000,
                prompt_retry_reflection=False,
                prompt_adaptive_stage_b_cot=False,
                prompt_repair_method_scaffold=False,
                prompt_retry_plan_then_patch=False,
                pytest_output_tail_lines=run_eval.DEFAULT_PYTEST_OUTPUT_TAIL_LINES,
                snapshot_artifact_max_string_chars=(
                    run_eval.DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS
                ),
                prompt_budget_total_chars=180_000,
                prompt_budget_retry_delta_chars=16_000,
                prompt_budget_stdout_chars=32_000,
                prompt_budget_snapshot_chars=32_000,
                prompt_budget_evidence_chars=64_000,
                prompt_budget_retry_history_chars=12_000,
                context_compaction_mode="auto_overflow",
                context_compaction_scope="snapshot,pytest,files",
                context_compaction_max_input_chars=120_000,
                context_compaction_target_chars=18_000,
                context_compaction_max_calls=1,
                openai_context_window_tokens=0,
                openai_context_window_source="",
                openai_context_window_probe_provider="",
                openai_context_window_probe_error="",
                openai_response_mode_autoswitch="off",
                adaptive_gate_policy="balanced",
                adaptive_max_helper_calls=1,
                adaptive_budget_topup_max_calls=run_eval.DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS,
                adaptive_stagnation_window=1,
                adaptive_saturation_threshold=0.75,
                adaptive_no_progress_token_cap=run_eval.DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP,
                adaptive_helper_budget_profile="default",
                adaptive_openai_helper_response_mode="inherit",
                adaptive_informativeness_pregate=False,
                adaptive_informativeness_pregate_threshold=0.7,
                syntax_preflight_enabled=True,
                syntax_preflight_max_cycles=run_eval.DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES,
                syntax_preflight_fallthrough=True,
                patch_sanitize_enabled=True,
                patch_verifier_scope=run_eval.PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
                patch_verifier_min_score=run_eval.PATCH_VERIFIER_MIN_SCORE,
                patch_verifier_infer_test_source_scope=True,
                adaptive_online_policy_controller=None,
                adaptive_online_policy_mode="off",
                adaptive_online_policy_algo="linucb",
                adaptive_online_feature_scaling_profile=(
                    run_eval.DEFAULT_POLICY_FEATURE_SCALING_PROFILE
                ),
                adaptive_budget_policy_controller=None,
                adaptive_budget_policy_mode="on",
                adaptive_budget_policy_algo="linucb",
                adaptive_budget_policy_architecture=run_eval.BUDGET_POLICY_ARCH_TIER_BINARY,
                adaptive_budget_feature_scaling_profile=(
                    run_eval.DEFAULT_POLICY_FEATURE_SCALING_PROFILE
                ),
                adaptive_budget_policy_tiers=run_eval.DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS,
                adaptive_budget_policy_max_ups_per_case=(
                    run_eval.DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE
                ),
                adaptive_budget_reward_rho_budget_exceeded=(
                    run_eval.ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT
                ),
                adaptive_reward_lambda_tokens=run_eval.ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT,
                adaptive_reward_mu_latency=run_eval.ADAPTIVE_REWARD_MU_LATENCY_DEFAULT,
                adaptive_reward_token_scale=run_eval.ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT,
                adaptive_reward_latency_scale=run_eval.ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT,
                adaptive_reward_eta_progress=run_eval.ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT,
                adaptive_reward_kappa_stagnation=run_eval.ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT,
                adaptive_controller_lambda_rlm=False,
                adaptive_controller_lambda_rlm_repo_dir="",
                adaptive_controller_lambda_rlm_python="",
                adaptive_controller_lambda_rlm_context_window_chars=(
                    run_eval.DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS
                ),
                adaptive_controller_lambda_rlm_timeout_sec=(
                    run_eval.DEFAULT_LAMBDA_RLM_TIMEOUT_SEC
                ),
                run_metadata={"run_id": "stream-run"},
                record_prompt="hash",
                record_evidence="hash",
                deterministic=False,
                seed=None,
                transport_recovery_gate=None,
                openai_transport_recovery=run_eval.DEFAULT_OPENAI_TRANSPORT_RECOVERY_ENABLED,
                record_sink=streamed.append,
            ),
        ),
    )

    assert len(streamed) == 1
    assert streamed[0]["condition"] == "layered_adaptive_llm"
    assert outcome.records_streamed is True


def test_execute_eval_unit_emits_heartbeat_callbacks_when_inline_progress_is_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case = _make_synthetic_case(tmp_path, "heartbeat_callback_case")
    heartbeat_events: list[dict[str, object]] = []

    def _fake_run_one_condition(**kwargs):
        kwargs["progress_update"]("attempt_start", {"attempt": 1, "max_attempts": 1})
        time.sleep(0.08)
        record = {
            "run_id": "heartbeat-run",
            "case_id": case.case_id,
            "condition": kwargs["condition"].value,
            "attempt": 1,
            "success": True,
            "note": "success",
            "patch_meta": {},
            "attempt_metadata": {},
        }
        kwargs["record_sink"](record)
        return [record]

    monkeypatch.setattr(run_eval, "run_one_condition", _fake_run_one_condition)

    outcome = run_eval._execute_eval_unit(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        artifacts_root=tmp_path / "artifacts",
        inputs=run_eval.EvalUnitInputs(
            patcher_args=argparse.Namespace(),
            patcher_override=SimpleNamespace(name="stub"),
            keep_workdirs=False,
            progress_verbosity="off",
            progress_heartbeat_sec=0.02,
            unit_progress_callback=lambda _case_id, _condition, event, payload: (
                heartbeat_events.append(payload)
                if event == "heartbeat" and isinstance(payload, dict)
                else None
            ),
            run_condition=run_eval.RunConditionInputs(
                patcher=None,
                k=1,
                output_format="diff",
                run_id="heartbeat-run",
                include_files_context=False,
                force_patch_attempt=False,
                run_metadata={"run_id": "heartbeat-run"},
            ),
        ),
    )

    assert outcome.runtime_error is None
    assert heartbeat_events
    heartbeat = heartbeat_events[0]
    assert heartbeat["phase"] == "attempt_start"
    assert heartbeat["records_observed"] == 0
    assert heartbeat["owner"] == ""
    assert heartbeat["source"] == ""
    assert heartbeat["activity_excerpt"] == ""
    assert heartbeat["attempts_completed"] == 0
    assert heartbeat["llm_calls_observed"] == 0
    assert heartbeat["eval_duration_sec_observed"] == 0.0
    assert float(heartbeat["unit_elapsed_sec"]) > 0.0
    assert float(heartbeat["phase_elapsed_sec"]) > 0.0
    assert "[HEARTBEAT]" not in capsys.readouterr().out


def test_execute_eval_unit_preserves_phase_elapsed_and_clears_wait_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evals import eval_attempt_flow

    case = _make_synthetic_case(tmp_path, "heartbeat_phase_state_case")
    heartbeat_events: list[dict[str, object]] = []
    clock = {"now": 100.0}

    monkeypatch.setattr(eval_attempt_flow.time, "monotonic", lambda: clock["now"])

    def _fake_run_one_condition(**kwargs):
        kwargs["progress_update"](
            "llm_request",
            {
                "attempt": 1,
                "max_attempts": 1,
                "phase_label": "llm: propose patch",
                "owner": "openai",
                "source": "phase",
                "waiting_for": "openai_server",
                "external_operation": "propose_patch",
                "external_provider": "openai",
            },
        )
        clock["now"] = 100.05
        time.sleep(0.04)
        kwargs["progress_update"](
            "llm_request",
            {
                "attempt": 1,
                "max_attempts": 1,
                "phase_label": "llm: propose patch",
                "owner": "openai",
                "source": "stdout",
                "waiting_for": "openai_server",
                "external_operation": "propose_patch",
                "external_provider": "openai",
                "activity_excerpt": "stream chunk",
            },
        )
        clock["now"] = 100.10
        time.sleep(0.04)
        kwargs["progress_update"](
            "patch_apply",
            {
                "attempt": 1,
                "max_attempts": 1,
                "phase_label": "harness: patch apply",
                "owner": "harness",
                "source": "phase",
                "note": "applying diff",
            },
        )
        clock["now"] = 100.15
        time.sleep(0.04)
        record = {
            "run_id": "heartbeat-phase-run",
            "case_id": case.case_id,
            "condition": kwargs["condition"].value,
            "attempt": 1,
            "success": True,
            "note": "success",
            "patch_meta": {},
            "attempt_metadata": {},
        }
        kwargs["record_sink"](record)
        return [record]

    monkeypatch.setattr(run_eval, "run_one_condition", _fake_run_one_condition)

    outcome = run_eval._execute_eval_unit(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        artifacts_root=tmp_path / "artifacts",
        inputs=run_eval.EvalUnitInputs(
            patcher_args=argparse.Namespace(),
            patcher_override=SimpleNamespace(name="stub"),
            keep_workdirs=False,
            progress_verbosity="off",
            progress_heartbeat_sec=0.01,
            unit_progress_callback=lambda _case_id, _condition, event, payload: (
                heartbeat_events.append(payload)
                if event == "heartbeat" and isinstance(payload, dict)
                else None
            ),
            run_condition=run_eval.RunConditionInputs(
                patcher=None,
                k=1,
                output_format="diff",
                run_id="heartbeat-phase-run",
                include_files_context=False,
                force_patch_attempt=False,
                run_metadata={"run_id": "heartbeat-phase-run"},
            ),
        ),
    )

    assert outcome.runtime_error is None
    llm_heartbeats = [
        payload for payload in heartbeat_events if payload.get("phase") == "llm: propose patch"
    ]
    assert llm_heartbeats
    assert max(float(payload["phase_elapsed_sec"]) for payload in llm_heartbeats) >= 0.09

    patch_apply_heartbeats = [
        payload for payload in heartbeat_events if payload.get("phase") == "harness: patch apply"
    ]
    assert patch_apply_heartbeats
    patch_apply_heartbeat = patch_apply_heartbeats[-1]
    assert patch_apply_heartbeat["owner"] == "harness"
    assert patch_apply_heartbeat["source"] == "phase"
    assert patch_apply_heartbeat["waiting_for"] == ""
    assert patch_apply_heartbeat["external_operation"] == ""
    assert patch_apply_heartbeat["external_provider"] == ""


@pytest.mark.parametrize(
    "config_name",
    [
        "openai_base_repair.latest.toml",
        "openai_cot_repair.template.toml",
        "openai_adaptive_gated.template.toml",
        "openai_adaptive.template.toml",
        "openai_adaptive_tuned.template.toml",
        "openai_base_repair_retry_reflection.template.toml",
        "openai_adaptive_stage_b_cot.template.toml",
        "openai_base_repair_repair_method_scaffold.template.toml",
        "openai_base_repair_retry_plan_then_patch.template.toml",
        "openai_base_repair_fair_v2.template.toml",
        "openai_cot_repair_fair_v2.template.toml",
        "baseline_ldb.template.toml",
        "baseline_intervenor.template.toml",
        "baseline_intervenor_fair_v2.template.toml",
        "baseline_tracecoder.template.toml",
        "baseline_tracecoder_fair_v2.template.toml",
        "baseline_lambda_rlm.template.toml",
        "baseline_rlm.template.toml",
    ],
)
def test_shipped_eval_configs_share_repo_default_k_jobs_and_openai_base_url(
    config_name: str,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "evals" / "configs" / config_name
    payload = eval_config._load_toml_document(config_path)
    run_eval_payload = payload["run_eval"]
    openai_payload = run_eval_payload["openai"]

    assert run_eval_payload["k"] == 2
    assert run_eval_payload["jobs"] == 3
    assert openai_payload["base_url"] == "http://127.0.0.1:12000/v1"


@pytest.mark.parametrize(
    ("config_name", "baseline_name", "expected_condition", "repo_name"),
    [
        ("baseline_ldb.template.toml", "ldb", "ldb", "LLMDebugger"),
        ("baseline_intervenor.template.toml", "intervenor", "intervenor", "INTERVENOR"),
        (
            "baseline_intervenor_fair_v2.template.toml",
            "intervenor",
            "intervenor",
            "INTERVENOR",
        ),
        ("baseline_tracecoder.template.toml", "tracecoder", "tracecoder", "TraceCoder"),
        (
            "baseline_tracecoder_fair_v2.template.toml",
            "tracecoder",
            "tracecoder",
            "TraceCoder",
        ),
        ("baseline_lambda_rlm.template.toml", "lambda_rlm", "lambda_rlm", "lambda-RLM"),
        ("baseline_rlm.template.toml", "rlm", "rlm", "rlm"),
    ],
)
def test_shipped_baseline_templates_resolve_expected_baseline_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    baseline_name: str,
    expected_condition: str,
    repo_name: str,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, f"{baseline_name}_template_case")
    (repo_root / "baselines" / "repos" / repo_name).mkdir(parents=True, exist_ok=True)
    env_python = repo_root / "baselines" / "envs" / baseline_name / "bin" / "python"
    env_python.parent.mkdir(parents=True, exist_ok=True)
    env_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    env_python.chmod(0o755)
    config_path = Path(__file__).resolve().parents[1] / "evals" / "configs" / config_name

    monkeypatch.setenv("OPENAI_API_KEY", "test-token")
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="baseline"))

    captured: dict[str, object] = {}

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        captured["condition"] = condition.value
        captured["run_metadata"] = dict(run_metadata)
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--config",
            str(config_path),
            "--run-id",
            f"shipped-{baseline_name}-template",
            "--max-cases",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    assert captured["condition"] == expected_condition
    run_metadata = captured["run_metadata"]
    assert isinstance(run_metadata, dict)
    patcher_config = run_metadata.get("patcher_config")
    assert isinstance(patcher_config, dict)
    assert patcher_config["baseline_name"] == baseline_name
    assert patcher_config["baseline_condition"] == expected_condition
    assert patcher_config["baseline_repo_dir"] == str(repo_root / "baselines" / "repos" / repo_name)
    assert patcher_config["baseline_python"] == str(env_python)


@pytest.mark.parametrize(
    "config_name",
    [
        "baseline_ldb.template.toml",
        "baseline_intervenor.template.toml",
        "baseline_intervenor_fair_v2.template.toml",
        "baseline_tracecoder.template.toml",
        "baseline_tracecoder_fair_v2.template.toml",
        "baseline_lambda_rlm.template.toml",
        "baseline_lambda_rlm_repair_fair.template.toml",
        "baseline_rlm.template.toml",
        "baseline_rlm_repair_fair.template.toml",
    ],
)
def test_shipped_baseline_templates_default_to_llamacpp_thinking_compat(
    config_name: str,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "evals" / "configs" / config_name
    payload = eval_config._load_toml_document(config_path)
    run_eval_payload = payload["run_eval"]

    assert run_eval_payload["baseline_openai_compat_profile"] == "lmstudio_model_yaml"
    assert run_eval_payload["baseline_openai_allow_reasoning_fallback"] is True
    assert run_eval_payload["baseline_openai_enable_thinking"] is True


@pytest.mark.parametrize(
    "config_name",
    [
        "openai_base_repair.latest.toml",
        "openai_cot_repair.template.toml",
        "openai_adaptive_gated.template.toml",
        "openai_adaptive.template.toml",
        "openai_base_repair_retry_reflection.template.toml",
        "openai_adaptive_stage_b_cot.template.toml",
        "openai_base_repair_repair_method_scaffold.template.toml",
        "openai_base_repair_retry_plan_then_patch.template.toml",
        "openai_base_repair_fair_v2.template.toml",
        "openai_cot_repair_fair_v2.template.toml",
    ],
)
def test_shipped_openai_templates_default_to_llamacpp_reasoning_fallback(
    config_name: str,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "evals" / "configs" / config_name
    payload = eval_config._load_toml_document(config_path)
    openai_payload = payload["run_eval"]["openai"]

    assert openai_payload["compat_profile"] == "lmstudio_model_yaml"
    assert openai_payload["allow_reasoning_fallback"] is True
    assert openai_payload["enable_thinking"] is True


@pytest.mark.parametrize(
    ("config_name", "expected_pairs"),
    [
        (
            "openai_base_repair_fair_v2.template.toml",
            {"openai_max_tokens": 4096},
        ),
        (
            "openai_cot_repair_fair_v2.template.toml",
            {"openai_max_tokens": 4096, "openai_plan_patch_max_tokens": 2048},
        ),
        (
            "baseline_intervenor_fair_v2.template.toml",
            {
                "baseline_intervenor_teacher_max_tokens": 3072,
                "baseline_intervenor_learner_max_tokens": 6144,
            },
        ),
        (
            "baseline_tracecoder_fair_v2.template.toml",
            {
                "baseline_tracecoder_analysis_max_tokens": 2048,
                "baseline_tracecoder_implementation_max_tokens": 4096,
            },
        ),
    ],
)
def test_shipped_fair_v2_templates_pin_budget_overrides(
    config_name: str,
    expected_pairs: dict[str, int],
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "evals" / "configs" / config_name
    payload = eval_config._load_toml_document(config_path)
    flat_payload = eval_config._flatten_config_mapping(payload["run_eval"])

    for key, value in expected_pairs.items():
        assert flat_payload[key] == value


@pytest.mark.parametrize(
    ("config_name", "expected_pairs"),
    [
        (
            "baseline_lambda_rlm.template.toml",
            {
                "baseline_lambda_prompt_contract": "canonical",
                "baseline_lambda_max_tokens": 8192,
                "baseline_lambda_context_window_chars": 100000,
                "baseline_lambda_accuracy_target": 0.8,
            },
        ),
        (
            "baseline_lambda_rlm_repair_fair.template.toml",
            {
                "baseline_lambda_prompt_contract": "repair_fair_v1",
                "baseline_lambda_max_tokens": 8192,
                "baseline_lambda_force_task_type": "qa",
            },
        ),
        (
            "baseline_rlm.template.toml",
            {
                "baseline_rlm_prompt_contract": "canonical",
                "baseline_rlm_max_depth": 2,
                "baseline_rlm_max_iterations": 24,
                "baseline_rlm_max_tokens": 3072,
                "baseline_rlm_max_total_tokens": 200000,
            },
        ),
        (
            "baseline_rlm_repair_fair.template.toml",
            {
                "baseline_rlm_prompt_contract": "repair_fair_v1",
                "baseline_rlm_max_depth": 1,
                "baseline_rlm_max_iterations": 8,
                "baseline_rlm_max_tokens": 3072,
                "baseline_rlm_max_total_tokens": 200000,
            },
        ),
    ],
)
def test_shipped_lambda_rlm_and_rlm_templates_pin_canonical_and_fair_budget_defaults(
    config_name: str,
    expected_pairs: dict[str, object],
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "evals" / "configs" / config_name
    payload = eval_config._load_toml_document(config_path)
    flat_payload = eval_config._flatten_config_mapping(payload["run_eval"])

    for key, value in expected_pairs.items():
        assert flat_payload[key] == value


def test_run_eval_openai_compat_settings_from_config_are_normalized_in_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "openai_config_compat_case")
    config_path = tmp_path / "openai-config.toml"
    config_path.write_text(
        """
[run_eval]
run_id = "replace-me"
patcher = "openai"
conditions = "base_repair"
k = 1
jobs = 1
openai_api_key_env = "OPENAI_API_KEY"

[run_eval.openai]
base_url = "http://127.0.0.1:12000/v1"
model = "dummy-model"
compat_profile = "lmstudio_model_yaml"
allow_reasoning_fallback = false
enable_thinking = true
timeout_sec = 30.0
response_mode = "edits"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-token")
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="openai"))

    captured_run_metadata: dict[str, object] = {}

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        captured_run_metadata.update(dict(run_metadata))
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--config",
            str(config_path),
            "--run-id",
            "openai-config-compat",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    patcher_config = captured_run_metadata.get("patcher_config")
    assert isinstance(patcher_config, dict)
    assert patcher_config["openai_compat_profile"] == "lmstudio_model_yaml"
    assert patcher_config["openai_enable_thinking"] is True
    assert patcher_config["openai_allow_reasoning_fallback"] is True


@pytest.mark.parametrize(
    ("config_name", "expected_condition", "baseline_section", "expected_inner_timeout"),
    [
        ("baseline_ldb.template.toml", "ldb", None, None),
        ("baseline_intervenor.template.toml", "intervenor", None, None),
        ("baseline_intervenor_fair_v2.template.toml", "intervenor", None, None),
        ("baseline_tracecoder.template.toml", "tracecoder", None, None),
        ("baseline_tracecoder_fair_v2.template.toml", "tracecoder", None, None),
        ("baseline_lambda_rlm.template.toml", "lambda_rlm", "baseline_lambda", 600),
        ("baseline_rlm.template.toml", "rlm", "baseline_rlm", 600),
    ],
)
def test_shipped_baseline_templates_pin_condition_and_comparable_timeouts(
    config_name: str,
    expected_condition: str,
    baseline_section: str | None,
    expected_inner_timeout: int | None,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "evals" / "configs" / config_name
    payload = eval_config._load_toml_document(config_path)
    run_eval_payload = payload["run_eval"]

    assert run_eval_payload["baseline"] == expected_condition
    assert run_eval_payload["conditions"] == expected_condition
    assert run_eval_payload["baseline_process_timeout_sec"] == 600.0

    if baseline_section is None:
        return

    baseline_payload = run_eval_payload[baseline_section]
    assert baseline_payload["timeout_sec"] == expected_inner_timeout


def test_run_pytest_dispatches_local_executor(tmp_path, monkeypatch) -> None:
    marker = run_eval.RunResult(returncode=0, duration_sec=0.0, stdout="ok", stderr="")
    monkeypatch.setattr(run_eval, "_run_pytest_local", lambda **kwargs: marker)
    result = run_eval.run_pytest(
        cwd=tmp_path,
        python_args=["-m", "pytest", "-q"],
        condition=run_eval.Condition.BASE_REPAIR,
        timeout_sec=1,
        output_format="json_compact",
        execution=run_eval.ExecutionConfig(executor="local"),
    )
    assert result is marker


def test_run_pytest_dispatches_testcontainers_executor(tmp_path, monkeypatch) -> None:
    marker = run_eval.RunResult(returncode=1, duration_sec=0.0, stdout="", stderr="x")
    monkeypatch.setattr(run_eval, "_run_pytest_testcontainers", lambda **kwargs: marker)
    result = run_eval.run_pytest(
        cwd=tmp_path,
        python_args=["-m", "pytest", "-q"],
        condition=run_eval.Condition.BASE_REPAIR,
        timeout_sec=1,
        output_format="json_compact",
        execution=run_eval.ExecutionConfig(executor="testcontainers"),
    )
    assert result is marker


def test_write_run_outputs_truncates_to_latest_lines(tmp_path) -> None:
    run = run_eval.RunResult(
        returncode=1,
        duration_sec=0.0,
        stdout="line-1\nline-2\nline-3\nline-4\n",
        stderr="err-1\nerr-2\n",
    )
    run_eval._write_run_outputs(tmp_path, run, prefix="pytest_before", tail_lines=2)

    stdout_text = (tmp_path / "pytest_before_stdout.txt").read_text(encoding="utf-8")
    stderr_text = (tmp_path / "pytest_before_stderr.txt").read_text(encoding="utf-8")
    combined_text = (tmp_path / "pytest_before_combined.txt").read_text(encoding="utf-8")

    assert "line-1" not in stdout_text
    assert "line-2" not in stdout_text
    assert "line-3\nline-4\n" in stdout_text
    assert "truncated 2 earlier line(s)" in stdout_text
    assert stderr_text == "err-1\nerr-2\n"
    assert "line-3\nline-4\n" in combined_text
    assert "err-1\nerr-2\n" in combined_text


def test_write_run_outputs_applies_hard_char_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_eval, "PERSISTED_PYTEST_OUTPUT_HARD_CHAR_CAP", 64)
    huge_line = "prefix\n" + ("x" * 500)
    run = run_eval.RunResult(returncode=1, duration_sec=0.0, stdout=huge_line, stderr="")
    run_eval._write_run_outputs(tmp_path, run, prefix="pytest_before", tail_lines=0)

    stdout_text = (tmp_path / "pytest_before_stdout.txt").read_text(encoding="utf-8")
    assert "character(s)" in stdout_text
    assert stdout_text.endswith("x" * 64)
    assert len(stdout_text) < 200


def test_truncate_snapshot_for_artifact_caps_nested_strings() -> None:
    huge = "X" * 1000
    snapshot = {
        "exception": {"type": "AssertionError", "message": huge},
        "frames": [{"code": huge, "locals": {"value": huge}}],
    }

    truncated = run_eval._truncate_snapshot_for_artifact(snapshot, max_string_chars=80)

    assert truncated is not snapshot
    assert "snapshot string truncated" in truncated["exception"]["message"]
    assert len(truncated["exception"]["message"]) <= 80
    assert len(truncated["frames"][0]["code"]) <= 80
    assert len(truncated["frames"][0]["locals"]["value"]) <= 80
    assert snapshot["exception"]["message"] == huge


def test_truncate_snapshot_for_artifact_zero_disables_truncation() -> None:
    huge = "Y" * 1000
    snapshot = {"exception": {"message": huge}}
    truncated = run_eval._truncate_snapshot_for_artifact(snapshot, max_string_chars=0)
    assert truncated["exception"]["message"] == huge


def test_run_eval_preflights_testcontainers_runtime(tmp_path, monkeypatch, capsys) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / "preflight_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    case = run_eval.CaseSpec(
        case_id="preflight_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=["preflight"],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(
        run_eval,
        "check_testcontainers_runtime_available",
        lambda: (False, "docker daemon is not reachable"),
    )

    exit_code = run_eval.main(
        [
            "--run-id",
            "preflight-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
        ]
    )
    assert exit_code == 2
    assert "Environment check failed: docker daemon is not reachable" in capsys.readouterr().out


def test_run_eval_prunes_stale_containers_before_testcontainers_run(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / "cleanup_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    case = run_eval.CaseSpec(
        case_id="cleanup_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(
        run_eval,
        "check_testcontainers_runtime_available",
        lambda: (True, "ok"),
    )
    cleanup_calls: dict[str, int] = {}

    def _cleanup(*, grace_sec: int, now_utc=None):
        cleanup_calls["grace_sec"] = grace_sec
        return 0, 0, None

    monkeypatch.setattr(run_eval, "cleanup_stale_eval_containers", _cleanup)
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "cleanup-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "testcontainers",
        ]
    )

    assert exit_code == 0
    assert cleanup_calls == {"grace_sec": run_eval.DEFAULT_CONTAINER_CLEANUP_STALE_GRACE_SEC}


def test_run_eval_openai_max_propose_default_in_patcher_config(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / "openai_default_budget_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    case = run_eval.CaseSpec(
        case_id="openai_default_budget_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="openai"))

    captured_run_metadata: dict[str, object] = {}

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        captured_run_metadata.update(dict(run_metadata))
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "openai-default-budget",
            "--patcher",
            "openai",
            "--openai-model",
            "dummy-model",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    patcher_config = captured_run_metadata.get("patcher_config")
    assert isinstance(patcher_config, dict)
    assert patcher_config["openai_max_propose_sec"] == 300.0
    assert patcher_config["openai_http2"] is True
    assert patcher_config["openai_transport_max_retries"] == 6
    assert patcher_config["openai_transport_retry_backoff_base_sec"] == 1.0
    assert patcher_config["openai_transport_retry_backoff_max_sec"] == 8.0
    assert patcher_config["openai_transport_recovery"] is True
    assert patcher_config["openai_transport_recovery_max_wait_sec"] == 60.0
    assert patcher_config["openai_transport_recovery_poll_sec"] == 2.0
    assert patcher_config["openai_compat_profile"] == "lmstudio_model_yaml"
    assert patcher_config["openai_allow_reasoning_fallback"] is True
    assert patcher_config["openai_enable_thinking"] is True


def test_run_eval_openai_transport_retry_settings_in_patcher_config(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / "openai_transport_budget_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    case = run_eval.CaseSpec(
        case_id="openai_transport_budget_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="openai"))

    captured_run_metadata: dict[str, object] = {}

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        captured_run_metadata.update(dict(run_metadata))
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "openai-transport-settings",
            "--patcher",
            "openai",
            "--openai-model",
            "dummy-model",
            "--openai-transport-max-retries",
            "9",
            "--openai-transport-retry-backoff-base-sec",
            "1.5",
            "--openai-transport-retry-backoff-max-sec",
            "12.0",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    patcher_config = captured_run_metadata.get("patcher_config")
    assert isinstance(patcher_config, dict)
    assert patcher_config["openai_transport_max_retries"] == 9
    assert patcher_config["openai_http2"] is True
    assert patcher_config["openai_transport_retry_backoff_base_sec"] == 1.5
    assert patcher_config["openai_transport_retry_backoff_max_sec"] == 12.0
    assert patcher_config["openai_transport_recovery"] is True


def test_run_eval_openai_http2_disable_in_patcher_config(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / "openai_http2_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    case = run_eval.CaseSpec(
        case_id="openai_http2_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="openai"))

    captured_run_metadata: dict[str, object] = {}

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        captured_run_metadata.update(dict(run_metadata))
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "openai-http2-off",
            "--patcher",
            "openai",
            "--openai-model",
            "dummy-model",
            "--no-openai-http2",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    patcher_config = captured_run_metadata.get("patcher_config")
    assert isinstance(patcher_config, dict)
    assert patcher_config["openai_http2"] is False


def test_run_eval_openai_thinking_auto_enables_reasoning_fallback_in_patcher_config(
    tmp_path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / "openai_thinking_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    case = run_eval.CaseSpec(
        case_id="openai_thinking_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="openai"))

    captured_run_metadata: dict[str, object] = {}

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        captured_run_metadata.update(dict(run_metadata))
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "openai-thinking-settings",
            "--patcher",
            "openai",
            "--openai-model",
            "dummy-model",
            "--openai-enable-thinking",
            "--no-openai-allow-reasoning-fallback",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    patcher_config = captured_run_metadata.get("patcher_config")
    assert isinstance(patcher_config, dict)
    assert patcher_config["openai_enable_thinking"] is True
    assert patcher_config["openai_allow_reasoning_fallback"] is True


def test_run_eval_case_list_file_filters_selected_cases(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_root = repo_root / "synthetic_cases"
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "case_a").mkdir(parents=True, exist_ok=True)
    (case_root / "case_b").mkdir(parents=True, exist_ok=True)
    (case_root / "case_c").mkdir(parents=True, exist_ok=True)

    case_a = run_eval.CaseSpec(
        case_id="case_a",
        path=case_root / "case_a",
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )
    case_b = run_eval.CaseSpec(
        case_id="case_b",
        path=case_root / "case_b",
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )
    case_c = run_eval.CaseSpec(
        case_id="case_c",
        path=case_root / "case_c",
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    case_list_file = tmp_path / "selected_cases.txt"
    case_list_file.write_text("# comment\n\ncase_b\n", encoding="utf-8")

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case_a, case_b, case_c])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    executed_case_ids: list[str] = []

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        executed_case_ids.append(case.case_id)
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "case-list-filter",
            "--patcher",
            "null",
            "--case",
            "case_a",
            "--case-list-file",
            str(case_list_file),
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    assert sorted(executed_case_ids) == ["case_a", "case_b"]


def test_run_eval_case_list_file_rejects_unknown_case_id(tmp_path, monkeypatch, capsys) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_root = repo_root / "synthetic_cases"
    case_root.mkdir(parents=True, exist_ok=True)

    case = run_eval.CaseSpec(
        case_id="known_case",
        path=case_root / "known_case",
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    case_list_file = tmp_path / "selected_cases.txt"
    case_list_file.write_text("unknown_case\n", encoding="utf-8")

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])

    exit_code = run_eval.main(
        [
            "--run-id",
            "case-list-unknown",
            "--patcher",
            "null",
            "--case-list-file",
            str(case_list_file),
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 2
    assert "Unknown case id(s) requested via --case/--case-list-file" in capsys.readouterr().out


def test_run_eval_case_list_file_read_error_returns_cli_error(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_root = repo_root / "synthetic_cases"
    case_root.mkdir(parents=True, exist_ok=True)
    case_dir = case_root / "known_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    case = run_eval.CaseSpec(
        case_id="known_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    case_list_file = tmp_path / "selected_cases.txt"
    case_list_file.write_text("known_case\n", encoding="utf-8")

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])

    original_read_text = Path.read_text

    def _fake_read_text(path: Path, *args, **kwargs) -> str:
        if path == case_list_file:
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _fake_read_text)

    exit_code = run_eval.main(
        [
            "--run-id",
            "case-list-read-error",
            "--patcher",
            "null",
            "--case-list-file",
            str(case_list_file),
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 2
    assert "Failed to read case-list file" in capsys.readouterr().out


def test_run_eval_forwards_pytest_output_tail_lines(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / "tail_lines_forward_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    case = run_eval.CaseSpec(
        case_id="tail_lines_forward_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    observed_tail_lines: dict[str, int] = {}

    def _run(*, case, condition, run_id, run_metadata, pytest_output_tail_lines, **_kwargs):
        observed_tail_lines["value"] = int(pytest_output_tail_lines)
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "tail-lines-forward",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
            "--pytest-output-tail-lines",
            "123",
        ]
    )

    assert exit_code == 0
    assert observed_tail_lines["value"] == 123


def test_run_eval_forwards_snapshot_artifact_max_string_chars(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / "snapshot_cap_forward_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    case = run_eval.CaseSpec(
        case_id="snapshot_cap_forward_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    observed_snapshot_cap: dict[str, int] = {}

    def _run(
        *,
        case,
        condition,
        run_id,
        run_metadata,
        snapshot_artifact_max_string_chars,
        **_kwargs,
    ):
        observed_snapshot_cap["value"] = int(snapshot_artifact_max_string_chars)
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "snapshot-cap-forward",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
            "--snapshot-artifact-max-string-chars",
            "321",
        ]
    )

    assert exit_code == 0
    assert observed_snapshot_cap["value"] == 321


def test_run_eval_config_precedence_and_resolved_lockfile(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / "config_precedence_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        """
[run_eval]
patcher = "null"
conditions = "base_repair"
k = 3
run_id = "run-from-config"
executor = "local"
prompt_retry_reflection = true
prompt_adaptive_stage_b_cot = true
prompt_repair_method_scaffold = true
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    case = run_eval.CaseSpec(
        case_id="config_precedence_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))
    observed: dict[str, object] = {}

    def _run(*, case, condition, run_id, run_metadata, k, **_kwargs):
        observed["run_id"] = run_id
        observed["k"] = int(k)
        observed["run_metadata"] = dict(run_metadata)
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--config",
            str(config_path),
            "--run-id",
            "run-from-cli",
            "--k",
            "1",
            "--no-prompt-retry-reflection",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    assert observed["run_id"] == "run-from-cli"
    assert observed["k"] == 1
    lock_path = repo_root / "evals" / "artifacts" / "configs" / "run-from-cli.resolved.json"
    assert lock_path.exists()
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["run_id"] == "run-from-cli"
    assert lock_payload["resolved_args"]["prompt_retry_reflection"] is False
    assert lock_payload["resolved_args"]["prompt_adaptive_stage_b_cot"] is True
    assert lock_payload["resolved_args"]["prompt_repair_method_scaffold"] is True
    assert lock_payload["config_files"] == [str(config_path.resolve())]
    assert lock_payload["resolved_args"]["k"] == 1
    assert lock_payload["value_sources"]["k"] == "cli"
    assert str(lock_payload["value_sources"]["patcher"]).startswith("config:")


def test_run_eval_rejects_plaintext_openai_api_key_in_config(tmp_path, capsys) -> None:
    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        """
[run_eval]
patcher = "openai"
openai_model = "demo"
openai_api_key = "sk-test-literal"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exit_code = run_eval.main(["--config", str(config_path)])
    assert exit_code == 2
    assert "openai_api_key" in capsys.readouterr().out


def test_run_eval_uses_run_scoped_adaptive_policy_state_paths_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "adaptive_state_defaults_case")

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    observed: dict[str, object] = {}

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        observed["run_metadata"] = dict(run_metadata)
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "adaptive-state-defaults",
            "--patcher",
            "null",
            "--conditions",
            "layered_adaptive_llm",
            "--k",
            "1",
            "--executor",
            "local",
            "--adaptive-online-policy-mode",
            "shadow",
        ]
    )

    assert exit_code == 0
    lock_path = (
        repo_root / "evals" / "artifacts" / "configs" / "adaptive-state-defaults.resolved.json"
    )
    assert lock_path.exists()
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["adaptive_online_state_path"] == str(
        repo_root / "evals" / "artifacts" / "policies" / "adaptive-state-defaults.json"
    )
    assert lock_payload["resolved_args"]["adaptive_budget_online_state_path"] == str(
        repo_root
        / "evals"
        / "artifacts"
        / "policies"
        / "adaptive-state-defaults.adaptive_budget_policy.json"
    )
    assert lock_payload["value_sources"]["adaptive_online_state_path"] == "derived_from_run_id"
    assert (
        lock_payload["value_sources"]["adaptive_budget_online_state_path"] == "derived_from_run_id"
    )


def test_run_eval_preserves_explicit_adaptive_policy_state_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "adaptive_state_explicit_case")
    online_state_path = repo_root / "evals" / "artifacts" / "policies" / "shared-online.json"
    budget_state_path = repo_root / "evals" / "artifacts" / "policies" / "shared-budget.json"

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))
    monkeypatch.setattr(
        run_eval,
        "run_one_condition",
        lambda *, case, condition, run_id, run_metadata, **_kwargs: [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ],
    )

    exit_code = run_eval.main(
        [
            "--run-id",
            "adaptive-state-explicit",
            "--patcher",
            "null",
            "--conditions",
            "layered_adaptive_llm",
            "--k",
            "1",
            "--executor",
            "local",
            "--adaptive-online-policy-mode",
            "shadow",
            "--adaptive-online-state-path",
            str(online_state_path),
            "--adaptive-budget-online-state-path",
            str(budget_state_path),
        ]
    )

    assert exit_code == 0
    lock_path = (
        repo_root / "evals" / "artifacts" / "configs" / "adaptive-state-explicit.resolved.json"
    )
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["adaptive_online_state_path"] == str(online_state_path)
    assert lock_payload["resolved_args"]["adaptive_budget_online_state_path"] == str(
        budget_state_path
    )
    assert lock_payload["value_sources"]["adaptive_online_state_path"] == "cli"
    assert lock_payload["value_sources"]["adaptive_budget_online_state_path"] == "cli"


def test_run_eval_forces_adaptive_only_features_off_for_non_adaptive_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "nonadaptive_force_off_case")
    observed: dict[str, object] = {}

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run(*, context_protocol, adaptive_controller_lambda_rlm, **kwargs):
        observed["context_protocol"] = context_protocol
        observed["adaptive_controller_lambda_rlm"] = adaptive_controller_lambda_rlm
        observed["adaptive_online_policy_mode"] = kwargs["adaptive_online_policy_mode"]
        observed["adaptive_budget_policy_mode"] = kwargs["adaptive_budget_policy_mode"]
        return [
            {
                "run_id": kwargs["run_id"],
                "case_id": case.case_id,
                "condition": kwargs["condition"].value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(kwargs["run_metadata"]),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "nonadaptive-force-off",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--context-protocol",
            "staged",
            "--adaptive-controller-lambda-rlm",
            "--adaptive-online-policy-mode",
            "shadow",
            "--adaptive-budget-policy-mode",
            "on",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    assert observed["context_protocol"] == "monolithic"
    assert observed["adaptive_controller_lambda_rlm"] is False
    assert observed["adaptive_online_policy_mode"] == "off"
    assert observed["adaptive_budget_policy_mode"] == "off"

    captured = capsys.readouterr().out
    assert "Non-adaptive run: forced adaptive-only features off" in captured

    lock_path = (
        repo_root / "evals" / "artifacts" / "configs" / "nonadaptive-force-off.resolved.json"
    )
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["context_protocol"] == "monolithic"
    assert lock_payload["resolved_args"]["adaptive_controller_lambda_rlm"] is False
    assert lock_payload["resolved_args"]["adaptive_online_policy_mode"] == "off"
    assert lock_payload["resolved_args"]["adaptive_budget_policy_mode"] == "off"
    assert lock_payload["value_sources"]["context_protocol"] == "forced_off:no_adaptive_condition"
    assert (
        lock_payload["value_sources"]["adaptive_controller_lambda_rlm"]
        == "forced_off:no_adaptive_condition"
    )
    assert (
        lock_payload["value_sources"]["adaptive_online_policy_mode"]
        == "forced_off:no_adaptive_condition"
    )
    assert (
        lock_payload["value_sources"]["adaptive_budget_policy_mode"]
        == "forced_off:no_adaptive_condition"
    )


def test_run_eval_records_budget_policy_architecture_in_lockfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "adaptive_budget_architecture_case")

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))
    monkeypatch.setattr(
        run_eval,
        "run_one_condition",
        lambda *, case, condition, run_id, run_metadata, **_kwargs: [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ],
    )

    exit_code = run_eval.main(
        [
            "--run-id",
            "adaptive-budget-architecture",
            "--patcher",
            "null",
            "--conditions",
            "layered_adaptive_llm",
            "--k",
            "1",
            "--executor",
            "local",
            "--adaptive-budget-policy-architecture",
            "hierarchical_v1",
        ]
    )

    assert exit_code == 0
    lock_path = (
        repo_root / "evals" / "artifacts" / "configs" / "adaptive-budget-architecture.resolved.json"
    )
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["adaptive_budget_policy_architecture"] == "hierarchical_v1"


def test_run_eval_resolves_openai_api_key_env_from_config_and_redacts_lock(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / "config_env_secret_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        """
[run_eval]
run_id = "config-env-secret-run"
patcher = "openai"
conditions = "base_repair"
executor = "local"
k = 1

[run_eval.openai]
model = "demo"
api_key_env = "LLMDEBUG_TEST_OPENAI_KEY"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLMDEBUG_TEST_OPENAI_KEY", "super-secret-token")

    case = run_eval.CaseSpec(
        case_id="config_env_secret_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    observed_openai_key: dict[str, str] = {}

    def _make_patcher(_args):
        observed_openai_key["value"] = str(_args.openai_api_key)
        return SimpleNamespace(name="openai")

    monkeypatch.setattr(run_eval, "make_patcher", _make_patcher)

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(["--config", str(config_path)])
    assert exit_code == 0
    assert observed_openai_key["value"] == "super-secret-token"

    lock_path = (
        repo_root / "evals" / "artifacts" / "configs" / "config-env-secret-run.resolved.json"
    )
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["openai_api_key"] == run_eval.RUN_EVAL_LOCKFILE_REDACTED


def test_run_eval_config_openai_api_key_env_uses_standard_fallback_chain(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / "config_openai_env_fallback_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        """
[run_eval]
run_id = "config-openai-env-fallback-run"
patcher = "openai"
conditions = "base_repair"
executor = "local"
k = 1
openai_api_key_env = "OPENAI_API_KEY"

[run_eval.openai]
model = "demo"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLMDEBUG_EVAL_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("Z_API", "z-secret-token")

    case = run_eval.CaseSpec(
        case_id="config_openai_env_fallback_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    observed_openai_key: dict[str, str] = {}

    def _make_patcher(_args):
        observed_openai_key["value"] = str(_args.openai_api_key)
        return SimpleNamespace(name="openai")

    monkeypatch.setattr(run_eval, "make_patcher", _make_patcher)
    monkeypatch.setattr(
        run_eval,
        "run_one_condition",
        lambda *, case, condition, run_id, run_metadata, **_kwargs: [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ],
    )

    exit_code = run_eval.main(["--config", str(config_path)])
    assert exit_code == 0
    assert observed_openai_key["value"] == "z-secret-token"


def test_run_eval_config_openai_api_key_env_defaults_to_local_when_standard_sources_missing(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case_dir = repo_root / "synthetic_cases" / "config_openai_env_default_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        """
[run_eval]
run_id = "config-openai-env-default-run"
patcher = "openai"
conditions = "base_repair"
executor = "local"
k = 1
openai_api_key_env = "OPENAI_API_KEY"

[run_eval.openai]
model = "demo"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("Z_API", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLMDEBUG_EVAL_OPENAI_API_KEY", raising=False)

    case = run_eval.CaseSpec(
        case_id="config_openai_env_default_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    observed_openai_key: dict[str, str] = {}

    def _make_patcher(_args):
        observed_openai_key["value"] = str(_args.openai_api_key)
        return SimpleNamespace(name="openai")

    monkeypatch.setattr(run_eval, "make_patcher", _make_patcher)
    monkeypatch.setattr(
        run_eval,
        "run_one_condition",
        lambda *, case, condition, run_id, run_metadata, **_kwargs: [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ],
    )

    exit_code = run_eval.main(["--config", str(config_path)])
    assert exit_code == 0
    assert observed_openai_key["value"] == "local"


def test_run_eval_banner_reports_adaptive_context_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "adaptive_banner_case")

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))
    monkeypatch.setattr(
        run_eval,
        "run_one_condition",
        lambda *, case, condition, run_id, run_metadata, **_kwargs: [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ],
    )

    exit_code = run_eval.main(
        [
            "--run-id",
            "adaptive-banner-run",
            "--patcher",
            "null",
            "--conditions",
            "layered_adaptive_llm",
            "--context-protocol",
            "staged",
            "--max-context-chars",
            "1234",
            "--context-stage-a-max-requests",
            "7",
            "--context-stage-a-max-chars",
            "4321",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Prompting / Adaptive:" in captured
    assert "Context: staged  max_chars=1234  files=on" in captured
    assert "Stage-A: max_requests=7  max_chars=4321" in captured


def test_run_eval_banner_prints_stage_a_limits_for_adaptive_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "adaptive_stage_a_banner_case")

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))
    monkeypatch.setattr(
        run_eval,
        "run_one_condition",
        lambda *, case, condition, run_id, run_metadata, **_kwargs: [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ],
    )

    exit_code = run_eval.main(
        [
            "--run-id",
            "adaptive-stage-a-banner-run",
            "--patcher",
            "null",
            "--conditions",
            "layered_adaptive_llm",
            "--context-protocol",
            "monolithic",
            "--context-stage-a-max-requests",
            "9",
            "--context-stage-a-max-chars",
            "2468",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Stage-A: max_requests=9  max_chars=2468" in captured


def test_run_eval_banner_prints_prompt_feature_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "prompt_feature_banner_case")

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="openai"))
    monkeypatch.setattr(
        run_eval,
        "run_one_condition",
        lambda *, case, condition, run_id, run_metadata, **_kwargs: [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ],
    )

    exit_code = run_eval.main(
        [
            "--run-id",
            "prompt-feature-banner-run",
            "--patcher",
            "openai",
            "--openai-model",
            "test-model",
            "--conditions",
            "layered_adaptive_llm",
            "--prompt-retry-reflection",
            "--prompt-adaptive-stage-b-cot",
            "--prompt-repair-method-scaffold",
            "--prompt-retry-plan-then-patch",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Resolved Config / Prompting & Budgets:" in captured
    assert "prompt_retry_reflection: true" in captured
    assert "prompt_adaptive_stage_b_cot: true" in captured
    assert "prompt_repair_method_scaffold: true" in captured
    assert "prompt_retry_plan_then_patch: true" in captured


def test_run_eval_rejects_retry_plan_then_patch_for_non_openai_patcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, "retry_plan_requires_openai_case")

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])

    exit_code = run_eval.main(
        [
            "--run-id",
            "retry-plan-invalid-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--prompt-retry-plan-then-patch",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr().out
    assert "--prompt-retry-plan-then-patch currently requires --patcher openai" in captured


@pytest.mark.parametrize(
    ("config_name", "expected_condition"),
    [
        ("openai_base_repair.latest.toml", "base_repair"),
        ("openai_base_repair_retry_reflection.template.toml", "base_repair"),
        ("openai_base_repair_repair_method_scaffold.template.toml", "base_repair"),
        ("openai_base_repair_retry_plan_then_patch.template.toml", "base_repair"),
        ("openai_base_repair_fair_v2.template.toml", "base_repair"),
        ("openai_cot_repair.template.toml", "cot_repair"),
        ("openai_cot_repair_fair_v2.template.toml", "cot_repair"),
        ("openai_adaptive_gated.template.toml", "layered_adaptive_gated"),
    ],
)
def test_shipped_nonadaptive_configs_disable_budget_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    expected_condition: str,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, f"{expected_condition}_config_case")
    config_path = Path(__file__).resolve().parents[1] / "evals" / "configs" / config_name

    monkeypatch.setenv("OPENAI_API_KEY", "test-token")
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="openai"))
    monkeypatch.setattr(
        run_eval,
        "run_one_condition",
        lambda *, case, condition, run_id, run_metadata, **_kwargs: [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ],
    )

    run_id = f"shipped-{expected_condition}-config"
    exit_code = run_eval.main(
        [
            "--config",
            str(config_path),
            "--run-id",
            run_id,
            "--max-cases",
            "1",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    lock_path = repo_root / "evals" / "artifacts" / "configs" / f"{run_id}.resolved.json"
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["adaptive_budget_policy_mode"] == "off"


def test_shipped_adaptive_gated_config_pins_rule_based_gate_controls() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "evals"
        / "configs"
        / "openai_adaptive_gated.template.toml"
    )
    payload = eval_config._load_toml_document(config_path)
    run_eval_payload = payload["run_eval"]
    adaptive_payload = run_eval_payload["adaptive"]
    context_payload = run_eval_payload["context"]

    assert run_eval_payload["conditions"] == "layered_adaptive_gated"
    assert adaptive_payload["gate_policy"] == "balanced"
    assert adaptive_payload["informativeness_pregate"] is True
    assert adaptive_payload["online_policy_mode"] == "off"
    assert adaptive_payload["budget_policy_mode"] == "off"
    assert adaptive_payload["max_helper_calls"] == 1
    assert adaptive_payload["budget_topup_max_calls"] == 1
    assert adaptive_payload["no_progress_token_cap"] == 15000
    assert context_payload["compaction_scope"] == "snapshot,pytest,files"


@pytest.mark.parametrize(
    ("config_name", "expected_condition", "expected_guidance_mode"),
    [
        ("study_ldb_hybrid_canonical.template.toml", "layered_adaptive_gated", "fair_v1"),
        ("study_intervenor_hybrid_canonical.template.toml", "layered_adaptive_gated", "fair_v1"),
        ("study_tracecoder_hybrid_canonical.template.toml", "layered_adaptive_gated", "fair_v1"),
        ("study_lambda_rlm_hybrid_canonical.template.toml", "layered_adaptive_gated", "fair_v1"),
        ("study_rlm_hybrid_canonical.template.toml", "layered_adaptive_gated", "fair_v1"),
        ("study_ldb_hybrid_tuned.template.toml", "layered_adaptive_gated", "tuned_v1"),
        ("study_intervenor_hybrid_tuned.template.toml", "layered_adaptive_gated", "tuned_v1"),
        ("study_tracecoder_hybrid_tuned.template.toml", "layered_adaptive_gated", "tuned_v1"),
        ("study_lambda_rlm_hybrid_tuned.template.toml", "layered_adaptive_gated", "tuned_v1"),
        ("study_rlm_hybrid_tuned.template.toml", "layered_adaptive_gated", "tuned_v1"),
    ],
)
def test_exact_study_hybrid_baseline_presets_pin_gated_outer_controller(
    config_name: str,
    expected_condition: str,
    expected_guidance_mode: str,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "evals" / "configs" / config_name
    payload = eval_config._load_toml_document(config_path)
    run_eval_payload = payload["run_eval"]
    adaptive_payload = run_eval_payload["adaptive"]
    context_payload = run_eval_payload["context"]

    assert run_eval_payload["conditions"] == expected_condition
    assert run_eval_payload["k"] == 2
    assert run_eval_payload["jobs"] == 2
    assert run_eval_payload["baseline_hybrid_guidance_mode"] == expected_guidance_mode
    assert run_eval_payload["adaptive_controller_lambda_rlm"] is False
    assert adaptive_payload["gate_policy"] == "balanced"
    assert adaptive_payload["informativeness_pregate"] is True
    assert adaptive_payload["online_policy_mode"] == "off"
    assert adaptive_payload["budget_policy_mode"] == "off"
    assert adaptive_payload["max_helper_calls"] == 1
    assert adaptive_payload["budget_topup_max_calls"] == 1
    assert adaptive_payload["no_progress_token_cap"] == 15000
    assert context_payload["compaction_scope"] == "snapshot,pytest,files"


def test_intervenor_fixed_evidence_v1_preset_pins_static_evidence_arm() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "evals"
        / "configs"
        / "study_intervenor_fixed_evidence_v1.template.toml"
    )
    payload = eval_config._load_toml_document(config_path)
    run_eval_payload = payload["run_eval"]
    adaptive_payload = run_eval_payload["adaptive"]
    context_payload = run_eval_payload["context"]

    assert run_eval_payload["conditions"] == "layered_adaptive_gated"
    assert run_eval_payload["k"] == 2
    assert run_eval_payload["jobs"] == 2
    assert run_eval_payload["baseline_hybrid_guidance_mode"] == "fixed_v1"
    assert run_eval_payload["adaptive_controller_lambda_rlm"] is False
    assert run_eval_payload["baseline_intervenor_teacher_guidance_max_chars"] == 4000
    assert run_eval_payload["baseline_intervenor_learner_guidance_max_chars"] == 4000
    assert adaptive_payload["gate_policy"] == "balanced"
    assert adaptive_payload["informativeness_pregate"] is True
    assert adaptive_payload["online_policy_mode"] == "off"
    assert adaptive_payload["budget_policy_mode"] == "off"
    assert adaptive_payload["max_helper_calls"] == 0
    assert adaptive_payload["budget_topup_max_calls"] == 0
    assert adaptive_payload["no_progress_token_cap"] == 15000
    assert context_payload["compaction_scope"] == "snapshot,pytest,files"


@pytest.mark.parametrize(
    ("config_name", "baseline_name"),
    [
        ("study_ldb_raw.template.toml", "ldb"),
        ("study_intervenor_raw.template.toml", "intervenor"),
        ("study_tracecoder_raw.template.toml", "tracecoder"),
        ("study_lambda_rlm_raw.template.toml", "lambda_rlm"),
        ("study_rlm_raw.template.toml", "rlm"),
    ],
)
def test_exact_study_raw_baseline_presets_pin_two_by_two_budget(
    config_name: str,
    baseline_name: str,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "evals" / "configs" / config_name
    payload = eval_config._load_toml_document(config_path)
    run_eval_payload = payload["run_eval"]
    adaptive_payload = run_eval_payload["adaptive"]

    assert run_eval_payload["baseline"] == baseline_name
    assert run_eval_payload["conditions"] == baseline_name
    assert run_eval_payload["k"] == 2
    assert run_eval_payload["jobs"] == 2
    assert run_eval_payload["baseline_process_timeout_sec"] == 600.0
    assert adaptive_payload["online_policy_mode"] == "off"
    assert adaptive_payload["budget_policy_mode"] == "off"
    assert adaptive_payload["max_helper_calls"] == 0


@pytest.mark.parametrize(
    ("config_name", "expected_condition", "expected_flags"),
    [
        (
            "openai_base_repair_retry_reflection.template.toml",
            "base_repair",
            {
                "prompt_retry_reflection": True,
                "prompt_adaptive_stage_b_cot": False,
                "prompt_repair_method_scaffold": False,
                "prompt_retry_plan_then_patch": False,
            },
        ),
        (
            "openai_adaptive_stage_b_cot.template.toml",
            "layered_adaptive_llm",
            {
                "prompt_retry_reflection": False,
                "prompt_adaptive_stage_b_cot": True,
                "prompt_repair_method_scaffold": False,
                "prompt_retry_plan_then_patch": False,
            },
        ),
        (
            "openai_base_repair_repair_method_scaffold.template.toml",
            "base_repair",
            {
                "prompt_retry_reflection": False,
                "prompt_adaptive_stage_b_cot": False,
                "prompt_repair_method_scaffold": True,
                "prompt_retry_plan_then_patch": False,
            },
        ),
        (
            "openai_base_repair_retry_plan_then_patch.template.toml",
            "base_repair",
            {
                "prompt_retry_reflection": False,
                "prompt_adaptive_stage_b_cot": False,
                "prompt_repair_method_scaffold": False,
                "prompt_retry_plan_then_patch": True,
            },
        ),
    ],
)
def test_shipped_prompt_feature_templates_enable_expected_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    expected_condition: str,
    expected_flags: dict[str, bool],
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_synthetic_case(repo_root, f"{expected_condition}_prompt_feature_config_case")
    config_path = Path(__file__).resolve().parents[1] / "evals" / "configs" / config_name

    monkeypatch.setenv("OPENAI_API_KEY", "test-token")
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="openai"))
    monkeypatch.setattr(
        run_eval,
        "run_one_condition",
        lambda *, case, condition, run_id, run_metadata, **_kwargs: [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "run_metadata": dict(run_metadata),
            }
        ],
    )

    run_id = f"shipped-{Path(config_name).stem}-config"
    argv = [
        "--config",
        str(config_path),
        "--run-id",
        run_id,
        "--max-cases",
        "1",
        "--k",
        "1",
        "--executor",
        "local",
    ]
    exit_code = run_eval.main(argv)

    assert exit_code == 0
    lock_path = repo_root / "evals" / "artifacts" / "configs" / f"{run_id}.resolved.json"
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["resolved_args"]["conditions"] == expected_condition
    for flag_name, expected_value in expected_flags.items():
        assert lock_payload["resolved_args"][flag_name] is expected_value
