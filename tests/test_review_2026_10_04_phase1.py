"""Regression tests for Phase 1 (result integrity) of the 2026-10-04 review remediation.

Critical findings live in ``test_review_2026_10_04_criticals.py``; this file covers the
High/Medium Phase 1 findings that need a run through ``run_one_condition``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from evals import run_eval
from evals.eval_types import Condition
from evals.patchers.base import PatchResult

_FIX = (
    "--- a/buggy.py\n+++ b/buggy.py\n@@ -1,2 +1,2 @@\n"
    " def f() -> int:\n-    return 1\n+    return 2\n"
)


@dataclass
class _DiffPatcher:
    diffs: list[str | None]
    name: str = "scripted"
    prompts: list[str] = field(default_factory=list)

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        diff = self.diffs[min(len(self.prompts), len(self.diffs)) - 1]
        return PatchResult(diff=diff, meta={})


def _case(tmp_path: Path) -> run_eval.CaseSpec:
    case_dir = tmp_path / "case_src"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return run_eval.CaseSpec(
        case_id="phase1_case",
        path=case_dir,
        category="logic",
        difficulty="easy",
        snapshot_dependency="helpful",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=60,
    )


def _run(
    tmp_path: Path, patcher: _DiffPatcher, *, k: int = 1, **inputs: Any
) -> list[dict[str, Any]]:
    case = _case(tmp_path)
    workdir = tmp_path / "work"
    shutil.copytree(case.path, workdir)
    previous = run_eval.set_active_execution_config(run_eval.ExecutionConfig(executor="local"))
    try:
        return run_eval.run_one_condition(
            case=case,
            condition=Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=tmp_path / "ctx",
            patcher=patcher,
            k=k,
            output_format="json_compact",
            run_id="phase1",
            include_files_context=True,
            **{"force_patch_attempt": False, **inputs},
        )
    finally:
        run_eval.set_active_execution_config(previous)


def _attempts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if int(r.get("attempt", 0)) >= 1]


def test_att14_patch_verifier_runs_for_non_adaptive_conditions_by_default(
    tmp_path: Path,
) -> None:
    records = _run(tmp_path, _DiffPatcher([_FIX]), force_patch_attempt=False)
    record = _attempts(records)[0]
    assert record["patch_meta"]["patch_verifier_scope"] == "all"
    assert record["patch_meta"]["patch_verifier_enabled"] is True
    assert record["success"] is True


_UNAPPLIABLE = (
    "--- a/buggy.py\n+++ b/buggy.py\n@@ -7,2 +7,2 @@\n"
    " def g() -> int:\n-    return 5\n+    return 6\n"
)


def test_att07_forced_marker_recovery_stays_apply_failure_and_is_reverted(
    tmp_path: Path,
) -> None:
    records = _run(
        tmp_path,
        _DiffPatcher([_UNAPPLIABLE]),
        force_patch_attempt=True,
        patch_verifier_scope="disabled",
    )
    record = _attempts(records)[0]
    meta = record["patch_meta"]
    assert record["note"] == "patch_apply_failed"
    assert record["success"] is False
    assert meta["runner_forced_patch_recovery"] is True
    assert meta["runner_forced_patch_recovery_reverted"] is True
    assert meta["patch_file_paths"] == ["buggy.py"]
    assert "llmdebug-force-attempt" not in (tmp_path / "work" / "buggy.py").read_text(
        encoding="utf-8"
    )


def parsed_cli_defaults(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Parse ``run_eval`` CLI defaults without running an eval."""
    captured: dict[str, Any] = {}

    def _capture(*, parser: Any, args: Any, argv_tokens: Any) -> int:
        del parser, argv_tokens
        captured.update(vars(args))
        return 0

    monkeypatch.setattr("evals.eval_main_flow.run_main_flow", _capture)
    assert run_eval.main([]) == 0
    return captured


def test_att07_force_patch_attempt_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    assert parsed_cli_defaults(monkeypatch)["force_patch_attempt"] is False


def test_llm13_patcher_forced_marker_diff_is_recognized_as_noop() -> None:
    from evals.eval_patch_constants import FORCED_PATCH_MARKER
    from evals.eval_patches import _is_forced_noop_marker_diff

    patcher_style = (
        "--- a/buggy.py\n+++ b/buggy.py\n@@ -1,2 +1,3 @@\n"
        f" def f() -> int:\n     return 1\n+{FORCED_PATCH_MARKER}\n"
    )
    runner_style = patcher_style.replace(FORCED_PATCH_MARKER, f"{FORCED_PATCH_MARKER}-2")
    assert _is_forced_noop_marker_diff(patcher_style)
    assert _is_forced_noop_marker_diff(runner_style)
    assert not _is_forced_noop_marker_diff(_FIX)


def test_att16_failed_strict_patch_leaves_no_rej_or_orig_files(tmp_path: Path) -> None:
    from evals.eval_patches import apply_diff

    (tmp_path / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "keep.orig").write_text("pre-existing\n", encoding="utf-8")
    ok, _msg = apply_diff(tmp_path, _UNAPPLIABLE)
    assert ok is False
    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.suffix in {".rej", ".orig"})
    assert leftovers == ["keep.orig"]


def test_att10_relaxed_application_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals.eval_patches import RELAXED_APPLY_MESSAGE

    def _relaxed_apply(cwd: Path, diff: str) -> tuple[bool, str]:
        del diff
        (cwd / "buggy.py").write_text("def f() -> int:\n    return 2\n", encoding="utf-8")
        return True, RELAXED_APPLY_MESSAGE

    monkeypatch.setattr(run_eval, "apply_diff", _relaxed_apply)
    records = _run(tmp_path, _DiffPatcher([_FIX]), patch_verifier_scope="disabled")
    record = _attempts(records)[0]
    assert record["patch_meta"]["patch_apply_mode"] == "relaxed_fallback"
    assert record["success"] is True


def test_att10_strict_application_is_recorded(tmp_path: Path) -> None:
    records = _run(tmp_path, _DiffPatcher([_FIX]), patch_verifier_scope="disabled")
    assert _attempts(records)[0]["patch_meta"]["patch_apply_mode"] == "strict"


def test_att10_analysis_counts_relaxed_and_test_surface_attempts() -> None:
    from evals.analyze_results import _derive_case_condition_rows

    def _record(attempt: int, **patch_meta: Any) -> dict[str, Any]:
        return {
            "run_id": "r",
            "case_id": "c",
            "condition": "base_repair",
            "attempt": attempt,
            "success": False,
            "patch_meta": patch_meta,
        }

    rows = _derive_case_condition_rows(
        {
            ("r", "c", "base_repair"): [
                _record(1, patch_apply_mode="relaxed_fallback", test_surface_modified=True),
                _record(2, patch_apply_mode="strict", test_surface_modified=False),
            ]
        }
    )
    assert rows[0]["relaxed_apply_attempts"] == 1
    assert rows[0]["test_surface_modified_attempts"] == 1


def test_orc06_abbreviated_flags_are_rejected(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("evals.eval_main_flow.run_main_flow", lambda **_kw: 0)
    with pytest.raises(SystemExit) as excinfo:
        run_eval.main(["--job", "3"])
    assert excinfo.value.code == 2
    assert "unrecognized arguments: --job" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("conditions", "message"),
    [("base_repair,base_repair", "duplicate condition"), (",", "no conditions given")],
)
def test_orc15_conditions_are_validated(conditions: str, message: str, tmp_path: Path) -> None:
    from argparse import Namespace

    from evals.eval_config import ConfigResolutionState

    args = Namespace(conditions=conditions, patcher="openai", native_baseline_mode="")
    with pytest.raises(ValueError, match=message):
        run_eval._resolve_conditions_for_run(
            args=args, config_state=ConfigResolutionState(config_paths=[], value_sources={})
        )


@pytest.mark.parametrize("run_id", ["..", "a/b", "../escape", "run id", "x" * 129])
def test_orc21_invalid_run_ids_are_rejected(run_id: str, capsys) -> None:
    exit_code = run_eval.main(["--run-id", run_id, "--patcher", "null", "--executor", "local"])
    assert exit_code == 2
    assert "Invalid --run-id" in capsys.readouterr().out


def test_pol05_pol23_budget_controllers_are_per_condition_with_own_seeds(
    tmp_path: Path,
) -> None:
    from evals.adaptive_budget_online_policy import BudgetPolicyConfig
    from evals.eval_main_flow import _per_condition_budget_configs

    base = BudgetPolicyConfig.normalize(
        mode="on",
        state_path=tmp_path / "budget.json",
        events_path=tmp_path / "budget_events.jsonl",
    )
    both = _per_condition_budget_configs(
        base,
        conditions=[
            Condition.BASE_REPAIR,
            Condition.LAYERED_ADAPTIVE_GATED,
            Condition.LAYERED_ADAPTIVE_LLM,
        ],
        seed=7,
    )
    assert sorted(both) == ["layered_adaptive_gated", "layered_adaptive_llm"]
    gated, llm = both["layered_adaptive_gated"], both["layered_adaptive_llm"]
    assert gated.state_path == tmp_path / "budget.layered_adaptive_gated.json"
    assert llm.events_path == tmp_path / "budget_events.layered_adaptive_llm.jsonl"
    assert gated.random_seed != llm.random_seed
    assert gated.random_seed is not None and gated.random_seed != 7

    single = _per_condition_budget_configs(
        base, conditions=[Condition.LAYERED_ADAPTIVE_LLM], seed=None
    )
    assert single["layered_adaptive_llm"].state_path == tmp_path / "budget.json"
    assert single["layered_adaptive_llm"].random_seed is None


def test_pol05_keyed_decisions_do_not_depend_on_decision_order() -> None:
    from evals.adaptive_online_policy import AdaptiveOnlinePolicyController, OnlinePolicyConfig

    def _controller() -> AdaptiveOnlinePolicyController:
        return AdaptiveOnlinePolicyController(
            OnlinePolicyConfig.normalize(mode="on", warmup_decisions=1000, random_seed=11)
        )

    features = {"bias": 1.0, "x": 0.3}
    first = _controller()
    keyed_first = first.decide(feature_values=features, rng_key="case-a|llm|1").selected_action

    second = _controller()
    for index in range(5):  # other units decide first under a different interleaving
        second.decide(feature_values=features, rng_key=f"case-b|llm|{index}")
    keyed_second = second.decide(feature_values=features, rng_key="case-a|llm|1").selected_action
    assert keyed_first == keyed_second

    draws = {
        _controller().decide(feature_values=features, rng_key=f"k{i}").selected_action
        for i in range(30)
    }
    assert len(draws) > 1  # different keys give different streams


def test_derive_seed_is_stable_and_separates_streams() -> None:
    from evals.seed_derivation import derive_seed

    assert derive_seed(1, "helper") == derive_seed(1, "helper")
    assert derive_seed(1, "helper") != derive_seed(1, "budget")
    assert derive_seed(1, "helper") != derive_seed(2, "helper")
    assert 0 <= derive_seed(123, "x", 4) < 2**63


def test_bas01_baseline_case_support_counts_convertible_cases(tmp_path: Path) -> None:
    from baselines.ldb.adapter import LDBAdapter
    from evals.patchers.baseline_patcher import baseline_case_support

    simple = tmp_path / "simple"
    simple.mkdir()
    (simple / "buggy.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (simple / "test_buggy.py").write_text(
        "from buggy import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "buggy.py").write_text("def add(a, b:\n", encoding="utf-8")
    (broken / "test_buggy.py").write_text("from buggy import add\n", encoding="utf-8")

    support = baseline_case_support(LDBAdapter(), [("simple", simple), ("broken", broken)])
    assert support == {
        "baseline": "ldb",
        "total_cases": 2,
        "convertible_cases": 1,
        "incompatible_reasons": {"buggy_syntax_error": 1},
    }


def test_bas01_condition_summary_excludes_unsupported_cases() -> None:
    from evals.analyze_results import _derive_case_condition_rows, _summarize_bucket

    def _record(case_id: str, success: bool, **patch_meta: Any) -> dict[str, Any]:
        return {
            "run_id": "r",
            "case_id": case_id,
            "condition": "ldb",
            "attempt": 1,
            "success": success,
            "patch_meta": patch_meta,
        }

    rows = _derive_case_condition_rows(
        {
            ("r", "c1", "ldb"): [_record("c1", True)],
            ("r", "c2", "ldb"): [
                _record("c2", False, baseline_supported=False, compatibility_reason="x")
            ],
        }
    )
    summary = _summarize_bucket(rows)["ldb"]
    assert summary["success_rate"] == 1.0
    assert summary["baseline_unsupported_cases"] == 1


def test_d4_every_baseline_gets_one_generation_contract(tmp_path: Path) -> None:
    """BAS-03/04/08/11: one config, recorded per record with an honest enforcement flag."""
    from baselines.adapter_base import BaselineGenerationSettings
    from evals.patchers.baseline_patcher import get_adapter_registry

    generation = BaselineGenerationSettings(
        temperature=0.0, seed=5, max_completion_tokens=2048, transport_max_retries=6
    )
    enforced = {}
    for name, adapter_cls in sorted(get_adapter_registry().items()):
        adapter = adapter_cls()
        enforced[name] = generation.record(
            enforced=bool(getattr(adapter, "honours_generation_contract", False))
        )
    assert {name: record["enforced"] for name, record in enforced.items()} == {
        "intervenor": True,
        "lambda_rlm": True,
        "ldb": False,
        "rlm": True,
        "tracecoder": False,
    }
    assert {record["max_completion_tokens"] for record in enforced.values()} == {2048}
    assert generation.to_env() == {
        "TRACEGATE_BASELINE_TEMPERATURE": "0.0",
        "TRACEGATE_BASELINE_SEED": "5",
        "TRACEGATE_BASELINE_MAX_TOKENS": "2048",
    }


def test_d4_intervenor_request_uses_the_shared_contract() -> None:
    from baselines.adapter_base import BaselineConfig, BaselineGenerationSettings
    from baselines.intervenor import adapter as intervenor

    config = BaselineConfig(
        generation=BaselineGenerationSettings(
            temperature=0.3, seed=9, max_completion_tokens=1000, transport_max_retries=4
        )
    )
    assert intervenor._generation_temperature(config) == 0.3
    assert intervenor._generation_max_tokens(config, 6144) == 1000
    assert intervenor._generation_seed_kwargs(config) == {"seed": 9}
    legacy = BaselineConfig()
    assert intervenor._generation_temperature(legacy) == 0.0
    assert intervenor._generation_max_tokens(legacy, 6144) == 6144


def test_bas12_tracecoder_test_timeout_falls_back_to_process_timeout() -> None:
    from baselines.adapter_base import BaselineConfig
    from baselines.tracecoder.adapter import _case_test_timeout_sec

    config = BaselineConfig(process_timeout_sec=600)
    assert _case_test_timeout_sec({"timeout_sec": 12}, config) == 12
    assert _case_test_timeout_sec({}, config) == 600
    assert _case_test_timeout_sec({"timeout_sec": 0}, config) == 600
