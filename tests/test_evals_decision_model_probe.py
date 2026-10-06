"""Tests for the offline decision-model probe (``evals.decision_model.probe``)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals import pytest_runner
from evals.decision_model import probe
from evals.decision_model.client import SystemOneClient
from evals.decision_model.config import (
    DECISION_MODEL_API_KEY_ENV,
    ROLE_SHADOW,
    DecisionModelClientConfig,
    DecisionModelPolicyConfig,
    DecisionModelSettings,
)
from evals.decision_model.runtime import SystemOneAdvisor
from evals.decision_model.schema import (
    DEFAULT_PROFILE_NAME,
    QUESTION_FIX_CONFIDENCE,
    QUESTION_MISSING_INFORMATION,
    QUESTION_NEXT_ACTION,
    QUESTION_TRACEBACK_SUFFICIENT,
    get_profile,
)

CASE_JSON = {
    "category": "async_temporal",
    "difficulty": "medium",
    "snapshot_dependency": "required",
    "expected_exception": "AssertionError",
    "description": "oracle text that must never reach the decision model",
    "python_args": ["-m", "pytest", "-q"],
    "timeout_sec": 30,
}
PYTEST_STDOUT = """\
FAILED test_buggy.py::test_total - AssertionError: expected 3, got 2
E       AssertionError: expected 3, got 2
E       assert 2 == 3
1 failed in 0.03s
"""
SNAPSHOT: dict[str, Any] = {
    "exception": {"type": "AssertionError", "message": "expected 3, got 2"},
    "pytest": {"nodeid": "test_buggy.py::test_total"},
    "frames": [
        {
            "file_rel": "buggy.py",
            "line": 4,
            "function": "total",
            "code": "return a + b - 1",
            "locals": [{"name": "a", "type": "int", "preview": "1"}],
        }
    ],
}


# --------------------------------------------------------------------------------------------
# Split assignment
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("async_cancel_swallowed_easy", "async_cancel_swallowed"),
        ("async_cancel_swallowed_medium", "async_cancel_swallowed"),
        ("async_cancel_swallowed_hard", "async_cancel_swallowed"),
        ("swebench_django__django_1234", "swebench_django__django_1234"),
        ("_hard", "_hard"),
    ],
)
def test_case_family_strips_only_difficulty_suffixes(case_id: str, expected: str) -> None:
    assert probe.case_family(case_id) == expected


def test_split_is_stable_and_shared_within_a_family() -> None:
    variants = [f"async_cancel_swallowed_{level}" for level in ("easy", "medium", "hard")]
    splits = {probe.split_for_case(case_id) for case_id in variants}
    assert len(splits) == 1
    assert splits <= {probe.SPLIT_DEV, probe.SPLIT_TEST}
    assert probe.split_for_case(variants[0]) == probe.split_for_case(variants[0])


def test_split_assigns_roughly_one_fifth_to_test() -> None:
    families = [f"case_family_{index}" for index in range(400)]
    test_count = sum(1 for family in families if probe.split_for_case(family) == probe.SPLIT_TEST)
    assert 40 <= test_count <= 120


# --------------------------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------------------------


def test_average_ranks_shares_rank_between_ties() -> None:
    assert probe.average_ranks([10.0, 20.0, 20.0, 40.0]) == [1.0, 2.5, 2.5, 4.0]


def test_auroc_perfect_and_inverted_separation() -> None:
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [False, False, True, True]
    assert probe.auroc(scores, labels) == pytest.approx(1.0)
    assert probe.auroc([-value for value in scores], labels) == pytest.approx(0.0)


def test_auroc_averages_ties_to_one_half() -> None:
    assert probe.auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == pytest.approx(0.5)
    # One tied pair between the classes: 2 clean wins + half credit for the tie out of 4 pairs.
    assert probe.auroc([0.3, 0.5, 0.5, 0.9], [False, False, True, True]) == pytest.approx(0.875)


def test_auroc_returns_none_without_both_classes() -> None:
    assert probe.auroc([0.1, 0.9], [True, True]) is None
    assert probe.auroc([], []) is None
    assert probe.auroc([0.1, 0.9], [True]) is None


def test_spearman_known_values() -> None:
    assert probe.spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert probe.spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    # Monotone but non-linear: Spearman is 1 where Pearson would not be.
    assert probe.spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]) == pytest.approx(1.0)
    # Ties on one side still produce a finite, lower correlation.
    rho = probe.spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.0, 3.0])
    assert rho is not None and 0.9 < rho < 1.0


def test_spearman_returns_none_for_degenerate_inputs() -> None:
    assert probe.spearman([1.0], [1.0]) is None
    assert probe.spearman([1.0, 2.0], [5.0, 5.0]) is None
    assert probe.spearman([1.0, 2.0], [1.0]) is None


def test_calibration_bins_group_scores_and_rates() -> None:
    bins = probe.calibration_bins([0.05, 0.15, 0.95, 1.0], [False, True, True, True])
    assert len(bins) == probe.CALIBRATION_BIN_COUNT
    assert bins[0]["count"] == 2
    assert bins[0]["mean_score"] == pytest.approx(0.1)
    assert bins[0]["observed_rate"] == pytest.approx(0.5)
    # 1.0 belongs to the last bin, not to a sixth one.
    assert bins[-1]["count"] == 2
    assert bins[-1]["observed_rate"] == pytest.approx(1.0)
    assert bins[2]["count"] == 0 and bins[2]["mean_score"] is None


def test_percentile_interpolates() -> None:
    assert probe.percentile([], 0.5) is None
    assert probe.percentile([7.0], 0.95) == pytest.approx(7.0)
    assert probe.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert probe.percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


# --------------------------------------------------------------------------------------------
# Available actions
# --------------------------------------------------------------------------------------------


def test_available_actions_include_lite_arms_only_when_they_change_the_prompt() -> None:
    full_replace = probe.available_actions_for_attempt_one(
        snapshot_available=True, reveal_mode="full", lite_snapshot_mode="replace"
    )
    assert full_replace == (
        "skip_helper",
        "invoke_helper",
        "frame_only",
        "file_only",
        "search_only",
        "snapshot_lite",
        "frame_locals_lite",
        "snapshot_full_lite",
    )
    assert probe.available_actions_for_attempt_one(
        snapshot_available=False, reveal_mode="full", lite_snapshot_mode="replace"
    ) == ("skip_helper", "invoke_helper", "frame_only", "file_only", "search_only")
    assert probe.available_actions_for_attempt_one(
        snapshot_available=True, reveal_mode="full", lite_snapshot_mode="supplement"
    ) == ("skip_helper", "invoke_helper", "frame_only", "file_only", "search_only")
    assert (
        probe.available_actions_for_attempt_one(
            snapshot_available=True,
            reveal_mode="traceback_first",
            lite_snapshot_mode="supplement",
        )
        == full_replace
    )
    # compact_then_patch and retry_helper_strict_json are never feasible on attempt 1.
    assert "compact_then_patch" not in full_replace
    assert "retry_helper_strict_json" not in full_replace


# --------------------------------------------------------------------------------------------
# Option-key renaming
# --------------------------------------------------------------------------------------------


def _systemone_body(options: list[str]) -> dict[str, Any]:
    share = 1.0 / len(options)
    probabilities = dict.fromkeys(options, share)
    probabilities[options[0]] = share + 0.1
    return {
        "model": "clef-flash",
        "answers": {
            QUESTION_NEXT_ACTION: {"probabilities": probabilities, "choice": options[0]},
            QUESTION_TRACEBACK_SUFFICIENT: {"noul": 0.25},
            QUESTION_MISSING_INFORMATION: {
                "probabilities": {
                    "none": 0.05,
                    "runtime_values": 0.6,
                    "call_path": 0.1,
                    "source_definition": 0.1,
                    "project_usage": 0.1,
                    "several_or_unclear": 0.05,
                }
            },
            QUESTION_FIX_CONFIDENCE: {"score": 1.0},
        },
        "usage": {"input_tokens": 1234},
    }


def test_every_helper_action_has_a_unique_explicit_permuted_key() -> None:
    from evals.decision_model.schema import HELPER_ACTIONS

    renamed = [probe.permuted_option_key(action) for action in HELPER_ACTIONS]
    # A missing entry would silently fall back to the shared "m" prefix and collide.
    assert [name for name in renamed if name.startswith("m_")] == []
    assert len(set(renamed)) == len(HELPER_ACTIONS)


def test_permuted_option_keys_scramble_alphabetical_order() -> None:
    assert probe.permuted_option_key("skip_helper") == "z_skip_helper"
    assert probe.permuted_option_key("frame_only") == "a_frame_only"
    actions = ["skip_helper", "invoke_helper", "frame_only", "file_only", "search_only"]
    renamed = [probe.permuted_option_key(action) for action in actions]
    assert len(set(renamed)) == len(renamed)
    original_order = sorted(actions)
    permuted_order = [name.split("_", 1)[1] for name in sorted(renamed)]
    assert permuted_order != original_order


def test_permute_and_restore_round_trip() -> None:
    actions = ["skip_helper", "invoke_helper", "frame_only"]
    request = {
        "model": "clef-flash",
        "state": {"0_note": "x"},
        "questions": {
            QUESTION_NEXT_ACTION: {
                "type": "choice",
                "instructions": "pick",
                "criteria": {action: {"choose_when": action} for action in actions},
            },
            QUESTION_TRACEBACK_SUFFICIENT: {"type": "noul", "criteria": {"true": "t"}},
        },
    }
    permuted = probe.permute_request_options(request)
    criteria = permuted["questions"][QUESTION_NEXT_ACTION]["criteria"]
    assert sorted(criteria) == sorted(probe.permuted_option_key(a) for a in actions)
    # Untouched questions and the rest of the body survive unchanged.
    assert permuted["questions"][QUESTION_TRACEBACK_SUFFICIENT] == {
        "type": "noul",
        "criteria": {"true": "t"},
    }
    assert permuted["state"] == {"0_note": "x"}
    assert request["questions"][QUESTION_NEXT_ACTION]["criteria"] == {
        action: {"choose_when": action} for action in actions
    }

    restored = probe.restore_response_options(
        _systemone_body([probe.permuted_option_key(action) for action in actions])
    )
    answer = restored["answers"][QUESTION_NEXT_ACTION]
    assert sorted(answer["probabilities"]) == sorted(actions)
    assert answer["choice"] == "skip_helper"
    assert restored["answers"][QUESTION_TRACEBACK_SUFFICIENT] == {"noul": 0.25}


def test_restore_is_a_noop_without_a_main_question() -> None:
    body = {"answers": {QUESTION_TRACEBACK_SUFFICIENT: {"noul": 0.5}}}
    assert probe.restore_response_options(body) == body
    assert probe.permute_request_options({"questions": "not-a-mapping"}) == {
        "questions": "not-a-mapping"
    }


def _mock_transport(seen: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"model": "clef-flash", "revision": "abc123"})
        body = json.loads(request.content)
        seen.append(body)
        options = list(body["questions"][QUESTION_NEXT_ACTION]["criteria"])
        return httpx.Response(200, json=_systemone_body(sorted(options)))

    return httpx.MockTransport(handler)


def _advisor(*, transport: httpx.BaseTransport, profile: str = DEFAULT_PROFILE_NAME) -> Any:
    settings = DecisionModelSettings(
        policy=DecisionModelPolicyConfig(role=ROLE_SHADOW, profile=profile),
        client=DecisionModelClientConfig(url="http://sidecar.invalid", timeout_sec=5.0),
    )
    return SystemOneAdvisor(settings, client=SystemOneClient(settings.client, transport=transport))


def test_option_key_permuting_transport_round_trips_through_the_advisor() -> None:
    seen: list[dict[str, Any]] = []
    transport = probe.OptionKeyPermutingTransport(_mock_transport(seen))
    advisor = _advisor(transport=transport)
    actions = ("skip_helper", "invoke_helper", "frame_only")
    try:
        advice = advisor.advise(
            probe.AdviceRequest(
                state_inputs=probe.DecisionStateInputs(stdout=PYTEST_STDOUT),
                available_actions=actions,
            )
        )
    finally:
        advisor.close()
    assert advice.status == "ok", advice.error
    # The sidecar saw permuted ids ...
    assert sorted(seen[0]["questions"][QUESTION_NEXT_ACTION]["criteria"]) == sorted(
        probe.permuted_option_key(action) for action in actions
    )
    # ... while the advisor parsed the answer under the original action ids.
    assert sorted(advice.answers.action_probs) == sorted(actions)
    assert advice.answers.action_probs["frame_only"] == pytest.approx(
        max(advice.answers.action_probs.values())
    )
    assert advice.input_tokens == 1234


def test_advisor_without_permutation_sends_the_original_option_ids() -> None:
    seen: list[dict[str, Any]] = []
    advisor = _advisor(transport=_mock_transport(seen))
    actions = ("skip_helper", "invoke_helper", "frame_only")
    try:
        advice = advisor.advise(
            probe.AdviceRequest(
                state_inputs=probe.DecisionStateInputs(stdout=PYTEST_STDOUT),
                available_actions=actions,
            )
        )
    finally:
        advisor.close()
    assert advice.status == "ok", advice.error
    assert sorted(seen[0]["questions"][QUESTION_NEXT_ACTION]["criteria"]) == sorted(actions)
    assert advice.answers.traceback_sufficient == pytest.approx(0.25)
    assert advice.answers.fix_confidence == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------------------------
# Report aggregation
# --------------------------------------------------------------------------------------------


def _record(
    case_id: str,
    *,
    snapshot_dependency: str,
    difficulty: str = "medium",
    traceback_sufficient: float = 0.5,
    runtime_values: float = 0.3,
    pregate_score: float = 0.5,
    fix_confidence: float = 0.5,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "family": probe.case_family(case_id),
        "split": probe.SPLIT_DEV,
        "labels": {
            "category": "async_temporal",
            "difficulty": difficulty,
            "snapshot_dependency": snapshot_dependency,
            "bug_source": "hand_crafted",
        },
        "pregate": {"score": pregate_score, "should_skip": False, "reason": "X_tierC"},
        "advice": {
            "status": "ok",
            "action_probs": {"skip_helper": 0.4, "frame_only": 0.6},
            "top_action": "frame_only",
            "traceback_sufficient": traceback_sufficient,
            "missing_information": {"runtime_values": runtime_values, "call_path": 0.0},
            "fix_confidence": fix_confidence,
            "latency_ms": 100.0,
            "state_chars": 3000,
        },
    }


def test_build_report_scores_the_decision_model_against_the_labels() -> None:
    records = [
        _record("a_easy", snapshot_dependency="none", traceback_sufficient=0.9, difficulty="easy"),
        _record("b_easy", snapshot_dependency="none", traceback_sufficient=0.8, difficulty="easy"),
        _record(
            "c_hard",
            snapshot_dependency="required",
            traceback_sufficient=0.2,
            runtime_values=0.9,
            difficulty="hard",
            fix_confidence=0.1,
        ),
        _record(
            "d_medium",
            snapshot_dependency="helpful",
            traceback_sufficient=0.3,
            runtime_values=0.7,
            fix_confidence=0.3,
        ),
    ]
    report = probe.build_report(records)
    assert report["cases"] == 4
    assert report["counts"]["by_status"] == {"ok": 4}
    assert report["counts"]["by_split"] == {probe.SPLIT_DEV: 4}
    assert report["scored_cases"] == 4
    auroc = report["auroc"]["snapshot_dependency_required_or_helpful"]
    assert auroc["dm_not_traceback_sufficient"] == pytest.approx(1.0)
    assert auroc["dm_need_runtime_values"] == pytest.approx(1.0)
    # The pregate score is constant here, so it cannot separate the classes.
    assert auroc["pregate_uninformative"] == pytest.approx(0.5)
    assert report["auroc"]["snapshot_dependency_required"]["dm_not_traceback_sufficient"] == (
        pytest.approx(1.0)
    )
    spearman = report["spearman"]["fix_confidence_vs_difficulty"]
    assert spearman["n"] == 4
    assert spearman["rho"] is not None and spearman["rho"] < 0.0
    assert report["action_mix_by_category"] == {"async_temporal": {"frame_only": 4}}
    calibration = report["calibration"]["traceback_sufficient_vs_snapshot_independent"]
    assert sum(entry["count"] for entry in calibration) == 4
    assert calibration[-1]["observed_rate"] == pytest.approx(1.0)
    assert report["latency_ms"]["p50"] == pytest.approx(100.0)
    assert report["latency_ms"]["p95"] == pytest.approx(100.0)
    assert report["state_chars"]["max"] == pytest.approx(3000.0)
    assert set(report["by_split"]) == {probe.SPLIT_DEV}


def test_build_report_ignores_unusable_records() -> None:
    usable = _record("a_easy", snapshot_dependency="none")
    failed = {
        "case_id": "b_easy",
        "split": probe.SPLIT_TEST,
        "labels": {"category": "x", "difficulty": "easy", "snapshot_dependency": "none"},
        "advice": {"status": "timeout", "error": "ReadTimeout"},
    }
    no_aux = _record("c_easy", snapshot_dependency="required")
    no_aux["advice"]["traceback_sufficient"] = None
    dry = _record("d_easy", snapshot_dependency="required")
    dry["advice"] = {"status": probe.STATUS_DRY_RUN, "state_chars": 500}

    report = probe.build_report([usable, failed, no_aux, dry])
    assert report["cases"] == 4
    assert report["counts"]["by_status"] == {"dry_run": 1, "ok": 2, "timeout": 1}
    assert report["counts"]["by_split"] == {probe.SPLIT_DEV: 3, probe.SPLIT_TEST: 1}
    assert report["scored_cases"] == 1
    # One class only -> no AUROC, but the report still renders.
    assert report["auroc"]["snapshot_dependency_required"]["dm_need_runtime_values"] is None
    # The timed-out record never rendered a state, so it is not counted.
    assert report["state_chars"]["n"] == 3


def test_build_report_on_no_records() -> None:
    report = probe.build_report([])
    assert report["cases"] == 0
    assert report["scored_cases"] == 0
    assert report["by_split"] == {}
    assert report["latency_ms"] == {"n": 0, "p50": None, "p95": None}


# --------------------------------------------------------------------------------------------
# End-to-end probe over a synthetic case directory
# --------------------------------------------------------------------------------------------


def _write_case(cases_dir: Path, case_id: str) -> Path:
    case_path = cases_dir / case_id
    case_path.mkdir(parents=True)
    (case_path / "case.json").write_text(json.dumps(CASE_JSON), encoding="utf-8")
    (case_path / "buggy.py").write_text(
        "def total(a, b):\n    return a + b - 1\n", encoding="utf-8"
    )
    (case_path / "test_buggy.py").write_text(
        "from buggy import total\n\n\ndef test_total():\n    assert total(1, 2) == 3\n",
        encoding="utf-8",
    )
    (case_path / "oracle.patch").write_text("", encoding="utf-8")
    return case_path


@pytest.fixture
def fake_pytest(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace the pytest run and snapshot read so the probe stays fast and hermetic."""
    import evals.run_eval as run_eval
    from evals.eval_types import RunResult

    seen_workdirs: list[Path] = []

    def fake_run_pytest(
        cwd: Path,
        python_args: list[str],
        condition: Any,
        timeout_sec: int,
        output_format: str,
        **_: Any,
    ) -> RunResult:
        seen_workdirs.append(Path(cwd))
        return RunResult(
            returncode=1,
            duration_sec=0.05,
            stdout=PYTEST_STDOUT,
            stderr="",
            timed_out=False,
            timeout_sec=timeout_sec,
        )

    monkeypatch.setattr(pytest_runner, "run_pytest", fake_run_pytest)
    monkeypatch.setattr(pytest_runner, "read_latest_snapshot", lambda _dir: (dict(SNAPSHOT), None))
    return seen_workdirs


def _case_spec(cases_dir: Path) -> Any:
    from evals.run_eval import load_cases

    cases = load_cases(cases_dir)
    assert len(cases) == 1
    return cases[0]


def test_probe_case_end_to_end(tmp_path: Path, fake_pytest: list[Path]) -> None:
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "async_cancel_swallowed_medium")
    case = _case_spec(cases_dir)
    options = probe.ProbeOptions(profile=get_profile(DEFAULT_PROFILE_NAME), run_id="dmprobe-test1")
    seen: list[dict[str, Any]] = []
    advisor = _advisor(transport=_mock_transport(seen))
    try:
        record = probe.probe_case(case, advisor=advisor, options=options)
    finally:
        advisor.close()

    assert record["case_id"] == "async_cancel_swallowed_medium"
    assert record["family"] == "async_cancel_swallowed"
    assert record["split"] in {probe.SPLIT_DEV, probe.SPLIT_TEST}
    assert record["labels"]["snapshot_dependency"] == "required"
    assert record["pytest"] == {"returncode": 1, "timed_out": False, "duration_sec": 0.05}
    assert record["snapshot_available"] is True
    assert record["pregate"]["reason"].startswith("AssertionError")
    assert record["available_actions"][:5] == [
        "skip_helper",
        "invoke_helper",
        "frame_only",
        "file_only",
        "search_only",
    ]
    advice = record["advice"]
    assert advice["status"] == "ok", advice["error"]
    assert sorted(advice["action_probs"]) == sorted(record["available_actions"])
    assert advice["top_action"] in record["available_actions"]
    assert advice["traceback_sufficient"] == pytest.approx(0.25)
    assert advice["missing_information"]["runtime_values"] == pytest.approx(0.6)
    assert advice["input_tokens"] == 1234
    assert advice["state_chars"] > 0 and len(advice["state_sha256"]) == 64
    assert "state" not in record

    # The case was copied into a neutral temp dir and stripped of its eval-only artifacts.
    workdir = fake_pytest[0]
    assert "async_cancel" not in str(workdir)
    assert workdir.name.startswith("llmdebug-dmprobe-")
    # No label, description or identifier reached the decision model.
    state = json.dumps(seen[0]["state"])
    for secret in ("async_cancel_swallowed", "oracle text", "async_temporal", "medium", "required"):
        assert secret not in state


def test_probe_case_dry_run_records_the_state(tmp_path: Path, fake_pytest: list[Path]) -> None:
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "async_cancel_swallowed_medium")
    case = _case_spec(cases_dir)
    options = probe.ProbeOptions(
        profile=get_profile(DEFAULT_PROFILE_NAME),
        run_id="dmprobe-test2",
        dry_run=True,
    )
    record = probe.probe_case(case, advisor=None, options=options)
    assert record["advice"]["status"] == probe.STATUS_DRY_RUN
    assert record["advice"]["state_chars"] == len(record["state"])
    assert record["state"].startswith('{"0_note"')
    assert "case.json" not in record["state"]
    assert fake_pytest and not (fake_pytest[0] / "case.json").exists()


def test_probe_case_records_execution_failures(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "async_cancel_swallowed_medium")
    case = _case_spec(cases_dir)
    options = probe.ProbeOptions(profile=get_profile(DEFAULT_PROFILE_NAME))
    missing = type(case)(**{**case.__dict__, "path": tmp_path / "does-not-exist"})
    record = probe.probe_case(missing, advisor=None, options=options)
    assert record["advice"]["status"].startswith("error:")
    assert record["labels"]["category"] == "async_temporal"


def test_run_probe_with_threads_returns_every_case(tmp_path: Path, fake_pytest: list[Path]) -> None:
    cases_dir = tmp_path / "cases"
    for index in range(3):
        _write_case(cases_dir, f"async_case_{index}_medium")
    from evals.run_eval import load_cases

    cases = load_cases(cases_dir)
    options = probe.ProbeOptions(profile=get_profile(DEFAULT_PROFILE_NAME), dry_run=True)
    written: list[dict[str, Any]] = []
    records = probe.run_probe(cases, advisor=None, options=options, jobs=3, writer=written.append)
    assert len(records) == 3
    assert {record["case_id"] for record in records} == {case.case_id for case in cases}
    assert len(written) == 3


def test_main_dry_run_writes_jsonl_and_report(
    tmp_path: Path,
    fake_pytest: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import evals.validate_cases as validate_cases

    monkeypatch.setattr(validate_cases, "check_pytest_available", lambda: (True, "ok"))
    cases_dir = tmp_path / "cases"
    for index in range(2):
        _write_case(cases_dir, f"async_case_{index}_medium")
    out_path = tmp_path / "out" / "probe.jsonl"

    exit_code = probe.main(
        [
            "--cases-dir",
            str(cases_dir),
            "--split",
            "all",
            "--executor",
            "local",
            "--dry-run",
            "--report",
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 0
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert all(record["advice"]["status"] == probe.STATUS_DRY_RUN for record in records)
    report = json.loads((tmp_path / "out" / "probe.report.json").read_text(encoding="utf-8"))
    assert report["cases"] == 2
    assert report["counts"]["by_status"] == {probe.STATUS_DRY_RUN: 2}
    assert math.isclose(
        float(report["state_chars"]["p50"]), float(records[0]["advice"]["state_chars"])
    )
    assert "Probed 2 case(s)" in capsys.readouterr().out


def test_select_cases_applies_filters_the_split_and_the_limit(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    for index in range(6):
        _write_case(cases_dir, f"async_case_{index}_medium")
    parser = probe.build_parser()
    all_cases = probe.select_cases(
        parser.parse_args(["--cases-dir", str(cases_dir), "--split", "all"])
    )
    assert len(all_cases) == 6
    dev = probe.select_cases(parser.parse_args(["--cases-dir", str(cases_dir)]))
    test = probe.select_cases(parser.parse_args(["--cases-dir", str(cases_dir), "--split", "test"]))
    assert len(dev) + len(test) == 6
    assert {case.case_id for case in dev} & {case.case_id for case in test} == set()
    limited = probe.select_cases(
        parser.parse_args(["--cases-dir", str(cases_dir), "--split", "all", "--limit", "2"])
    )
    assert len(limited) == 2
    assert (
        probe.select_cases(
            parser.parse_args(
                [
                    "--cases-dir",
                    str(cases_dir),
                    "--split",
                    "all",
                    "--case-id",
                    "async_case_3_medium",
                ]
            )
        )[0].case_id
        == "async_case_3_medium"
    )
    assert (
        probe.select_cases(
            parser.parse_args(
                ["--cases-dir", str(cases_dir), "--split", "all", "--category", "parse_value"]
            )
        )
        == []
    )


def test_options_from_args_uses_the_requested_profile_and_reveal_mode() -> None:
    args = probe.build_parser().parse_args(
        ["--profile", "evidence_atomic_v1", "--reveal-mode", "traceback_first"]
    )
    options = probe.options_from_args(args, run_id="dmprobe-opts")
    assert options.profile.name == "evidence_atomic_v1"
    assert options.reveal_mode == "traceback_first"
    assert options.lite_snapshot_mode == "replace"
    assert options.run_id == "dmprobe-opts"
    assert options.dry_run is False


def test_build_advisor_wires_the_permuting_transport_only_when_asked() -> None:
    parser = probe.build_parser()
    advisor = probe.build_advisor(
        parser.parse_args(["--rename-option-keys", "--model", "clef-x", "--timeout-sec", "3"])
    )
    try:
        assert advisor.client.config.model == "clef-x"
        assert advisor.client.config.timeout_sec == pytest.approx(3.0)
        assert isinstance(advisor.client._client._transport, probe.OptionKeyPermutingTransport)
    finally:
        advisor.close()
    plain = probe.build_advisor(parser.parse_args([]))
    try:
        assert not isinstance(plain.client._client._transport, probe.OptionKeyPermutingTransport)
    finally:
        plain.close()


def test_main_reports_an_empty_selection(tmp_path: Path) -> None:
    (tmp_path / "cases").mkdir()
    assert probe.main(["--cases-dir", str(tmp_path / "cases"), "--executor", "local"]) == 2


def test_main_rejects_invalid_jobs(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "async_cancel_swallowed_medium")
    exit_code = probe.main(
        ["--cases-dir", str(cases_dir), "--executor", "local", "--jobs", "0", "--split", "all"]
    )
    assert exit_code == 2


def test_build_advisor_sends_the_gateway_token_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe must authenticate like the harness; otherwise llama-swap answers 401."""
    monkeypatch.setenv(DECISION_MODEL_API_KEY_ENV, "dm-secret")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json={"status": "ok", "revision": "abc123"})

    advisor = probe.build_advisor(probe.build_parser().parse_args([]))
    try:
        assert advisor.client.config.api_key == "dm-secret"
        # The resolved token reaches the wire as a bearer header.
        authenticated = SystemOneClient(
            advisor.client.config, transport=httpx.MockTransport(handler)
        )
        try:
            assert authenticated.health().ok
        finally:
            authenticated.close()
    finally:
        advisor.close()

    assert seen == ["Bearer dm-secret"]


def test_probe_case_keeps_unexpected_advisor_failures_inside_the_case(
    tmp_path: Path, fake_pytest: list[Path]
) -> None:
    """An httpx error the client does not map must not abort run_probe (codex m1#8)."""
    cases_dir = tmp_path / "cases"
    for index in range(2):
        _write_case(cases_dir, f"async_case_{index}_medium")
    from evals.run_eval import load_cases

    cases = load_cases(cases_dir)
    options = probe.ProbeOptions(profile=get_profile(DEFAULT_PROFILE_NAME), run_id="dmprobe-boom")
    seen: list[dict[str, Any]] = []
    queries = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"model": "clef-flash", "revision": "abc123"})
        queries["count"] += 1
        if queries["count"] == 1:
            raise httpx.DecodingError("corrupt gzip body", request=request)
        body = json.loads(request.content)
        seen.append(body)
        options_seen = list(body["questions"][QUESTION_NEXT_ACTION]["criteria"])
        return httpx.Response(200, json=_systemone_body(sorted(options_seen)))

    advisor = _advisor(transport=httpx.MockTransport(handler))
    try:
        records = probe.run_probe(cases, advisor=advisor, options=options, jobs=1)
    finally:
        advisor.close()

    assert len(records) == 2
    statuses = sorted(str(record["advice"]["status"]) for record in records)
    assert statuses == ["error:DecodingError", "ok"]
