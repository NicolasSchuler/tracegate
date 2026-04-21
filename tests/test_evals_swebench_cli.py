"""Tests for the SWE-bench CLI subcommands (generate, score, smoke)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.swebench import cli as swebench_cli
from evals.swebench.types import (
    ConditionGenerationSummary,
    GenerationRunSummary,
    ScoreRunSummary,
    ScoringSmokeSubsetSummary,
)


def test_generate_lite_cli_parses_and_forwards_jobs(monkeypatch, capsys, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        return GenerationRunSummary(
            run_id="demo-run",
            run_dir=tmp_path / "evals" / "swebench_runs" / "demo-run",
            instance_count=2,
            condition_summaries=(
                ConditionGenerationSummary(
                    condition="baseline",
                    predictions_path=tmp_path / "baseline" / "predictions.jsonl",
                    total_instances=2,
                    generated=2,
                    resumed=0,
                    empty_patch_count=0,
                    request_error_count=0,
                ),
            ),
        )

    monkeypatch.setattr(swebench_cli, "generate_lite_predictions", _fake_generate)
    exit_code = swebench_cli.main(
        [
            "generate-lite",
            "--run-id",
            "demo-run",
            "--conditions",
            "baseline,llmdebug",
            "--openai-base-url",
            "http://localhost:19000",
            "--openai-model",
            "local-model",
            "--no-openai-http2",
            "--openai-async",
            "--jobs",
            "4",
        ]
    )

    assert exit_code == 0
    assert captured["run_id"] == "demo-run"
    assert captured["jobs"] == 4
    assert captured["openai_http2"] is False
    assert captured["openai_async"] is True
    assert captured["conditions"] == "baseline,llmdebug"
    assert "Run directory:" in capsys.readouterr().out


def test_prepare_handoff_fails_when_one_condition_missing(monkeypatch, capsys, tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "instance_ids.txt").write_text("inst-1\n", encoding="utf-8")
    (run_dir / "run_metadata.json").write_text('{"run_id":"demo-run"}\n', encoding="utf-8")
    (run_dir / "baseline" / "predictions.jsonl").write_text(
        (
            '{"instance_id":"inst-1",'
            '"model_patch":"diff --git a/a.py b/a.py\\n",'
            '"model_name_or_path":"baseline-model"}\n'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(swebench_cli, "run_dir_for_id", lambda _run_id: run_dir)
    exit_code = swebench_cli.main(["prepare-handoff", "--run-id", "demo-run"])

    assert exit_code == 1
    assert "Handoff validation failed" in capsys.readouterr().out


def test_score_official_cli_surfaces_failure(monkeypatch, capsys, tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    def _fake_score(**_kwargs):
        raise RuntimeError("simulated scorer failure")

    monkeypatch.setattr(swebench_cli, "run_official_scoring", _fake_score)
    exit_code = swebench_cli.main(["score-official", "--run-dir", str(run_dir)])

    assert exit_code == 1
    assert "Official scoring failed: simulated scorer failure" in capsys.readouterr().out


def test_score_official_cli_success_prints_condition_summary(monkeypatch, capsys, tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    condition_log = run_dir / "official_eval" / "baseline" / "run_evaluation.log"
    condition_log.parent.mkdir(parents=True, exist_ok=True)
    condition_log.write_text("", encoding="utf-8")

    def _fake_score(**_kwargs):
        return ScoreRunSummary(
            run_dir=run_dir,
            condition_summaries=(),
        )

    monkeypatch.setattr(swebench_cli, "run_official_scoring", _fake_score)
    exit_code = swebench_cli.main(["score-official", "--run-dir", str(run_dir)])

    assert exit_code == 0
    assert "Official scoring complete for run dir" in capsys.readouterr().out


def test_generate_lite_cli_rejects_single_condition(monkeypatch, capsys) -> None:
    exit_code = swebench_cli.main(
        [
            "generate-lite",
            "--run-id",
            "demo-run",
            "--conditions",
            "baseline",
            "--openai-base-url",
            "http://localhost:19000",
            "--openai-model",
            "local-model",
        ]
    )

    assert exit_code == 1
    assert (
        "Generation failed: conditions must include exactly baseline,llmdebug"
        in capsys.readouterr().out
    )


def test_prepare_scoring_smoke_cli_success(monkeypatch, capsys, tmp_path) -> None:
    source = tmp_path / "src-run"
    source.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "smoke-run"

    def _fake_prepare(**kwargs):
        assert kwargs["source_run_dir"] == source
        assert kwargs["max_instances"] == 2
        return ScoringSmokeSubsetSummary(
            source_run_dir=source,
            output_run_dir=output,
            instance_count=2,
            conditions=("baseline", "llmdebug"),
        )

    monkeypatch.setattr(swebench_cli, "prepare_scoring_smoke_subset", _fake_prepare)
    exit_code = swebench_cli.main(
        [
            "prepare-scoring-smoke",
            "--run-dir",
            str(source),
            "--max-instances",
            "2",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Scoring smoke subset prepared" in out
    assert "Instances selected: 2" in out


# ---------------------------------------------------------------------------
# _resolve_path — non-absolute path (line 27)
# ---------------------------------------------------------------------------


def test_resolve_path_converts_relative_to_absolute() -> None:
    from evals.swebench.cli import _resolve_path

    result = _resolve_path("relative/dir")
    assert result.is_absolute()
    assert result == Path.cwd() / "relative/dir"


# ---------------------------------------------------------------------------
# No command → print_help, return 2 (lines 140-141)
# ---------------------------------------------------------------------------


def test_main_no_command_returns_usage_error(capsys) -> None:
    exit_code = swebench_cli.main([])
    assert exit_code == 2


# ---------------------------------------------------------------------------
# Unknown command → print_help, return 2 (lines 279-280)
# ---------------------------------------------------------------------------


def test_main_unknown_command_returns_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        swebench_cli.main(["totally-unknown"])
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("argv", "expected_help"),
    [
        (
            ["generate-lite", "--help"],
            [
                "OpenAI-compatible chat completions endpoint",
                "request errors",
            ],
        ),
        (
            ["prepare-handoff", "--help"],
            [
                "handoff bundle should be prepared",
                "paired conditions expected in the run directory",
            ],
        ),
        (
            ["score-official", "--help"],
            [
                "paired prediction files",
                "official SWE-Bench scorer entrypoint",
            ],
        ),
        (
            ["prepare-scoring-smoke", "--help"],
            [
                "Source run directory with full predictions.",
                "Maximum paired instances to keep in the smoke subset.",
            ],
        ),
    ],
)
def test_subcommand_help_includes_descriptions(
    argv: list[str], expected_help: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        swebench_cli.main(argv)
    assert exc_info.value.code == 0
    output = " ".join(capsys.readouterr().out.split()).lower()
    for expected in expected_help:
        assert expected.lower() in output


# ---------------------------------------------------------------------------
# prepare-handoff ValueError → return 1 (lines 191-193)
# ---------------------------------------------------------------------------


def test_prepare_handoff_bad_conditions_returns_one(monkeypatch, capsys) -> None:
    exit_code = swebench_cli.main(["prepare-handoff", "--run-id", "r", "--conditions", "baseline"])
    assert exit_code == 1
    assert "Handoff validation failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# prepare-handoff run_dir doesn't exist → return 2 (lines 196-197)
# ---------------------------------------------------------------------------


def test_prepare_handoff_missing_run_dir_returns_two(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(
        swebench_cli,
        "run_dir_for_id",
        lambda _run_id: tmp_path / "nonexistent",
    )
    exit_code = swebench_cli.main(
        [
            "prepare-handoff",
            "--run-id",
            "r",
            "--conditions",
            "baseline,llmdebug",
        ]
    )
    assert exit_code == 2
    assert "Run directory not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# prepare-handoff success output (lines 220-225)
# ---------------------------------------------------------------------------


def test_prepare_handoff_success_prints_output(monkeypatch, capsys, tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    (run_dir / "instance_ids.txt").write_text("inst-1\n", encoding="utf-8")
    (run_dir / "run_metadata.json").write_text('{"run_id":"demo-run"}\n', encoding="utf-8")
    line = (
        '{"instance_id":"inst-1",'
        '"model_patch":"diff --git a/a.py b/a.py\\n",'
        '"model_name_or_path":"m"}\n'
    )
    (run_dir / "baseline" / "predictions.jsonl").write_text(line, encoding="utf-8")
    (run_dir / "llmdebug" / "predictions.jsonl").write_text(line, encoding="utf-8")
    monkeypatch.setattr(swebench_cli, "run_dir_for_id", lambda _run_id: run_dir)
    exit_code = swebench_cli.main(
        [
            "prepare-handoff",
            "--run-id",
            "demo-run",
            "--conditions",
            "baseline,llmdebug",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Prepared handoff run index" in out
    assert "Prepared handoff instructions" in out
    assert "Run directory is ready to copy" in out


# ---------------------------------------------------------------------------
# score-official success output (line 248)
# ---------------------------------------------------------------------------


def test_score_official_cli_prints_condition_detail(monkeypatch, capsys, tmp_path) -> None:
    from evals.swebench.types import ScoreConditionSummary, ScoreRunSummary

    run_dir = tmp_path / "demo-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    def _fake_score(**_kwargs):
        return ScoreRunSummary(
            run_dir=run_dir,
            condition_summaries=(
                ScoreConditionSummary(
                    condition="baseline",
                    run_id="demo-run-baseline",
                    predictions_path=run_dir / "baseline" / "predictions.jsonl",
                    working_dir=run_dir / "official_eval" / "baseline",
                    log_path=run_dir / "official_eval" / "baseline" / "run.log",
                    command=("python", "-m", "swebench"),
                    exit_code=0,
                ),
            ),
        )

    monkeypatch.setattr(swebench_cli, "run_official_scoring", _fake_score)
    exit_code = swebench_cli.main(["score-official", "--run-dir", str(run_dir)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "baseline: exit=0" in out


# ---------------------------------------------------------------------------
# prepare-scoring-smoke output (line 258) and error → return 1 (lines 269-271)
# ---------------------------------------------------------------------------


def test_prepare_scoring_smoke_cli_error_returns_one(monkeypatch, capsys, tmp_path) -> None:
    def _fake_prepare(**_kwargs):
        raise RuntimeError("no source data")

    monkeypatch.setattr(swebench_cli, "prepare_scoring_smoke_subset", _fake_prepare)
    exit_code = swebench_cli.main(
        [
            "prepare-scoring-smoke",
            "--run-dir",
            str(tmp_path / "no-such-run"),
        ]
    )
    assert exit_code == 1
    assert "Scoring smoke preparation failed" in capsys.readouterr().out
