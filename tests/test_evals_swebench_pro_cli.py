"""Tests for the SWE-bench Pro CLI subcommands.

Mirrors the structure of ``test_evals_swebench_cli.py`` — exercises argument
parsing and dispatch for every Pro subcommand without requiring Docker, network
access, or git operations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.swebench_pro import cli as pro_cli
from evals.swebench_pro.types import (
    ConditionGenerationSummary,
    GenerationRunSummary,
    ScoreConditionSummary,
    ScoreRunSummary,
    ScoringSmokeSubsetSummary,
)

# ---------------------------------------------------------------------------
# No subcommand → usage error (exit 2)
# ---------------------------------------------------------------------------


def test_no_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = pro_cli.main([])
    assert exit_code == 2


# ---------------------------------------------------------------------------
# generate-pro subcommand
# ---------------------------------------------------------------------------


def test_generate_pro_cli_parses_and_forwards(monkeypatch, capsys, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        return GenerationRunSummary(
            run_id="pro-run",
            run_dir=tmp_path / "evals" / "swebench_pro_runs" / "pro-run",
            instance_count=3,
            condition_summaries=(
                ConditionGenerationSummary(
                    condition="baseline",
                    predictions_path=tmp_path / "baseline" / "predictions.jsonl",
                    total_instances=3,
                    generated=3,
                    resumed=0,
                    empty_patch_count=0,
                    request_error_count=0,
                ),
            ),
        )

    monkeypatch.setattr(pro_cli, "generate_pro_predictions", _fake_generate)
    exit_code = pro_cli.main(
        [
            "generate-pro",
            "--run-id",
            "pro-run",
            "--conditions",
            "baseline,llmdebug",
            "--openai-base-url",
            "http://localhost:19000",
            "--openai-model",
            "local-model",
            "--no-openai-http2",
            "--openai-async",
            "--jobs",
            "8",
        ]
    )

    assert exit_code == 0
    assert captured["run_id"] == "pro-run"
    assert captured["jobs"] == 8
    assert captured["openai_http2"] is False
    assert captured["openai_async"] is True
    assert captured["conditions"] == "baseline,llmdebug"
    assert "Run directory:" in capsys.readouterr().out


def test_generate_pro_cli_rejects_single_condition(capsys) -> None:
    exit_code = pro_cli.main(
        [
            "generate-pro",
            "--run-id",
            "pro-run",
            "--conditions",
            "baseline",
            "--openai-base-url",
            "http://localhost:19000",
            "--openai-model",
            "local-model",
        ]
    )
    assert exit_code == 1
    assert "Generation failed" in capsys.readouterr().out


def test_generate_pro_cli_default_flags(monkeypatch, capsys, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        return GenerationRunSummary(
            run_id="run-defaults",
            run_dir=tmp_path / "run-defaults",
            instance_count=1,
            condition_summaries=(),
        )

    monkeypatch.setattr(pro_cli, "generate_pro_predictions", _fake_generate)
    exit_code = pro_cli.main(
        [
            "generate-pro",
            "--run-id",
            "run-defaults",
            "--openai-base-url",
            "http://localhost:19000",
            "--openai-model",
            "m",
        ]
    )
    assert exit_code == 0
    # Defaults
    assert captured["openai_http2"] is True
    assert captured["openai_async"] is False
    assert captured["jobs"] == 1
    assert captured["openai_temperature"] == 0.0
    assert captured["openai_max_tokens"] == 4096  # DAT-18: was 2048
    assert captured["max_instances"] == 0


# ---------------------------------------------------------------------------
# prepare-handoff subcommand
# ---------------------------------------------------------------------------


def test_prepare_handoff_fails_when_one_condition_missing(monkeypatch, capsys, tmp_path) -> None:
    run_dir = tmp_path / "pro-run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "instance_ids.txt").write_text("inst-1\n", encoding="utf-8")
    (run_dir / "run_metadata.json").write_text('{"run_id":"pro-run"}\n', encoding="utf-8")
    (run_dir / "baseline" / "predictions.jsonl").write_text(
        (
            '{"instance_id":"inst-1",'
            '"model_patch":"diff --git a/a.py b/a.py\\n",'
            '"model_name_or_path":"baseline-model"}\n'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(pro_cli, "run_dir_for_id", lambda _run_id: run_dir)
    exit_code = pro_cli.main(["prepare-handoff", "--run-id", "pro-run"])

    assert exit_code == 1
    assert "Handoff validation failed" in capsys.readouterr().out


def test_prepare_handoff_rejects_single_condition(capsys) -> None:
    exit_code = pro_cli.main(["prepare-handoff", "--run-id", "pro-run", "--conditions", "baseline"])
    assert exit_code == 1
    assert "Handoff validation failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# score-docker subcommand
# ---------------------------------------------------------------------------


def test_score_docker_cli_success(monkeypatch, capsys, tmp_path) -> None:
    run_dir = tmp_path / "pro-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    def _fake_score(**_kwargs):
        return ScoreRunSummary(
            run_dir=run_dir,
            condition_summaries=(
                ScoreConditionSummary(
                    condition="baseline",
                    run_id="pro-run",
                    predictions_path=run_dir / "baseline" / "predictions.jsonl",
                    output_dir=run_dir / "docker_eval" / "baseline",
                    total_instances=10,
                    passed_count=7,
                    failed_count=3,
                    error_count=0,
                    accuracy=0.7,
                ),
            ),
        )

    monkeypatch.setattr(pro_cli, "run_docker_scoring", _fake_score)
    # Provide explicit scripts/dockerfiles dirs so CLI doesn't call clone_or_update_repo
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    dockerfiles = tmp_path / "dockerfiles"
    dockerfiles.mkdir()

    exit_code = pro_cli.main(
        [
            "score-docker",
            "--run-dir",
            str(run_dir),
            "--run-scripts-dir",
            str(scripts_dir),
            "--dockerfiles-dir",
            str(dockerfiles),
            "--workers",
            "2",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Docker scoring complete for run dir" in out


def test_score_docker_cli_surfaces_failure(monkeypatch, capsys, tmp_path) -> None:
    run_dir = tmp_path / "pro-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    def _fake_score(**_kwargs):
        raise RuntimeError("simulated docker scoring failure")

    monkeypatch.setattr(pro_cli, "run_docker_scoring", _fake_score)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    dockerfiles = tmp_path / "dockerfiles"
    dockerfiles.mkdir()

    exit_code = pro_cli.main(
        [
            "score-docker",
            "--run-dir",
            str(run_dir),
            "--run-scripts-dir",
            str(scripts_dir),
            "--dockerfiles-dir",
            str(dockerfiles),
        ]
    )

    assert exit_code == 1
    assert "Docker scoring failed: simulated docker scoring failure" in capsys.readouterr().out


def test_score_docker_cli_auto_fetches_scripts(monkeypatch, capsys, tmp_path) -> None:
    """When no --run-scripts-dir given, CLI calls clone_or_update_repo."""
    run_dir = tmp_path / "pro-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = tmp_path / "cached-repo"
    repo_dir.mkdir()
    (repo_dir / "run_scripts").mkdir()
    (repo_dir / "dockerfiles").mkdir()

    clone_called = False

    def _fake_clone(**_kwargs):
        nonlocal clone_called
        clone_called = True
        return repo_dir

    def _fake_score(**_kwargs):
        return ScoreRunSummary(run_dir=run_dir, condition_summaries=())

    monkeypatch.setattr(pro_cli, "clone_or_update_repo", _fake_clone)
    monkeypatch.setattr(pro_cli, "run_docker_scoring", _fake_score)

    exit_code = pro_cli.main(
        [
            "score-docker",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert exit_code == 0
    assert clone_called


def test_score_docker_default_args(monkeypatch, capsys, tmp_path) -> None:
    """Verify default argument values for score-docker."""
    captured: dict[str, object] = {}

    def _fake_score(**kwargs):
        captured.update(kwargs)
        return ScoreRunSummary(run_dir=tmp_path, condition_summaries=())

    repo_dir = tmp_path / "cached-repo"
    repo_dir.mkdir()
    (repo_dir / "run_scripts").mkdir()
    (repo_dir / "dockerfiles").mkdir()

    monkeypatch.setattr(pro_cli, "clone_or_update_repo", lambda **_kw: repo_dir)
    monkeypatch.setattr(pro_cli, "run_docker_scoring", _fake_score)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    exit_code = pro_cli.main(["score-docker", "--run-dir", str(run_dir)])

    assert exit_code == 0
    assert captured["workers"] == 4
    assert captured["timeout"] == 3600
    assert captured["block_network"] is False
    assert captured["redo"] is False
    assert captured["dockerhub_username"] == "jefzda"


# ---------------------------------------------------------------------------
# prepare-scoring-smoke subcommand
# ---------------------------------------------------------------------------


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

    monkeypatch.setattr(pro_cli, "prepare_scoring_smoke_subset", _fake_prepare)
    exit_code = pro_cli.main(
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


def test_prepare_scoring_smoke_cli_failure(monkeypatch, capsys, tmp_path) -> None:
    source = tmp_path / "src-run"
    source.mkdir()

    def _fake_prepare(**_kwargs):
        raise RuntimeError("source run has no instance ids")

    monkeypatch.setattr(pro_cli, "prepare_scoring_smoke_subset", _fake_prepare)
    exit_code = pro_cli.main(
        [
            "prepare-scoring-smoke",
            "--run-dir",
            str(source),
        ]
    )

    assert exit_code == 1
    assert "Scoring smoke preparation failed" in capsys.readouterr().out


def test_prepare_scoring_smoke_custom_output_dir(monkeypatch, capsys, tmp_path) -> None:
    source = tmp_path / "src-run"
    source.mkdir()
    custom_output = tmp_path / "custom-smoke"

    captured: dict[str, object] = {}

    def _fake_prepare(**kwargs):
        captured.update(kwargs)
        return ScoringSmokeSubsetSummary(
            source_run_dir=source,
            output_run_dir=custom_output,
            instance_count=1,
            conditions=("baseline", "llmdebug"),
        )

    monkeypatch.setattr(pro_cli, "prepare_scoring_smoke_subset", _fake_prepare)
    exit_code = pro_cli.main(
        [
            "prepare-scoring-smoke",
            "--run-dir",
            str(source),
            "--output-run-dir",
            str(custom_output),
        ]
    )

    assert exit_code == 0
    assert captured["output_run_dir"] == custom_output


# ---------------------------------------------------------------------------
# fetch-run-scripts subcommand
# ---------------------------------------------------------------------------


def test_fetch_run_scripts_cli_success(monkeypatch, capsys, tmp_path) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / "run_scripts").mkdir(parents=True)
    (repo_dir / "dockerfiles").mkdir(parents=True)

    def _fake_clone(**_kwargs):
        return repo_dir

    monkeypatch.setattr(pro_cli, "clone_or_update_repo", _fake_clone)
    exit_code = pro_cli.main(
        [
            "fetch-run-scripts",
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Repository cloned to" in out
    assert "Run scripts:" in out
    assert "Dockerfiles:" in out


def test_fetch_run_scripts_cli_failure(monkeypatch, capsys) -> None:
    def _fake_clone(**_kwargs):
        raise RuntimeError("git clone failed")

    monkeypatch.setattr(pro_cli, "clone_or_update_repo", _fake_clone)
    exit_code = pro_cli.main(["fetch-run-scripts"])

    assert exit_code == 1
    assert "Fetch failed: git clone failed" in capsys.readouterr().out


def test_fetch_run_scripts_cli_force_flag(monkeypatch, capsys, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_clone(**kwargs):
        captured.update(kwargs)
        repo_dir = tmp_path / "repo"
        (repo_dir / "run_scripts").mkdir(parents=True, exist_ok=True)
        (repo_dir / "dockerfiles").mkdir(parents=True, exist_ok=True)
        return repo_dir

    monkeypatch.setattr(pro_cli, "clone_or_update_repo", _fake_clone)
    exit_code = pro_cli.main(["fetch-run-scripts", "--force"])

    assert exit_code == 0
    assert captured["force"] is True


def test_fetch_run_scripts_cli_custom_commit(monkeypatch, capsys, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_clone(**kwargs):
        captured.update(kwargs)
        repo_dir = tmp_path / "repo"
        (repo_dir / "run_scripts").mkdir(parents=True, exist_ok=True)
        (repo_dir / "dockerfiles").mkdir(parents=True, exist_ok=True)
        return repo_dir

    monkeypatch.setattr(pro_cli, "clone_or_update_repo", _fake_clone)
    exit_code = pro_cli.main(["fetch-run-scripts", "--commit", "abc123def"])

    assert exit_code == 0
    assert captured["commit"] == "abc123def"
