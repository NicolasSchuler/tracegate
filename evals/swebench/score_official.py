"""Official SWE-Bench scoring orchestration utilities."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

from .io import (
    parse_paired_conditions,
    prediction_path,
    read_predictions,
    validate_paired_predictions,
)
from .types import ScoreConditionSummary, ScoreRunSummary

VALID_CACHE_LEVELS = {"none", "base", "env", "instance"}
PINNED_SWEBENCH_VERSION = "4.1.0"


def build_run_evaluation_command(
    *,
    predictions_path: Path,
    run_id: str,
    dataset_name: str,
    split: str,
    workers: int,
    cache_level: str,
    clean: bool,
    python_executable: str,
    module: str,
) -> list[str]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    normalized_cache = cache_level.strip().lower()
    if normalized_cache not in VALID_CACHE_LEVELS:
        raise ValueError(
            f"unsupported cache level {cache_level!r}; expected one of {sorted(VALID_CACHE_LEVELS)}"
        )
    command = [
        python_executable,
        "-m",
        module,
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(workers),
        "--run_id",
        run_id,
        "--cache_level",
        normalized_cache,
    ]
    if clean:
        command.append("--clean")
    return command


def check_scoring_prerequisites(*, python_executable: str, module: str) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required for official SWE-Bench scoring")

    if module.startswith("swebench."):
        try:
            installed = importlib.metadata.version("swebench")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"SWE-Bench package is not installed. Expected swebench=={PINNED_SWEBENCH_VERSION}."
            ) from exc
        if installed != PINNED_SWEBENCH_VERSION:
            raise RuntimeError(
                "SWE-Bench version mismatch. "
                f"Expected swebench=={PINNED_SWEBENCH_VERSION}, found {installed!r}."
            )

    probe = [
        python_executable,
        "-c",
        f"import importlib; importlib.import_module({module!r})",
    ]
    proc = subprocess.run(
        probe,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            "SWE-Bench harness module is unavailable. "
            f"Install it (e.g. pip install swebench). Details: {message}"
        )


def run_official_scoring(
    *,
    run_dir: Path,
    conditions: str | list[str],
    dataset_name: str,
    split: str,
    workers: int,
    cache_level: str,
    clean: bool,
    python_executable: str = sys.executable,
    module: str = "swebench.harness.run_evaluation",
    check_prereqs: bool = True,
) -> ScoreRunSummary:
    if not run_dir.exists():
        raise RuntimeError(f"run directory does not exist: {run_dir}")
    condition_list = parse_paired_conditions(conditions)

    expected_ids_path = run_dir / "instance_ids.txt"
    expected_ids: list[str] | None
    if expected_ids_path.exists():
        expected_ids = [
            line.strip()
            for line in expected_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        expected_ids = None

    validate_paired_predictions(
        run_dir,
        conditions=condition_list,
        expected_ids=expected_ids,
        require_files=True,
        require_nonempty=True,
    )

    if check_prereqs:
        check_scoring_prerequisites(python_executable=python_executable, module=module)

    summaries: list[ScoreConditionSummary] = []
    base_run_id = run_dir.name
    for condition in condition_list:
        predictions = prediction_path(run_dir, condition)
        if not predictions.exists():
            raise RuntimeError(f"missing predictions file for {condition}: {predictions}")
        if not read_predictions(predictions):
            raise RuntimeError(f"predictions file is empty for {condition}: {predictions}")

        condition_run_id = f"{base_run_id}-{condition}"
        working_dir = run_dir / "official_eval" / condition
        working_dir.mkdir(parents=True, exist_ok=True)

        command = build_run_evaluation_command(
            predictions_path=predictions,
            run_id=condition_run_id,
            dataset_name=dataset_name,
            split=split,
            workers=workers,
            cache_level=cache_level,
            clean=clean,
            python_executable=python_executable,
            module=module,
        )
        log_path = working_dir / "run_evaluation.log"
        cmd_path = working_dir / "command.txt"
        cmd_path.write_text(
            " ".join(_shell_quote(part) for part in command) + "\n", encoding="utf-8"
        )

        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"Running condition: {condition}\n")
            handle.write(f"Working directory: {working_dir}\n")
            handle.write(f"Predictions path: {predictions}\n\n")
            handle.flush()
            proc = subprocess.run(
                command,
                cwd=working_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        summaries.append(
            ScoreConditionSummary(
                condition=condition,
                run_id=condition_run_id,
                predictions_path=predictions,
                working_dir=working_dir,
                log_path=log_path,
                command=tuple(command),
                exit_code=int(proc.returncode),
            )
        )

    failures = [item for item in summaries if item.exit_code != 0]
    if failures:
        rendered = ", ".join(f"{item.condition}={item.exit_code}" for item in failures)
        raise RuntimeError(f"official scoring failed for condition(s): {rendered}")

    return ScoreRunSummary(
        run_dir=run_dir,
        condition_summaries=tuple(summaries),
    )


def _shell_quote(token: str) -> str:
    if not token:
        return "''"
    if token.isalnum() or all(ch in "._/-" for ch in token):
        return token
    escaped = token.replace("'", "'\"'\"'")
    return f"'{escaped}'"
