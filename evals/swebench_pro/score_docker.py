"""Docker-based scoring for SWE-Bench Pro predictions."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import logging
import re
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import _json as json
from .io import (
    parse_paired_conditions,
    prediction_path,
    read_predictions,
    validate_paired_predictions,
)
from .run_scripts import (
    extract_env_commands,
    load_parser_script,
    load_run_script,
)
from .types import (
    SWEBENCH_PRO_DATASET_NAME,
    SWEBENCH_PRO_DEFAULT_SPLIT,
    ScoreConditionSummary,
    ScoreInstanceResult,
    ScoreRunSummary,
)

logger = logging.getLogger(__name__)


def get_dockerhub_image_uri(
    instance_id: str,
    dockerhub_username: str,
    dockerhub_tag: str,
) -> str:
    """Build the full DockerHub image URI for an instance.

    Args:
        instance_id: SWE-bench Pro instance identifier (unused but kept
            for logging).
        dockerhub_username: DockerHub username/org hosting the images.
        dockerhub_tag: Tag from the dataset's ``dockerhub_tag`` column.

    Returns:
        Full image URI like ``jefzda/sweap-images:tag``.
    """
    return f"{dockerhub_username}/sweap-images:{dockerhub_tag}"


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch."""
    if not patch:
        return patch

    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


APPLY_FAILED_EXIT_CODE = 3
"""Entry-script exit code when every patch-apply command failed (a model failure, DAT-09)."""

# Mirrors the official SWE-bench harness apply sequence (DAT-19).
_APPLY_COMMANDS = (
    "git apply --verbose /workspace/patch.diff",
    "git apply --verbose --reject /workspace/patch.diff",
    "patch --batch --fuzz=5 -p1 -i /workspace/patch.diff",
)


def build_entry_script(
    *,
    instance_id: str,
    base_commit: str,
    before_repo_set_cmd: str,
    selected_test_files: str,
    env_commands: list[str],
) -> str:
    """Build the entry script that runs inside the Docker container.

    Dataset values are shell-quoted (DAT-11); ``before_repo_set_cmd`` is a command by
    contract and runs as written. Each stage records its status under ``/workspace``
    so the scorer can tell a patch that didn't apply (exit
    :data:`APPLY_FAILED_EXIT_CODE`) from an infrastructure failure (DAT-09).

    Args:
        instance_id: Instance identifier (for logging).
        base_commit: Git commit to reset to before applying patch.
        before_repo_set_cmd: Repository setup command to run after
            applying patch.
        selected_test_files: Comma-separated list of test files to run.
        env_commands: Shell export statements extracted from Dockerfiles.

    Returns:
        Shell script content as a string.
    """
    del instance_id
    env_block = "\n".join(env_commands)
    # Use only the last line of before_repo_set_cmd (matching Pro eval
    # behavior)
    setup_cmd = before_repo_set_cmd.strip().split("\n")[-1] if before_repo_set_cmd.strip() else ""
    commit = shlex.quote(base_commit)
    apply_chain = " || ".join(_APPLY_COMMANDS)

    return f"""#!/bin/bash
set -e
{env_block}
# Apply patch when present
cd /app
git reset --hard {commit}
git checkout {commit}
if [[ -s /workspace/patch.diff ]]; then
  if {apply_chain}; then
    echo applied > /workspace/apply_status
  else
    echo failed > /workspace/apply_status
    exit {APPLY_FAILED_EXIT_CODE}
  fi
else
  echo "Patch is empty after cleanup; evaluating base commit."
  echo empty > /workspace/apply_status
fi
{setup_cmd}
# Run tests and parse results; failing tests are an outcome, not a script error
set +e
bash /workspace/run_script.sh {shlex.quote(selected_test_files)} > /workspace/stdout.log 2> /workspace/stderr.log
echo $? > /workspace/run_status
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""


@dataclass(frozen=True)
class _EvalContext:
    """Internal context for a single instance evaluation."""

    instance_id: str
    patch: str
    dockerhub_tag: str
    base_commit: str
    before_repo_set_cmd: str
    selected_test_files: str
    fail_to_pass: set[str]
    pass_to_pass: set[str]


def _parse_test_list(raw: str) -> set[str]:
    """Parse a test list field (may be a JSON/Python list string or plain text)."""
    if not raw or not raw.strip():
        return set()
    stripped = raw.strip()
    try:
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, (list, tuple, set)):
            return {str(item) for item in parsed}
    except (ValueError, SyntaxError):
        pass
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return {str(item) for item in parsed}
    except (json.JSONDecodeError, TypeError):
        pass
    return {stripped}


def _parse_selected_test_files(raw: str) -> str:
    """Parse selected_test_files_to_run into comma-separated string."""
    if not raw or not raw.strip():
        return ""
    try:
        parsed = ast.literal_eval(raw.strip())
        if isinstance(parsed, (list, tuple)):
            return ",".join(str(item) for item in parsed)
    except (ValueError, SyntaxError):
        pass
    return raw.strip()


def evaluate_instance_docker(
    ctx: _EvalContext,
    *,
    dockerhub_username: str,
    scripts_dir: Path,
    dockerfiles_dir: Path | None,
    output_dir: Path,
    prefix: str,
    timeout: int = 3600,
    block_network: bool = False,
    docker_platform: str | None = None,
    redo: bool = False,
) -> ScoreInstanceResult:
    """Evaluate a single instance using local Docker.

    Args:
        ctx: Evaluation context for this instance.
        dockerhub_username: DockerHub username for image pulls.
        scripts_dir: Path to cached run_scripts directory.
        dockerfiles_dir: Path to cached dockerfiles directory (for ENV
            extraction).
        output_dir: Root output directory for evaluation artifacts.
        prefix: Prefix for output files.
        timeout: Container timeout in seconds.
        block_network: Whether to block network inside container.
        docker_platform: Docker platform override (e.g.,
            ``linux/amd64``).
        redo: Re-evaluate even if output exists.

    Returns:
        Scoring result for this instance.
    """
    try:
        import docker as docker_sdk  # type: ignore[import-untyped]
    except ImportError as exc:
        return ScoreInstanceResult(
            instance_id=ctx.instance_id,
            passed=False,
            fail_to_pass_met=False,
            pass_to_pass_met=False,
            error=f"docker SDK not installed: {exc}",
            infrastructure_error=True,
        )

    uid = ctx.instance_id
    uid_dir = output_dir / uid
    uid_dir.mkdir(parents=True, exist_ok=True)

    if not ctx.fail_to_pass:
        # DAT-04: an empty FAIL_TO_PASS set would pass trivially.
        return _infrastructure_failure(uid, "empty fail_to_pass test list")

    # Clean patch
    cleaned_patch = strip_binary_hunks(ctx.patch)
    patch_digest = hashlib.sha256(cleaned_patch.encode("utf-8")).hexdigest()

    output_path = uid_dir / f"{prefix}_output.json"
    digest_path = uid_dir / f"{prefix}_output.patch_sha256"
    cached_digest = digest_path.read_text(encoding="utf-8").strip() if digest_path.exists() else ""
    # DAT-13: reuse a cached result only for the identical patch.
    if not redo and output_path.exists() and cached_digest == patch_digest:
        try:
            output = json.loads(output_path.read_text(encoding="utf-8"))
            if not _report_validation_error(output):
                return _score_output(ctx, output)
        except Exception:
            pass  # Re-evaluate on parse failure

    workspace_dir = uid_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Prepare workspace files
    try:
        run_script = load_run_script(scripts_dir, uid)
        parser_script = load_parser_script(scripts_dir, uid)
    except FileNotFoundError as exc:
        return ScoreInstanceResult(
            instance_id=uid,
            passed=False,
            fail_to_pass_met=False,
            pass_to_pass_met=False,
            error=f"Missing run scripts: {exc}",
            infrastructure_error=True,
        )

    # Extract ENV commands from Dockerfiles
    env_cmds: list[str] = []
    if dockerfiles_dir is not None:
        env_cmds = extract_env_commands(dockerfiles_dir, uid)

    # Build entry script
    entry_script = build_entry_script(
        instance_id=uid,
        base_commit=ctx.base_commit,
        before_repo_set_cmd=ctx.before_repo_set_cmd,
        selected_test_files=ctx.selected_test_files,
        env_commands=env_cmds,
    )

    # DAT-09: a retry or a changed patch must not inherit the previous execution's
    # receipts, report or logs; a leftover apply_status=failed would otherwise turn the
    # next container's infrastructure failure into patch_apply_failed, and a leftover
    # output.json would be scored and stamped with the new patch digest.
    _clear_execution_receipts(workspace_dir)

    # Write workspace files
    (workspace_dir / "patch.diff").write_text(cleaned_patch, encoding="utf-8")
    (workspace_dir / "run_script.sh").write_text(run_script, encoding="utf-8")
    (workspace_dir / "parser.py").write_text(parser_script, encoding="utf-8")
    (workspace_dir / "entryscript.sh").write_text(entry_script, encoding="utf-8")

    # Save patch snapshot
    (uid_dir / f"{prefix}_patch.diff").write_text(ctx.patch, encoding="utf-8")
    (uid_dir / f"{prefix}_entryscript.sh").write_text(entry_script, encoding="utf-8")

    # Run Docker container
    image_uri = get_dockerhub_image_uri(uid, dockerhub_username, ctx.dockerhub_tag)
    client: Any | None = None
    container: Any = None
    try:
        client = docker_sdk.from_env()

        # Pull image
        try:
            if docker_platform:
                client.images.pull(image_uri, platform=docker_platform)
            else:
                client.images.pull(image_uri)
        except Exception:
            try:
                client.images.get(image_uri)
            except Exception as pull_err:
                return ScoreInstanceResult(
                    instance_id=uid,
                    passed=False,
                    fail_to_pass_met=False,
                    pass_to_pass_met=False,
                    error=f"Failed to pull image {image_uri}: {pull_err}",
                    infrastructure_error=True,
                )

        abs_workspace = str(workspace_dir.resolve())
        run_kwargs: dict[str, Any] = {
            "volumes": {
                abs_workspace: {"bind": "/workspace", "mode": "rw"},
            },
            "detach": True,
            "remove": True,
            "entrypoint": "/bin/bash",
            "command": ["-c", "bash /workspace/entryscript.sh"],
        }
        if block_network:
            run_kwargs["network_mode"] = "none"
        if docker_platform:
            run_kwargs["platform"] = docker_platform

        container = client.containers.run(image_uri, **run_kwargs)
        try:
            result = container.wait(timeout=timeout)
        except Exception as wait_exc:
            _cleanup_container(container)
            return _infrastructure_failure(
                uid, f"container wait failed or timed out after {timeout}s: {wait_exc}"
            )
        status_code = result.get("StatusCode", 1) if isinstance(result, dict) else 1

        # Collect logs
        _copy_log(
            workspace_dir / "stdout.log",
            uid_dir / f"{prefix}_stdout.log",
        )
        _copy_log(
            workspace_dir / "stderr.log",
            uid_dir / f"{prefix}_stderr.log",
        )

        apply_status = _read_status(workspace_dir / "apply_status")
        if apply_status == "failed":
            # DAT-09: only this execution's apply-stage receipt classifies the patch as
            # un-appliable. Exit code 3 alone is not exclusive to patch application (a
            # setup command can exit 3 after the patch applied), so it must not suppress
            # the infrastructure-error gate below.
            return ScoreInstanceResult(
                instance_id=uid,
                passed=False,
                fail_to_pass_met=False,
                pass_to_pass_met=False,
                error="patch_apply_failed",
            )
        if status_code != 0:
            return _infrastructure_failure(uid, f"Container exited with code {status_code}")

        # DAT-09: the entry script runs the test runner under `set +e` and the parser's
        # exit code becomes the container status, so a runner that never ran (127) would
        # otherwise be scored as an ordinary wrong patch from an empty test list.
        runner_error = _runner_receipt_error(_read_status(workspace_dir / "run_status"))
        if runner_error:
            return _infrastructure_failure(uid, runner_error)

        # Parse output
        output_json_path = workspace_dir / "output.json"
        if not output_json_path.exists():
            return _infrastructure_failure(uid, "output.json not found after container execution")

        # DAT-09: reading, parsing or validating the report is scorer infrastructure, so
        # its failures must not be reported as a model failure.
        try:
            output = json.loads(output_json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return _infrastructure_failure(
                uid, f"unreadable output.json after container execution: {exc}"
            )
        report_error = _report_validation_error(output)
        if report_error:
            return _infrastructure_failure(uid, report_error)

        # Save output
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        digest_path.write_text(patch_digest + "\n", encoding="utf-8")

        return _score_output(ctx, output)

    except Exception as exc:
        if container is not None:
            _cleanup_container(container)
        return ScoreInstanceResult(
            instance_id=uid,
            passed=False,
            fail_to_pass_met=False,
            pass_to_pass_met=False,
            error=f"{type(exc).__name__}: {exc}",
            infrastructure_error=container is None,
        )
    finally:
        if client is not None:
            _close_docker_client(client)


def _infrastructure_failure(instance_id: str, error: str) -> ScoreInstanceResult:
    return ScoreInstanceResult(
        instance_id=instance_id,
        passed=False,
        fail_to_pass_met=False,
        pass_to_pass_met=False,
        error=error,
        infrastructure_error=True,
    )


_EXECUTION_RECEIPT_FILENAMES = (
    "apply_status",
    "run_status",
    "output.json",
    "stdout.log",
    "stderr.log",
)
"""Everything the container writes per execution; stale copies must never be read."""


def _clear_execution_receipts(workspace_dir: Path) -> None:
    """Delete the previous execution's status files, report and logs."""
    for name in _EXECUTION_RECEIPT_FILENAMES:
        (workspace_dir / name).unlink(missing_ok=True)


_RUNNER_INFRASTRUCTURE_EXIT_CODES = frozenset({126, 127})
"""``run_script.sh`` could not be executed (126) or its runner was missing (127)."""


def _runner_receipt_error(raw: str) -> str:
    """Classify the test runner's exit receipt; ``""`` when tests actually ran.

    A non-zero exit is normally just failing tests. ``126``/``127`` and signal exits
    (``>= 128``) mean the runner never produced test results, which is infrastructure.
    A missing receipt is not classified here: the entry script writes it and the report
    in the same step, so an absent receipt comes with an absent report.
    """
    text = raw.strip()
    if not text:
        return ""
    try:
        code = int(text)
    except ValueError:
        return f"unreadable run_status receipt: {text[:80]!r}"
    if code in _RUNNER_INFRASTRUCTURE_EXIT_CODES or code >= 128:
        return f"test runner exited with infrastructure code {code}"
    return ""


def _report_validation_error(output: Any) -> str:
    """Reject a scorer report the parser wrote but the scorer cannot trust."""
    if not isinstance(output, dict):
        return "output.json is not a JSON object"
    tests = output.get("tests")
    if not isinstance(tests, list):
        return "output.json has no 'tests' list"
    for entry in tests:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            return "output.json contains a malformed test entry"
    return ""


def _read_status(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _score_output(ctx: _EvalContext, output: dict[str, Any]) -> ScoreInstanceResult:
    """Score a parsed output.json against expected test results."""
    tests = output.get("tests", [])
    passed_tests = {t["name"] for t in tests if isinstance(t, dict) and t.get("status") == "PASSED"}

    f2p_met = ctx.fail_to_pass <= passed_tests
    p2p_met = ctx.pass_to_pass <= passed_tests
    overall = f2p_met and p2p_met

    return ScoreInstanceResult(
        instance_id=ctx.instance_id,
        passed=overall,
        fail_to_pass_met=f2p_met,
        pass_to_pass_met=p2p_met,
    )


def _copy_log(src: Path, dst: Path) -> None:
    """Copy a log file if it exists; a failed copy is logged, not raised (X-07)."""
    try:
        if src.exists():
            dst.write_text(
                src.read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8",
            )
    except Exception as exc:
        logger.warning("could not copy scoring log %s -> %s: %s", src, dst, exc)


def _cleanup_container(container: Any) -> None:
    """Best-effort cleanup for a launched Docker container.

    Failures are logged (X-07): a container that can't be removed keeps running.
    """
    try:
        container.stop(timeout=0)
    except Exception as exc:
        logger.warning("could not stop scoring container %s: %s", _container_label(container), exc)
    try:
        container.remove(force=True)
    except Exception as exc:
        logger.warning(
            "could not remove scoring container %s: %s", _container_label(container), exc
        )


def _container_label(container: Any) -> str:
    return str(getattr(container, "short_id", None) or getattr(container, "id", None) or "?")


def _close_docker_client(client: Any) -> None:
    """Best-effort cleanup for a Docker client; a failed close is logged (X-07)."""
    try:
        client.close()
    except Exception as exc:
        logger.debug("could not close the Docker client: %s", exc)


def _raise_if_scoring_errors(
    *,
    condition: str,
    results: list[ScoreInstanceResult],
) -> None:
    """Raise when any instance hit a scorer infrastructure error."""
    errored = [result for result in results if result.infrastructure_error]
    if not errored:
        return

    preview = "; ".join(
        f"{result.instance_id}: {result.error or 'unknown infrastructure error'}"
        for result in errored[:3]
    )
    if len(errored) > 3:
        preview = f"{preview}; ... ({len(errored) - 3} more)"
    raise RuntimeError(
        f"{condition}: Docker scoring encountered {len(errored)} infrastructure "
        f"error(s). Examples: {preview}"
    )


def run_docker_scoring(
    *,
    run_dir: Path,
    conditions: str | list[str],
    dataset_name: str = SWEBENCH_PRO_DATASET_NAME,
    split: str = SWEBENCH_PRO_DEFAULT_SPLIT,
    dockerhub_username: str,
    scripts_dir: Path,
    dockerfiles_dir: Path | None = None,
    workers: int = 4,
    timeout: int = 3600,
    block_network: bool = False,
    docker_platform: str | None = None,
    redo: bool = False,
) -> ScoreRunSummary:
    """Run Docker-based scoring for SWE-bench Pro predictions.

    Args:
        run_dir: Run directory containing prediction files.
        conditions: Conditions to score.
        dataset_name: HuggingFace dataset name.
        split: Dataset split.
        dockerhub_username: DockerHub username for images.
        scripts_dir: Path to per-instance run_scripts.
        dockerfiles_dir: Path to Dockerfiles (for ENV extraction).
        workers: Number of parallel Docker containers.
        timeout: Per-instance container timeout in seconds.
        block_network: Block network inside containers.
        docker_platform: Docker platform override.
        redo: Re-evaluate existing results.

    Returns:
        Scoring summary across conditions.
    """
    if not run_dir.exists():
        raise RuntimeError(f"run directory does not exist: {run_dir}")

    condition_list = parse_paired_conditions(conditions)
    validate_paired_predictions(
        run_dir,
        conditions=condition_list,
        require_files=True,
        require_nonempty=True,
    )

    # Load dataset for metadata
    rows_by_id = _load_dataset_rows(dataset_name=dataset_name, split=split)

    summaries: list[ScoreConditionSummary] = []
    for condition in condition_list:
        predictions = read_predictions(prediction_path(run_dir, condition))
        if not predictions:
            raise RuntimeError(f"predictions file is empty for {condition}")

        condition_output_dir = run_dir / "docker_eval" / condition
        condition_output_dir.mkdir(parents=True, exist_ok=True)

        prefix = condition
        results: list[ScoreInstanceResult] = []

        # Build evaluation contexts
        contexts: list[_EvalContext] = []
        for pred in predictions:
            row = rows_by_id.get(pred.instance_id)
            if row is None:
                results.append(
                    ScoreInstanceResult(
                        instance_id=pred.instance_id,
                        passed=False,
                        fail_to_pass_met=False,
                        pass_to_pass_met=False,
                        error=(f"Instance not found in dataset: {pred.instance_id}"),
                    )
                )
                continue

            contexts.append(
                _EvalContext(
                    instance_id=pred.instance_id,
                    patch=pred.model_patch,
                    dockerhub_tag=str(row.get("dockerhub_tag", "")),
                    base_commit=str(row.get("base_commit", "")),
                    before_repo_set_cmd=str(row.get("before_repo_set_cmd", "")),
                    selected_test_files=_parse_selected_test_files(
                        str(row.get("selected_test_files_to_run", ""))
                    ),
                    fail_to_pass=_parse_test_list(str(row.get("fail_to_pass", ""))),
                    pass_to_pass=_parse_test_list(str(row.get("pass_to_pass", ""))),
                )
            )

        # Run evaluations in parallel
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map: dict[concurrent.futures.Future[ScoreInstanceResult], str] = {}
            for ctx in contexts:
                future = pool.submit(
                    evaluate_instance_docker,
                    ctx,
                    dockerhub_username=dockerhub_username,
                    scripts_dir=scripts_dir,
                    dockerfiles_dir=dockerfiles_dir,
                    output_dir=condition_output_dir,
                    prefix=prefix,
                    timeout=timeout,
                    block_network=block_network,
                    docker_platform=docker_platform,
                    redo=redo,
                )
                future_map[future] = ctx.instance_id

            for future in as_completed(future_map):
                instance_id = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = ScoreInstanceResult(
                        instance_id=instance_id,
                        passed=False,
                        fail_to_pass_met=False,
                        pass_to_pass_met=False,
                        error=f"{type(exc).__name__}: {exc}",
                        infrastructure_error=True,
                    )
                results.append(result)

        # Save results before failing loudly, so the reasons survive (DAT-09).
        results_payload = {r.instance_id: r.passed for r in results}
        results_path = condition_output_dir / "eval_results.json"
        results_path.write_text(
            json.dumps(results_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        details_payload = {
            r.instance_id: {
                "passed": r.passed,
                "fail_to_pass_met": r.fail_to_pass_met,
                "pass_to_pass_met": r.pass_to_pass_met,
                "error": r.error,
                "infrastructure_error": r.infrastructure_error,
            }
            for r in sorted(results, key=lambda item: item.instance_id)
        }
        (condition_output_dir / "eval_results_detail.json").write_text(
            json.dumps(details_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _raise_if_scoring_errors(condition=condition, results=results)

        passed_count = sum(1 for r in results if r.passed)
        error_count = sum(1 for r in results if r.error is not None)
        total = len(results)
        accuracy = passed_count / total if total > 0 else 0.0

        summaries.append(
            ScoreConditionSummary(
                condition=condition,
                run_id=run_dir.name,
                predictions_path=prediction_path(run_dir, condition),
                output_dir=condition_output_dir,
                total_instances=total,
                passed_count=passed_count,
                failed_count=total - passed_count,
                error_count=error_count,
                accuracy=accuracy,
            )
        )

        print(
            f"[{condition}] Accuracy: {accuracy:.2%} "
            f"({passed_count}/{total} passed, {error_count} errors)"
        )

    return ScoreRunSummary(
        run_dir=run_dir,
        condition_summaries=tuple(summaries),
    )


def _load_dataset_rows(*, dataset_name: str, split: str) -> dict[str, dict[str, Any]]:
    """Load dataset rows indexed by instance_id."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for SWE-Bench Pro scoring. Install with: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    rows: dict[str, dict[str, Any]] = {}
    for row in dataset:
        payload = dict(row)
        instance_id = str(payload.get("instance_id", "")).strip()
        if instance_id:
            rows[instance_id] = payload
    return rows
