"""CLI for SWE-Bench Pro generation and Docker-based scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

from evals.swebench.prompt_policy import PromptInputs

from .generate import generate_pro_predictions
from .io import (
    instance_ids_path,
    parse_paired_conditions,
    run_dir_for_id,
    validate_paired_predictions,
    write_copy_instructions,
    write_run_index,
)
from .run_scripts import clone_or_update_repo, dockerfiles_dir, run_scripts_dir
from .score_docker import run_docker_scoring
from .smoke import prepare_scoring_smoke_subset
from .types import SWEBENCH_PRO_DATASET_NAME, SWEBENCH_PRO_DEFAULT_SPLIT


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--conditions", default="baseline,llmdebug")
    parser.add_argument("--dataset-name", default=SWEBENCH_PRO_DATASET_NAME)
    parser.add_argument("--split", default=SWEBENCH_PRO_DEFAULT_SPLIT)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--instance-ids-file", default="")
    parser.add_argument("--openai-base-url", required=True)
    parser.add_argument("--openai-model", required=True)
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument(
        "--openai-http2",
        dest="openai_http2",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-openai-http2",
        dest="openai_http2",
        action="store_false",
    )
    parser.add_argument("--openai-async", action="store_true")
    parser.add_argument("--openai-timeout-sec", type=float, default=180.0)
    parser.add_argument("--openai-max-tokens", type=int, default=2048)
    parser.add_argument("--openai-temperature", type=float, default=0.0)
    parser.add_argument("--openai-response-mode", choices=["diff", "edits"], default="diff")
    parser.add_argument(
        "--snapshot-dir",
        default="",
        help=(
            "Directory of per-instance llmdebug snapshot artifacts (<instance_id>.txt/.md/.json). "
            "Required when the llmdebug condition is requested."
        ),
    )
    parser.add_argument(
        "--include-test-ids",
        action="store_true",
        help="Add the fail_to_pass/pass_to_pass test IDs to the prompt (the scorer's tests).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-request-errors", type=int, default=25)
    parser.add_argument("--max-request-error-rate", type=float, default=0.5)
    parser.add_argument("--min-requests-for-error-rate-check", type=int, default=10)


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--conditions", default="baseline,llmdebug")


def _add_score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--conditions", default="baseline,llmdebug")
    parser.add_argument("--dataset-name", default=SWEBENCH_PRO_DATASET_NAME)
    parser.add_argument("--split", default=SWEBENCH_PRO_DEFAULT_SPLIT)
    parser.add_argument("--dockerhub-username", default="jefzda")
    parser.add_argument(
        "--run-scripts-dir",
        default="",
        help="Path to per-instance run_scripts. Auto-fetched if empty.",
    )
    parser.add_argument(
        "--dockerfiles-dir",
        default="",
        help="Path to Dockerfiles for ENV extraction. Auto-fetched if empty.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--block-network", action="store_true")
    parser.add_argument("--docker-platform", default=None)
    parser.add_argument("--redo", action="store_true")


def _add_prepare_scoring_smoke_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir", required=True, help="Source run directory with full predictions."
    )
    parser.add_argument(
        "--output-run-dir",
        default="",
        help="Output run directory for scoring smoke subset (default: sibling derived name).",
    )
    parser.add_argument("--max-instances", type=int, default=2)
    parser.add_argument("--conditions", default="baseline,llmdebug")


def _add_fetch_scripts_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Directory to cache the SWE-bench Pro repo (default: evals/swebench_pro_assets).",
    )
    parser.add_argument(
        "--commit",
        default="",
        help="Git commit SHA to pin (default: built-in pinned commit).",
    )
    parser.add_argument("--force", action="store_true", help="Force fresh clone.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SWE-Bench Pro helper CLI for llmdebug workflows.")
    sub = parser.add_subparsers(dest="command")

    generate = sub.add_parser("generate-pro", help="Generate SWE-Bench Pro predictions.")
    _add_generate_arguments(generate)

    prepare = sub.add_parser(
        "prepare-handoff", help="Validate run directory and write copy instructions."
    )
    _add_prepare_arguments(prepare)

    score = sub.add_parser(
        "score-docker", help="Run Docker-based scoring for SWE-Bench Pro predictions."
    )
    _add_score_arguments(score)

    smoke = sub.add_parser(
        "prepare-scoring-smoke",
        help="Create a small paired subset run for scoring smoke tests.",
    )
    _add_prepare_scoring_smoke_arguments(smoke)

    fetch = sub.add_parser(
        "fetch-run-scripts",
        help="Download/update per-instance run_scripts from SWE-bench Pro repo.",
    )
    _add_fetch_scripts_arguments(fetch)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for SWE-Bench Pro workflows.

    Subcommands:
      generate-pro           -- query a model and write per-condition prediction files.
      prepare-handoff        -- validate a run directory and emit copy instructions.
      score-docker           -- run Docker-based scoring against predictions.
      prepare-scoring-smoke  -- create a small paired subset run for scoring smoke tests.
      fetch-run-scripts      -- download per-instance run_scripts from SWE-bench Pro repo.

    Returns an integer exit code (0 = success, 1 = failure, 2 = usage error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    command = str(getattr(args, "command", "") or "")
    if not command:
        parser.print_help()
        return 2

    if command == "generate-pro":
        instance_ids_file = (
            _resolve_path(args.instance_ids_file) if str(args.instance_ids_file).strip() else None
        )
        snapshot_dir = _resolve_path(args.snapshot_dir) if str(args.snapshot_dir).strip() else None
        openai_api_key = str(args.openai_api_key).strip() or None
        try:
            summary = generate_pro_predictions(
                run_id=str(args.run_id),
                conditions=str(args.conditions),
                dataset_name=str(args.dataset_name),
                split=str(args.split),
                max_instances=int(args.max_instances),
                instance_ids_file=instance_ids_file,
                openai_base_url=str(args.openai_base_url),
                openai_model=str(args.openai_model),
                openai_api_key=openai_api_key,
                openai_http2=bool(args.openai_http2),
                openai_async=bool(args.openai_async),
                openai_timeout_sec=float(args.openai_timeout_sec),
                openai_max_tokens=int(args.openai_max_tokens),
                openai_temperature=float(args.openai_temperature),
                openai_response_mode=str(args.openai_response_mode),
                prompt_inputs=PromptInputs(
                    snapshot_dir=snapshot_dir, include_test_ids=bool(args.include_test_ids)
                ),
                resume=bool(args.resume),
                seed=args.seed if args.seed is None else int(args.seed),
                jobs=int(args.jobs),
                max_request_errors=int(args.max_request_errors),
                max_request_error_rate=float(args.max_request_error_rate),
                min_requests_for_error_rate_check=int(args.min_requests_for_error_rate_check),
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Generation failed: {exc}")
            return 1

        print(f"Run directory: {summary.run_dir}")
        print(f"Instances: {summary.instance_count}")
        for item in summary.condition_summaries:
            print(
                f"- {item.condition}: generated={item.generated} resumed={item.resumed} "
                f"empty={item.empty_patch_count} request_errors={item.request_error_count} "
                f"path={item.predictions_path}"
            )
        return 0

    if command == "prepare-handoff":
        try:
            condition_list = parse_paired_conditions(str(args.conditions))
        except ValueError as exc:
            print(f"Handoff validation failed: {exc}")
            return 1
        run_dir = run_dir_for_id(str(args.run_id))
        if not run_dir.exists():
            print(f"Run directory not found: {run_dir}")
            return 2

        expected_ids_path_val = instance_ids_path(run_dir)
        expected_ids = None
        if expected_ids_path_val.exists():
            expected_ids = [
                line.strip()
                for line in expected_ids_path_val.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        try:
            validate_paired_predictions(
                run_dir,
                conditions=condition_list,
                expected_ids=expected_ids,
                require_files=True,
                require_nonempty=True,
            )
        except ValueError as exc:
            print(f"Handoff validation failed: {exc}")
            return 1

        index_path = write_run_index(run_dir, run_id=str(args.run_id), conditions=condition_list)
        output = write_copy_instructions(run_dir, conditions=condition_list)
        print(f"Prepared handoff run index: {index_path}")
        print(f"Prepared handoff instructions: {output}")
        print(f"Run directory is ready to copy: {run_dir}")
        return 0

    if command == "score-docker":
        run_dir = _resolve_path(str(args.run_dir))
        scripts_dir_val = (
            _resolve_path(str(args.run_scripts_dir)) if str(args.run_scripts_dir).strip() else None
        )
        dockerfiles_dir_val = (
            _resolve_path(str(args.dockerfiles_dir)) if str(args.dockerfiles_dir).strip() else None
        )

        # Auto-fetch run_scripts if not provided
        if scripts_dir_val is None or dockerfiles_dir_val is None:
            print("Fetching SWE-bench Pro run_scripts and Dockerfiles...")
            repo_dir = clone_or_update_repo()
            if scripts_dir_val is None:
                scripts_dir_val = run_scripts_dir(repo_dir)
            if dockerfiles_dir_val is None:
                dockerfiles_dir_val = dockerfiles_dir(repo_dir)

        try:
            summary = run_docker_scoring(
                run_dir=run_dir,
                conditions=str(args.conditions),
                dataset_name=str(args.dataset_name),
                split=str(args.split),
                dockerhub_username=str(args.dockerhub_username),
                scripts_dir=scripts_dir_val,
                dockerfiles_dir=dockerfiles_dir_val,
                workers=int(args.workers),
                timeout=int(args.timeout),
                block_network=bool(args.block_network),
                docker_platform=args.docker_platform,
                redo=bool(args.redo),
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Docker scoring failed: {exc}")
            return 1

        print(f"Docker scoring complete for run dir: {summary.run_dir}")
        for item in summary.condition_summaries:
            print(
                f"- {item.condition}: accuracy={item.accuracy:.2%} "
                f"passed={item.passed_count}/{item.total_instances} "
                f"errors={item.error_count}"
            )
        return 0

    if command == "prepare-scoring-smoke":
        source_run_dir = _resolve_path(str(args.run_dir))
        output_arg = str(args.output_run_dir).strip()
        if output_arg:
            output_run_dir = _resolve_path(output_arg)
        else:
            suffix = f"score-smoke-{int(args.max_instances)}"
            output_run_dir = source_run_dir.parent / f"{source_run_dir.name}-{suffix}"
        try:
            summary = prepare_scoring_smoke_subset(
                source_run_dir=source_run_dir,
                output_run_dir=output_run_dir,
                max_instances=int(args.max_instances),
                conditions=str(args.conditions),
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Scoring smoke preparation failed: {exc}")
            return 1

        print(f"Scoring smoke subset prepared: {summary.output_run_dir}")
        print(f"Source run directory: {summary.source_run_dir}")
        print(f"Instances selected: {summary.instance_count}")
        print(f"Conditions: {','.join(summary.conditions)}")
        return 0

    if command == "fetch-run-scripts":
        cache_dir = _resolve_path(str(args.cache_dir)) if str(args.cache_dir).strip() else None
        commit = str(args.commit).strip() or None
        kwargs: dict[str, object] = {"force": bool(args.force)}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if commit is not None:
            kwargs["commit"] = commit
        try:
            repo_dir = clone_or_update_repo(**kwargs)  # type: ignore[arg-type]
        except RuntimeError as exc:
            print(f"Fetch failed: {exc}")
            return 1

        scripts = run_scripts_dir(repo_dir)
        dockerfiles = dockerfiles_dir(repo_dir)
        print(f"Repository cloned to: {repo_dir}")
        print(f"Run scripts: {scripts}")
        print(f"Dockerfiles: {dockerfiles}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
