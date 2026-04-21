"""CLI for SWE-Bench Lite generation and official scoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generate import generate_lite_predictions
from .io import (
    instance_ids_path,
    parse_paired_conditions,
    run_dir_for_id,
    validate_paired_predictions,
    write_copy_instructions,
    write_run_index,
)
from .score_official import run_official_scoring
from .smoke import prepare_scoring_smoke_subset
from .types import SWEBENCH_LITE_DATASET_NAME, SWEBENCH_LITE_DEFAULT_SPLIT


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-id", required=True, help="Run identifier used for the output run directory."
    )
    parser.add_argument(
        "--conditions",
        default="baseline,llmdebug",
        help="Comma-separated paired conditions to generate (default: baseline,llmdebug).",
    )
    parser.add_argument(
        "--dataset-name",
        default=SWEBENCH_LITE_DATASET_NAME,
        help="Dataset name passed through to the SWE-Bench Lite loader.",
    )
    parser.add_argument(
        "--split",
        default=SWEBENCH_LITE_DEFAULT_SPLIT,
        help="Dataset split to generate predictions for.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Optional cap on the number of instances to generate (0 = no cap).",
    )
    parser.add_argument(
        "--instance-ids-file",
        default="",
        help="Optional file with newline-delimited instance IDs to keep.",
    )
    parser.add_argument(
        "--openai-base-url",
        required=True,
        help="Base URL for the OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument(
        "--openai-model",
        required=True,
        help="Model name to request from the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--openai-api-key",
        default="",
        help="Optional API key override for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--openai-http2",
        dest="openai_http2",
        action="store_true",
        default=True,
        help="Use HTTP/2 for OpenAI-compatible requests (default: enabled).",
    )
    parser.add_argument(
        "--no-openai-http2",
        dest="openai_http2",
        action="store_false",
        help="Disable HTTP/2 for OpenAI-compatible requests.",
    )
    parser.add_argument(
        "--openai-async",
        action="store_true",
        help="Enable async request fan-out for generation workers.",
    )
    parser.add_argument(
        "--openai-timeout-sec",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds for OpenAI-compatible calls.",
    )
    parser.add_argument(
        "--openai-max-tokens",
        type=int,
        default=2048,
        help="Maximum completion tokens requested from the model.",
    )
    parser.add_argument(
        "--openai-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for OpenAI-compatible requests.",
    )
    parser.add_argument(
        "--openai-response-mode",
        choices=["diff", "edits"],
        default="diff",
        help="Patch contract to request from the model.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="",
        help="Optional directory for request/response snapshots and debug artifacts.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run directory instead of regenerating completed predictions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed recorded in generation metadata.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel generation workers to run.",
    )
    parser.add_argument(
        "--max-request-errors",
        type=int,
        default=25,
        help="Abort generation after this many request errors.",
    )
    parser.add_argument(
        "--max-request-error-rate",
        type=float,
        default=0.5,
        help="Abort generation when the request error rate exceeds this threshold.",
    )
    parser.add_argument(
        "--min-requests-for-error-rate-check",
        type=int,
        default=10,
        help="Minimum number of requests before error-rate abort checks apply.",
    )


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-id", required=True, help="Run identifier whose handoff bundle should be prepared."
    )
    parser.add_argument(
        "--conditions",
        default="baseline,llmdebug",
        help="Comma-separated paired conditions expected in the run directory.",
    )


def _add_score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir", required=True, help="Run directory containing paired prediction files."
    )
    parser.add_argument(
        "--conditions",
        default="baseline,llmdebug",
        help="Comma-separated paired conditions to score.",
    )
    parser.add_argument(
        "--dataset-name",
        default=SWEBENCH_LITE_DATASET_NAME,
        help="Dataset name passed to the official SWE-Bench scorer.",
    )
    parser.add_argument(
        "--split",
        default=SWEBENCH_LITE_DEFAULT_SPLIT,
        help="Dataset split passed to the official SWE-Bench scorer.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Worker count for the official SWE-Bench harness.",
    )
    parser.add_argument(
        "--cache-level",
        choices=["none", "base", "env", "instance"],
        default="env",
        help="Official scorer cache level to use.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Request a clean official scoring run instead of reusing prior artifacts.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch the official scorer module.",
    )
    parser.add_argument(
        "--module",
        default="swebench.harness.run_evaluation",
        help="Python module path for the official SWE-Bench scorer entrypoint.",
    )
    parser.add_argument(
        "--no-prereq-check",
        action="store_true",
        help="Skip local prerequisite checks before launching official scoring.",
    )


def _add_prepare_scoring_smoke_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir", required=True, help="Source run directory with full predictions."
    )
    parser.add_argument(
        "--output-run-dir",
        default="",
        help="Output run directory for scoring smoke subset (default: sibling derived name).",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=2,
        help="Maximum paired instances to keep in the smoke subset.",
    )
    parser.add_argument(
        "--conditions",
        default="baseline,llmdebug",
        help="Comma-separated paired conditions to keep in the smoke subset.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SWE-Bench Lite helper CLI for llmdebug workflows."
    )
    sub = parser.add_subparsers(dest="command")

    generate = sub.add_parser("generate-lite", help="Generate SWE-Bench Lite predictions.")
    _add_generate_arguments(generate)

    prepare = sub.add_parser(
        "prepare-handoff", help="Validate run directory and write copy instructions."
    )
    _add_prepare_arguments(prepare)

    score = sub.add_parser(
        "score-official", help="Run official SWE-Bench scoring for run predictions."
    )
    _add_score_arguments(score)

    smoke = sub.add_parser(
        "prepare-scoring-smoke",
        help="Create a small paired subset run for official scoring smoke tests.",
    )
    _add_prepare_scoring_smoke_arguments(smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for SWE-Bench Lite workflows.

    Subcommands:
      generate-lite          -- query a model and write per-condition prediction files.
      prepare-handoff        -- validate a run directory and emit copy instructions.
      score-official         -- run the official SWE-Bench harness against predictions.
      prepare-scoring-smoke  -- create a small paired subset run for scoring smoke tests.

    Returns an integer exit code (0 = success, 1 = failure, 2 = usage error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    command = str(getattr(args, "command", "") or "")
    if not command:
        parser.print_help()
        return 2

    if command == "generate-lite":
        instance_ids_file = (
            _resolve_path(args.instance_ids_file) if str(args.instance_ids_file).strip() else None
        )
        snapshot_dir = _resolve_path(args.snapshot_dir) if str(args.snapshot_dir).strip() else None
        openai_api_key = str(args.openai_api_key).strip() or None
        try:
            summary = generate_lite_predictions(
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
                snapshot_dir=snapshot_dir,
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

        expected_ids_path = instance_ids_path(run_dir)
        expected_ids = None
        if expected_ids_path.exists():
            expected_ids = [
                line.strip()
                for line in expected_ids_path.read_text(encoding="utf-8").splitlines()
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

    if command == "score-official":
        run_dir = _resolve_path(str(args.run_dir))
        try:
            summary = run_official_scoring(
                run_dir=run_dir,
                conditions=str(args.conditions),
                dataset_name=str(args.dataset_name),
                split=str(args.split),
                workers=int(args.workers),
                cache_level=str(args.cache_level),
                clean=bool(args.clean),
                python_executable=str(args.python_executable),
                module=str(args.module),
                check_prereqs=not bool(args.no_prereq_check),
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Official scoring failed: {exc}")
            return 1

        print(f"Official scoring complete for run dir: {summary.run_dir}")
        for item in summary.condition_summaries:
            print(
                f"- {item.condition}: exit={item.exit_code} "
                f"log={item.log_path} workdir={item.working_dir}"
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

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
