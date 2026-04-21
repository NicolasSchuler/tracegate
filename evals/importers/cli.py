"""CLI entry point for all dataset importers."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.run_eval import (
    add_execution_arguments,
    build_execution_config_from_args,
    check_testcontainers_runtime_available,
    set_active_execution_config,
)
from evals.validate_cases import check_pytest_available

from .. import _json as json


def _resolve_cli_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _print_case_import_summary(
    *, source: str, dry_run: bool, results: list[dict[str, Any]]
) -> None:
    mode = "previewed" if dry_run else "imported"
    print(f"\n{len(results)} case(s) {mode} from {source}.")

    if dry_run and results:
        print("\nPreview:")
        for result in results[:5]:
            print(
                f"  {result['case_id']}: "
                f"{json.dumps(result.get('metadata', {}), indent=None)[:120]}..."
            )
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more")


@dataclass(frozen=True)
class CommandSpec:
    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    handler: Callable[[argparse.Namespace], int]


def _build_importer_handler(
    *,
    module_path: str,
    function_name: str,
    kwargs_builder: Callable[[argparse.Namespace], dict[str, Any]],
) -> Callable[[argparse.Namespace], int]:
    def _handler(args: argparse.Namespace) -> int:
        output_dir = _resolve_cli_path(str(args.output_dir))
        module = importlib.import_module(module_path)
        importer = getattr(module, function_name)
        kwargs = kwargs_builder(args)
        results = importer(output_dir, **kwargs)
        _print_case_import_summary(
            source=str(args.source), dry_run=bool(args.dry_run), results=results
        )
        return 0

    return _handler


def _handle_prepare_top3(args: argparse.Namespace) -> int:
    from evals.importers.top3_bootstrap import prepare_top3_lock_manifest

    manifest_path = _resolve_cli_path(str(args.manifest))
    manifest = prepare_top3_lock_manifest(manifest_path)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Datasets pinned: {', '.join(dataset.source for dataset in manifest.datasets)}")
    return 0


def _handle_materialize_top3(args: argparse.Namespace) -> int:
    from evals.importers.top3_bootstrap import materialize_top3, report_has_hard_failures

    if int(args.jobs) < 1:
        print("Invalid --jobs: must be >= 1")
        return 2
    if int(args.max_cases) < 0:
        print("Invalid --max-cases: must be >= 0")
        return 2

    report, report_path = materialize_top3(
        manifest_path=_resolve_cli_path(str(args.manifest)),
        output_dir=_resolve_cli_path(str(args.output_dir)),
        data_root=_resolve_cli_path(str(args.data_root)),
        download_policy=str(args.download_policy),
        jobs=int(args.jobs),
        max_cases=int(args.max_cases),
        validate=bool(args.validate),
        dry_run=bool(args.dry_run),
    )

    total_attempted = sum(source.attempted for source in report.sources)
    total_imported = sum(source.imported for source in report.sources)
    total_skipped = sum(source.skipped for source in report.sources)
    print(f"\nMaterialization report: {report_path}")
    print(
        f"Summary: attempted={total_attempted}, imported={total_imported}, skipped={total_skipped}"
    )
    for source in report.sources:
        hard_flag = " hard_failed=yes" if source.hard_failed else ""
        print(
            f"  - {source.source}: "
            f"attempted={source.attempted}, imported={source.imported}, "
            f"skipped={source.skipped}{hard_flag}"
        )

    if report_has_hard_failures(report):
        print("Materialization completed with mandatory source failures. See report for details.")
        return 1
    return 0


def _add_common_import_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="evals/cases")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _add_debugbench_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-cases", type=int, default=0, help="Max cases to import (0 = unlimited)."
    )
    _add_common_import_arguments(parser)


def _add_pyresbugs_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-cases", type=int, default=0, help="Max cases to import (0 = unlimited)."
    )
    _add_common_import_arguments(parser)


def _add_humanevalpack_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-cases", type=int, default=20)
    _add_common_import_arguments(parser)


def _add_condefects_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-path", required=True, help="Path to cloned ConDefects repo.")
    parser.add_argument(
        "--test-zip",
        required=True,
        help="Path to Test.zip with test input/output data.",
    )
    parser.add_argument(
        "--max-cases", type=int, default=0, help="Max cases to import (0 = unlimited)."
    )
    _add_common_import_arguments(parser)


def _add_mutmut_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", required=True, help="Python module/package to mutate.")
    parser.add_argument("--max-cases", type=int, default=30)
    _add_common_import_arguments(parser)


def _add_quixbugs_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-path", required=True, help="Path to cloned QuixBugs repo.")
    parser.add_argument(
        "--max-cases", type=int, default=0, help="Max cases to import (0 = unlimited)."
    )
    _add_common_import_arguments(parser)


def _add_prepare_top3_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default="evals/manifests/top3.lock.json")


def _add_materialize_top3_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default="evals/manifests/top3.lock.json")
    parser.add_argument("--output-dir", default="evals/cases")
    parser.add_argument("--data-root", default="evals/data/top3")
    parser.add_argument(
        "--download-policy",
        choices=["never", "missing", "refresh"],
        default="missing",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


_COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name="debugbench",
        help="Import from DebugBench (HuggingFace).",
        add_arguments=_add_debugbench_arguments,
        handler=_build_importer_handler(
            module_path="evals.importers.debugbench_importer",
            function_name="import_debugbench",
            kwargs_builder=lambda args: {
                "max_cases": args.max_cases,
                "validate": args.validate,
                "dry_run": args.dry_run,
            },
        ),
    ),
    CommandSpec(
        name="pyresbugs",
        help="Import from PyResBugs (HuggingFace).",
        add_arguments=_add_pyresbugs_arguments,
        handler=_build_importer_handler(
            module_path="evals.importers.pyresbugs_importer",
            function_name="import_pyresbugs",
            kwargs_builder=lambda args: {
                "max_cases": args.max_cases,
                "validate": args.validate,
                "dry_run": args.dry_run,
            },
        ),
    ),
    CommandSpec(
        name="humanevalpack",
        help="Import from HumanEvalPack (HuggingFace).",
        add_arguments=_add_humanevalpack_arguments,
        handler=_build_importer_handler(
            module_path="evals.importers.humanevalpack_importer",
            function_name="import_humanevalpack",
            kwargs_builder=lambda args: {
                "max_cases": args.max_cases,
                "validate": args.validate,
                "dry_run": args.dry_run,
            },
        ),
    ),
    CommandSpec(
        name="condefects",
        help="Import from ConDefects (local clone + Test.zip).",
        add_arguments=_add_condefects_arguments,
        handler=_build_importer_handler(
            module_path="evals.importers.condefects_importer",
            function_name="import_condefects",
            kwargs_builder=lambda args: {
                "repo_path": Path(args.repo_path),
                "test_zip": Path(args.test_zip),
                "max_cases": args.max_cases,
                "validate": args.validate,
                "dry_run": args.dry_run,
            },
        ),
    ),
    CommandSpec(
        name="mutmut",
        help="Generate cases from mutmut mutations.",
        add_arguments=_add_mutmut_arguments,
        handler=_build_importer_handler(
            module_path="evals.importers.mutmut_generator",
            function_name="import_mutmut",
            kwargs_builder=lambda args: {
                "target": args.target,
                "max_cases": args.max_cases,
                "validate": args.validate,
                "dry_run": args.dry_run,
            },
        ),
    ),
    CommandSpec(
        name="quixbugs",
        help="Import from QuixBugs (local clone).",
        add_arguments=_add_quixbugs_arguments,
        handler=_build_importer_handler(
            module_path="evals.importers.quixbugs_importer",
            function_name="import_quixbugs",
            kwargs_builder=lambda args: {
                "repo_path": Path(args.repo_path),
                "max_cases": args.max_cases,
                "validate": args.validate,
                "dry_run": args.dry_run,
            },
        ),
    ),
    CommandSpec(
        name="prepare-top3",
        help="Generate lock manifest for BugsInPy, PyBugHive, TrickyBugs (no heavy downloads).",
        add_arguments=_add_prepare_top3_arguments,
        handler=_handle_prepare_top3,
    ),
    CommandSpec(
        name="materialize-top3",
        help="Materialize top-3 datasets from lock manifest into eval cases.",
        add_arguments=_add_materialize_top3_arguments,
        handler=_handle_materialize_top3,
    ),
)

_COMMAND_REGISTRY = {spec.name: spec for spec in _COMMAND_SPECS}


def _build_parser() -> argparse.ArgumentParser:
    execution_flags_parser = argparse.ArgumentParser(add_help=False)
    add_execution_arguments(execution_flags_parser, include_defaults=False)

    parser = argparse.ArgumentParser(
        description="Import external dataset cases into the llmdebug framework."
    )
    add_execution_arguments(parser)
    subparsers = parser.add_subparsers(dest="source", help="Dataset source to import from.")

    for spec in _COMMAND_SPECS:
        subparser = subparsers.add_parser(
            spec.name,
            help=spec.help,
            parents=[execution_flags_parser],
        )
        spec.add_arguments(subparser)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: import external dataset cases into the llmdebug framework.

    Dispatches to one of: ``debugbench``, ``pyresbugs``, ``humanevalpack``,
    ``condefects``, ``mutmut``, ``quixbugs``, ``prepare-top3``, or
    ``materialize-top3``.  Accepts ``--output-dir``, ``--validate``,
    ``--dry-run``, and execution-backend flags shared across importers.

    Returns an integer exit code (0 = success, 1 = failure, 2 = usage error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        execution_config = build_execution_config_from_args(args)
    except ValueError as exc:
        print(f"Invalid arguments: {exc}")
        return 2

    source = str(getattr(args, "source", "") or "")
    if not source:
        parser.print_help()
        return 2

    spec = _COMMAND_REGISTRY.get(source)
    if spec is None:
        print(f"Unknown source: {source}")
        return 2

    if source != "prepare-top3" and bool(getattr(args, "validate", False)):
        if execution_config.executor == "local":
            ok, message = check_pytest_available()
        else:
            ok, message = check_testcontainers_runtime_available()
        if not ok:
            print(f"Environment check failed: {message}")
            return 2

    previous_execution_config = set_active_execution_config(execution_config)
    try:
        return int(spec.handler(args))
    finally:
        set_active_execution_config(previous_execution_config)


if __name__ == "__main__":
    raise SystemExit(main())
