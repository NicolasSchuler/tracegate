"""Run-level collaborators of the eval flows, passed explicitly so tests can substitute them.

``run_eval`` builds an :class:`EvalSeams` from its own module attributes when a run
starts, so ``monkeypatch.setattr(run_eval, "load_cases", ...)`` keeps working for
anything entered through ``run_eval``. Code that calls a flow directly gets
:func:`default_eval_seams`, which reads the owning modules at call time. This
replaces the forwarding shims that made the flows import ``run_eval`` back (X-02).

Attempt-level collaborators (``run_pytest``, ``apply_diff``, ``read_latest_snapshot``,
...) are plain imports in ``evals.eval_attempt_flow``; patch them there.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals import (
    case_loading,
    eval_attempt_flow,
    eval_config,
    eval_containers,
    eval_recording,
    patcher_factory,
    run_state,
)
from evals.eval_types import CaseSpec

# Results, artifacts, policy state and baseline envs live under the checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalSeams:
    """The collaborators a run reaches through ``run_eval``'s module attributes."""

    repo_root: Path
    load_cases: Callable[[Path], list[CaseSpec]]
    make_patcher: Callable[[argparse.Namespace], Any]
    run_one_condition: Callable[..., list[dict[str, Any]]]
    build_run_settings_fingerprint: Callable[..., str]
    write_resolved_run_config_lockfile: Callable[..., Path]
    load_existing_run_state: Callable[..., Any]
    repair_trailing_results_tail: Callable[..., tuple[Path, int]]
    check_testcontainers_runtime_available: Callable[[], tuple[bool, str]]
    cleanup_stale_eval_containers: Callable[..., tuple[int, int, str | None]]
    run_results_lock: type[eval_recording._RunResultsLock]


def default_eval_seams() -> EvalSeams:
    """Seams bound to the owning modules' current attributes."""
    return EvalSeams(
        repo_root=REPO_ROOT,
        load_cases=case_loading.load_cases,
        make_patcher=patcher_factory.make_patcher,
        run_one_condition=eval_attempt_flow.run_one_condition_impl,
        build_run_settings_fingerprint=run_state._build_run_settings_fingerprint,
        write_resolved_run_config_lockfile=eval_config._write_resolved_run_config_lockfile,
        load_existing_run_state=run_state._load_existing_run_state,
        repair_trailing_results_tail=run_state._repair_trailing_results_tail,
        check_testcontainers_runtime_available=eval_containers.check_testcontainers_runtime_available,
        cleanup_stale_eval_containers=eval_containers.cleanup_stale_eval_containers,
        run_results_lock=eval_recording._RunResultsLock,
    )
