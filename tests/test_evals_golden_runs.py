"""Golden fixture runs: the eval harness must reproduce its committed records exactly.

Phase 5 of the 2026-10-04 review restructures the harness without changing behaviour.
This test runs every scenario in ``tests/golden_eval/scenarios.py`` against a scripted
stub model and compares records, policy logs, artifacts, model requests and the analysis
output with ``tests/golden/eval_runs/``. Golden files depend on the checkout path, the
interpreter and ``uv.lock``, so the test skips in any other environment; regenerate them
there with ``just golden-evals-update`` on a commit whose behaviour you trust.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.golden_eval import runner

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_golden_eval_runs_match(tmp_path: Path) -> None:
    mismatch = runner.golden_environment_mismatch()
    if mismatch is not None:
        pytest.skip(mismatch)

    names = sorted(runner.SCENARIOS)
    errors = runner.run_scenarios(names, tmp_path)
    assert not errors, errors

    differences = {
        name: runner.compare(name, (tmp_path / f"{name}.json").read_text(encoding="utf-8"))
        for name in names
    }
    assert not any(differences.values()), "\n".join(
        f"{name}:\n  " + "\n  ".join(lines) for name, lines in differences.items() if lines
    )


def test_every_scenario_has_a_golden_file() -> None:
    expected = {f"{name}.json.gz" for name in runner.SCENARIOS}
    present = {path.name for path in runner.GOLDEN_DIR.glob("*.json.gz")}
    assert present == expected


def test_the_golden_environment_file_is_committed() -> None:
    """Without it every comparison exits 2 and this suite skips, so the lock is unused."""
    environment_path = runner.environment_path()
    assert environment_path.is_file(), (
        f"{environment_path} is missing: regenerate the golden files with "
        "`just golden-evals-update` and commit it alongside them"
    )
    recorded = json.loads(environment_path.read_text(encoding="utf-8"))
    assert set(recorded) == set(runner.environment())
