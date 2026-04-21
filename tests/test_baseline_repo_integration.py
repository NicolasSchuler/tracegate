"""Static checks around baseline setup scripts and committed patch assets."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SETUP_SCRIPT = _REPO_ROOT / "baselines" / "setup_repos.sh"
_SETUP_ENVS_SCRIPT = _REPO_ROOT / "baselines" / "setup_envs.sh"


def _patch_index_entries(patch_path: Path) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    current_path: str | None = None
    for line in patch_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("diff --git a/") and " b/" in line:
            current_path = line.removeprefix("diff --git a/").split(" b/", maxsplit=1)[1]
            continue
        if current_path is None or not line.startswith("index "):
            continue
        hashes = line.split(maxsplit=2)[1]
        before_hash, after_hash = hashes.split("..", maxsplit=1)
        entries.append((current_path, before_hash, after_hash))
        current_path = None
    return entries


def test_setup_repos_script_parses() -> None:
    result = subprocess.run(
        ["bash", "-n", str(_SETUP_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_setup_envs_script_parses() -> None:
    result = subprocess.run(
        ["bash", "-n", str(_SETUP_ENVS_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_baseline_patch_files_are_syntactically_valid() -> None:
    patch_paths = sorted((_REPO_ROOT / "baselines").glob("*/patches/*.patch"))
    assert patch_paths, "expected at least one committed baseline patch file"

    for patch_path in patch_paths:
        result = subprocess.run(
            ["git", "apply", "--stat", str(patch_path)],
            check=False,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0, f"{patch_path}:\n{result.stderr}{result.stdout}"


def test_baseline_patch_stacks_have_consistent_index_chains() -> None:
    patch_dirs = sorted((_REPO_ROOT / "baselines").glob("*/patches"))
    assert patch_dirs, "expected at least one committed baseline patch directory"

    for patch_dir in patch_dirs:
        patch_paths = sorted(patch_dir.glob("*.patch"))
        previous_outputs: dict[str, str] = {}
        for patch_path in patch_paths:
            for relative_path, before_hash, after_hash in _patch_index_entries(patch_path):
                previous_output = previous_outputs.get(relative_path)
                if previous_output is not None:
                    assert before_hash == previous_output, (
                        f"{patch_path}: expected preimage {previous_output} for {relative_path}, "
                        f"found {before_hash}; regenerate the dependent patch stack in order."
                    )
                previous_outputs[relative_path] = after_hash


def test_setup_scripts_reference_lambda_and_rlm_backends() -> None:
    repos_script = _SETUP_SCRIPT.read_text(encoding="utf-8")
    envs_script = _SETUP_ENVS_SCRIPT.read_text(encoding="utf-8")

    assert "lambda-RLM" in repos_script
    assert "baselines.pins" in repos_script
    assert "PINNED_REPOS" in repos_script
    assert "lambda_rlm" in envs_script
    assert "rlm" in envs_script
    assert "BASELINE_REPO_ROOT" in envs_script
