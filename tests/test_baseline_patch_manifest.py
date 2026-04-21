"""Tests for explicit baseline patch/env-shim inventory."""

from __future__ import annotations

from pathlib import Path

from baselines.patch_manifest import PATCH_MANIFEST


def test_patch_manifest_entries_point_to_existing_files() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for entry in PATCH_MANIFEST:
        path = repo_root / entry.relative_path
        assert path.is_file(), f"missing manifest file: {entry.relative_path}"


def test_patch_manifest_entries_have_scope_and_rationale() -> None:
    for entry in PATCH_MANIFEST:
        assert entry.scope
        assert entry.behavioral_impact.strip()
        assert entry.rationale.strip()
        assert entry.upstream_commit.strip()


def test_patch_manifest_covers_repo_patches_and_env_shims() -> None:
    relative_paths = {entry.relative_path for entry in PATCH_MANIFEST}
    assert "baselines/ldb/patches/0001-add-llmdebug-dataset-support.patch" in relative_paths
    assert "baselines/tracecoder/patches/0001-add-llmdebug-repair-mode.patch" in relative_paths
    assert "baselines/intervenor/env_client.py" in relative_paths
    assert "baselines/lambda_rlm/env_client.py" in relative_paths
    assert "baselines/rlm/env_client.py" in relative_paths
