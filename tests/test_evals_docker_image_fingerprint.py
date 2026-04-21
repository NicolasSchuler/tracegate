"""Tests for eval Docker image fingerprint inputs."""

from __future__ import annotations

from pathlib import Path

from evals.docker_image_fingerprint import compute_eval_image_build_fingerprint


def _write_minimal_repo(root: Path) -> None:
    (root / "README.md").write_text("readme\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        """
[project]
name = "fingerprint-demo"
version = "0.0.0"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    (root / "src").mkdir()
    (root / "src" / "pkg.py").write_text("VALUE = 1\n", encoding="utf-8")

    (root / "evals" / "docker").mkdir(parents=True)
    (root / "evals" / "docker" / "eval.Dockerfile").write_text(
        "FROM scratch\n",
        encoding="utf-8",
    )

    (root / "baselines").mkdir()


def test_compute_eval_image_build_fingerprint_changes_when_eval_dockerfile_changes(
    tmp_path: Path,
) -> None:
    _write_minimal_repo(tmp_path)

    before = compute_eval_image_build_fingerprint(tmp_path, include_baselines=False)

    dockerfile = tmp_path / "evals" / "docker" / "eval.Dockerfile"
    dockerfile.write_text("FROM busybox\n", encoding="utf-8")

    after = compute_eval_image_build_fingerprint(tmp_path, include_baselines=False)

    assert after != before
