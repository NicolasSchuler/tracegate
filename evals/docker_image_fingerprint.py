"""Compute source fingerprints for the eval Docker runner image."""

from __future__ import annotations

import hashlib
from pathlib import Path

_FINGERPRINT_FILES = ("README.md", "pyproject.toml", "uv.lock")
_FINGERPRINT_DIRS = ("src", "evals", "baselines")
_EXCLUDED_PREFIXES = (
    "baselines/envs",
    "baselines/repos",
    "evals/artifacts",
    "evals/results",
    "evals/swebench_runs",
)
_EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".llmdebug",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
_EXCLUDED_SUFFIXES = {".pyc", ".pyd", ".pyo"}
_READ_CHUNK_BYTES = 1024 * 1024


def _should_skip_relative_path(relative_path: Path) -> bool:
    relative_posix = relative_path.as_posix()
    if any(
        relative_posix == prefix or relative_posix.startswith(f"{prefix}/")
        for prefix in _EXCLUDED_PREFIXES
    ):
        return True
    if any(part in _EXCLUDED_DIR_NAMES for part in relative_path.parts):
        return True
    return relative_path.suffix in _EXCLUDED_SUFFIXES


def _iter_fingerprint_entries(repo_root: Path) -> list[tuple[str, Path | None]]:
    entries: list[tuple[str, Path | None]] = []

    for rel_str in _FINGERPRINT_FILES:
        relative_path = Path(rel_str)
        path = repo_root / relative_path
        entries.append((relative_path.as_posix(), path if path.is_file() else None))

    for rel_str in _FINGERPRINT_DIRS:
        relative_root = Path(rel_str)
        root = repo_root / relative_root
        if not root.exists():
            entries.append((f"{relative_root.as_posix()}/", None))
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(repo_root)
            if _should_skip_relative_path(relative_path):
                continue
            entries.append((relative_path.as_posix(), path))

    return entries


def compute_eval_image_build_fingerprint(repo_root: Path, *, include_baselines: bool) -> str:
    """Return a deterministic fingerprint for the eval Docker image inputs."""

    resolved_root = repo_root.resolve()
    # Keep this helper stdlib-only so run_eval_docker.sh can compute fingerprints
    # with plain python3, even before project dependencies are installed.
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(f"include_baselines={int(include_baselines)}\n".encode())

    for relative_name, path in _iter_fingerprint_entries(resolved_root):
        if path is None:
            hasher.update(f"MISSING:{relative_name}\n".encode())
            continue

        hasher.update(f"FILE:{relative_name}\n".encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
                hasher.update(chunk)
        hasher.update(b"\n")

    return hasher.hexdigest()
