"""Utilities for downloading and caching SWE-bench Pro per-instance run scripts."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

from .types import SWEBENCH_PRO_GITHUB_REPO

# Default commit to pin for reproducibility
DEFAULT_PIN_COMMIT = "66a831e8104b3205801113bccb255fc523e000b6"


def default_cache_dir() -> Path:
    """Return the default cache directory for SWE-bench Pro assets.

    Returns:
        ``<repo_root>/evals/swebench_pro_assets``.
    """
    return Path(__file__).resolve().parents[1] / "swebench_pro_assets"


def clone_or_update_repo(
    *,
    cache_dir: Path | None = None,
    commit: str = DEFAULT_PIN_COMMIT,
    force: bool = False,
) -> Path:
    """Clone (or update) the SWE-bench Pro repository for run_scripts and Dockerfiles.

    Uses a sparse checkout to fetch only the ``run_scripts/``, ``dockerfiles/``,
    and ``helper_code/`` directories, keeping the clone lightweight.

    Args:
        cache_dir: Directory to store the cloned repo.  Defaults to
            :func:`default_cache_dir`.
        commit: Git commit SHA to pin.  Defaults to :data:`DEFAULT_PIN_COMMIT`.
        force: If ``True``, remove existing clone and start fresh.

    Returns:
        Path to the cloned repository directory.

    Raises:
        RuntimeError: If git commands fail.
    """
    base = cache_dir or default_cache_dir()
    repo_dir = base / "SWE-bench_Pro-os"

    if force and repo_dir.exists():
        shutil.rmtree(repo_dir)

    if repo_dir.exists() and (repo_dir / ".git").exists():
        # Update existing clone
        _run_git(["fetch", "origin"], cwd=repo_dir)
        _run_git(["checkout", commit], cwd=repo_dir)
        return repo_dir

    # Fresh sparse clone
    base.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://github.com/{SWEBENCH_PRO_GITHUB_REPO}.git"

    _run_git(
        ["clone", "--no-checkout", "--filter=blob:none", repo_url, str(repo_dir)],
        cwd=base,
    )
    _run_git(["sparse-checkout", "init", "--cone"], cwd=repo_dir)
    _run_git(
        ["sparse-checkout", "set", "run_scripts", "dockerfiles", "helper_code"],
        cwd=repo_dir,
    )
    _run_git(["checkout", commit], cwd=repo_dir)

    return repo_dir


def run_scripts_dir(repo_dir: Path) -> Path:
    """Return the path to the ``run_scripts/`` directory within the cloned repo."""
    return repo_dir / "run_scripts"


def dockerfiles_dir(repo_dir: Path) -> Path:
    """Return the path to the ``dockerfiles/`` directory within the cloned repo."""
    return repo_dir / "dockerfiles"


def validate_instance_scripts(
    *,
    scripts_dir: Path,
    instance_id: str,
) -> tuple[Path, Path]:
    """Validate that run_script.sh and parser.py exist for an instance.

    Args:
        scripts_dir: Root ``run_scripts/`` directory.
        instance_id: SWE-bench Pro instance identifier.

    Returns:
        Tuple of ``(run_script_path, parser_path)``.

    Raises:
        FileNotFoundError: If either required file is missing.
    """
    instance_dir = scripts_dir / instance_id
    run_script = instance_dir / "run_script.sh"
    parser = instance_dir / "parser.py"

    if not run_script.exists():
        raise FileNotFoundError(f"run_script.sh not found for {instance_id}: {run_script}")
    if not parser.exists():
        raise FileNotFoundError(f"parser.py not found for {instance_id}: {parser}")

    return run_script, parser


def load_run_script(scripts_dir: Path, instance_id: str) -> str:
    """Load the run_script.sh content for an instance."""
    run_script, _ = validate_instance_scripts(scripts_dir=scripts_dir, instance_id=instance_id)
    return run_script.read_text(encoding="utf-8")


def load_parser_script(scripts_dir: Path, instance_id: str) -> str:
    """Load the parser.py content for an instance."""
    _, parser = validate_instance_scripts(scripts_dir=scripts_dir, instance_id=instance_id)
    return parser.read_text(encoding="utf-8")


def load_base_dockerfile(dockerfiles_root: Path, instance_id: str) -> str:
    """Load the base Dockerfile content for an instance.

    Args:
        dockerfiles_root: Root ``dockerfiles/`` directory.
        instance_id: SWE-bench Pro instance identifier.

    Returns:
        Dockerfile content as a string.

    Raises:
        FileNotFoundError: If the Dockerfile is missing.
    """
    path = dockerfiles_root / "base_dockerfile" / instance_id / "Dockerfile"
    if not path.exists():
        raise FileNotFoundError(f"Base Dockerfile not found for {instance_id}: {path}")
    return path.read_text(encoding="utf-8")


def load_instance_dockerfile(dockerfiles_root: Path, instance_id: str) -> str:
    """Load the instance Dockerfile content for an instance.

    Args:
        dockerfiles_root: Root ``dockerfiles/`` directory.
        instance_id: SWE-bench Pro instance identifier.

    Returns:
        Dockerfile content as a string.

    Raises:
        FileNotFoundError: If the Dockerfile is missing.
    """
    path = dockerfiles_root / "instance_dockerfile" / instance_id / "Dockerfile"
    if not path.exists():
        raise FileNotFoundError(f"Instance Dockerfile not found for {instance_id}: {path}")
    return path.read_text(encoding="utf-8")


def extract_env_commands(
    dockerfiles_root: Path,
    instance_id: str,
) -> list[str]:
    """Extract ENV commands from both base and instance Dockerfiles.

    Converts Docker ``ENV`` directives to shell ``export`` statements.

    Args:
        dockerfiles_root: Root ``dockerfiles/`` directory.
        instance_id: SWE-bench Pro instance identifier.

    Returns:
        List of ``export VAR=VALUE`` shell statements.
    """
    env_cmds: list[str] = []
    for loader in (load_base_dockerfile, load_instance_dockerfile):
        try:
            content = loader(dockerfiles_root, instance_id)
        except FileNotFoundError:
            continue
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("ENV"):
                env_cmds.append(_translate_env_directive(stripped))
    return env_cmds


def _translate_env_directive(raw_line: str) -> str:
    """Translate one Docker ``ENV`` directive into a shell ``export`` line."""
    body = raw_line[3:].strip()
    if not body:
        return "export"

    try:
        tokens = shlex.split(body, posix=True)
    except ValueError:
        tokens = body.split()

    if not tokens:
        return "export"
    if "=" in tokens[0]:
        return f"export {body}"

    key = tokens[0]
    value = " ".join(tokens[1:])
    if not value:
        return f"export {key}"
    if any(char.isspace() for char in value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
        return f'export {key}="{escaped}"'
    return f"export {key}={value}"


def list_available_instances(scripts_dir: Path) -> list[str]:
    """Return sorted list of instance IDs available in the run_scripts directory."""
    if not scripts_dir.exists():
        return []
    return sorted(
        d.name for d in scripts_dir.iterdir() if d.is_dir() and d.name.startswith("instance_")
    )


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a git command and raise on failure."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"git {' '.join(args[:3])} failed (exit {result.returncode}): {stderr}")
    return result
