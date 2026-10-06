"""Shared helpers for strict, native dataset reproducibility importers."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import _json as json

SKIP_UNSUPPORTED_NATIVE_COMMAND = "unsupported_native_test_command"
SKIP_NATIVE_COMMAND_MISSING_SELECTOR = "native_command_missing_test_selector"
SKIP_UNSAFE_DATASET_PATH = "unsafe_dataset_path"
SKIP_GIT_CLONE_FAILED = "git_clone_failed"
SKIP_UNSAFE_REPO_URL = "unsafe_repo_url"
SKIP_GIT_SHOW_FAILED = "git_show_failed"
SKIP_UNSUPPORTED_PATCH_FILE_COUNT = "unsupported_patch_file_count"


_SINGLE_LINE_SET_RE = re.compile(r"^set\s+[-+a-zA-Z]+$")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


@dataclass(frozen=True)
class NativePytestCommand:
    """Normalized, lossless representation of a native pytest command."""

    source_test_command: str
    python_args: list[str]
    source_test_selector: str
    test_paths: list[str]


@dataclass(frozen=True)
class GitShowFileRequest:
    commitish: str
    rel_path: str


def normalize_relative_path(raw_path: str) -> str | None:
    """Return a safe, normalized POSIX-like relative path."""
    normalized = raw_path.strip().replace("\\", "/")
    if not normalized:
        return None
    pure = Path(normalized)
    if pure.is_absolute():
        return None
    if any(part in {"", ".", ".."} for part in pure.parts):
        return None
    return "/".join(pure.parts)


def normalize_native_pytest_command(command_text: str) -> NativePytestCommand | None:
    """Normalize native pytest commands without semantic rewrite.

    Accepted forms:
    - pytest ...
    - py.test ...
    - python -m pytest ...
    - python3 -m pytest ...

    Any shell composition, env-wrapper execution, or multi-command scripts are rejected.
    """
    line = _extract_single_command_line(command_text)
    if line is None:
        return None
    if any(token in line for token in ("&&", "||", "|", ";", "`", "$(", "<", ">")):
        return None

    try:
        tokens = shlex.split(line, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None
    if _starts_with_env_assignment(tokens):
        return None

    normalized_tail: list[str]
    head = Path(tokens[0]).name.lower()
    if head in {"pytest", "py.test"}:
        normalized_tail = tokens[1:]
    elif head in {"python", "python3"}:
        if len(tokens) < 3 or tokens[1] != "-m" or tokens[2] != "pytest":
            return None
        normalized_tail = tokens[3:]
    else:
        return None

    python_args = ["-m", "pytest", *normalized_tail]
    selector_tokens = _collect_selector_tokens(normalized_tail)
    test_paths = _extract_test_paths(selector_tokens)
    if not test_paths:
        return None

    selector = " ".join(selector_tokens) if selector_tokens else " ".join(normalized_tail)
    selector = selector.strip()
    if not selector:
        selector = " ".join(normalized_tail)

    return NativePytestCommand(
        source_test_command=line,
        python_args=python_args,
        source_test_selector=selector,
        test_paths=test_paths,
    )


# Dataset-supplied clone URLs: https or scp-style ssh only. A leading "-" would be parsed
# as a git option, and transports such as ext:: can run commands (DAT-12).
_SAFE_REPO_URL_RE = re.compile(r"^(?:https://[^\s]+|git@[A-Za-z0-9.-]+:[^\s]+)$")


def is_safe_repo_url(repo_url: str) -> bool:
    """Whether *repo_url* is an ``https://`` or ``git@host:path`` URL."""
    return bool(_SAFE_REPO_URL_RE.match(repo_url))


def ensure_git_repo(repo_url: str, clone_dir: Path) -> tuple[bool, str | None]:
    """Clone or refresh a git repository for deterministic file reads."""
    if not is_safe_repo_url(repo_url):
        return False, SKIP_UNSAFE_REPO_URL
    try:
        if clone_dir.exists() and (clone_dir / ".git").exists():
            proc = subprocess.run(
                ["git", "fetch", "--all", "--tags", "--prune"],
                cwd=clone_dir,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if proc.returncode == 0:
                return True, None
            shutil.rmtree(clone_dir, ignore_errors=True)

        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["git", "clone", "--filter=blob:none", "--", repo_url, str(clone_dir)],
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        if proc.returncode != 0:
            return False, SKIP_GIT_CLONE_FAILED
        return True, None
    except (OSError, subprocess.TimeoutExpired):
        return False, SKIP_GIT_CLONE_FAILED


def git_show_file(repo_dir: Path, request: GitShowFileRequest) -> str | None:
    """Read file content at `commitish:rel_path` from a local git repo."""
    rel_path = normalize_relative_path(request.rel_path)
    if rel_path is None or request.commitish.startswith("-"):
        return None  # a leading "-" would be read as a git option (DAT-12)
    try:
        proc = subprocess.run(
            ["git", "show", f"{request.commitish}:{rel_path}"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def write_native_case_files(
    case_dir: Path,
    *,
    buggy_files: dict[str, str],
    test_files: dict[str, str],
    oracle_patch: str,
    metadata: dict[str, Any],
) -> None:
    """Write a case preserving source-relative paths for native tests."""
    case_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, content in buggy_files.items():
        _write_relative_file(case_dir, rel_path, content)
    for rel_path, content in test_files.items():
        _write_relative_file(case_dir, rel_path, content)

    # Contract compatibility file. Native `python_args` will usually target dataset tests.
    if "test_buggy.py" not in {Path(p).name for p in [*buggy_files, *test_files]}:
        (case_dir / "test_buggy.py").write_text(
            "def test_native_dataset_placeholder_contract_only():\n    assert True\n",
            encoding="utf-8",
        )

    (case_dir / "oracle.patch").write_text(oracle_patch, encoding="utf-8")
    (case_dir / "case.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _extract_single_command_line(command_text: str) -> str | None:
    lines: list[str] = []
    for raw in command_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#!"):
            continue
        if line.startswith("#"):
            continue
        if _SINGLE_LINE_SET_RE.match(line):
            continue
        lines.append(line)
    if len(lines) != 1:
        return None
    return lines[0]


def _starts_with_env_assignment(tokens: list[str]) -> bool:
    if not tokens:
        return False
    return bool(_ENV_ASSIGN_RE.match(tokens[0]))


def _collect_selector_tokens(pytest_args: list[str]) -> list[str]:
    selectors: list[str] = []
    i = 0
    while i < len(pytest_args):
        token = pytest_args[i]
        if token == "--":
            i += 1
            continue
        if token.startswith("-"):
            i += _option_arity(token)
            continue
        selectors.append(token)
        i += 1
    return selectors


def _option_arity(option: str) -> int:
    # Conservative set of options that take exactly one value.
    value_options = {
        "-k",
        "-m",
        "-p",
        "-c",
        "--maxfail",
        "--junitxml",
        "--rootdir",
        "--confcutdir",
        "--basetemp",
        "--ignore",
        "--ignore-glob",
        "--deselect",
        "--log-level",
        "--log-format",
        "--log-date-format",
    }
    if option in value_options:
        return 2
    if option.startswith("--") and "=" in option:
        return 1
    return 1


def _extract_test_paths(selector_tokens: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in selector_tokens:
        raw = token.split("::", 1)[0]
        normalized = normalize_relative_path(raw)
        if normalized is None:
            continue
        if not _is_supported_selector_path(raw=raw, normalized=normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _is_supported_selector_path(*, raw: str, normalized: str) -> bool:
    if normalized.endswith(".py"):
        return True
    # Accept deterministic directory selectors (for example `tests/`, `tests/unit`, `pkg/tests`).
    if raw.endswith("/") or raw.endswith("\\"):
        return True
    if "/" in normalized:
        return True
    return normalized in {"test", "tests"}


def _write_relative_file(case_dir: Path, rel_path: str, content: str) -> None:
    normalized = normalize_relative_path(rel_path)
    if normalized is None:
        raise ValueError(f"unsafe relative path: {rel_path!r}")
    target = case_dir / normalized
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
