"""Smoke-verify cloned baseline repositories used by llmdebug.

This is intentionally lightweight and stdlib-only so it can run during
`baselines/setup_repos.sh` and Docker image builds before the project itself
is installed. The goal is to fail fast on broken compatibility patches,
especially syntax errors in patched upstream entrypoints.
"""

from __future__ import annotations

import py_compile
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path


class BaselineRepoVerificationError(RuntimeError):
    """Raised when a cloned baseline repo fails smoke verification."""


@dataclass(frozen=True)
class RepoSpec:
    name: str
    required_files: tuple[str, ...]
    marker_checks: tuple[tuple[str, tuple[str, ...]], ...] = ()


_REPO_SPECS = (
    RepoSpec(
        name="LLMDebugger",
        required_files=(
            "programming/ldb.py",
            "programming/main.py",
            "programming/utils.py",
            "programming/generators/model.py",
            "programming/generators/py_generate.py",
        ),
        marker_checks=(
            ("programming/utils.py", ('"llmdebug"',)),
            (
                "programming/ldb.py",
                (
                    '"llmdebug"',
                    "llm_duration_sec",
                    "usage_available",
                    "is_passing",
                    "test_feedback",
                ),
            ),
            (
                "programming/generators/model.py",
                (
                    "_record_metrics",
                    "LLMDEBUG_BASELINE_OPENAI_REQUESTED_MODEL",
                    "super().__init__(model_name)",
                ),
            ),
            (
                "programming/generators/py_generate.py",
                ('dataset_type in ["HumanEval", "MBPP", "llmdebug"]',),
            ),
        ),
    ),
    RepoSpec(
        name="INTERVENOR",
        required_files=(
            "prompt/code_teacher.txt",
            "prompt/code_learner.txt",
        ),
    ),
    RepoSpec(
        name="TraceCoder",
        required_files=(
            "config.py",
            "datasets/llmdebug/__init__.py",
            "datasets/llmdebug/evaluation.py",
            "problem_processor.py",
            "reporting.py",
            "src/dataset_loader.py",
            "src/exact_test_context.py",
            "trace_learn_coder.py",
            "src/generation.py",
            "src/progress.py",
            "src/traceRunner.py",
        ),
        marker_checks=(
            ("config.py", ('"llmdebug"', "--dataset-path")),
            (
                "problem_processor.py",
                (
                    "repair_seeded",
                    "execution_mode",
                    "prompt_chars_total",
                    "completion_chars_total",
                    "usage_missing_call_count",
                    "method_instrumentation",
                    "method_two_step_repair",
                    "direct_gen_passed",
                    "final_passed",
                    "stopped_reason",
                    "total_llm_duration_sec",
                    "total_eval_duration_sec",
                    "build_exact_test_context",
                    "exact_test_context_meta",
                ),
            ),
            (
                "reporting.py",
                ("build_llmdebug_failure_feedback", "save_metrics_sidecar"),
            ),
            ("trace_learn_coder.py", ("llmdebug_metrics_path",)),
            (
                "src/generation.py",
                (
                    "OPENAI_BASE_URL",
                    "TRACECODER_FAKE_RESPONSES_PATH",
                    "TRACECODER_ANALYSIS_MAX_TOKENS",
                    "TRACECODER_IMPLEMENTATION_MAX_TOKENS",
                    "input_tokens",
                    "output_tokens",
                ),
            ),
            (
                "src/exact_test_context.py",
                ("build_exact_test_context", "TRACECODER_EXACT_TEST_CONTEXT_MAX_CHARS"),
            ),
        ),
    ),
    RepoSpec(
        name="lambda-RLM",
        required_files=("pyproject.toml",),
    ),
    RepoSpec(
        name="rlm",
        required_files=(
            "pyproject.toml",
            "rlm/__init__.py",
            "rlm/core/rlm.py",
            "rlm/environments/local_repl.py",
            "rlm/clients/openai.py",
        ),
    ),
)

_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "results",
    }
)


def _iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in _SKIP_DIR_NAMES for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return sorted(files)


def _assert_required_files(spec: RepoSpec, repo_dir: Path) -> None:
    for relative_path in spec.required_files:
        path = repo_dir / relative_path
        if not path.is_file():
            raise BaselineRepoVerificationError(
                f"{spec.name}: missing required file {relative_path}"
            )


def _assert_markers(spec: RepoSpec, repo_dir: Path) -> None:
    for relative_path, markers in spec.marker_checks:
        path = repo_dir / relative_path
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise BaselineRepoVerificationError(
                f"{spec.name}: failed to read {relative_path}: {exc}"
            ) from exc
        missing = [marker for marker in markers if marker not in text]
        if missing:
            rendered = ", ".join(repr(marker) for marker in missing)
            raise BaselineRepoVerificationError(
                f"{spec.name}: missing llmdebug compatibility marker(s) in {relative_path}: {rendered}"
            )


def _compile_python_files(spec: RepoSpec, repo_dir: Path) -> int:
    python_files = _iter_python_files(repo_dir)
    if not python_files:
        raise BaselineRepoVerificationError(f"{spec.name}: no Python files found to compile")
    for path in python_files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            raise BaselineRepoVerificationError(
                f"{spec.name}: syntax verification failed for {path.relative_to(repo_dir)}: {exc.msg}"
            ) from exc
    return len(python_files)


def verify_baseline_repos(root: Path) -> dict[str, int]:
    """Validate required repo layout, llmdebug patch markers, and Python syntax."""
    resolved_root = root.resolve()
    results: dict[str, int] = {}
    for spec in _REPO_SPECS:
        repo_dir = resolved_root / spec.name
        if not repo_dir.is_dir():
            raise BaselineRepoVerificationError(
                f"{spec.name}: repo directory not found at {repo_dir}"
            )
        _assert_required_files(spec, repo_dir)
        _assert_markers(spec, repo_dir)
        results[spec.name] = _compile_python_files(spec, repo_dir)
    return results


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else Path(__file__).resolve().parent / "repos"
    try:
        results = verify_baseline_repos(root)
    except BaselineRepoVerificationError as exc:
        print(f"[baseline-verify] {exc}", file=sys.stderr)
        return 1

    for name, compiled_count in results.items():
        print(f"[baseline-verify] {name}: compiled {compiled_count} Python files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
