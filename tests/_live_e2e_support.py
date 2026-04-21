"""Shared support for live repair end-to-end integration tests."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from baselines.adapter_base import BaselineResult

# Keep each live test job inside a single 10-minute envelope by default.
# Increase these locally only when you intentionally want longer live runs.
API_TIMEOUT_SEC = 600.0
PROCESS_TIMEOUT_SEC = 600.0
CASE_TIMEOUT_SEC = 600.0

SOLVABLE_BUGGY = textwrap.dedent("""\
    def multiply(a, b):
        return a + b
""")

SOLVABLE_TEST = textwrap.dedent("""\
    from buggy import multiply


    def test_multiply_positive():
        assert multiply(3, 4) == 12


    def test_multiply_zero():
        assert multiply(0, 5) == 0


    def test_multiply_negative():
        assert multiply(-2, 3) == -6
""")

UNSOLVABLE_BUGGY = textwrap.dedent("""\
    def evaluate(expr):
        \"\"\"Evaluate a mathematical expression string.

        Supports: +, -, *, /, ** (power), unary minus, parentheses.
        Precedence (high to low): ** > unary - > * / > + -
        ** is right-associative; all others left-associative.
        Returns float result.
        \"\"\"
        tokens = _tokenize(expr)
        pos = [0]
        result = _parse_expr(tokens, pos)
        if pos[0] != len(tokens):
            raise ValueError(f"Unexpected token at position {pos[0]}")
        return result


    def _tokenize(expr):
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i].isspace():
                i += 1
            elif expr[i].isdigit() or expr[i] == '.':
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                tokens.append(float(expr[i:j]))
                i = j
            elif expr[i] in '+-*/()':
                if expr[i] == '*' and i + 1 < len(expr) and expr[i + 1] == '*':
                    tokens.append('**')
                    i += 2
                else:
                    tokens.append(expr[i])
                    i += 1
            else:
                raise ValueError(f"Unexpected character: {expr[i]}")
        return tokens


    def _parse_expr(tokens, pos):
        left = _parse_term(tokens, pos)
        while pos[0] < len(tokens) and tokens[pos[0]] in ('+', '-'):
            op = tokens[pos[0]]
            pos[0] += 1
            right = _parse_term(tokens, pos)
            if op == '+':
                left = left + right
            else:
                left = left - right
        return left


    def _parse_term(tokens, pos):
        left = _parse_power(tokens, pos)
        while pos[0] < len(tokens) and tokens[pos[0]] in ('*', '/'):
            op = tokens[pos[0]]
            pos[0] += 1
            right = _parse_power(tokens, pos)
            if op == '*':
                left = left * right
            else:
                left = left / right
        return left


    def _parse_power(tokens, pos):
        # BUG: while-loop makes ** left-associative instead of right-associative.
        # 2**3**2 computes (2**3)**2=64 instead of 2**(3**2)=512.
        base = _parse_unary(tokens, pos)
        while pos[0] < len(tokens) and tokens[pos[0]] == '**':
            pos[0] += 1
            exp = _parse_unary(tokens, pos)
            base = base ** exp
        return base


    def _parse_unary(tokens, pos):
        if pos[0] < len(tokens) and tokens[pos[0]] == '-':
            pos[0] += 1
            return -_parse_unary(tokens, pos)
        return _parse_atom(tokens, pos)


    def _parse_atom(tokens, pos):
        if pos[0] >= len(tokens):
            raise ValueError("Unexpected end of expression")
        token = tokens[pos[0]]
        if isinstance(token, float):
            pos[0] += 1
            return token
        if token == '(':
            pos[0] += 1
            result = _parse_expr(tokens, pos)
            if pos[0] >= len(tokens) or tokens[pos[0]] != ')':
                raise ValueError("Missing closing parenthesis")
            pos[0] += 1
            return result
        raise ValueError(f"Unexpected token: {token}")
""")

UNSOLVABLE_TEST = textwrap.dedent("""\
    from buggy import evaluate


    def test_basic_addition():
        assert evaluate("2 + 3") == 5.0


    def test_precedence():
        assert evaluate("2 + 3 * 4") == 14.0


    def test_parentheses():
        assert evaluate("(2 + 3) * 4") == 20.0


    def test_single_power():
        assert evaluate("2 ** 3") == 8.0


    def test_power_right_associative():
        # 2 ** 3 ** 2 should be 2 ** (3 ** 2) = 2 ** 9 = 512
        assert evaluate("2 ** 3 ** 2") == 512.0


    def test_chained_power():
        # 2 ** 2 ** 3 should be 2 ** (2 ** 3) = 2 ** 8 = 256
        assert evaluate("2 ** 2 ** 3") == 256.0


    def test_unary_minus():
        assert evaluate("-3 + 5") == 2.0


    def test_complex_expression():
        assert evaluate("(1 + 2) * 3 ** 2") == 27.0
""")


def _server_reachable(base_url: str) -> bool:
    """Check whether the live OpenAI-compatible server is available."""
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{base_url}/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def _discover_first_model(base_url: str) -> str:
    """Return the first served model id from an OpenAI-compatible endpoint."""
    try:
        import json
        import urllib.request

        req = urllib.request.Request(
            f"{base_url}/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise pytest.UsageError(
            "--live-openai-model was not provided and model discovery failed "
            f"for {base_url}: {type(exc).__name__}: {exc}"
        ) from exc

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise pytest.UsageError(
            f"--live-openai-model was not provided and {base_url}/models returned no models"
        )

    first = data[0]
    model_id = first.get("id") if isinstance(first, dict) else None
    if not isinstance(model_id, str) or not model_id.strip():
        raise pytest.UsageError(
            f"--live-openai-model was not provided and {base_url}/models returned no usable id"
        )
    return model_id.strip()


@pytest.fixture(scope="session")
def live_openai_base_url(pytestconfig: pytest.Config) -> str:
    base_url = str(pytestconfig.getoption("--live-openai-base-url") or "").strip().rstrip("/")
    if not base_url:
        raise pytest.UsageError("--live-openai-base-url must be non-empty")
    return base_url


@pytest.fixture(scope="session")
def live_openai_model(pytestconfig: pytest.Config, live_openai_base_url: str) -> str:
    model = str(pytestconfig.getoption("--live-openai-model") or "").strip()
    return model or _discover_first_model(live_openai_base_url)


@pytest.fixture(scope="session")
def live_openai_compat_profile(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--live-openai-compat-profile") or "auto").strip()


@pytest.fixture(scope="session", autouse=True)
def _skip_if_live_server_unreachable(live_openai_base_url: str) -> None:
    if not _server_reachable(live_openai_base_url):
        pytest.skip(f"LLM server not reachable at {live_openai_base_url}")


def _write_case(case_dir: Path, *, buggy: str, test_code: str) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text(buggy, encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(test_code, encoding="utf-8")
    return case_dir


@pytest.fixture
def solvable_case(tmp_path: Path) -> Path:
    """Create a trivially solvable case directory."""
    return _write_case(tmp_path / "solvable", buggy=SOLVABLE_BUGGY, test_code=SOLVABLE_TEST)


@pytest.fixture
def unsolvable_case(tmp_path: Path) -> Path:
    """Create a hard case that a small local model is unlikely to solve."""
    return _write_case(tmp_path / "unsolvable", buggy=UNSOLVABLE_BUGGY, test_code=UNSOLVABLE_TEST)


def run_case_pytest(
    case_dir: Path,
    *,
    python_executable: str | None = None,
    timeout_sec: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run the case-local pytest command and return the completed process."""
    return subprocess.run(
        [python_executable or sys.executable, "-m", "pytest", "-x", "-q", "test_buggy.py"],
        cwd=case_dir,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


def assert_python_file_parses(path: Path) -> None:
    """Fail the test if *path* does not contain valid Python."""
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(f"{path} is not valid Python: {exc}")


def _assert_baseline_attempt_parses(result: BaselineResult, *, baseline_name: str) -> None:
    assert isinstance(result, BaselineResult), f"Expected BaselineResult, got {type(result)}"
    if result.repaired_code is None or not result.repaired_code.strip():
        return
    try:
        ast.parse(result.repaired_code)
    except SyntaxError:
        pytest.fail(
            f"{baseline_name}: produced syntactically invalid Python:\n{result.repaired_code[:500]}"
        )


def assert_baseline_solved_result(
    result: BaselineResult,
    *,
    baseline_name: str,
    original_buggy: str,
    test_code: str,
    tmp_path: Path,
) -> None:
    _assert_baseline_attempt_parses(result, baseline_name=baseline_name)
    assert result.repaired_code is not None, f"{baseline_name}: expected repaired_code but got None"
    assert result.repaired_code.strip(), f"{baseline_name}: repaired_code is empty"
    assert result.repaired_code.strip() != original_buggy.strip(), (
        f"{baseline_name}: repaired code is identical to buggy code"
    )

    work = tmp_path / "verify"
    work.mkdir(exist_ok=True)
    (work / "buggy.py").write_text(result.repaired_code, encoding="utf-8")
    (work / "test_buggy.py").write_text(test_code, encoding="utf-8")
    rerun = run_case_pytest(work, timeout_sec=30)
    assert rerun.returncode == 0, (
        f"{baseline_name}: repaired code does not pass tests.\n"
        f"Repaired code:\n{result.repaired_code[:500]}"
    )


def assert_baseline_classified_failure(
    result: BaselineResult,
    *,
    baseline_name: str,
    original_buggy: str = "",
) -> None:
    _assert_baseline_attempt_parses(result, baseline_name=baseline_name)
    if result.repaired_code is not None and original_buggy:
        assert result.repaired_code.strip() != original_buggy.strip(), (
            f"{baseline_name}: surfaced unchanged buggy code as a failed repair"
        )
    meta = result.meta if isinstance(result.meta, dict) else {}
    metrics = meta.get("metrics") if isinstance(meta.get("metrics"), dict) else {}
    has_classification = any(
        (
            str(meta.get("error") or "").strip(),
            str(meta.get("stopped_reason") or "").strip(),
            str(meta.get("compat_message") or "").strip(),
            str(metrics.get("last_llm_failure_kind") or "").strip(),
            str(metrics.get("stopped_reason") or "").strip(),
        )
    )
    assert has_classification, f"{baseline_name}: failure was not explicitly classified"


def assert_baseline_unsolved_result(
    result: BaselineResult,
    *,
    baseline_name: str,
    original_buggy: str = "",
) -> None:
    _assert_baseline_attempt_parses(result, baseline_name=baseline_name)
    if result.success:
        return
    if result.repaired_code is None or not result.repaired_code.strip():
        assert_baseline_classified_failure(
            result,
            baseline_name=baseline_name,
            original_buggy=original_buggy,
        )
        return
    if original_buggy:
        assert result.repaired_code.strip() != original_buggy.strip(), (
            f"{baseline_name}: surfaced unchanged buggy code as a failed repair"
        )
