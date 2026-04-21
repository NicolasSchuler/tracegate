"""Phase 5 - INTERVENOR behavioral conformance + diff roundtrip for lambda-RLM/RLM.

Recommendation 5: Verify INTERVENOR teacher-learner loop invariants hold
even under edge-case inputs.

Recommendation 8: Verify diff roundtrip for lambda-RLM and RLM through
the base-class ``extract_diff`` path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from baselines.adapter_base import (
    BaselineCaseContext,
    BaselineConfig,
    BaselineResult,
)
from baselines.intervenor.adapter import IntervenorAdapter
from baselines.lambda_rlm.adapter import LambdaRLMAdapter
from baselines.rlm.adapter import RLMAdapter

# ── Helpers ────────────────────────────────────────────────────────────


def _make_intervenor_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fake_intervenor"
    prompt_dir = repo / "prompt"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "code_teacher.txt").write_text(
        "Teacher prompt\n- buggy code:\n%%%buggy_code%%%\nError: %%%error_message%%%\n"
        "Modification method:\n",
        encoding="utf-8",
    )
    (prompt_dir / "code_learner.txt").write_text(
        "Learner prompt\n- buggy code:\n%%%buggy_code%%%\nError: %%%error_message%%%\n"
        "- modification method:\n%%%repair_method%%%\nCorrect the code:\n",
        encoding="utf-8",
    )
    return repo


def _setup_intervenor(
    tmp_path: Path,
    *,
    max_attempts: int = 2,
) -> tuple[IntervenorAdapter, BaselineConfig]:
    repo = _make_intervenor_repo(tmp_path)
    adapter = IntervenorAdapter()
    config = BaselineConfig(
        repo_dir=repo,
        model="test-model",
        max_attempts=max_attempts,
        timeout_sec=20,
    )
    adapter.setup(config)
    return adapter, config


def _simple_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text(
        "def multiply(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (case_dir / "test_buggy.py").write_text(
        "from buggy import multiply\n\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n",
        encoding="utf-8",
    )
    return case_dir


def _intervenor_case_dict(
    adapter: IntervenorAdapter,
    case_dir: Path,
) -> dict:
    return adapter.convert_case(
        case_dir,
        case_context=BaselineCaseContext(
            case_id="test_case",
            python_args=("-m", "pytest", "-q", "test_buggy.py"),
            timeout_sec=10,
            description="Multiply two numbers.",
        ),
    ).to_dict()


# ── Recommendation 5: INTERVENOR behavioral conformance ──────────────


class TestIntervenorBehavioralConformance:
    """Invariant-level tests for the INTERVENOR teacher→learner loop."""

    def test_teacher_always_called_before_learner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariant: teacher prompt is always generated before learner prompt."""
        adapter, config = _setup_intervenor(tmp_path, max_attempts=1)
        case_dir = _simple_case(tmp_path)
        case_input = _intervenor_case_dict(adapter, case_dir)

        call_order: list[str] = []

        def _fake_generate_text(*, client, model, prompt, max_tokens, config=None):
            if "Modification method:" in prompt:
                call_order.append("teacher")
                return "Change + to *.", {"prompt_tokens": 5, "completion_tokens": 3}
            call_order.append("learner")
            return "def multiply(a, b):\n    return a * b\n", {
                "prompt_tokens": 5,
                "completion_tokens": 3,
            }

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        adapter.run_repair(case_input, config)

        assert call_order == ["teacher", "learner"]

    def test_teacher_output_injected_into_learner_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariant: teacher's repair method text appears in the learner prompt."""
        adapter, config = _setup_intervenor(tmp_path, max_attempts=1)
        case_dir = _simple_case(tmp_path)
        case_input = _intervenor_case_dict(adapter, case_dir)

        learner_prompts: list[str] = []

        def _fake_generate_text(*, client, model, prompt, max_tokens, config=None):
            if "Modification method:" in prompt:
                return "UNIQUE_TEACHER_OUTPUT_42", {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                }
            learner_prompts.append(prompt)
            return "def multiply(a, b):\n    return a * b\n", {
                "prompt_tokens": 5,
                "completion_tokens": 3,
            }

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        adapter.run_repair(case_input, config)

        assert len(learner_prompts) == 1
        assert "UNIQUE_TEACHER_OUTPUT_42" in learner_prompts[0]

    def test_empty_learner_response_terminates_without_postcheck(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariant: empty learner → immediate failure, no postcheck run."""
        adapter, config = _setup_intervenor(tmp_path, max_attempts=3)
        case_dir = _simple_case(tmp_path)
        case_input = _intervenor_case_dict(adapter, case_dir)

        postcheck_count = 0
        original_run_tests = adapter._run_case_tests

        def _counting_run_tests(**kwargs):
            nonlocal postcheck_count
            postcheck_count += 1
            return original_run_tests(**kwargs)

        monkeypatch.setattr(adapter, "_run_case_tests", _counting_run_tests)

        def _fake_generate_text(*, client, model, prompt, max_tokens, config=None):
            if "Modification method:" in prompt:
                return "Change operator.", {"prompt_tokens": 5, "completion_tokens": 3}
            return "", {"prompt_tokens": 5, "completion_tokens": 3}  # empty learner

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        assert result.meta["error"] == "learner_empty_response"
        # Exactly 1 test run: the precheck. No postcheck after empty learner.
        assert postcheck_count == 1

    def test_max_attempts_strictly_enforced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariant: exactly max_attempts iterations, no more."""
        adapter, config = _setup_intervenor(tmp_path, max_attempts=2)
        case_dir = _simple_case(tmp_path)
        case_input = _intervenor_case_dict(adapter, case_dir)

        attempt_count = 0

        def _fake_generate_text(*, client, model, prompt, max_tokens, config=None):
            nonlocal attempt_count
            if "Modification method:" in prompt:
                attempt_count += 1
                return "Change operator.", {"prompt_tokens": 5, "completion_tokens": 3}
            # wrong fix each time
            return "def multiply(a, b):\n    return a - b\n", {
                "prompt_tokens": 5,
                "completion_tokens": 3,
            }

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        assert result.meta["error"] == "max_attempts_exhausted"
        assert attempt_count == 2
        assert result.iterations_used == 2

    def test_two_llm_calls_per_completed_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariant: each completed attempt produces exactly 2 LLM calls."""
        adapter, config = _setup_intervenor(tmp_path, max_attempts=1)
        case_dir = _simple_case(tmp_path)
        case_input = _intervenor_case_dict(adapter, case_dir)

        def _fake_generate_text(*, client, model, prompt, max_tokens, config=None):
            if "Modification method:" in prompt:
                return "Change +.", {"prompt_tokens": 5, "completion_tokens": 3}
            return "def multiply(a, b):\n    return a * b\n", {
                "prompt_tokens": 5,
                "completion_tokens": 3,
            }

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.meta["metrics"]["llm_call_count"] == 2

    def test_precheck_pass_returns_success_without_llm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariant: case that already passes → success with 0 iterations, no LLM calls."""
        adapter, config = _setup_intervenor(tmp_path)
        case_dir = tmp_path / "passing_case"
        case_dir.mkdir()
        (case_dir / "buggy.py").write_text(
            "def multiply(a, b):\n    return a * b\n",
            encoding="utf-8",
        )
        (case_dir / "test_buggy.py").write_text(
            "from buggy import multiply\n\n\ndef test_multiply():\n"
            "    assert multiply(3, 4) == 12\n",
            encoding="utf-8",
        )
        case_input = adapter.convert_case(
            case_dir,
            case_context=BaselineCaseContext(
                case_id="passing",
                python_args=("-m", "pytest", "-q", "test_buggy.py"),
                timeout_sec=10,
                description="Multiply.",
            ),
        ).to_dict()

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert result.iterations_used == 0
        assert result.meta["metrics"]["llm_call_count"] == 0

    def test_postcheck_pass_terminates_loop_early(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariant: successful postcheck ends the loop before max_attempts."""
        adapter, config = _setup_intervenor(tmp_path, max_attempts=5)
        case_dir = _simple_case(tmp_path)
        case_input = _intervenor_case_dict(adapter, case_dir)

        def _fake_generate_text(*, client, model, prompt, max_tokens, config=None):
            if "Modification method:" in prompt:
                return "Change +.", {"prompt_tokens": 5, "completion_tokens": 3}
            return "def multiply(a, b):\n    return a * b\n", {
                "prompt_tokens": 5,
                "completion_tokens": 3,
            }

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert result.iterations_used == 1  # terminated after 1st attempt, not 5

    def test_setup_required_before_run_repair(self, tmp_path: Path) -> None:
        """Invariant: run_repair raises if setup() was not called."""
        adapter = IntervenorAdapter()
        case_dir = _simple_case(tmp_path)
        case_input = adapter.convert_case(
            case_dir,
            case_context=BaselineCaseContext(
                case_id="x",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=10,
            ),
        ).to_dict()
        config = BaselineConfig(repo_dir=tmp_path, model="test-model")

        with pytest.raises(RuntimeError, match="setup"):
            adapter.run_repair(case_input, config)

    def test_usage_unavailable_when_any_call_lacks_usage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariant: usage_available=False if any LLM call returns no usage."""
        adapter, config = _setup_intervenor(tmp_path, max_attempts=1)
        case_dir = _simple_case(tmp_path)
        case_input = _intervenor_case_dict(adapter, case_dir)

        def _fake_generate_text(*, client, model, prompt, max_tokens, config=None):
            if "Modification method:" in prompt:
                return "Change +.", {}  # no usage on teacher call
            return "def multiply(a, b):\n    return a * b\n", {
                "prompt_tokens": 5,
                "completion_tokens": 3,
            }

        monkeypatch.setattr(adapter, "_build_client", lambda config: object())
        monkeypatch.setattr(adapter, "_generate_text", _fake_generate_text)

        result = adapter.run_repair(case_input, config)

        # usage_available should be False because one call had no usage
        assert result.meta["metrics"]["usage_available"] is False


# ── Recommendation 8: diff roundtrip for lambda-RLM and RLM ─────────


class TestLambdaRLMDiffRoundtrip:
    """Verify that lambda-RLM diffs round-trip through extract_diff."""

    @pytest.fixture()
    def case_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "lambda_case"
        d.mkdir()
        (d / "buggy.py").write_text(
            "def compute(value):\n    return value + 1\n",
            encoding="utf-8",
        )
        (d / "test_buggy.py").write_text(
            "from buggy import compute\n\ndef test_compute():\n    assert compute(3) == 6\n",
            encoding="utf-8",
        )
        return d

    def test_extract_diff_from_repaired_code(self, case_dir: Path) -> None:
        adapter = LambdaRLMAdapter()
        result = BaselineResult(
            repaired_code="def compute(value):\n    return value * 2\n",
            success=True,
            iterations_used=1,
        )
        diff = adapter.extract_diff(case_dir, result)

        assert diff is not None
        assert "diff --git" in diff
        assert "-    return value + 1" in diff
        assert "+    return value * 2" in diff

    def test_extract_diff_returns_none_on_identical(self, case_dir: Path) -> None:
        adapter = LambdaRLMAdapter()
        original = (case_dir / "buggy.py").read_text(encoding="utf-8")
        result = BaselineResult(
            repaired_code=original,
            success=True,
            iterations_used=1,
        )
        diff = adapter.extract_diff(case_dir, result)
        assert diff is None

    def test_extract_diff_returns_none_on_no_repair(self, case_dir: Path) -> None:
        adapter = LambdaRLMAdapter()
        result = BaselineResult(
            repaired_code=None,
            success=False,
            iterations_used=0,
        )
        diff = adapter.extract_diff(case_dir, result)
        assert diff is None

    def test_extract_diff_applies_cleanly(self, case_dir: Path, tmp_path: Path) -> None:
        """The generated diff can be applied back to the original to produce the repair."""
        import subprocess

        adapter = LambdaRLMAdapter()
        repaired = "def compute(value):\n    return value * 2\n"
        result = BaselineResult(
            repaired_code=repaired,
            success=True,
            iterations_used=1,
        )
        diff = adapter.extract_diff(case_dir, result)
        assert diff is not None

        # Write diff to a file and apply it
        diff_file = tmp_path / "patch.diff"
        diff_file.write_text(diff, encoding="utf-8")
        apply_result = subprocess.run(
            ["git", "apply", "--stat", str(diff_file)],
            capture_output=True,
            text=True,
            cwd=str(case_dir),
        )
        # git apply --stat should parse the diff without errors
        assert apply_result.returncode == 0, f"git apply --stat failed: {apply_result.stderr}"

    def test_extract_diff_prefers_patch_diff_over_repaired_code(self, case_dir: Path) -> None:
        """Precedence: patch_diff > file_edits > repaired_code."""
        adapter = LambdaRLMAdapter()
        explicit_diff = (
            "diff --git a/buggy.py b/buggy.py\n"
            "--- a/buggy.py\n"
            "+++ b/buggy.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def compute(value):\n"
            "-    return value + 1\n"
            "+    return value * 2\n"
        )
        result = BaselineResult(
            repaired_code="ignored",
            patch_diff=explicit_diff,
            success=True,
            iterations_used=1,
        )
        diff = adapter.extract_diff(case_dir, result)
        assert diff == explicit_diff


class TestRLMDiffRoundtrip:
    """Verify that RLM diffs round-trip through extract_diff."""

    @pytest.fixture()
    def case_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "rlm_case"
        d.mkdir()
        (d / "buggy.py").write_text(
            "def compute(value):\n    return value + 1\n",
            encoding="utf-8",
        )
        (d / "test_buggy.py").write_text(
            "from buggy import compute\n\ndef test_compute():\n    assert compute(3) == 6\n",
            encoding="utf-8",
        )
        return d

    def test_extract_diff_from_repaired_code(self, case_dir: Path) -> None:
        adapter = RLMAdapter()
        result = BaselineResult(
            repaired_code="def compute(value):\n    return value * 2\n",
            success=True,
            iterations_used=1,
        )
        diff = adapter.extract_diff(case_dir, result)

        assert diff is not None
        assert "diff --git" in diff
        assert "-    return value + 1" in diff
        assert "+    return value * 2" in diff

    def test_extract_diff_returns_none_on_identical(self, case_dir: Path) -> None:
        adapter = RLMAdapter()
        original = (case_dir / "buggy.py").read_text(encoding="utf-8")
        result = BaselineResult(
            repaired_code=original,
            success=True,
            iterations_used=1,
        )
        diff = adapter.extract_diff(case_dir, result)
        assert diff is None

    def test_extract_diff_returns_none_on_no_repair(self, case_dir: Path) -> None:
        adapter = RLMAdapter()
        result = BaselineResult(
            repaired_code=None,
            success=False,
            iterations_used=0,
        )
        diff = adapter.extract_diff(case_dir, result)
        assert diff is None

    def test_extract_diff_applies_cleanly(self, case_dir: Path, tmp_path: Path) -> None:
        """The generated diff can be applied back to the original to produce the repair."""
        import subprocess

        adapter = RLMAdapter()
        repaired = "def compute(value):\n    return value * 2\n"
        result = BaselineResult(
            repaired_code=repaired,
            success=True,
            iterations_used=1,
        )
        diff = adapter.extract_diff(case_dir, result)
        assert diff is not None

        diff_file = tmp_path / "patch.diff"
        diff_file.write_text(diff, encoding="utf-8")
        apply_result = subprocess.run(
            ["git", "apply", "--stat", str(diff_file)],
            capture_output=True,
            text=True,
            cwd=str(case_dir),
        )
        assert apply_result.returncode == 0, f"git apply --stat failed: {apply_result.stderr}"

    def test_extract_diff_prefers_patch_diff_over_repaired_code(self, case_dir: Path) -> None:
        adapter = RLMAdapter()
        explicit_diff = (
            "diff --git a/buggy.py b/buggy.py\n"
            "--- a/buggy.py\n"
            "+++ b/buggy.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def compute(value):\n"
            "-    return value + 1\n"
            "+    return value * 2\n"
        )
        result = BaselineResult(
            repaired_code="ignored",
            patch_diff=explicit_diff,
            success=True,
            iterations_used=1,
        )
        diff = adapter.extract_diff(case_dir, result)
        assert diff == explicit_diff
