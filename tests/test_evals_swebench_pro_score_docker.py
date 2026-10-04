"""Tests for SWE-bench Pro Docker-based scoring logic.

All tests run WITHOUT requiring an actual Docker daemon.  The ``docker`` SDK
is mocked entirely so that unit tests exercise entry-script generation,
output parsing, pass/fail computation, and error handling in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import evals.swebench_pro.score_docker as score_docker_mod
from evals.swebench.types import PredictionRecord
from evals.swebench_pro.score_docker import (
    _EvalContext,
    _parse_selected_test_files,
    _parse_test_list,
    _score_output,
    build_entry_script,
    evaluate_instance_docker,
    get_dockerhub_image_uri,
    run_docker_scoring,
    strip_binary_hunks,
)
from evals.swebench_pro.types import ScoreInstanceResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SENTINEL: object = object()


def _make_ctx(
    *,
    instance_id: str = "repo__issue__42",
    patch: str = "diff --git a/a.py b/a.py\n",
    dockerhub_tag: str = "abc123",
    base_commit: str = "deadbeef",
    before_repo_set_cmd: str = "",
    selected_test_files: str = "tests/test_foo.py",
    fail_to_pass: set[str] | object = _SENTINEL,
    pass_to_pass: set[str] | object = _SENTINEL,
) -> _EvalContext:
    return _EvalContext(
        instance_id=instance_id,
        patch=patch,
        dockerhub_tag=dockerhub_tag,
        base_commit=base_commit,
        before_repo_set_cmd=before_repo_set_cmd,
        selected_test_files=selected_test_files,
        fail_to_pass={"test_a"} if fail_to_pass is _SENTINEL else fail_to_pass,  # type: ignore[arg-type]
        pass_to_pass={"test_b"} if pass_to_pass is _SENTINEL else pass_to_pass,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# get_dockerhub_image_uri
# ---------------------------------------------------------------------------


class TestGetDockerhubImageUri:
    def test_basic_uri(self) -> None:
        uri = get_dockerhub_image_uri(
            instance_id="repo__issue__1",
            dockerhub_username="jefzda",
            dockerhub_tag="abc123",
        )
        assert uri == "jefzda/sweap-images:abc123"

    def test_custom_username(self) -> None:
        uri = get_dockerhub_image_uri(
            instance_id="x",
            dockerhub_username="myuser",
            dockerhub_tag="tag999",
        )
        assert uri == "myuser/sweap-images:tag999"


# ---------------------------------------------------------------------------
# strip_binary_hunks
# ---------------------------------------------------------------------------


class TestStripBinaryHunks:
    def test_empty_patch(self) -> None:
        assert strip_binary_hunks("") == ""

    def test_none_like_patch(self) -> None:
        assert strip_binary_hunks("") == ""

    def test_no_binary_sections(self) -> None:
        patch = (
            "diff --git a/file.py b/file.py\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        assert strip_binary_hunks(patch) == patch

    def test_removes_binary_files_differ(self) -> None:
        patch = (
            "diff --git a/file.py b/file.py\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "diff --git a/image.png b/image.png\n"
            "Binary files a/image.png and b/image.png differ\n"
        )
        result = strip_binary_hunks(patch)
        assert "image.png" not in result
        assert "file.py" in result

    def test_removes_git_binary_patch(self) -> None:
        patch = (
            "diff --git a/data.bin b/data.bin\n"
            "GIT binary patch\n"
            "literal 1234\n"
            "zcmV;>1T2&iHg1+N\n"
            "\n"
            "diff --git a/code.py b/code.py\n"
            "--- a/code.py\n"
            "+++ b/code.py\n"
            "@@ -1 +1 @@\n"
            "-x\n"
            "+y\n"
        )
        result = strip_binary_hunks(patch)
        assert "data.bin" not in result
        assert "code.py" in result


# ---------------------------------------------------------------------------
# build_entry_script
# ---------------------------------------------------------------------------


class TestBuildEntryScript:
    def test_basic_script_structure(self) -> None:
        script = build_entry_script(
            instance_id="repo__1",
            base_commit="abc123",
            before_repo_set_cmd="pip install -e .",
            selected_test_files="tests/test_a.py",
            env_commands=["export FOO=bar"],
        )
        assert script.startswith("#!/bin/bash\n")
        assert "set -e" in script
        assert "export FOO=bar" in script
        assert "git reset --hard abc123" in script
        assert "git checkout abc123" in script
        assert "if [[ -s /workspace/patch.diff ]]; then" in script
        # DAT-19: the official apply sequence with fallbacks.
        assert "git apply --verbose /workspace/patch.diff" in script
        assert "git apply --verbose --reject /workspace/patch.diff" in script
        assert "patch --batch --fuzz=5 -p1 -i /workspace/patch.diff" in script
        assert "exit 3" in script
        assert 'echo "Patch is empty after cleanup; evaluating base commit."' in script
        assert "pip install -e ." in script
        assert "bash /workspace/run_script.sh tests/test_a.py" in script
        assert "echo $? > /workspace/run_status" in script
        assert "python /workspace/parser.py" in script

    def test_empty_env_commands(self) -> None:
        script = build_entry_script(
            instance_id="repo__2",
            base_commit="def456",
            before_repo_set_cmd="",
            selected_test_files="tests/t.py",
            env_commands=[],
        )
        assert "export" not in script
        assert "git reset --hard def456" in script

    def test_multiline_before_repo_set_cmd_uses_last_line(self) -> None:
        script = build_entry_script(
            instance_id="repo__3",
            base_commit="aaa",
            before_repo_set_cmd="echo first\necho second\npip install -e .",
            selected_test_files="t.py",
            env_commands=[],
        )
        # Only the last line should appear (matching Pro eval behavior)
        assert "pip install -e ." in script
        assert "echo first" not in script
        assert "echo second" not in script

    def test_multiple_env_commands(self) -> None:
        script = build_entry_script(
            instance_id="repo__4",
            base_commit="bbb",
            before_repo_set_cmd="",
            selected_test_files="t.py",
            env_commands=["export A=1", "export B=2", "export C=3"],
        )
        assert "export A=1" in script
        assert "export B=2" in script
        assert "export C=3" in script


# ---------------------------------------------------------------------------
# _parse_test_list
# ---------------------------------------------------------------------------


class TestParseTestList:
    def test_empty_string(self) -> None:
        assert _parse_test_list("") == set()

    def test_whitespace_only(self) -> None:
        assert _parse_test_list("   ") == set()

    def test_python_list_literal(self) -> None:
        assert _parse_test_list('["test_a", "test_b"]') == {"test_a", "test_b"}

    def test_python_tuple_literal(self) -> None:
        assert _parse_test_list('("test_x",)') == {"test_x"}

    def test_json_list(self) -> None:
        assert _parse_test_list('["one", "two"]') == {"one", "two"}

    def test_plain_text_fallback(self) -> None:
        assert _parse_test_list("test_foo") == {"test_foo"}

    def test_single_item_list(self) -> None:
        assert _parse_test_list('["only"]') == {"only"}


# ---------------------------------------------------------------------------
# _parse_selected_test_files
# ---------------------------------------------------------------------------


class TestParseSelectedTestFiles:
    def test_empty(self) -> None:
        assert _parse_selected_test_files("") == ""

    def test_whitespace(self) -> None:
        assert _parse_selected_test_files("   ") == ""

    def test_python_list(self) -> None:
        result = _parse_selected_test_files('["a.py", "b.py"]')
        assert result == "a.py,b.py"

    def test_plain_text(self) -> None:
        assert _parse_selected_test_files("test_file.py") == "test_file.py"


# ---------------------------------------------------------------------------
# _score_output
# ---------------------------------------------------------------------------


class TestScoreOutput:
    def test_all_tests_pass(self) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass={"test_b"})
        output: dict[str, Any] = {
            "tests": [
                {"name": "test_a", "status": "PASSED"},
                {"name": "test_b", "status": "PASSED"},
            ]
        }
        result = _score_output(ctx, output)
        assert result.passed is True
        assert result.fail_to_pass_met is True
        assert result.pass_to_pass_met is True
        assert result.error is None

    def test_fail_to_pass_not_met(self) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass={"test_b"})
        output: dict[str, Any] = {
            "tests": [
                {"name": "test_a", "status": "FAILED"},
                {"name": "test_b", "status": "PASSED"},
            ]
        }
        result = _score_output(ctx, output)
        assert result.passed is False
        assert result.fail_to_pass_met is False
        assert result.pass_to_pass_met is True

    def test_pass_to_pass_not_met(self) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass={"test_b"})
        output: dict[str, Any] = {
            "tests": [
                {"name": "test_a", "status": "PASSED"},
                {"name": "test_b", "status": "FAILED"},
            ]
        }
        result = _score_output(ctx, output)
        assert result.passed is False
        assert result.fail_to_pass_met is True
        assert result.pass_to_pass_met is False

    def test_both_criteria_fail(self) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass={"test_b"})
        output: dict[str, Any] = {
            "tests": [
                {"name": "test_a", "status": "FAILED"},
                {"name": "test_b", "status": "FAILED"},
            ]
        }
        result = _score_output(ctx, output)
        assert result.passed is False
        assert result.fail_to_pass_met is False
        assert result.pass_to_pass_met is False

    def test_empty_tests_list(self) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        output: dict[str, Any] = {"tests": []}
        result = _score_output(ctx, output)
        assert result.passed is False
        assert result.fail_to_pass_met is False
        # Empty pass_to_pass is trivially met
        assert result.pass_to_pass_met is True

    def test_empty_expectations_trivially_pass(self) -> None:
        ctx = _make_ctx(fail_to_pass=set(), pass_to_pass=set())
        output: dict[str, Any] = {"tests": []}
        result = _score_output(ctx, output)
        assert result.passed is True

    def test_extra_tests_ignored(self) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        output: dict[str, Any] = {
            "tests": [
                {"name": "test_a", "status": "PASSED"},
                {"name": "test_extra", "status": "FAILED"},
            ]
        }
        result = _score_output(ctx, output)
        assert result.passed is True
        assert result.fail_to_pass_met is True

    def test_non_dict_tests_are_skipped(self) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        output: dict[str, Any] = {
            "tests": [
                "not a dict",
                {"name": "test_a", "status": "PASSED"},
            ]
        }
        result = _score_output(ctx, output)
        assert result.passed is True
        assert result.fail_to_pass_met is True


# ---------------------------------------------------------------------------
# evaluate_instance_docker - error paths (no real Docker)
# ---------------------------------------------------------------------------


class TestEvaluateInstanceDockerErrors:
    def test_missing_docker_sdk(self, tmp_path: Path) -> None:
        """When docker SDK is not importable, returns error result."""
        ctx = _make_ctx()
        # Simulate ImportError by patching builtins.__import__
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def _block_docker(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "docker":
                raise ImportError("No module named 'docker'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_docker):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=tmp_path / "scripts",
                dockerfiles_dir=tmp_path / "dockerfiles",
                output_dir=tmp_path / "output",
                prefix="baseline",
            )
        assert result.passed is False
        assert result.error is not None
        assert "docker" in result.error.lower()
        assert result.infrastructure_error is True

    def test_missing_run_scripts(self, tmp_path: Path) -> None:
        """When run_script.sh is missing, returns error result."""
        ctx = _make_ctx()
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        # Instance directory exists but no run_script.sh
        (scripts_dir / ctx.instance_id).mkdir()

        mock_docker = MagicMock()
        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=scripts_dir,
                dockerfiles_dir=None,
                output_dir=tmp_path / "output",
                prefix="baseline",
            )
        assert result.passed is False
        assert result.error is not None
        assert "Missing run scripts" in result.error
        assert result.infrastructure_error is True

    def test_cached_output_is_reused(self, tmp_path: Path) -> None:
        """When output.json already exists and redo=False, reuse cached result."""
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        uid_dir = tmp_path / "output" / ctx.instance_id
        uid_dir.mkdir(parents=True)
        cached = {
            "tests": [{"name": "test_a", "status": "PASSED"}],
        }
        import json

        (uid_dir / "baseline_output.json").write_text(json.dumps(cached), encoding="utf-8")
        # DAT-13: the cache is only reused for the identical (cleaned) patch.
        import hashlib

        digest = hashlib.sha256(strip_binary_hunks(ctx.patch).encode("utf-8")).hexdigest()
        (uid_dir / "baseline_output.patch_sha256").write_text(digest, encoding="utf-8")

        mock_docker = MagicMock()
        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=tmp_path / "scripts",
                dockerfiles_dir=None,
                output_dir=tmp_path / "output",
                prefix="baseline",
                redo=False,
            )
        assert result.passed is True
        assert result.fail_to_pass_met is True

    def test_cached_output_for_a_different_patch_is_not_reused(self, tmp_path: Path) -> None:
        """DAT-13: a regenerated prediction must not reuse the old outcome."""
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        uid_dir = tmp_path / "output" / ctx.instance_id
        uid_dir.mkdir(parents=True)
        import json

        cached = {"tests": [{"name": "test_a", "status": "PASSED"}]}
        (uid_dir / "baseline_output.json").write_text(json.dumps(cached), encoding="utf-8")
        (uid_dir / "baseline_output.patch_sha256").write_text("0" * 64, encoding="utf-8")

        mock_docker = MagicMock()
        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=tmp_path / "scripts",
                dockerfiles_dir=None,
                output_dir=tmp_path / "output",
                prefix="baseline",
                redo=False,
            )
        assert result.passed is False
        assert "Missing run scripts" in (result.error or "")

    def test_redo_ignores_cached_output(self, tmp_path: Path) -> None:
        """When redo=True, cached output.json is ignored and scripts are required."""
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        uid_dir = tmp_path / "output" / ctx.instance_id
        uid_dir.mkdir(parents=True)
        import json

        cached = {"tests": [{"name": "test_a", "status": "PASSED"}]}
        (uid_dir / "baseline_output.json").write_text(json.dumps(cached), encoding="utf-8")

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        # No instance scripts → error
        (scripts_dir / ctx.instance_id).mkdir()

        mock_docker = MagicMock()
        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=scripts_dir,
                dockerfiles_dir=None,
                output_dir=tmp_path / "output",
                prefix="baseline",
                redo=True,
            )
        assert result.passed is False
        assert "Missing run scripts" in (result.error or "")

    def test_container_nonzero_exit(self, tmp_path: Path) -> None:
        """Container exiting with non-zero code yields error result."""
        ctx = _make_ctx()
        scripts_dir = tmp_path / "scripts"
        instance_dir = scripts_dir / ctx.instance_id
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        (instance_dir / "parser.py").write_text("import sys\n", encoding="utf-8")

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 1}

        mock_client = MagicMock()
        mock_client.images.pull.return_value = None
        mock_client.containers.run.return_value = mock_container

        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=scripts_dir,
                dockerfiles_dir=None,
                output_dir=tmp_path / "output",
                prefix="baseline",
            )
        assert result.passed is False
        assert result.error is not None
        assert "exited with code 1" in result.error
        # DAT-09: an unexplained container failure is infrastructure, not a wrong patch.
        assert result.infrastructure_error is True

    def test_output_json_missing_after_success(self, tmp_path: Path) -> None:
        """Container exits 0 but output.json missing → error result."""
        ctx = _make_ctx()
        scripts_dir = tmp_path / "scripts"
        instance_dir = scripts_dir / ctx.instance_id
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (instance_dir / "parser.py").write_text("import sys\n", encoding="utf-8")

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}

        mock_client = MagicMock()
        mock_client.images.pull.return_value = None
        mock_client.containers.run.return_value = mock_container

        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=scripts_dir,
                dockerfiles_dir=None,
                output_dir=tmp_path / "output",
                prefix="baseline",
            )
        assert result.passed is False
        assert result.error is not None
        assert "output.json not found" in result.error
        assert result.infrastructure_error is True  # DAT-09

    def test_empty_patch_is_scored_at_base_commit(self, tmp_path: Path) -> None:
        """Empty prediction diffs should not be treated as container failures."""
        ctx = _make_ctx(patch="", fail_to_pass={"test_a"}, pass_to_pass=set())
        scripts_dir = tmp_path / "scripts"
        instance_dir = scripts_dir / ctx.instance_id
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (instance_dir / "parser.py").write_text("import sys\n", encoding="utf-8")

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}

        mock_client = MagicMock()
        mock_client.images.pull.return_value = None
        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client

        output_dir = tmp_path / "output"

        def _fake_run(_image: str, **_kwargs: Any) -> MagicMock:
            workspace_dir = output_dir / ctx.instance_id / "workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            import json

            (workspace_dir / "output.json").write_text(
                json.dumps({"tests": [{"name": "test_a", "status": "PASSED"}]}),
                encoding="utf-8",
            )
            return mock_container

        mock_client.containers.run.side_effect = _fake_run

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=scripts_dir,
                dockerfiles_dir=None,
                output_dir=output_dir,
                prefix="baseline",
            )

        entry_script = (output_dir / ctx.instance_id / f"{'baseline'}_entryscript.sh").read_text(
            encoding="utf-8"
        )
        patch_text = (output_dir / ctx.instance_id / "workspace" / "patch.diff").read_text(
            encoding="utf-8"
        )

        assert patch_text == ""
        assert "if [[ -s /workspace/patch.diff ]]; then" in entry_script
        assert result.passed is True
        assert result.error is None

    def test_image_pull_failure_falls_back_to_local(self, tmp_path: Path) -> None:
        """When pull fails but image exists locally, proceed with evaluation."""
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        scripts_dir = tmp_path / "scripts"
        instance_dir = scripts_dir / ctx.instance_id
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (instance_dir / "parser.py").write_text("import sys\n", encoding="utf-8")

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}

        mock_client = MagicMock()
        mock_client.images.pull.side_effect = Exception("network error")
        mock_client.images.get.return_value = MagicMock()  # image exists locally
        mock_client.containers.run.return_value = mock_container

        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client

        # We need output.json to exist after container runs
        output_dir = tmp_path / "output"

        def _fake_run(image, **kwargs):
            # Write output.json into the workspace
            workspace_dir = output_dir / ctx.instance_id / "workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            import json

            (workspace_dir / "output.json").write_text(
                json.dumps({"tests": [{"name": "test_a", "status": "PASSED"}]}), encoding="utf-8"
            )
            return mock_container

        mock_client.containers.run.side_effect = _fake_run

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=scripts_dir,
                dockerfiles_dir=None,
                output_dir=output_dir,
                prefix="baseline",
            )
        # Empty expectations → passes trivially
        assert result.passed is True

    def test_image_pull_and_get_both_fail(self, tmp_path: Path) -> None:
        """When both pull and get fail, returns image pull error."""
        ctx = _make_ctx()
        scripts_dir = tmp_path / "scripts"
        instance_dir = scripts_dir / ctx.instance_id
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (instance_dir / "parser.py").write_text("import sys\n", encoding="utf-8")

        mock_client = MagicMock()
        mock_client.images.pull.side_effect = Exception("network error")
        mock_client.images.get.side_effect = Exception("image not found locally")

        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=scripts_dir,
                dockerfiles_dir=None,
                output_dir=tmp_path / "output",
                prefix="baseline",
            )
        assert result.passed is False
        assert result.error is not None
        assert "Failed to pull image" in result.error
        assert result.infrastructure_error is True

    def test_container_cleanup_on_wait_timeout(self, tmp_path: Path) -> None:
        """Timeouts after launch stop/remove the container and close the client."""
        ctx = _make_ctx()
        scripts_dir = tmp_path / "scripts"
        instance_dir = scripts_dir / ctx.instance_id
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (instance_dir / "parser.py").write_text("import sys\n", encoding="utf-8")

        mock_container = MagicMock()
        mock_container.wait.side_effect = TimeoutError("timed out waiting for container")

        mock_client = MagicMock()
        mock_client.images.pull.return_value = None
        mock_client.containers.run.return_value = mock_container

        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=scripts_dir,
                dockerfiles_dir=None,
                output_dir=tmp_path / "output",
                prefix="baseline",
            )

        assert result.passed is False
        assert result.error is not None
        assert "timed out waiting for container" in result.error
        assert result.infrastructure_error is True  # DAT-09
        mock_container.stop.assert_called_once_with(timeout=0)
        mock_container.remove.assert_called_once_with(force=True)
        mock_client.close.assert_called_once_with()

    def test_cleanup_failures_do_not_mask_original_error(self, tmp_path: Path) -> None:
        """Cleanup errors after launch should not replace the original failure."""
        ctx = _make_ctx()
        scripts_dir = tmp_path / "scripts"
        instance_dir = scripts_dir / ctx.instance_id
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (instance_dir / "parser.py").write_text("import sys\n", encoding="utf-8")

        mock_container = MagicMock()
        mock_container.wait.side_effect = RuntimeError("post-launch failure")
        mock_container.stop.side_effect = RuntimeError("stop cleanup failed")
        mock_container.remove.side_effect = RuntimeError("remove cleanup failed")

        mock_client = MagicMock()
        mock_client.images.pull.return_value = None
        mock_client.containers.run.return_value = mock_container
        mock_client.close.side_effect = RuntimeError("client close failed")

        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=scripts_dir,
                dockerfiles_dir=None,
                output_dir=tmp_path / "output",
                prefix="baseline",
            )

        assert result.passed is False
        assert result.error is not None
        assert "post-launch failure" in result.error
        mock_container.stop.assert_called_once_with(timeout=0)
        mock_container.remove.assert_called_once_with(force=True)
        mock_client.close.assert_called_once_with()

    def test_client_closed_after_successful_evaluation(self, tmp_path: Path) -> None:
        """Successful evaluations explicitly close the Docker client."""
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        scripts_dir = tmp_path / "scripts"
        instance_dir = scripts_dir / ctx.instance_id
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (instance_dir / "parser.py").write_text("import sys\n", encoding="utf-8")

        output_dir = tmp_path / "output"
        workspace_dir = output_dir / ctx.instance_id / "workspace"
        workspace_dir.mkdir(parents=True)

        import json

        (workspace_dir / "output.json").write_text(
            json.dumps({"tests": [{"name": "test_a", "status": "PASSED"}]}), encoding="utf-8"
        )

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}

        mock_client = MagicMock()
        mock_client.images.pull.return_value = None
        mock_client.containers.run.return_value = mock_container

        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=scripts_dir,
                dockerfiles_dir=None,
                output_dir=output_dir,
                prefix="baseline",
            )

        assert result.passed is True
        mock_client.close.assert_called_once_with()


class TestRunDockerScoring:
    def test_records_nonfatal_instance_errors_in_summary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "pro-run"
        run_dir.mkdir()

        monkeypatch.setattr(score_docker_mod, "parse_paired_conditions", lambda _raw: ["baseline"])
        monkeypatch.setattr(
            score_docker_mod, "validate_paired_predictions", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            score_docker_mod,
            "prediction_path",
            lambda run_dir, condition: run_dir / condition / "predictions.jsonl",
        )
        monkeypatch.setattr(
            score_docker_mod,
            "read_predictions",
            lambda _path: [PredictionRecord("inst-1", "diff --git a/a.py b/a.py\n", "model")],
        )
        monkeypatch.setattr(
            score_docker_mod,
            "_load_dataset_rows",
            lambda **_kwargs: {
                "inst-1": {
                    "dockerhub_tag": "tag123",
                    "base_commit": "deadbeef",
                    "before_repo_set_cmd": "",
                    "selected_test_files_to_run": "",
                    "fail_to_pass": "[]",
                    "pass_to_pass": "[]",
                }
            },
        )
        monkeypatch.setattr(
            score_docker_mod,
            "evaluate_instance_docker",
            lambda *args, **kwargs: ScoreInstanceResult(
                instance_id="inst-1",
                passed=False,
                fail_to_pass_met=False,
                pass_to_pass_met=False,
                error="output.json not found after container execution",
            ),
        )

        summary = run_docker_scoring(
            run_dir=run_dir,
            conditions="baseline,llmdebug",
            dockerhub_username="testuser",
            scripts_dir=tmp_path / "scripts",
        )

        assert len(summary.condition_summaries) == 1
        item = summary.condition_summaries[0]
        assert item.condition == "baseline"
        assert item.passed_count == 0
        assert item.failed_count == 1
        assert item.error_count == 1
        assert item.accuracy == 0.0
        assert (run_dir / "docker_eval" / "baseline" / "eval_results.json").exists()

    def test_raises_when_instance_scoring_returns_infrastructure_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "pro-run"
        run_dir.mkdir()

        monkeypatch.setattr(score_docker_mod, "parse_paired_conditions", lambda _raw: ["baseline"])
        monkeypatch.setattr(
            score_docker_mod, "validate_paired_predictions", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            score_docker_mod,
            "prediction_path",
            lambda run_dir, condition: run_dir / condition / "predictions.jsonl",
        )
        monkeypatch.setattr(
            score_docker_mod,
            "read_predictions",
            lambda _path: [PredictionRecord("inst-1", "diff --git a/a.py b/a.py\n", "model")],
        )
        monkeypatch.setattr(
            score_docker_mod,
            "_load_dataset_rows",
            lambda **_kwargs: {
                "inst-1": {
                    "dockerhub_tag": "tag123",
                    "base_commit": "deadbeef",
                    "before_repo_set_cmd": "",
                    "selected_test_files_to_run": "",
                    "fail_to_pass": "[]",
                    "pass_to_pass": "[]",
                }
            },
        )
        monkeypatch.setattr(
            score_docker_mod,
            "evaluate_instance_docker",
            lambda *args, **kwargs: ScoreInstanceResult(
                instance_id="inst-1",
                passed=False,
                fail_to_pass_met=False,
                pass_to_pass_met=False,
                error="docker SDK not installed",
                infrastructure_error=True,
            ),
        )

        with pytest.raises(RuntimeError, match="Docker scoring encountered 1 infrastructure error"):
            run_docker_scoring(
                run_dir=run_dir,
                conditions="baseline,llmdebug",
                dockerhub_username="testuser",
                scripts_dir=tmp_path / "scripts",
            )

        # DAT-09: results and their reasons are persisted before failing loudly.
        detail_path = run_dir / "docker_eval" / "baseline" / "eval_results_detail.json"
        assert json.loads(detail_path.read_text(encoding="utf-8"))["inst-1"] == {
            "passed": False,
            "fail_to_pass_met": False,
            "pass_to_pass_met": False,
            "error": "docker SDK not installed",
            "infrastructure_error": True,
        }

    def test_returns_summary_when_no_instance_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "pro-run"
        run_dir.mkdir()

        monkeypatch.setattr(score_docker_mod, "parse_paired_conditions", lambda _raw: ["baseline"])
        monkeypatch.setattr(
            score_docker_mod, "validate_paired_predictions", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            score_docker_mod,
            "prediction_path",
            lambda run_dir, condition: run_dir / condition / "predictions.jsonl",
        )
        monkeypatch.setattr(
            score_docker_mod,
            "read_predictions",
            lambda _path: [PredictionRecord("inst-1", "diff --git a/a.py b/a.py\n", "model")],
        )
        monkeypatch.setattr(
            score_docker_mod,
            "_load_dataset_rows",
            lambda **_kwargs: {
                "inst-1": {
                    "dockerhub_tag": "tag123",
                    "base_commit": "deadbeef",
                    "before_repo_set_cmd": "",
                    "selected_test_files_to_run": "",
                    "fail_to_pass": "[]",
                    "pass_to_pass": "[]",
                }
            },
        )
        monkeypatch.setattr(
            score_docker_mod,
            "evaluate_instance_docker",
            lambda *args, **kwargs: ScoreInstanceResult(
                instance_id="inst-1",
                passed=True,
                fail_to_pass_met=True,
                pass_to_pass_met=True,
                error=None,
            ),
        )

        summary = run_docker_scoring(
            run_dir=run_dir,
            conditions="baseline,llmdebug",
            dockerhub_username="testuser",
            scripts_dir=tmp_path / "scripts",
        )

        assert len(summary.condition_summaries) == 1
        item = summary.condition_summaries[0]
        assert item.condition == "baseline"
        assert item.passed_count == 1
        assert item.error_count == 0
        assert item.accuracy == 1.0
        assert (run_dir / "docker_eval" / "baseline" / "eval_results.json").exists()


class TestReviewScoringIntegrity:
    """2026-10-04 review: DAT-04, DAT-09, DAT-11."""

    def test_empty_fail_to_pass_is_an_infrastructure_error(self, tmp_path: Path) -> None:
        ctx = _make_ctx(fail_to_pass=set(), pass_to_pass=set())
        with patch.dict("sys.modules", {"docker": MagicMock()}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=tmp_path / "scripts",
                dockerfiles_dir=None,
                output_dir=tmp_path / "output",
                prefix="baseline",
            )
        assert result.passed is False
        assert result.infrastructure_error is True
        assert result.error == "empty fail_to_pass test list"

    def test_patch_that_does_not_apply_is_a_model_failure(self, tmp_path: Path) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        instance_dir = tmp_path / "scripts" / ctx.instance_id
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (instance_dir / "parser.py").write_text("import sys\n", encoding="utf-8")
        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": score_docker_mod.APPLY_FAILED_EXIT_CODE}
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client
        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = evaluate_instance_docker(
                ctx,
                dockerhub_username="testuser",
                scripts_dir=tmp_path / "scripts",
                dockerfiles_dir=None,
                output_dir=tmp_path / "output",
                prefix="baseline",
            )
        assert result.passed is False
        assert result.infrastructure_error is False
        assert result.error == "patch_apply_failed"

    def test_dataset_values_are_shell_quoted(self) -> None:
        script = build_entry_script(
            instance_id="repo__1",
            base_commit="abc; rm -rf /",
            before_repo_set_cmd="",
            selected_test_files="tests/test a.py,tests/test_b.py",
            env_commands=[],
        )
        assert "git reset --hard 'abc; rm -rf /'" in script
        assert "bash /workspace/run_script.sh 'tests/test a.py,tests/test_b.py'" in script
