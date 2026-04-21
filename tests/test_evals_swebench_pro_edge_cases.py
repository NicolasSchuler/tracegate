"""Edge-case tests for SWE-bench Pro integration.

Covers unicode handling, empty/missing fields, special-character instance IDs,
malformed Dockerfile content, Docker scoring edge cases, and IO boundary
conditions — all WITHOUT network access, Docker, or git operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals.swebench.types import CONDITION_BASELINE, CONDITION_LLMDEBUG, PredictionRecord
from evals.swebench_pro.generate import _build_prompt_pro, _stringify_field
from evals.swebench_pro.io import (
    load_instance_ids,
    load_json,
    read_predictions,
    validate_run_id,
    write_instance_ids,
    write_json,
    write_predictions,
)
from evals.swebench_pro.run_scripts import extract_env_commands
from evals.swebench_pro.score_docker import (
    _EvalContext,
    _parse_test_list,
    _score_output,
    build_entry_script,
    strip_binary_hunks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL: object = object()


def _make_row(**overrides: Any) -> dict[str, Any]:
    """Return a standard Pro dataset row, optionally overriding fields."""
    row: dict[str, Any] = {
        "instance_id": "pro-test-123",
        "repo": "demo/pro-repo",
        "base_commit": "abc123",
        "problem_statement": "Fix the Pro bug.",
        "repo_language": "Python",
        "requirements": "",
        "interface": "",
        "fail_to_pass": None,
        "pass_to_pass": None,
    }
    row.update(overrides)
    return row


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


def _setup_dockerfiles(
    tmp_path: Path,
    instance_id: str,
    *,
    base_content: str | None = None,
    instance_content: str | None = None,
) -> Path:
    """Create a dockerfiles directory tree with optional base/instance Dockerfiles."""
    dockerfiles = tmp_path / "dockerfiles"
    if base_content is not None:
        base_dir = dockerfiles / "base_dockerfile" / instance_id
        base_dir.mkdir(parents=True)
        (base_dir / "Dockerfile").write_text(base_content, encoding="utf-8")
    if instance_content is not None:
        inst_dir = dockerfiles / "instance_dockerfile" / instance_id
        inst_dir.mkdir(parents=True)
        (inst_dir / "Dockerfile").write_text(instance_content, encoding="utf-8")
    dockerfiles.mkdir(parents=True, exist_ok=True)
    return dockerfiles


# ===========================================================================
# 1. Unicode / special characters
# ===========================================================================


class TestUnicodeBuildPromptPro:
    """_build_prompt_pro with unicode in problem_statement and fields."""

    def test_chinese_characters_in_problem_statement(self) -> None:
        row = _make_row(problem_statement="修复这个错误：变量未定义")  # noqa: RUF001
        prompt = _build_prompt_pro(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

        assert "修复这个错误：变量未定义" in prompt  # noqa: RUF001

    def test_emoji_in_problem_statement(self) -> None:
        row = _make_row(problem_statement="Fix the 🐛 bug in 📦 package")
        prompt = _build_prompt_pro(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

        assert "🐛" in prompt
        assert "📦" in prompt

    def test_rtl_text_in_problem_statement(self) -> None:
        row = _make_row(problem_statement="إصلاح الخطأ في البرنامج")
        prompt = _build_prompt_pro(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

        assert "إصلاح الخطأ في البرنامج" in prompt

    def test_mixed_unicode_scripts(self) -> None:
        row = _make_row(
            problem_statement="Fix bug: 日本語テスト → résultat überprüfen",
            requirements="パッケージ>=1.0",
            interface="def lösung(входные_данные: str) -> str: ...",
        )
        prompt = _build_prompt_pro(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

        assert "日本語テスト" in prompt
        assert "パッケージ>=1.0" in prompt
        assert "lösung" in prompt


class TestUnicodePredictionsRoundTrip:
    """write_predictions / read_predictions round-trip with unicode content."""

    def test_unicode_model_patch(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        records = [
            PredictionRecord(
                instance_id="inst-1",
                model_patch="diff --git a/f.py b/f.py\n+# 修复: 添加中文注释\n",
                model_name_or_path="model-a",
            ),
        ]
        write_predictions(path, records)
        loaded = read_predictions(path)

        assert len(loaded) == 1
        assert "修复" in loaded[0].model_patch
        assert loaded[0] == records[0]

    def test_emoji_in_model_patch(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        records = [
            PredictionRecord(
                instance_id="inst-2",
                model_patch="diff --git a/f.py b/f.py\n+# 🎉 Fix complete\n",
                model_name_or_path="model-b",
            ),
        ]
        write_predictions(path, records)
        loaded = read_predictions(path)

        assert loaded[0].model_patch == records[0].model_patch

    def test_rtl_and_bidi_in_model_patch(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        records = [
            PredictionRecord(
                instance_id="inst-3",
                model_patch="diff --git a/f.py b/f.py\n+# مرحبا بالعالم\n",
                model_name_or_path="model-c",
            ),
        ]
        write_predictions(path, records)
        loaded = read_predictions(path)

        assert "مرحبا" in loaded[0].model_patch


class TestUnicodeStripBinaryHunks:
    """strip_binary_hunks with unicode diff content."""

    def test_unicode_diff_preserved(self) -> None:
        patch = (
            "diff --git a/файл.py b/файл.py\n"
            "--- a/файл.py\n"
            "+++ b/файл.py\n"
            "@@ -1 +1 @@\n"
            "-旧内容\n"
            "+新内容\n"
        )
        result = strip_binary_hunks(patch)

        assert "файл.py" in result
        assert "新内容" in result

    def test_unicode_diff_with_binary_section_removed(self) -> None:
        patch = (
            "diff --git a/코드.py b/코드.py\n"
            "--- a/코드.py\n"
            "+++ b/코드.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "diff --git a/画像.png b/画像.png\n"
            "Binary files a/画像.png and b/画像.png differ\n"
        )
        result = strip_binary_hunks(patch)

        assert "코드.py" in result
        assert "画像.png" not in result


class TestUnicodeExtractEnvCommands:
    """extract_env_commands with unicode in ENV values."""

    def test_unicode_env_value(self, tmp_path: Path) -> None:
        dockerfiles = _setup_dockerfiles(
            tmp_path,
            "inst_unicode",
            base_content="FROM python:3.11\nENV GREETING=こんにちは\n",
        )
        cmds = extract_env_commands(dockerfiles, "inst_unicode")

        assert len(cmds) == 1
        assert cmds[0] == "export GREETING=こんにちは"

    def test_emoji_env_value(self, tmp_path: Path) -> None:
        dockerfiles = _setup_dockerfiles(
            tmp_path,
            "inst_emoji",
            base_content="FROM python:3.11\nENV TAG=🚀release\n",
        )
        cmds = extract_env_commands(dockerfiles, "inst_emoji")

        assert len(cmds) == 1
        assert "🚀" in cmds[0]


# ===========================================================================
# 2. Empty / missing dataset fields
# ===========================================================================


class TestEmptyMissingFields:
    """_build_prompt_pro when optional fields are all None or empty."""

    def test_all_optional_fields_none(self) -> None:
        row = _make_row(
            requirements=None,
            interface=None,
            repo_language=None,
            fail_to_pass=None,
            pass_to_pass=None,
        )
        prompt = _build_prompt_pro(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

        assert "Fix the Pro bug." in prompt
        assert "Requirements:" not in prompt
        assert "Interface:" not in prompt
        assert "Language:" not in prompt
        assert "Fail-to-pass tests:" not in prompt
        assert "Pass-to-pass tests:" not in prompt

    def test_all_optional_fields_empty_string(self) -> None:
        row = _make_row(
            requirements="",
            interface="",
            repo_language="",
            fail_to_pass="",
            pass_to_pass="",
        )
        prompt = _build_prompt_pro(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

        assert "Requirements:" not in prompt
        assert "Interface:" not in prompt
        assert "Language:" not in prompt

    def test_empty_problem_statement(self) -> None:
        row = _make_row(problem_statement="")
        prompt = _build_prompt_pro(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

        # Prompt still includes structural elements
        assert "Repository: demo/pro-repo" in prompt
        assert "Instance ID: pro-test-123" in prompt
        assert "Issue:" in prompt

    def test_whitespace_only_problem_statement(self) -> None:
        row = _make_row(problem_statement="   \n  \t  ")
        prompt = _build_prompt_pro(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

        assert "Repository: demo/pro-repo" in prompt
        assert "Issue:" in prompt

    def test_empty_fail_to_pass_string(self) -> None:
        row = _make_row(fail_to_pass="", pass_to_pass="")
        prompt = _build_prompt_pro(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

        assert "Fail-to-pass tests:" not in prompt
        assert "Pass-to-pass tests:" not in prompt

    def test_empty_list_fail_to_pass(self) -> None:
        row = _make_row(fail_to_pass=[], pass_to_pass=[])
        prompt = _build_prompt_pro(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

        assert "Fail-to-pass tests:" not in prompt
        assert "Pass-to-pass tests:" not in prompt

    def test_llmdebug_with_all_empty_fields(self) -> None:
        row = _make_row(
            requirements=None,
            interface=None,
            repo_language=None,
            fail_to_pass=None,
            pass_to_pass=None,
        )
        prompt = _build_prompt_pro(row=row, condition=CONDITION_LLMDEBUG, snapshot_dir=None)

        # llmdebug section still appears
        assert "llmdebug runtime evidence" in prompt
        assert "No snapshot directory was provided" in prompt


# ===========================================================================
# 3. Special characters in instance_ids
# ===========================================================================


class TestSpecialInstanceIds:
    """Predictions and instance IDs with underscores, double underscores, hyphens."""

    def test_write_predictions_double_underscore_id(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        records = [
            PredictionRecord(
                instance_id="instance_OWNER__REPO-abc123-v1",
                model_patch="diff --git a/f.py b/f.py\n+line\n",
                model_name_or_path="model",
            ),
        ]
        write_predictions(path, records)
        loaded = read_predictions(path)

        assert loaded[0].instance_id == "instance_OWNER__REPO-abc123-v1"

    def test_write_predictions_many_special_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        ids = [
            "a__b__c",
            "org-name__repo-name-123",
            "x_y-z.1.2.3",
            "UPPER__lower__MiXeD",
        ]
        records = [
            PredictionRecord(
                instance_id=iid,
                model_patch=f"diff --git a/f.py b/f.py\n+{iid}\n",
                model_name_or_path="m",
            )
            for iid in ids
        ]
        write_predictions(path, records)
        loaded = read_predictions(path)

        assert [r.instance_id for r in loaded] == ids

    def test_instance_ids_round_trip_double_underscore(self, tmp_path: Path) -> None:
        path = tmp_path / "ids.txt"
        ids = [
            "instance_OWNER__REPO-abc123-v1",
            "another__double__underscore",
            "simple-id",
        ]
        write_instance_ids(path, ids)
        loaded = load_instance_ids(path)

        assert loaded == ids

    def test_instance_ids_with_dots_and_numbers(self, tmp_path: Path) -> None:
        path = tmp_path / "ids.txt"
        ids = ["v1.2.3__issue__42", "123__456__789"]
        write_instance_ids(path, ids)
        loaded = load_instance_ids(path)

        assert loaded == ids


# ===========================================================================
# 4. Malformed Dockerfile / script content
# ===========================================================================


class TestMalformedDockerfileContent:
    """extract_env_commands with unusual Dockerfile content."""

    def test_dockerfile_only_comments(self, tmp_path: Path) -> None:
        dockerfiles = _setup_dockerfiles(
            tmp_path,
            "inst_comments",
            base_content="# This is a comment\n# Another comment\n# ENV NOT_REAL=1\n",
        )
        cmds = extract_env_commands(dockerfiles, "inst_comments")

        assert cmds == []

    def test_env_inside_run_not_extracted(self, tmp_path: Path) -> None:
        """ENV inside a RUN command should NOT be extracted as a top-level ENV."""
        dockerfiles = _setup_dockerfiles(
            tmp_path,
            "inst_run_env",
            base_content=(
                'FROM python:3.11\nRUN export ENV_VAR=1 && echo done\nRUN bash -c "ENV FOO=bar"\n'
            ),
        )
        cmds = extract_env_commands(dockerfiles, "inst_run_env")

        # Lines starting with "RUN" should not produce export commands
        assert cmds == []

    def test_env_with_backslash_continuation(self, tmp_path: Path) -> None:
        """Multi-line ENV with backslash continuation — only first line is extracted."""
        dockerfiles = _setup_dockerfiles(
            tmp_path,
            "inst_continuation",
            base_content=("FROM python:3.11\nENV MY_VAR=hello \\\n    world\nENV SIMPLE=yes\n"),
        )
        cmds = extract_env_commands(dockerfiles, "inst_continuation")

        # The line-by-line parser extracts lines starting with ENV
        # so the continuation line ("    world") is NOT extracted
        assert any("MY_VAR" in c for c in cmds)
        assert any("SIMPLE" in c for c in cmds)
        # The continuation line should not appear as a standalone export
        assert not any(c.strip() == "export world" for c in cmds)

    def test_empty_dockerfile(self, tmp_path: Path) -> None:
        dockerfiles = _setup_dockerfiles(
            tmp_path,
            "inst_empty",
            base_content="",
        )
        cmds = extract_env_commands(dockerfiles, "inst_empty")

        assert cmds == []

    def test_dockerfile_with_env_no_value(self, tmp_path: Path) -> None:
        """ENV with no assignment — still extracted as-is."""
        dockerfiles = _setup_dockerfiles(
            tmp_path,
            "inst_no_val",
            base_content="FROM python:3.11\nENV\n",
        )
        cmds = extract_env_commands(dockerfiles, "inst_no_val")

        # "ENV" alone becomes "export" (stripped ENV line)
        assert len(cmds) == 1
        assert cmds[0] == "export"

    def test_env_with_equals_in_value(self, tmp_path: Path) -> None:
        dockerfiles = _setup_dockerfiles(
            tmp_path,
            "inst_eq",
            base_content="FROM python:3.11\nENV OPTS=--flag=value --other=x\n",
        )
        cmds = extract_env_commands(dockerfiles, "inst_eq")

        assert len(cmds) == 1
        assert cmds[0] == "export OPTS=--flag=value --other=x"


class TestParseTestListEdgeCases:
    """_parse_test_list with malformed or unusual input."""

    def test_malformed_json(self) -> None:
        result = _parse_test_list("{not valid json")
        # Falls back to treating the whole string as a single test name
        assert result == {"{not valid json"}

    def test_nested_lists(self) -> None:
        result = _parse_test_list('[["inner_a", "inner_b"], "outer"]')
        # ast.literal_eval parses this as a list; items are stringified
        assert "outer" in result
        # Nested list is stringified
        assert len(result) == 2

    def test_mixed_types_in_list(self) -> None:
        result = _parse_test_list('["test_a", 42, true]')
        # JSON parse succeeds; items are stringified
        assert "test_a" in result
        assert "42" in result

    def test_python_set_literal(self) -> None:
        result = _parse_test_list('{"test_x", "test_y"}')
        assert result == {"test_x", "test_y"}

    def test_single_quoted_string(self) -> None:
        result = _parse_test_list("'just_a_string'")
        # ast.literal_eval parses to a string (not list/tuple/set), so the
        # literal_eval branch does not capture it.  JSON parse also fails.
        # Plain-text fallback returns the raw input (with quotes) as a set.
        assert result == {"'just_a_string'"}

    def test_integer_literal(self) -> None:
        result = _parse_test_list("42")
        # Not a list/tuple/set, so falls through to plain text
        assert result == {"42"}

    def test_none_string(self) -> None:
        result = _parse_test_list("None")
        # ast.literal_eval("None") returns None, which is not list/tuple/set
        # Falls through to plain text
        assert result == {"None"}


# ===========================================================================
# 5. Docker scoring edge cases
# ===========================================================================


class TestScoreOutputEdgeCases:
    """_score_output with unusual test statuses and names."""

    def test_skipped_status_not_counted_as_passed(self) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass={"test_b"})
        output: dict[str, Any] = {
            "tests": [
                {"name": "test_a", "status": "SKIPPED"},
                {"name": "test_b", "status": "SKIPPED"},
            ]
        }
        result = _score_output(ctx, output)

        assert result.passed is False
        assert result.fail_to_pass_met is False
        assert result.pass_to_pass_met is False

    def test_error_status_not_counted_as_passed(self) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        output: dict[str, Any] = {
            "tests": [
                {"name": "test_a", "status": "ERROR"},
            ]
        }
        result = _score_output(ctx, output)

        assert result.passed is False
        assert result.fail_to_pass_met is False

    def test_mixed_statuses(self) -> None:
        ctx = _make_ctx(
            fail_to_pass={"test_a"},
            pass_to_pass={"test_b"},
        )
        output: dict[str, Any] = {
            "tests": [
                {"name": "test_a", "status": "PASSED"},
                {"name": "test_b", "status": "SKIPPED"},
                {"name": "test_c", "status": "ERROR"},
                {"name": "test_d", "status": "FAILED"},
            ]
        }
        result = _score_output(ctx, output)

        assert result.fail_to_pass_met is True
        assert result.pass_to_pass_met is False
        assert result.passed is False

    def test_extremely_long_test_name(self) -> None:
        long_name = "test_" + "x" * 10000
        ctx = _make_ctx(fail_to_pass={long_name}, pass_to_pass=set())
        output: dict[str, Any] = {
            "tests": [
                {"name": long_name, "status": "PASSED"},
            ]
        }
        result = _score_output(ctx, output)

        assert result.passed is True
        assert result.fail_to_pass_met is True

    def test_duplicate_test_names_in_output(self) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        output: dict[str, Any] = {
            "tests": [
                {"name": "test_a", "status": "FAILED"},
                {"name": "test_a", "status": "PASSED"},
            ]
        }
        result = _score_output(ctx, output)

        # Second occurrence overrides in the set comprehension (PASSED wins)
        assert result.fail_to_pass_met is True
        assert result.passed is True

    def test_empty_status_string(self) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        output: dict[str, Any] = {
            "tests": [
                {"name": "test_a", "status": ""},
            ]
        }
        result = _score_output(ctx, output)

        assert result.fail_to_pass_met is False

    def test_missing_status_key(self) -> None:
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        output: dict[str, Any] = {
            "tests": [
                {"name": "test_a"},
            ]
        }
        result = _score_output(ctx, output)

        assert result.fail_to_pass_met is False

    def test_missing_name_key_raises(self) -> None:
        """Missing 'name' key in a test dict triggers KeyError — documents current behavior."""
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        output: dict[str, Any] = {
            "tests": [
                {"status": "PASSED"},
            ]
        }

        with pytest.raises(KeyError):
            _score_output(ctx, output)

    def test_no_tests_key_in_output(self) -> None:
        ctx = _make_ctx(fail_to_pass=set(), pass_to_pass=set())
        output: dict[str, Any] = {"other_key": "value"}
        result = _score_output(ctx, output)

        # Empty expectations → trivially true
        assert result.passed is True

    def test_case_sensitive_status(self) -> None:
        """Status comparison is case-sensitive; 'passed' != 'PASSED'."""
        ctx = _make_ctx(fail_to_pass={"test_a"}, pass_to_pass=set())
        output: dict[str, Any] = {
            "tests": [
                {"name": "test_a", "status": "passed"},
            ]
        }
        result = _score_output(ctx, output)

        assert result.fail_to_pass_met is False


class TestBuildEntryScriptEdgeCases:
    """build_entry_script with edge-case inputs."""

    def test_empty_before_repo_set_cmd(self) -> None:
        script = build_entry_script(
            instance_id="repo__1",
            base_commit="abc123",
            before_repo_set_cmd="",
            selected_test_files="tests/t.py",
            env_commands=[],
        )

        assert "git reset --hard abc123" in script
        # No stray command appears for the empty setup line
        lines = script.splitlines()
        assert any("git apply" in ln for ln in lines)

    def test_before_repo_set_cmd_with_shell_metacharacters(self) -> None:
        cmd = 'pip install -e . && echo "done" | tee log.txt; rm -rf /tmp/*'
        script = build_entry_script(
            instance_id="repo__2",
            base_commit="def456",
            before_repo_set_cmd=cmd,
            selected_test_files="t.py",
            env_commands=[],
        )

        # The last line of the multi-statement is used verbatim
        assert cmd in script

    def test_very_long_patch_in_context(self) -> None:
        """build_entry_script handles long commit hashes and test file args."""
        long_commit = "a" * 128
        long_test = ",".join(f"tests/test_{i}.py" for i in range(100))
        script = build_entry_script(
            instance_id="repo__3",
            base_commit=long_commit,
            before_repo_set_cmd="",
            selected_test_files=long_test,
            env_commands=[],
        )

        assert long_commit in script
        assert long_test in script

    def test_env_commands_with_special_characters(self) -> None:
        script = build_entry_script(
            instance_id="repo__4",
            base_commit="abc",
            before_repo_set_cmd="",
            selected_test_files="t.py",
            env_commands=[
                "export PATH=/usr/local/bin:$PATH",
                'export OPTS="--flag=value --other"',
                "export EMPTY=",
            ],
        )

        assert "export PATH=/usr/local/bin:$PATH" in script
        assert 'export OPTS="--flag=value --other"' in script
        assert "export EMPTY=" in script


# ===========================================================================
# 6. IO edge cases
# ===========================================================================


class TestValidateRunIdEdgeCases:
    """validate_run_id with boundary and adversarial inputs."""

    def test_very_long_run_id(self) -> None:
        long_id = "a" * 300
        # Accepted because the regex only checks character classes, not length
        assert validate_run_id(long_id) == long_id

    def test_single_character_run_id(self) -> None:
        assert validate_run_id("x") == "x"

    def test_all_digits_run_id(self) -> None:
        assert validate_run_id("12345") == "12345"

    def test_dots_and_underscores_only(self) -> None:
        assert validate_run_id("._-._") == "._-._"

    def test_rejects_dollar_sign(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            validate_run_id("run$id")

    def test_rejects_backtick(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            validate_run_id("run`id")

    def test_rejects_pipe(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            validate_run_id("run|id")

    def test_rejects_semicolon(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            validate_run_id("run;id")

    def test_rejects_ampersand(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            validate_run_id("run&id")

    def test_rejects_newline(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            validate_run_id("run\nid")

    def test_rejects_null_byte(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            validate_run_id("run\x00id")

    def test_rejects_tilde(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            validate_run_id("run~id")


class TestWriteJsonLoadJsonEdgeCases:
    """write_json / load_json with deeply nested structures."""

    def test_nested_dict_structure(self, tmp_path: Path) -> None:
        path = tmp_path / "nested.json"
        payload: dict[str, Any] = {
            "level1": {
                "level2": {
                    "level3": {
                        "values": [1, 2, 3],
                        "flag": True,
                    }
                }
            }
        }
        write_json(path, payload)
        loaded = load_json(path)

        assert loaded["level1"]["level2"]["level3"]["values"] == [1, 2, 3]  # type: ignore[index]
        assert loaded["level1"]["level2"]["level3"]["flag"] is True  # type: ignore[index]

    def test_unicode_keys_and_values(self, tmp_path: Path) -> None:
        path = tmp_path / "unicode.json"
        payload: dict[str, Any] = {"名前": "テスト", "emoji": "🎉", "العربية": True}
        write_json(path, payload)
        loaded = load_json(path)

        assert loaded["名前"] == "テスト"
        assert loaded["emoji"] == "🎉"

    def test_empty_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        write_json(path, {})
        loaded = load_json(path)

        assert loaded == {}

    def test_large_nested_structure(self, tmp_path: Path) -> None:
        path = tmp_path / "large.json"
        payload: dict[str, Any] = {
            f"key_{i}": {"nested": list(range(50)), "flag": i % 2 == 0} for i in range(100)
        }
        write_json(path, payload)
        loaded = load_json(path)

        assert len(loaded) == 100
        assert loaded["key_0"]["nested"] == list(range(50))  # type: ignore[index]

    def test_null_values(self, tmp_path: Path) -> None:
        path = tmp_path / "nulls.json"
        payload: dict[str, Any] = {"a": None, "b": [None, 1, None], "c": {"d": None}}
        write_json(path, payload)
        loaded = load_json(path)

        assert loaded["a"] is None
        assert loaded["b"] == [None, 1, None]  # type: ignore[index]


class TestInstanceIdsEdgeCases:
    """load_instance_ids / write_instance_ids with edge-case content."""

    def test_ids_with_leading_trailing_whitespace_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "ids.txt"
        path.write_text("  inst-1  \n  inst-2  \n", encoding="utf-8")
        loaded = load_instance_ids(path)

        assert loaded == ["inst-1", "inst-2"]

    def test_ids_with_unicode(self, tmp_path: Path) -> None:
        path = tmp_path / "ids.txt"
        ids = ["inst-日本語", "inst-中文", "inst-한국어"]
        write_instance_ids(path, ids)
        loaded = load_instance_ids(path)

        assert loaded == ids

    def test_single_id(self, tmp_path: Path) -> None:
        path = tmp_path / "ids.txt"
        write_instance_ids(path, ["only-one"])
        loaded = load_instance_ids(path)

        assert loaded == ["only-one"]

    def test_many_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "ids.txt"
        ids = [f"inst-{i:06d}" for i in range(10000)]
        write_instance_ids(path, ids)
        loaded = load_instance_ids(path)

        assert loaded == ids

    def test_ids_with_only_blanks(self, tmp_path: Path) -> None:
        path = tmp_path / "ids.txt"
        path.write_text("\n\n  \n\t\n", encoding="utf-8")
        loaded = load_instance_ids(path)

        assert loaded == []


class TestStringifyFieldEdgeCases:
    """_stringify_field with edge-case values."""

    def test_empty_list(self) -> None:
        assert _stringify_field([]) == ""

    def test_list_with_empty_strings(self) -> None:
        assert _stringify_field(["", "  ", "valid"]) == "valid"

    def test_nested_dict(self) -> None:
        result = _stringify_field({"outer": {"inner": [1, 2, 3]}})
        assert '"outer"' in result
        assert '"inner"' in result

    def test_boolean_value(self) -> None:
        assert _stringify_field(True) == "True"
        assert _stringify_field(False) == "False"

    def test_float_value(self) -> None:
        assert _stringify_field(3.14) == "3.14"

    def test_zero(self) -> None:
        assert _stringify_field(0) == "0"

    def test_empty_string(self) -> None:
        assert _stringify_field("") == ""

    def test_whitespace_only_string(self) -> None:
        assert _stringify_field("   \n\t   ") == ""
