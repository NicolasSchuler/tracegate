"""Tests for SWE-bench Pro run_scripts utilities.

Exercises Dockerfile ENV extraction, instance script validation, path helpers,
and instance listing — all WITHOUT network access, git operations, or Docker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.swebench_pro.run_scripts import (
    DEFAULT_PIN_COMMIT,
    default_cache_dir,
    dockerfiles_dir,
    extract_env_commands,
    list_available_instances,
    load_base_dockerfile,
    load_instance_dockerfile,
    load_parser_script,
    load_run_script,
    run_scripts_dir,
    validate_instance_scripts,
)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestPathHelpers:
    def test_default_cache_dir_is_under_evals(self) -> None:
        result = default_cache_dir()
        assert result.name == "swebench_pro_assets"
        assert "evals" in str(result)

    def test_default_cache_dir_is_absolute(self) -> None:
        result = default_cache_dir()
        assert result.is_absolute()

    def test_run_scripts_dir(self, tmp_path: Path) -> None:
        assert run_scripts_dir(tmp_path) == tmp_path / "run_scripts"

    def test_dockerfiles_dir(self, tmp_path: Path) -> None:
        assert dockerfiles_dir(tmp_path) == tmp_path / "dockerfiles"

    def test_default_pin_commit_is_sha(self) -> None:
        assert len(DEFAULT_PIN_COMMIT) == 40
        assert all(c in "0123456789abcdef" for c in DEFAULT_PIN_COMMIT)


# ---------------------------------------------------------------------------
# validate_instance_scripts
# ---------------------------------------------------------------------------


class TestValidateInstanceScripts:
    def test_both_files_present(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        instance_dir = scripts / "instance__42"
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (instance_dir / "parser.py").write_text("import sys\n", encoding="utf-8")

        run_script_path, parser_path = validate_instance_scripts(
            scripts_dir=scripts,
            instance_id="instance__42",
        )

        assert run_script_path == instance_dir / "run_script.sh"
        assert parser_path == instance_dir / "parser.py"

    def test_missing_run_script(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        instance_dir = scripts / "instance__42"
        instance_dir.mkdir(parents=True)
        (instance_dir / "parser.py").write_text("import sys\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match=r"run_script\.sh not found"):
            validate_instance_scripts(
                scripts_dir=scripts,
                instance_id="instance__42",
            )

    def test_missing_parser(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        instance_dir = scripts / "instance__42"
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("#!/bin/bash\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match=r"parser\.py not found"):
            validate_instance_scripts(
                scripts_dir=scripts,
                instance_id="instance__42",
            )

    def test_missing_instance_directory(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        scripts.mkdir()

        with pytest.raises(FileNotFoundError, match=r"run_script\.sh not found"):
            validate_instance_scripts(
                scripts_dir=scripts,
                instance_id="nonexistent__99",
            )


# ---------------------------------------------------------------------------
# load_run_script / load_parser_script
# ---------------------------------------------------------------------------


class TestLoadScripts:
    def test_load_run_script(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        instance_dir = scripts / "instance__1"
        instance_dir.mkdir(parents=True)
        content = "#!/bin/bash\nset -e\npytest tests/\n"
        (instance_dir / "run_script.sh").write_text(content, encoding="utf-8")
        (instance_dir / "parser.py").write_text("import sys\n", encoding="utf-8")

        assert load_run_script(scripts, "instance__1") == content

    def test_load_parser_script(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        instance_dir = scripts / "instance__2"
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        content = "import json\nprint(json.dumps({}))\n"
        (instance_dir / "parser.py").write_text(content, encoding="utf-8")

        assert load_parser_script(scripts, "instance__2") == content

    def test_load_run_script_not_found(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        scripts.mkdir()

        with pytest.raises(FileNotFoundError):
            load_run_script(scripts, "missing__99")

    def test_load_parser_script_not_found(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        scripts.mkdir()

        with pytest.raises(FileNotFoundError):
            load_parser_script(scripts, "missing__99")


# ---------------------------------------------------------------------------
# load_base_dockerfile / load_instance_dockerfile
# ---------------------------------------------------------------------------


class TestLoadDockerfiles:
    def test_load_base_dockerfile(self, tmp_path: Path) -> None:
        dockerfiles = tmp_path / "dockerfiles"
        path = dockerfiles / "base_dockerfile" / "instance__1" / "Dockerfile"
        path.parent.mkdir(parents=True)
        content = "FROM python:3.11\nRUN pip install foo\n"
        path.write_text(content, encoding="utf-8")

        assert load_base_dockerfile(dockerfiles, "instance__1") == content

    def test_load_instance_dockerfile(self, tmp_path: Path) -> None:
        dockerfiles = tmp_path / "dockerfiles"
        path = dockerfiles / "instance_dockerfile" / "instance__1" / "Dockerfile"
        path.parent.mkdir(parents=True)
        content = "FROM base\nENV LANG=C.UTF-8\n"
        path.write_text(content, encoding="utf-8")

        assert load_instance_dockerfile(dockerfiles, "instance__1") == content

    def test_load_base_dockerfile_missing(self, tmp_path: Path) -> None:
        dockerfiles = tmp_path / "dockerfiles"
        dockerfiles.mkdir()

        with pytest.raises(FileNotFoundError, match="Base Dockerfile not found"):
            load_base_dockerfile(dockerfiles, "missing__1")

    def test_load_instance_dockerfile_missing(self, tmp_path: Path) -> None:
        dockerfiles = tmp_path / "dockerfiles"
        dockerfiles.mkdir()

        with pytest.raises(FileNotFoundError, match="Instance Dockerfile not found"):
            load_instance_dockerfile(dockerfiles, "missing__1")


# ---------------------------------------------------------------------------
# extract_env_commands
# ---------------------------------------------------------------------------


class TestExtractEnvCommands:
    def test_extracts_from_both_dockerfiles(self, tmp_path: Path) -> None:
        dockerfiles = tmp_path / "dockerfiles"

        # Base Dockerfile
        base_dir = dockerfiles / "base_dockerfile" / "instance__1"
        base_dir.mkdir(parents=True)
        (base_dir / "Dockerfile").write_text(
            "FROM python:3.11\n"
            "ENV PYTHONDONTWRITEBYTECODE=1\n"
            "RUN pip install foo\n"
            "ENV PATH=/opt/bin:$PATH\n",
            encoding="utf-8",
        )

        # Instance Dockerfile
        inst_dir = dockerfiles / "instance_dockerfile" / "instance__1"
        inst_dir.mkdir(parents=True)
        (inst_dir / "Dockerfile").write_text(
            "FROM base_image\nENV LANG=C.UTF-8\nCOPY . /app\n",
            encoding="utf-8",
        )

        cmds = extract_env_commands(dockerfiles, "instance__1")
        assert len(cmds) == 3
        assert cmds[0] == "export PYTHONDONTWRITEBYTECODE=1"
        assert cmds[1] == "export PATH=/opt/bin:$PATH"
        assert cmds[2] == "export LANG=C.UTF-8"

    def test_no_env_commands(self, tmp_path: Path) -> None:
        dockerfiles = tmp_path / "dockerfiles"

        base_dir = dockerfiles / "base_dockerfile" / "instance__2"
        base_dir.mkdir(parents=True)
        (base_dir / "Dockerfile").write_text(
            "FROM python:3.11\nRUN pip install foo\n",
            encoding="utf-8",
        )

        cmds = extract_env_commands(dockerfiles, "instance__2")
        assert cmds == []

    def test_missing_both_dockerfiles_returns_empty(self, tmp_path: Path) -> None:
        dockerfiles = tmp_path / "dockerfiles"
        dockerfiles.mkdir()

        cmds = extract_env_commands(dockerfiles, "nonexistent__99")
        assert cmds == []

    def test_missing_instance_dockerfile_only(self, tmp_path: Path) -> None:
        """Base Dockerfile exists but instance doesn't."""
        dockerfiles = tmp_path / "dockerfiles"

        base_dir = dockerfiles / "base_dockerfile" / "instance__3"
        base_dir.mkdir(parents=True)
        (base_dir / "Dockerfile").write_text(
            "FROM ubuntu\nENV DEBIAN_FRONTEND=noninteractive\n",
            encoding="utf-8",
        )

        cmds = extract_env_commands(dockerfiles, "instance__3")
        assert len(cmds) == 1
        assert cmds[0] == "export DEBIAN_FRONTEND=noninteractive"

    def test_env_with_spaces(self, tmp_path: Path) -> None:
        dockerfiles = tmp_path / "dockerfiles"

        base_dir = dockerfiles / "base_dockerfile" / "instance__4"
        base_dir.mkdir(parents=True)
        (base_dir / "Dockerfile").write_text(
            "ENV MY_VAR=hello world\n",
            encoding="utf-8",
        )

        cmds = extract_env_commands(dockerfiles, "instance__4")
        assert len(cmds) == 1
        assert cmds[0] == "export MY_VAR=hello world"

    def test_env_multiword_syntax(self, tmp_path: Path) -> None:
        """Docker supports ``ENV KEY VALUE`` (legacy) and ``ENV KEY=VALUE``."""
        dockerfiles = tmp_path / "dockerfiles"

        base_dir = dockerfiles / "base_dockerfile" / "instance__5"
        base_dir.mkdir(parents=True)
        (base_dir / "Dockerfile").write_text(
            "ENV FOO bar\n",
            encoding="utf-8",
        )

        cmds = extract_env_commands(dockerfiles, "instance__5")
        assert len(cmds) == 1
        assert cmds[0] == "export FOO=bar"

    def test_env_multiword_legacy_value_is_shell_safe(self, tmp_path: Path) -> None:
        dockerfiles = tmp_path / "dockerfiles"

        base_dir = dockerfiles / "base_dockerfile" / "instance__6"
        base_dir.mkdir(parents=True)
        (base_dir / "Dockerfile").write_text(
            "ENV FOO bar baz\n",
            encoding="utf-8",
        )

        cmds = extract_env_commands(dockerfiles, "instance__6")
        assert len(cmds) == 1
        assert cmds[0] == 'export FOO="bar baz"'


# ---------------------------------------------------------------------------
# list_available_instances
# ---------------------------------------------------------------------------


class TestListAvailableInstances:
    def test_lists_instance_directories(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        (scripts / "instance__alpha").mkdir(parents=True)
        (scripts / "instance__beta").mkdir(parents=True)
        (scripts / "instance__gamma").mkdir(parents=True)

        result = list_available_instances(scripts)
        assert result == ["instance__alpha", "instance__beta", "instance__gamma"]

    def test_excludes_non_instance_dirs(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        (scripts / "instance__alpha").mkdir(parents=True)
        (scripts / "README.md").parent.mkdir(parents=True, exist_ok=True)
        (scripts / "README.md").write_text("readme", encoding="utf-8")
        (scripts / "other_dir").mkdir(parents=True)

        result = list_available_instances(scripts)
        # Only dirs starting with "instance_" are included
        assert result == ["instance__alpha"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        scripts.mkdir()

        assert list_available_instances(scripts) == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        scripts = tmp_path / "nonexistent"
        assert list_available_instances(scripts) == []

    def test_sorted_output(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        (scripts / "instance__zzz").mkdir(parents=True)
        (scripts / "instance__aaa").mkdir(parents=True)
        (scripts / "instance__mmm").mkdir(parents=True)

        result = list_available_instances(scripts)
        assert result == ["instance__aaa", "instance__mmm", "instance__zzz"]

    def test_ignores_files_named_instance(self, tmp_path: Path) -> None:
        scripts = tmp_path / "run_scripts"
        scripts.mkdir()
        (scripts / "instance__file.txt").write_text("not a dir", encoding="utf-8")

        result = list_available_instances(scripts)
        assert result == []
