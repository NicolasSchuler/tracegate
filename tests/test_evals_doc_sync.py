"""Tests for evals doc_sync: changelog consistency and case inventory rendering."""

from __future__ import annotations

import subprocess
import types

import pytest

import evals.doc_sync as doc_sync
from evals.doc_sync import (
    BEGIN_CASE_INVENTORY_MARKER,
    END_CASE_INVENTORY_MARKER,
    CategorySummary,
    InventorySummary,
    _first_h2_heading,
    _format_source_name,
    _load_project_version_regex_fallback,
    build_inventory_summary,
    extract_case_inventory_block,
    get_git_tag_versions,
    render_case_inventory_block,
    replace_case_inventory_block,
    validate_changelog_consistency,
    write_case_inventory_block,
)
from llmdebug import _json as json


def _write_case(
    cases_dir,
    *,
    case_id: str,
    category: str,
    difficulty: str,
    snapshot_dependency: str,
    bug_source: str | None = None,
) -> None:
    case_dir = cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "description": f"Synthetic case: {case_id}",
        "category": category,
        "difficulty": difficulty,
        "snapshot_dependency": snapshot_dependency,
        "expected_exception": "AssertionError",
        "tags": ["synthetic"],
        "python_args": ["-m", "pytest", "-q"],
        "timeout_sec": 30,
    }
    if bug_source is not None:
        metadata["bug_source"] = bug_source
    (case_dir / "case.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_build_inventory_summary_from_synthetic_cases(tmp_path) -> None:
    _write_case(
        tmp_path,
        case_id="case_a",
        category="logic_error",
        difficulty="easy",
        snapshot_dependency="required",
        bug_source=None,
    )
    _write_case(
        tmp_path,
        case_id="case_b",
        category="logic_error",
        difficulty="hard",
        snapshot_dependency="helpful",
        bug_source="debugbench",
    )
    _write_case(
        tmp_path,
        case_id="case_c",
        category="hidden_state",
        difficulty="medium",
        snapshot_dependency="required",
        bug_source="condefects",
    )

    summary = build_inventory_summary(tmp_path)
    assert summary.total_cases == 3
    assert summary.source_counts == (("condefects", 1), ("debugbench", 1), ("hand_crafted", 1))
    assert summary.categories[0].name == "logic_error"
    assert summary.categories[0].count == 2
    assert summary.categories[0].difficulty_counts == (("easy", 1), ("hard", 1))
    assert summary.categories[1].name == "hidden_state"
    assert summary.categories[1].difficulty_counts == (("medium", 1),)


def test_render_case_inventory_block_is_deterministic() -> None:
    summary = InventorySummary(
        total_cases=2,
        source_counts=(("hand_crafted", 1), ("debugbench", 1)),
        categories=(
            CategorySummary(name="logic_error", count=1, difficulty_counts=(("easy", 1),)),
            CategorySummary(name="hidden_state", count=1, difficulty_counts=(("hard", 1),)),
        ),
    )

    rendered = render_case_inventory_block(summary)
    expected = "\n".join(
        [
            BEGIN_CASE_INVENTORY_MARKER,
            "_This section is auto-generated from `evals/cases` by "
            "`python -m evals.doc_sync --write`._",
            "",
            "2 validated cases across 2 categories and 2 sources:",
            "",
            "| Source | Cases |",
            "|--------|-------|",
            "| Hand-crafted | 1 |",
            "| DebugBench | 1 |",
            "| **Total** | **2** |",
            "",
            "Category distribution:",
            "",
            "| Category | Count | Difficulty breakdown |",
            "|----------|-------|---------------------|",
            "| logic_error | 1 | 1 easy |",
            "| hidden_state | 1 | 1 hard |",
            END_CASE_INVENTORY_MARKER,
        ]
    )
    assert rendered == expected


def test_replace_case_inventory_block_updates_marked_content() -> None:
    readme_text = (
        f"prefix\n{BEGIN_CASE_INVENTORY_MARKER}\nold-content\n{END_CASE_INVENTORY_MARKER}\nsuffix\n"
    )
    replacement = f"{BEGIN_CASE_INVENTORY_MARKER}\nnew-content\n{END_CASE_INVENTORY_MARKER}"
    updated = replace_case_inventory_block(readme_text, replacement)
    assert "old-content" not in updated
    assert "new-content" in updated
    assert updated.startswith("prefix\n")
    assert updated.endswith("suffix\n")


def test_replace_case_inventory_block_requires_markers() -> None:
    with pytest.raises(ValueError, match="Case inventory markers were not found"):
        replace_case_inventory_block("missing markers", "replacement")


def test_validate_changelog_consistency_reports_missing_tag_and_current_section() -> None:
    changelog = "# CHANGELOG\n\n## v1.0.0\n\n- Initial release\n"
    errors = validate_changelog_consistency(
        changelog,
        tagged_versions=["1.0.0", "1.1.0"],
        project_version="1.2.0",
    )
    assert any("v1.1.0" in error for error in errors)
    assert any("Unreleased" in error or "Current" in error for error in errors)


def test_validate_changelog_consistency_passes_when_current_section_is_present() -> None:
    changelog = (
        "# CHANGELOG\n\n"
        "## Current (1.2.0)\n\n"
        "- Pending release notes.\n\n"
        "## v1.1.0\n\n"
        "- Patch release.\n\n"
        "## v1.0.0\n\n"
        "- Initial release.\n"
    )
    errors = validate_changelog_consistency(
        changelog,
        tagged_versions=["1.0.0", "1.1.0"],
        project_version="1.2.0",
    )
    assert errors == []


def test_load_project_version_uses_regex_fallback_without_tomllib(tmp_path, monkeypatch) -> None:
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        "\n".join(
            [
                "[build-system]",
                'requires = ["hatchling"]',
                "",
                "[project]",
                'name = "llmdebug"',
                'version = "9.8.7"',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(doc_sync, "tomllib", None)

    version = doc_sync.load_project_version(pyproject_path)
    assert version == "9.8.7"


def test_run_checks_fails_when_case_inventory_block_is_stale(tmp_path, monkeypatch) -> None:
    cases_dir = tmp_path / "cases"
    _write_case(
        cases_dir,
        case_id="case_a",
        category="logic_error",
        difficulty="easy",
        snapshot_dependency="required",
        bug_source=None,
    )

    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# evals",
                BEGIN_CASE_INVENTORY_MARKER,
                "stale block",
                END_CASE_INVENTORY_MARKER,
                "",
            ]
        ),
        encoding="utf-8",
    )

    changelog_path = tmp_path / "CHANGELOG.md"
    changelog_path.write_text(
        "# CHANGELOG\n\n## Current (1.2.0)\n\n- Pending.\n\n## v1.1.0\n\n- Notes.\n",
        encoding="utf-8",
    )

    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        '[project]\nname = "llmdebug"\nversion = "1.2.0"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(doc_sync, "CASES_DIR", cases_dir)
    monkeypatch.setattr(doc_sync, "EVALS_README_PATH", readme_path)
    monkeypatch.setattr(doc_sync, "CHANGELOG_PATH", changelog_path)
    monkeypatch.setattr(doc_sync, "PYPROJECT_PATH", pyproject_path)
    monkeypatch.setattr(doc_sync, "get_git_tag_versions", lambda repo_root=tmp_path: ["1.1.0"])

    errors = doc_sync.run_checks()
    assert any("out of date" in error for error in errors)


# ---------------------------------------------------------------------------
# _format_source_name - unknown source (line 67, branch [65,67])
# ---------------------------------------------------------------------------


def test_format_source_name_unknown_source_replaces_underscores() -> None:
    """Unknown source key should have underscores replaced with spaces."""
    assert _format_source_name("some_unknown_source") == "some unknown source"


def test_format_source_name_known_source_returns_label() -> None:
    """Known source key should return its label from SOURCE_LABELS."""
    assert _format_source_name("debugbench") == "DebugBench"


# ---------------------------------------------------------------------------
# get_git_tag_versions - failure (lines 198-199, branch [198,199])
# ---------------------------------------------------------------------------


def test_get_git_tag_versions_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-zero returncode from git tag should raise RuntimeError."""

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="fatal: not a repo")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="Failed to read git tags"):
        get_git_tag_versions()


# ---------------------------------------------------------------------------
# get_git_tag_versions - parsing valid/invalid tags
# (lines 201-207, branches [198,201], [202,203], [202,207], [205,202], [205,206])
# ---------------------------------------------------------------------------


def test_get_git_tag_versions_parses_valid_and_skips_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only well-formed version tags (v1.2.3 or 1.2.3) should be extracted."""
    tag_output = "\n".join(
        [
            "v0.1.0",
            "v1.2.3",
            "some-random-tag",
            "",
            "  ",
            "2.0.0",
            "not-a-version",
            "v3.4.5",
        ]
    )

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=tag_output, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    versions = get_git_tag_versions()
    assert versions == ["0.1.0", "1.2.3", "2.0.0", "3.4.5"]


def test_get_git_tag_versions_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty git tag output should return an empty list."""

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert get_git_tag_versions() == []


# ---------------------------------------------------------------------------
# _load_project_version_regex_fallback
# (lines 221, 225-226, branches [220,221], [223,225])
# ---------------------------------------------------------------------------


def test_regex_fallback_valid_version() -> None:
    """Regex fallback should parse a well-formed [project] version."""
    text = '[project]\nname = "x"\nversion = "1.2.3"\n'
    assert _load_project_version_regex_fallback(text) == "1.2.3"


def test_regex_fallback_no_project_section_raises() -> None:
    """Missing [project] section should raise ValueError."""
    text = '[build-system]\nrequires = ["hatchling"]\n'
    with pytest.raises(ValueError, match=r"Could not parse project.version"):
        _load_project_version_regex_fallback(text)


def test_regex_fallback_bad_version_format_raises() -> None:
    """version line that doesn't match the pattern → break → ValueError."""
    text = "[project]\nversion = no_quotes\n"
    with pytest.raises(ValueError, match=r"Could not parse project.version"):
        _load_project_version_regex_fallback(text)


def test_regex_fallback_empty_version_string_raises() -> None:
    """version = "" (empty string) → match is None → break → ValueError."""
    text = '[project]\nversion = ""\n'
    with pytest.raises(ValueError, match=r"Could not parse project.version"):
        _load_project_version_regex_fallback(text)


def test_regex_fallback_whitespace_only_version_raises() -> None:
    """version = " " (whitespace only) → strips to empty → break → ValueError."""
    text = '[project]\nversion = " "\n'
    with pytest.raises(ValueError, match=r"Could not parse project.version"):
        _load_project_version_regex_fallback(text)


# ---------------------------------------------------------------------------
# _first_h2_heading - returns None (line 245, branch [242,245])
# ---------------------------------------------------------------------------


def test_first_h2_heading_returns_none_when_no_headings() -> None:
    """Changelog with no ## headings should return None."""
    assert _first_h2_heading("# Title\nSome text\nMore text\n") is None


def test_first_h2_heading_returns_first_match() -> None:
    """Should return the first ## heading found."""
    text = "# Title\n## First\n## Second\n"
    assert _first_h2_heading(text) == "## First"


# ---------------------------------------------------------------------------
# validate_changelog_consistency - missing tagged versions (line 237, branch [236,237])
# ---------------------------------------------------------------------------


def test_validate_changelog_missing_tagged_versions() -> None:
    """Missing tagged versions should produce an error listing them."""
    changelog = "# CHANGELOG\n\n## v1.0.0\n\n- Release\n"
    errors = validate_changelog_consistency(
        changelog,
        tagged_versions=["1.0.0", "1.1.0", "1.2.0"],
        project_version="1.0.0",
    )
    assert len(errors) == 1
    assert "v1.1.0" in errors[0]
    assert "v1.2.0" in errors[0]


def test_validate_changelog_project_version_already_tagged_no_error() -> None:
    """When project_version IS tagged and in changelog, no errors expected."""
    changelog = "# CHANGELOG\n\n## v1.0.0\n\n- Release\n"
    errors = validate_changelog_consistency(
        changelog,
        tagged_versions=["1.0.0"],
        project_version="1.0.0",
    )
    assert errors == []


# ---------------------------------------------------------------------------
# load_project_version - tomllib path with bad version (line 237, branch [236,237])
# ---------------------------------------------------------------------------


def test_load_project_version_tomllib_non_string_raises(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """project.version that is not a non-empty string should raise."""
    from pathlib import Path

    p = Path(str(tmp_path)) / "pyproject.toml"
    p.write_text('[project]\nname = "x"\nversion = ""\n', encoding="utf-8")

    # Ensure tomllib is available (use a fake module)
    fake_tomllib = types.ModuleType("fake_tomllib")
    fake_tomllib.loads = lambda text: {  # type: ignore[attr-defined]
        "project": {"version": ""}
    }
    monkeypatch.setattr(doc_sync, "tomllib", fake_tomllib)

    with pytest.raises(ValueError, match="non-empty string"):
        doc_sync.load_project_version(p)


# ---------------------------------------------------------------------------
# write_case_inventory_block (lines 279-286, branches [283,284], [283,285])
# ---------------------------------------------------------------------------


def test_write_case_inventory_block_updates_stale(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stale readme should be updated → (True, updated message)."""
    from pathlib import Path

    tp = Path(str(tmp_path))

    summary = InventorySummary(
        total_cases=1,
        source_counts=(("hand_crafted", 1),),
        categories=(
            CategorySummary(
                name="logic_error",
                count=1,
                difficulty_counts=(("easy", 1),),
            ),
        ),
    )
    monkeypatch.setattr(doc_sync, "build_inventory_summary", lambda: summary)

    readme_path = tp / "README.md"
    readme_path.write_text(
        f"# evals\n{BEGIN_CASE_INVENTORY_MARKER}\nstale block\n{END_CASE_INVENTORY_MARKER}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(doc_sync, "EVALS_README_PATH", readme_path)

    changed, message = write_case_inventory_block()
    assert changed is True
    assert "Updated" in message
    updated = readme_path.read_text(encoding="utf-8")
    assert BEGIN_CASE_INVENTORY_MARKER in updated


def test_write_case_inventory_block_no_change(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When block is already correct → (False, already up to date)."""
    from pathlib import Path

    tp = Path(str(tmp_path))

    # Build a known summary and its rendered block.
    summary = InventorySummary(
        total_cases=1,
        source_counts=(("hand_crafted", 1),),
        categories=(
            CategorySummary(
                name="logic_error",
                count=1,
                difficulty_counts=(("easy", 1),),
            ),
        ),
    )
    desired_block = render_case_inventory_block(summary)

    # Stub build_inventory_summary to return the fixed summary.
    monkeypatch.setattr(doc_sync, "build_inventory_summary", lambda: summary)

    readme_path = tp / "README.md"
    readme_path.write_text(f"# evals\n{desired_block}\n", encoding="utf-8")
    monkeypatch.setattr(doc_sync, "EVALS_README_PATH", readme_path)

    changed, message = write_case_inventory_block()
    assert changed is False
    assert "up to date" in message


# ---------------------------------------------------------------------------
# run_checks - extract_case_inventory_block raises ValueError
# (lines 297-298, branch [300,306])
# ---------------------------------------------------------------------------


def test_run_checks_extract_block_raises_valueerror(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ValueError from extract_case_inventory_block should be appended."""
    from pathlib import Path

    tp = Path(str(tmp_path))

    summary = InventorySummary(
        total_cases=1,
        source_counts=(("hand_crafted", 1),),
        categories=(
            CategorySummary(
                name="logic_error",
                count=1,
                difficulty_counts=(("easy", 1),),
            ),
        ),
    )
    monkeypatch.setattr(doc_sync, "build_inventory_summary", lambda: summary)

    # README WITHOUT markers → triggers ValueError
    readme_path = tp / "README.md"
    readme_path.write_text("# evals\nNo markers here.\n", encoding="utf-8")

    changelog_path = tp / "CHANGELOG.md"
    changelog_path.write_text(
        "# CHANGELOG\n\n## Unreleased (9.8.7)\n\n- stuff\n",
        encoding="utf-8",
    )

    pyproject_path = tp / "pyproject.toml"
    pyproject_path.write_text('[project]\nname = "x"\nversion = "9.8.7"\n', encoding="utf-8")

    monkeypatch.setattr(doc_sync, "EVALS_README_PATH", readme_path)
    monkeypatch.setattr(doc_sync, "CHANGELOG_PATH", changelog_path)
    monkeypatch.setattr(doc_sync, "get_git_tag_versions", lambda repo_root=tp: [])
    monkeypatch.setattr(doc_sync, "load_project_version", lambda pyproject_path=None: "9.8.7")

    errors = doc_sync.run_checks()
    assert any("markers were not found" in e for e in errors)


def test_run_checks_block_matches_no_errors(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When everything matches, run_checks returns an empty error list."""
    from pathlib import Path

    tp = Path(str(tmp_path))

    # Build a known summary and its rendered block.
    summary = InventorySummary(
        total_cases=1,
        source_counts=(("hand_crafted", 1),),
        categories=(
            CategorySummary(
                name="logic_error",
                count=1,
                difficulty_counts=(("easy", 1),),
            ),
        ),
    )
    desired_block = render_case_inventory_block(summary)

    # Stub build_inventory_summary to return the fixed summary.
    monkeypatch.setattr(doc_sync, "build_inventory_summary", lambda: summary)

    readme_path = tp / "README.md"
    readme_path.write_text(f"# evals\n{desired_block}\n", encoding="utf-8")

    changelog_path = tp / "CHANGELOG.md"
    changelog_path.write_text(
        "# CHANGELOG\n\n## Unreleased (9.8.7)\n\n- stuff\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(doc_sync, "EVALS_README_PATH", readme_path)
    monkeypatch.setattr(doc_sync, "CHANGELOG_PATH", changelog_path)
    monkeypatch.setattr(doc_sync, "get_git_tag_versions", lambda repo_root=tp: [])
    monkeypatch.setattr(doc_sync, "load_project_version", lambda pyproject_path=None: "9.8.7")

    errors = doc_sync.run_checks()
    assert errors == []


# ---------------------------------------------------------------------------
# main() (lines 321-342, branches for --write and --check)
# ---------------------------------------------------------------------------


def test_main_write_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """main(["--write"]) should call write_case_inventory_block and return 0."""
    monkeypatch.setattr(
        doc_sync,
        "write_case_inventory_block",
        lambda: (True, "Updated evals/README.md case inventory block."),
    )
    assert doc_sync.main(["--write"]) == 0


def test_main_check_mode_no_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """main(["--check"]) with no errors should print success and return 0."""
    monkeypatch.setattr(doc_sync, "run_checks", lambda: [])
    assert doc_sync.main(["--check"]) == 0


def test_main_check_mode_with_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main(["--check"]) with errors should print to stderr and return 1."""
    monkeypatch.setattr(doc_sync, "run_checks", lambda: ["error one", "error two"])
    result = doc_sync.main(["--check"])
    assert result == 1
    captured = capsys.readouterr()
    assert "error one" in captured.err
    assert "error two" in captured.err
