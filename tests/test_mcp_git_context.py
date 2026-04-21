"""Tests for MCP git context evidence collection and commit-range parsing."""

from __future__ import annotations

import pytest

from llmdebug.mcp_git_context import collect_git_evidence_refs, parse_max_commits


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (5, 5),
        ("3", 3),
        (0, 1),
        (-99, 1),
        (999, 50),
    ],
)
def test_parse_max_commits_clamps_to_supported_range(raw: object, expected: int) -> None:
    assert parse_max_commits(raw) == expected


def test_parse_max_commits_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Invalid max_commits value"):
        parse_max_commits("abc")


def test_collect_git_evidence_refs_extracts_head_and_crash_refs() -> None:
    enhanced = {
        "head": {"commit": "abc1234", "branch": "main"},
        "crash_file": {
            "file_rel": "src/app.py",
            "blame": {"commit": "def5678"},
        },
    }
    assert collect_git_evidence_refs(enhanced) == [
        "git_head_commit:abc1234",
        "git_branch:main",
        "git_crash_file:src/app.py",
        "git_blame_commit:def5678",
    ]


def test_collect_git_evidence_refs_ignores_malformed_sections() -> None:
    enhanced = {"head": "not-a-dict", "crash_file": {"blame": "not-a-dict"}}
    assert collect_git_evidence_refs(enhanced) == []


# ---------------------------------------------------------------------------
# Partial branch coverage — lines 27->29, 29->32, 33->45, 42->45
# ---------------------------------------------------------------------------


class TestCollectGitEvidenceRefsBranches:
    """Cover conditional branches for empty/missing fields."""

    def test_empty_dict_returns_empty(self) -> None:
        """Both head and crash_file missing entirely."""
        assert collect_git_evidence_refs({}) == []

    def test_head_empty_commit_and_branch(self) -> None:
        """Branch 27->29 + 29->32: head present but commit/branch are empty strings."""
        enhanced = {"head": {"commit": "", "branch": ""}}
        assert collect_git_evidence_refs(enhanced) == []

    def test_head_commit_only(self) -> None:
        """Branch 29->32: commit present, branch empty → only commit ref."""
        enhanced = {"head": {"commit": "abc", "branch": ""}}
        assert collect_git_evidence_refs(enhanced) == ["git_head_commit:abc"]

    def test_head_branch_only(self) -> None:
        """Branch 27->29: commit empty, branch present → only branch ref."""
        enhanced = {"head": {"commit": "", "branch": "main"}}
        assert collect_git_evidence_refs(enhanced) == ["git_branch:main"]

    def test_head_whitespace_only_values(self) -> None:
        """Whitespace-only values are stripped to empty → no refs."""
        enhanced = {"head": {"commit": "  ", "branch": " \t "}}
        assert collect_git_evidence_refs(enhanced) == []

    def test_crash_file_not_dict(self) -> None:
        """Branch 33->45: crash_file is not a dict → skip crash file block."""
        enhanced = {"crash_file": "string-not-dict"}
        assert collect_git_evidence_refs(enhanced) == []

    def test_crash_file_missing(self) -> None:
        """Branch 33->45: crash_file key absent → skip crash file block."""
        enhanced = {"head": {"commit": "abc", "branch": "main"}}
        refs = collect_git_evidence_refs(enhanced)
        assert "git_crash_file:" not in " ".join(refs)
        assert "git_head_commit:abc" in refs

    def test_crash_file_empty_file_rel(self) -> None:
        """crash_file present but file_rel is empty → no file ref."""
        enhanced = {"crash_file": {"file_rel": "", "blame": {"commit": "xyz"}}}
        refs = collect_git_evidence_refs(enhanced)
        assert "git_crash_file:" not in " ".join(refs)
        assert "git_blame_commit:xyz" in refs

    def test_blame_commit_empty(self) -> None:
        """Branch 42->45: blame present but commit is empty → no blame ref."""
        enhanced = {"crash_file": {"file_rel": "f.py", "blame": {"commit": ""}}}
        refs = collect_git_evidence_refs(enhanced)
        assert refs == ["git_crash_file:f.py"]

    def test_blame_missing(self) -> None:
        """blame key absent from crash_file → no blame ref."""
        enhanced = {"crash_file": {"file_rel": "f.py"}}
        refs = collect_git_evidence_refs(enhanced)
        assert refs == ["git_crash_file:f.py"]

    @pytest.mark.parametrize(
        ("head", "crash_file", "expected_count"),
        [
            (None, None, 0),
            ({"commit": "a", "branch": "b"}, None, 2),
            (None, {"file_rel": "f.py"}, 1),
            (None, {"file_rel": "f.py", "blame": {"commit": "c"}}, 2),
        ],
    )
    def test_partial_fields_combinations(
        self,
        head: dict[str, str] | None,
        crash_file: dict[str, object] | None,
        expected_count: int,
    ) -> None:
        enhanced: dict[str, object] = {}
        if head is not None:
            enhanced["head"] = head
        if crash_file is not None:
            enhanced["crash_file"] = crash_file
        refs = collect_git_evidence_refs(enhanced)
        assert len(refs) == expected_count
