"""Tests for coverage-guided snapshot enrichment."""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from llmdebug.coverage_context import (
    _extract_file_coverage,
    _match_file,
    extract_coverage_from_plugin,
    get_coverage_for_frames,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockCoverageData:
    """Minimal stand-in for coverage.CoverageData."""

    def __init__(
        self,
        files: set[str] | None = None,
        lines_map: dict[str, list[int]] | None = None,
    ) -> None:
        self._files = files or set()
        self._lines_map = lines_map or {}

    def measured_files(self) -> set[str]:
        return self._files

    def lines(self, path: str) -> list[int] | None:
        return self._lines_map.get(path)


class MockCoverage:
    """Minimal stand-in for coverage.Coverage."""

    def __init__(
        self,
        data: MockCoverageData | None = None,
        analysis2_map: dict[str, tuple[Any, ...]] | None = None,
        branch_stats_map: dict[str, dict[int, tuple[int, int]]] | None = None,
        *,
        analysis2_error: Exception | None = None,
        get_data_error: Exception | None = None,
        has_branch_stats: bool = True,
    ) -> None:
        self._data = data or MockCoverageData()
        self._analysis2_map = analysis2_map or {}
        self._branch_stats_map = branch_stats_map or {}
        self._analysis2_error = analysis2_error
        self._get_data_error = get_data_error
        self._has_branch_stats = has_branch_stats
        self.save_called = False

    def get_data(self) -> MockCoverageData:
        if self._get_data_error:
            raise self._get_data_error
        return self._data

    def analysis2(self, path: str) -> tuple[Any, ...]:
        if self._analysis2_error:
            raise self._analysis2_error
        if path in self._analysis2_map:
            return self._analysis2_map[path]
        raise KeyError(f"No analysis for {path}")

    def branch_stats(self) -> dict[str, dict[int, tuple[int, int]]]:
        if not self._has_branch_stats:
            raise AttributeError("no branch_stats")
        return self._branch_stats_map

    def save(self) -> None:
        self.save_called = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_cov() -> MockCoverage:
    """Coverage mock with two small files."""
    data = MockCoverageData(
        files={"/src/app.py", "/src/utils.py"},
        lines_map={
            "/src/app.py": [1, 2, 3, 5, 7],
            "/src/utils.py": [1, 4, 6],
        },
    )
    analysis = {
        # (filename, executed, excluded, missing, formatted_missing)
        "/src/app.py": ("/src/app.py", [1, 2, 3, 5], [], [7, 9], "7, 9"),
        "/src/utils.py": ("/src/utils.py", [1, 4], [], [6], "6"),
    }
    return MockCoverage(data=data, analysis2_map=analysis)


# ---------------------------------------------------------------------------
# Tests for _match_file
# ---------------------------------------------------------------------------


class TestMatchFile:
    def test_exact_match(self) -> None:
        measured = {"/src/app.py", "/src/utils.py"}
        assert _match_file(measured, "/src/app.py") == "/src/app.py"

    def test_realpath_match(self, tmp_path: Any) -> None:
        real_file = tmp_path / "real.py"
        real_file.touch()
        link = tmp_path / "link.py"
        link.symlink_to(real_file)

        measured = {str(real_file)}
        result = _match_file(measured, str(link))
        assert result == str(real_file)

    def test_realpath_match_reverse(self, tmp_path: Any) -> None:
        """Measured file is a symlink, target is the real path."""
        real_file = tmp_path / "real.py"
        real_file.touch()
        link = tmp_path / "link.py"
        link.symlink_to(real_file)

        measured = {str(link)}
        result = _match_file(measured, str(real_file))
        # The function iterates measured files and checks realpath match
        assert result == str(link)

    def test_no_match_returns_none(self) -> None:
        measured = {"/src/app.py"}
        assert _match_file(measured, "/other/module.py") is None

    def test_empty_measured_set(self) -> None:
        assert _match_file(set(), "/src/app.py") is None


# ---------------------------------------------------------------------------
# Tests for _extract_file_coverage
# ---------------------------------------------------------------------------


class TestExtractFileCoverage:
    def test_normal_case(self, simple_cov: MockCoverage) -> None:
        result = _extract_file_coverage(simple_cov, "/src/app.py")
        assert result is not None
        assert result["executed_lines"] == [1, 2, 3, 5]
        assert result["missing_lines"] == [7, 9]
        assert result["line_rate"] == round(4 / 6, 4)

    def test_large_file_summary_only(self) -> None:
        """Files exceeding max_lines get counts only, no line lists."""
        # 300 executed + 100 missing = 400 > max_lines=200
        executed = list(range(1, 301))
        missing = list(range(301, 401))
        analysis = {
            "/big.py": ("/big.py", executed, [], missing, "301-400"),
        }
        cov = MockCoverage(
            data=MockCoverageData(files={"/big.py"}),
            analysis2_map=analysis,
        )
        result = _extract_file_coverage(cov, "/big.py", max_lines=200)
        assert result is not None
        assert "executed_lines" not in result
        assert "missing_lines" not in result
        assert result["executable_count"] == 400
        assert result["executed_count"] == 300
        assert result["missing_count"] == 100
        assert result["line_rate"] == round(300 / 400, 4)

    def test_analysis2_failure_falls_back_to_lines(self) -> None:
        """When analysis2 raises, fall back to CoverageData.lines()."""
        data = MockCoverageData(
            files={"/src/app.py"},
            lines_map={"/src/app.py": [3, 1, 5]},
        )
        cov = MockCoverage(
            data=data,
            analysis2_error=RuntimeError("analysis failed"),
        )
        result = _extract_file_coverage(cov, "/src/app.py")
        assert result is not None
        # Fallback has executed lines sorted, but missing coverage is unknown.
        assert result["executed_lines"] == [1, 3, 5]
        assert "missing_lines" not in result
        assert "line_rate" not in result
        assert result["line_rate_unknown"] is True

    def test_analysis2_failure_large_file_summary_omits_line_rate(self) -> None:
        """Summary fallback should report executed counts without a synthetic rate."""
        executed = list(range(1, 301))
        data = MockCoverageData(
            files={"/big.py"},
            lines_map={"/big.py": executed},
        )
        cov = MockCoverage(
            data=data,
            analysis2_error=RuntimeError("analysis failed"),
        )
        result = _extract_file_coverage(cov, "/big.py", max_lines=200)
        assert result is not None
        assert result["executed_count"] == 300
        assert result["executable_count"] == 300
        assert "missing_count" not in result
        assert "line_rate" not in result
        assert result["line_rate_unknown"] is True

    def test_complete_failure_returns_none(self) -> None:
        """When both analysis2 and get_data raise, return None."""
        cov = MockCoverage(
            analysis2_error=RuntimeError("analysis failed"),
            get_data_error=RuntimeError("get_data failed"),
        )
        result = _extract_file_coverage(cov, "/src/app.py")
        assert result is None

    def test_fallback_lines_returns_none(self) -> None:
        """When analysis2 fails and lines() returns None, return None."""
        data = MockCoverageData(
            files={"/src/app.py"},
            lines_map={},  # lines() returns None for unknown file
        )
        cov = MockCoverage(
            data=data,
            analysis2_error=RuntimeError("analysis failed"),
        )
        result = _extract_file_coverage(cov, "/src/app.py")
        assert result is None

    def test_branch_stats_included_when_partial(self) -> None:
        """Partial branches appear in result."""
        analysis = {
            "/src/app.py": ("/src/app.py", [1, 2, 3], [], [5], "5"),
        }
        branch_stats = {
            "/src/app.py": {
                2: (2, 1),  # partial: taken 1 of 2
                3: (2, 2),  # fully covered: taken 2 of 2
            },
        }
        cov = MockCoverage(
            data=MockCoverageData(files={"/src/app.py"}),
            analysis2_map=analysis,
            branch_stats_map=branch_stats,
        )
        result = _extract_file_coverage(cov, "/src/app.py")
        assert result is not None
        assert "partial_branches" in result
        assert result["partial_branches"] == {"2": [2, 1]}
        # Line 3 is fully covered, so not included
        assert "3" not in result["partial_branches"]

    def test_branch_stats_omitted_when_no_partials(self) -> None:
        """No partial_branches key when all branches are fully covered."""
        analysis = {
            "/src/app.py": ("/src/app.py", [1, 2], [], [], ""),
        }
        branch_stats = {
            "/src/app.py": {
                2: (2, 2),  # fully covered
            },
        }
        cov = MockCoverage(
            data=MockCoverageData(files={"/src/app.py"}),
            analysis2_map=analysis,
            branch_stats_map=branch_stats,
        )
        result = _extract_file_coverage(cov, "/src/app.py")
        assert result is not None
        assert "partial_branches" not in result

    def test_no_branch_stats_method(self) -> None:
        """Old coverage version without branch_stats is handled gracefully."""
        analysis = {
            "/src/app.py": ("/src/app.py", [1, 2], [], [3], "3"),
        }
        cov = MockCoverage(
            data=MockCoverageData(files={"/src/app.py"}),
            analysis2_map=analysis,
            has_branch_stats=False,
        )
        result = _extract_file_coverage(cov, "/src/app.py")
        assert result is not None
        assert "partial_branches" not in result

    def test_zero_total_lines(self) -> None:
        """Edge case: file with no executable lines."""
        analysis = {
            "/empty.py": ("/empty.py", [], [], [], ""),
        }
        cov = MockCoverage(
            data=MockCoverageData(files={"/empty.py"}),
            analysis2_map=analysis,
        )
        result = _extract_file_coverage(cov, "/empty.py")
        assert result is not None
        assert result["line_rate"] == 1.0


# ---------------------------------------------------------------------------
# Tests for get_coverage_for_frames
# ---------------------------------------------------------------------------


class TestGetCoverageForFrames:
    def test_returns_coverage_for_matching_files(self, simple_cov: MockCoverage) -> None:
        result = get_coverage_for_frames(simple_cov, ["/src/app.py", "/src/utils.py"])
        assert result is not None
        assert "/src/app.py" in result
        assert "/src/utils.py" in result
        assert result["/src/app.py"]["executed_lines"] == [1, 2, 3, 5]

    def test_large_file_summary_only(self) -> None:
        """Files exceeding max_lines get summary-only in frame results."""
        executed = list(range(1, 301))
        missing = list(range(301, 401))
        data = MockCoverageData(files={"/big.py"})
        analysis = {"/big.py": ("/big.py", executed, [], missing, "301-400")}
        cov = MockCoverage(data=data, analysis2_map=analysis)

        result = get_coverage_for_frames(cov, ["/big.py"], max_lines=200)
        assert result is not None
        file_cov = result["/big.py"]
        assert "executed_lines" not in file_cov
        assert "missing_lines" not in file_cov
        assert file_cov["executable_count"] == 400
        assert file_cov["executed_count"] == 300
        assert file_cov["missing_count"] == 100

    def test_unmatched_files_skipped(self, simple_cov: MockCoverage) -> None:
        result = get_coverage_for_frames(simple_cov, ["/nonexistent/module.py"])
        assert result is None

    def test_branch_stats_included(self) -> None:
        """Branch stats propagate through to frame-level results."""
        analysis = {
            "/src/app.py": ("/src/app.py", [1, 2, 3], [], [5], "5"),
        }
        branch_stats = {
            "/src/app.py": {10: (3, 1)},
        }
        data = MockCoverageData(files={"/src/app.py"})
        cov = MockCoverage(
            data=data,
            analysis2_map=analysis,
            branch_stats_map=branch_stats,
        )
        result = get_coverage_for_frames(cov, ["/src/app.py"])
        assert result is not None
        assert "partial_branches" in result["/src/app.py"]
        assert result["/src/app.py"]["partial_branches"] == {"10": [3, 1]}

    def test_analysis2_failure_falls_back(self) -> None:
        """Graceful fallback when analysis2 raises for a frame file."""
        data = MockCoverageData(
            files={"/src/app.py"},
            lines_map={"/src/app.py": [2, 4, 6]},
        )
        cov = MockCoverage(
            data=data,
            analysis2_error=RuntimeError("analysis failed"),
        )
        result = get_coverage_for_frames(cov, ["/src/app.py"])
        assert result is not None
        assert result["/src/app.py"]["executed_lines"] == [2, 4, 6]
        assert "missing_lines" not in result["/src/app.py"]
        assert "line_rate" not in result["/src/app.py"]
        assert result["/src/app.py"]["line_rate_unknown"] is True

    def test_returns_none_when_no_measured_files(self) -> None:
        data = MockCoverageData(files=set())
        cov = MockCoverage(data=data)
        result = get_coverage_for_frames(cov, ["/src/app.py"])
        assert result is None

    def test_returns_none_when_get_data_raises(self) -> None:
        cov = MockCoverage(get_data_error=RuntimeError("no data"))
        result = get_coverage_for_frames(cov, ["/src/app.py"])
        assert result is None

    def test_deduplicates_frame_files(self, simple_cov: MockCoverage) -> None:
        """Duplicate frame files should only produce one coverage entry."""
        result = get_coverage_for_frames(
            simple_cov,
            ["/src/app.py", "/src/app.py", "/src/app.py"],
        )
        assert result is not None
        assert len(result) == 1
        assert "/src/app.py" in result

    def test_mixed_matched_and_unmatched(self, simple_cov: MockCoverage) -> None:
        """Only matched files appear in the result."""
        result = get_coverage_for_frames(
            simple_cov,
            ["/src/app.py", "/nonexistent.py", "/src/utils.py"],
        )
        assert result is not None
        assert len(result) == 2
        assert "/nonexistent.py" not in result


# ---------------------------------------------------------------------------
# Tests for extract_coverage_from_plugin
# ---------------------------------------------------------------------------


def _make_tb_with_frames(filenames: list[str]) -> types.TracebackType:
    """Build a real traceback with controlled filenames.

    Compiles code objects with the desired filenames so that
    traceback.extract_tb returns frames we control.
    """
    tb: types.TracebackType | None = None
    for fname in reversed(filenames):
        code_str = "raise RuntimeError('tb')"
        code = compile(code_str, fname, "exec")
        try:
            exec(code, {})
        except RuntimeError:
            import sys

            inner_tb = sys.exc_info()[2]
            assert inner_tb is not None
            tb = inner_tb
    return tb  # type: ignore[return-value]


def _make_item_mock(
    cov_plugin: Any = None,
    *,
    getplugin_error: Exception | None = None,
) -> MagicMock:
    """Create a pytest Item mock with pluginmanager."""
    item = MagicMock()
    if getplugin_error:
        item.config.pluginmanager.getplugin.side_effect = getplugin_error
    else:
        item.config.pluginmanager.getplugin.return_value = cov_plugin
    return item


class TestExtractCoverageFromPlugin:
    def test_returns_none_when_env_var_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "false")
        warnings: list[str] = []
        result = extract_coverage_from_plugin(MagicMock(), None, warnings)
        assert result is None
        assert warnings == []

    def test_returns_none_when_env_var_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "0")
        warnings: list[str] = []
        result = extract_coverage_from_plugin(MagicMock(), None, warnings)
        assert result is None

    def test_returns_none_when_env_var_no(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "no")
        warnings: list[str] = []
        result = extract_coverage_from_plugin(MagicMock(), None, warnings)
        assert result is None

    def test_returns_none_when_cov_plugin_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "true")
        item = _make_item_mock(cov_plugin=None)
        warnings: list[str] = []
        result = extract_coverage_from_plugin(item, None, warnings)
        assert result is None
        assert warnings == []

    def test_returns_none_when_getplugin_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "true")
        item = _make_item_mock(getplugin_error=RuntimeError("no plugin"))
        warnings: list[str] = []
        result = extract_coverage_from_plugin(item, None, warnings)
        assert result is None

    def test_returns_none_when_cov_controller_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "true")
        plugin = MagicMock()
        plugin.cov_controller = None
        item = _make_item_mock(cov_plugin=plugin)
        warnings: list[str] = []
        result = extract_coverage_from_plugin(item, MagicMock(), warnings)
        assert result is None
        assert warnings == []

    def test_returns_none_when_cov_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "true")
        plugin = MagicMock()
        plugin.cov_controller = MagicMock()
        plugin.cov_controller.cov = None
        item = _make_item_mock(cov_plugin=plugin)
        warnings: list[str] = []
        result = extract_coverage_from_plugin(item, MagicMock(), warnings)
        assert result is None

    def test_appends_warning_on_cov_controller_attr_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "true")
        plugin = MagicMock(spec=[])  # No attributes at all
        item = _make_item_mock(cov_plugin=plugin)
        warnings: list[str] = []
        result = extract_coverage_from_plugin(item, MagicMock(), warnings)
        assert result is None
        assert len(warnings) == 1
        assert "coverage extraction failed" in warnings[0]

    def test_returns_coverage_data_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "true")

        # Build a real traceback pointing to a known filename
        tb = _make_tb_with_frames(["/src/app.py"])

        # Build mock coverage
        data = MockCoverageData(
            files={"/src/app.py"},
            lines_map={"/src/app.py": [1, 2, 3]},
        )
        analysis = {
            "/src/app.py": ("/src/app.py", [1, 2], [], [3], "3"),
        }
        mock_cov = MockCoverage(data=data, analysis2_map=analysis)

        plugin = MagicMock()
        plugin.cov_controller = MagicMock()
        plugin.cov_controller.cov = mock_cov

        item = _make_item_mock(cov_plugin=plugin)
        warnings: list[str] = []

        result = extract_coverage_from_plugin(item, tb, warnings)
        assert result is not None
        assert "/src/app.py" in result
        assert result["/src/app.py"]["executed_lines"] == [1, 2]
        assert result["/src/app.py"]["missing_lines"] == [3]
        assert warnings == []
        assert mock_cov.save_called

    def test_save_failure_is_non_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cov.save() failure should not prevent data extraction."""
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "true")

        tb = _make_tb_with_frames(["/src/app.py"])

        data = MockCoverageData(
            files={"/src/app.py"},
            lines_map={"/src/app.py": [1, 2]},
        )
        analysis = {
            "/src/app.py": ("/src/app.py", [1, 2], [], [], ""),
        }

        class FailingSaveCoverage(MockCoverage):
            def save(self) -> None:
                raise OSError("disk full")

        mock_cov = FailingSaveCoverage(data=data, analysis2_map=analysis)
        plugin = MagicMock()
        plugin.cov_controller = MagicMock()
        plugin.cov_controller.cov = mock_cov

        item = _make_item_mock(cov_plugin=plugin)
        warnings: list[str] = []

        result = extract_coverage_from_plugin(item, tb, warnings)
        assert result is not None
        assert "/src/app.py" in result

    def test_appends_warning_on_extraction_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When get_coverage_for_frames raises, warning is appended."""
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "true")

        tb = _make_tb_with_frames(["/src/app.py"])

        # Coverage that explodes on get_data during frame extraction
        mock_cov = MockCoverage(
            get_data_error=RuntimeError("boom"),
        )

        plugin = MagicMock()
        plugin.cov_controller = MagicMock()
        plugin.cov_controller.cov = mock_cov

        item = _make_item_mock(cov_plugin=plugin)
        warnings: list[str] = []

        result = extract_coverage_from_plugin(item, tb, warnings)
        # get_coverage_for_frames returns None gracefully, no warning
        assert result is None

    def test_returns_none_when_no_frame_files(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty traceback produces None."""
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "true")

        mock_cov = MockCoverage(
            data=MockCoverageData(files={"/src/app.py"}),
        )
        plugin = MagicMock()
        plugin.cov_controller = MagicMock()
        plugin.cov_controller.cov = mock_cov

        item = _make_item_mock(cov_plugin=plugin)
        warnings: list[str] = []

        # Pass None as tb - extract_tb(None) returns empty list
        result = extract_coverage_from_plugin(item, None, warnings)
        assert result is None

    def test_env_var_defaults_to_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When LLMDEBUG_INCLUDE_COVERAGE is not set, coverage is enabled."""
        monkeypatch.delenv("LLMDEBUG_INCLUDE_COVERAGE", raising=False)

        # Plugin not found, but the env-var check should not block
        item = _make_item_mock(cov_plugin=None)
        warnings: list[str] = []
        result = extract_coverage_from_plugin(item, None, warnings)
        # Returns None because plugin is None, not because env var blocked it
        assert result is None

    def test_env_var_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FALSE and False should both disable coverage."""
        for value in ("FALSE", "False", "NO", "Off"):
            monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", value)
            warnings: list[str] = []
            result = extract_coverage_from_plugin(MagicMock(), None, warnings)
            assert result is None


# ---------------------------------------------------------------------------
# Coverage gap tests — lines 27-28, 36-37, 224-226, 233-235
# ---------------------------------------------------------------------------


class TestMatchFileRealpathError:
    """Lines 27-28 and 36-37: OSError from os.path.realpath in _match_file."""

    def test_realpath_raises_oserror_on_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Line 27: except (OSError, ValueError) when realpath fails for target."""
        original_realpath = __import__("os").path.realpath

        def _broken_realpath(p: str, **kwargs: Any) -> str:
            if p == "/broken/target.py":
                raise OSError("bad path")
            return original_realpath(p, **kwargs)

        monkeypatch.setattr("os.path.realpath", _broken_realpath)
        measured = {"/src/app.py"}
        result = _match_file(measured, "/broken/target.py")
        assert result is None

    def test_realpath_raises_oserror_on_measured_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Line 36-37: except (OSError, ValueError) when realpath fails for measured file."""
        call_count = 0
        original_realpath = __import__("os").path.realpath

        def _selective_broken_realpath(p: str, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            # First call is for target (succeeds), subsequent for measured files (fail)
            if call_count > 1:
                raise OSError("measured file broken")
            return original_realpath(p, **kwargs)

        monkeypatch.setattr("os.path.realpath", _selective_broken_realpath)
        measured = {"/unmatchable/file.py"}
        result = _match_file(measured, "/other/target.py")
        assert result is None

    def test_realpath_raises_valueerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 27-28: ValueError variant."""

        def _raise_value_error(p: str, **kwargs: Any) -> str:
            raise ValueError("embedded null byte")

        monkeypatch.setattr("os.path.realpath", _raise_value_error)
        result = _match_file({"/src/app.py"}, "/some/target.py")
        assert result is None


class TestExtractCoverageFromPluginExceptions:
    """Lines 224-226, 233-235: exception handling in extract_coverage_from_plugin."""

    def test_extract_tb_raises_appends_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 224-226: exception in tb_module.extract_tb."""
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "true")

        import traceback as tb_mod

        monkeypatch.setattr(
            tb_mod, "extract_tb", lambda tb: (_ for _ in ()).throw(RuntimeError("extract failed"))
        )

        mock_cov = MockCoverage(
            data=MockCoverageData(files={"/src/app.py"}),
        )
        plugin = MagicMock()
        plugin.cov_controller = MagicMock()
        plugin.cov_controller.cov = mock_cov

        item = _make_item_mock(cov_plugin=plugin)
        warnings_list: list[str] = []
        result = extract_coverage_from_plugin(item, "fake_tb", warnings_list)
        assert result is None
        assert any("coverage frame extraction failed" in w for w in warnings_list)

    def test_get_coverage_for_frames_raises_appends_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lines 233-235: exception in get_coverage_for_frames."""
        monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "true")

        tb = _make_tb_with_frames(["/src/app.py"])

        mock_cov = MockCoverage(
            data=MockCoverageData(files={"/src/app.py"}),
            analysis2_map={
                "/src/app.py": ("/src/app.py", [1, 2], [], [3], "3"),
            },
        )
        plugin = MagicMock()
        plugin.cov_controller = MagicMock()
        plugin.cov_controller.cov = mock_cov

        item = _make_item_mock(cov_plugin=plugin)
        warnings_list: list[str] = []

        # Monkeypatch get_coverage_for_frames to raise
        monkeypatch.setattr(
            "llmdebug.coverage_context.get_coverage_for_frames",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("data extraction boom")),
        )
        result = extract_coverage_from_plugin(item, tb, warnings_list)
        assert result is None
        assert any("coverage data extraction failed" in w for w in warnings_list)
