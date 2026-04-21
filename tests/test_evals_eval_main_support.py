"""Tests for evals/eval_main_support.py coverage gaps.

Covers:
- _DynamicAttrProxy.__fspath__, __repr__, __str__, __truediv__
- Module-level proxy instances _REPO_ROOT and _RunResultsLock
- Delegating wrapper functions (lines 75, 79, 83, 87, 91, 95, 99, 103)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import evals.eval_main_support as _support
from evals.eval_main_support import (  # type: ignore[attr-defined]
    _REPO_ROOT,
    _DynamicAttrProxy,
    _RunResultsLock,
)


class TestDynamicAttrProxyProtocol:
    """_DynamicAttrProxy proxies path-like and string-like dunder methods."""

    def test_fspath_returns_string_for_path_attribute(self) -> None:
        # _REPO_ROOT proxies a Path, so __fspath__ should return a string
        result = _REPO_ROOT.__fspath__()
        assert isinstance(result, str)

    def test_repr_returns_string(self) -> None:
        result = repr(_REPO_ROOT)
        assert isinstance(result, str)

    def test_str_returns_string(self) -> None:
        result = str(_REPO_ROOT)
        assert isinstance(result, str)

    def test_truediv_appends_path_component(self) -> None:
        # _REPO_ROOT / "somefile" should behave like a Path division
        result = _REPO_ROOT / "somefile.txt"
        assert isinstance(result, Path)
        assert str(result).endswith("somefile.txt")

    def test_run_results_lock_repr_is_string(self) -> None:
        result = repr(_RunResultsLock)
        assert isinstance(result, str)

    def test_getattr_proxies_through_to_underlying_value(self) -> None:
        # _REPO_ROOT is a Path; Path objects have .name attribute
        name_attr = _REPO_ROOT.name
        assert isinstance(name_attr, str)

    def test_manual_proxy_with_simple_object(self) -> None:
        """Test _DynamicAttrProxy directly using a simple mock attribute on run_eval."""
        import evals.run_eval as _legacy

        # _REPO_ROOT wraps _legacy._REPO_ROOT which is a Path
        proxy = _DynamicAttrProxy("_REPO_ROOT")
        assert proxy._value() == _legacy._REPO_ROOT
        assert str(proxy) == str(_legacy._REPO_ROOT)
        assert repr(proxy) == repr(_legacy._REPO_ROOT)
        assert proxy.__fspath__() == str(_legacy._REPO_ROOT)
        assert (proxy / "x") == (_legacy._REPO_ROOT / "x")


class TestDelegatingWrapperFunctions:
    """Wrapper functions delegate to evals.run_eval (lines 75, 79, 83, 87, 91, 95, 99, 103)."""

    def test_make_patcher_delegates_to_legacy(self) -> None:
        sentinel = object()
        with patch("evals.run_eval.make_patcher", return_value=sentinel) as mock_fn:
            result = _support.make_patcher("arg1", key="val")
            mock_fn.assert_called_once_with("arg1", key="val")
            assert result is sentinel

    def test_load_cases_delegates_to_legacy(self) -> None:
        sentinel = object()
        with patch("evals.run_eval.load_cases", return_value=sentinel) as mock_fn:
            result = _support.load_cases("x", y=2)
            mock_fn.assert_called_once_with("x", y=2)
            assert result is sentinel

    def test_build_run_settings_fingerprint_delegates(self) -> None:
        sentinel = object()
        with patch(
            "evals.run_eval._build_run_settings_fingerprint", return_value=sentinel
        ) as mock_fn:
            result = _support._build_run_settings_fingerprint("a", b=1)  # type: ignore[attr-defined]
            mock_fn.assert_called_once_with("a", b=1)
            assert result is sentinel

    def test_write_resolved_run_config_lockfile_delegates(self) -> None:
        sentinel = object()
        with patch(
            "evals.run_eval._write_resolved_run_config_lockfile", return_value=sentinel
        ) as mock_fn:
            result = _support._write_resolved_run_config_lockfile("p")  # type: ignore[attr-defined]
            mock_fn.assert_called_once_with("p")
            assert result is sentinel

    def test_load_existing_run_state_delegates(self) -> None:
        sentinel = object()
        with patch("evals.run_eval._load_existing_run_state", return_value=sentinel) as mock_fn:
            result = _support._load_existing_run_state("p")  # type: ignore[attr-defined]
            mock_fn.assert_called_once_with("p")
            assert result is sentinel

    def test_repair_trailing_results_tail_delegates(self) -> None:
        sentinel = object()
        with patch(
            "evals.run_eval._repair_trailing_results_tail", return_value=sentinel
        ) as mock_fn:
            result = _support._repair_trailing_results_tail("p")  # type: ignore[attr-defined]
            mock_fn.assert_called_once_with("p")
            assert result is sentinel

    def test_check_testcontainers_runtime_available_delegates(self) -> None:
        sentinel = object()
        with patch(
            "evals.run_eval.check_testcontainers_runtime_available", return_value=sentinel
        ) as mock_fn:
            result = _support.check_testcontainers_runtime_available()
            mock_fn.assert_called_once_with()
            assert result is sentinel

    def test_cleanup_stale_eval_containers_delegates(self) -> None:
        sentinel = object()
        with patch(
            "evals.run_eval.cleanup_stale_eval_containers", return_value=sentinel
        ) as mock_fn:
            result = _support.cleanup_stale_eval_containers()
            mock_fn.assert_called_once_with()
            assert result is sentinel
