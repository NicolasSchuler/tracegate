"""Pytest plugin for automatic snapshot capture on test failures."""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Callable, Generator
from typing import Any, cast

import pytest

from .capture import capture_exception
from .config import GitDirtyMode, OutputFormat, SnapshotConfig
from .redaction_policy import normalize_redaction_profile, resolve_redaction_policy
from .serialize import safe_repr, truncate_str

# Stash key for tracking test outcomes across reruns (flaky detection)
_flaky_key = pytest.StashKey[list[str]]()

# Internal pytest fixtures that are noisy/uninformative in snapshots
_FIXTURE_EXCLUDE = frozenset(
    {
        "request",
        "pytestconfig",
        "_session_faker",
        "capsys",
        "capfd",
        "capfdbinary",
        "capsysbinary",
        "cache",
        "record_property",
        "record_testsuite_property",
        "record_xml_attribute",
        "recwarn",
        "monkeypatch",
        "doctest_namespace",
    }
)

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _parse_env_bool(raw: str) -> bool | None:
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return None


def _env_bool(name: str, default: str = "") -> bool:
    """Read a boolean from an environment variable.

    Treats ``"1"``, ``"true"``, ``"yes"``, ``"on"`` (case-insensitive) as
    True; everything else as False.
    """
    raw = os.getenv(name)
    candidate = default if raw is None else raw
    parsed = _parse_env_bool(candidate)
    return parsed if parsed is not None else False


def _env_optional_bool(name: str) -> bool | None:
    """Read an optional boolean from an environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return None
    return _parse_env_bool(raw)


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """Read an integer from an environment variable, falling back to *default*."""
    from ._coerce import read_env_int

    return read_env_int(name, default, minimum=minimum)


def pytest_configure(config: pytest.Config) -> None:
    """Register the no_snapshot marker."""
    config.addinivalue_line(
        "markers",
        "no_snapshot: disable llmdebug snapshot capture for this test",
    )


def _record_call_outcome(item: pytest.Item, report: pytest.TestReport) -> None:
    if report.when != "call":
        return
    try:
        outcomes_list = item.stash.setdefault(_flaky_key, [])
        outcomes_list.append(report.outcome)
    except Exception:  # nosec B110 — best-effort outcome record
        pass  # Best effort — stash may not be available in all pytest versions


def _parse_include_modules(value: str) -> tuple[str, ...] | None:
    """Parse comma-separated module filter from env var."""
    modules = tuple(module.strip() for module in value.split(",") if module.strip())
    return modules or None


def _resolve_output_format(raw_value: str) -> OutputFormat:
    """Map raw string to OutputFormat literal."""
    if raw_value == "json":
        return "json"
    if raw_value == "toon":
        return "toon"
    return "json_compact"


def _resolve_git_dirty_mode(raw_value: str) -> GitDirtyMode:
    """Map raw string to GitDirtyMode literal, raising on invalid."""
    if raw_value == "always":
        return "always"
    if raw_value == "cached":
        return "cached"
    if raw_value == "off":
        return "off"
    raise ValueError(
        f"LLMDEBUG_GIT_DIRTY_MODE must be one of: always, cached, off (got {raw_value!r})"
    )


def _build_snapshot_config_from_env(debug: bool) -> SnapshotConfig:
    """Construct a SnapshotConfig from ``LLMDEBUG_*`` environment variables.

    Boolean vars accept 1/true/yes/on (case-insensitive).  Missing vars
    use defaults matching the library's documented API.
    """
    include_modules_env = os.getenv("LLMDEBUG_INCLUDE_MODULES", "")
    include_modules = _parse_include_modules(include_modules_env)

    max_exception_depth = _env_int("LLMDEBUG_MAX_EXCEPTION_DEPTH", 5, minimum=1)
    include_git = _env_bool("LLMDEBUG_INCLUDE_GIT", "true")
    git_dirty_mode_raw = os.getenv("LLMDEBUG_GIT_DIRTY_MODE", "always").strip().lower()
    try:
        git_dirty_mode = _resolve_git_dirty_mode(git_dirty_mode_raw)
    except ValueError as e:
        git_dirty_mode = "always"
        if debug:
            sys.stderr.write(f"llmdebug: {e}\n")
    include_array_stats = _env_bool("LLMDEBUG_ARRAY_STATS")
    include_args = _env_bool("LLMDEBUG_INCLUDE_ARGS", "true")
    categorize_errors = _env_bool("LLMDEBUG_CATEGORIZE_ERRORS", "true")
    include_async_context = _env_bool("LLMDEBUG_ASYNC_CONTEXT", "true")
    capture_logs = _env_bool("LLMDEBUG_CAPTURE_LOGS")
    log_max_records = _env_int("LLMDEBUG_LOG_MAX_RECORDS", 20, minimum=0)
    include_coverage = _env_bool("LLMDEBUG_INCLUDE_COVERAGE", "true")

    profile_raw = os.getenv("LLMDEBUG_REDACTION_PROFILE", "ci")
    try:
        redaction_profile = normalize_redaction_profile(profile_raw)
    except ValueError as e:
        redaction_profile = None
        if debug:
            sys.stderr.write(f"llmdebug: {e}\n")

    policy = resolve_redaction_policy(
        redaction_profile=redaction_profile,
        redact=None,
        redact_keys=_env_optional_bool("LLMDEBUG_REDACT_KEYS"),
        redact_traceback=_env_optional_bool("LLMDEBUG_REDACT_TRACEBACK"),
        redact_exception_strings=_env_optional_bool("LLMDEBUG_REDACT_EXCEPTION_STRINGS"),
        legacy_redact=(),
        legacy_redact_keys=False,
        legacy_redact_traceback=False,
        legacy_redact_exception_strings=False,
    )

    output_format = _resolve_output_format(os.getenv("LLMDEBUG_OUTPUT_FORMAT", "json_compact"))
    return SnapshotConfig(
        out_dir=".llmdebug",
        debug=debug,
        include_modules=include_modules,
        max_exception_depth=max_exception_depth,
        redaction_profile=policy.profile,
        redact=policy.patterns,
        redact_keys=policy.redact_keys,
        redact_traceback=policy.redact_traceback,
        redact_exception_strings=policy.redact_exception_strings,
        include_git=include_git,
        git_dirty_mode=git_dirty_mode,
        include_array_stats=include_array_stats,
        include_args=include_args,
        categorize_errors=categorize_errors,
        include_async_context=include_async_context,
        capture_logs=capture_logs,
        log_max_records=log_max_records,
        include_coverage=include_coverage,
        output_format=output_format,
    )


def _build_base_pytest_context(
    item: pytest.Item,
    report: pytest.TestReport,
    cfg: SnapshotConfig,
) -> tuple[dict[str, object], Any]:
    """Build the baseline pytest context dict (nodeid, params, markers)."""
    params = None
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        try:
            params = {k: safe_repr(v, cfg) for k, v in callspec.params.items()}
        except Exception:
            params = None

    context: dict[str, object] = {
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "when": report.when,
        "longrepr": _truncate_optional_text(getattr(report, "longreprtext", None), cfg.max_str),
        "capstdout": _truncate_optional_text(getattr(report, "capstdout", None), cfg.max_str),
        "capstderr": _truncate_optional_text(getattr(report, "capstderr", None), cfg.max_str),
        "params": params,
        "repro": [sys.executable, "-m", "pytest", report.nodeid, "-q"],
    }
    return context, callspec


def _capture_fixture_context(
    item: pytest.Item,
    cfg: SnapshotConfig,
    pytest_context: dict[str, object],
) -> dict[str, str]:
    """Capture active fixture names/values, excluding noisy internal fixtures."""
    fixture_values: dict[str, str] = {}
    fixture_names = cast(list[str], getattr(item, "fixturenames", []))
    func_args = cast(dict[str, Any], getattr(item, "funcargs", {}))
    for fixture_name in fixture_names:
        if fixture_name in _FIXTURE_EXCLUDE or fixture_name not in func_args:
            continue
        fixture_values[fixture_name] = safe_repr(func_args[fixture_name], cfg)
    if fixture_values:
        pytest_context["fixtures"] = fixture_values
    return fixture_values


def _capture_marker_context(
    item: pytest.Item,
    cfg: SnapshotConfig,
    pytest_context: dict[str, object],
) -> dict[str, dict[str, object] | list[dict[str, object]]]:
    """Capture pytest markers as structured dicts (skipping parametrize/usefixtures)."""
    marker_dict: dict[str, dict[str, object] | list[dict[str, object]]] = {}
    for marker in item.iter_markers():
        if marker.name in {"parametrize", "usefixtures"}:
            continue
        entry: dict[str, object] = {}
        if marker.args:
            entry["args"] = [safe_repr(arg, cfg) for arg in marker.args]
        if marker.kwargs:
            entry["kwargs"] = {k: safe_repr(v, cfg) for k, v in marker.kwargs.items()}
        existing = marker_dict.get(marker.name)
        if existing is None:
            marker_dict[marker.name] = entry
        elif isinstance(existing, list):
            existing.append(entry)
        else:
            marker_dict[marker.name] = [existing, entry]
    if marker_dict:
        pytest_context["markers"] = marker_dict
    return marker_dict


def _capture_timeout_context(
    item: pytest.Item,
    cfg: SnapshotConfig,
    pytest_context: dict[str, object],
) -> dict[str, object]:
    """Capture timeout settings from pytest-timeout if active."""
    timeout_marker = item.get_closest_marker("timeout")
    timeout_config: dict[str, object] = {}
    if timeout_marker is None:
        return timeout_config
    if timeout_marker.args:
        timeout_config["seconds"] = timeout_marker.args[0]
    if timeout_marker.kwargs:
        timeout_config["kwargs"] = {
            key: safe_repr(value, cfg) for key, value in timeout_marker.kwargs.items()
        }
    if timeout_config:
        pytest_context["timeout_config"] = timeout_config
    return timeout_config


def _capture_param_indices(callspec: Any, pytest_context: dict[str, object]) -> dict[str, int]:
    """Capture parametrize indices for reproducing specific parameter combos."""
    indices = dict(callspec.indices)
    if indices:
        pytest_context["param_indices"] = indices
    return indices


def _capture_xdist_worker(item: pytest.Item, pytest_context: dict[str, object]) -> str | None:
    """Capture xdist worker ID when running in parallel."""
    workerinput = getattr(item.config, "workerinput", None)
    if workerinput is None:
        return None
    worker_id = workerinput.get("workerid")
    if worker_id is not None:
        pytest_context["xdist_worker"] = worker_id
        return str(worker_id)
    return None


def _capture_flaky_metadata(
    item: pytest.Item, pytest_context: dict[str, object]
) -> tuple[int, bool] | None:
    """Capture rerun history for flaky test detection."""
    stash_outcomes = item.stash.get(_flaky_key, cast(list[str], []))
    if len(stash_outcomes) <= 1:
        return None
    rerun_count = len(stash_outcomes)
    flaky = "passed" in stash_outcomes and "failed" in stash_outcomes
    pytest_context["rerun_count"] = rerun_count
    pytest_context["flaky"] = flaky
    return rerun_count, flaky


def _capture_coverage_context(
    item: pytest.Item, tb: types.TracebackType | None, plugin_warnings: list[str]
) -> dict[str, Any] | None:
    """Capture coverage data for the test file from pytest-cov."""
    from .coverage_context import extract_coverage_from_plugin

    coverage_data = extract_coverage_from_plugin(item, tb, plugin_warnings)
    if coverage_data:
        return coverage_data
    return None


def _capture_context_block(
    plugin_warnings: list[str],
    warning_prefix: str,
    func: Callable[[], Any],
) -> None:
    try:
        func()
    except Exception as e:
        plugin_warnings.append(f"{warning_prefix}: {type(e).__name__}: {e}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> Generator[None, Any, None]:
    """Capture snapshot on test failure."""
    outcome = yield
    report = outcome.get_result()

    _record_call_outcome(item, report)

    # Only capture on test call failures (not setup/teardown)
    if report.when != "call" or not report.failed:
        return

    # Respect opt-out marker
    if item.get_closest_marker("no_snapshot"):
        return

    # Get exception info
    if call.excinfo is None:
        return

    exc = call.excinfo.value
    tb = call.excinfo.tb

    # Use test node ID as snapshot name
    name = item.nodeid.replace("/", "_").replace("::", "_")

    debug = _env_bool("LLMDEBUG_DEBUG")
    cfg = _build_snapshot_config_from_env(debug)

    include_fixtures = _env_bool("LLMDEBUG_INCLUDE_FIXTURES", "true")
    include_markers = _env_bool("LLMDEBUG_INCLUDE_MARKERS", "true")
    include_pytest_meta = _env_bool("LLMDEBUG_INCLUDE_PYTEST_META", "true")

    pytest_context, callspec = _build_base_pytest_context(item, report, cfg)
    _plugin_warnings: list[str] = []

    if include_fixtures:
        _capture_context_block(
            _plugin_warnings,
            "fixture capture failed",
            lambda: _capture_fixture_context(item, cfg, pytest_context),
        )

    if include_markers:
        _capture_context_block(
            _plugin_warnings,
            "marker capture failed",
            lambda: _capture_marker_context(item, cfg, pytest_context),
        )
        _capture_context_block(
            _plugin_warnings,
            "timeout config failed",
            lambda: _capture_timeout_context(item, cfg, pytest_context),
        )

    if include_pytest_meta and callspec is not None:
        _capture_context_block(
            _plugin_warnings,
            "param indices failed",
            lambda: _capture_param_indices(callspec, pytest_context),
        )

    if include_pytest_meta:
        _capture_context_block(
            _plugin_warnings,
            "xdist worker failed",
            lambda: _capture_xdist_worker(item, pytest_context),
        )
        _capture_context_block(
            _plugin_warnings,
            "flaky detection failed",
            lambda: _capture_flaky_metadata(item, pytest_context),
        )

    if cfg.include_coverage:

        def _attach_coverage() -> None:
            coverage_data = _capture_coverage_context(item, tb, _plugin_warnings)
            if coverage_data:
                pytest_context["coverage"] = coverage_data

        _capture_context_block(
            _plugin_warnings,
            "coverage enrichment failed",
            _attach_coverage,
        )

    if _plugin_warnings:
        pytest_context["_plugin_warnings"] = _plugin_warnings

    try:
        capture_exception(name, exc, tb, cfg, extra={"pytest": pytest_context})
    except Exception as capture_exc:
        if cfg.debug:
            sys.stderr.write(
                f"llmdebug: capture failed for {name}: "
                f"{type(capture_exc).__name__}: {capture_exc}\n"
            )


def _truncate_optional_text(value: object, max_len: int) -> str | None:
    if not isinstance(value, str):
        return None
    return truncate_str(value, max_len)
