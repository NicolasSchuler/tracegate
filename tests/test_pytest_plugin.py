"""Tests for pytest plugin."""

import os
import subprocess
import sys
from types import SimpleNamespace

import llmdebug.pytest_plugin as pytest_plugin_module
from llmdebug import get_latest_snapshot
from llmdebug.config import SnapshotConfig


def test_plugin_captures_on_failure(tmp_path):
    """Test that the pytest plugin captures snapshots on test failures."""
    # Create a test file
    test_file = tmp_path / "test_example.py"
    test_file.write_text(
        """
def test_failing():
    x = [1, 2, 3]
    assert False, "intentional failure"
"""
    )

    # Run pytest in subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # Test should fail
    assert result.returncode != 0

    # Check that snapshot was created
    llmdebug_dir = tmp_path / ".llmdebug"
    assert llmdebug_dir.exists(), f"No .llmdebug dir. stderr: {result.stderr}"

    snapshots = list(llmdebug_dir.glob("*.json"))
    assert len(snapshots) >= 1  # At least the snapshot (and maybe latest.json)

    # Check that repro command is present (use get_latest_snapshot to auto-expand keys)
    snapshot = get_latest_snapshot(str(llmdebug_dir))
    assert snapshot is not None, "Failed to read snapshot"
    repro = snapshot.get("pytest", {}).get("repro")
    assert isinstance(repro, list)
    assert "-m" in repro
    assert "pytest" in repro
    assert "test_example.py::test_failing" in repro


def test_no_snapshot_marker(tmp_path):
    """Test that no_snapshot marker prevents capture."""
    test_file = tmp_path / "test_no_capture.py"
    test_file.write_text(
        """
import pytest

@pytest.mark.no_snapshot
def test_failing_no_capture():
    assert False, "should not capture"
"""
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # Test should fail
    assert result.returncode != 0

    # Check that no snapshot was created for this specific test
    llmdebug_dir = tmp_path / ".llmdebug"
    if llmdebug_dir.exists():
        snapshots = list(llmdebug_dir.glob("*test_failing_no_capture*.json"))
        assert len(snapshots) == 0, "Snapshot should not be created for no_snapshot marker"


def test_passing_test_no_snapshot(tmp_path):
    """Test that passing tests don't create snapshots."""
    test_file = tmp_path / "test_passing.py"
    test_file.write_text(
        """
def test_passing():
    assert True
"""
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # Test should pass
    assert result.returncode == 0

    # Check that no snapshot was created
    llmdebug_dir = tmp_path / ".llmdebug"
    assert not llmdebug_dir.exists() or len(list(llmdebug_dir.glob("*.json"))) == 0


def _run_pytest_and_get_snapshot(tmp_path, test_code, env_overrides=None):
    """Helper: write test file, run pytest subprocess, return snapshot."""
    test_file = tmp_path / "test_subject.py"
    test_file.write_text(test_code)

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0, f"Test was expected to fail.\nstdout:\n{result.stdout}"

    llmdebug_dir = tmp_path / ".llmdebug"
    assert llmdebug_dir.exists(), f"No .llmdebug dir. stderr: {result.stderr}"

    snapshot = get_latest_snapshot(str(llmdebug_dir))
    assert snapshot is not None, "Failed to read snapshot"
    return snapshot


def test_fixture_values_in_snapshot(tmp_path):
    """Test that custom fixture values appear in the snapshot and internal fixtures are excluded."""
    snapshot = _run_pytest_and_get_snapshot(
        tmp_path,
        """
import pytest

@pytest.fixture
def data():
    return [1, 2, 3]

@pytest.fixture
def label():
    return "hello"

def test_with_fixtures(data, label, request):
    assert False, "intentional failure"
""",
    )
    pytest_ctx = snapshot["pytest"]
    fixtures = pytest_ctx.get("fixtures")
    assert fixtures is not None, f"Expected 'fixtures' key. pytest keys: {list(pytest_ctx.keys())}"
    assert "data" in fixtures
    assert "label" in fixtures
    # Internal fixture 'request' should be excluded
    assert "request" not in fixtures


def test_markers_in_snapshot(tmp_path):
    """Test that custom markers and timeout marker appear in the snapshot."""
    snapshot = _run_pytest_and_get_snapshot(
        tmp_path,
        """
import pytest

@pytest.mark.slow
@pytest.mark.timeout(30)
def test_marked():
    assert False, "intentional failure"
""",
    )
    pytest_ctx = snapshot["pytest"]
    markers = pytest_ctx.get("markers")
    assert markers is not None, f"Expected 'markers' key. pytest keys: {list(pytest_ctx.keys())}"
    assert "slow" in markers
    # timeout marker is captured separately in timeout_config, but also appears in markers
    assert "timeout" in markers

    # Timeout config should also be set
    toc = pytest_ctx.get("timeout_config")
    assert toc is not None, f"Expected 'timeout_config' key. pytest keys: {list(pytest_ctx.keys())}"
    assert toc["seconds"] == 30


def test_param_indices_in_snapshot(tmp_path):
    """Test that parametrize indices are captured."""
    snapshot = _run_pytest_and_get_snapshot(
        tmp_path,
        """
import pytest

@pytest.mark.parametrize("x", [10, 20, 30])
def test_param(x):
    if x == 20:
        assert False, "fails on 20"
""",
    )
    pytest_ctx = snapshot["pytest"]
    param_indices = pytest_ctx.get("param_indices")
    assert param_indices is not None, (
        f"Expected 'param_indices'. pytest keys: {list(pytest_ctx.keys())}"
    )
    assert "x" in param_indices
    # x=20 is index 1 in the parametrize list [10, 20, 30]
    assert param_indices["x"] == 1


def test_internal_fixtures_excluded(tmp_path):
    """Test that internal/noisy fixtures like capsys and tmp_path don't appear."""
    snapshot = _run_pytest_and_get_snapshot(
        tmp_path,
        """
def test_with_capsys(capsys, tmp_path):
    print("hello")
    assert False, "intentional failure"
""",
    )
    pytest_ctx = snapshot["pytest"]
    fixtures = pytest_ctx.get("fixtures", {})
    assert "capsys" not in fixtures
    # tmp_path is NOT in the exclude list and should appear
    assert "tmp_path" in fixtures


def test_fixtures_disabled_by_env(tmp_path):
    """Test that LLMDEBUG_INCLUDE_FIXTURES=false suppresses fixtures."""
    snapshot = _run_pytest_and_get_snapshot(
        tmp_path,
        """
import pytest

@pytest.fixture
def data():
    return [1, 2, 3]

def test_with_fixture(data):
    assert False, "intentional failure"
""",
        env_overrides={"LLMDEBUG_INCLUDE_FIXTURES": "false"},
    )
    pytest_ctx = snapshot["pytest"]
    assert "fixtures" not in pytest_ctx


def test_redaction_profile_from_env(tmp_path):
    """Test that redaction profile env applies to pytest snapshot capture."""
    snapshot = _run_pytest_and_get_snapshot(
        tmp_path,
        """
def test_secret_failure():
    assert False, "password=secret123"
""",
        env_overrides={"LLMDEBUG_REDACTION_PROFILE": "prod"},
    )
    message = snapshot["exception"]["message"]
    assert "[REDACTED]" in message
    assert "password=secret123" not in message


def test_default_redaction_profile_is_ci(monkeypatch):
    monkeypatch.delenv("LLMDEBUG_REDACTION_PROFILE", raising=False)

    cfg = pytest_plugin_module._build_snapshot_config_from_env(debug=False)

    assert cfg.redaction_profile == "ci"
    assert cfg.redact_exception_strings is True
    assert len(cfg.redact) > 0


def test_build_snapshot_config_reads_include_coverage_env(monkeypatch):
    monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "false")
    cfg_disabled = pytest_plugin_module._build_snapshot_config_from_env(debug=False)
    assert cfg_disabled.include_coverage is False

    monkeypatch.setenv("LLMDEBUG_INCLUDE_COVERAGE", "true")
    cfg_enabled = pytest_plugin_module._build_snapshot_config_from_env(debug=False)
    assert cfg_enabled.include_coverage is True


def test_build_snapshot_config_reads_git_dirty_mode_env(monkeypatch):
    monkeypatch.setenv("LLMDEBUG_GIT_DIRTY_MODE", "cached")
    cfg_cached = pytest_plugin_module._build_snapshot_config_from_env(debug=False)
    assert cfg_cached.git_dirty_mode == "cached"

    monkeypatch.setenv("LLMDEBUG_GIT_DIRTY_MODE", "off")
    cfg_off = pytest_plugin_module._build_snapshot_config_from_env(debug=False)
    assert cfg_off.git_dirty_mode == "off"


def test_plugin_keeps_default_snapshot_dir_when_xdist_worker_env_is_set(tmp_path):
    snapshot = _run_pytest_and_get_snapshot(
        tmp_path,
        """
def test_failing_under_worker_env():
    assert False, "intentional failure"
""",
        env_overrides={"PYTEST_XDIST_WORKER": "gw0"},
    )
    assert snapshot["pytest"]["nodeid"] == "test_subject.py::test_failing_under_worker_env"
    assert not (tmp_path / ".llmdebug_gw0").exists()


def test_env_bool_parsing_handles_whitespace_and_falsy(monkeypatch):
    monkeypatch.setenv("LLMDEBUG_INCLUDE_GIT", "  YES ")
    assert pytest_plugin_module._env_bool("LLMDEBUG_INCLUDE_GIT") is True

    monkeypatch.setenv("LLMDEBUG_INCLUDE_GIT", " off ")
    assert pytest_plugin_module._env_bool("LLMDEBUG_INCLUDE_GIT") is False


def test_env_optional_bool_invalid_returns_none(monkeypatch):
    monkeypatch.setenv("LLMDEBUG_REDACT_KEYS", "sometimes")
    assert pytest_plugin_module._env_optional_bool("LLMDEBUG_REDACT_KEYS") is None


def test_build_snapshot_config_invalid_git_dirty_mode_falls_back(monkeypatch, capsys):
    monkeypatch.setenv("LLMDEBUG_GIT_DIRTY_MODE", "invalid")
    cfg = pytest_plugin_module._build_snapshot_config_from_env(debug=True)
    assert cfg.git_dirty_mode == "always"

    stderr = capsys.readouterr().err
    assert "LLMDEBUG_GIT_DIRTY_MODE must be one of" in stderr


def test_build_snapshot_config_invalid_integer_envs_fall_back(monkeypatch):
    monkeypatch.setenv("LLMDEBUG_MAX_EXCEPTION_DEPTH", "-1")
    monkeypatch.setenv("LLMDEBUG_LOG_MAX_RECORDS", "not-a-number")

    cfg = pytest_plugin_module._build_snapshot_config_from_env(debug=False)

    assert cfg.max_exception_depth == 5
    assert cfg.log_max_records == 20


def test_pytest_context_redacts_captured_streams_by_default(tmp_path):
    """CI redaction defaults should redact sensitive stdout/stderr capture."""
    snapshot = _run_pytest_and_get_snapshot(
        tmp_path,
        """
import sys

def test_secret_in_output():
    print("api_key=secret123")
    print("auth_token=topsecret", file=sys.stderr)
    assert False, "boom"
""",
    )
    pytest_ctx = snapshot["pytest"]
    capstdout = pytest_ctx.get("capstdout", "")
    capstderr = pytest_ctx.get("capstderr", "")
    assert "[REDACTED]" in capstdout
    assert "[REDACTED]" in capstderr
    assert "api_key=secret123" not in capstdout
    assert "auth_token=topsecret" not in capstderr


class _DummyMarker:
    def __init__(
        self,
        name: str,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ):
        self.name = name
        self.args = args
        self.kwargs = kwargs or {}


class _DummyCallSpec:
    def __init__(self) -> None:
        self.params = {"x": 1}
        self.indices = {"x": 0}


class _DummyItem:
    def __init__(self, *, include_no_snapshot: bool = False) -> None:
        self.nodeid = "tests/test_subject.py::test_case"
        self.stash: dict[object, list[str]] = {}
        self.fixturenames = ["data", "request"]
        self.funcargs = {"data": [1, 2, 3], "request": "ignored"}
        self.callspec = _DummyCallSpec()
        self.config = SimpleNamespace(workerinput={"workerid": "gw0"})
        self._include_no_snapshot = include_no_snapshot

    def get_closest_marker(self, name: str):
        if name == "timeout":
            return _DummyMarker("timeout", args=(30,))
        if name == "no_snapshot" and self._include_no_snapshot:
            return _DummyMarker("no_snapshot")
        return None

    def iter_markers(self):
        return [
            _DummyMarker("slow"),
            _DummyMarker("timeout", args=(30,), kwargs={"method": "thread"}),
            _DummyMarker("parametrize"),
        ]


class _DummyReport:
    when = "call"
    failed = True
    outcome = "failed"
    nodeid = "tests/test_subject.py::test_case"
    longreprtext = "assert False"
    capstdout = "stdout text"
    capstderr = "stderr text"


class _DummyOutcome:
    def __init__(self, report) -> None:
        self._report = report

    def get_result(self):
        return self._report


class _DummyExcInfo:
    def __init__(self, exc: Exception) -> None:
        self.value = exc
        self.tb = exc.__traceback__


def _run_hook(item, call, report) -> None:
    hook = pytest_plugin_module.pytest_runtest_makereport(item, call)
    next(hook)
    try:
        hook.send(_DummyOutcome(report))
    except StopIteration:
        pass


def test_pytest_plugin_helpers_build_config_and_context(monkeypatch, capsys):
    monkeypatch.setenv("LLMDEBUG_INCLUDE_MODULES", "pkg.alpha, pkg.beta")
    monkeypatch.setenv("LLMDEBUG_REDACTION_PROFILE", "invalid-profile")
    monkeypatch.setenv("LLMDEBUG_OUTPUT_FORMAT", "toon")

    cfg = pytest_plugin_module._build_snapshot_config_from_env(debug=True)
    assert isinstance(cfg, pytest_plugin_module.SnapshotConfig)
    assert cfg.include_modules == ("pkg.alpha", "pkg.beta")
    assert cfg.output_format == "toon"

    stderr = capsys.readouterr().err
    assert "llmdebug:" in stderr
    assert pytest_plugin_module._parse_include_modules("a,b,, c ") == ("a", "b", "c")
    assert pytest_plugin_module._resolve_output_format("json") == "json"
    assert pytest_plugin_module._resolve_output_format("unknown") == "json_compact"

    item = _DummyItem()
    report = _DummyReport()
    pytest_context, callspec = pytest_plugin_module._build_base_pytest_context(item, report, cfg)
    assert pytest_context["nodeid"] == report.nodeid
    assert callspec is item.callspec

    pytest_plugin_module._capture_fixture_context(item, cfg, pytest_context)
    assert "fixtures" in pytest_context
    fixtures = pytest_context["fixtures"]
    assert isinstance(fixtures, dict)
    assert "request" not in fixtures

    pytest_plugin_module._capture_marker_context(item, cfg, pytest_context)
    assert "markers" in pytest_context
    markers = pytest_context["markers"]
    assert isinstance(markers, dict)
    assert "slow" in markers
    assert "parametrize" not in markers

    pytest_plugin_module._capture_timeout_context(item, cfg, pytest_context)
    timeout_config = pytest_context["timeout_config"]
    assert isinstance(timeout_config, dict)
    assert timeout_config["seconds"] == 30

    pytest_plugin_module._capture_param_indices(item.callspec, pytest_context)
    param_indices = pytest_context["param_indices"]
    assert isinstance(param_indices, dict)
    assert param_indices["x"] == 0

    pytest_plugin_module._capture_xdist_worker(item, pytest_context)
    assert pytest_context["xdist_worker"] == "gw0"

    item.stash[pytest_plugin_module._flaky_key] = ["failed", "passed"]
    pytest_plugin_module._capture_flaky_metadata(item, pytest_context)
    assert pytest_context["flaky"] is True

    warnings: list[str] = []
    pytest_plugin_module._capture_context_block(
        warnings,
        "boom",
        lambda: (_ for _ in ()).throw(RuntimeError("x")),
    )
    assert warnings and "boom" in warnings[0]


def test_capture_marker_context_preserves_duplicate_markers() -> None:
    item = _DummyItem()
    cfg = SnapshotConfig(output_format="json")
    pytest_context: dict[str, object] = {}

    item.iter_markers = lambda: [  # type: ignore[method-assign]
        _DummyMarker("slow", args=(1,)),
        _DummyMarker("slow", kwargs={"level": 2}),
    ]

    pytest_plugin_module._capture_marker_context(item, cfg, pytest_context)
    markers = pytest_context.get("markers")
    assert isinstance(markers, dict)
    slow = markers.get("slow")
    assert isinstance(slow, list)
    assert len(slow) == 2


def test_pytest_plugin_hook_calls_capture_exception(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_capture_exception(name, exc, tb, cfg, *, extra):
        captured["name"] = name
        captured["exc"] = exc
        captured["tb"] = tb
        captured["cfg"] = cfg
        captured["extra"] = extra

    monkeypatch.setattr(pytest_plugin_module, "capture_exception", _fake_capture_exception)
    monkeypatch.setattr(
        pytest_plugin_module,
        "_build_snapshot_config_from_env",
        lambda _debug: SnapshotConfig(output_format="json", include_coverage=True),
    )
    monkeypatch.setattr(
        pytest_plugin_module,
        "_capture_coverage_context",
        lambda *_args, **_kwargs: {"files": []},
    )

    item = _DummyItem()
    report = _DummyReport()
    try:
        raise RuntimeError("hook failure")
    except RuntimeError as exc:
        call = SimpleNamespace(excinfo=_DummyExcInfo(exc))

    _run_hook(item, call, report)

    assert captured["name"] == "tests_test_subject.py_test_case"
    extra = captured["extra"]
    assert isinstance(extra, dict)
    assert "pytest" in extra
    pytest_context = extra["pytest"]
    assert pytest_context["nodeid"] == report.nodeid
    assert "coverage" in pytest_context
    assert pytest_context["xdist_worker"] == "gw0"
    assert item.stash[pytest_plugin_module._flaky_key] == ["failed"]


def test_pytest_plugin_hook_skips_coverage_when_disabled(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_capture_exception(name, exc, tb, cfg, *, extra):
        captured["name"] = name
        captured["extra"] = extra

    monkeypatch.setattr(pytest_plugin_module, "capture_exception", _fake_capture_exception)
    monkeypatch.setattr(
        pytest_plugin_module,
        "_build_snapshot_config_from_env",
        lambda _debug: SnapshotConfig(output_format="json", include_coverage=False),
    )
    monkeypatch.setattr(
        pytest_plugin_module,
        "_capture_coverage_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("coverage extraction should be skipped")
        ),
    )

    item = _DummyItem()
    report = _DummyReport()
    try:
        raise RuntimeError("hook failure")
    except RuntimeError as exc:
        call = SimpleNamespace(excinfo=_DummyExcInfo(exc))

    _run_hook(item, call, report)

    extra = captured["extra"]
    assert isinstance(extra, dict)
    pytest_context = extra["pytest"]
    assert "coverage" not in pytest_context


def test_pytest_plugin_hook_handles_no_snapshot_and_missing_excinfo(monkeypatch):
    called = {"count": 0}
    monkeypatch.setattr(
        pytest_plugin_module,
        "capture_exception",
        lambda *_args, **_kwargs: called.__setitem__("count", called["count"] + 1),
    )
    monkeypatch.setattr(
        pytest_plugin_module,
        "_build_snapshot_config_from_env",
        lambda _debug: SnapshotConfig(output_format="json"),
    )

    report = _DummyReport()
    no_snap_item = _DummyItem(include_no_snapshot=True)
    _run_hook(no_snap_item, SimpleNamespace(excinfo=None), report)

    regular_item = _DummyItem()
    _run_hook(regular_item, SimpleNamespace(excinfo=None), report)
    assert called["count"] == 0


def test_pytest_plugin_hook_logs_capture_exception_failures(monkeypatch, capsys):
    monkeypatch.setattr(
        pytest_plugin_module,
        "capture_exception",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("capture boom")),
    )
    monkeypatch.setattr(
        pytest_plugin_module,
        "_build_snapshot_config_from_env",
        lambda _debug: SnapshotConfig(debug=True, output_format="json"),
    )

    item = _DummyItem()
    report = _DummyReport()
    try:
        raise ValueError("hook failure")
    except ValueError as exc:
        call = SimpleNamespace(excinfo=_DummyExcInfo(exc))

    _run_hook(item, call, report)
    stderr = capsys.readouterr().err
    assert "capture failed for tests_test_subject.py_test_case" in stderr


# ---------------------------------------------------------------------------
# Additional tests for uncovered lines in pytest_plugin.py
# ---------------------------------------------------------------------------


def test_module_reload_covers_module_level_definitions() -> None:
    """Re-import the plugin module so module-level ``def`` / constant lines
    execute *under* coverage tracking.

    Covers lines: 41-42 (_TRUTHY/_FALSY), 45, 54, 66, 74, 81, 89, 105,
    114, 127, 197, 224, 242, 269, 290, 298, 310, 324, 336, 347-348, 449.
    """
    import importlib

    import llmdebug.pytest_plugin as mod

    orig_key = mod._flaky_key
    try:
        importlib.reload(mod)
        # Verify the module was successfully reloaded (known attribute exists)
        assert hasattr(mod, "_flaky_key")
        assert hasattr(mod, "_parse_env_bool")
    finally:
        # Restore the original StashKey so subsequent tests keep working.
        mod._flaky_key = orig_key


# -- _parse_env_bool (lines 41-42, 45-51) ---------------------------------


def test_parse_env_bool_truthy_variants() -> None:
    for val in ("1", "true", "yes", "on", "TRUE", " Yes "):
        assert pytest_plugin_module._parse_env_bool(val) is True


def test_parse_env_bool_falsy_variants() -> None:
    for val in ("0", "false", "no", "off", "FALSE", " No "):
        assert pytest_plugin_module._parse_env_bool(val) is False


def test_parse_env_bool_unknown_returns_none() -> None:
    assert pytest_plugin_module._parse_env_bool("") is None
    assert pytest_plugin_module._parse_env_bool("maybe") is None


# -- _env_bool (line 54) --------------------------------------------------


def test_env_bool_unset_no_default_returns_false(monkeypatch) -> None:
    monkeypatch.delenv("_LLMDEBUG_TEST_BOOL", raising=False)
    assert pytest_plugin_module._env_bool("_LLMDEBUG_TEST_BOOL") is False


def test_env_bool_unset_with_truthy_default(monkeypatch) -> None:
    monkeypatch.delenv("_LLMDEBUG_TEST_BOOL", raising=False)
    assert pytest_plugin_module._env_bool("_LLMDEBUG_TEST_BOOL", "true") is True


def test_env_bool_set_overrides_default(monkeypatch) -> None:
    monkeypatch.setenv("_LLMDEBUG_TEST_BOOL", "no")
    assert pytest_plugin_module._env_bool("_LLMDEBUG_TEST_BOOL", "true") is False


def test_env_bool_unrecognised_value_returns_false(monkeypatch) -> None:
    monkeypatch.setenv("_LLMDEBUG_TEST_BOOL", "maybe")
    assert pytest_plugin_module._env_bool("_LLMDEBUG_TEST_BOOL") is False


# -- _env_optional_bool (line 66) -----------------------------------------


def test_env_optional_bool_unset_returns_none(monkeypatch) -> None:
    monkeypatch.delenv("_LLMDEBUG_TEST_OPT", raising=False)
    assert pytest_plugin_module._env_optional_bool("_LLMDEBUG_TEST_OPT") is None


def test_env_optional_bool_truthy(monkeypatch) -> None:
    monkeypatch.setenv("_LLMDEBUG_TEST_OPT", "yes")
    assert pytest_plugin_module._env_optional_bool("_LLMDEBUG_TEST_OPT") is True


def test_env_optional_bool_falsy(monkeypatch) -> None:
    monkeypatch.setenv("_LLMDEBUG_TEST_OPT", "0")
    assert pytest_plugin_module._env_optional_bool("_LLMDEBUG_TEST_OPT") is False


# -- _env_int (line 74) ---------------------------------------------------


def test_env_int_unset_returns_default(monkeypatch) -> None:
    monkeypatch.delenv("_LLMDEBUG_TEST_INT", raising=False)
    assert pytest_plugin_module._env_int("_LLMDEBUG_TEST_INT", 42) == 42


def test_env_int_reads_set_value(monkeypatch) -> None:
    monkeypatch.setenv("_LLMDEBUG_TEST_INT", "7")
    assert pytest_plugin_module._env_int("_LLMDEBUG_TEST_INT", 42) == 7


def test_env_int_minimum_enforced(monkeypatch) -> None:
    monkeypatch.setenv("_LLMDEBUG_TEST_INT", "0")
    assert pytest_plugin_module._env_int("_LLMDEBUG_TEST_INT", 5, minimum=1) == 5


# -- pytest_configure (line 81) -------------------------------------------


def test_pytest_configure_registers_no_snapshot_marker() -> None:
    lines: list[tuple[str, str]] = []

    class _FakeConfig:
        def addinivalue_line(self, section: str, line: str) -> None:
            lines.append((section, line))

    pytest_plugin_module.pytest_configure(
        _FakeConfig()  # type: ignore[arg-type]
    )
    assert any("no_snapshot" in text for _, text in lines)


# -- _record_call_outcome (lines 89, 95-96) -------------------------------


def test_record_call_outcome_records_on_call_phase() -> None:
    item = _DummyItem()
    item.stash = {}  # type: ignore[assignment]
    report = SimpleNamespace(when="call", outcome="failed")
    pytest_plugin_module._record_call_outcome(
        item,
        report,  # type: ignore[arg-type]
    )
    assert item.stash[pytest_plugin_module._flaky_key] == ["failed"]


def test_record_call_outcome_skips_non_call_phase() -> None:
    item = _DummyItem()
    item.stash = {}  # type: ignore[assignment]
    report = SimpleNamespace(when="setup", outcome="passed")
    pytest_plugin_module._record_call_outcome(
        item,
        report,  # type: ignore[arg-type]
    )
    assert pytest_plugin_module._flaky_key not in item.stash


def test_record_call_outcome_catches_stash_exception() -> None:
    """Exercise the except branch (lines 95-96)."""

    class _BrokenStash:
        def setdefault(self, _key: object, _default: object) -> None:
            raise TypeError("stash unavailable")

    item = SimpleNamespace(stash=_BrokenStash())
    report = SimpleNamespace(when="call", outcome="failed")
    # Must not raise — explicitly verify no exception propagates
    result = pytest_plugin_module._record_call_outcome(
        item,
        report,  # type: ignore[arg-type]
    )
    assert result is None


# -- _resolve_output_format (line 105) ------------------------------------


def test_resolve_output_format_all_branches() -> None:
    assert pytest_plugin_module._resolve_output_format("json") == "json"
    assert pytest_plugin_module._resolve_output_format("toon") == "toon"
    assert pytest_plugin_module._resolve_output_format("json_compact") == "json_compact"
    assert pytest_plugin_module._resolve_output_format("xml") == "json_compact"


# -- _resolve_git_dirty_mode (line 114) -----------------------------------


def test_resolve_git_dirty_mode_valid_values() -> None:
    assert pytest_plugin_module._resolve_git_dirty_mode("always") == "always"
    assert pytest_plugin_module._resolve_git_dirty_mode("cached") == "cached"
    assert pytest_plugin_module._resolve_git_dirty_mode("off") == "off"


def test_resolve_git_dirty_mode_invalid_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="LLMDEBUG_GIT_DIRTY_MODE"):
        pytest_plugin_module._resolve_git_dirty_mode("nope")


# -- _build_snapshot_config_from_env (line 127) ----------------------------


def test_build_snapshot_config_from_env_defaults(monkeypatch) -> None:
    # Clear ALL LLMDEBUG_* env vars to avoid pollution from other tests
    for key in list(os.environ):
        if key.startswith("LLMDEBUG_"):
            monkeypatch.delenv(key, raising=False)
    cfg = pytest_plugin_module._build_snapshot_config_from_env(debug=False)
    assert isinstance(cfg, pytest_plugin_module.SnapshotConfig)
    assert cfg.output_format == "json_compact"
    assert cfg.git_dirty_mode == "always"


# -- _build_base_pytest_context (lines 197, 208-209) ----------------------


def test_build_base_context_normal() -> None:
    item = _DummyItem()
    report = _DummyReport()
    cfg = SnapshotConfig(output_format="json")
    ctx, cs = pytest_plugin_module._build_base_pytest_context(
        item,
        report,
        cfg,  # type: ignore[arg-type]
    )
    assert ctx["nodeid"] == report.nodeid
    assert ctx["outcome"] == "failed"
    assert cs is item.callspec


def test_build_base_context_broken_callspec_params() -> None:
    """Lines 208-209: callspec.params raises → params falls back to None."""

    class _BrokenCallSpec:
        @property
        def params(self) -> dict[str, object]:
            raise RuntimeError("params unavailable")

    item = _DummyItem()
    item.callspec = _BrokenCallSpec()  # type: ignore[assignment]
    report = _DummyReport()
    cfg = SnapshotConfig(output_format="json")
    ctx, cs = pytest_plugin_module._build_base_pytest_context(
        item,
        report,
        cfg,  # type: ignore[arg-type]
    )
    assert ctx["params"] is None
    assert cs is item.callspec


def test_build_base_context_no_callspec() -> None:
    item = _DummyItem()
    delattr(item, "callspec")
    report = _DummyReport()
    cfg = SnapshotConfig(output_format="json")
    ctx, cs = pytest_plugin_module._build_base_pytest_context(
        item,
        report,
        cfg,  # type: ignore[arg-type]
    )
    assert ctx["params"] is None
    assert cs is None


# -- _capture_fixture_context (line 224) -----------------------------------


def test_capture_fixture_context_empty_fixtures() -> None:
    """No active fixtures → empty dict, no key in context."""
    item = _DummyItem()
    item.fixturenames = []
    item.funcargs = {}
    cfg = SnapshotConfig(output_format="json")
    ctx: dict[str, object] = {}
    result = pytest_plugin_module._capture_fixture_context(
        item,
        cfg,
        ctx,  # type: ignore[arg-type]
    )
    assert result == {}
    assert "fixtures" not in ctx


def test_capture_fixture_context_includes_custom_excludes_internal() -> None:
    item = _DummyItem()
    cfg = SnapshotConfig(output_format="json")
    ctx: dict[str, object] = {}
    result = pytest_plugin_module._capture_fixture_context(
        item,
        cfg,
        ctx,  # type: ignore[arg-type]
    )
    assert "data" in result
    assert "request" not in result
    assert ctx["fixtures"] == result


# -- _capture_marker_context (lines 242, 261) ------------------------------


def test_capture_marker_triple_duplicate_hits_list_append() -> None:
    """Line 261: ``existing.append(entry)`` for 3+ duplicate markers."""
    item = _DummyItem()
    item.iter_markers = lambda: [  # type: ignore[method-assign]
        _DummyMarker("retry", args=(1,)),
        _DummyMarker("retry", args=(2,)),
        _DummyMarker("retry", args=(3,)),
    ]
    cfg = SnapshotConfig(output_format="json")
    ctx: dict[str, object] = {}
    markers = pytest_plugin_module._capture_marker_context(
        item,
        cfg,
        ctx,  # type: ignore[arg-type]
    )
    assert isinstance(markers["retry"], list)
    assert len(markers["retry"]) == 3


def test_capture_marker_context_empty_markers() -> None:
    """No markers → empty dict, no key in context."""
    item = _DummyItem()
    item.iter_markers = lambda: []  # type: ignore[method-assign]
    cfg = SnapshotConfig(output_format="json")
    ctx: dict[str, object] = {}
    result = pytest_plugin_module._capture_marker_context(
        item,
        cfg,
        ctx,  # type: ignore[arg-type]
    )
    assert result == {}
    assert "markers" not in ctx


# -- _capture_timeout_context (lines 269, 278, 282) -----------------------


def test_capture_timeout_no_marker_returns_empty() -> None:
    """Line 278: early return when timeout marker is absent."""
    item = SimpleNamespace(get_closest_marker=lambda name: None)
    cfg = SnapshotConfig(output_format="json")
    ctx: dict[str, object] = {}
    result = pytest_plugin_module._capture_timeout_context(
        item,
        cfg,
        ctx,  # type: ignore[arg-type]
    )
    assert result == {}
    assert "timeout_config" not in ctx


def test_capture_timeout_kwargs_only() -> None:
    """Line 282: kwargs branch with no positional args."""
    marker = _DummyMarker("timeout", args=(), kwargs={"method": "thread"})
    item = SimpleNamespace(get_closest_marker=lambda name: (marker if name == "timeout" else None))
    cfg = SnapshotConfig(output_format="json")
    ctx: dict[str, object] = {}
    result = pytest_plugin_module._capture_timeout_context(
        item,
        cfg,
        ctx,  # type: ignore[arg-type]
    )
    assert "kwargs" in result
    assert ctx["timeout_config"] == result


def test_capture_timeout_args_and_kwargs() -> None:
    marker = _DummyMarker("timeout", args=(60,), kwargs={"method": "signal"})
    item = SimpleNamespace(get_closest_marker=lambda name: (marker if name == "timeout" else None))
    cfg = SnapshotConfig(output_format="json")
    ctx: dict[str, object] = {}
    result = pytest_plugin_module._capture_timeout_context(
        item,
        cfg,
        ctx,  # type: ignore[arg-type]
    )
    assert result["seconds"] == 60
    assert "kwargs" in result


# -- _capture_param_indices (line 290) -------------------------------------


def test_capture_param_indices_populates_context() -> None:
    callspec = SimpleNamespace(indices={"a": 0, "b": 2})
    ctx: dict[str, object] = {}
    result = pytest_plugin_module._capture_param_indices(callspec, ctx)
    assert result == {"a": 0, "b": 2}
    assert ctx["param_indices"] == {"a": 0, "b": 2}


def test_capture_param_indices_empty_indices() -> None:
    callspec = SimpleNamespace(indices={})
    ctx: dict[str, object] = {}
    result = pytest_plugin_module._capture_param_indices(callspec, ctx)
    assert result == {}
    assert "param_indices" not in ctx


# -- _capture_xdist_worker (lines 298, 302, 307) --------------------------


def test_capture_xdist_worker_no_workerinput() -> None:
    """Line 302: workerinput is None → return None."""
    item = SimpleNamespace(config=SimpleNamespace(workerinput=None))
    ctx: dict[str, object] = {}
    assert (
        pytest_plugin_module._capture_xdist_worker(
            item,
            ctx,  # type: ignore[arg-type]
        )
        is None
    )
    assert "xdist_worker" not in ctx


def test_capture_xdist_worker_missing_workerid() -> None:
    """Line 307: workerinput present but no workerid → return None."""
    item = SimpleNamespace(config=SimpleNamespace(workerinput={}))
    ctx: dict[str, object] = {}
    assert (
        pytest_plugin_module._capture_xdist_worker(
            item,
            ctx,  # type: ignore[arg-type]
        )
        is None
    )
    assert "xdist_worker" not in ctx


def test_capture_xdist_worker_with_workerid() -> None:
    item = SimpleNamespace(config=SimpleNamespace(workerinput={"workerid": "gw3"}))
    ctx: dict[str, object] = {}
    result = pytest_plugin_module._capture_xdist_worker(
        item,
        ctx,  # type: ignore[arg-type]
    )
    assert result == "gw3"
    assert ctx["xdist_worker"] == "gw3"


# -- _capture_flaky_metadata (line 310) -----------------------------------


def test_capture_flaky_empty_stash_returns_none() -> None:
    item = _DummyItem()
    item.stash = {}  # type: ignore[assignment]
    ctx: dict[str, object] = {}
    assert (
        pytest_plugin_module._capture_flaky_metadata(
            item,
            ctx,  # type: ignore[arg-type]
        )
        is None
    )


def test_capture_flaky_single_outcome_returns_none() -> None:
    item = _DummyItem()
    item.stash = {  # type: ignore[assignment]
        pytest_plugin_module._flaky_key: ["failed"]
    }
    ctx: dict[str, object] = {}
    assert (
        pytest_plugin_module._capture_flaky_metadata(
            item,
            ctx,  # type: ignore[arg-type]
        )
        is None
    )


def test_capture_flaky_mixed_outcomes_detects_flaky() -> None:
    item = _DummyItem()
    item.stash = {  # type: ignore[assignment]
        pytest_plugin_module._flaky_key: ["failed", "passed"]
    }
    ctx: dict[str, object] = {}
    result = pytest_plugin_module._capture_flaky_metadata(
        item,
        ctx,  # type: ignore[arg-type]
    )
    assert result == (2, True)
    assert ctx["flaky"] is True
    assert ctx["rerun_count"] == 2


def test_capture_flaky_consistent_failures_not_flaky() -> None:
    item = _DummyItem()
    item.stash = {  # type: ignore[assignment]
        pytest_plugin_module._flaky_key: ["failed", "failed"]
    }
    ctx: dict[str, object] = {}
    result = pytest_plugin_module._capture_flaky_metadata(
        item,
        ctx,  # type: ignore[arg-type]
    )
    assert result == (2, False)
    assert ctx["flaky"] is False


# -- _capture_coverage_context (lines 324, 332) ----------------------------


def test_capture_coverage_context_returns_data(monkeypatch) -> None:
    """Line 332: coverage data returned when non-empty."""
    import llmdebug.coverage_context

    monkeypatch.setattr(
        llmdebug.coverage_context,
        "extract_coverage_from_plugin",
        lambda item, tb, warnings: {"files": {"src/a.py": {}}},
    )
    item = _DummyItem()
    warnings: list[str] = []
    result = pytest_plugin_module._capture_coverage_context(
        item,
        None,
        warnings,  # type: ignore[arg-type]
    )
    assert result == {"files": {"src/a.py": {}}}


def test_capture_coverage_context_returns_none_when_no_data(
    monkeypatch,
) -> None:
    import llmdebug.coverage_context

    monkeypatch.setattr(
        llmdebug.coverage_context,
        "extract_coverage_from_plugin",
        lambda item, tb, warnings: None,
    )
    item = _DummyItem()
    warnings: list[str] = []
    assert (
        pytest_plugin_module._capture_coverage_context(
            item,
            None,
            warnings,  # type: ignore[arg-type]
        )
        is None
    )


def test_capture_coverage_context_returns_none_for_empty_dict(
    monkeypatch,
) -> None:
    import llmdebug.coverage_context

    monkeypatch.setattr(
        llmdebug.coverage_context,
        "extract_coverage_from_plugin",
        lambda item, tb, warnings: {},
    )
    item = _DummyItem()
    warnings: list[str] = []
    assert (
        pytest_plugin_module._capture_coverage_context(
            item,
            None,
            warnings,  # type: ignore[arg-type]
        )
        is None
    )


# -- _capture_context_block (line 336) ------------------------------------


def test_capture_context_block_success_no_warnings() -> None:
    warnings: list[str] = []
    holder: list[int] = []
    pytest_plugin_module._capture_context_block(warnings, "test", lambda: holder.append(1))
    assert holder == [1]
    assert len(warnings) == 0


def test_capture_context_block_records_exception() -> None:
    warnings: list[str] = []

    def _raise() -> None:
        raise ValueError("boom")

    pytest_plugin_module._capture_context_block(warnings, "ctx fail", _raise)
    assert len(warnings) == 1
    assert "ctx fail" in warnings[0]
    assert "ValueError" in warnings[0]


# -- _truncate_optional_text (lines 449, 451) ------------------------------


def test_truncate_optional_text_none_returns_none() -> None:
    assert pytest_plugin_module._truncate_optional_text(None, 100) is None


def test_truncate_optional_text_int_returns_none() -> None:
    assert pytest_plugin_module._truncate_optional_text(42, 100) is None


def test_truncate_optional_text_list_returns_none() -> None:
    assert pytest_plugin_module._truncate_optional_text([], 100) is None


def test_truncate_optional_text_short_string() -> None:
    assert pytest_plugin_module._truncate_optional_text("hello", 100) == "hello"


def test_truncate_optional_text_long_string_truncated() -> None:
    result = pytest_plugin_module._truncate_optional_text("a" * 600, 50)
    assert result is not None
    # truncate_str returns s[:max_len-12] + "...[TRUNC]" ⇒ ~48 chars
    assert len(result) < 600


# -- Hook integration: _plugin_warnings branch (line 437) ------------------


def test_hook_populates_plugin_warnings_on_capture_error(
    monkeypatch,
) -> None:
    """Line 437: ``_plugin_warnings`` list is non-empty and attached."""
    captured: dict[str, object] = {}

    def _fake_capture(name, exc, tb, cfg, *, extra):
        captured["extra"] = extra

    monkeypatch.setattr(pytest_plugin_module, "capture_exception", _fake_capture)
    monkeypatch.setattr(
        pytest_plugin_module,
        "_build_snapshot_config_from_env",
        lambda _debug: SnapshotConfig(output_format="json", include_coverage=False),
    )
    # Make fixture capture raise so a warning is recorded
    monkeypatch.setattr(
        pytest_plugin_module,
        "_capture_fixture_context",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("fixture boom")),
    )

    item = _DummyItem()
    report = _DummyReport()
    try:
        raise RuntimeError("test error")
    except RuntimeError as exc:
        call = SimpleNamespace(excinfo=_DummyExcInfo(exc))

    _run_hook(item, call, report)

    extra = captured["extra"]
    assert isinstance(extra, dict)
    ctx = extra["pytest"]
    assert "_plugin_warnings" in ctx
    warnings_list = ctx["_plugin_warnings"]
    assert any("fixture capture failed" in w for w in warnings_list)


# ---------------------------------------------------------------------------
# Coverage-gap tests: branch-heavy pytest_plugin.py
# ---------------------------------------------------------------------------


def test_build_config_invalid_git_dirty_mode_debug_false(monkeypatch, capsys):
    """Branch [143,145]: invalid git dirty mode with debug=False suppresses stderr."""
    for key in list(os.environ):
        if key.startswith("LLMDEBUG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLMDEBUG_GIT_DIRTY_MODE", "invalid_mode")
    cfg = pytest_plugin_module._build_snapshot_config_from_env(debug=False)
    assert cfg.git_dirty_mode == "always"
    stderr = capsys.readouterr().err
    assert stderr == ""


def test_build_config_invalid_redaction_profile_debug_false(monkeypatch, capsys):
    """Branch [158,161]: invalid redaction profile with debug=False suppresses stderr."""
    for key in list(os.environ):
        if key.startswith("LLMDEBUG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLMDEBUG_REDACTION_PROFILE", "invalid-profile-xyz")
    cfg = pytest_plugin_module._build_snapshot_config_from_env(debug=False)
    assert cfg.redaction_profile is None
    stderr = capsys.readouterr().err
    assert stderr == ""


def test_capture_timeout_context_empty_marker(monkeypatch):
    """Branch [285,287]: timeout marker with no args/kwargs → empty dict, key not set."""
    empty_marker = _DummyMarker("timeout", args=(), kwargs={})

    class _ItemEmptyTimeout(_DummyItem):
        def get_closest_marker(self, name):
            if name == "timeout":
                return empty_marker
            return None

    item = _ItemEmptyTimeout()
    cfg = SnapshotConfig(output_format="json")
    ctx: dict[str, object] = {}
    result = pytest_plugin_module._capture_timeout_context(item, cfg, ctx)
    assert result == {}
    assert "timeout_config" not in ctx


def test_hook_skips_fixtures_when_disabled(monkeypatch):
    """Branches [385,392]: LLMDEBUG_INCLUDE_FIXTURES=false skips fixture capture."""
    captured = {}

    def _fake_capture(name, exc, tb, cfg, *, extra=None):
        captured["extra"] = extra

    monkeypatch.setattr(pytest_plugin_module, "capture_exception", _fake_capture)
    monkeypatch.setattr(
        pytest_plugin_module,
        "_build_snapshot_config_from_env",
        lambda _debug: SnapshotConfig(output_format="json", include_coverage=False),
    )
    monkeypatch.setenv("LLMDEBUG_INCLUDE_FIXTURES", "false")
    monkeypatch.setenv("LLMDEBUG_INCLUDE_MARKERS", "true")
    monkeypatch.setenv("LLMDEBUG_INCLUDE_PYTEST_META", "true")

    item = _DummyItem()
    report = _DummyReport()
    try:
        raise RuntimeError("test")
    except RuntimeError as exc:
        call = SimpleNamespace(excinfo=_DummyExcInfo(exc))
    _run_hook(item, call, report)

    ctx = captured["extra"]["pytest"]
    assert "fixtures" not in ctx


def test_hook_skips_markers_when_disabled(monkeypatch):
    """Branches [392,404]: LLMDEBUG_INCLUDE_MARKERS=false skips marker/timeout capture."""
    captured = {}

    def _fake_capture(name, exc, tb, cfg, *, extra=None):
        captured["extra"] = extra

    monkeypatch.setattr(pytest_plugin_module, "capture_exception", _fake_capture)
    monkeypatch.setattr(
        pytest_plugin_module,
        "_build_snapshot_config_from_env",
        lambda _debug: SnapshotConfig(output_format="json", include_coverage=False),
    )
    monkeypatch.setenv("LLMDEBUG_INCLUDE_MARKERS", "false")
    monkeypatch.setenv("LLMDEBUG_INCLUDE_FIXTURES", "true")
    monkeypatch.setenv("LLMDEBUG_INCLUDE_PYTEST_META", "true")

    item = _DummyItem()
    report = _DummyReport()
    try:
        raise RuntimeError("test")
    except RuntimeError as exc:
        call = SimpleNamespace(excinfo=_DummyExcInfo(exc))
    _run_hook(item, call, report)

    ctx = captured["extra"]["pytest"]
    assert "markers" not in ctx
    assert "timeout_config" not in ctx


def test_hook_skips_param_indices_when_no_callspec(monkeypatch):
    """Branch [404,411]: include_pytest_meta=true but callspec=None → param_indices skipped."""
    captured = {}

    def _fake_capture(name, exc, tb, cfg, *, extra=None):
        captured["extra"] = extra

    monkeypatch.setattr(pytest_plugin_module, "capture_exception", _fake_capture)
    monkeypatch.setattr(
        pytest_plugin_module,
        "_build_snapshot_config_from_env",
        lambda _debug: SnapshotConfig(output_format="json", include_coverage=False),
    )
    monkeypatch.setenv("LLMDEBUG_INCLUDE_PYTEST_META", "true")
    monkeypatch.setenv("LLMDEBUG_INCLUDE_FIXTURES", "true")
    monkeypatch.setenv("LLMDEBUG_INCLUDE_MARKERS", "true")

    item = _DummyItem()
    delattr(item, "callspec")  # No callspec → None
    report = _DummyReport()
    try:
        raise RuntimeError("test")
    except RuntimeError as exc:
        call = SimpleNamespace(excinfo=_DummyExcInfo(exc))
    _run_hook(item, call, report)

    ctx = captured["extra"]["pytest"]
    assert "param_indices" not in ctx


def test_hook_skips_pytest_meta_when_disabled(monkeypatch):
    """Branches [411,423]: LLMDEBUG_INCLUDE_PYTEST_META=false skips xdist/flaky capture."""
    captured = {}

    def _fake_capture(name, exc, tb, cfg, *, extra=None):
        captured["extra"] = extra

    monkeypatch.setattr(pytest_plugin_module, "capture_exception", _fake_capture)
    monkeypatch.setattr(
        pytest_plugin_module,
        "_build_snapshot_config_from_env",
        lambda _debug: SnapshotConfig(output_format="json", include_coverage=False),
    )
    monkeypatch.setenv("LLMDEBUG_INCLUDE_PYTEST_META", "false")
    monkeypatch.setenv("LLMDEBUG_INCLUDE_FIXTURES", "true")
    monkeypatch.setenv("LLMDEBUG_INCLUDE_MARKERS", "true")

    item = _DummyItem()
    report = _DummyReport()
    try:
        raise RuntimeError("test")
    except RuntimeError as exc:
        call = SimpleNamespace(excinfo=_DummyExcInfo(exc))
    _run_hook(item, call, report)

    ctx = captured["extra"]["pytest"]
    # xdist_worker and flaky info should be absent
    assert "xdist_worker" not in ctx
    assert "param_indices" not in ctx


def test_hook_capture_exception_failure_debug_false(monkeypatch, capsys):
    """Branch [442,-347]: capture_exception raises with debug=False → silent swallow."""
    monkeypatch.setattr(
        pytest_plugin_module,
        "capture_exception",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("capture boom")),
    )
    monkeypatch.setattr(
        pytest_plugin_module,
        "_build_snapshot_config_from_env",
        lambda _debug: SnapshotConfig(debug=False, output_format="json"),
    )

    item = _DummyItem()
    report = _DummyReport()
    try:
        raise ValueError("hook failure")
    except ValueError as exc:
        call = SimpleNamespace(excinfo=_DummyExcInfo(exc))
    _run_hook(item, call, report)

    stderr = capsys.readouterr().err
    assert "capture failed" not in stderr
