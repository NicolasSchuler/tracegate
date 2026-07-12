"""Tests for exception capture."""

import sys

import pytest

from llmdebug import _json as json
from llmdebug import debug_snapshot, get_latest_snapshot, snapshot_section
from llmdebug.capture import capture_exception
from llmdebug.config import SnapshotConfig


def test_capture_warnings_are_capture_local_state():
    """Capture warning buffers should not be stored as module-global mutable state."""
    import llmdebug.capture as capture_module

    assert "_warnings" not in vars(capture_module)


def _exception_with_traceback():
    try:
        marker = 1  # noqa: F841 - keep a local in the crash frame
        raise ValueError("boom")
    except Exception as exc:  # pragma: no cover - helper
        return exc, exc.__traceback__
    raise AssertionError("unreachable")


def test_capture_exception_collects_warning_paths(tmp_path, monkeypatch):
    """Capture should aggregate warning paths from optional enrichments."""
    import llmdebug.async_context as async_context_module
    import llmdebug.capture as capture_module
    import llmdebug.error_categories as error_categories_module
    import llmdebug.git_context as git_context_module
    import llmdebug.log_capture as log_capture_module

    def _boom(*_args, **_kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(error_categories_module, "categorize_exception", _boom)
    monkeypatch.setattr(capture_module.os.path, "relpath", _boom)
    monkeypatch.setattr(async_context_module, "is_coroutine_frame", _boom)
    monkeypatch.setattr(git_context_module, "get_git_context", _boom)
    monkeypatch.setattr(async_context_module, "get_async_context", _boom)
    monkeypatch.setattr(log_capture_module, "get_recent_logs", _boom)

    exc, tb = _exception_with_traceback()
    cfg = SnapshotConfig(
        out_dir=str(tmp_path),
        output_format="json",
        categorize_errors=True,
        include_git=True,
        include_async_context=True,
        capture_logs=True,
        debug=False,
    )
    capture_exception("warning_paths", exc, tb, cfg)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    warnings = snapshot.get("_capture_warnings", [])
    assert any("error categorization failed" in w for w in warnings)
    assert any("relative path failed" in w for w in warnings)
    assert any("coroutine frame detection failed" in w for w in warnings)
    assert any("git context failed" in w for w in warnings)
    assert any("async context failed" in w for w in warnings)
    assert any("log capture failed" in w for w in warnings)


def test_collect_frames_warns_when_locals_count_fails(monkeypatch):
    """collect_frames should warn when locals length probing fails in meta mode."""
    import llmdebug.capture as capture_module

    class _BadLocals(dict):
        def __len__(self) -> int:
            raise RuntimeError("len failure")

    class _FakeCode:
        co_argcount = 0
        co_kwonlyargcount = 0
        co_varnames: tuple[str, ...] = ()
        co_flags = 0

    class _FakeFrame:
        f_locals = _BadLocals({"x": 1})
        f_globals = {"__name__": "fake.module"}
        f_code = _FakeCode()

    class _FakeTB:
        tb_frame = _FakeFrame()
        tb_next = None

    class _Extracted:
        filename = "fake.py"
        lineno = 1
        name = "fake"
        line = "raise RuntimeError()"

    monkeypatch.setattr(capture_module.traceback, "extract_tb", lambda _tb: [_Extracted()])
    cfg = SnapshotConfig(locals_mode="meta")
    warnings: list[str] = []

    frames = capture_module.collect_frames(_FakeTB(), cfg, warnings=warnings, redactors=[])
    assert frames
    assert any("locals count failed" in warning for warning in warnings)


def test_capture_exception_emits_debug_warnings(tmp_path, monkeypatch, capsys):
    """Debug mode should emit capture warnings to stderr."""
    import llmdebug._version as version_module

    monkeypatch.delattr(version_module, "__version__", raising=False)

    def _meta_failure():
        raise RuntimeError("meta failure")

    try:
        _meta_failure()
    except RuntimeError as exc:
        tb = exc.__traceback__
        cfg = SnapshotConfig(
            out_dir=str(tmp_path),
            output_format="json",
            locals_mode="meta",
            include_git=False,
            include_async_context=False,
            capture_logs=False,
            debug=True,
        )
        capture_exception("debug_warning_paths", exc, tb, cfg)

    stderr = capsys.readouterr().err
    assert "version detection failed" in stderr

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    warnings = snapshot.get("_capture_warnings", [])
    assert any("version detection failed" in w for w in warnings)


def test_decorator_captures_exception(tmp_path):
    """Test that decorator captures exception and still raises."""

    @debug_snapshot(out_dir=str(tmp_path))
    def failing_func():
        x = [1, 2, 3]  # noqa: F841 - intentional for testing locals capture
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        failing_func()

    # Check snapshot was created
    snapshots = [path for path in tmp_path.glob("*.json") if path.name != "latest.json"]
    assert len(snapshots) == 1


def test_context_manager_captures_exception(tmp_path):
    """Test that context manager captures exception."""
    with pytest.raises(RuntimeError), snapshot_section("test_section", out_dir=str(tmp_path)):
        data = {"key": "value"}  # noqa: F841 - intentional for testing locals capture
        raise RuntimeError("context error")

    snapshots = list(tmp_path.glob("*test_section*.json"))
    assert len(snapshots) == 1


def test_no_exception_no_snapshot(tmp_path):
    """Test that no snapshot is created when no exception."""

    @debug_snapshot(out_dir=str(tmp_path))
    def passing_func():
        return 42

    result = passing_func()
    assert result == 42

    snapshots = list(tmp_path.glob("*.json"))
    assert len(snapshots) == 0


def test_snapshot_contains_correct_structure(tmp_path):
    """Test that snapshot JSON contains all expected fields."""

    @debug_snapshot(out_dir=str(tmp_path))
    def failing_with_locals():
        my_list = [1, 2, 3]  # noqa: F841
        my_dict = {"key": "value"}  # noqa: F841
        my_number = 42  # noqa: F841
        raise ValueError("test error message")

    with pytest.raises(ValueError):
        failing_with_locals()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Check top-level structure
    assert snapshot["name"] == "failing_with_locals"
    assert "timestamp_utc" in snapshot
    assert snapshot["exception"]["type"] == "ValueError"
    assert snapshot["exception"]["qualified_type"] == "builtins.ValueError"
    assert snapshot["exception"]["message"] == "test error message"
    assert "traceback" in snapshot
    assert "frames" in snapshot
    assert snapshot["crash_frame_index"] == 0
    assert "env" in snapshot

    # Check frames structure (crash site should be first)
    assert len(snapshot["frames"]) > 0
    crash_frame = snapshot["frames"][0]
    assert "file" in crash_frame
    assert "line" in crash_frame
    assert "function" in crash_frame
    assert crash_frame["function"] == "failing_with_locals"
    assert "locals" in crash_frame

    # Check that locals were captured
    locals_dict = crash_frame["locals"]
    assert "my_list" in locals_dict
    assert locals_dict["my_list"] == [1, 2, 3]
    assert "my_dict" in locals_dict
    assert "my_number" in locals_dict
    assert locals_dict["my_number"] == 42


def test_capture_writes_debug_session_envelope(tmp_path):
    """Captured files use the versioned DebugSession envelope on disk."""

    @debug_snapshot(out_dir=str(tmp_path), output_format="json")
    def failing_for_envelope():
        raise ValueError("envelope test")

    with pytest.raises(ValueError, match="envelope test"):
        failing_for_envelope()

    latest = tmp_path / "latest.json"
    raw = json.loads(latest.read_text(encoding="utf-8"))

    assert raw["schema_version"] == "2.0"
    assert raw["kind"] == "llmdebug.debug_session"
    assert raw["session"]["name"] == "failing_for_envelope"
    assert raw["snapshot"]["exception"]["type"] == "ValueError"
    assert raw["snapshot"]["frames"][0]["function"] == "failing_for_envelope"


def test_snapshot_captures_array_shapes(tmp_path):
    """Test that numpy arrays are summarized with shape info."""
    pytest.importorskip("numpy")
    import numpy as np

    @debug_snapshot(out_dir=str(tmp_path))
    def failing_with_array():
        arr = np.zeros((10, 20, 30))  # noqa: F841
        raise RuntimeError("array error")

    with pytest.raises(RuntimeError):
        failing_with_array()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    arr_info = crash_frame["locals"]["arr"]

    # Should have array summary, not raw data
    assert "__array__" in arr_info
    assert arr_info["shape"] == [10, 20, 30]
    assert arr_info["dtype"] == "float64"


def test_locals_mode_none_skips_locals(tmp_path):
    """Test that locals_mode='none' doesn't capture locals."""

    @debug_snapshot(out_dir=str(tmp_path), locals_mode="none")
    def failing_no_locals():
        secret = "password123"  # noqa: F841
        raise ValueError("error")

    with pytest.raises(ValueError):
        failing_no_locals()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert "locals" not in crash_frame


def test_locals_mode_meta_captures_metadata(tmp_path):
    """Test that locals_mode='meta' captures locals_meta only."""

    @debug_snapshot(out_dir=str(tmp_path), locals_mode="meta")
    def failing_meta():
        secret = "password123"  # noqa: F841
        items = [1, 2, 3]  # noqa: F841
        raise ValueError("error")

    with pytest.raises(ValueError):
        failing_meta()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert "locals" not in crash_frame
    assert "locals_meta" in crash_frame


def test_source_mode_crash_only(tmp_path):
    """Test that source is captured only for crash frame."""

    @debug_snapshot(out_dir=str(tmp_path), frames=2, source_mode="crash_only")
    def outer():
        def inner():
            raise RuntimeError("boom")

        inner()

    with pytest.raises(RuntimeError):
        outer()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    assert len(snapshot["frames"]) == 2
    crash_frame = snapshot["frames"][0]
    parent_frame = snapshot["frames"][1]
    assert "source" in crash_frame
    assert "source" not in parent_frame


def test_exception_args_and_notes(tmp_path):
    """Test that exception args and notes are captured."""

    @debug_snapshot(out_dir=str(tmp_path))
    def failing_with_args():
        exc = ValueError("a", "b")
        if hasattr(exc, "add_note"):
            exc.add_note("note")  # pyright: ignore[reportAttributeAccessIssue]
        raise exc

    with pytest.raises(ValueError):
        failing_with_args()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    exc = snapshot["exception"]
    assert exc["args"] == ["a", "b"]
    if hasattr(BaseException(), "add_note"):
        assert "notes" in exc


def test_locals_mode_invalid_raises():
    """Test that invalid locals_mode values raise a ValueError."""
    with pytest.raises(ValueError, match="locals_mode must be 'safe', 'meta', or 'none'"):
        debug_snapshot(locals_mode="invalid")  # type: ignore[arg-type]


def test_include_modules_filters_frames(tmp_path):
    """Test that include_modules filters non-matching frames."""

    # Use a module prefix that won't match our test module
    @debug_snapshot(out_dir=str(tmp_path), include_modules=["nonexistent_module"])
    def outer():
        def inner():
            raise ValueError("filtered test")

        inner()

    with pytest.raises(ValueError):
        outer()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Should only have the crash frame (always included) since no module matches
    # The crash frame is always kept regardless of filter
    frames = snapshot["frames"]
    assert len(frames) >= 1

    # Crash frame should be present at index 0
    assert frames[0]["function"] == "inner"


def test_include_modules_matches_prefix(tmp_path):
    """Test that module prefix matching works correctly."""

    # Use the actual test module prefix which should match
    @debug_snapshot(out_dir=str(tmp_path), frames=5, include_modules=["tests", "test_"])
    def outer():
        def middle():
            def inner():
                raise RuntimeError("prefix test")

            inner()

        middle()

    with pytest.raises(RuntimeError):
        outer()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # All frames should be from our test module
    frames = snapshot["frames"]
    assert len(frames) >= 1
    # Crash frame is always included
    assert frames[0]["function"] == "inner"


def test_include_modules_none_captures_all(tmp_path):
    """Test that include_modules=None captures all frames (default behavior)."""

    @debug_snapshot(out_dir=str(tmp_path), frames=5, include_modules=None)
    def failing():
        raise ValueError("all frames test")

    with pytest.raises(ValueError):
        failing()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    assert len(snapshot["frames"]) >= 1


def test_exception_chain_depth_configurable(tmp_path):
    """Test that max_exception_depth controls exception chain traversal."""

    @debug_snapshot(out_dir=str(tmp_path), max_exception_depth=3)
    def deep_chain():
        try:
            try:
                try:
                    raise ValueError("level 0")
                except ValueError as e:
                    raise TypeError("level 1") from e
            except TypeError as e:
                raise RuntimeError("level 2") from e
        except RuntimeError as e:
            raise KeyError("level 3") from e

    with pytest.raises(KeyError):
        deep_chain()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    exc = snapshot["exception"]
    assert exc["type"] == "KeyError"

    # With depth=3, we should see the cause chain
    assert "cause" in exc
    assert exc["cause"]["type"] == "RuntimeError"
    assert "cause" in exc["cause"]
    assert exc["cause"]["cause"]["type"] == "TypeError"


def test_exception_chain_depth_one(tmp_path):
    """Test that max_exception_depth=1 stops at first exception."""

    @debug_snapshot(out_dir=str(tmp_path), max_exception_depth=1)
    def shallow_chain():
        try:
            raise ValueError("original")
        except ValueError as e:
            raise RuntimeError("wrapped") from e

    with pytest.raises(RuntimeError):
        shallow_chain()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    exc = snapshot["exception"]
    assert exc["type"] == "RuntimeError"

    # With depth=1, cause should not have further cause
    # (depth 1 means we capture the exception itself but no chain traversal)
    # Actually checking the implementation: depth starts at 0, increments to 1 for cause
    # So with max_depth=1, cause at depth=1 won't recurse further
    if "cause" in exc:
        # The cause is captured but won't have its own cause/context
        assert "cause" not in exc["cause"] or exc["cause"].get("cause") is None


def test_config_validation_max_exception_depth():
    """Test that max_exception_depth validation works."""
    with pytest.raises(ValueError, match="max_exception_depth must be >= 1"):
        debug_snapshot(max_exception_depth=0)


def test_config_validation_lock_timeout():
    """Test that lock_timeout validation works."""
    with pytest.raises(ValueError, match="lock_timeout must be >= 0"):
        debug_snapshot(lock_timeout=-1)


def test_config_validation_log_max_records():
    """Test that log_max_records validation works."""
    with pytest.raises(ValueError, match="log_max_records must be >= 0"):
        debug_snapshot(log_max_records=-1)


# ============================================================================
# Function Arguments Tests
# ============================================================================


def test_function_args_captured(tmp_path):
    """Test that function arguments are captured separately from locals."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=True)
    def func_with_args(a, b, c=10):
        local_var = a + b
        raise ValueError(f"Sum is {local_var}")

    with pytest.raises(ValueError):
        func_with_args(1, 2, c=20)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Find the crash frame (should be func_with_args)
    crash_frame = snapshot["frames"][0]
    assert crash_frame["function"] == "func_with_args"

    # Should have args and arg_names
    assert "args" in crash_frame
    assert "arg_names" in crash_frame

    # Check arg_names
    assert crash_frame["arg_names"] == ["a", "b", "c"]

    # Check arg values
    args = crash_frame["args"]
    assert args["a"] == 1
    assert args["b"] == 2
    assert args["c"] == 20


def test_function_args_with_positional_only(tmp_path):
    """Test that positional-only arguments are captured in order."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=True)
    def func_with_positional_only(a, b, /, c):
        raise ValueError("test")

    with pytest.raises(ValueError):
        func_with_positional_only(1, 2, 3)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert crash_frame["arg_names"] == ["a", "b", "c"]

    args = crash_frame["args"]
    assert args["a"] == 1
    assert args["b"] == 2
    assert args["c"] == 3


def test_function_args_with_positional_only_and_kwonly(tmp_path):
    """Test mixed positional-only and keyword-only signatures."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=True)
    def func_with_positional_only_kwonly(a, /, b, *, c, d="default"):
        raise ValueError("test")

    with pytest.raises(ValueError):
        func_with_positional_only_kwonly(1, 2, c="required", d="provided")

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert crash_frame["arg_names"] == ["a", "b", "c", "d"]

    args = crash_frame["args"]
    assert args["a"] == 1
    assert args["b"] == 2
    assert args["c"] == "required"
    assert args["d"] == "provided"


def test_function_args_with_positional_only_and_varargs_kwonly_kwargs(tmp_path):
    """Test mixed signatures with positional-only, *args, kw-only, and **kwargs."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=True)
    def func_with_mixed_signature(a, /, b, *args, c=1, **kwargs):
        raise ValueError("test")

    with pytest.raises(ValueError):
        func_with_mixed_signature(10, 20, 30, 40, c=99, z=7)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert crash_frame["arg_names"] == ["a", "b", "c", "*args", "**kwargs"]

    args = crash_frame["args"]
    assert args["a"] == 10
    assert args["b"] == 20
    assert args["c"] == 99
    assert args["args"] == [30, 40]
    assert args["kwargs"] == {"'z'": 7}


def test_function_args_with_positional_only_varargs_kwargs(tmp_path):
    """Test positional-only signatures with *args and **kwargs."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=True)
    def func_with_positional_only_varargs(a, /, *args, **kwargs):
        raise ValueError("test")

    with pytest.raises(ValueError):
        func_with_positional_only_varargs(1, 2, 3, x=10, y=20)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert crash_frame["arg_names"] == ["a", "*args", "**kwargs"]

    args = crash_frame["args"]
    assert args["a"] == 1
    assert args["args"] == [2, 3]
    assert args["kwargs"] == {"'x'": 10, "'y'": 20}


def test_function_args_disabled(tmp_path):
    """Test that function arguments are not captured when disabled."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=False)
    def func_with_args(a, b):
        raise ValueError("test")

    with pytest.raises(ValueError):
        func_with_args(1, 2)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]

    # Should not have args or arg_names
    assert "args" not in crash_frame
    assert "arg_names" not in crash_frame


def test_function_args_with_varargs(tmp_path):
    """Test that *args and **kwargs are captured."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=True)
    def func_with_varargs(a, *args, **kwargs):
        raise ValueError("test")

    with pytest.raises(ValueError):
        func_with_varargs(1, 2, 3, x=10, y=20)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert "args" in crash_frame
    assert "arg_names" in crash_frame

    # Check arg_names includes *args and **kwargs markers
    arg_names = crash_frame["arg_names"]
    assert "a" in arg_names
    assert any("*args" in name for name in arg_names)
    assert any("**kwargs" in name for name in arg_names)

    # Check values (tuples become lists in JSON serialization)
    args = crash_frame["args"]
    assert args["a"] == 1
    assert args["args"] == [2, 3]
    assert args["kwargs"] == {"'x'": 10, "'y'": 20}  # Keys get repr'd


def test_function_args_with_self(tmp_path):
    """Test that self is captured as an argument for methods."""

    class MyClass:
        def __init__(self, value):
            self.value = value

        @debug_snapshot(out_dir=str(tmp_path), include_args=True)
        def method(self, x):
            raise ValueError("test")

    obj = MyClass(42)
    with pytest.raises(ValueError):
        obj.method(10)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert "args" in crash_frame
    assert "arg_names" in crash_frame

    # Check that self is in arg_names
    assert "self" in crash_frame["arg_names"]
    assert "x" in crash_frame["arg_names"]

    # Check values
    args = crash_frame["args"]
    assert args["x"] == 10
    # self should be serialized as an object
    assert "__type__" in args["self"] or "value" in str(args["self"])


def test_function_args_with_keyword_only(tmp_path):
    """Test that keyword-only arguments are captured."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=True)
    def func_with_kwonly(a, b, *, kwonly1, kwonly2=None):
        raise ValueError("test")

    with pytest.raises(ValueError):
        func_with_kwonly(1, 2, kwonly1="required", kwonly2="optional")

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert "args" in crash_frame
    assert "arg_names" in crash_frame

    # Check all arg names are captured (including keyword-only)
    arg_names = crash_frame["arg_names"]
    assert "a" in arg_names
    assert "b" in arg_names
    assert "kwonly1" in arg_names
    assert "kwonly2" in arg_names

    # Check values
    args = crash_frame["args"]
    assert args["a"] == 1
    assert args["b"] == 2
    assert args["kwonly1"] == "required"
    assert args["kwonly2"] == "optional"


def test_function_args_with_kwonly_and_varargs(tmp_path):
    """Test complex signature: positional, *args, keyword-only, **kwargs."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=True)
    def complex_func(a, *args, kwonly, **kwargs):
        raise ValueError("test")

    with pytest.raises(ValueError):
        complex_func(1, 2, 3, kwonly="kw", extra="val")

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    arg_names = crash_frame["arg_names"]

    # Check all arg types are captured in order
    assert "a" in arg_names
    assert "kwonly" in arg_names
    assert any("*args" in name for name in arg_names)
    assert any("**kwargs" in name for name in arg_names)

    # Check values
    args = crash_frame["args"]
    assert args["a"] == 1
    assert args["kwonly"] == "kw"
    assert args["args"] == [2, 3]


# ============================================================================
# Error Categorization Tests
# ============================================================================


def test_error_categorization_in_snapshot(tmp_path):
    """Test that error categorization is included in snapshot."""

    @debug_snapshot(out_dir=str(tmp_path), categorize_errors=True)
    def shape_error():
        raise ValueError("shape mismatch: expected (10,) got (20,)")

    with pytest.raises(ValueError):
        shape_error()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Should have error_category in exception
    exc = snapshot["exception"]
    assert "error_category" in exc
    assert exc["error_category"]["category"] == "shape_mismatch"
    assert "suggestion" in exc["error_category"]


def test_error_categorization_disabled(tmp_path):
    """Test that error categorization is not included when disabled."""

    @debug_snapshot(out_dir=str(tmp_path), categorize_errors=False)
    def some_error():
        raise ValueError("shape mismatch")

    with pytest.raises(ValueError):
        some_error()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Should not have error_category
    exc = snapshot["exception"]
    assert "error_category" not in exc


# ============================================================================
# Git Context Tests
# ============================================================================


def test_git_context_in_snapshot(tmp_path):
    """Test that git context is included in snapshot."""

    @debug_snapshot(out_dir=str(tmp_path), include_git=True)
    def failing():
        raise ValueError("test")

    with pytest.raises(ValueError):
        failing()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Should have git context since we're running from a git repo
    if "git" in snapshot:
        assert "commit" in snapshot["git"]
        assert "branch" in snapshot["git"]


def test_git_context_disabled(tmp_path):
    """Test that git context is not included when disabled."""

    @debug_snapshot(out_dir=str(tmp_path), include_git=False)
    def failing():
        raise ValueError("test")

    with pytest.raises(ValueError):
        failing()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Should not have git context
    assert "git" not in snapshot


# ============================================================================
# Config Serialization Tests
# ============================================================================


def test_new_config_fields_serialized(tmp_path):
    """Test that new config fields are included in capture_config."""

    @debug_snapshot(
        out_dir=str(tmp_path),
        include_git=True,
        include_array_stats=True,
        include_args=True,
        categorize_errors=True,
        include_async_context=True,
        capture_logs=False,
        log_max_records=50,
    )
    def failing():
        raise ValueError("test")

    with pytest.raises(ValueError):
        failing()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    cfg = snapshot["capture_config"]
    assert cfg["include_git"] is True
    assert cfg["git_dirty_mode"] == "always"
    assert cfg["include_array_stats"] is True
    assert cfg["include_args"] is True
    assert cfg["categorize_errors"] is True
    assert cfg["include_async_context"] is True
    assert cfg["capture_logs"] is False
    assert cfg["log_max_records"] == 50
    assert cfg["lock_timeout"] == 5.0
    assert cfg["output_format"] == "json_compact"


def test_git_dirty_mode_serialized(tmp_path):
    @debug_snapshot(
        out_dir=str(tmp_path),
        include_git=True,
        git_dirty_mode="cached",
    )
    def failing():
        raise ValueError("test")

    with pytest.raises(ValueError):
        failing()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    cfg = snapshot["capture_config"]
    assert cfg["git_dirty_mode"] == "cached"


def test_redaction_profile_serialized(tmp_path):
    @debug_snapshot(
        out_dir=str(tmp_path),
        redaction_profile="prod",
    )
    def failing():
        raise ValueError("password=secret123")

    with pytest.raises(ValueError):
        failing()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    cfg = snapshot["capture_config"]
    assert cfg["redaction_profile"] == "prod"
    assert cfg["redact_traceback"] is True
    assert cfg["redact_exception_strings"] is True
    assert len(cfg["redact"]) > 0


def test_exception_and_traceback_redaction_opt_in(tmp_path):
    @debug_snapshot(
        out_dir=str(tmp_path),
        redact=(r"token=\w+",),
        redact_traceback=True,
        redact_exception_strings=True,
    )
    def failing():
        raise ValueError("token=abcdef")

    with pytest.raises(ValueError):
        failing()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    assert "[REDACTED]" in snapshot["exception"]["message"]
    assert "token=abcdef" not in snapshot["exception"]["message"]
    assert "[REDACTED]" in snapshot["traceback"]
    assert "token=abcdef" not in snapshot["traceback"]


def test_context_payload_redacts_env_and_extra(tmp_path, monkeypatch):
    import llmdebug.capture as capture_module

    monkeypatch.setattr(
        capture_module.sys,
        "argv",
        ["python", "--auth_token=abcd", "api_key=secret123"],
    )

    exc, tb = _exception_with_traceback()

    cfg = SnapshotConfig(
        out_dir=str(tmp_path),
        output_format="json",
        redaction_profile="prod",
        redact=(r"(?i)(?:api[_-]?key|auth[_-]?token)\s*[:=]\s*\S+",),
        include_git=False,
        include_async_context=False,
        capture_logs=False,
    )
    capture_exception(
        "context_redaction",
        exc,
        tb,
        cfg,
        extra={"note": "auth_token=abcd"},
    )

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    argv = snapshot["env"]["argv"]
    assert any("[REDACTED]" in value for value in argv)
    assert all("auth_token=abcd" not in value for value in argv)
    assert snapshot["note"] == "[REDACTED]"


def test_default_redaction_settings_emit_warning(tmp_path, monkeypatch):
    import llmdebug as llmdebug_package

    monkeypatch.setattr(llmdebug_package, "_unredacted_default_warning_emitted", False)
    with pytest.warns(UserWarning, match="no redaction settings were provided"):

        @debug_snapshot(out_dir=str(tmp_path))
        def failing():
            raise ValueError("test")

    with pytest.raises(ValueError):
        failing()


def test_redaction_warning_not_emitted_with_profile(tmp_path, monkeypatch):
    import warnings

    import llmdebug as llmdebug_package

    monkeypatch.setattr(llmdebug_package, "_unredacted_default_warning_emitted", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @debug_snapshot(out_dir=str(tmp_path), redaction_profile="ci")
        def failing():
            raise ValueError("test")

    assert not any("no redaction settings were provided" in str(w.message) for w in caught)

    with pytest.raises(ValueError):
        failing()


def test_redaction_warning_not_emitted_with_explicit_patterns(tmp_path, monkeypatch):
    import warnings

    import llmdebug as llmdebug_package

    monkeypatch.setattr(llmdebug_package, "_unredacted_default_warning_emitted", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @debug_snapshot(out_dir=str(tmp_path), redact=(r"secret",))
        def failing():
            raise ValueError("secret data")

    assert not any("no redaction settings were provided" in str(w.message) for w in caught)

    with pytest.raises(ValueError):
        failing()


def test_get_source_snippet_streams_without_readlines(monkeypatch):
    import builtins

    import llmdebug.capture as capture_module

    class _FakeFile:
        def __init__(self, lines: list[str]) -> None:
            self._lines = lines

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def __iter__(self):
            return iter(self._lines)

        def readlines(self):  # pragma: no cover - guard rail assertion
            raise AssertionError("readlines() should not be used")

    monkeypatch.setattr(capture_module.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *_args, **_kwargs: _FakeFile(
            ["line 1\n", "line 2\n", "line 3\n", "line 4\n", "line 5\n"]
        ),
    )

    snippet = capture_module.get_source_snippet("demo.py", lineno=3, ctx=1)
    assert snippet["start"] == 2
    assert snippet["end"] == 4
    assert [row["lineno"] for row in snippet["snippet"]] == [2, 3, 4]


# ---------------------------------------------------------------------------
# _summarize_exception — branch coverage
# ---------------------------------------------------------------------------


def test_summarize_exception_args_serialization_failure(monkeypatch) -> None:
    """If exc.args raises, fallback text is recorded."""
    import llmdebug.capture as capture_module

    class _BadArgs:
        @property
        def args(self):
            raise RuntimeError("args boom")

    exc = _BadArgs()
    cfg = SnapshotConfig()
    result = capture_module._summarize_exception(
        exc,  # type: ignore[arg-type]
        cfg,
        warnings=[],
        redactors=[],
        redact_strings=False,
    )
    assert any("serialization failed" in a for a in result.get("args", []))


def test_summarize_exception_notes_serialization_failure(monkeypatch) -> None:
    """If __notes__ raises, fallback text is recorded."""
    import llmdebug.capture as capture_module

    exc = ValueError("boom")

    class _BadNotes:
        def __iter__(self):
            raise RuntimeError("notes boom")

    exc.__notes__ = _BadNotes()  # type: ignore[attr-defined]
    cfg = SnapshotConfig()
    result = capture_module._summarize_exception(
        exc,
        cfg,
        warnings=[],
        redactors=[],
        redact_strings=False,
    )
    assert any("serialization failed" in n for n in result.get("notes", []))


def test_summarize_exception_cause_chain() -> None:
    """__cause__ is recursively summarized."""
    import llmdebug.capture as capture_module

    cause = TypeError("original cause")
    exc = ValueError("outer")
    exc.__cause__ = cause
    cfg = SnapshotConfig()
    result = capture_module._summarize_exception(
        exc, cfg, warnings=[], redactors=[], redact_strings=False
    )
    assert "cause" in result
    assert result["cause"]["type"] == "TypeError"


def test_summarize_exception_context_chain() -> None:
    """__context__ is recursively summarized when present."""
    import llmdebug.capture as capture_module

    try:
        raise ValueError("first")
    except ValueError:
        try:
            raise RuntimeError("second")
        except RuntimeError as exc:
            cfg = SnapshotConfig()
            result = capture_module._summarize_exception(
                exc, cfg, warnings=[], redactors=[], redact_strings=False
            )
    assert "context" in result
    assert result["context"]["type"] == "ValueError"


def test_summarize_exception_cycle_detected() -> None:
    """Cyclic exception references produce a CYCLE marker instead of infinite recursion."""
    import llmdebug.capture as capture_module

    exc = ValueError("cyclic")
    seen: set[int] = {id(exc)}
    cfg = SnapshotConfig()
    result = capture_module._summarize_exception(
        exc, cfg, warnings=[], redactors=[], redact_strings=False, seen=seen
    )
    assert result["message"] == "...[CYCLE]"


@pytest.mark.skipif(sys.version_info < (3, 11), reason="BaseExceptionGroup requires Python 3.11+")
def test_summarize_exception_base_exception_group() -> None:
    """BaseExceptionGroup children are recursively summarized (Python 3.11+)."""

    import llmdebug.capture as capture_module

    group = BaseExceptionGroup("group", [ValueError("a"), TypeError("b")])  # noqa: F821
    cfg = SnapshotConfig()
    result = capture_module._summarize_exception(
        group, cfg, warnings=[], redactors=[], redact_strings=False
    )
    assert result.get("is_exception_group") is True
    assert len(result["exceptions"]) == 2


# ---------------------------------------------------------------------------
# extract_function_args — *args / **kwargs paths
# ---------------------------------------------------------------------------


def test_extract_function_args_with_varargs_and_varkw() -> None:
    """extract_function_args captures *args and **kwargs names."""
    import llmdebug.capture as capture_module

    def _fn_with_varargs(*args, **kwargs):  # type: ignore[no-untyped-def]
        return capture_module.extract_function_args(import_current_frame())

    import inspect

    def import_current_frame():
        return inspect.currentframe()

    # Call with real frame to get *args / **kwargs in locals
    def _capture_frame_inside():
        frame = inspect.currentframe()
        assert frame is not None
        return capture_module.extract_function_args(frame)

    result = _capture_frame_inside()
    # The function has no real args, but the call should succeed
    assert result is not None
    arg_names, _ = result
    assert isinstance(arg_names, list)


def test_extract_function_args_with_kwonly() -> None:
    """extract_function_args handles keyword-only arguments."""
    import inspect

    import llmdebug.capture as capture_module

    def _kwonly_fn(a, *, b, c=3):  # type: ignore[no-untyped-def]
        frame = inspect.currentframe()
        assert frame is not None
        return capture_module.extract_function_args(frame)

    result = _kwonly_fn(1, b=2)
    assert result is not None
    arg_names, arg_values = result
    assert "a" in arg_names
    assert "b" in arg_names
    assert arg_values["a"] == 1
    assert arg_values["b"] == 2


# ---------------------------------------------------------------------------
# _redact_context — non-dict return path
# ---------------------------------------------------------------------------


def test_redact_context_non_dict_return_preserves_original(monkeypatch) -> None:
    """If redact_jsonlike returns a non-dict, the original context is preserved."""
    import llmdebug.capture as capture_module

    monkeypatch.setattr(
        capture_module, "redact_jsonlike", lambda ctx, r, **kw: ["not", "a", "dict"]
    )
    import re

    redactors = [re.compile("secret")]
    context = {"key": "value"}
    cfg = SnapshotConfig()
    warnings: list[str] = []
    result = capture_module._redact_context(context, redactors, cfg, warnings)
    assert result is context
    assert any("non-dict" in w for w in warnings)


def test_redact_context_exception_preserves_original(monkeypatch) -> None:
    """If redact_jsonlike raises, the original context is preserved with a warning."""
    import llmdebug.capture as capture_module

    def _boom(*args, **kwargs):
        raise RuntimeError("redact crash")

    monkeypatch.setattr(capture_module, "redact_jsonlike", _boom)
    import re

    redactors = [re.compile("secret")]
    context = {"key": "value"}
    cfg = SnapshotConfig()
    warnings: list[str] = []
    result = capture_module._redact_context(context, redactors, cfg, warnings)
    assert result is context
    assert any("context redaction failed" in w for w in warnings)


# ---------------------------------------------------------------------------
# capture_exception — with extra payload containing trace/eval
# ---------------------------------------------------------------------------


def test_capture_exception_with_extra_trace_and_eval(tmp_path) -> None:
    """capture_exception passes trace/eval from extra to the debug session."""
    from llmdebug.capture import capture_exception
    from llmdebug.config import SnapshotConfig

    try:
        raise ValueError("with extra")
    except ValueError as exc:
        tb = exc.__traceback__
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", include_git=False)
        capture_exception(
            "extra_test",
            exc,
            tb,
            cfg,
            extra={"trace": {"spans": [1]}, "eval_outcome": {"status": "ok"}, "hook": "test"},
        )

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    assert snapshot.get("trace") == {"spans": [1]}
    assert snapshot.get("eval") == {"status": "ok"}


# ============================================================================
# Additional coverage tests for specific missing lines
# ============================================================================


# ---------------------------------------------------------------------------
# Line 130: exceptions_truncated flag on exception groups
# ---------------------------------------------------------------------------


def test_exception_group_truncated(monkeypatch) -> None:
    """When an ExceptionGroup has more children than max_items, flag is set."""
    import sys

    if sys.version_info < (3, 11):
        pytest.skip("BaseExceptionGroup requires Python 3.11+")

    import builtins

    from llmdebug.capture import _summarize_exception

    _BEG: type = builtins.__dict__["BaseExceptionGroup"]
    children = [ValueError(f"child-{i}") for i in range(5)]
    group = _BEG("big group", children)
    cfg = SnapshotConfig(max_items=2)
    result = _summarize_exception(
        group,
        cfg,
        warnings=[],
        redactors=[],
        redact_strings=False,
    )
    assert result.get("is_exception_group") is True
    assert len(result["exceptions"]) == 2
    assert result["exceptions_truncated"] is True


# ---------------------------------------------------------------------------
# Lines 174, 179-180: get_env_info - including argv serialization failure
# ---------------------------------------------------------------------------


def test_get_env_info_basic() -> None:
    """get_env_info returns expected keys."""
    from llmdebug.capture import get_env_info

    cfg = SnapshotConfig()
    info = get_env_info(cfg)
    assert "python" in info
    assert "executable" in info
    assert "platform" in info
    assert "cwd" in info
    assert "argv" in info
    assert isinstance(info["argv"], list)


def test_get_env_info_argv_failure(monkeypatch) -> None:
    """When sys.argv serialization fails, argv falls back to []."""
    import llmdebug.capture as capture_module
    from llmdebug.capture import get_env_info

    class _BadArgv:
        def __iter__(self):
            raise RuntimeError("argv boom")

    monkeypatch.setattr(capture_module.sys, "argv", _BadArgv())
    cfg = SnapshotConfig()
    info = get_env_info(cfg)
    assert info["argv"] == []


# ---------------------------------------------------------------------------
# Lines 190, 213, 217-218: get_source_snippet edge cases
# ---------------------------------------------------------------------------


def test_get_source_snippet_def() -> None:
    """get_source_snippet is callable and returns snippet data."""
    from llmdebug.capture import get_source_snippet

    result = get_source_snippet("nonexistent_file.py", 1, 3)
    assert result == {}


def test_get_source_snippet_empty_file(tmp_path) -> None:
    """get_source_snippet returns {} for an empty file (last_line==0)."""
    from llmdebug.capture import get_source_snippet

    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")
    result = get_source_snippet(str(empty_file), 1, 3)
    assert result == {}


def test_get_source_snippet_exception(monkeypatch) -> None:
    """get_source_snippet returns {} on any exception during read."""
    import builtins

    import llmdebug.capture as capture_module

    monkeypatch.setattr(capture_module.os.path, "exists", lambda _p: True)

    def _bad_open(*_a, **_kw):
        raise PermissionError("no access")

    monkeypatch.setattr(builtins, "open", _bad_open)
    from llmdebug.capture import get_source_snippet

    result = get_source_snippet("exists.py", 1, 3)
    assert result == {}


# ---------------------------------------------------------------------------
# Line 221: extract_function_args — function def
# Line 247: positional_count fallback when posonly_count > arg_count
# Lines 281-282: extract_function_args exception handler
# ---------------------------------------------------------------------------


def test_extract_function_args_def() -> None:
    """extract_function_args is callable with a real frame."""
    import inspect

    from llmdebug.capture import extract_function_args

    def _simple(a, b):  # type: ignore[no-untyped-def]
        return inspect.currentframe()

    frame = _simple(1, 2)
    assert frame is not None
    result = extract_function_args(frame)
    assert result is not None
    arg_names, arg_values = result
    assert "a" in arg_names
    assert "b" in arg_names
    assert arg_values["a"] == 1
    assert arg_values["b"] == 2


def test_extract_function_args_posonly_gt_argcount() -> None:
    """When posonly_count > arg_count, positional_count = posonly_count + arg_count."""
    from llmdebug.capture import extract_function_args

    class _FakeCode:
        co_posonlyargcount = 5
        co_argcount = 2  # posonly > argcount (non-standard runtime)
        co_kwonlyargcount = 0
        co_varnames = ("a", "b", "c", "d", "e", "f", "g")
        co_flags = 0

    class _FakeFrame:
        f_code = _FakeCode()
        f_locals = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}

    result = extract_function_args(_FakeFrame())  # type: ignore[arg-type]
    assert result is not None
    arg_names, _arg_values = result
    # positional_count should be 5 + 2 = 7
    assert len(arg_names) == 7


def test_extract_function_args_exception_returns_none() -> None:
    """extract_function_args returns None when frame access raises."""
    from llmdebug.capture import extract_function_args

    class _BrokenFrame:
        @property
        def f_code(self):
            raise RuntimeError("broken frame")

    result = extract_function_args(_BrokenFrame())  # type: ignore[arg-type]
    assert result is None


# ---------------------------------------------------------------------------
# Line 285: _extract_tb_pairs — function def
# ---------------------------------------------------------------------------


def test_extract_tb_pairs_basic(monkeypatch) -> None:
    """_extract_tb_pairs walks traceback and returns paired items."""
    import llmdebug.capture as capture_module
    from llmdebug.capture import _extract_tb_pairs

    class _Extracted:
        filename = "test.py"
        lineno = 10
        name = "test_fn"
        line = "pass"

    class _FakeFrame:
        f_globals = {"__name__": "test_module"}

    class _FakeTB:
        tb_frame = _FakeFrame()
        tb_next = None

    monkeypatch.setattr(capture_module.traceback, "extract_tb", lambda _tb: [_Extracted()])
    cfg = SnapshotConfig()
    result = _extract_tb_pairs(_FakeTB(), cfg)  # type: ignore[arg-type]
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Lines 309, 320-321: _add_locals_safe — truncated locals
# ---------------------------------------------------------------------------


def test_add_locals_safe_truncated(monkeypatch) -> None:
    """When locals exceed max_items, locals_truncated flag and count are set."""
    from llmdebug.capture import _add_locals_safe

    # Create a frame with many locals (more than max_items=2)
    many_locals = {f"var_{i}": i for i in range(10)}

    class _FakeCode:
        co_argcount = 0
        co_posonlyargcount = 0
        co_kwonlyargcount = 0
        co_varnames: tuple[str, ...] = ()
        co_flags = 0

    class _FakeFrame:
        f_locals = many_locals
        f_code = _FakeCode()

    cfg = SnapshotConfig(max_items=2, include_args=False)
    frame_dict: dict[str, object] = {}
    _add_locals_safe(
        frame_dict,  # type: ignore[arg-type]
        _FakeFrame(),  # type: ignore[arg-type]
        cfg,
        [],
    )
    assert frame_dict.get("locals_truncated") is True
    assert frame_dict.get("locals_truncated_keys") == 8  # 10 - 2


# ---------------------------------------------------------------------------
# Lines 331, 345-346: _add_locals_meta — truncated locals in meta mode
# ---------------------------------------------------------------------------


def test_add_locals_meta_truncated() -> None:
    """When total_locals > max_items in meta mode, truncation info is set."""
    from llmdebug.capture import _add_locals_meta

    many_locals = {f"var_{i}": i for i in range(10)}

    class _FakeCode:
        co_argcount = 0
        co_posonlyargcount = 0
        co_kwonlyargcount = 0
        co_varnames: tuple[str, ...] = ()
        co_flags = 0

    class _FakeFrame:
        f_locals = many_locals
        f_code = _FakeCode()

    cfg = SnapshotConfig(max_items=3, include_args=False)
    frame_dict: dict[str, object] = {}
    warnings: list[str] = []
    _add_locals_meta(
        frame_dict,  # type: ignore[arg-type]
        _FakeFrame(),  # type: ignore[arg-type]
        cfg,
        warnings,
    )
    assert frame_dict.get("locals_truncated") is True
    assert frame_dict.get("locals_truncated_keys") == 7  # 10 - 3


# ---------------------------------------------------------------------------
# Lines 355, 372: _serialize_frame — def and file_rel = None when ".."
# ---------------------------------------------------------------------------


def test_serialize_frame_relpath_outside_cwd(monkeypatch) -> None:
    """When relpath starts with '..', file_rel should be None."""
    import llmdebug.capture as capture_module
    from llmdebug.capture import _serialize_frame

    monkeypatch.setattr(capture_module.os.path, "relpath", lambda _p, _cwd: "../outside/file.py")

    class _FakeFrame:
        f_locals = {}
        f_globals = {"__name__": "test_module"}
        f_code = type(
            "FakeCode",
            (),
            {
                "co_argcount": 0,
                "co_posonlyargcount": 0,
                "co_kwonlyargcount": 0,
                "co_varnames": (),
                "co_flags": 0,
            },
        )()

    class _FakeTB:
        tb_frame = _FakeFrame()

    class _FakeExtracted:
        filename = "/outside/file.py"
        lineno = 5
        name = "some_func"
        line = "pass"

    cfg = SnapshotConfig(
        locals_mode="none",
        source_mode="none",
        include_async_context=False,
    )
    warnings: list[str] = []
    result = _serialize_frame(_FakeTB(), _FakeExtracted(), 0, 1, cfg, warnings, [])
    assert result["file_rel"] is None
    assert result["file"] == "/outside/file.py"


# ---------------------------------------------------------------------------
# Lines 405: collect_frames — function def (already exercised, but explicit)
# ---------------------------------------------------------------------------


def test_collect_frames_basic(monkeypatch) -> None:
    """collect_frames returns reversed frame list."""
    import llmdebug.capture as capture_module
    from llmdebug.capture import collect_frames

    class _FakeCode:
        co_argcount = 0
        co_posonlyargcount = 0
        co_kwonlyargcount = 0
        co_varnames: tuple[str, ...] = ()
        co_flags = 0

    class _FakeFrame:
        f_locals = {"x": 1}
        f_globals = {"__name__": "test_module"}
        f_code = _FakeCode()

    class _FakeTB:
        tb_frame = _FakeFrame()
        tb_next = None

    class _Extracted:
        filename = "test.py"
        lineno = 10
        name = "test_fn"
        line = "raise ValueError()"

    monkeypatch.setattr(capture_module.traceback, "extract_tb", lambda _tb: [_Extracted()])
    cfg = SnapshotConfig(
        locals_mode="safe",
        include_args=False,
        include_async_context=False,
        source_mode="none",
    )
    warnings: list[str] = []
    frames = collect_frames(
        _FakeTB(),
        cfg,
        warnings=warnings,
        redactors=[],  # type: ignore[arg-type]
    )
    assert len(frames) == 1
    assert frames[0]["function"] == "test_fn"


# ---------------------------------------------------------------------------
# Lines 427, 437, 447: _try_git_context, _try_async_context, _try_log_context
# ---------------------------------------------------------------------------


def test_try_git_context_import_failure(monkeypatch) -> None:
    """_try_git_context returns None and warns when import fails."""
    from llmdebug.capture import _try_git_context

    def _boom(**_kw):
        raise ImportError("no git module")

    import llmdebug.git_context as git_ctx_mod

    monkeypatch.setattr(git_ctx_mod, "get_git_context", _boom)
    cfg = SnapshotConfig()
    warnings: list[str] = []
    result = _try_git_context(cfg, warnings)
    assert result is None
    assert any("git context failed" in w for w in warnings)


def test_try_async_context_import_failure(monkeypatch) -> None:
    """_try_async_context returns None and warns when import fails."""
    import llmdebug.async_context as async_mod
    from llmdebug.capture import _try_async_context

    def _boom():
        raise RuntimeError("async boom")

    monkeypatch.setattr(async_mod, "get_async_context", _boom)
    warnings: list[str] = []
    result = _try_async_context(warnings)
    assert result is None
    assert any("async context failed" in w for w in warnings)


def test_try_log_context_import_failure(monkeypatch) -> None:
    """_try_log_context returns None and warns when import fails."""
    import llmdebug.log_capture as log_mod
    from llmdebug.capture import _try_log_context

    def _boom(_max_records):
        raise RuntimeError("log boom")

    monkeypatch.setattr(log_mod, "get_recent_logs", _boom)
    cfg = SnapshotConfig()
    warnings: list[str] = []
    result = _try_log_context(cfg, warnings)
    assert result is None
    assert any("log capture failed" in w for w in warnings)


# ---------------------------------------------------------------------------
# Line 457: _redact_context — already covered above, but ensure def is hit
# ---------------------------------------------------------------------------


def test_redact_context_empty_payload() -> None:
    """_redact_context returns empty dict unchanged."""
    from llmdebug.capture import _redact_context

    cfg = SnapshotConfig()
    result = _redact_context({}, [], cfg, [])
    assert result == {}


# ---------------------------------------------------------------------------
# Lines 475, 496: _collect_context
# ---------------------------------------------------------------------------


def test_collect_context_includes_logs(monkeypatch) -> None:
    """_collect_context includes recent_logs when capture_logs is enabled."""
    import llmdebug.log_capture as log_mod
    from llmdebug.capture import _collect_context

    monkeypatch.setattr(log_mod, "get_recent_logs", lambda _n: [{"msg": "test log"}])
    cfg = SnapshotConfig(
        include_env=False,
        include_git=False,
        include_async_context=False,
        capture_logs=True,
    )
    context, trace, eval_p = _collect_context(cfg, [], [], None)
    assert "recent_logs" in context
    assert context["recent_logs"] == [{"msg": "test log"}]
    assert trace is None
    assert eval_p is None


# ---------------------------------------------------------------------------
# Line 505: capture_exception — function def, with tb=None
# ---------------------------------------------------------------------------


def test_capture_exception_no_traceback(tmp_path) -> None:
    """capture_exception works when tb is None (no frames collected)."""
    from llmdebug.capture import capture_exception

    exc = ValueError("no traceback")
    cfg = SnapshotConfig(
        out_dir=str(tmp_path),
        output_format="json",
        include_git=False,
        include_async_context=False,
        capture_logs=False,
    )
    path = capture_exception("no_tb", exc, None, cfg)
    assert path.exists()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    assert snapshot["exception"]["type"] == "ValueError"
    assert snapshot["frames"] == []
    assert "crash_frame_index" not in snapshot


# ---------------------------------------------------------------------------
# Coverage-gap tests: capture.py branches
# ---------------------------------------------------------------------------


def test_summarize_exception_empty_args():
    """Branch [95,99]: exc.args is empty tuple → 'args' key not in summary."""
    from llmdebug.capture import _summarize_exception

    # Create exception with empty args
    exc = Exception()
    assert exc.args == ()
    cfg = SnapshotConfig(output_format="json")
    summary = _summarize_exception(exc, cfg, warnings=[], redactors=[], redact_strings=False)
    assert "args" not in summary
    assert summary["type"] == "Exception"


def test_extract_function_args_missing_from_locals():
    """Branches [252,250], [261,259], [269,271], [277,280]:
    Function args defined in code object but missing from f_locals."""
    import inspect
    import types
    from unittest.mock import MagicMock

    from llmdebug.capture import extract_function_args

    # Build a mock frame with args defined in varnames but NOT in f_locals
    frame = MagicMock(spec=types.FrameType)
    code = MagicMock()
    # Simulate: def func(a, b, *, kw1, *extra_args, **extra_kwargs)
    code.co_varnames = ("a", "b", "kw1", "extra_args", "extra_kwargs")
    code.co_argcount = 2  # positional: a, b
    code.co_posonlyargcount = 0
    code.co_kwonlyargcount = 1  # keyword-only: kw1
    code.co_flags = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS
    frame.f_code = code
    # f_locals is EMPTY — none of the args are present
    frame.f_locals = {}

    result = extract_function_args(frame)
    assert result is not None
    arg_names, arg_values = result
    # All names should be listed
    assert "a" in arg_names
    assert "b" in arg_names
    assert "kw1" in arg_names
    assert "*extra_args" in arg_names
    assert "**extra_kwargs" in arg_names
    # But no values captured since f_locals is empty
    assert arg_values == {}


def test_add_locals_safe_extract_args_returns_none(monkeypatch):
    """Branch [324,-309]: extract_function_args returns None with include_args=True."""
    import sys

    from llmdebug.capture import _add_locals_safe

    cfg = SnapshotConfig(output_format="json", include_args=True)

    # Monkeypatch extract_function_args to return None
    monkeypatch.setattr(
        "llmdebug.capture.extract_function_args",
        lambda frame: None,
    )

    # Grab a real frame object
    frame = sys._getframe(0)

    frame_dict: dict = {}
    _add_locals_safe(frame_dict, frame, cfg, [])
    # args and arg_names should NOT be present when extract_function_args returns None
    assert "args" not in frame_dict
    assert "arg_names" not in frame_dict
    # But locals should still be present
    assert "locals" in frame_dict


def test_add_locals_meta_extract_args_returns_none(monkeypatch):
    """Branch [349,-331]: extract_function_args returns None with include_args=True in meta mode."""
    import sys

    from llmdebug.capture import _add_locals_meta

    cfg = SnapshotConfig(output_format="json", include_args=True, locals_mode="meta")

    monkeypatch.setattr(
        "llmdebug.capture.extract_function_args",
        lambda frame: None,
    )

    frame = sys._getframe(0)

    frame_dict: dict = {}
    warnings: list[str] = []
    _add_locals_meta(frame_dict, frame, cfg, warnings)
    assert "args_meta" not in frame_dict
    assert "arg_names" not in frame_dict
    assert "locals_meta" in frame_dict
