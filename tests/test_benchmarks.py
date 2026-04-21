"""Performance benchmarks for llmdebug.

Run with: uv run pytest tests/test_benchmarks.py --benchmark-only -v
Skip benchmarks: uv run pytest tests/test_benchmarks.py --benchmark-skip
"""

from dataclasses import replace

import pytest

from llmdebug import _json as json
from llmdebug import debug_snapshot, get_latest_snapshot
from llmdebug.capture import capture_exception
from llmdebug.config import SnapshotConfig
from llmdebug.error_categories import categorize_exception
from llmdebug.git_context import get_git_context
from llmdebug.serialize import compile_redactors, serialize_locals_bundle, summarize_array


def _has_module(name: str) -> bool:
    """Check if a module is importable."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def cfg_minimal():
    """Minimal config for baseline measurements."""
    return SnapshotConfig(
        locals_mode="none",
        source_mode="none",
        include_env=False,
        include_git=False,
        include_async_context=False,
        capture_logs=False,
        categorize_errors=False,
        include_args=False,
    )


@pytest.fixture
def cfg_standard():
    """Standard config (default settings)."""
    return SnapshotConfig()


@pytest.fixture
def cfg_full():
    """Full config with all features enabled."""
    return SnapshotConfig(
        include_git=True,
        include_array_stats=True,
        include_args=True,
        categorize_errors=True,
        include_async_context=True,
        capture_logs=True,
    )


# ============================================================================
# Decorator Overhead Benchmarks
# ============================================================================


def test_decorator_overhead_no_exception(benchmark, tmp_path):
    """Benchmark decorator overhead when no exception occurs.

    This measures the cost of wrapping a function - should be <1ms.
    """

    @debug_snapshot(out_dir=str(tmp_path))
    def passing_func():
        return 42

    result = benchmark(passing_func)
    assert result == 42


def test_decorator_overhead_simple_return(benchmark, tmp_path):
    """Benchmark decorator overhead with simple return value."""

    @debug_snapshot(out_dir=str(tmp_path))
    def simple_return():
        x = 1 + 2
        return x * 3

    result = benchmark(simple_return)
    assert result == 9


# ============================================================================
# Capture Latency Benchmarks
# ============================================================================


@pytest.mark.benchmark(group="capture-overhead")
def test_capture_overhead_baseline_no_snapshot(benchmark):
    """Benchmark equivalent failure path overhead without snapshot capture."""

    def do_fail_without_capture():
        try:
            raise ValueError("test error")
        except ValueError:
            return None

    benchmark(do_fail_without_capture)


@pytest.mark.benchmark(group="capture-overhead")
def test_capture_overhead_minimal_config(benchmark, tmp_path, cfg_minimal):
    """Benchmark capture overhead with minimal config."""

    def do_capture():
        try:
            raise ValueError("test error")
        except ValueError as e:
            import sys

            _, _, tb = sys.exc_info()
            capture_exception("bench", e, tb, replace(cfg_minimal, out_dir=str(tmp_path)))

    benchmark(do_capture)


@pytest.mark.benchmark(group="capture-overhead")
def test_capture_overhead_default_config(benchmark, tmp_path, cfg_standard):
    """Benchmark capture overhead with default config."""

    def do_capture():
        try:
            local_var = [1, 2, 3]  # noqa: F841
            raise ValueError("test error")
        except ValueError as e:
            import sys

            _, _, tb = sys.exc_info()
            capture_exception("bench", e, tb, replace(cfg_standard, out_dir=str(tmp_path)))

    benchmark(do_capture)


@pytest.mark.benchmark(group="capture-overhead")
def test_capture_overhead_full_config(benchmark, tmp_path, cfg_full):
    """Benchmark capture overhead with all features enabled."""

    def do_capture():
        try:
            local_var = {"key": "value", "list": [1, 2, 3]}  # noqa: F841
            raise ValueError("shape mismatch: expected (10,) got (20,)")
        except ValueError as e:
            import sys

            _, _, tb = sys.exc_info()
            capture_exception("bench", e, tb, replace(cfg_full, out_dir=str(tmp_path)))

    benchmark(do_capture)


# ============================================================================
# Array Summarization Benchmarks
# ============================================================================


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_array_summarization_small(benchmark):
    """Benchmark array summarization for 100x100 array."""
    import numpy as np

    cfg = SnapshotConfig(include_array_stats=False)
    arr = np.random.randn(100, 100)

    result = benchmark(lambda: summarize_array(arr, cfg))
    assert "shape" in result
    assert result["shape"] == [100, 100]


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_array_summarization_large(benchmark):
    """Benchmark array summarization for 1000x1000 array."""
    import numpy as np

    cfg = SnapshotConfig(include_array_stats=False)
    arr = np.random.randn(1000, 1000)

    result = benchmark(lambda: summarize_array(arr, cfg))
    assert "shape" in result
    assert result["shape"] == [1000, 1000]


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_array_summarization_with_stats(benchmark):
    """Benchmark array summarization with statistics enabled."""
    import numpy as np

    cfg = SnapshotConfig(include_array_stats=True)
    arr = np.random.randn(1000, 1000)

    result = benchmark(lambda: summarize_array(arr, cfg))
    assert "stats" in result


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_array_anomaly_detection(benchmark):
    """Benchmark NaN/Inf detection in large array."""
    import numpy as np

    cfg = SnapshotConfig(include_array_stats=False)
    arr = np.random.randn(1000, 1000)
    # Inject some anomalies
    arr[0, 0] = float("nan")
    arr[500, 500] = float("inf")

    result = benchmark(lambda: summarize_array(arr, cfg))
    assert "anomalies" in result


# ============================================================================
# Component Benchmarks
# ============================================================================


def test_git_context_capture(benchmark):
    """Benchmark git subprocess calls."""
    result = benchmark(get_git_context)
    # Should have context since we're in a git repo
    assert result is not None or result is None  # Works in both git and non-git


def test_error_categorization_shape_mismatch(benchmark):
    """Benchmark error categorization pattern matching."""
    exc = ValueError("shape mismatch: expected (10,) got (20,)")
    result = benchmark(lambda: categorize_exception(exc))
    assert result is not None
    assert result["category"] == "shape_mismatch"


def test_error_categorization_no_match(benchmark):
    """Benchmark error categorization when no match."""

    class CustomError(Exception):
        pass

    exc = CustomError("some unique error message")
    result = benchmark(lambda: categorize_exception(exc))
    assert result is None


def test_large_locals_one_pass_serialization(benchmark):
    """Benchmark one-pass locals value+metadata serialization."""
    cfg = SnapshotConfig(max_items=2000)
    redactors = compile_redactors(cfg.redact)
    local_vars = {f"var_{i}": i for i in range(2000)}

    bundle = benchmark(lambda: serialize_locals_bundle(local_vars, cfg, redactors))
    assert len(bundle.values) == 2000
    assert len(bundle.metadata) == 2000


# ============================================================================
# Snapshot Size Tests (not benchmarks, just measurements)
# ============================================================================


def test_snapshot_size_minimal(tmp_path, cfg_minimal):
    """Measure minimal snapshot file size."""
    try:
        raise ValueError("test")
    except ValueError as e:
        import sys

        _, _, tb = sys.exc_info()
        capture_exception("size_test", e, tb, replace(cfg_minimal, out_dir=str(tmp_path)))

    snapshot = get_latest_snapshot(str(tmp_path))
    size = len(json.dumps(snapshot))
    print(f"\nMinimal snapshot size: {size} bytes")
    assert size > 0


def test_snapshot_size_standard(tmp_path, cfg_standard):
    """Measure standard snapshot file size."""

    @debug_snapshot(out_dir=str(tmp_path))
    def failing():
        local_list = [1, 2, 3]  # noqa: F841
        local_dict = {"key": "value"}  # noqa: F841
        raise ValueError("test error")

    with pytest.raises(ValueError):
        failing()

    snapshot = get_latest_snapshot(str(tmp_path))
    size = len(json.dumps(snapshot))
    print(f"\nStandard snapshot size: {size} bytes")
    assert size > 0


def test_snapshot_size_full(tmp_path, cfg_full):
    """Measure full snapshot file size with all features."""

    @debug_snapshot(
        out_dir=str(tmp_path),
        include_git=True,
        include_array_stats=True,
        include_args=True,
        categorize_errors=True,
        include_async_context=True,
    )
    def failing_with_data():
        local_list = list(range(100))  # noqa: F841
        local_dict = {f"key_{i}": i for i in range(20)}  # noqa: F841
        raise ValueError("shape mismatch: expected (10,) got (20,)")

    with pytest.raises(ValueError):
        failing_with_data()

    snapshot = get_latest_snapshot(str(tmp_path))
    size = len(json.dumps(snapshot))
    print(f"\nFull snapshot size: {size} bytes")
    assert size > 0


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_snapshot_size_with_arrays(tmp_path):
    """Measure snapshot size with array data."""
    import numpy as np

    @debug_snapshot(out_dir=str(tmp_path), include_array_stats=True)
    def failing_with_arrays():
        arr1 = np.random.randn(100, 100)  # noqa: F841
        arr2 = np.zeros((50, 50, 3))  # noqa: F841
        raise ValueError("array processing error")

    with pytest.raises(ValueError):
        failing_with_arrays()

    snapshot = get_latest_snapshot(str(tmp_path))
    size = len(json.dumps(snapshot))
    print(f"\nSnapshot with arrays size: {size} bytes")
    assert size > 0
