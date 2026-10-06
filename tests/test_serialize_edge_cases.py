"""Additional branch-heavy tests for serialize helpers."""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

from llmdebug.config import SnapshotConfig
from llmdebug.serialize import (
    _build_local_metadata_entry,
    _compute_array_stats_and_anomalies,
    _jax_stats_and_anomalies,
    _numpy_like_stats_and_anomalies,
    _sample_array_head,
    _torch_stats_and_anomalies,
    get_array_element_count,
    get_requires_grad,
    get_tensor_device,
    serialize_locals_bundle,
    summarize_dataframe,
    summarize_series,
    to_jsonlike,
)


def test_get_tensor_device_paths() -> None:
    """Verifies that get_tensor_device returns None for CPU devices, a string for CUDA, the set representation when .devices() is present, and None when the device property raises."""

    class _CpuDevice:
        type = "cpu"

        def __str__(self) -> str:
            return "cpu"

    class _CudaDevice:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:0"

    class _WithCpuDevice:
        device = _CpuDevice()

    class _WithCudaDevice:
        device = _CudaDevice()

    class _WithDevices:
        def devices(self) -> set[str]:
            return {"gpu:0"}

    class _WithBadDevice:
        @property
        def device(self) -> object:
            raise RuntimeError("boom")

    assert get_tensor_device(_WithCpuDevice()) is None
    assert get_tensor_device(_WithCudaDevice()) == "cuda:0"
    assert get_tensor_device(_WithDevices()) == "{'gpu:0'}"
    assert get_tensor_device(_WithBadDevice()) is None


def test_get_requires_grad_paths() -> None:
    """Verifies that get_requires_grad returns True when the flag is set, None when it is False, and None when accessing the property raises."""

    class _TrueGrad:
        requires_grad = True

    class _FalseGrad:
        requires_grad = False

    class _BadGrad:
        @property
        def requires_grad(self) -> bool:
            raise RuntimeError("boom")

    assert get_requires_grad(_TrueGrad()) is True
    assert get_requires_grad(_FalseGrad()) is None
    assert get_requires_grad(_BadGrad()) is None


def test_get_array_element_count_fallbacks() -> None:
    """Ensures element-count logic falls back from .size to .shape product and returns None for negative or non-integer shape dimensions."""

    class _FromShape:
        size = "not-an-int"
        shape = (3, 4)

    class _NegativeShape:
        shape = (3, -1)

    class _UnknownShape:
        shape = ("?", 4)

    assert get_array_element_count(_FromShape()) == 12
    assert get_array_element_count(_NegativeShape()) is None
    assert get_array_element_count(_UnknownShape()) is None
    assert get_array_element_count(object()) is None


def test_sample_array_head_fallback_and_error_paths() -> None:
    """Verifies that _sample_array_head falls back from reshape to ravel, then to flatten, and raises TypeError when none of the methods are available."""

    class _RavelOnly:
        def reshape(self, _shape: int) -> np.ndarray:
            raise RuntimeError("no reshape")

        def ravel(self) -> np.ndarray:
            return np.array([1, 2, 3], dtype=np.int32)

    class _FlattenOnly:
        def flatten(self) -> list[int]:
            return [4, 5, 6]

    assert _sample_array_head(_RavelOnly(), max_items=2) == [1.0, 2.0]
    assert _sample_array_head(_FlattenOnly(), max_items=2) == [4.0, 5.0]

    with pytest.raises(TypeError, match="array does not support"):
        _sample_array_head(object(), max_items=2)


def test_compute_array_stats_and_anomalies_short_circuit_paths() -> None:
    """Ensures _compute_array_stats_and_anomalies returns (None, None) for arrays below the element threshold, empty arrays, and unknown-module objects."""
    arr = np.arange(10, dtype=np.float32)
    assert _compute_array_stats_and_anomalies(arr, max_elements=3) == (None, None)
    assert _compute_array_stats_and_anomalies(np.array([], dtype=np.float32)) == (None, None)

    class _Unknown:
        __module__ = "custom.module"

    assert _compute_array_stats_and_anomalies(_Unknown()) == (None, None)


def test_torch_stats_and_anomalies_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies torch stats and NaN/Inf anomaly counts using a monkeypatched torch module, and confirms the no-op path for non-numeric dtypes."""

    class _Scalar:
        def __init__(self, value: float) -> None:
            self._value = value

        def item(self) -> float:
            return self._value

    class _Tensor:
        def __init__(self, values: list[float]) -> None:
            self.values = np.asarray(values, dtype=np.float32)

        def detach(self) -> _Tensor:
            return self

        def cpu(self) -> _Tensor:
            return self

        def float(self) -> _Tensor:
            return self

    class _TensorMask:
        def __init__(self, mask: np.ndarray) -> None:
            self.mask = mask

        def sum(self) -> _Scalar:
            return _Scalar(float(np.sum(self.mask)))

    fake_torch = types.SimpleNamespace(
        isnan=lambda t: _TensorMask(np.isnan(t.values)),
        isinf=lambda t: _TensorMask(np.isinf(t.values)),
        min=lambda t: _Scalar(float(np.min(t.values))),
        max=lambda t: _Scalar(float(np.max(t.values))),
        mean=lambda t: _Scalar(float(np.mean(t.values))),
        std=lambda t, unbiased=False: _Scalar(float(np.std(t.values))),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    stats, anomalies = _torch_stats_and_anomalies(
        _Tensor([1.0, float("nan"), float("inf")]), is_numeric_dtype=True
    )
    assert stats is not None
    assert np.isnan(stats["min"])
    assert anomalies == {"nan": 1, "inf": 1}

    stats2, anomalies2 = _torch_stats_and_anomalies(_Tensor([1.0]), is_numeric_dtype=False)
    assert stats2 is None
    assert anomalies2 is None


def test_jax_stats_and_anomalies_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies JAX array stats and NaN/Inf anomaly counts using a monkeypatched jax module, and confirms the no-op path for non-numeric dtypes."""
    fake_jnp = types.SimpleNamespace(
        float32=np.float32,
        isnan=np.isnan,
        isinf=np.isinf,
        min=np.min,
        max=np.max,
        mean=np.mean,
        std=np.std,
    )
    fake_jax = types.SimpleNamespace(numpy=fake_jnp)
    monkeypatch.setitem(__import__("sys").modules, "jax", fake_jax)
    monkeypatch.setitem(__import__("sys").modules, "jax.numpy", fake_jnp)

    class _JaxArray:
        def __init__(self, values: list[float]) -> None:
            self._values = np.asarray(values, dtype=np.float32)

        def astype(self, dtype: Any) -> np.ndarray:
            return self._values.astype(dtype)

        def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
            # Anomalies are counted on the array itself, not the float32 copy (CAP-23).
            return self._values

    stats, anomalies = _jax_stats_and_anomalies(
        _JaxArray([1.0, float("nan"), float("inf")]), is_numeric_dtype=True
    )
    assert stats is not None
    assert anomalies == {"nan": 1, "inf": 1}

    stats2, anomalies2 = _jax_stats_and_anomalies(_JaxArray([1.0]), is_numeric_dtype=False)
    assert stats2 is None
    assert anomalies2 is None


def test_numpy_like_stats_and_anomalies_cupy_path() -> None:
    """Ensures the CuPy array path transfers data via .get() before computing numpy-based stats and anomalies."""

    class _FakeCupy:
        __module__ = "cupy.core.core"

        def __init__(self, values: list[float]) -> None:
            self._values = np.asarray(values, dtype=np.float32)
            self.dtype = self._values.dtype

        def get(self) -> np.ndarray:
            return self._values

    stats, anomalies = _numpy_like_stats_and_anomalies(
        _FakeCupy([1.0, 2.0, 3.0]),
        module="cupy.core.core",
        is_numeric_dtype=True,
    )
    assert stats is not None
    assert anomalies is None


def test_dataframe_and_series_summaries_with_fakes() -> None:
    class _Values:
        def tolist(self) -> list[list[int]]:
            return [[1, 2], [3, 4]]

    class _Counts:
        def items(self) -> list[tuple[str, int]]:
            return [("a", 1), ("b", 0), ("c", 4)]

    class _Isna:
        def sum(self) -> _Counts:
            return _Counts()

    class _Head:
        values = _Values()

    class _DTypes:
        def items(self) -> list[tuple[str, str]]:
            return [("a", "int64"), ("b", "float64"), ("c", "object")]

    class DataFrame:
        __module__ = "pandas.core.frame"

        def __init__(self) -> None:
            self.shape = (2, 2)
            self.columns = ["a", "b", "c"]
            self.dtypes = _DTypes()

        def __len__(self) -> int:
            return 2

        def head(self, _rows: int) -> _Head:
            return _Head()

        def isna(self) -> _Isna:
            return _Isna()

    class Series:
        __module__ = "pandas.core.series"

        def __init__(self) -> None:
            self.shape = (6,)
            self.name = "s"
            self.dtype = "int64"

        def head(self, _rows: int) -> Series:
            return self

        def tolist(self) -> list[int]:
            return [1, 2, 3, 4, 5]

        def __len__(self) -> int:
            return 6

    cfg = SnapshotConfig(max_items=2)
    df_summary = summarize_dataframe(DataFrame(), cfg)
    assert df_summary["__dataframe__"] is True
    assert df_summary["columns"] == ["a", "b"]
    assert df_summary["columns_truncated"] is True
    assert df_summary["dtypes"] == {"a": "int64", "b": "float64"}
    assert df_summary["dtypes_truncated"] is True
    assert df_summary["null_counts"] == {"a": 1}
    assert df_summary["null_counts_truncated"] is True
    assert "head" in df_summary

    series_summary = summarize_series(Series(), cfg)
    assert series_summary["__series__"] is True
    assert series_summary["name"] == "s"
    assert series_summary["head_truncated"] is True


def test_dataframe_and_series_error_paths() -> None:
    class DataFrame:
        __module__ = "pandas.core.frame"

        @property
        def shape(self) -> tuple[int, int]:
            raise RuntimeError("shape fail")

        @property
        def columns(self) -> list[str]:
            raise RuntimeError("cols fail")

        def __len__(self) -> int:
            return 0

        def head(self, _rows: int) -> object:
            raise RuntimeError("head fail")

    class Series:
        __module__ = "pandas.core.series"

        @property
        def shape(self) -> tuple[int]:
            raise RuntimeError("shape fail")

        name = "bad"
        dtype = "int64"

        def head(self, _rows: int) -> object:
            raise RuntimeError("head fail")

    cfg = SnapshotConfig()
    df_summary = summarize_dataframe(DataFrame(), cfg)
    assert str(df_summary["shape"]).startswith("<error:")
    assert str(df_summary["columns"]).startswith("<error:")
    assert str(df_summary["head"]).startswith("<error:")

    series_summary = summarize_series(Series(), cfg)
    assert str(series_summary["shape"]).startswith("<error:")
    assert str(series_summary["head"]).startswith("<error:")


def test_jsonlike_dict_and_object_truncation_paths() -> None:
    cfg = SnapshotConfig(max_items=2)

    data = {1: "a", 2: "b", 3: "c"}
    out = to_jsonlike(data, cfg)
    assert isinstance(out, dict)
    assert "...[TRUNC]" in out

    class _NonDictDunder:
        @property
        def __dict__(self) -> int:  # type: ignore[override]
            return 123

        def __repr__(self) -> str:
            return "non-dict-repr"

    repr_out = to_jsonlike(_NonDictDunder(), cfg)
    assert repr_out == "non-dict-repr"

    class _BigObject:
        pass

    obj = _BigObject()
    for i in range(30):
        setattr(obj, f"k{i}", i)
    obj_out = to_jsonlike(obj, cfg)
    assert isinstance(obj_out, dict)
    assert "...[TRUNC]" in obj_out


def test_build_local_metadata_entry_handles_len_and_shape_errors() -> None:
    class _BadShape:
        def __iter__(self):
            raise RuntimeError("shape fail")

    class _ArrayLike:
        __module__ = "numpy"

        def __init__(self) -> None:
            self.shape = _BadShape()

        def __len__(self) -> int:
            raise RuntimeError("len fail")

    entry = _build_local_metadata_entry(_ArrayLike())
    assert "len" not in entry
    assert str(entry["shape"]).startswith("<error:")


def test_serialize_locals_bundle_handles_len_error_and_redaction_return_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llmdebug.serialize as serialize_module

    class _LenErrorMapping(dict[str, Any]):
        def __len__(self) -> int:
            raise RuntimeError("len fail")

    cfg = SnapshotConfig(max_items=2)
    bundle = serialize_locals_bundle(_LenErrorMapping({"a": 1, "b": 2}), cfg, [])
    assert bundle.truncated is False
    assert bundle.truncated_keys == 0

    monkeypatch.setattr(serialize_module, "redact_jsonlike", lambda *_a, **_k: ["not", "dict"])
    with pytest.raises(TypeError, match="expected dict"):
        serialize_locals_bundle({"x": 1}, cfg, [])


# ---------------------------------------------------------------------------
# Additional edge-case tests for uncovered lines
# ---------------------------------------------------------------------------


def test_safe_repr_raises_fallback() -> None:
    """Lines 150-151: safe_repr — repr raises exception → fallback."""
    from llmdebug.serialize import safe_repr

    class _Unreprable:
        def __repr__(self) -> str:
            raise RuntimeError("repr exploded")

    cfg = SnapshotConfig()
    result = safe_repr(_Unreprable(), cfg)
    assert "<unreprable _Unreprable>" in result


def test_stable_redaction_sort_key_repr_fallback() -> None:
    """Lines 86, 92-93: _stable_redaction_sort_key — repr raises → fallback."""
    from llmdebug.serialize import _stable_redaction_sort_key

    class _Unreprable:
        def __repr__(self) -> str:
            raise RuntimeError("repr fail")

    key = _stable_redaction_sort_key(_Unreprable())
    assert key[2] == "<unreprable _Unreprable>"


def test_get_tensor_device_jax_without_cpu() -> None:
    """Line 191: get_tensor_device — JAX device.type without 'cpu'."""

    class _TpuDevice:
        type = "tpu"

        def __str__(self) -> str:
            return "tpu:0"

    class _WithTpuDevice:
        device = _TpuDevice()

    assert get_tensor_device(_WithTpuDevice()) == "tpu:0"


def test_sample_array_head_ravel_fallback() -> None:
    """Lines 258-259: _get_flatten_array — ravel fallback when reshape fails."""

    class _ReshapeFailsRavelWorks:
        def reshape(self, _shape: int) -> np.ndarray:
            raise RuntimeError("reshape failed")

        def ravel(self) -> np.ndarray:
            return np.array([10, 20, 30], dtype=np.float64)

    result = _sample_array_head(_ReshapeFailsRavelWorks(), max_items=2)
    assert result == [10.0, 20.0]


def test_sample_array_head_scalar_from_tolist() -> None:
    """Line 269: _sample_array_head — when raw is not a list (scalar from tolist)."""

    # Wrap it so it has reshape/tolist but tolist returns a scalar
    class _ScalarArray:
        def reshape(self, _shape: int) -> _ScalarArray:
            return self

        def __getitem__(self, idx: Any) -> _ScalarArray:
            return self

        def tolist(self) -> float:
            return 42.0

    result = _sample_array_head(_ScalarArray(), max_items=5)
    assert result == [42.0]


def test_count_anomalies_unsupported_module() -> None:
    """Line 307: count_anomalies — returns None for unsupported module."""
    from llmdebug.serialize import count_anomalies

    class _CustomArray:
        __module__ = "custom.tensor"

    assert count_anomalies(_CustomArray()) is None


def test_count_anomalies_numpy_with_nan() -> None:
    """count_anomalies with numpy array containing NaN."""
    from llmdebug.serialize import count_anomalies

    arr = np.array([1.0, float("nan"), float("inf"), 2.0])
    result = count_anomalies(arr)
    assert result is not None
    assert result["nan"] == 1
    assert result["inf"] == 1


def test_summarize_array_shape_dtype_raise() -> None:
    """Lines 594-595, 600-601: summarize_array — shape/dtype extraction failure."""
    from llmdebug.serialize import summarize_array

    class _BadShape:
        """Shape that exists but raises on iteration (list() conversion)."""

        def __iter__(self) -> Any:
            raise RuntimeError("shape boom")

    class _BadDtype:
        """Dtype that exists but raises on str()."""

        def __str__(self) -> str:
            raise RuntimeError("dtype boom")

    class _BadArray:
        __module__ = "numpy"
        __qualname__ = "ndarray"
        shape = _BadShape()
        dtype = _BadDtype()

    cfg = SnapshotConfig()
    summary = summarize_array(_BadArray(), cfg)
    assert "<error:" in str(summary.get("shape", ""))
    assert "<error:" in str(summary.get("dtype", ""))


def test_summarize_dataframe_dtypes_truncated() -> None:
    """Lines 662-663: summarize_dataframe — dtypes truncated."""

    class _DTypes:
        def items(self) -> list[tuple[str, str]]:
            return [("a", "int64"), ("b", "float64"), ("c", "object"), ("d", "bool")]

    class _Counts:
        def items(self) -> list[tuple[str, int]]:
            return [("a", 0), ("b", 3), ("c", 0), ("d", 1)]

    class _Isna:
        def sum(self) -> _Counts:
            return _Counts()

    class _Values:
        def tolist(self) -> list[list[int]]:
            return [[1, 2, 3, 4]]

    class _Head:
        values = _Values()

    class DataFrame:
        __module__ = "pandas.core.frame"

        def __init__(self) -> None:
            self.shape = (1, 4)
            self.columns = ["a", "b", "c", "d"]
            self.dtypes = _DTypes()

        def __len__(self) -> int:
            return 1

        def head(self, _rows: int) -> _Head:
            return _Head()

        def isna(self) -> _Isna:
            return _Isna()

    cfg = SnapshotConfig(max_items=2)
    summary = summarize_dataframe(DataFrame(), cfg)
    assert summary["dtypes_truncated"] is True
    assert len(summary["dtypes"]) == 2


def test_summarize_dataframe_null_counts_exception() -> None:
    """Lines 684-685: summarize_dataframe — null_counts exception."""

    class _DTypes:
        def items(self) -> list[tuple[str, str]]:
            return [("a", "int64")]

    class _Values:
        def tolist(self) -> list[list[int]]:
            return [[1]]

    class _Head:
        values = _Values()

    class DataFrame:
        __module__ = "pandas.core.frame"

        def __init__(self) -> None:
            self.shape = (1, 1)
            self.columns = ["a"]
            self.dtypes = _DTypes()

        def __len__(self) -> int:
            return 1

        def head(self, _rows: int) -> _Head:
            return _Head()

        def isna(self) -> Any:
            raise RuntimeError("isna broken")

    cfg = SnapshotConfig()
    summary = summarize_dataframe(DataFrame(), cfg)
    # null_counts should be absent (exception silently swallowed)
    assert "null_counts" not in summary


def test_jsonlike_value_dataframe_series_dispatch() -> None:
    """Lines 778, 780: _jsonlike_value — dataframe/series dispatch."""

    class _Values:
        def tolist(self) -> list[list[int]]:
            return [[1]]

    class _DTypes:
        def items(self) -> list[tuple[str, str]]:
            return [("x", "int64")]

    class _Counts:
        def items(self) -> list[tuple[str, int]]:
            return [("x", 0)]

    class _Isna:
        def sum(self) -> _Counts:
            return _Counts()

    class _Head:
        values = _Values()

    class DataFrame:
        __module__ = "pandas.core.frame"

        def __init__(self) -> None:
            self.shape = (1, 1)
            self.columns = ["x"]
            self.dtypes = _DTypes()

        def __len__(self) -> int:
            return 1

        def head(self, _rows: int) -> _Head:
            return _Head()

        def isna(self) -> _Isna:
            return _Isna()

    class Series:
        __module__ = "pandas.core.series"

        def __init__(self) -> None:
            self.shape = (3,)
            self.name = "s"
            self.dtype = "int64"

        def head(self, _rows: int) -> Series:
            return self

        def tolist(self) -> list[int]:
            return [1, 2, 3]

        def __len__(self) -> int:
            return 3

    cfg = SnapshotConfig()
    df_out = to_jsonlike(DataFrame(), cfg)
    assert isinstance(df_out, dict)
    assert df_out.get("__dataframe__") is True

    series_out = to_jsonlike(Series(), cfg)
    assert isinstance(series_out, dict)
    assert series_out.get("__series__") is True


def test_locals_metadata_truncation_at_max_items() -> None:
    """Line 885: locals_metadata — truncation at max_items."""
    from llmdebug.serialize import locals_metadata

    local_vars = {f"var_{i}": i for i in range(20)}
    cfg = SnapshotConfig(max_items=5)
    meta = locals_metadata(local_vars, cfg)
    assert len(meta) == 5


def test_numpy_like_stats_non_numeric_dtype() -> None:
    """Lines 513-514: numpy-like stats — non-numeric dtype path."""
    # Bool dtype can be cast to float64 but is not considered "numeric"
    arr = np.array([True, False, True])
    stats, _anomalies = _numpy_like_stats_and_anomalies(arr, module="numpy", is_numeric_dtype=False)
    assert stats is None


def test_add_array_stats_path() -> None:
    """Lines 575-577: _add_array_stats — stats computation path."""
    from llmdebug.serialize import _add_array_stats

    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    summary: dict[str, Any] = {}
    cfg = SnapshotConfig(include_array_stats=True)
    _add_array_stats(summary, arr, cfg, element_count=5, is_non_cpu=False)
    assert "stats" in summary
    assert "min" in summary["stats"]


def test_compute_array_stats_and_anomalies_torch_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lines 414-415: _compute_array_stats_and_anomalies — torch dispatch."""
    import types as _types

    class _Scalar:
        def __init__(self, value: float) -> None:
            self._value = value

        def item(self) -> float:
            return self._value

    class _TensorMask:
        def __init__(self, mask: np.ndarray) -> None:
            self.mask = mask

        def sum(self) -> _Scalar:
            return _Scalar(float(np.sum(self.mask)))

    class _Tensor:
        __module__ = "torch"
        size = 3
        shape = (3,)
        dtype = "float32"

        def __init__(self, values: list[float]) -> None:
            self.values = np.asarray(values, dtype=np.float32)

        def detach(self) -> _Tensor:
            return self

        def cpu(self) -> _Tensor:
            return self

        def float(self) -> _Tensor:
            return self

    fake_torch = _types.SimpleNamespace(
        isnan=lambda t: _TensorMask(np.isnan(t.values)),
        isinf=lambda t: _TensorMask(np.isinf(t.values)),
        min=lambda t: _Scalar(float(np.min(t.values))),
        max=lambda t: _Scalar(float(np.max(t.values))),
        mean=lambda t: _Scalar(float(np.mean(t.values))),
        std=lambda t, unbiased=False: _Scalar(float(np.std(t.values))),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    stats, anomalies = _compute_array_stats_and_anomalies(_Tensor([1.0, 2.0, 3.0]))
    assert stats is not None
    assert anomalies is None  # no NaN/Inf


def test_compute_array_stats_and_anomalies_exception_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lines 422-423: _compute_array_stats_and_anomalies — exception fallback."""

    class _BadArray:
        __module__ = "numpy"
        size = 5
        shape = (5,)
        dtype = "float32"

    monkeypatch.setattr(
        "llmdebug.serialize._is_numeric_array_dtype",
        lambda _arr: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    stats, anomalies = _compute_array_stats_and_anomalies(_BadArray())
    assert stats is None
    assert anomalies is None


def test_head_sampling_failure_in_numpy_like() -> None:
    """Line 541-542: head sampling failure."""
    from llmdebug.serialize import _add_head_sample

    class _BadArray:
        __module__ = "numpy"

        def reshape(self, _shape: int) -> None:
            raise RuntimeError("reshape fail")

        def ravel(self) -> None:
            raise RuntimeError("ravel fail")

        def flatten(self) -> None:
            raise RuntimeError("flatten fail")

    summary: dict[str, Any] = {}
    _add_head_sample(summary, _BadArray(), element_count=5, is_non_cpu=False)
    assert "<error:" in str(summary.get("head", ""))


def test_compute_array_stats_cupy_fallback() -> None:
    """Lines 382, 390-392: compute_array_stats — CuPy path."""
    from llmdebug.serialize import compute_array_stats

    class _FakeCupy:
        __module__ = "cupy.core.core"

        def __init__(self, values: list[float]) -> None:
            self._values = np.asarray(values, dtype=np.float32)
            self.dtype = self._values.dtype
            self.size = self._values.size

        def get(self) -> np.ndarray:
            return self._values

        def astype(self, dtype: Any) -> np.ndarray:
            return self._values.astype(dtype)

    stats = compute_array_stats(_FakeCupy([1.0, 2.0, 3.0]))
    assert stats is not None
    assert "min" in stats
    assert stats["min"] == 1.0


def test_compute_array_stats_exception_returns_none() -> None:
    """Lines 390-392: compute_array_stats — exception fallback returns None."""
    from llmdebug.serialize import compute_array_stats

    class _BrokenArray:
        __module__ = "numpy"
        dtype = "float32"
        size = 5

        def astype(self, _dtype: Any) -> Any:
            raise RuntimeError("astype exploded")

    assert compute_array_stats(_BrokenArray()) is None


def test_jax_anomalies_are_counted_on_the_original_array(monkeypatch: pytest.MonkeyPatch) -> None:
    """CAP-23: the float32 copy turned 1e300 into inf, a false anomaly."""
    fake_jnp = types.SimpleNamespace(
        float32=np.float32,
        isnan=np.isnan,
        isinf=np.isinf,
        min=np.min,
        max=np.max,
        mean=np.mean,
        std=np.std,
    )
    monkeypatch.setitem(__import__("sys").modules, "jax", types.SimpleNamespace(numpy=fake_jnp))
    monkeypatch.setitem(__import__("sys").modules, "jax.numpy", fake_jnp)

    with np.errstate(over="ignore"):
        _, anomalies = _jax_stats_and_anomalies(np.array([1e300, 2.0]), is_numeric_dtype=True)

    assert anomalies is None


def test_cupy_arrays_get_anomaly_counts() -> None:
    """CAP-23: the combined path counted NaN/Inf for NumPy only."""

    class _FakeCupy:
        __module__ = "cupy.core.core"

        def __init__(self, values: list[float]) -> None:
            self._values = np.asarray(values, dtype=np.float32)
            self.dtype = self._values.dtype

        def get(self) -> np.ndarray:
            return self._values

    from llmdebug.serialize import count_anomalies

    arr = _FakeCupy([1.0, float("nan"), float("inf")])
    assert count_anomalies(arr) == {"nan": 1, "inf": 1}
    assert _compute_array_stats_and_anomalies(arr)[1] == {"nan": 1, "inf": 1}


def _past_depth_limit(value: Any, cfg: SnapshotConfig) -> Any:
    """Serialize *value* where ``to_jsonlike`` has to fall back to a repr (CAP-03)."""
    rendered = to_jsonlike({"a": {"b": value}}, cfg)
    assert isinstance(rendered, dict)
    return rendered["a"]["b"]


def test_depth_limit_repr_keeps_the_plain_repr_without_redaction() -> None:
    cfg = SnapshotConfig(redaction_profile="dev")
    assert _past_depth_limit({"password": "hunter2"}, cfg) == "{'password': 'hunter2'}"


def test_depth_limit_repr_elides_oversized_containers() -> None:
    cfg = SnapshotConfig(redaction_profile="prod", max_items=2, max_str=200)
    wide = {"k0": 0, "k1": 1, "k2": 2, "k3": 3}

    assert _past_depth_limit(wide, cfg) == "{'k0': 0, 'k1': 1, '...[TRUNC]': '+2 more keys'}"
    assert _past_depth_limit([0, 1, 2, 3], cfg) == "[0, 1, '...[TRUNC]']"


def test_depth_limit_repr_preserves_tuples_and_redacts_pairs() -> None:
    cfg = SnapshotConfig(redaction_profile="prod", max_items=5, max_str=200)

    assert _past_depth_limit((1, {"api_key": "abc"}), cfg) == "(1, {'api_key': '[REDACTED]'})"
    assert _past_depth_limit(["api_key", "abc"], cfg) == "['api_key', '[REDACTED]']"


def test_depth_limit_repr_stops_at_its_own_depth_bound() -> None:
    """Below the bound the repr is handed to the text patterns unchanged."""
    cfg = SnapshotConfig(redaction_profile="prod", max_str=500)
    deep: Any = {"password": "hunter2"}
    for _ in range(7):
        deep = {"wrap": deep}

    assert "'password': 'hunter2'" in _past_depth_limit(deep, cfg)


def test_depth_limit_repr_fails_closed_on_an_unwalkable_container() -> None:
    class _Hostile(dict):  # type: ignore[type-arg]
        def items(self) -> Any:
            raise RuntimeError("no items")

    cfg = SnapshotConfig(redaction_profile="prod")

    assert _past_depth_limit(_Hostile(password="hunter2"), cfg) == (
        "<serialization failed: RuntimeError>"
    )
