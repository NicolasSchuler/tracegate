"""Safe serialization with smart array handling."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from re import Pattern
from typing import Any, cast

from .config import SnapshotConfig
from .redaction_policy import SENSITIVE_NAMES

JsonLike = None | bool | int | float | str | dict[str, Any] | list[Any]

MAX_ARRAY_ELEMENTS_FOR_HEAD = 1_000_000
MAX_ARRAY_ELEMENTS_FOR_STATS = 2_000_000
MAX_ARRAY_ELEMENTS_FOR_ANOMALIES = 2_000_000
# orjson encodes ints only within the signed/unsigned 64-bit range and raises otherwise.
JSON_INT_MIN = -(2**63)
JSON_INT_MAX = 2**64 - 1


@dataclass(frozen=True)
class LocalsSerializationBundle:
    """Serialized local variables with per-key type metadata.

    Produced by :func:`serialize_locals_bundle` after applying size limits,
    redaction, and type introspection to a frame's ``f_locals``.

    Attributes:
        values: Serialized local variable values keyed by variable name.
        metadata: Per-variable type/shape metadata keyed by variable name.
        truncated: ``True`` when the number of locals exceeded the
            configured ``max_items`` limit and the result was clipped.
        truncated_keys: Number of local variable keys that were dropped
            due to the ``max_items`` limit (zero when *truncated* is
            ``False``).
    """

    values: dict[str, Any]
    metadata: dict[str, dict[str, Any]]
    truncated: bool
    truncated_keys: int


_last_redactor_patterns: tuple[str | Pattern[str], ...] | None = None
_last_redactor_result: list[Pattern[str]] = []


def compile_redactors(patterns: tuple[str | Pattern[str], ...] | None) -> list[Pattern[str]]:
    """Compile regex patterns for redaction."""
    global _last_redactor_patterns, _last_redactor_result
    normalized_patterns = patterns or ()
    if normalized_patterns is _last_redactor_patterns:
        return _last_redactor_result
    compiled: list[Pattern[str]] = []
    for p in normalized_patterns:
        if isinstance(p, str):
            try:
                compiled.append(re.compile(p))
            except re.error as e:
                import sys

                sys.stderr.write(f"llmdebug: invalid redaction pattern {p!r}: {e}\n")
        else:
            compiled.append(p)
    _last_redactor_patterns = normalized_patterns
    _last_redactor_result = compiled
    return compiled


REDACTED = "[REDACTED]"

# CAP-03: matched against local-variable names and dict keys whenever any redaction pattern
# is active; the value under a matching name is replaced whole.
SENSITIVE_NAME_PATTERN = re.compile(f"(?i){SENSITIVE_NAMES}")


def is_sensitive_name(name: str) -> bool:
    """Whether a variable or key name marks its value as a secret."""
    return SENSITIVE_NAME_PATTERN.search(name) is not None


def redact_text(text: str, redactors: list[Pattern[str]]) -> str:
    """Apply redaction patterns to text."""
    for rx in redactors:
        text = rx.sub(REDACTED, text)
    return text


def _insert_redacted_key(out: dict[str, Any], key: str, value: JsonLike) -> None:
    """Insert a key/value pair while preserving collisions via stable suffixes."""
    if key not in out:
        out[key] = value
        return
    suffix = 2
    while True:
        candidate = f"{key}#{suffix}"
        if candidate not in out:
            out[candidate] = value
            return
        suffix += 1


def _stable_redaction_sort_key(item: Any) -> tuple[str, str, str]:
    try:
        rendered = repr(item)
    except Exception:
        rendered = f"<unreprable {type(item).__name__}>"
    return (type(item).__module__, type(item).__name__, rendered)


def redact_jsonlike(
    value: JsonLike,
    redactors: list[Pattern[str]],
    *,
    redact_keys: bool = False,
) -> JsonLike:
    """Recursively redact only string leaf values in JSON-like data.

    This avoids the fragile serialize-then-regex-then-parse flow, where
    aggressive patterns can corrupt JSON structure. When any redactor is
    active, the whole value under a dict key with a sensitive name (see
    :func:`is_sensitive_name`) is replaced, whatever its type.
    """
    if isinstance(value, str):
        return redact_text(value, redactors)
    if isinstance(value, tuple):
        tuple_items = cast(tuple[Any, ...], value)
        return [
            redact_jsonlike(cast(JsonLike, item), redactors, redact_keys=redact_keys)
            for item in tuple_items
        ]
    if isinstance(value, (set, frozenset)):
        ordered_items = sorted(
            cast(set[Any] | frozenset[Any], value), key=_stable_redaction_sort_key
        )
        return [
            redact_jsonlike(cast(JsonLike, item), redactors, redact_keys=redact_keys)
            for item in ordered_items
        ]
    if isinstance(value, list):
        return [redact_jsonlike(item, redactors, redact_keys=redact_keys) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            redacted_key = redact_text(key, redactors) if redact_keys else key
            redacted_item: JsonLike
            if redactors and is_sensitive_name(str(key)):
                redacted_item = REDACTED
            else:
                redacted_item = redact_jsonlike(item, redactors, redact_keys=redact_keys)
            if redact_keys:
                _insert_redacted_key(out, redacted_key, redacted_item)
            else:
                out[redacted_key] = redacted_item
        return out
    return value


def truncate_str(s: str, max_len: int) -> str:
    """Truncate string with marker."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 12] + "...[TRUNC]"


def safe_repr(x: Any, cfg: SnapshotConfig) -> str:
    """Get repr with fallback for unreprable objects."""
    try:
        r = repr(x)
    except Exception:
        r = f"<unreprable {type(x).__name__}>"
    return truncate_str(r, cfg.max_str)


def is_array_like(x: Any) -> bool:
    """Check if object is an array type (numpy, jax, torch, etc)."""
    module = type(x).__module__
    return any(
        module.startswith(prefix) for prefix in ("numpy", "jax", "torch", "tensorflow", "cupy")
    ) and hasattr(x, "shape")


def is_dataframe(x: Any) -> bool:
    """Check if object is a pandas DataFrame."""
    return type(x).__module__.startswith("pandas") and type(x).__name__ == "DataFrame"


def is_series(x: Any) -> bool:
    """Check if object is a pandas Series."""
    return type(x).__module__.startswith("pandas") and type(x).__name__ == "Series"


def get_tensor_device(arr: Any) -> str | None:
    """Get device string for PyTorch/JAX tensors.

    Returns None for CPU (default) or if device cannot be determined.
    Only returns non-None for non-CPU devices to highlight noteworthy info.
    """
    try:
        # PyTorch: arr.device -> device(type='cuda', index=0)
        if hasattr(arr, "device"):
            device = arr.device
            # PyTorch device object
            if hasattr(device, "type"):
                if device.type != "cpu":
                    return str(device)
                return None
            # JAX SingleDeviceSharding or similar
            device_str = str(device)
            if "cpu" not in device_str.lower():
                return device_str
            return None
        # JAX: arr.devices() -> {gpu(id=0)}
        if hasattr(arr, "devices"):
            devices = arr.devices()
            if devices:
                # Check if any non-CPU device
                devices_str = str(devices)
                if "cpu" not in devices_str.lower():
                    return devices_str
    except Exception:  # nosec B110 — safe serialization fallback
        pass
    return None


def get_requires_grad(arr: Any) -> bool | None:
    """Check if tensor requires gradients (PyTorch only).

    Returns True only if requires_grad is True, None otherwise.
    """
    try:
        if hasattr(arr, "requires_grad") and arr.requires_grad:
            return True
    except Exception:  # nosec B110 — safe serialization fallback
        pass
    return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def get_array_element_count(arr: Any) -> int | None:
    """Estimate total number of elements for array-like objects."""
    size = getattr(arr, "size", None)
    if size is not None:
        size_int = _to_int(size)
        if size_int is not None and size_int >= 0:
            return size_int

    shape = getattr(arr, "shape", None)
    if isinstance(shape, (list, tuple)):
        total = 1
        for dim in cast(tuple[Any, ...], shape):
            dim_int = _to_int(dim)
            if dim_int is None or dim_int < 0:
                return None
            total *= dim_int
        return total

    return None


def _sample_array_head(arr: Any, max_items: int = 5) -> list[float]:
    """Return up to ``max_items`` scalar values from the start of an array."""
    flat = None
    if hasattr(arr, "reshape"):
        try:
            flat = arr.reshape(-1)
        except Exception:  # nosec B110 — safe serialization fallback
            flat = None
    if flat is None and hasattr(arr, "ravel"):
        try:
            flat = arr.ravel()
        except Exception:  # nosec B110 — safe serialization fallback
            flat = None
    if flat is None and hasattr(arr, "flatten"):
        flat = arr.flatten()
    if flat is None:
        raise TypeError("array does not support flatten/ravel/reshape")

    sample = flat[:max_items]
    raw: Any = sample.tolist() if hasattr(sample, "tolist") else list(sample)

    if not isinstance(raw, list):
        raw = [raw]
    values: list[Any] = cast(list[Any], raw)
    return [float(x) for x in values[:max_items]]


def count_anomalies(arr: Any) -> dict[str, int] | None:
    """Count NaN and Inf values in tensor.

    Returns dict with counts only if anomalies exist, None otherwise.
    Works with NumPy, PyTorch, and JAX arrays.
    """
    try:
        module = type(arr).__module__

        nan_count = 0
        inf_count = 0

        if module.startswith("torch"):
            # PyTorch
            import torch  # type: ignore[import-not-found]

            _torch: Any = torch
            nan_count = int(_torch.isnan(arr).sum().item())
            inf_count = int(_torch.isinf(arr).sum().item())
        elif module.startswith("jax"):
            # JAX
            import jax.numpy as jnp  # type: ignore[import-not-found]

            _jnp: Any = jnp
            nan_count = int(_jnp.isnan(arr).sum())
            inf_count = int(_jnp.isinf(arr).sum())
        elif module.startswith("numpy"):
            # NumPy
            import numpy as np

            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
        else:
            return None

        if nan_count > 0 or inf_count > 0:
            result: dict[str, int] = {}
            if nan_count > 0:
                result["nan"] = nan_count
            if inf_count > 0:
                result["inf"] = inf_count
            return result
    except Exception:  # nosec B110 — safe serialization fallback
        pass
    return None


def compute_array_stats(
    arr: Any,
    *,
    max_elements: int = MAX_ARRAY_ELEMENTS_FOR_STATS,
) -> dict[str, float] | None:
    """Compute min/max/mean/std statistics for numeric arrays.

    Returns:
        Dict with statistics rounded to 6 decimal places, or None if not applicable.
        Returns None for empty arrays, non-numeric dtypes, or on error.
    """
    try:
        element_count = get_array_element_count(arr)
        if element_count is not None and element_count > max_elements:
            return None

        module = type(arr).__module__

        # Check if numeric dtype
        dtype_str = str(getattr(arr, "dtype", ""))
        # Complex types are intentionally excluded: "min/max/mean/std" are ambiguous
        # without choosing a projection (real/imag/magnitude), and converting to float
        # would silently drop information.
        if not any(t in dtype_str for t in ("float", "int", "uint")):
            return None

        # Check for empty array
        if hasattr(arr, "size") and arr.size == 0:
            return None

        if module.startswith("torch"):
            # PyTorch - move to CPU for stats
            import torch  # type: ignore[import-not-found]

            _torch: Any = torch
            arr_cpu = arr.detach().cpu().float()
            return {
                "min": round(float(_torch.min(arr_cpu).item()), 6),
                "max": round(float(_torch.max(arr_cpu).item()), 6),
                "mean": round(float(_torch.mean(arr_cpu).item()), 6),
                # Match NumPy/JAX default semantics (population std, not unbiased).
                "std": round(float(_torch.std(arr_cpu, unbiased=False).item()), 6),
            }
        elif module.startswith("jax"):
            # JAX
            import jax.numpy as jnp  # type: ignore[import-not-found]

            _jnp: Any = jnp
            arr_float = arr.astype(_jnp.float32)
            return {
                "min": round(float(_jnp.min(arr_float)), 6),
                "max": round(float(_jnp.max(arr_float)), 6),
                "mean": round(float(_jnp.mean(arr_float)), 6),
                "std": round(float(_jnp.std(arr_float)), 6),
            }
        elif module.startswith(("numpy", "cupy")):
            # NumPy/CuPy
            import numpy as np

            # Handle CuPy by converting to numpy
            if module.startswith("cupy"):
                arr = arr.get()
            arr_float = arr.astype(np.float64)
            return {
                "min": round(float(np.min(arr_float)), 6),
                "max": round(float(np.max(arr_float)), 6),
                "mean": round(float(cast(Any, np.mean(arr_float))), 6),
                "std": round(float(np.std(arr_float)), 6),
            }
    except Exception:  # nosec B110 — safe serialization fallback
        pass
    return None


def _compute_array_stats_and_anomalies(
    arr: Any,
    *,
    max_elements: int = MAX_ARRAY_ELEMENTS_FOR_STATS,
) -> tuple[dict[str, float] | None, dict[str, int] | None]:
    """Compute stats + anomaly counts in one backend pass.

    Intended for summarize_array(include_array_stats=True) to avoid duplicate scans.
    """
    try:
        if _array_exceeds_limit(arr, max_elements=max_elements):
            return None, None
        if _is_empty_array(arr):
            return None, None

        module = type(arr).__module__
        is_numeric_dtype = _is_numeric_array_dtype(arr)
        # Dispatch to backend-specific implementations by module prefix.
        # Order doesn't matter — these frameworks never share module namespaces.
        if module.startswith("torch"):
            return _torch_stats_and_anomalies(arr, is_numeric_dtype=is_numeric_dtype)
        if module.startswith("jax"):
            return _jax_stats_and_anomalies(arr, is_numeric_dtype=is_numeric_dtype)
        if module.startswith(("numpy", "cupy")):
            return _numpy_like_stats_and_anomalies(
                arr, module=module, is_numeric_dtype=is_numeric_dtype
            )
    except Exception:  # nosec B110 — safe serialization fallback
        pass
    return None, None


def _array_exceeds_limit(arr: Any, *, max_elements: int) -> bool:
    element_count = get_array_element_count(arr)
    return element_count is not None and element_count > max_elements


def _is_empty_array(arr: Any) -> bool:
    return bool(hasattr(arr, "size") and arr.size == 0)


def _is_numeric_array_dtype(arr: Any) -> bool:
    dtype_str = str(getattr(arr, "dtype", ""))
    return any(token in dtype_str for token in ("float", "int", "uint"))


def _build_anomaly_counts(nan_count: int, inf_count: int) -> dict[str, int] | None:
    if nan_count <= 0 and inf_count <= 0:
        return None
    anomalies: dict[str, int] = {}
    if nan_count > 0:
        anomalies["nan"] = nan_count
    if inf_count > 0:
        anomalies["inf"] = inf_count
    return anomalies


def _torch_stats_and_anomalies(
    arr: Any, *, is_numeric_dtype: bool
) -> tuple[dict[str, float] | None, dict[str, int] | None]:
    import torch  # type: ignore[import-not-found]

    _torch: Any = torch
    arr_float = arr.detach().cpu().float()
    anomalies = _build_anomaly_counts(
        int(_torch.isnan(arr_float).sum().item()),
        int(_torch.isinf(arr_float).sum().item()),
    )
    if not is_numeric_dtype:
        return None, anomalies
    stats = {
        "min": round(float(_torch.min(arr_float).item()), 6),
        "max": round(float(_torch.max(arr_float).item()), 6),
        "mean": round(float(_torch.mean(arr_float).item()), 6),
        # Match NumPy/JAX default semantics (population std, not unbiased).
        "std": round(float(_torch.std(arr_float, unbiased=False).item()), 6),
    }
    return stats, anomalies


def _jax_stats_and_anomalies(
    arr: Any, *, is_numeric_dtype: bool
) -> tuple[dict[str, float] | None, dict[str, int] | None]:
    import jax.numpy as jnp  # type: ignore[import-not-found]

    _jnp: Any = jnp
    arr_float = arr.astype(_jnp.float32)
    anomalies = _build_anomaly_counts(
        int(_jnp.isnan(arr_float).sum()),
        int(_jnp.isinf(arr_float).sum()),
    )
    if not is_numeric_dtype:
        return None, anomalies
    stats = {
        "min": round(float(_jnp.min(arr_float)), 6),
        "max": round(float(_jnp.max(arr_float)), 6),
        "mean": round(float(_jnp.mean(arr_float)), 6),
        "std": round(float(_jnp.std(arr_float)), 6),
    }
    return stats, anomalies


def _numpy_like_stats_and_anomalies(
    arr: Any, *, module: str, is_numeric_dtype: bool
) -> tuple[dict[str, float] | None, dict[str, int] | None]:
    import numpy as np

    if module.startswith("cupy"):
        arr = arr.get()
    arr_float = arr.astype(np.float64)

    anomalies = None
    if module.startswith("numpy"):
        anomalies = _build_anomaly_counts(
            int(np.isnan(arr_float).sum()),
            int(np.isinf(arr_float).sum()),
        )

    if not is_numeric_dtype:
        return None, anomalies
    stats = {
        "min": round(float(np.min(arr_float)), 6),
        "max": round(float(np.max(arr_float)), 6),
        "mean": round(float(cast(Any, np.mean(arr_float))), 6),
        "std": round(float(np.std(arr_float)), 6),
    }
    return stats, anomalies


def _add_head_sample(
    summary: dict[str, Any],
    arr: Any,
    element_count: int | None,
    *,
    is_non_cpu: bool,
) -> None:
    """Add head sample (or omission reason) to the array summary."""
    if is_non_cpu:
        summary["head_omitted_reason"] = "non_cpu_device"
    elif element_count is not None and element_count > MAX_ARRAY_ELEMENTS_FOR_HEAD:
        summary["head_omitted_reason"] = "too_large"
    else:
        try:
            summary["head"] = _sample_array_head(arr, max_items=5)
            if element_count is not None and element_count > 5:
                summary["head_truncated"] = True
        except Exception as e:
            summary["head"] = f"<error: {type(e).__name__}>"


def _add_anomalies(
    summary: dict[str, Any],
    arr: Any,
    element_count: int | None,
) -> None:
    """Add anomaly counts (or omission reason) to the array summary."""
    if element_count is not None and element_count > MAX_ARRAY_ELEMENTS_FOR_ANOMALIES:
        summary["anomalies_omitted_reason"] = "too_large"
    else:
        anomalies = count_anomalies(arr)
        if anomalies:
            summary["anomalies"] = anomalies


def _add_array_stats(
    summary: dict[str, Any],
    arr: Any,
    cfg: SnapshotConfig,
    element_count: int | None,
    *,
    is_non_cpu: bool,
) -> None:
    """Add statistical summary (or omission reason) to the array summary."""
    if not cfg.include_array_stats:
        return
    if is_non_cpu:
        summary["stats_omitted_reason"] = "non_cpu_device"
    elif element_count is not None and element_count > MAX_ARRAY_ELEMENTS_FOR_STATS:
        summary["stats_omitted_reason"] = "too_large"
    else:
        stats = compute_array_stats(arr)
        if stats:
            summary["stats"] = stats


def summarize_array(arr: Any, cfg: SnapshotConfig) -> dict[str, Any]:
    """Summarize an array-like object.

    Returns basic info (type, shape, dtype, head) plus ML-specific metadata:
    - device: only if non-CPU (highlights CUDA/TPU tensors)
    - requires_grad: only if True (highlights trainable tensors)
    - anomalies: NaN/Inf counts only if present (highlights corruption)
    """
    type_name = f"{type(arr).__module__}.{type(arr).__name__}"
    summary: dict[str, Any] = {"__array__": type_name}

    if hasattr(arr, "shape"):
        try:
            summary["shape"] = list(arr.shape)
        except Exception as e:
            summary["shape"] = f"<error: {type(e).__name__}>"

    if hasattr(arr, "dtype"):
        try:
            summary["dtype"] = str(arr.dtype)
        except Exception as e:
            summary["dtype"] = f"<error: {type(e).__name__}>"

    device = get_tensor_device(arr)
    if device is not None:
        summary["device"] = device
    is_non_cpu = device is not None
    element_count = get_array_element_count(arr)

    _add_head_sample(summary, arr, element_count, is_non_cpu=is_non_cpu)

    requires_grad = get_requires_grad(arr)
    if requires_grad:
        summary["requires_grad"] = True

    used_combined_stats_path = False
    if (
        cfg.include_array_stats
        and not is_non_cpu
        and not (element_count is not None and element_count > MAX_ARRAY_ELEMENTS_FOR_ANOMALIES)
        and not (element_count is not None and element_count > MAX_ARRAY_ELEMENTS_FOR_STATS)
    ):
        stats, anomalies = _compute_array_stats_and_anomalies(arr)
        if anomalies:
            summary["anomalies"] = anomalies
        if stats:
            summary["stats"] = stats
        used_combined_stats_path = True

    if not used_combined_stats_path:
        _add_anomalies(summary, arr, element_count)
        _add_array_stats(summary, arr, cfg, element_count, is_non_cpu=is_non_cpu)

    return summary


def _preview_cell(value: Any, cfg: SnapshotConfig) -> Any:
    """Keep a pandas preview cell as-is unless it is an int orjson can't encode."""
    if isinstance(value, int) and not JSON_INT_MIN <= value <= JSON_INT_MAX:
        return safe_repr(value, cfg)
    return value


def summarize_dataframe(df: Any, cfg: SnapshotConfig) -> dict[str, Any]:
    """Summarize a pandas DataFrame."""
    summary: dict[str, Any] = {"__dataframe__": True}

    try:
        summary["shape"] = list(df.shape)
    except Exception as e:
        summary["shape"] = f"<error: {type(e).__name__}>"

    try:
        summary["columns"] = list(df.columns)[: cfg.max_items]
        if len(df.columns) > cfg.max_items:
            summary["columns_truncated"] = True
    except Exception as e:
        summary["columns"] = f"<error: {type(e).__name__}>"

    try:
        dtypes_raw = getattr(df, "dtypes", None)
        if dtypes_raw is not None:
            dtype_items = list(dtypes_raw.items()) if hasattr(dtypes_raw, "items") else []
            if dtype_items:
                summary["dtypes"] = {
                    str(column): str(dtype) for column, dtype in dtype_items[: cfg.max_items]
                }
                if len(dtype_items) > cfg.max_items:
                    summary["dtypes_truncated"] = True
    except Exception:  # nosec B110 — safe dataframe metadata fallback
        pass

    try:
        isna = getattr(df, "isna", None)
        if callable(isna):
            null_mask = cast(Any, isna())
            sum_method = getattr(null_mask, "sum", None)
            if callable(sum_method):
                null_counts_obj = cast(Any, sum_method())
                items_method = getattr(null_counts_obj, "items", None)
                if callable(items_method):
                    null_items = cast(list[tuple[Any, Any]], list(cast(Any, items_method())))
                    rendered_nulls = {
                        str(column): count_int
                        for column, count in null_items[: cfg.max_items]
                        if not isinstance(count, bool) and (count_int := int(count)) > 0
                    }
                    if rendered_nulls:
                        summary["null_counts"] = rendered_nulls
                    if len(null_items) > cfg.max_items:
                        summary["null_counts_truncated"] = True
    except Exception:  # nosec B110 — safe dataframe metadata fallback
        pass

    try:
        head_rows = min(3, len(df))
        rows = cast(list[list[Any]], df.head(head_rows).values.tolist())
        summary["head"] = [[_preview_cell(cell, cfg) for cell in row] for row in rows]
    except Exception as e:
        summary["head"] = f"<error: {type(e).__name__}>"

    return summary


def summarize_series(series: Any, cfg: SnapshotConfig) -> dict[str, Any]:
    """Summarize a pandas Series."""
    summary: dict[str, Any] = {"__series__": True}

    try:
        summary["shape"] = list(series.shape)
    except Exception as e:
        summary["shape"] = f"<error: {type(e).__name__}>"

    if hasattr(series, "name") and series.name is not None:
        summary["name"] = str(series.name)

    if hasattr(series, "dtype"):
        summary["dtype"] = str(series.dtype)

    try:
        cells = cast(list[Any], series.head(5).tolist())
        summary["head"] = [_preview_cell(cell, cfg) for cell in cells]
        if len(series) > 5:
            summary["head_truncated"] = True
    except Exception as e:
        summary["head"] = f"<error: {type(e).__name__}>"

    return summary


def _jsonlike_collection(x: Any, cfg: SnapshotConfig, depth: int) -> JsonLike:
    """Serialize list/tuple/set/frozenset to JSON-like form."""
    if isinstance(x, (list, tuple)):
        items: list[Any] = list(cast(list[Any], x)[: cfg.max_items + 1])
    else:
        ordered: list[Any] = sorted(
            x,
            key=lambda item: (type(item).__module__, type(item).__name__, safe_repr(item, cfg)),
        )
        items = ordered[: cfg.max_items + 1]
    truncated = len(items) > cfg.max_items
    if truncated:
        items = items[: cfg.max_items]
    result: list[Any] = [to_jsonlike(i, cfg, depth + 1) for i in items]
    if truncated:
        result.append("...[TRUNC]")
    return result


def _jsonlike_dict(x: dict[Any, Any], cfg: SnapshotConfig, depth: int) -> JsonLike:
    """Serialize dict to JSON-like form."""
    out: dict[str, Any] = {}
    for i, (k, v) in enumerate(x.items()):
        if i >= cfg.max_items:
            _insert_redacted_key(out, "...[TRUNC]", f"+{len(x) - cfg.max_items} more keys")
            break
        # str keys stay plain (str.__str__ also unwraps str subclasses); others use repr.
        # A collision such as {1: ..., "1": ...} keeps both entries as "1" and "1#2".
        key = truncate_str(str.__str__(k), cfg.max_str) if isinstance(k, str) else safe_repr(k, cfg)
        _insert_redacted_key(out, key, to_jsonlike(v, cfg, depth + 1))
    return out


def _jsonlike_object(x: Any, cfg: SnapshotConfig, depth: int) -> JsonLike:
    """Serialize an object with __dict__ to JSON-like form."""
    d = getattr(x, "__dict__", {})
    if not isinstance(d, dict):
        return safe_repr(x, cfg)
    d_typed: dict[str, Any] = cast(dict[str, Any], d)
    out: dict[str, Any] = {"__type__": type(x).__name__}
    for i, (k, v) in enumerate(d_typed.items()):
        if i >= min(cfg.max_items, 20):
            out["...[TRUNC]"] = "more fields omitted"
            break
        out[str(k)] = to_jsonlike(v, cfg, depth + 1)
    return out


def _serialization_failed(error: Exception) -> str:
    return f"<serialization failed: {type(error).__name__}>"


def to_jsonlike(x: Any, cfg: SnapshotConfig, depth: int = 0) -> JsonLike:
    """Convert arbitrary Python object to JSON-serializable form.

    Never raises: a value whose introspection fails (for example a
    ``__getattribute__`` that raises) becomes a ``<serialization failed: ...>``
    marker, so one hostile object can't abort the capture or hide its siblings.
    """
    try:
        return _to_jsonlike(x, cfg, depth)
    except Exception as e:
        return _serialization_failed(e)


def _to_jsonlike(x: Any, cfg: SnapshotConfig, depth: int) -> JsonLike:
    if x is None or isinstance(x, (bool, float)):
        return x
    if isinstance(x, int):
        return x if JSON_INT_MIN <= x <= JSON_INT_MAX else safe_repr(x, cfg)
    if isinstance(x, str):
        return truncate_str(x, cfg.max_str)
    if depth >= 2:
        return safe_repr(x, cfg)
    if is_array_like(x):
        return summarize_array(x, cfg)
    if is_dataframe(x):
        return summarize_dataframe(x, cfg)
    if is_series(x):
        return summarize_series(x, cfg)
    if isinstance(x, (list, tuple, set, frozenset)):
        return _jsonlike_collection(x, cfg, depth)
    if isinstance(x, dict):
        return _jsonlike_dict(cast(dict[Any, Any], x), cfg, depth)
    if depth < 1 and hasattr(x, "__dict__"):
        return _jsonlike_object(x, cfg, depth)
    return safe_repr(x, cfg)


def serialize_locals(
    local_vars: dict[str, Any], cfg: SnapshotConfig, redactors: list[Pattern[str]]
) -> dict[str, Any]:
    """Serialize local variables with redaction."""
    result, _, _ = serialize_locals_with_stats(local_vars, cfg, redactors)
    return result


def _local_type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def _build_local_metadata_entry(value: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"type": _local_type_name(value)}
    try:
        entry["len"] = len(value)  # type: ignore[arg-type]
    except Exception:  # nosec B110 — safe serialization fallback
        pass
    try:
        if (is_array_like(value) or is_dataframe(value) or is_series(value)) and hasattr(
            value, "shape"
        ):
            entry["shape"] = list(value.shape)
    except Exception as e:
        entry["shape"] = f"<error: {type(e).__name__}>"
    return entry


def _serialize_local_value(value: Any, cfg: SnapshotConfig) -> JsonLike:
    """Serialize one local; modules, functions and classes collapse to a type tag."""
    try:
        is_definition = (
            inspect.ismodule(value) or inspect.isfunction(value) or inspect.isclass(value)
        )
    except Exception as e:
        return _serialization_failed(e)
    return f"<{type(value).__name__}>" if is_definition else to_jsonlike(value, cfg)


def serialize_locals_bundle(
    local_vars: dict[str, Any],
    cfg: SnapshotConfig,
    redactors: list[Pattern[str]],
) -> LocalsSerializationBundle:
    """Serialize locals and metadata in one pass over the frame locals."""
    values: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}

    total_count: int | None = None
    try:
        total_count = len(local_vars)
    except Exception:
        total_count = None

    for i, (key, value) in enumerate(local_vars.items()):
        if i >= cfg.max_items:
            break
        if redactors and is_sensitive_name(str(key)):
            # Never repr a secret, and keep its length out of the metadata.
            values[key] = REDACTED
            metadata[str(key)] = {"type": _local_type_name(value)}
            continue
        values[key] = _serialize_local_value(value, cfg)
        metadata[str(key)] = _build_local_metadata_entry(value)

    cleaned = redact_jsonlike(values, redactors, redact_keys=bool(cfg.redact_keys))
    if not isinstance(cleaned, dict):
        raise TypeError(f"redact_jsonlike returned {type(cleaned).__name__}, expected dict")
    cleaned_metadata: dict[str, dict[str, Any]] = metadata
    if cfg.redact_keys:
        cleaned_metadata = {}
        for key in values:
            key_text = str(key)
            redacted_key = redact_text(key_text, redactors)
            _insert_redacted_key(cleaned_metadata, redacted_key, metadata.get(key_text, {}))

    truncated = False
    truncated_keys = 0
    if total_count is not None and total_count > cfg.max_items:
        truncated = True
        truncated_keys = total_count - cfg.max_items

    return LocalsSerializationBundle(
        values=cleaned,
        metadata=cleaned_metadata,
        truncated=truncated,
        truncated_keys=truncated_keys,
    )


def serialize_locals_with_stats(
    local_vars: dict[str, Any], cfg: SnapshotConfig, redactors: list[Pattern[str]]
) -> tuple[dict[str, Any], bool, int]:
    """Serialize local values with redaction and truncation stats.

    Kept as compatibility wrapper while internals use :func:`serialize_locals_bundle`.
    """
    bundle = serialize_locals_bundle(local_vars, cfg, redactors)
    return bundle.values, bundle.truncated, bundle.truncated_keys


def locals_metadata(local_vars: dict[str, Any], cfg: SnapshotConfig) -> dict[str, dict[str, Any]]:
    """Lightweight metadata for local variables (type + size hints).

    Kept as compatibility wrapper while internals use :func:`serialize_locals_bundle`.
    """
    meta: dict[str, dict[str, Any]] = {}

    for i, (k, v) in enumerate(local_vars.items()):
        if i >= cfg.max_items:
            break
        meta[str(k)] = _build_local_metadata_entry(v)

    return meta
