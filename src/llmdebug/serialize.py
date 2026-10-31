"""Safe serialization with smart array handling."""

from __future__ import annotations

import collections
import copy
import functools
import heapq
import inspect
import re
from dataclasses import dataclass
from re import Pattern
from typing import Any, NamedTuple, cast

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
        truncated: ``True`` when the locals exceeded ``max_items`` and the result was clipped.
        truncated_keys: Number of locals dropped by the ``max_items`` limit (zero if not
            *truncated*).
    """

    values: dict[str, Any]
    metadata: dict[str, dict[str, Any]]
    truncated: bool
    truncated_keys: int


def compile_redactors(patterns: tuple[str | Pattern[str], ...] | None) -> list[Pattern[str]]:
    """Compile regex patterns for redaction; invalid patterns are reported and skipped.

    Args:
        patterns: Pattern strings or precompiled patterns; ``None`` means none.

    Returns:
        A fresh list of compiled patterns that callers may mutate.
    """
    # A fresh list per call: callers may mutate it without touching the shared cache.
    return list(_compile_redactors_cached(tuple(patterns or ())))


# lru_cache keeps its bookkeeping consistent across threads (threading.excepthook can capture
# concurrently); the former one-slot cache stored patterns and result in two statements (CAP-07).
@functools.lru_cache(maxsize=32)
def _compile_redactors_cached(
    patterns: tuple[str | Pattern[str], ...],
) -> tuple[Pattern[str], ...]:
    compiled: list[Pattern[str]] = []
    for p in patterns:
        if isinstance(p, str):
            try:
                compiled.append(re.compile(p))
            except re.error as e:
                import sys

                sys.stderr.write(f"llmdebug: invalid redaction pattern {p!r}: {e}\n")
        else:
            compiled.append(p)
    return tuple(compiled)


REDACTED = "[REDACTED]"

# CAP-03: matched against local-variable names and dict keys whenever any redaction pattern
# is active; the value under a matching name is replaced whole.
SENSITIVE_NAME_PATTERN = re.compile(f"(?i){SENSITIVE_NAMES}")


def is_sensitive_name(name: str) -> bool:
    """Return whether a variable or key name marks its value as a secret.

    Args:
        name: Variable, argument, attribute, or dict key name.

    Returns:
        True if the name contains a ``SENSITIVE_NAMES`` term (case-insensitive).
    """
    return SENSITIVE_NAME_PATTERN.search(name) is not None


def _is_sensitive_pair_name(first: Any) -> bool:
    return isinstance(first, str) and is_sensitive_name(first)


def redact_text(text: str, redactors: list[Pattern[str]]) -> str:
    """Apply redaction patterns to text.

    Args:
        text: Text to redact.
        redactors: Compiled patterns, applied in order.

    Returns:
        The text with every match replaced by ``[REDACTED]``.
    """
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
    """Recursively redact string leaf values (and optionally keys) in JSON-like data.

    This avoids the fragile serialize-then-regex-then-parse flow, where
    aggressive patterns can corrupt JSON structure. When any redactor is
    active, the whole value under a dict key with a sensitive name (see
    :func:`is_sensitive_name`) is replaced, whatever its type, and so is the
    second item of a ``[name, value]`` pair (a header or form field).

    Args:
        value: JSON-like data; tuples become lists and sets become stably sorted lists.
        redactors: Compiled patterns; an empty list disables name-based redaction too.
        redact_keys: Also redact dict keys; colliding keys get ``#2``-style suffixes.

    Returns:
        A redacted copy of *value*.
    """
    if isinstance(value, str):
        return redact_text(value, redactors)
    if isinstance(value, (tuple, list)):
        sequence = cast(list[Any] | tuple[Any, ...], value)
        items: list[Any] = [
            redact_jsonlike(cast(JsonLike, item), redactors, redact_keys=redact_keys)
            for item in sequence
        ]
        if redactors and len(items) == 2 and _is_sensitive_pair_name(sequence[0]):
            items[1] = REDACTED
        return items
    if isinstance(value, (set, frozenset)):
        ordered_items = sorted(
            cast(set[Any] | frozenset[Any], value), key=_stable_redaction_sort_key
        )
        return [
            redact_jsonlike(cast(JsonLike, item), redactors, redact_keys=redact_keys)
            for item in ordered_items
        ]
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
    """Truncate a string with a marker.

    Args:
        s: String to truncate.
        max_len: Length above which *s* is truncated.

    Returns:
        *s* if it fits, else its first ``max_len - 12`` characters plus ``...[TRUNC]``.
    """
    if len(s) <= max_len:
        return s
    return s[: max_len - 12] + "...[TRUNC]"


def safe_repr(x: Any, cfg: SnapshotConfig) -> str:
    """Return a truncated ``repr`` with a fallback for unreprable objects.

    Args:
        x: Object to render.
        cfg: Config supplying ``max_str``.

    Returns:
        The repr cut to ``max_str``, or ``<unreprable TypeName>`` if ``repr()`` raises.
    """
    try:
        r = repr(x)
    except Exception:
        r = f"<unreprable {type(x).__name__}>"
    return truncate_str(r, cfg.max_str)


def is_array_like(x: Any) -> bool:
    """Check if object is an array type (numpy, jax, torch, tensorflow, cupy).

    Args:
        x: Object to test.

    Returns:
        True if the type comes from one of those libraries and has a ``shape``.
    """
    module = type(x).__module__
    return any(
        module.startswith(prefix) for prefix in ("numpy", "jax", "torch", "tensorflow", "cupy")
    ) and hasattr(x, "shape")


def is_dataframe(x: Any) -> bool:
    """Check if object is a pandas DataFrame.

    Args:
        x: Object to test.

    Returns:
        True for a type named ``DataFrame`` from a ``pandas`` module.
    """
    return type(x).__module__.startswith("pandas") and type(x).__name__ == "DataFrame"


def is_series(x: Any) -> bool:
    """Check if object is a pandas Series.

    Args:
        x: Object to test.

    Returns:
        True for a type named ``Series`` from a ``pandas`` module.
    """
    return type(x).__module__.startswith("pandas") and type(x).__name__ == "Series"


def get_tensor_device(arr: Any) -> str | None:
    """Get device string for PyTorch/JAX tensors.

    Args:
        arr: Array-like object.

    Returns:
        The device string for non-CPU devices only, to highlight noteworthy info;
        ``None`` for CPU (default) or if the device cannot be determined.
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

    Args:
        arr: Array-like object.

    Returns:
        True only if ``requires_grad`` is truthy, ``None`` otherwise.
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
    """Estimate total number of elements for array-like objects.

    Args:
        arr: Array-like object.

    Returns:
        A non-negative ``size``, else the product of ``shape``, else ``None``.
    """
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
    """Count NaN and Inf values in a NumPy, CuPy, PyTorch or JAX array.

    Args:
        arr: Array to scan.

    Returns:
        Nonzero ``nan``/``inf`` counts if anomalies exist, ``None`` otherwise or on error.
    """
    try:
        return _stats_and_anomalies(arr, want_stats=False)[1]
    except Exception:  # nosec B110 — safe serialization fallback
        return None


def compute_array_stats(
    arr: Any,
    *,
    max_elements: int = MAX_ARRAY_ELEMENTS_FOR_STATS,
) -> dict[str, float] | None:
    """Compute min/max/mean/std statistics for numeric arrays.

    Args:
        arr: NumPy, CuPy, PyTorch or JAX array.
        max_elements: Arrays with more elements are skipped.

    Returns:
        Dict with statistics rounded to 6 decimal places, or None if not applicable.
        Returns None for oversized or empty arrays, non-numeric dtypes, or on error.
    """
    try:
        if _array_exceeds_limit(arr, max_elements=max_elements) or _is_empty_array(arr):
            return None
        return _stats_and_anomalies(arr, want_anomalies=False)[0]
    except Exception:  # nosec B110 — safe serialization fallback
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
        if _array_exceeds_limit(arr, max_elements=max_elements) or _is_empty_array(arr):
            return None, None
        return _stats_and_anomalies(arr)
    except Exception:  # nosec B110 — safe serialization fallback
        return None, None


def _stats_and_anomalies(
    arr: Any, *, want_stats: bool = True, want_anomalies: bool = True
) -> tuple[dict[str, float] | None, dict[str, int] | None]:
    """Compute array statistics and NaN/Inf counts in the single backend dispatch (CAP-23).

    ``count_anomalies`` and ``compute_array_stats`` used to be separate copies of the
    backend code and had drifted from this path: anomalies were counted on the up-cast
    copy (a float32 cast turns large float64 values into Inf) and CuPy got none.
    Complex dtypes get anomaly counts but no stats (min/max/mean/std need a projection).
    """
    module = type(arr).__module__
    is_numeric_dtype = want_stats and _is_numeric_array_dtype(arr)
    # Order doesn't matter — these frameworks never share module namespaces.
    if module.startswith("torch"):
        return _torch_stats_and_anomalies(
            arr, is_numeric_dtype=is_numeric_dtype, want_anomalies=want_anomalies
        )
    if module.startswith("jax"):
        return _jax_stats_and_anomalies(
            arr, is_numeric_dtype=is_numeric_dtype, want_anomalies=want_anomalies
        )
    if module.startswith(("numpy", "cupy")):
        return _numpy_like_stats_and_anomalies(
            arr, module=module, is_numeric_dtype=is_numeric_dtype, want_anomalies=want_anomalies
        )
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
    arr: Any, *, is_numeric_dtype: bool, want_anomalies: bool = True
) -> tuple[dict[str, float] | None, dict[str, int] | None]:
    import torch  # type: ignore[import-not-found]

    _torch: Any = torch
    tensor = arr.detach()
    anomalies = (
        _build_anomaly_counts(
            int(_torch.isnan(tensor).sum().item()),
            int(_torch.isinf(tensor).sum().item()),
        )
        if want_anomalies
        else None
    )
    if not is_numeric_dtype:
        return None, anomalies
    arr_float = tensor.cpu().float()
    stats = {
        "min": round(float(_torch.min(arr_float).item()), 6),
        "max": round(float(_torch.max(arr_float).item()), 6),
        "mean": round(float(_torch.mean(arr_float).item()), 6),
        # Match NumPy/JAX default semantics (population std, not unbiased).
        "std": round(float(_torch.std(arr_float, unbiased=False).item()), 6),
    }
    return stats, anomalies


def _jax_stats_and_anomalies(
    arr: Any, *, is_numeric_dtype: bool, want_anomalies: bool = True
) -> tuple[dict[str, float] | None, dict[str, int] | None]:
    import jax.numpy as jnp  # type: ignore[import-not-found]

    _jnp: Any = jnp
    anomalies = (
        _build_anomaly_counts(int(_jnp.isnan(arr).sum()), int(_jnp.isinf(arr).sum()))
        if want_anomalies
        else None
    )
    if not is_numeric_dtype:
        return None, anomalies
    arr_float = arr.astype(_jnp.float32)
    stats = {
        "min": round(float(_jnp.min(arr_float)), 6),
        "max": round(float(_jnp.max(arr_float)), 6),
        "mean": round(float(_jnp.mean(arr_float)), 6),
        "std": round(float(_jnp.std(arr_float)), 6),
    }
    return stats, anomalies


def _numpy_like_stats_and_anomalies(
    arr: Any, *, module: str, is_numeric_dtype: bool, want_anomalies: bool = True
) -> tuple[dict[str, float] | None, dict[str, int] | None]:
    import numpy as np

    if module.startswith("cupy"):
        arr = arr.get()
    anomalies = (
        _build_anomaly_counts(int(np.isnan(arr).sum()), int(np.isinf(arr).sum()))
        if want_anomalies
        else None
    )
    if not is_numeric_dtype:
        return None, anomalies
    arr_float = arr.astype(np.float64)
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

    Args:
        arr: Array-like object (see :func:`is_array_like`).
        cfg: Config; ``include_array_stats`` adds a ``stats`` entry.

    Returns:
        Basic info (type, shape, dtype, head) plus ML-specific metadata: ``device`` only
        if non-CPU, ``requires_grad`` only if True, and ``anomalies`` (NaN/Inf counts)
        only if present. Skipped parts get a ``*_omitted_reason`` entry.
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


def _preview_cell(value: Any, cfg: SnapshotConfig, depth: int = 0) -> Any:
    """Bound a pandas preview cell or column label like any other value (CAP-01, CAP-13).

    Object-dtype cells hold arbitrary Python values, so the normalization has to recurse:
    a wide integer nested in a cell's dict or list reached the JSON encoder unchanged,
    which raises and loses the whole snapshot rather than one value.
    """
    if isinstance(value, str):
        return truncate_str(value, cfg.max_str)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not JSON_INT_MIN <= value <= JSON_INT_MAX:
        return safe_repr(value, cfg)
    if isinstance(value, dict):
        if depth >= 2:
            return safe_repr(value, cfg)
        items = list(cast(dict[Any, Any], value).items())[: cfg.max_items]
        return {
            str(_preview_cell(key, cfg, depth + 1)): _preview_cell(item, cfg, depth + 1)
            for key, item in items
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        if depth >= 2:
            return safe_repr(value, cfg)
        entries = list(cast("set[Any]", value))[: cfg.max_items]
        return [_preview_cell(item, cfg, depth + 1) for item in entries]
    return value


def _add_dataframe_columns(summary: dict[str, Any], df: Any, cfg: SnapshotConfig) -> None:
    """Record the listed column labels and whether they were cut (CAP-01, CAP-13).

    Labels are arbitrary objects too, so they get the same normalization as preview
    cells: a wide-integer column label used to break the JSON encoder.
    """
    try:
        columns = list(cast(list[Any], df.columns))
        summary["columns"] = [_preview_cell(column, cfg) for column in columns[: cfg.max_items]]
        if len(columns) > cfg.max_items:
            summary["columns_truncated"] = True
    except Exception as e:
        summary["columns"] = f"<error: {type(e).__name__}>"


def summarize_dataframe(df: Any, cfg: SnapshotConfig) -> dict[str, Any]:
    """Summarize a pandas DataFrame.

    Args:
        df: DataFrame to summarize.
        cfg: Config whose ``max_items`` and ``max_str`` bound columns and cells.

    Returns:
        Shape, columns, dtypes, nonzero null counts, and up to 3 head rows; lists cut by
        ``max_items`` are flagged with ``*_truncated``.
    """
    summary: dict[str, Any] = {"__dataframe__": True}

    try:
        summary["shape"] = list(df.shape)
    except Exception as e:
        summary["shape"] = f"<error: {type(e).__name__}>"

    _add_dataframe_columns(summary, df, cfg)

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
        summary["head"] = _dataframe_head(df, cfg)
    except Exception as e:
        summary["head"] = f"<error: {type(e).__name__}>"

    return summary


def _dataframe_head(df: Any, cfg: SnapshotConfig) -> list[list[Any]]:
    # Only the columns listed in "columns" (max_items), not every column (CAP-13).
    head = df.iloc[: min(3, len(df)), : cfg.max_items]
    rows = cast(list[list[Any]], head.values.tolist())
    return [[_preview_cell(cell, cfg) for cell in row] for row in rows]


def summarize_series(series: Any, cfg: SnapshotConfig) -> dict[str, Any]:
    """Summarize a pandas Series.

    Args:
        series: Series to summarize.
        cfg: Config bounding the preview cells.

    Returns:
        Shape, name, dtype, and the first 5 values as ``head``.
    """
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


# Up to this size a set is sorted by repr, as always; larger sets are previewed without
# rendering every element (CAP-10: a 200,000-element set took 40-160 ms).
_SET_FULL_SORT_LIMIT = 1000


class _SetPreview(NamedTuple):
    """Up to ``max_items`` preview elements, plus whether the set held more."""

    items: list[Any]
    truncated: bool


def _set_preview(x: set[Any] | frozenset[Any], cfg: SnapshotConfig) -> _SetPreview:
    """Return the lowest ``max_items`` elements of a set, in a process-independent order.

    Truncation comes from the set's own size: inferring it from the preview length
    reported a 3,000-element set as complete whenever the preview hit an internal
    sampling cap first.
    """
    limit = cfg.max_items
    total = len(x)

    def by_repr(item: Any) -> tuple[str, str, str]:
        return (type(item).__module__, type(item).__name__, safe_repr(item, cfg))

    if total <= _SET_FULL_SORT_LIMIT:
        return _SetPreview(sorted(x, key=by_repr)[:limit], total > limit)
    try:
        return _SetPreview(heapq.nsmallest(limit, x), total > limit)
    except Exception:
        # Ordering the elements failed -- incomparable types, but also e.g. a
        # `Decimal("NaN")` raising `InvalidOperation`, which used to escape and turn the
        # whole local into a `<serialization failed>` marker. Falling back to canonical
        # repr order over the whole set costs one repr per element, but a sample taken in
        # set-iteration order changed contents (not just order) with the hash seed.
        return _SetPreview(heapq.nsmallest(limit, x, key=by_repr), total > limit)


def _jsonlike_collection(x: Any, cfg: SnapshotConfig, depth: int) -> JsonLike:
    """Serialize list/tuple/set/frozenset to JSON-like form."""
    if isinstance(x, (list, tuple)):
        items: list[Any] = list(cast(list[Any], x)[: cfg.max_items + 1])
        truncated = len(items) > cfg.max_items
        if truncated:
            items = items[: cfg.max_items]
    else:
        items, truncated = _set_preview(cast(set[Any] | frozenset[Any], x), cfg)
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
        if isinstance(k, str):
            name = str.__str__(k)
            key = truncate_str(name, cfg.max_str)
        else:
            name = key = safe_repr(k, cfg)
        # Sensitivity is decided on the untruncated name: a key longer than ``max_str``
        # loses its secret suffix, and the name-aware redaction that runs afterwards only
        # ever sees the truncated key (CAP-03).
        value: JsonLike = (
            REDACTED if cfg.redact and is_sensitive_name(name) else to_jsonlike(v, cfg, depth + 1)
        )
        _insert_redacted_key(out, key, value)
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


# Past the depth limit a container becomes a single repr string, which only the text patterns
# can inspect afterwards. Composite values survived them: the repr of
# {'password': ['alpha', 'beta']} kept "beta", and a nested ['Authorization', '<creds>'] pair
# has no "name: value" shape at all (CAP-03). Replacing those values structurally first keeps
# the surrounding shape of the repr readable without the secrets.
_REPR_REDACTION_MAX_DEPTH = 6


def _is_exact_type(x: Any, expected: type[Any]) -> bool:
    """Return ``type(x) is expected``; a subclass takes the rebuild path in `_redact_for_repr`."""
    return type(x) is expected


def _redacted_dict_for_repr(
    mapping: dict[Any, Any], cfg: SnapshotConfig, depth: int
) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for i, (key, item) in enumerate(mapping.items()):
        if i >= cfg.max_items:
            out["...[TRUNC]"] = f"+{len(mapping) - cfg.max_items} more keys"
            break
        if isinstance(key, str) and is_sensitive_name(key):
            out[key] = REDACTED
        else:
            out[key] = _redact_for_repr(item, cfg, depth + 1)
    return out


def _redacted_items_for_repr(
    sequence: list[Any] | tuple[Any, ...], cfg: SnapshotConfig, depth: int
) -> list[Any]:
    items: list[Any] = [
        _redact_for_repr(item, cfg, depth + 1) for item in sequence[: cfg.max_items]
    ]
    if len(sequence) == 2 and _is_sensitive_pair_name(sequence[0]):
        items[1] = REDACTED
    elif len(sequence) > cfg.max_items:
        items.append("...[TRUNC]")
    return items


def _redact_namedtuple_for_repr(x: Any, cfg: SnapshotConfig, depth: int) -> Any:
    """Return the namedtuple with fields under sensitive names replaced, keeping its type.

    Rebuilding it as a plain tuple would drop the field names from the repr, and the text
    patterns would then no longer see the ``password=`` that protects the value. Raising is
    the fail-closed outcome: the caller turns it into a marker rather than repr the original.
    """
    fields = cast(tuple[str, ...], x._fields)
    values = list(cast(tuple[Any, ...], x))
    replacement: dict[str, Any] = {}
    # strict=True: a _fields/values mismatch is not a namedtuple, and failing is closed.
    for i, (field, item) in enumerate(zip(fields, values, strict=True)):
        if is_sensitive_name(field):
            replacement[field] = REDACTED
        elif i >= cfg.max_items:
            replacement[field] = "...[TRUNC]"
        else:
            replacement[field] = _redact_for_repr(item, cfg, depth + 1)
    # The same pair rule as for a plain tuple: ("password", value) is a header or form field.
    if len(fields) == 2 and _is_sensitive_pair_name(values[0]):
        replacement[fields[1]] = REDACTED
    return x._replace(**replacement)


def _redact_container_subclass_for_repr(x: Any, redacted: Any) -> Any:
    """Return a dict/list/tuple subclass holding *redacted*, or a rendering of its type name.

    Never returns *x*: a subclass may define its own ``__repr__``, and a custom ``redact``
    tuple need not contain the text patterns at all, so leaving its contents in place would
    fail open.
    """
    try:
        clone = copy.copy(x)
        if isinstance(clone, dict):
            mapping = cast(dict[Any, Any], clone)
            mapping.clear()
            mapping.update(cast(dict[Any, Any], redacted))
            return cast(Any, clone)
        if isinstance(clone, list):
            sequence = cast(list[Any], clone)
            sequence[:] = cast(list[Any], redacted)
            return cast(Any, clone)
        return type(x)(redacted)
    except Exception:  # nosec B110 — the fallback below never keeps the original contents
        pass
    try:
        return f"{type(x).__name__}({redacted!r})"
    except Exception:
        return REDACTED


def _redact_for_repr(x: Any, cfg: SnapshotConfig, depth: int) -> Any:
    """Return a copy of *x* with values under sensitive names replaced, for repr rendering.

    Exact ``dict``/``list``/``tuple`` instances are rebuilt as themselves and a namedtuple
    keeps its type and field names. Any other container subclass is rebuilt through
    :func:`_redact_container_subclass_for_repr`, which never hands back the original.
    """
    if depth >= _REPR_REDACTION_MAX_DEPTH:
        return x
    if isinstance(x, dict):
        redacted_mapping = _redacted_dict_for_repr(cast(dict[Any, Any], x), cfg, depth)
        if _is_exact_type(x, dict):
            return redacted_mapping
        return _redact_container_subclass_for_repr(x, redacted_mapping)
    if hasattr(x, "_fields") and hasattr(x, "_replace") and isinstance(x, tuple):
        return _redact_namedtuple_for_repr(cast(Any, x), cfg, depth)
    if isinstance(x, (list, tuple)):
        items = _redacted_items_for_repr(cast("list[Any] | tuple[Any, ...]", x), cfg, depth)
        if _is_exact_type(x, list):
            return items
        if _is_exact_type(x, tuple):
            return tuple(items)
        return _redact_container_subclass_for_repr(x, items)
    return x


def _depth_limit_repr(x: Any, cfg: SnapshotConfig) -> str:
    """Return ``safe_repr`` of a value past the depth limit, with secrets removed first."""
    if not cfg.redact:
        return safe_repr(x, cfg)
    try:
        reduced = _redact_for_repr(x, cfg, 0)
    except Exception as e:
        # Fail closed: a container that cannot be walked is not repr'd either.
        return _serialization_failed(e)
    return safe_repr(reduced, cfg)


def to_jsonlike(x: Any, cfg: SnapshotConfig, depth: int = 0) -> JsonLike:
    """Convert arbitrary Python object to JSON-serializable form.

    Never raises: a value whose introspection fails (for example a
    ``__getattribute__`` that raises) becomes a ``<serialization failed: ...>``
    marker, so one hostile object can't abort the capture or hide its siblings.

    Args:
        x: Value to convert.
        cfg: Config bounding strings and collections and enabling redaction.
        depth: Nesting level; from depth 2 containers collapse to a redacted repr.

    Returns:
        A JSON-like value; arrays and pandas objects become summary dicts.
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
        return _depth_limit_repr(x, cfg)
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
    """Serialize local variables with redaction.

    Args:
        local_vars: Frame locals mapping.
        cfg: Snapshot configuration.
        redactors: Compiled redaction patterns.

    Returns:
        Serialized values of the first ``max_items`` locals, keyed by name.
    """
    result, _, _ = serialize_locals_with_stats(local_vars, cfg, redactors)
    return result


def _local_type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


# Types whose len() is known to be cheap and side-effect free. Calling len() on arbitrary
# objects ran user code: a lazy ORM query set executes its query (CAP-21).
_SAFE_LEN_TYPES: tuple[type, ...] = (
    str,
    bytes,
    bytearray,
    list,
    tuple,
    dict,
    set,
    frozenset,
    range,
    collections.deque,
)


def _has_builtin_len(value: Any) -> bool:
    """Return whether ``value`` is a safe container that has not overridden ``__len__``.

    An ``isinstance`` check alone still ran user code: a ``list`` subclass that
    overrides ``__len__`` passed it, which is exactly the arbitrary call CAP-21 removed
    from metadata-only mode.
    """
    value_type = cast("type[Any]", type(value))
    if value_type in _SAFE_LEN_TYPES:
        return True
    if not isinstance(value, _SAFE_LEN_TYPES):
        return False
    own_len = cast(object, getattr(value_type, "__len__", None))
    return any(
        own_len is cast(object, getattr(base, "__len__", None))
        for base in _SAFE_LEN_TYPES
        if isinstance(value, base)
    )


def _safe_len(value: Any) -> int | None:
    """Return ``len(value)`` for containers and array-likes only; ``None`` otherwise or on error."""
    try:
        if (
            _has_builtin_len(value)
            or is_array_like(value)
            or is_dataframe(value)
            or is_series(value)
        ):
            return len(value)
    except Exception:  # nosec B110 — safe serialization fallback
        pass
    return None


def _build_local_metadata_entry(value: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"type": _local_type_name(value)}
    length = _safe_len(value)
    if length is not None:
        entry["len"] = length
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
    """Serialize locals and metadata in one pass over the frame locals.

    Args:
        local_vars: Frame locals mapping.
        cfg: Config supplying limits and ``redact_keys``.
        redactors: Compiled patterns; if any, secret-named locals become ``[REDACTED]``.

    Returns:
        Values, metadata, and truncation info for the first ``max_items`` locals.
    """
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

    Args:
        local_vars: Frame locals mapping.
        cfg: Snapshot configuration.
        redactors: Compiled redaction patterns.

    Returns:
        A ``(values, truncated, truncated_keys)`` tuple.
    """
    bundle = serialize_locals_bundle(local_vars, cfg, redactors)
    return bundle.values, bundle.truncated, bundle.truncated_keys


def locals_metadata(
    local_vars: dict[str, Any],
    cfg: SnapshotConfig,
    *,
    redactors: list[Pattern[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build lightweight metadata for local variables (type + size hints).

    Used by metadata-only capture (``locals_mode="meta"``); values are not serialized.

    Args:
        local_vars: Frame locals mapping.
        cfg: Config whose ``max_items`` caps the number of entries.
        redactors: Compiled patterns; if any, secret-named locals keep only their
            ``type`` so that ``len`` and ``shape`` cannot leak the size of a secret.

    Returns:
        Per-name ``type``, plus ``len`` and ``shape`` where cheaply available.
    """
    meta: dict[str, dict[str, Any]] = {}

    for i, (k, v) in enumerate(local_vars.items()):
        if i >= cfg.max_items:
            break
        if redactors and is_sensitive_name(str(k)):
            meta[str(k)] = {"type": _local_type_name(v)}
            continue
        meta[str(k)] = _build_local_metadata_entry(v)

    return meta
