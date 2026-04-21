"""Hypothesis generation engine — auto-generate debugging hypotheses from snapshots.

Pure logic module with zero external dependencies. Follows the snapshot_diff.py
architecture: frozen dataclasses, pure functions, text formatter.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from ._debug_session import coerce_crash_frame_index
from .snapshot_types import ExceptionDict, FrameDict, SnapshotDict

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_ARRAY_KEYS = ("__array__", "__dataframe__", "__series__")


@dataclass(frozen=True)
class Hypothesis:
    """A ranked debugging hypothesis derived from snapshot patterns."""

    confidence: float  # 0.0-1.0
    pattern: str  # machine-readable: "empty_array", "shape_mismatch", etc.
    description: str  # human-readable explanation
    evidence: tuple[str, ...]  # data points from snapshot
    suggestion: str  # what to do next


# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------

_detectors: list[Callable[[dict[str, Any]], list[Hypothesis]]] = []


def _register_detector(
    fn: Callable[[dict[str, Any]], list[Hypothesis]],
) -> Callable[[dict[str, Any]], list[Hypothesis]]:
    _detectors.append(fn)
    return fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_crash_frame(snapshot: dict[str, Any]) -> FrameDict | None:
    """Extract the crash frame from a snapshot."""
    view = cast(SnapshotDict, snapshot)
    frames_obj = view.get("frames")
    if not isinstance(frames_obj, list) or not frames_obj:
        return None

    crash_idx = coerce_crash_frame_index(snapshot)
    if 0 <= crash_idx < len(frames_obj) and isinstance(frames_obj[crash_idx], dict):
        return cast(FrameDict, frames_obj[crash_idx])
    if isinstance(frames_obj[0], dict):
        return cast(FrameDict, frames_obj[0])
    return None


def _get_locals(frame: FrameDict) -> dict[str, Any]:
    """Get locals from a frame, filtering pytest internals."""
    locals_dict = frame.get("locals", {})
    if not isinstance(locals_dict, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        return {}
    return {k: v for k, v in locals_dict.items() if not str(k).startswith("@py_")}


def _is_array_summary(value: Any) -> bool:
    """Check if a value is an array/dataframe/series summary."""
    if not isinstance(value, dict):
        return False
    d = cast(dict[str, Any], value)
    return any(d.get(k) for k in _ARRAY_KEYS)


def _get_shape(value: Any) -> list[int] | None:
    """Extract shape from an array summary."""
    if isinstance(value, dict):
        d = cast(dict[str, Any], value)
        shape = d.get("shape")
        if isinstance(shape, list) and all(isinstance(s, int) for s in cast(list[Any], shape)):
            return cast(list[int], shape)
    return None


def _truncate(s: str, max_len: int = 80) -> str:
    """Safe string truncation."""
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _exc_type(snapshot: dict[str, Any]) -> str:
    """Get the exception type name."""
    exc = snapshot.get("exception", {})
    if not isinstance(exc, dict):
        return ""
    return str(cast(ExceptionDict, exc).get("type", ""))


def _exc_message(snapshot: dict[str, Any]) -> str:
    """Get the exception message."""
    exc = snapshot.get("exception", {})
    if not isinstance(exc, dict):
        return ""
    return str(cast(ExceptionDict, exc).get("message", ""))


# ---------------------------------------------------------------------------
# 10 Pattern Detectors
# ---------------------------------------------------------------------------


@_register_detector
def _detect_empty_array(snapshot: dict[str, Any]) -> list[Hypothesis]:  # pyright: ignore[reportUnusedFunction]
    """Detect arrays with shape[0] == 0."""
    frame = _get_crash_frame(snapshot)
    if not frame:
        return []
    locals_dict = _get_locals(frame)
    results: list[Hypothesis] = []
    for name, value in locals_dict.items():
        shape = _get_shape(value)
        if shape and shape[0] == 0:
            results.append(
                Hypothesis(
                    confidence=0.85,
                    pattern="empty_array",
                    description=f"Variable '{name}' is an empty array (shape {shape})",
                    evidence=(f"{name}.shape = {shape}",),
                    suggestion=(
                        f"Check why '{name}' has no elements. "
                        "Trace upstream data loading or filtering."
                    ),
                )
            )
    return results


@_register_detector
def _detect_shape_mismatch(snapshot: dict[str, Any]) -> list[Hypothesis]:  # pyright: ignore[reportUnusedFunction]
    """Detect shape mismatches when broadcasting/shape errors occur."""
    exc_type = _exc_type(snapshot)
    exc_msg = _exc_message(snapshot)
    if exc_type not in ("ValueError", "RuntimeError"):
        return []
    shape_keywords = ("shape", "broadcast", "dimension", "mismatch", "size")
    if not any(kw in exc_msg.lower() for kw in shape_keywords):
        return []

    frame = _get_crash_frame(snapshot)
    if not frame:
        return []
    locals_dict = _get_locals(frame)
    shapes: list[tuple[str, list[int]]] = []
    for name, value in locals_dict.items():
        shape = _get_shape(value)
        if shape:
            shapes.append((name, shape))

    if len(shapes) < 2:
        return []

    evidence = tuple(f"{name}.shape = {shape}" for name, shape in shapes)
    return [
        Hypothesis(
            confidence=0.9,
            pattern="shape_mismatch",
            description=(
                f"Shape mismatch between arrays: {', '.join(f'{n} {s}' for n, s in shapes)}"
            ),
            evidence=evidence,
            suggestion="Check that array shapes are compatible for the operation.",
        )
    ]


@_register_detector
def _detect_none_dereference(snapshot: dict[str, Any]) -> list[Hypothesis]:  # pyright: ignore[reportUnusedFunction]
    """Detect None dereference (AttributeError/TypeError on None)."""
    exc_type = _exc_type(snapshot)
    exc_msg = _exc_message(snapshot)
    if exc_type not in ("AttributeError", "TypeError"):
        return []

    frame = _get_crash_frame(snapshot)
    if not frame:
        return []
    locals_dict = _get_locals(frame)

    none_vars = [name for name, value in locals_dict.items() if value is None]
    if not none_vars:
        return []

    # Higher confidence if "NoneType" appears in the message
    has_nonetype = "NoneType" in exc_msg or "'None'" in exc_msg
    confidence = 0.9 if has_nonetype else 0.75

    evidence = tuple(f"{name} = None" for name in none_vars)
    return [
        Hypothesis(
            confidence=confidence,
            pattern="none_dereference",
            description=f"Variable(s) {', '.join(none_vars)} are None — likely unexpected",
            evidence=evidence,
            suggestion=(
                f"Check initialization of {none_vars[0]}. It may not be set on all code paths."
            ),
        )
    ]


@_register_detector
def _detect_off_by_one(snapshot: dict[str, Any]) -> list[Hypothesis]:  # pyright: ignore[reportUnusedFunction]
    """Detect index == length (off-by-one) with IndexError."""
    if _exc_type(snapshot) != "IndexError":
        return []

    frame = _get_crash_frame(snapshot)
    if not frame:
        return []
    locals_dict = _get_locals(frame)

    results: list[Hypothesis] = []
    # Single pass: separate integer vars (potential indices) and length-bearing vars
    int_vars: dict[str, int] = {}
    length_vars: dict[str, int] = {}  # name -> length
    for name, value in locals_dict.items():
        if isinstance(value, int):
            int_vars[name] = value
        if isinstance(value, (list, str)):
            length_vars[name] = len(cast(list[Any], value))
        elif _is_array_summary(value):
            shape = _get_shape(value)
            if shape:
                length_vars[name] = shape[0]

    # Check int vars against length-bearing vars only (O(int_vars * length_vars))
    for idx_name, idx_val in int_vars.items():
        for col_name, length in length_vars.items():
            if col_name == idx_name:
                continue
            if idx_val == length:
                results.append(
                    Hypothesis(
                        confidence=0.9,
                        pattern="off_by_one",
                        description=(
                            f"Index '{idx_name}' ({idx_val}) equals length of "
                            f"'{col_name}' ({length}) — off-by-one error"
                        ),
                        evidence=(
                            f"{idx_name} = {idx_val}",
                            f"len({col_name}) = {length}",
                        ),
                        suggestion=f"Use '{idx_name} < len({col_name})' or '{idx_name} - 1'.",
                    )
                )
    return results


@_register_detector
def _detect_nan_inf(snapshot: dict[str, Any]) -> list[Hypothesis]:  # pyright: ignore[reportUnusedFunction]
    """Detect NaN/Inf anomalies in arrays."""
    frame = _get_crash_frame(snapshot)
    if not frame:
        return []
    locals_dict = _get_locals(frame)

    results: list[Hypothesis] = []
    for name, value in locals_dict.items():
        if not isinstance(value, dict):
            continue
        vd = cast(dict[str, Any], value)
        anomalies_raw = vd.get("anomalies", {})
        if not isinstance(anomalies_raw, dict):
            continue
        anomalies = cast(dict[str, Any], anomalies_raw)
        nan_count: Any = anomalies.get("nan", 0)
        inf_count: Any = anomalies.get("inf", 0)
        if nan_count or inf_count:
            parts: list[str] = []
            if nan_count:
                parts.append(f"{nan_count} NaN")
            if inf_count:
                parts.append(f"{inf_count} Inf")
            results.append(
                Hypothesis(
                    confidence=0.85,
                    pattern="nan_inf_anomaly",
                    description=f"Array '{name}' contains {', '.join(parts)} values",
                    evidence=(f"{name}.anomalies = {anomalies}",),
                    suggestion=(
                        f"Check computations upstream of '{name}' for division by zero or overflow."
                    ),
                )
            )
    return results


@_register_detector
def _detect_type_mismatch(snapshot: dict[str, Any]) -> list[Hypothesis]:  # pyright: ignore[reportUnusedFunction]
    """Detect TypeError with type info in message."""
    if _exc_type(snapshot) != "TypeError":
        return []
    exc_msg = _exc_message(snapshot)

    frame = _get_crash_frame(snapshot)
    if not frame:
        return []
    locals_meta: dict[str, Any] = cast(dict[str, Any], frame.get("locals_meta", {}))
    locals_dict = _get_locals(frame)

    # Try to find a type mentioned in the error message
    results: list[Hypothesis] = []
    for name in locals_dict:
        meta: dict[str, Any] = cast(dict[str, Any], locals_meta.get(name, {}))
        var_type = meta.get("type", "")
        if var_type and var_type in exc_msg:
            results.append(
                Hypothesis(
                    confidence=0.8,
                    pattern="type_mismatch",
                    description=f"Variable '{name}' has type '{var_type}' mentioned in TypeError",
                    evidence=(f"{name}: {var_type}", f"message: {_truncate(exc_msg)}"),
                    suggestion=f"Check the expected type for '{name}' at this call site.",
                )
            )
    return results


@_register_detector
def _detect_key_not_found(snapshot: dict[str, Any]) -> list[Hypothesis]:  # pyright: ignore[reportUnusedFunction]
    """Detect KeyError with dict locals showing available keys."""
    if _exc_type(snapshot) != "KeyError":
        return []
    exc_msg = _exc_message(snapshot)

    frame = _get_crash_frame(snapshot)
    if not frame:
        return []
    locals_dict = _get_locals(frame)

    results: list[Hypothesis] = []
    for name, value in locals_dict.items():
        if isinstance(value, dict):
            vd = cast(dict[str, Any], value)
            keys_str = ", ".join(str(k) for k in list(vd.keys())[:10])
            results.append(
                Hypothesis(
                    confidence=0.85,
                    pattern="key_not_found",
                    description=f"KeyError({exc_msg}) — dict '{name}' has keys: [{keys_str}]",
                    evidence=(
                        f"missing key: {exc_msg}",
                        f"{name}.keys() = [{keys_str}]",
                    ),
                    suggestion=f"Check spelling of the key or use '{name}.get(key, default)'.",
                )
            )
    return results


@_register_detector
def _detect_import_failure(snapshot: dict[str, Any]) -> list[Hypothesis]:  # pyright: ignore[reportUnusedFunction]
    """Detect ImportError / ModuleNotFoundError."""
    exc_type = _exc_type(snapshot)
    if exc_type not in ("ImportError", "ModuleNotFoundError"):
        return []
    exc_msg = _exc_message(snapshot)

    # Try to extract module name from message
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", exc_msg)
    module_name = match.group(1) if match else exc_msg

    return [
        Hypothesis(
            confidence=0.95,
            pattern="import_failure",
            description=f"Cannot import '{module_name}'",
            evidence=(f"exception: {exc_type}: {_truncate(exc_msg)}",),
            suggestion=f"Install the missing package: pip install {module_name.split('.')[0]}",
        )
    ]


@_register_detector
def _detect_file_not_found(snapshot: dict[str, Any]) -> list[Hypothesis]:  # pyright: ignore[reportUnusedFunction]
    """Detect FileNotFoundError with path in message."""
    if _exc_type(snapshot) != "FileNotFoundError":
        return []
    exc_msg = _exc_message(snapshot)

    # Try to extract the path
    match = re.search(r"['\"]([^'\"]+)['\"]", exc_msg)
    path = match.group(1) if match else "unknown"

    return [
        Hypothesis(
            confidence=0.95,
            pattern="file_not_found",
            description=f"File not found: '{path}'",
            evidence=(f"exception: FileNotFoundError: {_truncate(exc_msg)}",),
            suggestion=f"Check that '{path}' exists and the path is correct.",
        )
    ]


@_register_detector
def _detect_assertion_failure(snapshot: dict[str, Any]) -> list[Hypothesis]:  # pyright: ignore[reportUnusedFunction]
    """Detect AssertionError with comparison values in locals."""
    if _exc_type(snapshot) != "AssertionError":
        return []

    frame = _get_crash_frame(snapshot)
    if not frame:
        return []
    locals_dict = _get_locals(frame)

    if not locals_dict:
        return [
            Hypothesis(
                confidence=0.7,
                pattern="assertion_failure",
                description="Assertion failed — no locals available for comparison",
                evidence=("AssertionError raised",),
                suggestion="Add explicit assertion messages to see expected vs actual values.",
            )
        ]

    evidence = tuple(
        f"{name} = {_truncate(str(value))}" for name, value in list(locals_dict.items())[:5]
    )
    return [
        Hypothesis(
            confidence=0.8,
            pattern="assertion_failure",
            description="Assertion failed — check variable values at crash site",
            evidence=evidence,
            suggestion="Compare the actual values above against expected values.",
        )
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_hypotheses(snapshot: dict[str, Any]) -> list[Hypothesis]:
    """Run all detectors and return hypotheses sorted by confidence (descending).

    Detector exceptions are silently swallowed to ensure the engine never crashes.
    """
    results: list[Hypothesis] = []
    seen_patterns: set[str] = set()

    for detector in _detectors:
        try:
            hypotheses = detector(snapshot)
        except Exception:  # nosec B112 — fault isolation per detector
            continue
        for h in hypotheses:
            # Deduplicate by pattern (keep highest confidence)
            if h.pattern in seen_patterns:
                continue
            seen_patterns.add(h.pattern)
            results.append(h)

    results.sort(key=lambda h: h.confidence, reverse=True)
    return results


def format_hypotheses_text(hypotheses: list[Hypothesis]) -> str:
    """Format hypotheses as markdown text for CLI/MCP output."""
    if not hypotheses:
        return "No hypotheses generated — the snapshot may lack sufficient data."

    lines: list[str] = []
    for i, h in enumerate(hypotheses, 1):
        conf_pct = int(h.confidence * 100)
        lines.append(f"## Hypothesis #{i} [{conf_pct}%] — {h.pattern}")
        lines.append(f"{h.description}")
        lines.append("")
        lines.append("Evidence:")
        lines.extend(f"  - {e}" for e in h.evidence)
        lines.append("")
        lines.append(f"Suggestion: {h.suggestion}")
        lines.append("")

    return "\n".join(lines)
