"""Shared TypedDict contracts for snapshot payloads.

These types intentionally allow partial/malformed payloads so callers can
perform robust validation and fallback logic without assuming strict shape.
"""

from __future__ import annotations

from typing import Any, TypedDict


class ErrorCategoryDict(TypedDict, total=False):
    """Heuristic classification of a captured exception.

    Attributes:
        category: Error category identifier, such as ``shape_mismatch``.
        suggestion: Short remediation hint associated with the category.
    """

    category: str
    suggestion: str


class ExceptionDict(TypedDict, total=False):
    """Structured summary of the exception that triggered a capture.

    Attributes:
        type: Unqualified exception class name, such as ``ValueError``.
        qualified_type: Module-qualified exception class name.
        message: Exception message text, truncated and optionally redacted.
        error_category: Heuristic error classification with a remediation hint.
    """

    type: str
    qualified_type: str
    message: str
    error_category: ErrorCategoryDict


class SourceLineDict(TypedDict, total=False):
    """One line of a captured source snippet.

    Attributes:
        lineno: One-based line number in the source file.
        code: Source text of the line.
    """

    lineno: int
    code: str


class SourceDict(TypedDict, total=False):
    """Source snippet surrounding a frame's current line.

    Attributes:
        start: First line number included in the snippet.
        end: Last line number included in the snippet.
        snippet: Ordered source lines between ``start`` and ``end``.
    """

    start: int
    end: int
    snippet: list[SourceLineDict]


class FrameDict(TypedDict, total=False):
    """One stack frame captured from a traceback.

    Attributes:
        file: Absolute path to the source file.
        file_rel: Path relative to the working directory, or ``None`` outside it.
        line: Line number (int) or a sentinel string for unavailable frames.
        function: Qualified function name at this frame.
        module: Python module name containing the function, when known.
        code: Source line text at the frame's current line, when available.
        source: Surrounding source snippet with start/end line numbers.
        locals: Local variable values serialised to a JSON-compatible form.
        locals_meta: Per-variable metadata (type, shape, array stats, etc.).
        locals_truncated: ``True`` when locals beyond ``max_items`` were dropped.
        locals_truncated_keys: How many locals were dropped.
        args: Function argument values serialised to a JSON-compatible form.
        args_meta: Per-argument metadata mirroring ``locals_meta``.
        arg_names: Ordered list of positional argument names.
        is_coroutine: ``True`` when the frame belongs to an async function.
    """

    file: str
    file_rel: str | None
    line: int | str
    function: str
    module: str | None
    code: str | None
    source: SourceDict
    locals: dict[str, Any]
    locals_meta: dict[str, dict[str, Any]]
    locals_truncated: bool
    locals_truncated_keys: int
    args: dict[str, Any]
    args_meta: dict[str, dict[str, Any]]
    arg_names: list[str]
    is_coroutine: bool


class SessionDict(TypedDict, total=False):
    """Session metadata stored in the DebugSession envelope.

    Attributes:
        name: Human-readable capture label.
        timestamp_utc: ISO-8601 UTC timestamp of capture.
        llmdebug_version: Library version that produced the snapshot.
    """

    name: str
    timestamp_utc: str
    llmdebug_version: str


class SnapshotDict(TypedDict, total=False):
    """Top-level snapshot payload in either the legacy flat or the DebugSession envelope shape.

    Attributes:
        schema_version: Wire-format version string (legacy flat shape).
        name: Human-readable capture label.
        timestamp_utc: ISO-8601 UTC timestamp of capture.
        llmdebug_version: Library version that produced the snapshot.
        exception: Structured exception summary.
        traceback: Raw formatted traceback string.
        frames: Ordered list of captured stack frames.
        crash_frame_index: Index into ``frames`` identifying the crash frame.
        capture_config: Serialised ``SnapshotConfig`` used during capture.
        git: Git repository metadata at capture time.
        pytest: pytest session metadata when captured inside a test.
        rca: Root-cause analysis result attached after initial capture.
        hypothesis: Root-cause hypothesis text or related RCA hypothesis metadata.
        verification_result: Outcome of an automated verification pass.
        trace: Raw execution trace data (format is version-dependent).
        eval: Evaluation harness annotations.
        kind: Envelope discriminator for the DebugSession shape.
        session: Session-level metadata (DebugSession envelope shape).
        snapshot: Inner snapshot dict (DebugSession envelope shape).
        context: Extra ambient context attached to the session.
    """

    # Legacy flat fields
    schema_version: str
    name: str
    timestamp_utc: str
    llmdebug_version: str
    exception: ExceptionDict
    traceback: str
    frames: list[FrameDict | Any]
    crash_frame_index: int | str
    capture_config: dict[str, Any]
    git: dict[str, Any]
    pytest: dict[str, Any]
    rca: dict[str, Any]
    hypothesis: str
    verification_result: dict[str, Any]
    trace: Any
    eval: Any
    # DebugSession envelope fields
    kind: str
    session: SessionDict
    snapshot: dict[str, Any]
    context: dict[str, Any]


__all__ = [
    "ErrorCategoryDict",
    "ExceptionDict",
    "FrameDict",
    "SessionDict",
    "SnapshotDict",
    "SourceDict",
    "SourceLineDict",
]
