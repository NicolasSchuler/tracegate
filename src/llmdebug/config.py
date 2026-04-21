"""Configuration dataclass for snapshot capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from re import Pattern
from typing import Literal

SourceMode = Literal["all", "crash_only", "none"]
LocalsMode = Literal["safe", "meta", "none"]
OutputFormat = Literal["json", "json_compact", "toon"]
DetailLevel = Literal["crash", "full", "context"]
RedactionProfile = Literal["dev", "ci", "prod"]
GitDirtyMode = Literal["always", "cached", "off"]


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuration for debug snapshot capture.

    The same object can be reused across the public capture entry points
    (`debug_snapshot`, `snapshot_section`, hooks, middleware). A few options
    are especially important when configuring behavior:

    - `source_mode` / `locals_mode` control how much code and local state is
      captured.
    - `redaction_profile` and the `redact_*` overrides control privacy.
    - `include_args` separates function arguments from other locals in frame
      summaries.
    - `include_coverage` only has an effect when pytest-plugin capture runs
      alongside pytest-cov.
    - `output_format` selects between readability (`json`) and smaller
      context-efficient encodings (`json_compact`, `toon`).
    """

    out_dir: str = ".llmdebug"
    frames: int = 5
    source_context: int = 3
    source_mode: SourceMode = "all"
    locals_mode: LocalsMode = "safe"
    max_str: int = 500
    max_items: int = 50
    redaction_profile: RedactionProfile | None = None
    redact: tuple[str | Pattern[str], ...] = field(default_factory=tuple)
    redact_keys: bool = False  # Redact dict keys as well as values (can collapse keys)
    redact_traceback: bool = False  # Redact traceback string payloads
    redact_exception_strings: bool = False  # Redact exception message/args/notes strings
    include_env: bool = True
    debug: bool = False
    max_snapshots: int = 50  # 0 = unlimited
    include_modules: tuple[str, ...] | None = None  # Filter frames by module prefix
    max_exception_depth: int = 5  # Exception chain recursion limit
    lock_timeout: float = 5.0  # Seconds to wait for file lock
    # New feature flags
    include_git: bool = True  # Capture git commit/branch/dirty status
    git_dirty_mode: GitDirtyMode = "always"  # Dirty-check strategy when include_git=True
    include_array_stats: bool = False  # Compute min/max/mean/std for arrays
    include_args: bool = True  # Separate function arguments from locals
    categorize_errors: bool = True  # Auto-classify errors with suggestions
    include_async_context: bool = True  # Capture asyncio task info
    capture_logs: bool = False  # Capture recent log records
    log_max_records: int = 20  # Max log records to capture
    output_format: OutputFormat = "json_compact"
    include_coverage: bool = True  # Include coverage data from pytest-cov

    def __post_init__(self) -> None:
        if self.output_format not in {"json", "json_compact", "toon"}:
            raise ValueError("output_format must be 'json', 'json_compact', or 'toon'")
        if self.locals_mode not in {"safe", "meta", "none"}:
            raise ValueError("locals_mode must be 'safe', 'meta', or 'none'")
        if self.source_mode not in {"all", "crash_only", "none"}:
            raise ValueError("source_mode must be 'all', 'crash_only', or 'none'")
        if self.frames < 1:
            raise ValueError("frames must be >= 1")
        if self.max_str < 13:
            raise ValueError("max_str must be >= 13 (12-char truncation marker + 1)")
        if self.max_items < 1:
            raise ValueError("max_items must be >= 1")
        if self.max_snapshots < 0:
            raise ValueError("max_snapshots must be >= 0 (0 = unlimited)")
        if self.max_exception_depth < 1:
            raise ValueError("max_exception_depth must be >= 1")
        if self.lock_timeout < 0:
            raise ValueError("lock_timeout must be >= 0")
        if self.log_max_records < 0:
            raise ValueError("log_max_records must be >= 0")
        if self.redaction_profile not in {None, "dev", "ci", "prod"}:
            raise ValueError("redaction_profile must be one of: dev, ci, prod, or None")
        if self.git_dirty_mode not in {"always", "cached", "off"}:
            raise ValueError("git_dirty_mode must be 'always', 'cached', or 'off'")
