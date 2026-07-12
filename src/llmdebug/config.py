"""Configuration dataclass for snapshot capture."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from re import Pattern
from typing import Any, Literal, cast

SourceMode = Literal["all", "crash_only", "none"]
LocalsMode = Literal["safe", "meta", "none"]
OutputFormat = Literal["json", "json_compact", "toon"]
DetailLevel = Literal["crash", "full", "context"]
RedactionProfile = Literal["dev", "ci", "prod"]
GitDirtyMode = Literal["always", "cached", "off"]


class _InheritRedactionDefault:
    """Constructor-only marker replaced with concrete values in ``__post_init__``."""

    __llmdebug_inherit_redaction__ = True

    def __repr__(self) -> str:
        return "<inherit>"


_INHERIT_REDACTION = _InheritRedactionDefault()
_RedactionInheritance = tuple[bool, bool, bool, bool]
_ResolvedRedactionValues = tuple[tuple[str | Pattern[str], ...], bool, bool, bool]


def _inherits_redaction_default(value: object) -> bool:
    """Recognize constructor-only inheritance markers before normalization."""
    return value is None or getattr(value, "__llmdebug_inherit_redaction__", False) is True


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuration for debug snapshot capture.

    The same object can be reused across the public capture entry points
    (`debug_snapshot`, `snapshot_section`, hooks, middleware). A few options
    are especially important when configuring behavior:

    - `source_mode` / `locals_mode` control how much code and local state is
      captured.
    - `redaction_profile` and the `redact_*` overrides control privacy. Omitted
      overrides inherit from the selected profile (or the hard default when no
      profile is selected). Public attributes are always concrete after
      construction.
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
    redact: tuple[str | Pattern[str], ...] = cast(
        tuple[str | Pattern[str], ...], _INHERIT_REDACTION
    )
    redact_keys: bool = cast(bool, _INHERIT_REDACTION)
    redact_traceback: bool = cast(bool, _INHERIT_REDACTION)
    redact_exception_strings: bool = cast(bool, _INHERIT_REDACTION)
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
    _redaction_inheritance: _RedactionInheritance | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _resolved_redaction_values: _ResolvedRedactionValues | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _redaction_profile_at_resolution: RedactionProfile | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        _validate_snapshot_config(self)
        inheritance = _resolve_redaction_inheritance(
            config=self,
            prior_inheritance=self._redaction_inheritance,
            prior_values=self._resolved_redaction_values,
            prior_profile=self._redaction_profile_at_resolution,
        )
        resolved_values = _resolve_redaction_values(self, inheritance)
        _apply_resolved_redaction(self, inheritance, resolved_values)

    @property
    def has_explicit_redaction_settings(self) -> bool:
        """Whether a profile or any redaction override was supplied by the caller."""
        return self.redaction_profile is not None or not all(self._redaction_inheritance or ())

    def with_updates(self, **changes: object) -> SnapshotConfig:
        """Return an updated config while preserving exact redaction inheritance.

        ``dataclasses.replace`` cannot tell an omitted inherited override from
        an explicitly repeated resolved value when the profile also changes.
        This method retains the unresolved intent and applies only the fields
        present in ``changes``.
        """
        public_fields = {
            config_field.name
            for config_field in fields(self)
            if config_field.init and not config_field.name.startswith("_")
        }
        unknown = sorted(set(changes) - public_fields)
        if unknown:
            rendered = ", ".join(repr(name) for name in unknown)
            raise TypeError(f"SnapshotConfig.with_updates() got unknown fields: {rendered}")

        inheritance_by_field: dict[str, bool] = dict(
            zip(
                (
                    "redact",
                    "redact_keys",
                    "redact_traceback",
                    "redact_exception_strings",
                ),
                self._redaction_inheritance or (False, False, False, False),
                strict=True,
            )
        )
        values: dict[str, object] = {}
        for config_field in fields(self):
            name = config_field.name
            if not config_field.init or name.startswith("_"):
                continue
            if name in changes:
                values[name] = changes[name]
            elif inheritance_by_field.get(name, False):
                values[name] = _INHERIT_REDACTION
            else:
                values[name] = getattr(self, name)
        return type(self)(**cast(Any, values))


def _validate_snapshot_config(config: SnapshotConfig) -> None:
    """Validate independent capture settings before resolving redaction policy."""
    if config.output_format not in {"json", "json_compact", "toon"}:
        raise ValueError("output_format must be 'json', 'json_compact', or 'toon'")
    if config.locals_mode not in {"safe", "meta", "none"}:
        raise ValueError("locals_mode must be 'safe', 'meta', or 'none'")
    if config.source_mode not in {"all", "crash_only", "none"}:
        raise ValueError("source_mode must be 'all', 'crash_only', or 'none'")
    if config.frames < 1:
        raise ValueError("frames must be >= 1")
    if config.max_str < 13:
        raise ValueError("max_str must be >= 13 (12-char truncation marker + 1)")
    if config.max_items < 1:
        raise ValueError("max_items must be >= 1")
    if config.max_snapshots < 0:
        raise ValueError("max_snapshots must be >= 0 (0 = unlimited)")
    if config.max_exception_depth < 1:
        raise ValueError("max_exception_depth must be >= 1")
    if config.lock_timeout < 0:
        raise ValueError("lock_timeout must be >= 0")
    if config.log_max_records < 0:
        raise ValueError("log_max_records must be >= 0")
    if config.redaction_profile not in {None, "dev", "ci", "prod"}:
        raise ValueError("redaction_profile must be one of: dev, ci, prod, or None")
    if config.git_dirty_mode not in {"always", "cached", "off"}:
        raise ValueError("git_dirty_mode must be 'always', 'cached', or 'off'")


def _resolve_redaction_inheritance(
    *,
    config: SnapshotConfig,
    prior_inheritance: _RedactionInheritance | None,
    prior_values: _ResolvedRedactionValues | None,
    prior_profile: RedactionProfile | None,
) -> _RedactionInheritance:
    """Recover which redaction inputs remain inherited across safe updates."""
    if (
        prior_inheritance is not None
        and config.redaction_profile != prior_profile
        and any(prior_inheritance)
    ):
        raise ValueError(
            "dataclasses.replace() cannot safely change redaction_profile when settings "
            "are inherited; use SnapshotConfig.with_updates(...)"
        )

    incoming_values = (
        config.redact,
        config.redact_keys,
        config.redact_traceback,
        config.redact_exception_strings,
    )
    if prior_inheritance is None:
        return cast(
            _RedactionInheritance,
            tuple(_inherits_redaction_default(value) for value in incoming_values),
        )
    if prior_values is None:
        raise ValueError("_resolved_redaction_values is required with _redaction_inheritance")
    return cast(
        _RedactionInheritance,
        tuple(
            was_inherited and incoming == previous
            for was_inherited, incoming, previous in zip(
                prior_inheritance,
                incoming_values,
                prior_values,
                strict=True,
            )
        ),
    )


def _resolve_redaction_values(
    config: SnapshotConfig,
    inheritance: _RedactionInheritance,
) -> _ResolvedRedactionValues:
    """Resolve concrete redaction values from profile inheritance and overrides."""
    from .redaction_policy import resolve_redaction_policy

    policy = resolve_redaction_policy(
        redaction_profile=config.redaction_profile,
        redact=None if inheritance[0] else tuple(config.redact),
        redact_keys=None if inheritance[1] else config.redact_keys,
        redact_traceback=None if inheritance[2] else config.redact_traceback,
        redact_exception_strings=(None if inheritance[3] else config.redact_exception_strings),
    )
    return (
        policy.patterns,
        policy.redact_keys,
        policy.redact_traceback,
        policy.redact_exception_strings,
    )


def _apply_resolved_redaction(
    config: SnapshotConfig,
    inheritance: _RedactionInheritance,
    values: _ResolvedRedactionValues,
) -> None:
    """Store concrete public values and private inheritance provenance."""
    patterns, redact_keys, redact_traceback, redact_exception_strings = values
    object.__setattr__(config, "redact", patterns)
    object.__setattr__(config, "redact_keys", redact_keys)
    object.__setattr__(config, "redact_traceback", redact_traceback)
    object.__setattr__(config, "redact_exception_strings", redact_exception_strings)
    object.__setattr__(config, "_redaction_inheritance", inheritance)
    object.__setattr__(config, "_resolved_redaction_values", values)
    object.__setattr__(config, "_redaction_profile_at_resolution", config.redaction_profile)


def resolve_snapshot_config(config: SnapshotConfig) -> SnapshotConfig:
    """Return a capture-ready config.

    Redaction inheritance is resolved during ``SnapshotConfig`` construction,
    so this boundary helper is intentionally an idempotent compatibility seam.
    """
    return config
