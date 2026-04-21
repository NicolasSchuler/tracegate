"""Additional tests for SnapshotConfig validation branches."""

from __future__ import annotations

import pytest

from llmdebug.config import SnapshotConfig


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"output_format": "invalid"}, "output_format must be"),
        ({"locals_mode": "invalid"}, "locals_mode must be"),
        ({"source_mode": "invalid"}, "source_mode must be"),
        ({"frames": 0}, "frames must be >= 1"),
        ({"max_str": 12}, "max_str must be >= 13"),
        ({"max_items": 0}, "max_items must be >= 1"),
        ({"max_snapshots": -1}, "max_snapshots must be >= 0"),
        ({"max_exception_depth": 0}, "max_exception_depth must be >= 1"),
        ({"lock_timeout": -0.1}, "lock_timeout must be >= 0"),
        ({"log_max_records": -1}, "log_max_records must be >= 0"),
        ({"redaction_profile": "staging"}, "redaction_profile must be one of"),
        ({"git_dirty_mode": "sometimes"}, "git_dirty_mode must be"),
    ],
)
def test_snapshot_config_rejects_invalid_values(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        SnapshotConfig(**overrides)  # type: ignore[arg-type]


def test_snapshot_config_accepts_boundary_values() -> None:
    cfg = SnapshotConfig(
        frames=1,
        max_str=13,
        max_items=1,
        max_snapshots=0,
        max_exception_depth=1,
        lock_timeout=0.0,
        log_max_records=0,
        redaction_profile="dev",
        git_dirty_mode="off",
        output_format="toon",
    )
    assert cfg.frames == 1
    assert cfg.max_str == 13
    assert cfg.max_items == 1
    assert cfg.max_snapshots == 0
    assert cfg.max_exception_depth == 1
    assert cfg.lock_timeout == 0.0
    assert cfg.log_max_records == 0
    assert cfg.redaction_profile == "dev"
    assert cfg.git_dirty_mode == "off"
    assert cfg.output_format == "toon"
