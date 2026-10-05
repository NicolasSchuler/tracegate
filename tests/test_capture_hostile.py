"""Hostile-object suite: capture must survive whatever user code leaves behind.

Phase 4.1 of ``docs/2026-10-04-code-review-remediation-handoff.md``. Every case
goes end-to-end through ``capture_exception`` and reads the written snapshot, so
failures in serialization *and* in the JSON writer are both caught.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from llmdebug import get_latest_snapshot
from llmdebug.capture import capture_exception
from llmdebug.config import SnapshotConfig


def _capture(
    tmp_path: Path, raiser: Callable[[], None], *, name: str = "hostile", **cfg_kwargs: Any
) -> dict[str, Any]:
    try:
        raiser()
    except Exception as exc:
        cfg = SnapshotConfig(out_dir=str(tmp_path), include_git=False, **cfg_kwargs)
        path = capture_exception(name, exc, exc.__traceback__, cfg)
        assert path.exists()
    else:  # pragma: no cover - every raiser raises
        raise AssertionError("raiser did not raise")
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    return snapshot


def _crash_locals(snapshot: dict[str, Any]) -> dict[str, Any]:
    return snapshot["frames"][0]["locals"]


# ---------------------------------------------------------------------------
# CAP-01: ints outside the 64-bit range orjson can encode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [2**64, -(2**63) - 1, 2**70, -(2**200)])
def test_wide_int_local_is_kept_as_its_repr(tmp_path: Path, value: int) -> None:
    def boom() -> None:
        wide = value  # noqa: F841 - captured local
        raise ValueError("wide int")

    snapshot = _capture(tmp_path, boom)

    assert _crash_locals(snapshot)["wide"] == repr(value)


@pytest.mark.parametrize("value", [2**64 - 1, -(2**63), 0, True])
def test_ints_at_the_64_bit_bounds_stay_numbers(tmp_path: Path, value: int) -> None:
    def boom() -> None:
        edge = value  # noqa: F841 - captured local
        raise ValueError("edge int")

    snapshot = _capture(tmp_path, boom)

    assert _crash_locals(snapshot)["edge"] == value


def test_wide_ints_nested_in_containers(tmp_path: Path) -> None:
    def boom() -> None:
        nested = [1, 2**65, {"k": 2**80}]  # noqa: F841 - captured local
        raise ValueError("nested wide ints")

    snapshot = _capture(tmp_path, boom)

    nested = _crash_locals(snapshot)["nested"]
    assert nested[0] == 1
    assert nested[1] == repr(2**65)
    assert repr(2**80) in str(nested[2])


def test_wide_ints_in_pandas_previews(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    def boom() -> None:
        frame = pd.DataFrame({"a": [2**70, 1]})  # noqa: F841 - captured local
        series = pd.Series([2**70, 2])  # noqa: F841 - captured local
        raise ValueError("pandas wide ints")

    snapshot = _capture(tmp_path, boom)

    local_vars = _crash_locals(snapshot)
    assert local_vars["frame"]["head"][0] == [repr(2**70)]
    assert local_vars["series"]["head"] == [repr(2**70), 2]
