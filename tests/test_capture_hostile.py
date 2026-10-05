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


# ---------------------------------------------------------------------------
# CAP-02: exception text whose __str__ raises
# ---------------------------------------------------------------------------


class _Unprintable:
    def __str__(self) -> str:
        raise RuntimeError("no str")

    def __repr__(self) -> str:
        raise RuntimeError("no repr")


class _UnprintableError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("no str")


def test_exception_whose_str_raises(tmp_path: Path) -> None:
    def boom() -> None:
        raise _UnprintableError("hidden")

    snapshot = _capture(tmp_path, boom)

    exc = snapshot["exception"]
    assert exc["type"] == "_UnprintableError"
    assert exc["message"] == "<str() failed: RuntimeError>"
    assert exc["args"] == ["hidden"]


def test_exception_arg_whose_str_raises(tmp_path: Path) -> None:
    def boom() -> None:
        raise ValueError(_Unprintable(), "readable")

    snapshot = _capture(tmp_path, boom)

    exc = snapshot["exception"]
    assert exc["message"] == "<str() failed: RuntimeError>"
    assert exc["args"] == ["<str() failed: RuntimeError>", "readable"]


def test_exception_note_whose_str_raises(tmp_path: Path) -> None:
    def boom() -> None:
        err = ValueError("noted")
        err.__notes__ = ["readable", _Unprintable()]
        raise err

    snapshot = _capture(tmp_path, boom)

    assert snapshot["exception"]["notes"] == ["readable", "<str() failed: RuntimeError>"]


def test_chained_exception_whose_str_raises(tmp_path: Path) -> None:
    def boom() -> None:
        try:
            raise _UnprintableError()
        except _UnprintableError as inner:
            raise ValueError("outer") from inner

    snapshot = _capture(tmp_path, boom)

    exc = snapshot["exception"]
    assert exc["message"] == "outer"
    assert exc["cause"]["message"] == "<str() failed: RuntimeError>"


def _raise_no_str(_self: object) -> str:
    raise RuntimeError("no str")


def test_unprintable_exception_is_still_categorized(tmp_path: Path) -> None:
    # Categories match on the exact type name, so keep the name "KeyError".
    unprintable_key_error = type("KeyError", (KeyError,), {"__str__": _raise_no_str})

    def boom() -> None:
        raise unprintable_key_error("k")

    snapshot = _capture(tmp_path, boom)

    assert snapshot["exception"]["error_category"]["category"]
    assert not any("categorization" in w for w in snapshot.get("_capture_warnings", []))


# ---------------------------------------------------------------------------
# Hostile locals: one bad value must not abort the capture or hide its siblings
# ---------------------------------------------------------------------------


class _NoRepr:
    def __repr__(self) -> str:
        raise RuntimeError("no repr")


class _NoLen(list[int]):
    def __len__(self) -> int:
        raise RuntimeError("no len")


class _NoAttributes:
    """Every attribute lookup but ``__class__`` raises a non-AttributeError."""

    def __getattribute__(self, name: str) -> Any:
        if name == "__class__":
            return object.__getattribute__(self, name)
        raise RuntimeError(f"no {name}")


class _NoClass:
    """Even ``__class__`` raises, so ``isinstance`` against other types raises."""

    def __getattribute__(self, name: str) -> Any:
        raise RuntimeError(f"no {name}")


class _NoGetattr:
    def __init__(self) -> None:
        self.visible = 1

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"no {name}")


class _Lazy:
    """Stands in for a lazy query object whose ``__len__`` does work."""

    def __init__(self) -> None:
        self.len_calls = 0

    def __len__(self) -> int:
        self.len_calls += 1
        return 3


_HOSTILE_FACTORIES: dict[str, Callable[[], Any]] = {
    "no_repr": _NoRepr,
    "no_len": lambda: _NoLen([1, 2]),
    "no_attributes": _NoAttributes,
    "no_class": _NoClass,
    "no_getattr": _NoGetattr,
    "lazy": _Lazy,
}


@pytest.mark.parametrize("locals_mode", ["safe", "meta"])
@pytest.mark.parametrize("kind", sorted(_HOSTILE_FACTORIES))
def test_hostile_local_keeps_its_siblings(tmp_path: Path, kind: str, locals_mode: str) -> None:
    def boom() -> None:
        before = "kept"  # noqa: F841 - captured local
        hostile = _HOSTILE_FACTORIES[kind]()  # noqa: F841 - captured local
        after = [1, 2]  # noqa: F841 - captured local
        raise ValueError(kind)

    snapshot = _capture(tmp_path, boom, locals_mode=locals_mode)

    frame = snapshot["frames"][0]
    assert {"before", "hostile", "after"} <= set(frame["locals_meta"])
    if locals_mode == "safe":
        local_vars = frame["locals"]
        assert local_vars["before"] == "kept"
        assert local_vars["after"] == [1, 2]
        assert isinstance(local_vars["hostile"], (str, dict, list))


@pytest.mark.parametrize("kind", ["no_attributes", "no_class"])
def test_raising_getattribute_is_marked(tmp_path: Path, kind: str) -> None:
    def boom() -> None:
        hostile = _HOSTILE_FACTORIES[kind]()  # noqa: F841 - captured local
        raise ValueError(kind)

    snapshot = _capture(tmp_path, boom)

    assert _crash_locals(snapshot)["hostile"] == "<serialization failed: RuntimeError>"


def test_hostile_values_nested_in_containers(tmp_path: Path) -> None:
    def boom() -> None:
        mixed = [1, _NoClass(), _NoRepr(), "ok"]  # noqa: F841 - captured local
        mapping = {"bad": _NoAttributes(), "good": 2}  # noqa: F841 - captured local
        raise ValueError("nested hostile")

    snapshot = _capture(tmp_path, boom)

    local_vars = _crash_locals(snapshot)
    assert local_vars["mixed"][0] == 1
    assert local_vars["mixed"][1] == "<serialization failed: RuntimeError>"
    assert local_vars["mixed"][2] == "<unreprable _NoRepr>"
    assert local_vars["mixed"][3] == "ok"
    good, bad = sorted(local_vars["mapping"].values(), key=lambda v: isinstance(v, str))
    assert good == 2
    assert "_NoAttributes object" in bad


def test_recursive_containers(tmp_path: Path) -> None:
    def boom() -> None:
        loop: list[Any] = [1]
        loop.append(loop)
        cycle: dict[str, Any] = {"n": 1}
        cycle["self"] = cycle
        raise ValueError("recursive")

    snapshot = _capture(tmp_path, boom)

    local_vars = _crash_locals(snapshot)
    assert local_vars["loop"][0] == 1
    assert "[...]" in str(local_vars["loop"][1])
    assert "{...}" in str(local_vars["cycle"])


def test_very_long_local_names(tmp_path: Path) -> None:
    long_name = "v" * 300
    namespace: dict[str, Any] = {}
    exec(  # builds a frame with a 300-character local name
        f"def boom():\n    {long_name} = 1\n    raise ValueError('long name')\n",
        namespace,
    )

    snapshot = _capture(tmp_path, namespace["boom"])

    assert _crash_locals(snapshot)[long_name] == 1


@pytest.mark.parametrize("locals_mode", ["safe", "meta"])
def test_capture_does_not_call_len_on_arbitrary_objects(tmp_path: Path, locals_mode: str) -> None:
    """CAP-21: len() on a lazy ORM query set executes the query."""
    lazy = _Lazy()

    def boom() -> None:
        query = lazy  # noqa: F841 - captured local
        rows = [1, 2, 3]  # noqa: F841 - captured local
        raise ValueError("lazy")

    snapshot = _capture(tmp_path, boom, locals_mode=locals_mode)

    meta = snapshot["frames"][0]["locals_meta"]
    assert lazy.len_calls == 0
    assert "len" not in meta["query"]
    assert meta["rows"]["len"] == 3
