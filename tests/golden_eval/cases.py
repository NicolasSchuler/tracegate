"""Synthetic eval cases and the scripted model behaviour for the golden runs.

The cases live here as data, not as files under ``tests/``, because their
``test_*.py`` files fail by design and must not be collected by the suite.
Each case scripts what the stubbed model answers to the n-th patch call of a
unit, so one run exercises retries, syntax cycles, failed applies and
unfixable cases deterministically.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llmdebug import _json as json


@dataclass(frozen=True)
class Edit:
    path: str
    find: str
    replace: str


@dataclass(frozen=True)
class PatchReply:
    """What the model returns for one patch call.

    ``edit`` is rendered as an edits payload or a unified diff, depending on
    the requested response mode. ``raw`` overrides both with literal text.
    ``stale_diff`` renders the diff against a context that is not in the file,
    so it can't apply.
    """

    edit: Edit | None = None
    raw: str | None = None
    stale_diff: bool = False


PatchScript = Callable[[int, dict[str, str]], PatchReply]


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    metadata: dict[str, Any]
    files: dict[str, str]
    patch_script: PatchScript
    stage_a_requests: list[dict[str, Any]] = field(default_factory=list)


def _metadata(category: str, exception: str, dependency: str) -> dict[str, Any]:
    return {
        "description": f"golden fixture case ({category})",
        "category": category,
        "difficulty": "easy",
        "snapshot_dependency": dependency,
        "expected_exception": exception,
        "tags": ["golden"],
        "python_args": ["-m", "pytest", "-q", "-p", "no:cacheprovider"],
        "timeout_sec": 30,
    }


# --- golden_sum_skip_last: wrong patch first, right patch second -------------------------

_SUM_SOURCE = (
    "def total(values):\n"
    "    acc = 0\n"
    "    for index in range(len(values) - 1):\n"
    "        acc += values[index]\n"
    "    return acc\n"
)


def _sum_script(_n: int, files: dict[str, str]) -> PatchReply:
    source = files.get("buggy.py", "")
    if "range(len(values) - 1)" in source:
        return PatchReply(Edit("buggy.py", "range(len(values) - 1)", "range(1, len(values))"))
    return PatchReply(Edit("buggy.py", "range(1, len(values))", "range(len(values))"))


# --- golden_alias_keyerror: snapshot-dependent, fixed on the first patch ----------------

_ALIAS_SOURCE = (
    'DEFAULTS = {"timeout": 30, "retries": 3}\n'
    'ALIASES = {"retry": "retries", "wait": "timeout"}\n'
    "\n"
    "\n"
    "def resolve(config, key):\n"
    "    merged = dict(DEFAULTS)\n"
    "    merged.update(config)\n"
    "    alias = ALIASES.get(key, key)\n"
    "    return merged[key]\n"
)


def _alias_script(_n: int, _files: dict[str, str]) -> PatchReply:
    return PatchReply(Edit("buggy.py", "return merged[key]", "return merged[alias]"))


# --- golden_concat_syntax: a syntax error first, then the fix ---------------------------

_CONCAT_SOURCE = "def scale(text, factor):\n    base = text\n    return base + factor\n"


def _concat_script(n: int, files: dict[str, str]) -> PatchReply:
    source = files.get("buggy.py", "")
    if n == 1:
        return PatchReply(Edit("buggy.py", "return base + factor", "return int(base + factor"))
    if n == 2:
        # A literal backslash-n in code: the patch sanitizer rejects this one.
        return PatchReply(
            Edit("buggy.py", "return base + factor", "return int(base) + factor  \\n")
        )
    if "return int(base + factor" in source:
        return PatchReply(Edit("buggy.py", "return int(base + factor", "return int(base) + factor"))
    return PatchReply(Edit("buggy.py", "return base + factor", "return int(base) + factor"))


# --- golden_port_unfixable: every failure mode, never fixed -----------------------------

_PORT_SOURCE = "def parse_port(raw):\n    host, port = raw.split(':')\n    return int(port)\n"


def _port_script(n: int, _files: dict[str, str]) -> PatchReply:
    cycle = (n - 1) % 4
    if cycle in {0, 1}:
        # Edits mode: the find text is missing; diff mode: the context is stale.
        return PatchReply(
            Edit("buggy.py", "return int(port.strip())", "return int(port)"), stale_diff=True
        )
    if cycle == 2:
        return PatchReply(raw="I am not sure what to change here.")
    return PatchReply(Edit("buggy.py", "    return int(port)\n", "    return int(port)  # port\n"))


# --- golden_cross_file_units: the fix is in a helper module ------------------------------

_UNITS_HELPER = "MS_PER_SECOND = 100\n\n\ndef to_ms(seconds):\n    return seconds * MS_PER_SECOND\n"
_UNITS_SOURCE = (
    "from units import to_ms\n"
    "\n"
    "\n"
    "def budget_ms(timeouts):\n"
    "    return sum(to_ms(value) for value in timeouts)\n"
)


def _units_script(_n: int, _files: dict[str, str]) -> PatchReply:
    return PatchReply(Edit("units.py", "MS_PER_SECOND = 100", "MS_PER_SECOND = 1000"))


# --- golden_already_green: passes before any patch ------------------------------------


def _green_script(_n: int, _files: dict[str, str]) -> PatchReply:
    return PatchReply(raw="No change needed.")


GOLDEN_CASES: tuple[GoldenCase, ...] = (
    GoldenCase(
        case_id="golden_sum_skip_last",
        metadata=_metadata("logic_error", "AssertionError", "none"),
        files={
            "buggy.py": _SUM_SOURCE,
            "test_buggy.py": (
                "from buggy import total\n\n\ndef test_total():\n    assert total([1, 2, 3]) == 6\n"
            ),
        },
        patch_script=_sum_script,
    ),
    GoldenCase(
        case_id="golden_alias_keyerror",
        metadata=_metadata("reference_error", "KeyError", "required"),
        files={
            "buggy.py": _ALIAS_SOURCE,
            "test_buggy.py": (
                "from buggy import resolve\n\n\ndef test_alias():\n"
                '    assert resolve({}, "retry") == 3\n'
            ),
        },
        patch_script=_alias_script,
        stage_a_requests=[
            {"type": "local_var", "frame": 0, "name": "alias"},
            {"type": "file", "path": "buggy.py"},
        ],
    ),
    GoldenCase(
        case_id="golden_concat_syntax",
        metadata=_metadata("type_coercion", "TypeError", "helpful"),
        files={
            "buggy.py": _CONCAT_SOURCE,
            "test_buggy.py": (
                'from buggy import scale\n\n\ndef test_scale():\n    assert scale("2", 3) == 5\n'
            ),
        },
        patch_script=_concat_script,
        stage_a_requests=[{"type": "search", "path": "buggy.py", "pattern": "base"}],
    ),
    GoldenCase(
        case_id="golden_port_unfixable",
        metadata=_metadata("parse_value", "ValueError", "helpful"),
        files={
            "buggy.py": _PORT_SOURCE,
            "test_buggy.py": (
                "from buggy import parse_port\n\n\ndef test_ipv6():\n"
                '    assert parse_port("[::1]:8080") == 8080\n'
            ),
        },
        patch_script=_port_script,
        stage_a_requests=[{"type": "frame", "index": 0}],
    ),
    GoldenCase(
        case_id="golden_cross_file_units",
        metadata=_metadata("cross_file_contract", "AssertionError", "helpful"),
        files={
            "units.py": _UNITS_HELPER,
            "buggy.py": _UNITS_SOURCE,
            "test_buggy.py": (
                "from buggy import budget_ms\n\n\ndef test_budget():\n"
                "    assert budget_ms([1, 2]) == 3000\n"
            ),
        },
        patch_script=_units_script,
        stage_a_requests=[{"type": "file", "path": "units.py"}],
    ),
    GoldenCase(
        case_id="golden_already_green",
        metadata=_metadata("logic_error", "AssertionError", "none"),
        files={
            "buggy.py": "def double(value):\n    return value * 2\n",
            "test_buggy.py": (
                "from buggy import double\n\n\ndef test_double():\n    assert double(4) == 8\n"
            ),
        },
        patch_script=_green_script,
    ),
)


def render_edits(reply: PatchReply) -> str:
    if reply.raw is not None:
        return reply.raw
    assert reply.edit is not None
    edit = reply.edit
    return json.dumps({"edits": [{"path": edit.path, "find": edit.find, "replace": edit.replace}]})


def render_diff(reply: PatchReply, files: dict[str, str]) -> str:
    if reply.raw is not None:
        return reply.raw
    assert reply.edit is not None
    edit = reply.edit
    before = files.get(edit.path, "")
    if reply.stale_diff:
        before = "# context that is not in the file\n" + before
    after = before.replace(edit.find, edit.replace, 1)
    if reply.stale_diff:
        after = after + "# appended\n"
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{edit.path}",
        tofile=f"b/{edit.path}",
    )
    return "".join(lines)
