"""AST-based determinism check for imported cases (DAT-14).

The previous filter matched six substrings, so ``time.time()``, ``datetime.now()``,
``random.randint``/``choice``/``shuffle``/``sample``, ``uuid``, ``os.urandom`` and set
iteration all passed. A case whose outcome varies between runs adds variance that looks
like a condition effect.

Scan a corpus with ``python -m evals.importers.determinism [cases_dir]``.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

_CLOCK_CALLS = frozenset(
    {
        "time.time",
        "time.time_ns",
        "time.perf_counter",
        "time.perf_counter_ns",
        "time.monotonic",
        "time.monotonic_ns",
        "time.process_time",
        "time.process_time_ns",
        "time.sleep",
        "time.localtime",
        "time.gmtime",
        "time.ctime",
        "time.asctime",
        "datetime.datetime.now",
        "datetime.datetime.utcnow",
        "datetime.datetime.today",
        "datetime.date.today",
    }
)
_ENTROPY_CALLS = frozenset(
    {"uuid.uuid1", "uuid.uuid4", "os.urandom", "os.getrandom", "random.SystemRandom"}
)
_ENTROPY_MODULES = ("secrets.",)
_NETWORK_MODULES = ("requests.", "urllib.", "http.", "socket.", "httpx.", "aiohttp.")
_RANDOM_MODULES = ("random.", "numpy.random.")
_SEED_CALLS = frozenset({"random.seed", "numpy.random.seed"})
_SEEDABLE_CONSTRUCTORS = frozenset(
    {"random.Random", "numpy.random.default_rng", "numpy.random.RandomState"}
)
_SET_BUILDERS = frozenset({"set", "frozenset"})
_FALLBACK_MARKERS = ("requests.", "urllib.", "http.", "socket.", "time.sleep", "random.random")


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    """Map local names to the qualified names they were imported as."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    aliases[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _qualified_name(func: ast.expr, aliases: dict[str, str]) -> str:
    parts: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(aliases.get(node.id, node.id))
    return ".".join(reversed(parts))


def _iterates_a_set(iterable: ast.expr) -> bool:
    if isinstance(iterable, (ast.Set, ast.SetComp)):
        return True
    return (
        isinstance(iterable, ast.Call)
        and isinstance(iterable.func, ast.Name)
        and iterable.func.id in _SET_BUILDERS
    )


def nondeterminism_reasons(code: str) -> list[str]:
    """Return why *code* may behave differently between runs (empty when deterministic).

    Reasons: ``clock`` (wall-clock or sleep), ``entropy`` (uuid, ``os.urandom``,
    ``secrets``), ``network``, ``unseeded_random`` (module-level ``random``/``numpy.random``
    without a constant seed, or an unseeded generator), and ``set_iteration`` (string
    hashes vary unless ``PYTHONHASHSEED`` is fixed).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        lower = code.lower()
        return ["unparsable_risky_call"] if any(m in lower for m in _FALLBACK_MARKERS) else []

    aliases = _import_aliases(tree)
    reasons: set[str] = set()
    if any(f"{name}.".startswith(_NETWORK_MODULES) for name in aliases.values()):
        reasons.add("network")  # importing a network client is enough to flag the case
    seeded = False
    random_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and _iterates_a_set(
            node.iter
        ):
            reasons.add("set_iteration")
        if not isinstance(node, ast.Call):
            continue
        name = _qualified_name(node.func, aliases)
        if not name:
            continue
        if name in _CLOCK_CALLS:
            reasons.add("clock")
        elif name in _ENTROPY_CALLS or name.startswith(_ENTROPY_MODULES):
            reasons.add("entropy")
        elif name.startswith(_NETWORK_MODULES):
            reasons.add("network")
        elif name in _SEED_CALLS:
            seeded = seeded or bool(node.args or node.keywords)
        elif name in _SEEDABLE_CONSTRUCTORS:
            if not (node.args or node.keywords):
                reasons.add("unseeded_random")
        elif name.startswith(_RANDOM_MODULES):
            random_calls += 1
    if random_calls and not seeded:
        reasons.add("unseeded_random")
    return sorted(reasons)


def is_deterministic(code: str) -> bool:
    """Whether *code* has no detectable source of run-to-run variation."""
    return not nondeterminism_reasons(code)


_SCANNED_FILES = ("buggy.py", "test_buggy.py")


def scan_cases(cases_dir: Path) -> dict[str, list[str]]:
    """Return ``{case_id: ["<file>:<reason>", ...]}`` for every flagged case under *cases_dir*."""
    flagged: dict[str, list[str]] = {}
    for case_json in sorted(cases_dir.rglob("case.json")):
        reasons = [
            f"{name}:{reason}"
            for name in _SCANNED_FILES
            if (case_json.parent / name).is_file()
            for reason in nondeterminism_reasons(
                (case_json.parent / name).read_text(encoding="utf-8", errors="replace")
            )
        ]
        if reasons:
            flagged[case_json.parent.name] = reasons
    return flagged


def main(argv: list[str] | None = None) -> int:
    """Print flagged cases; exit 1 when any case is flagged."""
    parser = argparse.ArgumentParser(
        description="Flag cases whose code may behave differently between runs (DAT-14)."
    )
    parser.add_argument("cases_dir", nargs="?", type=Path, default=Path("evals/cases"))
    args = parser.parse_args(argv)
    flagged = scan_cases(args.cases_dir)
    for case_id, reasons in flagged.items():
        print(f"{case_id}: {', '.join(reasons)}")
    print(f"{len(flagged)} flagged case(s)")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
