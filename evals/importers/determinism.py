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
from collections.abc import Iterator
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
# Longest prefix first: "numpy.random." also matches nothing in "random.".
_RANDOM_FAMILY_BY_PREFIX = (("numpy.random.", "numpy"), ("random.", "random"))
_SEED_CALLS = {"random.seed": "random", "numpy.random.seed": "numpy"}
_SEED_KEYWORDS = frozenset({"a", "seed"})
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


def _is_a_set_expression(value: ast.expr) -> bool:
    if isinstance(value, (ast.Set, ast.SetComp)):
        return True
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in _SET_BUILDERS
    )


def _set_valued_names(tree: ast.AST) -> set[str]:
    """Names bound to a literal set, set comprehension or ``set()``/``frozenset()`` call.

    Iterating such a binding has the same hash-seed-dependent order as iterating the
    literal, so the checker must see through the indirection (DAT-14).
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_a_set_expression(node.value):
            targets = node.targets
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and _is_a_set_expression(node.value)
        ):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _iterates_a_set(iterable: ast.expr, set_valued_names: set[str]) -> bool:
    if _is_a_set_expression(iterable):
        return True
    return isinstance(iterable, ast.Name) and iterable.id in set_valued_names


def _random_family(name: str) -> str:
    for prefix, family in _RANDOM_FAMILY_BY_PREFIX:
        if name.startswith(prefix):
            return family
    return ""


def _constant_seed_argument(node: ast.Call) -> bool:
    """Whether the call passes a seed that is a constant other than ``None``.

    ``random.seed(None)`` and ``default_rng(seed=None)`` reseed from entropy, so an
    argument alone does not make the RNG reproducible.
    """
    values = [*node.args, *(kw.value for kw in node.keywords if kw.arg in _SEED_KEYWORDS)]
    if not values:
        return False
    return all(isinstance(value, ast.Constant) and value.value is not None for value in values)


def _walk_with_scope(
    node: ast.AST, scope: ast.AST | None = None
) -> Iterator[tuple[ast.AST, ast.AST | None]]:
    """Yield every descendant with the function scope that encloses it (``None`` = module)."""
    for child in ast.iter_child_nodes(node):
        child_scope = (
            child
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            else scope
        )
        yield child, child_scope
        yield from _walk_with_scope(child, child_scope)


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
    set_valued_names = _set_valued_names(tree)
    reasons: set[str] = set()
    if any(f"{name}.".startswith(_NETWORK_MODULES) for name in aliases.values()):
        reasons.add("network")  # importing a network client is enough to flag the case
    # (family, scope, lineno) of constant seeds, and of the random calls they may exempt.
    seeds: list[tuple[str, ast.AST | None, int]] = []
    random_uses: list[tuple[str, ast.AST | None, int]] = []
    for node, scope in _walk_with_scope(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and _iterates_a_set(
            node.iter, set_valued_names
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
            if _constant_seed_argument(node):
                seeds.append((_SEED_CALLS[name], scope, node.lineno))
        elif name in _SEEDABLE_CONSTRUCTORS:
            if not _constant_seed_argument(node):
                reasons.add("unseeded_random")
        elif name.startswith(_RANDOM_MODULES):
            random_uses.append((_random_family(name), scope, node.lineno))
    # DAT-14: a seed only exempts a random call when it is for the same RNG and is known to
    # run first -- earlier in the same function, or earlier at module level.
    for family, scope, lineno in random_uses:
        covered = any(
            seed_family == family
            and seed_lineno < lineno
            and (seed_scope is scope or seed_scope is None)
            for seed_family, seed_scope, seed_lineno in seeds
        )
        if not covered:
            reasons.add("unseeded_random")
            break
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
