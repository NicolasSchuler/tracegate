"""Synchronize auto-generated eval documentation blocks and release metadata checks.

This module owns the generated inventory block in ``evals/README.md`` and a
small set of documentation-consistency checks that are cheap enough to run in
local quality flows. The block markers are part of the contract: replacement
must update exactly one marked region.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    tomllib = None  # type: ignore[assignment]

from .case_loading import load_cases
from .doc_sync_cli import GENERATED_CLI_BLOCKS
from .doc_sync_policy import stale_policy_defaults_docs, write_policy_defaults_blocks
from .eval_types import CaseSpec

BEGIN_CASE_INVENTORY_MARKER = "<!-- BEGIN AUTO-GENERATED: CASE_INVENTORY -->"
END_CASE_INVENTORY_MARKER = "<!-- END AUTO-GENERATED: CASE_INVENTORY -->"

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = REPO_ROOT / "evals" / "cases"
EVALS_README_PATH = REPO_ROOT / "evals" / "README.md"
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

DIFFICULTY_ORDER = ("easy", "medium", "hard")

SOURCE_LABELS = {
    "hand_crafted": "Hand-crafted",
    "debugbench": "DebugBench",
    "pyresbugs": "PyResBugs",
    "condefects": "ConDefects",
    "humanevalpack": "HumanEvalPack",
    "bugsinpy": "BugsInPy",
    "pybughive": "PyBugHive",
    "trickybugs": "TrickyBugs",
    "mutmut": "mutmut",
}


@dataclass(frozen=True)
class CategorySummary:
    """Aggregate count for one eval category plus its difficulty breakdown."""

    name: str
    count: int
    difficulty_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class InventorySummary:
    """Rendered inventory inputs for the eval case catalog."""

    total_cases: int
    source_counts: tuple[tuple[str, int], ...]
    categories: tuple[CategorySummary, ...]


def _normalize_source(raw_source: str | None) -> str:
    if raw_source is None:
        return "hand_crafted"
    value = raw_source.strip()
    return value or "hand_crafted"


def _format_source_name(source: str) -> str:
    if source in SOURCE_LABELS:
        return SOURCE_LABELS[source]
    return source.replace("_", " ")


def summarize_cases(cases: Iterable[CaseSpec]) -> InventorySummary:
    """Aggregate loaded cases into the README inventory summary shape."""
    source_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    category_difficulty_counter: dict[str, Counter[str]] = defaultdict(Counter)
    total = 0

    for case in cases:
        total += 1
        source = _normalize_source(case.bug_source)
        source_counter[source] += 1
        category_counter[case.category] += 1
        category_difficulty_counter[case.category][case.difficulty] += 1

    sorted_sources = tuple(sorted(source_counter.items(), key=lambda item: (-item[1], item[0])))

    categories: list[CategorySummary] = []
    for category, count in sorted(category_counter.items(), key=lambda item: (-item[1], item[0])):
        raw_difficulties = category_difficulty_counter[category]
        known = [
            (difficulty, raw_difficulties[difficulty])
            for difficulty in DIFFICULTY_ORDER
            if raw_difficulties[difficulty] > 0
        ]
        extra = sorted(
            [
                (difficulty, diff_count)
                for difficulty, diff_count in raw_difficulties.items()
                if difficulty not in DIFFICULTY_ORDER and diff_count > 0
            ],
            key=lambda item: item[0],
        )
        categories.append(
            CategorySummary(
                name=category,
                count=count,
                difficulty_counts=(*known, *extra),
            )
        )

    return InventorySummary(
        total_cases=total,
        source_counts=sorted_sources,
        categories=tuple(categories),
    )


def build_inventory_summary(cases_dir: Path = CASES_DIR) -> InventorySummary:
    """Load cases from disk and summarize them for documentation output."""
    return summarize_cases(load_cases(cases_dir))


def _format_difficulty_breakdown(difficulty_counts: tuple[tuple[str, int], ...]) -> str:
    return ", ".join(f"{count} {difficulty}" for difficulty, count in difficulty_counts)


def render_case_inventory_block(summary: InventorySummary) -> str:
    """Render the exact auto-generated block expected inside ``evals/README.md``."""
    source_count = len(summary.source_counts)
    category_count = len(summary.categories)

    lines = [
        BEGIN_CASE_INVENTORY_MARKER,
        "_This section is auto-generated from `evals/cases` by "
        "`python -m evals.doc_sync --write`._",
        "",
        f"{summary.total_cases} validated cases across {category_count} categories "
        f"and {source_count} sources:",
        "",
        "| Source | Cases |",
        "|--------|-------|",
    ]
    for source, count in summary.source_counts:
        lines.append(f"| {_format_source_name(source)} | {count} |")
    lines.append(f"| **Total** | **{summary.total_cases}** |")
    lines.extend(
        [
            "",
            "Category distribution:",
            "",
            "| Category | Count | Difficulty breakdown |",
            "|----------|-------|---------------------|",
        ]
    )
    lines.extend(
        f"| {category.name} | {category.count} | "
        f"{_format_difficulty_breakdown(category.difficulty_counts)} |"
        for category in summary.categories
    )
    lines.append(END_CASE_INVENTORY_MARKER)
    return "\n".join(lines)


def _block_pattern() -> re.Pattern[str]:
    return re.compile(
        f"{re.escape(BEGIN_CASE_INVENTORY_MARKER)}.*?{re.escape(END_CASE_INVENTORY_MARKER)}",
        re.DOTALL,
    )


def extract_case_inventory_block(readme_text: str) -> str:
    """Extract the single generated inventory block from README text.

    Raises ``ValueError`` when the marker pair is missing so callers can report
    a broken documentation contract instead of silently writing a malformed file.
    """
    match = _block_pattern().search(readme_text)
    if match is None:
        raise ValueError(
            "Case inventory markers were not found in evals/README.md. "
            f"Expected markers: {BEGIN_CASE_INVENTORY_MARKER} ... {END_CASE_INVENTORY_MARKER}"
        )
    return match.group(0)


def replace_case_inventory_block(readme_text: str, replacement_block: str) -> str:
    """Replace the marked inventory block and preserve the rest of the README."""
    if _block_pattern().search(readme_text) is None:
        raise ValueError(
            "Case inventory markers were not found in evals/README.md. "
            f"Expected markers: {BEGIN_CASE_INVENTORY_MARKER} ... {END_CASE_INVENTORY_MARKER}"
        )
    return _block_pattern().sub(replacement_block, readme_text, count=1)


def parse_changelog_versions(changelog_text: str) -> set[str]:
    """Return released semantic versions declared as H2 changelog headings."""
    return set(re.findall(r"^##\s+v(\d+\.\d+\.\d+)\b", changelog_text, flags=re.MULTILINE))


def get_git_tag_versions(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return semantic-version git tags normalized without the leading ``v``."""
    proc = subprocess.run(
        ["git", "tag", "--sort=version:refname"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to read git tags: {proc.stderr.strip() or proc.stdout.strip()}")

    versions: list[str] = []
    for raw_tag in proc.stdout.splitlines():
        tag = raw_tag.strip()
        match = re.fullmatch(r"v?(\d+\.\d+\.\d+)", tag)
        if match:
            versions.append(match.group(1))
    return versions


def _load_project_version_regex_fallback(pyproject_text: str) -> str:
    in_project_section = False
    for raw_line in pyproject_text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project_section = line == "[project]"
            continue
        if not in_project_section or not line.startswith("version"):
            continue
        match = re.fullmatch(r'version\s*=\s*"([^"]+)"', line)
        if match is None:
            break
        version = match.group(1).strip()
        if version:
            return version
        break
    raise ValueError("Could not parse project.version from pyproject.toml")


def load_project_version(pyproject_path: Path = PYPROJECT_PATH) -> str:
    """Load ``project.version`` from ``pyproject.toml`` with a regex fallback."""
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    if tomllib is None:
        return _load_project_version_regex_fallback(pyproject_text)

    pyproject = tomllib.loads(pyproject_text)
    version = pyproject["project"]["version"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError("project.version in pyproject.toml must be a non-empty string")
    return version.strip()


def _first_h2_heading(changelog_text: str) -> str | None:
    for line in changelog_text.splitlines():
        if line.startswith("## "):
            return line.strip()
    return None


def validate_changelog_consistency(
    changelog_text: str,
    *,
    tagged_versions: list[str],
    project_version: str,
) -> list[str]:
    """Return human-readable changelog/tag consistency errors.

    Tagged releases must appear in ``CHANGELOG.md``. If ``project.version`` is
    newer than the newest tag, the top changelog heading must be an
    unreleased/current section that mentions that version.
    """
    errors: list[str] = []
    changelog_versions = parse_changelog_versions(changelog_text)
    missing_tagged_versions = [v for v in tagged_versions if v not in changelog_versions]
    if missing_tagged_versions:
        errors.append(
            "CHANGELOG is missing tagged versions: "
            + ", ".join(f"v{version}" for version in missing_tagged_versions)
        )

    if project_version not in tagged_versions:
        first_heading = _first_h2_heading(changelog_text)
        heading_pattern = re.compile(
            rf"^##\s*(?:Unreleased|Current)\b.*\b{re.escape(project_version)}\b",
            flags=re.IGNORECASE,
        )
        if first_heading is None or heading_pattern.search(first_heading) is None:
            errors.append(
                "project.version is not tagged; expected a top changelog section with "
                f"`Unreleased` or `Current` that mentions {project_version}."
            )

    return errors


def write_case_inventory_block() -> tuple[bool, str]:
    """Rewrite the README inventory block when the rendered content changed."""
    summary = build_inventory_summary()
    desired_block = render_case_inventory_block(summary)
    current_text = EVALS_README_PATH.read_text(encoding="utf-8")
    updated_text = replace_case_inventory_block(current_text, desired_block)
    if updated_text == current_text:
        return False, "evals/README.md already up to date."
    EVALS_README_PATH.write_text(updated_text, encoding="utf-8")
    return True, "Updated evals/README.md case inventory block."


def run_checks() -> list[str]:
    """Run the non-mutating documentation drift checks used by docs-check."""
    errors: list[str] = []

    summary = build_inventory_summary()
    expected_block = render_case_inventory_block(summary)
    readme_text = EVALS_README_PATH.read_text(encoding="utf-8")
    try:
        current_block = extract_case_inventory_block(readme_text)
    except ValueError as e:
        errors.append(str(e))
    else:
        if current_block != expected_block:
            errors.append(
                "Case inventory block in evals/README.md is out of date. "
                "Run `python -m evals.doc_sync --write`."
            )

    errors.extend(
        f"Policy defaults block in {doc} is out of date. Run `python -m evals.doc_sync --write`."
        for doc in stale_policy_defaults_docs()
    )
    errors.extend(
        f"{block.name} block in {doc} is out of date. Run `python -m evals.doc_sync --write`."
        for block in GENERATED_CLI_BLOCKS
        for doc in block.stale_docs()
    )

    changelog_text = CHANGELOG_PATH.read_text(encoding="utf-8")
    tagged_versions = get_git_tag_versions()
    project_version = load_project_version()
    errors.extend(
        validate_changelog_consistency(
            changelog_text,
            tagged_versions=tagged_versions,
            project_version=project_version,
        )
    )

    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m evals.doc_sync``."""
    parser = argparse.ArgumentParser(description="Synchronize and verify documentation drift.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Write auto-generated docs content.")
    mode.add_argument(
        "--check",
        action="store_true",
        help="Check docs content and fail with non-zero status on drift.",
    )
    args = parser.parse_args(argv)

    if args.write:
        _changed, message = write_case_inventory_block()
        print(message)
        for doc in write_policy_defaults_blocks():
            print(f"Updated {doc} policy defaults block.")
        for block in GENERATED_CLI_BLOCKS:
            for doc in block.write():
                print(f"Updated {doc} {block.name} block.")
        return 0

    errors = run_checks()
    if errors:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Documentation sync checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
