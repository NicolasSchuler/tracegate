"""Generated CLI-defaults tables for docs that list ``run_eval`` flags (X-08).

``docs/baseline-adapters.md`` restated the ``--baseline-*`` defaults by hand and had drifted:
it gave ``600.0`` and ``2`` for the process timeout and iteration count, which default to
other settings, and ``3072``/``6144`` for the INTERVENOR token caps, which are
``1024``/``2048``. The block below is rendered from ``build_parser()``, and
``evals.doc_sync --check`` fails when a doc's copy is stale.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .eval_cli_args import build_parser

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GeneratedBlock:
    """A marker-delimited block that one or more docs carry verbatim."""

    name: str
    docs: tuple[Path, ...]
    render_body: Callable[[], list[str]]

    @property
    def begin(self) -> str:
        return f"<!-- BEGIN AUTO-GENERATED: {self.name} -->"

    @property
    def end(self) -> str:
        return f"<!-- END AUTO-GENERATED: {self.name} -->"

    def render(self) -> str:
        return "\n".join(
            [
                self.begin,
                "<!-- Rendered from the code by `python -m evals.doc_sync --write`; do not edit. -->",
                "",
                *self.render_body(),
                self.end,
            ]
        )

    def _pattern(self) -> re.Pattern[str]:
        return re.compile(f"{re.escape(self.begin)}.*?{re.escape(self.end)}", re.DOTALL)

    def stale_docs(self) -> list[str]:
        """Docs whose block is missing or out of date (relative paths)."""
        expected = self.render()
        stale: list[str] = []
        for path in self.docs:
            match = self._pattern().search(path.read_text(encoding="utf-8"))
            if match is None or match.group(0) != expected:
                stale.append(str(path.relative_to(REPO_ROOT)))
        return stale

    def write(self) -> list[str]:
        """Rewrite each doc's block; returns the docs that changed. Docs need the markers."""
        expected = self.render()
        changed: list[str] = []
        for path in self.docs:
            text = path.read_text(encoding="utf-8")
            if self._pattern().search(text) is None:
                raise ValueError(f"{path}: missing {self.begin} ... {self.end}")
            updated = self._pattern().sub(lambda _match: expected, text, count=1)
            if updated != text:
                path.write_text(updated, encoding="utf-8")
                changed.append(str(path.relative_to(REPO_ROOT)))
        return changed


def _format_default(value: object) -> str:
    if value is None or value == "":
        return "unset"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    return f"`{value}`"


def _describe(text: str | None) -> str:
    """The help text on one line, without its "(default: ...)" (the table has a column)."""
    collapsed = " ".join((text or "").split())
    collapsed = re.sub(r"\s*\(default: [^)]*\)", "", collapsed)
    return collapsed.replace("|", "\\|")


def cli_flag_rows(prefix: str, parser: argparse.ArgumentParser | None = None) -> list[str]:
    """One table row per destination whose flag starts with *prefix*, in parser order.

    A ``--no-<flag>`` negation shares its destination and is listed with it.
    """
    parser = parser or build_parser()
    by_dest: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if any(option.startswith(prefix) for option in action.option_strings) or any(
            option.startswith(f"--no-{prefix[2:]}") for option in action.option_strings
        ):
            by_dest.setdefault(action.dest, []).append(action)
    rows: list[str] = []
    for dest, actions in by_dest.items():
        flags = " / ".join(
            f"`{option}`"
            for action in actions
            for option in action.option_strings
            if option.startswith("--")
        )
        primary = next(
            (a for a in actions if not a.option_strings[0].startswith("--no-")), actions[0]
        )
        rows.append(
            f"| {flags} | `{dest}` | {_format_default(parser.get_default(dest))} | "
            f"{_describe(primary.help)} |"
        )
    return rows


def _baseline_cli_body() -> list[str]:
    return [
        "| Flag | TOML key | CLI default | Description |",
        "|------|----------|-------------|-------------|",
        *cli_flag_rows("--baseline"),
        "",
        "`unset` means the flag has no value of its own; its description says what applies "
        "instead. Shipped templates under `evals/configs/` may set other values.",
    ]


BASELINE_CLI_DEFAULTS = GeneratedBlock(
    name="BASELINE_CLI_DEFAULTS",
    docs=(REPO_ROOT / "docs" / "baseline-adapters.md",),
    render_body=_baseline_cli_body,
)

GENERATED_CLI_BLOCKS: tuple[GeneratedBlock, ...] = (BASELINE_CLI_DEFAULTS,)
