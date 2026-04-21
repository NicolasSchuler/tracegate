"""Live and plain progress-display implementations for eval runs.

The display layer turns resolved config plus per-unit progress events into a
stable terminal view. ``DisplayUnitState`` models one case/condition pair and
tracks both machine state (running, skipped, done) and the latest human-facing
activity hints.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, TextIO

from evals.eval_config import _resolved_args_for_lockfile
from evals.progress_output import humanize_failure_reason

try:
    from rich.columns import Columns
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _stdout_is_tty(stream: TextIO | None = None) -> bool:
    target = sys.stdout if stream is None else stream
    isatty = getattr(target, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    whole_seconds = max(0, round(float(seconds)))
    hours, rem = divmod(whole_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _join_nonempty(parts: Iterable[str], *, sep: str = "  ") -> str:
    return sep.join(part for part in parts if str(part).strip())


def _format_csv(values: Sequence[str]) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(cleaned) if cleaned else "-"


def _format_case_filters(
    *,
    cases: Sequence[Any],
    attr_name: str,
) -> str:
    values = sorted(
        {
            str(getattr(case, attr_name, "") or "").strip()
            for case in cases
            if str(getattr(case, attr_name, "") or "").strip()
        }
    )
    return _format_csv(values)


def _top_failure_reasons_text(failure_reasons: Counter[str], *, top_n: int = 3) -> str:
    if not failure_reasons:
        return "none"
    ranked = sorted(
        failure_reasons.items(),
        key=lambda item: (-item[1], item[0]),
    )[: max(1, int(top_n))]
    return ", ".join(f"{humanize_failure_reason(reason)} {count}" for reason, count in ranked)


def _format_resolved_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value if value else '""'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_resolved_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{key}={_format_resolved_value(raw)}" for key, raw in sorted(value.items())
            )
            + "}"
        )
    return str(value)


def _resolved_config_section_title_for_key(key: str) -> str:
    if key in {
        "run_id",
        "conditions",
        "patcher",
        "jobs",
        "k",
        "keep_workdirs",
        "deterministic",
        "seed",
    }:
        return "Resolved Config / General"
    if key in {
        "cases_dir",
        "case",
        "case_list_file",
        "category",
        "difficulty",
        "snapshot_dependency",
        "bug_source",
        "max_cases",
    }:
        return "Resolved Config / Selection Filters"
    if key == "executor" or key.startswith("container_"):
        return "Resolved Config / Execution Backend"
    if key.startswith("openai_"):
        return "Resolved Config / OpenAI"
    if key.startswith("context_") or key.startswith("retry_") or key.startswith("prompt_"):
        return "Resolved Config / Prompting & Budgets"
    if key.startswith("adaptive_"):
        return "Resolved Config / Adaptive"
    if key == "baseline" or key.startswith("baseline_"):
        return "Resolved Config / Baseline"
    if key.startswith("patch_") or key.startswith("syntax_"):
        return "Resolved Config / Patch & Verification"
    if key in {
        "output_format",
        "transport_fail_fast_streak",
        "case_fail_fast_on_request_budget_exceeded",
        "force_patch_attempt",
        "openai_force_patch_attempt",
    }:
        return "Resolved Config / Runtime Controls"
    if (
        key.startswith("record_")
        or key.startswith("progress_")
        or key
        in {
            "pytest_output_tail_lines",
            "snapshot_artifact_max_string_chars",
        }
    ):
        return "Resolved Config / Recording & Progress"
    return "Resolved Config / Other"


def _build_resolved_config_sections(*, args: Any, excluded_keys: set[str]) -> list[StartupSection]:
    resolved_args = _resolved_args_for_lockfile(args)
    grouped_rows: dict[str, list[tuple[str, str]]] = {}
    for key, value in resolved_args.items():
        if key in excluded_keys:
            continue
        title = _resolved_config_section_title_for_key(str(key))
        grouped_rows.setdefault(title, []).append((str(key), _format_resolved_value(value)))

    section_order = [
        "Resolved Config / General",
        "Resolved Config / Selection Filters",
        "Resolved Config / Execution Backend",
        "Resolved Config / OpenAI",
        "Resolved Config / Prompting & Budgets",
        "Resolved Config / Adaptive",
        "Resolved Config / Baseline",
        "Resolved Config / Patch & Verification",
        "Resolved Config / Runtime Controls",
        "Resolved Config / Recording & Progress",
        "Resolved Config / Other",
    ]
    return [
        StartupSection(title=title, rows=tuple(grouped_rows[title]))
        for title in section_order
        if grouped_rows.get(title)
    ]


@dataclass(frozen=True)
class StartupSection:
    """One startup panel or plaintext section rendered before the run begins."""

    title: str
    rows: tuple[tuple[str, str], ...]


@dataclass
class DisplayRunSummary:
    """Top-level aggregate state shared by all progress display backends."""

    total_units: int = 0
    completed_units: int = 0
    skipped_units: int = 0
    success_units: int = 0
    failed_units: int = 0
    interrupted_units: int = 0
    runtime_error_units: int = 0
    failure_reasons: Counter[str] = field(default_factory=Counter)
    elapsed_sec: float = 0.0
    eta_sec: float | None = None
    prompt_chars_total: int = 0
    completion_chars_total: int = 0
    token_usage_text: str = "tokens=n/a"

    @property
    def progress_pct(self) -> float:
        if self.total_units <= 0:
            return 100.0
        return 100.0 * (self.completed_units / self.total_units)


@dataclass
class DisplayUnitState:
    """Mutable per-unit display state for one ``case_id``/condition pair.

    ``status`` is the coarse lifecycle state, while ``phase`` and the activity
    fields capture the latest meaningful work the runner reported.
    """

    case_id: str
    condition: str
    status: str = "queued"
    attempt: int = 0
    max_attempts: int = 0
    phase: str = "queued"
    last_note: str = ""
    observed_records: int = 0
    failure_reason: str = ""
    waiting_for: str = ""
    external_operation: str = ""
    external_provider: str = ""
    owner: str = ""
    source: str = ""
    activity_excerpt: str = ""
    attempts_completed: int = 0
    llm_calls_observed: int = 0
    eval_duration_sec_observed: float = 0.0
    started_monotonic: float | None = None
    phase_started_monotonic: float | None = None
    last_update_monotonic: float | None = None
    last_heartbeat_monotonic: float | None = None
    last_meaningful_progress_monotonic: float | None = None

    def start(self) -> None:
        now = time.monotonic()
        self.status = "running"
        self.phase = "queued"
        self.attempt = 0
        self.max_attempts = 0
        self.last_note = ""
        self.observed_records = 0
        self.failure_reason = ""
        self.waiting_for = ""
        self.external_operation = ""
        self.external_provider = ""
        self.owner = ""
        self.source = ""
        self.activity_excerpt = ""
        self.attempts_completed = 0
        self.llm_calls_observed = 0
        self.eval_duration_sec_observed = 0.0
        self.started_monotonic = now
        self.phase_started_monotonic = now
        self.last_update_monotonic = now
        self.last_heartbeat_monotonic = None
        self.last_meaningful_progress_monotonic = now

    def skip(self) -> None:
        now = time.monotonic()
        self.status = "skipped"
        self.phase = "resumed"
        self.failure_reason = ""
        self.last_update_monotonic = now
        self.last_meaningful_progress_monotonic = now

    def finish(self, *, status: str, failure_reason: str = "") -> None:
        now = time.monotonic()
        self.last_update_monotonic = now
        self.last_meaningful_progress_monotonic = now
        self.status = status
        self.phase = "done" if status != "error" else "error"
        self.failure_reason = str(failure_reason or "")

    def apply_progress(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        now = time.monotonic()
        self.last_update_monotonic = now
        if event == "heartbeat":
            self.last_heartbeat_monotonic = now
            if payload is not None:
                phase_value = payload.get("phase")
                if str(phase_value or "").strip():
                    self.phase = str(phase_value)
                phase_elapsed_value = payload.get("phase_elapsed_sec")
                if isinstance(phase_elapsed_value, (int, float)):
                    self.phase_started_monotonic = now - max(0.0, float(phase_elapsed_value))
        else:
            phase_label = (
                str(payload.get("phase_label") or "").strip() if payload is not None else ""
            )
            phase_value = phase_label or str(event or self.phase)
            phase_changed = self.phase != phase_value
            self.phase = phase_value
            if phase_changed:
                self.phase_started_monotonic = now
                self.waiting_for = ""
                self.external_operation = ""
                self.external_provider = ""
                self.owner = ""
                self.source = ""
                self.activity_excerpt = ""
            self.last_meaningful_progress_monotonic = now
        if payload is None:
            return
        attempt_value = payload.get("attempt")
        max_attempts_value = payload.get("max_attempts")
        if isinstance(attempt_value, int):
            self.attempt = int(attempt_value)
        if isinstance(max_attempts_value, int):
            self.max_attempts = int(max_attempts_value)
        note_value = payload.get("note")
        if note_value is not None:
            self.last_note = str(note_value)
        for payload_key, attr_name in (
            ("waiting_for", "waiting_for"),
            ("external_operation", "external_operation"),
            ("external_provider", "external_provider"),
            ("owner", "owner"),
            ("source", "source"),
            ("activity_excerpt", "activity_excerpt"),
        ):
            if payload_key in payload:
                setattr(self, attr_name, str(payload.get(payload_key) or ""))
        attempts_completed_value = payload.get("attempts_completed")
        if isinstance(attempts_completed_value, int):
            self.attempts_completed = max(0, int(attempts_completed_value))
        llm_calls_observed_value = payload.get("llm_calls_observed")
        if isinstance(llm_calls_observed_value, int):
            self.llm_calls_observed = max(0, int(llm_calls_observed_value))
        eval_duration_sec_observed_value = payload.get("eval_duration_sec_observed")
        if isinstance(eval_duration_sec_observed_value, (int, float)):
            self.eval_duration_sec_observed = max(0.0, float(eval_duration_sec_observed_value))
        records_value = payload.get("records_observed")
        if isinstance(records_value, int):
            self.observed_records = max(0, int(records_value))
        elif event == "record_emitted":
            self.observed_records += 1

    def phase_elapsed_text(self) -> str:
        if self.phase_started_monotonic is None:
            return "-"
        return _format_duration(time.monotonic() - self.phase_started_monotonic)

    def activity_status_text(self) -> str:
        if self.status != "running":
            return self.status
        if self.last_meaningful_progress_monotonic is None:
            return "starting"
        quiet_sec = max(0.0, time.monotonic() - self.last_meaningful_progress_monotonic)
        if quiet_sec < 3.0:
            return "running"
        return f"quiet {_format_duration(quiet_sec)}"

    def notes_text(self) -> str:
        parts = [f"recs={self.observed_records}"]
        owner = self.owner.strip()
        source = self.source.strip()
        if owner:
            parts.append(f"owner={owner}")
        if source:
            parts.append(f"source={source}")
        wait_for = self.waiting_for.strip()
        if wait_for:
            parts.append(f"wait={wait_for}")
        provider = self.external_provider.strip()
        operation = self.external_operation.strip()
        if provider and not wait_for:
            parts.append(f"provider={provider}")
        if operation:
            parts.append(f"op={operation}")
        if self.attempts_completed > 0:
            parts.append(f"iters={self.attempts_completed}")
        if self.llm_calls_observed > 0:
            parts.append(f"llm_calls={self.llm_calls_observed}")
        if self.eval_duration_sec_observed > 0:
            parts.append(f"eval={self.eval_duration_sec_observed:.1f}s")
        if self.last_note.strip():
            parts.append(self.last_note.strip())
        if self.activity_excerpt.strip():
            parts.append(f"activity={self.activity_excerpt.strip()}")
        return " | ".join(parts)


class RunDisplay(Protocol):
    """Interface shared by plain and rich run-display implementations."""

    suppress_inline_progress: bool

    def render_startup(self, sections: Sequence[StartupSection]) -> None: ...

    def start_runtime(self) -> None: ...

    def update_summary(self, summary: DisplayRunSummary) -> None: ...

    def start_unit(self, *, case_id: str, condition: str) -> None: ...

    def skip_unit(self, *, case_id: str, condition: str) -> None: ...

    def update_unit_progress(
        self,
        *,
        case_id: str,
        condition: str,
        event: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None: ...

    def finish_unit(
        self,
        *,
        case_id: str,
        condition: str,
        status: str,
        failure_reason: str = "",
    ) -> None: ...

    def add_event(self, *, message: str, style: str = "info") -> None: ...

    def render_final_lines(self, lines: Sequence[str]) -> None: ...

    def close(self) -> None: ...


class PlainRunDisplay:
    """Minimal stdout display used when rich/TTY rendering is unavailable."""

    suppress_inline_progress = False

    def __init__(self) -> None:
        self.summary: DisplayRunSummary | None = None

    def render_startup(self, sections: Sequence[StartupSection]) -> None:
        for section in sections:
            print(f"{section.title}:")
            for label, value in section.rows:
                print(f"  {label}: {value}")

    def update_summary(self, summary: DisplayRunSummary) -> None:
        self.summary = summary

    def start_runtime(self) -> None:
        return

    def start_unit(self, *, case_id: str, condition: str) -> None:
        del case_id, condition

    def skip_unit(self, *, case_id: str, condition: str) -> None:
        del case_id, condition

    def update_unit_progress(
        self,
        *,
        case_id: str,
        condition: str,
        event: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        del case_id, condition, event, payload

    def finish_unit(
        self,
        *,
        case_id: str,
        condition: str,
        status: str,
        failure_reason: str = "",
    ) -> None:
        del case_id, condition, status, failure_reason

    def add_event(self, *, message: str, style: str = "info") -> None:
        del style
        print(message, flush=True)

    def render_final_lines(self, lines: Sequence[str]) -> None:
        for line in lines:
            print(line)

    def close(self) -> None:
        return


class RichRunDisplay:
    """Interactive Rich-based display with live summary and per-unit panels."""

    suppress_inline_progress = True

    def __init__(self, *, total_units: int, verbosity: str) -> None:
        if not HAS_RICH:
            raise RuntimeError("RichRunDisplay requires rich")
        self.console = Console(stderr=False)
        self._state_lock = threading.Lock()
        self.summary = DisplayRunSummary(
            total_units=max(0, int(total_units)),
            completed_units=0,
            skipped_units=0,
            success_units=0,
            failed_units=0,
            interrupted_units=0,
            runtime_error_units=0,
        )
        self.verbosity = str(verbosity or "basic")
        self.units: dict[tuple[str, str], DisplayUnitState] = {}
        self.events: deque[tuple[str, str]] = deque(maxlen=8)
        self._summary_updated_monotonic = time.monotonic()
        self._runtime_started = False
        self.progress = Progress(
            SpinnerColumn(style="bold green"),
            TextColumn("[bold cyan]Progress"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("{task.percentage:>5.1f}%"),
            console=self.console,
            expand=True,
        )
        self.progress_task_id = self.progress.add_task(
            "evals",
            total=max(1, int(total_units)),
            completed=0,
        )
        self.live = Live(
            console=self.console,
            refresh_per_second=6,
            transient=False,
            get_renderable=self._build_dashboard,
        )

    def _unit_key(self, *, case_id: str, condition: str) -> tuple[str, str]:
        return (str(case_id), str(condition))

    def _ensure_unit(self, *, case_id: str, condition: str) -> DisplayUnitState:
        key = self._unit_key(case_id=case_id, condition=condition)
        unit = self.units.get(key)
        if unit is None:
            unit = DisplayUnitState(case_id=str(case_id), condition=str(condition))
            self.units[key] = unit
        return unit

    def _copy_summary(self, summary: DisplayRunSummary) -> DisplayRunSummary:
        return replace(summary, failure_reasons=Counter(summary.failure_reasons))

    def _refresh(self) -> None:
        with self._state_lock:
            runtime_started = self._runtime_started
        if runtime_started:
            self.live.refresh()

    def _render_summary_from_snapshot(
        self,
        *,
        summary: DisplayRunSummary,
        runtime_started: bool,
        summary_updated_monotonic: float,
    ) -> DisplayRunSummary:
        if (not runtime_started) or summary.completed_units >= summary.total_units:
            return summary
        delta_sec = max(0.0, time.monotonic() - summary_updated_monotonic)
        if delta_sec <= 0.0:
            return summary
        eta_sec = summary.eta_sec
        live_eta_sec = None if eta_sec is None else max(0.0, eta_sec - delta_sec)
        return replace(
            summary,
            elapsed_sec=summary.elapsed_sec + delta_sec,
            eta_sec=live_eta_sec,
        )

    def _render_summary(self) -> DisplayRunSummary:
        with self._state_lock:
            summary = self._copy_summary(self.summary)
            runtime_started = self._runtime_started
            summary_updated_monotonic = self._summary_updated_monotonic
        return self._render_summary_from_snapshot(
            summary=summary,
            runtime_started=runtime_started,
            summary_updated_monotonic=summary_updated_monotonic,
        )

    def _event_text(self, message: str, style: str) -> Text:
        text = Text()
        prefix_style = {
            "success": "bold green",
            "error": "bold red",
            "warning": "bold yellow",
            "skip": "bold cyan",
        }.get(style, "bold white")
        text.append(message, style=prefix_style)
        return text

    def _build_summary_panel(self, summary: DisplayRunSummary) -> Panel:
        summary_table = Table.grid(expand=True)
        summary_table.add_column()
        summary_table.add_column()
        summary_table.add_column()
        summary_table.add_column()
        summary_table.add_row(
            f"[green]success[/green] {summary.success_units}",
            f"[red]failed[/red] {summary.failed_units}",
            f"[yellow]interrupted[/yellow] {summary.interrupted_units}",
            f"[yellow]runtime errors[/yellow] {summary.runtime_error_units}",
        )
        summary_table.add_row(
            f"[cyan]skipped[/cyan] {summary.skipped_units}",
            "",
            "",
            "",
        )
        summary_table.add_row(
            f"elapsed {_format_duration(summary.elapsed_sec)}",
            f"eta {_format_duration(summary.eta_sec)}",
            f"chars in {summary.prompt_chars_total}",
            f"chars out {summary.completion_chars_total}",
        )
        summary_table.add_row(
            f"top failures {_top_failure_reasons_text(summary.failure_reasons)}",
            summary.token_usage_text,
            "",
            "",
        )
        return Panel(summary_table, title="Summary", border_style="cyan")

    def _build_active_units_panel(self, units: Sequence[DisplayUnitState] | None = None) -> Panel:
        if units is None:
            with self._state_lock:
                units = [replace(unit) for unit in self.units.values()]
        table = Table(expand=True)
        table.add_column("Unit")
        table.add_column("Status")
        table.add_column("Attempt", justify="right")
        table.add_column("Phase")
        table.add_column("Elapsed", justify="right")
        table.add_column("Notes")
        rows_added = 0
        running_units = [unit for unit in units if unit.status == "running"]
        running_units.sort(
            key=lambda unit: (
                unit.started_monotonic if unit.started_monotonic is not None else float("inf"),
                unit.case_id,
                unit.condition,
            )
        )
        for unit in running_units[:6]:
            attempt_text = (
                f"{unit.attempt}/{unit.max_attempts}"
                if unit.attempt > 0 and unit.max_attempts > 0
                else ("-" if unit.attempt <= 0 else str(unit.attempt))
            )
            table.add_row(
                f"{unit.case_id} / {unit.condition}",
                unit.activity_status_text(),
                attempt_text,
                unit.phase,
                unit.phase_elapsed_text(),
                unit.notes_text(),
            )
            rows_added += 1
        if rows_added == 0:
            table.add_row("No active units", "-", "-", "-", "-", "-")
        return Panel(table, title="Active Units", border_style="magenta")

    def _build_events_panel(self, events: Sequence[tuple[str, str]] | None = None) -> Panel:
        if events is None:
            with self._state_lock:
                events = tuple(self.events)
        if not events:
            body: Any = Text("No events yet", style="dim")
        else:
            body = Group(*[self._event_text(message, style) for message, style in events])
        return Panel(body, title="Latest Events", border_style="green")

    def _build_dashboard(self) -> Group:
        with self._state_lock:
            summary = self._copy_summary(self.summary)
            runtime_started = self._runtime_started
            summary_updated_monotonic = self._summary_updated_monotonic
            units = [replace(unit) for unit in self.units.values()]
            events = tuple(self.events)
        summary = self._render_summary_from_snapshot(
            summary=summary,
            runtime_started=runtime_started,
            summary_updated_monotonic=summary_updated_monotonic,
        )
        header = Text.assemble(
            ("llmdebug eval run", "bold"),
            "  ",
            (f"{summary.completed_units}/{summary.total_units} units", "bold cyan"),
            "  ",
            (f"{summary.progress_pct:.1f}%", "bold magenta"),
            "  ",
            (f"elapsed {_format_duration(summary.elapsed_sec)}", "white"),
            "  ",
            (f"eta {_format_duration(summary.eta_sec)}", "white"),
        )
        self.progress.update(
            self.progress_task_id,
            completed=max(0, int(summary.completed_units)),
            total=max(1, int(summary.total_units)),
        )
        body = Columns(
            [self._build_summary_panel(summary), self._build_active_units_panel(units)],
            equal=True,
            expand=True,
        )
        return Group(header, self.progress, body, self._build_events_panel(events))

    def render_startup(self, sections: Sequence[StartupSection]) -> None:
        if not sections:
            return
        featured = sections[:1]
        remaining = list(sections[1:])
        for section in featured:
            self.console.print(self._section_panel(section, expand=True))
        if remaining:
            paired: list[Any] = []
            for index in range(0, len(remaining), 2):
                chunk = remaining[index : index + 2]
                paired.append(Columns([self._section_panel(item) for item in chunk], equal=True))
            for renderable in paired:
                self.console.print(renderable)

    def _section_panel(self, section: StartupSection, *, expand: bool = False) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(style="bold cyan", ratio=1)
        table.add_column(style="white", ratio=3)
        for label, value in section.rows:
            table.add_row(label, value)
        return Panel(table, title=section.title, border_style="blue", expand=expand)

    def update_summary(self, summary: DisplayRunSummary) -> None:
        with self._state_lock:
            self.summary = self._copy_summary(summary)
            self._summary_updated_monotonic = time.monotonic()
        self._refresh()

    def start_runtime(self) -> None:
        with self._state_lock:
            if self._runtime_started:
                return
            self._runtime_started = True
            self._summary_updated_monotonic = time.monotonic()
        self.live.start()
        self._refresh()

    def start_unit(self, *, case_id: str, condition: str) -> None:
        with self._state_lock:
            self._ensure_unit(case_id=case_id, condition=condition).start()
        self._refresh()

    def skip_unit(self, *, case_id: str, condition: str) -> None:
        with self._state_lock:
            self._ensure_unit(case_id=case_id, condition=condition).skip()
        self._refresh()

    def update_unit_progress(
        self,
        *,
        case_id: str,
        condition: str,
        event: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        with self._state_lock:
            self._ensure_unit(case_id=case_id, condition=condition).apply_progress(
                event=event,
                payload=payload,
            )
        self._refresh()

    def finish_unit(
        self,
        *,
        case_id: str,
        condition: str,
        status: str,
        failure_reason: str = "",
    ) -> None:
        with self._state_lock:
            self._ensure_unit(case_id=case_id, condition=condition).finish(
                status=status,
                failure_reason=failure_reason,
            )
        self._refresh()

    def add_event(self, *, message: str, style: str = "info") -> None:
        with self._state_lock:
            self.events.appendleft((message, style))
        self._refresh()

    def render_final_lines(self, lines: Sequence[str]) -> None:
        for line in lines:
            self.console.print(line)

    def close(self) -> None:
        with self._state_lock:
            runtime_started = self._runtime_started
        if runtime_started:
            self.live.stop()


def create_run_display(
    *,
    total_units: int,
    verbosity: str,
    stream: TextIO | None = None,
) -> RunDisplay:
    """Create the best available display backend for the current terminal."""
    if HAS_RICH and _stdout_is_tty(stream) and os.getenv("NO_COLOR") is None:
        return RichRunDisplay(total_units=total_units, verbosity=verbosity)
    return PlainRunDisplay()


def build_startup_sections(
    *,
    args: Any,
    cases: Sequence[Any],
    conditions: Sequence[Any],
    has_adaptive_condition: bool,
    run_id: str,
    results_path: Path,
    config_paths: Sequence[str],
    resolved_config_lock_path: Path | None,
    total_units: int,
    skipped_units: int,
    remaining_units: int,
    execution_config: Any,
) -> list[StartupSection]:
    """Build the startup summary sections shown before eval execution begins."""
    selection_rows: list[tuple[str, str]] = [
        ("Cases", str(len(cases))),
        (
            "Conditions",
            _format_csv([getattr(condition, "value", str(condition)) for condition in conditions]),
        ),
        ("Categories", _format_case_filters(cases=cases, attr_name="category")),
        ("Difficulties", _format_case_filters(cases=cases, attr_name="difficulty")),
        ("Snapshot dep", _format_case_filters(cases=cases, attr_name="snapshot_dependency")),
    ]
    execution_rows: list[tuple[str, str]] = [
        ("Executor", str(getattr(execution_config, "executor", "unknown"))),
        ("Jobs", str(int(getattr(args, "jobs", 1)))),
        ("Attempts", f"k={int(getattr(args, 'k', 0))}"),
        ("Keep workdirs", "on" if bool(getattr(args, "keep_workdirs", False)) else "off"),
        ("Progress", f"{skipped_units} complete, {remaining_units} remaining, {total_units} total"),
    ]

    patcher = str(getattr(args, "patcher", "") or "")
    model = str(getattr(args, "openai_model", "") or "").strip() or "-"
    model_rows: list[tuple[str, str]] = [
        ("Patcher", patcher),
        ("Model", model),
        ("Base URL", str(getattr(args, "openai_base_url", "") or "-")),
        (
            "Params",
            _join_nonempty(
                [
                    f"temp={float(getattr(args, 'openai_temperature', 0.0)):.1f}",
                    f"max_tokens={int(getattr(args, 'openai_max_tokens', 0))}",
                    f"response={getattr(args, 'openai_response_mode', '-')!s}",
                    f"timeout={float(getattr(args, 'openai_timeout_sec', 0.0)):.0f}s",
                ]
            ),
        ),
        (
            "Transport",
            _join_nonempty(
                [
                    f"retries={int(getattr(args, 'openai_transport_max_retries', 0))}",
                    "recovery=on"
                    if bool(getattr(args, "openai_transport_recovery", False))
                    else "recovery=off",
                    f"max_propose={float(getattr(args, 'openai_max_propose_sec', 0.0)):.0f}s",
                ]
            ),
        ),
    ]
    native_baseline_mode = str(getattr(args, "native_baseline_mode", "") or "").strip()
    if native_baseline_mode:
        model_rows.insert(1, ("Native Baseline", native_baseline_mode))
    if patcher == "baseline":
        baseline_name = str(getattr(args, "baseline_name", "") or "").strip()
        if baseline_name:
            insert_index = 2 if native_baseline_mode else 1
            model_rows.insert(insert_index, ("Baseline", baseline_name))

    context_label = str(getattr(args, "context_protocol", "monolithic"))
    adaptive_label = (
        _join_nonempty(
            [
                f"gate={getattr(args, 'adaptive_gate_policy', '-')!s}",
                f"helper_calls={int(getattr(args, 'adaptive_max_helper_calls', 0))}",
                "pregate=on"
                if bool(getattr(args, "adaptive_informativeness_pregate", False))
                else "pregate=off",
                f"budget_policy={getattr(args, 'adaptive_budget_policy_mode', 'off')!s}",
            ]
        )
        if has_adaptive_condition
        else "off"
    )
    prompting_rows: list[tuple[str, str]] = [
        (
            "Context",
            _join_nonempty(
                [
                    context_label,
                    f"max_chars={int(getattr(args, 'max_context_chars', 0))}",
                    "files=off" if bool(getattr(args, "no_files_context", False)) else "files=on",
                ]
            ),
        ),
        (
            "Stage-A",
            _join_nonempty(
                [
                    f"max_requests={int(getattr(args, 'context_stage_a_max_requests', 0))}",
                    f"max_chars={int(getattr(args, 'context_stage_a_max_chars', 0))}",
                ]
            ),
        ),
        (
            "Retry",
            _join_nonempty(
                [
                    f"prompt={getattr(args, 'retry_prompt_mode', '-')!s}",
                    f"context={getattr(args, 'retry_context_mode', '-')!s}",
                    f"history={int(getattr(args, 'retry_history_max_chars', 0))}",
                ]
            ),
        ),
        ("Adaptive", adaptive_label),
    ]

    run_rows: list[tuple[str, str]] = [
        ("Run ID", run_id),
        ("Results", str(results_path)),
        ("Config", _format_csv([str(path) for path in config_paths]) if config_paths else "-"),
        ("Resolved lock", str(resolved_config_lock_path) if resolved_config_lock_path else "-"),
        ("Resume", f"{skipped_units} complete, {remaining_units} remaining, {total_units} total"),
    ]

    recording_rows: list[tuple[str, str]] = [
        (
            "Recording",
            _join_nonempty(
                [
                    f"prompt={getattr(args, 'record_prompt', '-')!s}",
                    f"evidence={getattr(args, 'record_evidence', '-')!s}",
                    f"deps={getattr(args, 'record_deps', '-')!s}",
                ]
            ),
        ),
        ("Output", str(getattr(args, "output_format", "-"))),
        (
            "Artifacts",
            _join_nonempty(
                [
                    f"pytest_tail={int(getattr(args, 'pytest_output_tail_lines', 0))}",
                    (f"heartbeat={float(getattr(args, 'progress_heartbeat_sec', 0.0)):.1f}s"),
                    f"verbosity={getattr(args, 'progress_verbosity', 'basic')!s}",
                ]
            ),
        ),
    ]

    summary_sections = [
        StartupSection(title="Run", rows=tuple(run_rows)),
        StartupSection(title="Selection", rows=tuple(selection_rows)),
        StartupSection(title="Execution", rows=tuple(execution_rows)),
        StartupSection(title="Model / Patcher", rows=tuple(model_rows)),
        StartupSection(title="Prompting / Adaptive", rows=tuple(prompting_rows)),
        StartupSection(title="Artifacts", rows=tuple(recording_rows)),
    ]
    displayed_arg_keys = {
        "adaptive_budget_policy_mode",
        "adaptive_gate_policy",
        "adaptive_informativeness_pregate",
        "adaptive_max_helper_calls",
        "baseline_name",
        "conditions",
        "config",
        "context_protocol",
        "context_stage_a_max_chars",
        "context_stage_a_max_requests",
        "executor",
        "jobs",
        "k",
        "keep_workdirs",
        "native_baseline_mode",
        "max_context_chars",
        "no_files_context",
        "openai_base_url",
        "openai_max_propose_sec",
        "openai_max_tokens",
        "openai_model",
        "openai_response_mode",
        "openai_temperature",
        "openai_timeout_sec",
        "openai_transport_max_retries",
        "openai_transport_recovery",
        "output_format",
        "patcher",
        "progress_heartbeat_sec",
        "progress_verbosity",
        "pytest_output_tail_lines",
        "record_deps",
        "record_evidence",
        "record_prompt",
        "retry_context_mode",
        "retry_history_max_chars",
        "retry_prompt_mode",
        "run_id",
    }
    return summary_sections + _build_resolved_config_sections(
        args=args,
        excluded_keys=displayed_arg_keys,
    )
