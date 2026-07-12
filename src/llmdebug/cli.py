"""llmdebug CLI — View and manage debug snapshots.

The CLI renders in two modes: Rich (panels/tables when rich is installed and
stdout is a TTY) and plain text (click.echo fallback).  ``_get_console()``
returns a Rich Console or None; callers branch on this to pick the rendering
path.
"""

from __future__ import annotations

import shlex

try:
    import click
except ImportError:
    import sys

    print(
        "llmdebug CLI requires extra dependencies.\nInstall with: pip install 'llmdebug[cli]'",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from . import _json as json
from ._cli_display import (
    _emit_json_payload,  # pyright: ignore[reportPrivateUsage]
    _get_console,  # pyright: ignore[reportPrivateUsage]
    _print_context_if_requested,  # pyright: ignore[reportPrivateUsage]
    _print_diff_rich,  # pyright: ignore[reportPrivateUsage]
    _print_exception_panel,  # pyright: ignore[reportPrivateUsage]
    _print_filtered_frames,  # pyright: ignore[reportPrivateUsage]
    _print_frame,  # pyright: ignore[reportPrivateUsage]
    _print_frame_rich,  # pyright: ignore[reportPrivateUsage]
    _print_show_footer,  # pyright: ignore[reportPrivateUsage]
    _resolve_show_detail,  # pyright: ignore[reportPrivateUsage]
    _stdout_is_tty,  # pyright: ignore[reportPrivateUsage]
    _truncate_tail,  # pyright: ignore[reportPrivateUsage]
    _truncate_text,  # pyright: ignore[reportPrivateUsage]
)
from ._debug_session import coerce_crash_frame_index
from ._formatting import (
    format_enhanced_git_context_text,
)
from ._snapshot_loader import (
    find_all_snapshots as _find_all_snapshots_impl,
)
from ._snapshot_loader import (
    load_snapshot as _load_snapshot_impl,
)
from ._snapshot_loader import (
    load_snapshot_from_file as _load_snapshot_from_file_impl,
)
from ._snapshot_loader import (
    load_snapshot_from_file_raw as _load_snapshot_from_file_raw_impl,
)
from ._snapshot_loader import (
    load_snapshot_raw as _load_snapshot_raw_impl,
)
from ._snapshot_loader import (
    resolve_snapshot_ref_path as _resolve_snapshot_ref_path_impl,
)
from .export_artifact import (
    apply_export_redaction,
    build_export_bundle_manifest,
    build_export_payload,
    default_export_path,
    render_bundle_index_html,
    render_export_html,
)

# Rich Table is used directly by the list command; other Rich rendering
# lives in _cli_display.  The conditional import keeps Rich optional.
if TYPE_CHECKING:
    from rich.table import Table

try:
    from rich.table import Table

except ImportError:
    pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# pyright strict mode narrows isinstance(x, dict) to dict[Unknown, Unknown].
# We use cast(dict[str, Any], x) after isinstance checks to restore the type.


@dataclass(frozen=True)
class _LoadedExportState:
    requested_snapshot_ref: str
    snapshot_ref: str | None
    snapshot_path: Path
    raw_payload: dict[str, Any]
    snapshot: dict[str, Any]
    compare_ref: str | None
    compare_path: Path | None
    compare_snapshot: dict[str, Any] | None


@dataclass(frozen=True)
class _ExportPayloadTarget:
    snapshot: dict[str, Any]
    raw_payload: dict[str, Any]
    snapshot_path: Path
    snapshot_ref: str | None
    compare_ref: str | None
    compare_path: Path | None
    compare_snapshot: dict[str, Any] | None


def _quickstart_hint(out_dir: str | Path) -> str:
    """Return a copy-paste-safe quickstart command for a snapshot store."""
    return shlex.join(("llmdebug", "quickstart", "--dir", str(out_dir)))


def _echo_snapshot_error(error: Exception, *, out_dir: str | Path) -> None:
    """Print a snapshot-loading error with an actionable empty-store hint."""
    click.echo(str(error), err=True)
    if isinstance(error, FileNotFoundError) and str(error).startswith("No snapshots found"):
        click.echo(f"Try: {_quickstart_hint(out_dir)}", err=True)


def _load_snapshot(path: str | None, out_dir: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction] — used in tests
    """Load a snapshot from an explicit path or fall back to latest."""
    try:
        return _load_snapshot_impl(path, out_dir, _trust_path=True)
    except FileNotFoundError as e:
        _echo_snapshot_error(e, out_dir=out_dir)
        raise SystemExit(1) from e


def _load_snapshot_raw(path: str | None, out_dir: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction] — used in tests
    """Load a raw snapshot payload (DebugSession envelope when present)."""
    try:
        return _load_snapshot_raw_impl(path, out_dir, _trust_path=True)
    except FileNotFoundError as e:
        _echo_snapshot_error(e, out_dir=out_dir)
        raise SystemExit(1) from e


def _load_snapshot_ref(ref: str | None, out_dir: str) -> dict[str, Any]:
    """Load a snapshot from a shared CLI reference syntax."""
    try:
        path = _resolve_snapshot_ref_path_impl(ref, out_dir, _trust_path=True)
        return _load_snapshot_from_file_impl(path)
    except (FileNotFoundError, ImportError, ValueError) as e:
        _echo_snapshot_error(e, out_dir=out_dir)
        raise SystemExit(1) from e


def _load_snapshot_ref_raw(ref: str | None, out_dir: str) -> dict[str, Any]:
    """Load a raw snapshot payload from a shared CLI reference syntax."""
    try:
        path = _resolve_snapshot_ref_path_impl(ref, out_dir, _trust_path=True)
        return _load_snapshot_from_file_raw_impl(path)
    except (FileNotFoundError, ImportError, ValueError) as e:
        _echo_snapshot_error(e, out_dir=out_dir)
        raise SystemExit(1) from e


def _latest_snapshot_path_for_export(out_dir: str) -> Path | None:
    """Resolve the latest export source to the immutable snapshot file."""
    entries = _find_all_snapshots_impl(out_dir, limit=1)
    if not entries:
        return None
    entry = entries[0]
    path_value = entry.get("path")
    if not path_value:
        return None
    return Path(str(path_value))


def _resolve_export_snapshot_path(snapshot_ref: str | None, out_dir: str) -> Path:
    """Resolve export refs to stable snapshot files instead of latest pointers."""
    path = _resolve_snapshot_ref_path_impl(snapshot_ref, out_dir, _trust_path=True)
    path_is_latest_pointer = False
    if path.name in {"latest.json", "latest.toon"}:
        try:
            path_is_latest_pointer = path.parent.resolve() == Path(out_dir).resolve()
        except OSError:
            path_is_latest_pointer = False
    if snapshot_ref in {None, "latest"} or path_is_latest_pointer:
        latest_snapshot_path = _latest_snapshot_path_for_export(out_dir)
        if latest_snapshot_path is not None:
            return latest_snapshot_path
    try:
        if path.is_symlink():
            return path.resolve()
    except OSError:
        return path
    return path


def _load_export_state(
    snapshot_ref: str | None,
    compare_ref: str | None,
    out_dir: str,
) -> _LoadedExportState:
    requested_snapshot_ref = snapshot_ref or "latest"
    snapshot_path = _resolve_export_snapshot_path(snapshot_ref, out_dir)
    raw_payload = _load_snapshot_from_file_raw_impl(snapshot_path)
    snapshot = _load_snapshot_from_file_impl(snapshot_path)
    compare_path: Path | None = None
    compare_snapshot: dict[str, Any] | None = None
    if compare_ref is not None:
        compare_path = _resolve_export_snapshot_path(compare_ref, out_dir)
        compare_snapshot = _load_snapshot_from_file_impl(compare_path)
    return _LoadedExportState(
        requested_snapshot_ref=requested_snapshot_ref,
        snapshot_ref=snapshot_ref,
        snapshot_path=snapshot_path,
        raw_payload=raw_payload,
        snapshot=snapshot,
        compare_ref=compare_ref,
        compare_path=compare_path,
        compare_snapshot=compare_snapshot,
    )


def _resolve_export_defaults(
    export_mode: str | None,
    redaction_profile: str | None,
    bundle_dir: Path | None,
) -> tuple[str, str | None]:
    effective_mode = export_mode or ("redacted" if bundle_dir is not None else "full")
    effective_redaction_profile = redaction_profile
    if effective_mode == "redacted" and effective_redaction_profile is None:
        effective_redaction_profile = "ci"
    return effective_mode, effective_redaction_profile


def _build_export_report_payload(
    target: _ExportPayloadTarget,
    *,
    out_dir: str,
    export_mode: str,
    redaction_profile: str | None,
) -> dict[str, Any]:
    export_payload = build_export_payload(
        snapshot=target.snapshot,
        raw_payload=target.raw_payload,
        snapshot_path=target.snapshot_path,
        out_dir=out_dir,
        snapshot_ref=target.snapshot_ref,
        compare_ref=target.compare_ref,
        compare_path=target.compare_path,
        compare_snapshot=target.compare_snapshot,
        export_mode=export_mode,
        export_redaction_profile=cast(Any, redaction_profile),
    )
    return apply_export_redaction(
        export_payload,
        export_mode=export_mode,
        redaction_profile=redaction_profile,
    )


def _prepare_export_payloads(
    state: _LoadedExportState,
    *,
    out_dir: str,
    export_mode: str,
    redaction_profile: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    export_payload = _build_export_report_payload(
        _ExportPayloadTarget(
            snapshot=state.snapshot,
            raw_payload=state.raw_payload,
            snapshot_path=state.snapshot_path,
            snapshot_ref=state.snapshot_ref,
            compare_ref=state.compare_ref,
            compare_path=state.compare_path,
            compare_snapshot=state.compare_snapshot,
        ),
        out_dir=out_dir,
        export_mode=export_mode,
        redaction_profile=redaction_profile,
    )
    if state.compare_path is None or state.compare_snapshot is None or state.compare_ref is None:
        return export_payload, None
    compare_export_payload = _build_export_report_payload(
        _ExportPayloadTarget(
            snapshot=state.compare_snapshot,
            raw_payload=_load_snapshot_from_file_raw_impl(state.compare_path),
            snapshot_path=state.compare_path,
            snapshot_ref=state.compare_ref,
            compare_ref=state.requested_snapshot_ref,
            compare_path=state.snapshot_path,
            compare_snapshot=state.snapshot,
        ),
        out_dir=out_dir,
        export_mode=export_mode,
        redaction_profile=redaction_profile,
    )
    return export_payload, compare_export_payload


def _render_export_text(export_format: str, export_payload: dict[str, Any]) -> str:
    if export_format == "json":
        return json.dumps(export_payload, indent=2, ensure_ascii=False, default=str)
    return render_export_html(export_payload)


def _write_export_bundle(
    bundle_dir: Path,
    *,
    export_format: str,
    export_payload: dict[str, Any],
    compare_export_payload: dict[str, Any] | None,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"report.{export_format}"
    artifact_path = bundle_dir / artifact_name
    artifact_path.write_text(_render_export_text(export_format, export_payload), encoding="utf-8")

    compare_artifact_name: str | None = None
    if compare_export_payload is not None:
        compare_artifact_name = f"compare.{export_format}"
        (bundle_dir / compare_artifact_name).write_text(
            _render_export_text(export_format, compare_export_payload),
            encoding="utf-8",
        )

    manifest = build_export_bundle_manifest(
        payload=export_payload,
        export_format=export_format,
        artifact_filename=artifact_name,
        compare_payload=compare_export_payload,
        compare_artifact_filename=compare_artifact_name,
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (bundle_dir / "index.html").write_text(
        render_bundle_index_html(manifest),
        encoding="utf-8",
    )


def _write_single_export(
    output: Path | None,
    *,
    snapshot_path: Path,
    export_format: str,
    export_payload: dict[str, Any],
) -> Path:
    output_path = output or default_export_path(snapshot_path=snapshot_path, fmt=export_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_export_text(export_format, export_payload), encoding="utf-8")
    return output_path


def _load_snapshot_from_file(path: Path) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction] — used in tests
    """Load and auto-expand a single snapshot file."""
    try:
        return _load_snapshot_from_file_impl(path)
    except (FileNotFoundError, ImportError) as e:
        click.echo(str(e), err=True)
        raise SystemExit(1) from e


def _find_all_snapshots(out_dir: str) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction] — used in tests
    """Scan output dir and return snapshot metadata sorted newest-first."""
    return _find_all_snapshots_impl(out_dir)


def _format_timestamp(ts_str: str | None) -> str:  # pyright: ignore[reportUnusedFunction] — used in tests
    """Format an ISO timestamp into a short human-readable form."""
    if not ts_str:
        return "?"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return ts_str[:19] if len(ts_str) >= 19 else ts_str


# ---------------------------------------------------------------------------
# Click commands
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.option("--dir", "out_dir", default=".llmdebug", help="Snapshot directory")
@click.pass_context
def main(ctx: click.Context, out_dir: str) -> None:
    """llmdebug - View and manage debug snapshots."""
    ctx.ensure_object(dict)
    ctx.obj["out_dir"] = out_dir
    if ctx.invoked_subcommand is None:
        ctx.invoke(show, out_dir=out_dir)


@main.command()
@click.argument("snapshot_ref", required=False)
@click.option("--full", is_flag=True, help="Show all frames (shortcut for --detail full)")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output expanded JSON at the selected detail level.",
)
@click.option(
    "--detail",
    type=click.Choice(["crash", "full", "context"]),
    default=None,
    help="Detail level: crash (minimal), full (all frames), context (everything)",
)
@click.option(
    "--raw-session",
    is_flag=True,
    help="Output raw DebugSession envelope JSON for the selected snapshot.",
)
@click.option("--dir", "out_dir", default=".llmdebug", help="Snapshot directory")
def show(
    snapshot_ref: str | None,
    full: bool,
    as_json: bool,
    detail: str | None,
    raw_session: bool,
    out_dir: str,
) -> None:
    """Display a debug snapshot (default: latest)."""
    if raw_session:
        _emit_json_payload(_load_snapshot_ref_raw(snapshot_ref, out_dir))
        return

    detail = _resolve_show_detail(detail, full)

    snapshot = _load_snapshot_ref(snapshot_ref, out_dir)

    # Apply detail filter
    from .detail_filter import filter_snapshot

    filtered = filter_snapshot(snapshot, detail)  # type: ignore[arg-type]

    if as_json:
        _emit_json_payload(filtered)
        return

    console = _get_console()

    # 1. Exception panel
    _print_exception_panel(filtered, console)

    # 2. Frames
    _print_filtered_frames(filtered, snapshot, detail, console)

    _print_context_if_requested(filtered, detail, console)

    # 3. Footer
    _print_show_footer(snapshot, console)


@main.command()
@click.option(
    "--dir",
    "out_dir",
    type=click.Path(path_type=Path),
    default=Path(".llmdebug"),
    help="Snapshot directory",
)
@click.pass_context
def quickstart(ctx: click.Context, out_dir: Path) -> None:
    """Create and display an offline example failure snapshot."""
    from ._quickstart import create_quickstart_snapshot

    result = create_quickstart_snapshot(out_dir)
    ctx.invoke(
        show,
        snapshot_ref=str(result.snapshot_path),
        full=False,
        as_json=False,
        detail="crash",
        raw_session=False,
        out_dir=str(out_dir),
    )
    click.echo(f"\nSnapshot written to: {result.snapshot_path}")
    click.echo("Next commands:")
    click.echo(f"  {result.show_command}")
    click.echo(f"  {result.list_command}")


@main.command(name="list")
@click.option("--limit", "-n", default=10, type=int, help="Max snapshots to show")
@click.option("--search", default="", help="Case-insensitive metadata search query")
@click.option("--dir", "out_dir", default=".llmdebug", help="Snapshot directory")
def list_cmd(limit: int, search: str, out_dir: str) -> None:
    """List available snapshots."""
    entries = _find_all_snapshots_impl(out_dir, limit=limit, search=search or None)

    if not entries:
        click.echo(f"No snapshots found in {out_dir}")
        click.echo(f"Try: {_quickstart_hint(out_dir)}")
        if _stdout_is_tty():
            click.echo("Run a failing test to create a snapshot, or pass --dir to another folder.")
        return

    console = _get_console()
    name_width = 20
    exception_width = 18
    location_width = 28
    search_suffix = f", search={search!r}" if search.strip() else ""

    if console:
        table = Table(
            title=f"Snapshots ({len(entries)} shown, UTC{search_suffix})", show_lines=False
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Timestamp", style="cyan", no_wrap=True)
        table.add_column("Name")
        table.add_column("Exception", style="red")
        table.add_column("Location", style="dim")

        for i, raw_entry in enumerate(cast(list[Any], entries), 1):
            entry = (
                cast(dict[str, Any], raw_entry)
                if isinstance(raw_entry, dict)
                else cast(dict[str, Any], {})
            )
            try:
                ts = datetime.fromtimestamp(
                    float(entry.get("mtime", 0.0)), tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M")
            except (TypeError, ValueError, OSError, OverflowError):
                ts = "?"
            name = _truncate_text(entry.get("name", "?"), name_width)
            exc_type = _truncate_text(entry.get("exc_type", "?"), exception_width)
            loc = _truncate_tail(
                f"{entry.get('file') or '?'}:{entry.get('line', '?')}", location_width
            )
            table.add_row(
                str(i),
                ts,
                name,
                exc_type,
                loc,
            )
        console.print(table)
    else:
        click.echo(f"Snapshots ({len(entries)} shown, UTC{search_suffix}):")
        for i, raw_entry in enumerate(cast(list[Any], entries), 1):
            entry = (
                cast(dict[str, Any], raw_entry)
                if isinstance(raw_entry, dict)
                else cast(dict[str, Any], {})
            )
            try:
                ts = datetime.fromtimestamp(
                    float(entry.get("mtime", 0.0)), tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M")
            except (TypeError, ValueError, OSError, OverflowError):
                ts = "?"
            name = _truncate_text(entry.get("name", "?"), name_width)
            exc_type = _truncate_text(entry.get("exc_type", "?"), exception_width)
            loc = _truncate_tail(
                f"{entry.get('file') or '?'}:{entry.get('line', '?')}", location_width
            )
            click.echo(f"  {i}. [{ts}] {name}  {exc_type}  {loc}")


@main.command()
@click.argument("snapshot_ref", required=False)
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["html", "json"]),
    default="html",
    show_default=True,
    help="Export artifact format",
)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output path")
@click.option(
    "--mode",
    "export_mode",
    type=click.Choice(["full", "redacted"]),
    default=None,
    help="Export fidelity mode. Defaults to full for single-file output and redacted for bundles.",
)
@click.option(
    "--redaction-profile",
    type=click.Choice(["dev", "ci", "prod"]),
    default=None,
    help="Redaction profile used when --mode=redacted. Defaults to ci.",
)
@click.option(
    "--bundle-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Write a bundle directory with manifest and index pages instead of a single file.",
)
@click.option(
    "--compare",
    "compare_ref",
    default=None,
    help="Optional snapshot reference to compare against in the export artifact",
)
@click.option("--dir", "out_dir", default=".llmdebug", help="Snapshot directory")
def export(
    snapshot_ref: str | None,
    export_format: str,
    output: Path | None,
    export_mode: str | None,
    redaction_profile: str | None,
    bundle_dir: Path | None,
    compare_ref: str | None,
    out_dir: str,
) -> None:
    """Export a snapshot as a single-file report or a redacted bundle.

    Single-file exports default to full detail, while bundle exports default to
    redacted mode with the CI profile unless you override those flags.
    """
    if output is not None and bundle_dir is not None:
        raise click.UsageError("Use either --output or --bundle-dir, not both.")

    try:
        export_state = _load_export_state(snapshot_ref, compare_ref, out_dir)
    except (FileNotFoundError, ImportError, ValueError) as e:
        _echo_snapshot_error(e, out_dir=out_dir)
        raise SystemExit(1) from e

    effective_mode, effective_redaction_profile = _resolve_export_defaults(
        export_mode,
        redaction_profile,
        bundle_dir,
    )
    export_payload, compare_export_payload = _prepare_export_payloads(
        export_state,
        out_dir=out_dir,
        export_mode=effective_mode,
        redaction_profile=effective_redaction_profile,
    )

    if bundle_dir is not None:
        _write_export_bundle(
            bundle_dir,
            export_format=export_format,
            export_payload=export_payload,
            compare_export_payload=compare_export_payload,
        )
        click.echo(f"Exported bundle {bundle_dir}")
        return

    output_path = _write_single_export(
        output,
        snapshot_path=export_state.snapshot_path,
        export_format=export_format,
        export_payload=export_payload,
    )
    click.echo(f"Exported {output_path}")


def _compute_delete_list(
    out_path: Path,
    snapshot_files: list[Path],
    keep: int,
    delete_all: bool,
) -> list[Path]:
    """Compute the list of files to delete without performing any I/O side effects."""
    to_delete: list[Path] = []
    if delete_all:
        to_delete = list(snapshot_files)
        for name in ("latest.json", "latest.toon", ".llmdebug.lock"):
            extra = out_path / name
            if extra.exists():
                to_delete.append(extra)
        to_delete.extend(
            p
            for p in out_path.iterdir()
            if p.is_file() and p.suffix == ".txt" and ".traceback" in p.name
        )
    else:
        for snap in snapshot_files[keep:] if keep < len(snapshot_files) else []:
            to_delete.append(snap)
            tb_file = snap.with_suffix("").with_suffix(".traceback.txt")
            if tb_file.exists():
                to_delete.append(tb_file)
    return to_delete


def _perform_deletions(to_delete: list[Path]) -> tuple[int, int]:
    """Delete files and return (deleted_count, failed_count)."""
    deleted = 0
    failed = 0
    for p in to_delete:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            failed += 1
            click.echo(f"  Warning: could not delete {p.name}: {e}", err=True)
    return deleted, failed


@main.command()
@click.option(
    "--keep",
    "-k",
    default=5,
    type=click.IntRange(min=0),
    help="Keep N most recent snapshots",
)
@click.option("--all", "delete_all", is_flag=True, help="Delete all snapshots")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.option("--dir", "out_dir", default=".llmdebug", help="Snapshot directory")
def clean(keep: int, delete_all: bool, yes: bool, out_dir: str) -> None:
    """Remove old snapshots."""
    out_path = Path(out_dir)
    if not out_path.is_dir():
        click.echo(f"Directory not found: {out_dir}", err=True)
        return

    snapshot_files = sorted(
        (
            p
            for p in out_path.iterdir()
            if p.is_file()
            and p.suffix in {".json", ".toon"}
            and not p.name.startswith("latest")
            and not p.name.endswith(".tmp")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    to_delete = _compute_delete_list(out_path, snapshot_files, keep, delete_all)
    if not to_delete:
        click.echo("Nothing to delete.")
        return

    msg = f"Will delete {len(to_delete)} file(s)"
    if not delete_all:
        msg += f" (keeping {min(keep, len(snapshot_files))} snapshot(s))"
    click.echo(msg)

    if not yes and not click.confirm("Continue?"):
        click.echo("Aborted.")
        return

    deleted, failed = _perform_deletions(to_delete)
    msg = f"Deleted {deleted} file(s)."
    if failed:
        msg += f" Failed to delete {failed} file(s)."
    click.echo(msg)


@main.command()
@click.argument("snapshot_ref", required=False)
@click.option("--index", "-i", type=int, default=None, help="Show specific frame by index")
@click.option("--dir", "out_dir", default=".llmdebug", help="Snapshot directory")
def frames(snapshot_ref: str | None, index: int | None, out_dir: str) -> None:
    """Show stack frames in detail."""
    snapshot = _load_snapshot_ref(snapshot_ref, out_dir)
    frame_list_obj = snapshot.get("frames", [])
    frame_list: list[Any] = (
        cast(list[Any], frame_list_obj) if isinstance(frame_list_obj, list) else cast(list[Any], [])
    )
    crash_idx = coerce_crash_frame_index(snapshot)
    console = _get_console()

    if not frame_list:
        click.echo("No frames in snapshot.")
        return

    if index is not None:
        if index < 0 or index >= len(frame_list):
            click.echo(f"Frame index {index} out of range (0-{len(frame_list) - 1})", err=True)
            raise SystemExit(1)
        frame_raw = frame_list[index]
        if not isinstance(frame_raw, dict):
            click.echo(f"Frame index {index} is malformed in snapshot.", err=True)
            raise SystemExit(1)
        frame_d = cast(dict[str, Any], frame_raw)
        if console:
            _print_frame_rich(frame_d, index, index == crash_idx, console)
        else:
            _print_frame(frame_d, index, index == crash_idx, console)
    else:
        for i, frame_item in enumerate(frame_list):
            if not isinstance(frame_item, dict):
                continue
            frame_d = cast(dict[str, Any], frame_item)
            if console:
                _print_frame_rich(frame_d, i, i == crash_idx, console)
                console.print()
            else:
                _print_frame(frame_d, i, i == crash_idx, console)
                click.echo()


@main.command()
@click.argument("old_ref", required=False)
@click.argument("new_ref", required=False)
@click.option("--json", "as_json", is_flag=True, help="Output diff as JSON")
@click.option("--dir", "out_dir", default=".llmdebug", help="Snapshot directory")
def diff(old_ref: str | None, new_ref: str | None, as_json: bool, out_dir: str) -> None:
    """Compare two snapshots to show what changed.

    With no arguments, compares the two most recent snapshots.
    References: file path, #N (list index), "latest", "previous".

    \b
    Examples:
        llmdebug diff                    # latest vs previous
        llmdebug diff #2 #1             # by list index
        llmdebug diff old.json new.json  # by path
    """
    from dataclasses import asdict

    from .snapshot_diff import diff_snapshots, format_diff_text

    old = _load_snapshot_ref(old_ref or "previous", out_dir)
    new = _load_snapshot_ref(new_ref or "latest", out_dir)

    result = diff_snapshots(old, new)

    if as_json:
        click.echo(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
        return

    console = _get_console()

    if console:
        _print_diff_rich(result, console)
    else:
        click.echo(format_diff_text(result))


@main.command(name="git-context")
@click.argument("snapshot_ref", required=False)
@click.option("--json", "as_json", is_flag=True, help="Output enhanced git context as JSON")
@click.option(
    "--max-commits",
    type=click.IntRange(1, 50),
    default=5,
    show_default=True,
    help="Maximum number of recent commits to include.",
)
@click.option("--dir", "out_dir", default=".llmdebug", help="Snapshot directory")
def git_context(snapshot_ref: str | None, as_json: bool, max_commits: int, out_dir: str) -> None:
    """Show on-demand enhanced git context for a snapshot."""
    from .git_context import get_enhanced_git_context

    snapshot = _load_snapshot_ref(snapshot_ref, out_dir)

    context = get_enhanced_git_context(snapshot, out_dir=out_dir, max_commits=max_commits)
    if context is None:
        click.echo(
            "Enhanced git context unavailable (not in a git repository or git is unavailable)."
        )
        return

    if as_json:
        click.echo(json.dumps(context, indent=2, ensure_ascii=False, default=str))
        return

    click.echo(format_enhanced_git_context_text(context))
