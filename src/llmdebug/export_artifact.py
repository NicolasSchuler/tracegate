"""Build shareable JSON, HTML, and bundle export artifacts for snapshots and RCA state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, cast

from . import _json as json
from ._formatting import frame_file_label
from ._presentation import format_frame_text
from .rca_state import build_failure_signature, derive_session_id, get_rca_history, get_rca_status
from .redaction_policy import (
    RedactionProfile,
    normalize_redaction_profile,
    resolve_redaction_policy,
)
from .serialize import compile_redactors, redact_jsonlike
from .snapshot_diff import diff_snapshots, format_diff_text, structured_diff_summary

ExportMode = str
JsonDict = dict[str, Any]
JsonList = list[Any]

_EXPORT_FORMAT_VERSION = "1.1"
_EXCEPTION_STRING_KEYS = {"message", "args", "notes", "note"}


def default_export_path(*, snapshot_path: Path, fmt: str) -> Path:
    """Return a stable default export path next to the source snapshot.

    Args:
        snapshot_path: Snapshot file being exported.
        fmt: Export format; ``"html"`` selects ``.html``, anything else ``.json``.

    Returns:
        Sibling path named ``<snapshot stem>.report.html`` or ``.report.json``.
    """
    suffix = ".html" if fmt == "html" else ".json"
    return snapshot_path.with_name(f"{snapshot_path.stem}.report{suffix}")


def _path_basename(value: Any) -> str:
    """Return the last POSIX- or Windows-style path component of *value*."""
    text = str(value or "").strip().rstrip("/\\")
    if not text:
        return ""
    return text.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def _looks_like_filesystem_path(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    candidate = Path(text)
    return candidate.is_absolute() or text.startswith("~") or "/" in text or "\\" in text


def _sanitize_redacted_locator(value: Any, *, fallback: str = "") -> Any:
    """Reduce a path-like locator to its basename; keep non-path refs such as ``#2``."""
    text = str(value or "").strip()
    if not text:
        return fallback or None
    if _looks_like_filesystem_path(text):
        return _path_basename(text) or fallback or None
    return value


def _sanitize_redacted_export_metadata(payload: dict[str, Any]) -> None:
    """Strip local directory paths from the source and comparison metadata in place."""
    source = payload.get("source")
    if isinstance(source, dict):
        source_dict = cast(JsonDict, source)
        snapshot_file = str(source_dict.get("snapshot_file") or "")
        source_dict["out_dir"] = None
        source_dict["snapshot_path"] = _sanitize_redacted_locator(
            source_dict.get("snapshot_path"),
            fallback=snapshot_file,
        )
        source_dict["snapshot_ref"] = _sanitize_redacted_locator(
            source_dict.get("snapshot_ref"),
            fallback=snapshot_file,
        )

    comparison = payload.get("comparison")
    if isinstance(comparison, dict):
        comparison_dict = cast(JsonDict, comparison)
        sanitized_snapshot_path = _sanitize_redacted_locator(comparison_dict.get("snapshot_path"))
        comparison_dict["snapshot_path"] = sanitized_snapshot_path
        comparison_dict["ref"] = _sanitize_redacted_locator(
            comparison_dict.get("ref"),
            fallback=str(sanitized_snapshot_path or ""),
        )


def _utc_now() -> str:
    """Return the current UTC time as second-precision ISO-8601 with a ``Z`` suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _crash_frame(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return the crash frame, falling back to the first frame or an empty dict."""
    frames_raw = snapshot.get("frames")
    frames = cast(list[Any], frames_raw) if isinstance(frames_raw, list) else cast(list[Any], [])
    if not frames:
        return {}
    crash_idx_raw = snapshot.get("crash_frame_index", 0)
    crash_idx = crash_idx_raw if isinstance(crash_idx_raw, int) else 0
    if 0 <= crash_idx < len(frames) and isinstance(frames[crash_idx], dict):
        return cast(dict[str, Any], frames[crash_idx])
    first = frames[0]
    return cast(dict[str, Any], first) if isinstance(first, dict) else {}


def _serialize_frame_sections(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-frame index, crash flag, location, function, and rendered text."""
    out: list[dict[str, Any]] = []
    frames_raw = snapshot.get("frames")
    frames = cast(list[Any], frames_raw) if isinstance(frames_raw, list) else cast(list[Any], [])
    crash_idx_raw = snapshot.get("crash_frame_index", 0)
    crash_idx = crash_idx_raw if isinstance(crash_idx_raw, int) else 0
    for index, raw_frame in enumerate(frames):
        if not isinstance(raw_frame, dict):
            continue
        frame = cast(dict[str, Any], raw_frame)
        out.append(
            {
                "index": index,
                "is_crash": index == crash_idx,
                "location": f"{frame_file_label(frame)}:{frame.get('line', '?')}",
                "function": str(frame.get("function") or "?"),
                "text": format_frame_text(frame, index, index == crash_idx, crash_tag=False),
            }
        )
    return out


def _comparison_payload(
    *,
    compare_ref: str | None,
    compare_path: Path | None,
    compare_snapshot: dict[str, Any] | None,
    snapshot: dict[str, Any],
) -> dict[str, Any] | None:
    """Diff the comparison snapshot against *snapshot*, or return None without one."""
    if compare_snapshot is None or compare_path is None:
        return None
    diff = diff_snapshots(compare_snapshot, snapshot)
    return {
        "ref": compare_ref or str(compare_path),
        "snapshot_path": str(compare_path),
        "structured_summary": structured_diff_summary(diff),
        "diff": asdict(diff),
        "text": format_diff_text(diff),
    }


def _find_retry_summaries(value: Any) -> list[dict[str, Any]]:
    """Collect ``retry_summary``/``retry_summaries`` dicts at any depth, deduplicated."""
    found: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        """Append retry summaries found in *node* and its descendants to ``found``."""
        if isinstance(node, dict):
            node_dict = cast(JsonDict, node)
            explicit_summary = node_dict.get("retry_summary")
            if isinstance(explicit_summary, dict):
                found.append(dict(cast(JsonDict, explicit_summary)))
            explicit_summaries = node_dict.get("retry_summaries")
            if isinstance(explicit_summaries, list):
                found.extend(
                    dict(cast(JsonDict, item))
                    for item in cast(JsonList, explicit_summaries)
                    if isinstance(item, dict)
                )
            for item in node_dict.values():
                _walk(item)
            return
        if isinstance(node, list):
            for item in cast(JsonList, node):
                _walk(item)

    _walk(value)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for summary in found:
        marker = json.dumps(summary, sort_keys=True, ensure_ascii=False, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(summary)
    return deduped


def build_export_payload(
    *,
    snapshot: dict[str, Any],
    raw_payload: dict[str, Any],
    snapshot_path: Path,
    out_dir: str,
    snapshot_ref: str | None = None,
    compare_ref: str | None = None,
    compare_path: Path | None = None,
    compare_snapshot: dict[str, Any] | None = None,
    export_mode: ExportMode = "full",
    export_redaction_profile: RedactionProfile | None = None,
) -> dict[str, Any]:
    """Build a shareable export payload for JSON/HTML reports.

    Args:
        snapshot: Snapshot normalized to the legacy flat view.
        raw_payload: Snapshot as stored on disk; exported only when it differs from
            *snapshot* (for example a DebugSession envelope) and searched for retry
            summaries.
        snapshot_path: Path of the exported snapshot file.
        out_dir: Snapshot directory; scopes the session ID and the RCA lookup.
        snapshot_ref: Reference the caller used to select the snapshot; recorded as
            ``"latest"`` when omitted.
        compare_ref: Reference used to select the comparison snapshot.
        compare_path: Path of the comparison snapshot; required for a comparison.
        compare_snapshot: Earlier snapshot to diff against; required for a
            comparison.
        export_mode: Mode label recorded in the payload; this function does not
            redact (see :func:`apply_export_redaction`).
        export_redaction_profile: Redaction profile label recorded in the payload.

    Returns:
        A ``llmdebug.export`` payload with source metadata, session ID, failure
        signature, exception, crash frame, rendered frames, retry summaries, the
        optional comparison, and the session's RCA status and last 20 history records.
    """
    session_id = derive_session_id(snapshot, out_dir=out_dir, fallback="export")
    failure_signature = build_failure_signature(snapshot)
    crash_frame = _crash_frame(snapshot)
    raw_payload_to_export: dict[str, Any] | None = None
    if raw_payload != snapshot:
        raw_payload_to_export = raw_payload
    retry_summaries = _find_retry_summaries(raw_payload_to_export or raw_payload)
    return {
        "kind": "llmdebug.export",
        "format_version": _EXPORT_FORMAT_VERSION,
        "exported_at_utc": _utc_now(),
        "export_mode": export_mode,
        "export_redaction_profile": export_redaction_profile,
        "source": {
            "out_dir": str(Path(out_dir).resolve()),
            "snapshot_ref": str(snapshot_ref or "latest"),
            "snapshot_path": str(snapshot_path),
            "snapshot_file": snapshot_path.name,
        },
        "session_id": session_id,
        "failure_signature": failure_signature,
        "exception": cast(dict[str, Any], snapshot.get("exception") or {}),
        "crash_frame": crash_frame,
        "snapshot": snapshot,
        "raw_payload": raw_payload_to_export,
        "frames": _serialize_frame_sections(snapshot),
        "retry_summaries": retry_summaries,
        "comparison": _comparison_payload(
            compare_ref=compare_ref,
            compare_path=compare_path,
            compare_snapshot=compare_snapshot,
            snapshot=snapshot,
        ),
        "rca": {
            "status": get_rca_status(out_dir=out_dir, session_id=session_id),
            "history": get_rca_history(out_dir=out_dir, session_id=session_id, limit=20),
        },
    }


def _restore_path_value(redacted: Any, original: Any, *, key_name: str) -> None:
    """Copy every *key_name* value from *original* back into *redacted*, recursively."""
    if isinstance(redacted, dict) and isinstance(original, dict):
        redacted_dict = cast(JsonDict, redacted)
        original_dict = cast(JsonDict, original)
        if key_name in original_dict:
            redacted_dict[key_name] = deepcopy(original_dict[key_name])
        for child_key, child_value in redacted_dict.items():
            original_child = original_dict.get(child_key)
            _restore_path_value(child_value, original_child, key_name=key_name)
        return
    if isinstance(redacted, list) and isinstance(original, list):
        redacted_list = cast(JsonList, redacted)
        original_list = cast(JsonList, original)
        for redacted_item, original_item in zip(redacted_list, original_list, strict=False):
            _restore_path_value(redacted_item, original_item, key_name=key_name)


def _restore_exception_text_fields(redacted: Any, original: Any) -> None:
    """Restore exception message, args, and notes (and their causes) from *original*."""
    if isinstance(redacted, dict) and isinstance(original, dict):
        redacted_dict = cast(JsonDict, redacted)
        original_dict = cast(JsonDict, original)
        if "exception" in redacted_dict and "exception" in original_dict:
            redacted_exception = redacted_dict["exception"]
            original_exception = original_dict["exception"]
            if isinstance(redacted_exception, dict) and isinstance(original_exception, dict):
                redacted_exception_dict = cast(JsonDict, redacted_exception)
                original_exception_dict = cast(JsonDict, original_exception)
                for key in _EXCEPTION_STRING_KEYS:
                    if key in original_exception_dict:
                        redacted_exception_dict[key] = deepcopy(original_exception_dict[key])
        for child_key, child_value in redacted_dict.items():
            if child_key in {"cause", "context"}:
                original_child = original_dict.get(child_key)
                if isinstance(child_value, dict) and isinstance(original_child, dict):
                    child_dict = cast(JsonDict, child_value)
                    original_child_dict = cast(JsonDict, original_child)
                    for key in _EXCEPTION_STRING_KEYS:
                        if key in original_child_dict:
                            child_dict[key] = deepcopy(original_child_dict[key])
            _restore_exception_text_fields(child_value, original_dict.get(child_key))
        return
    if isinstance(redacted, list) and isinstance(original, list):
        redacted_list = cast(JsonList, redacted)
        original_list = cast(JsonList, original)
        for redacted_item, original_item in zip(redacted_list, original_list, strict=False):
            _restore_exception_text_fields(redacted_item, original_item)


def apply_export_redaction(
    payload: dict[str, Any],
    *,
    export_mode: ExportMode,
    redaction_profile: str | None,
) -> dict[str, Any]:
    """Return a redacted export payload without mutating the source payload.

    Outside ``redacted`` mode the payload is deep-copied and labelled ``full``.
    In ``redacted`` mode the profile's patterns are applied, traceback and
    exception text are restored when the profile does not redact them, and local
    paths are reduced to file names in the source and comparison metadata.

    Args:
        payload: Export payload from :func:`build_export_payload`.
        export_mode: ``"redacted"`` (any case) to redact; any other value exports
            full fidelity.
        redaction_profile: Redaction profile name; ``ci`` or ``prod`` is required
            in redacted mode.

    Returns:
        A new payload with ``export_mode`` and ``export_redaction_profile`` set.

    Raises:
        ValueError: If the profile name is invalid, or if redacted mode is requested
            with a profile that defines no redaction patterns.
    """
    normalized_mode = str(export_mode or "full").strip().lower()
    normalized_profile = normalize_redaction_profile(redaction_profile)
    if normalized_mode != "redacted":
        out = deepcopy(payload)
        out["export_mode"] = "full"
        out["export_redaction_profile"] = normalized_profile
        return out

    policy = resolve_redaction_policy(redaction_profile=normalized_profile)
    redactors = compile_redactors(policy.patterns)
    if not redactors:
        # CAP-06: an artifact labelled "redacted" must have had redaction applied.
        raise ValueError(
            "redacted export needs a redaction profile with patterns (ci or prod), "
            f"got {normalized_profile!r}"
        )
    redacted = cast(
        dict[str, Any],
        redact_jsonlike(payload, redactors, redact_keys=policy.redact_keys),
    )
    if not policy.redact_traceback:
        _restore_path_value(redacted, payload, key_name="traceback")
    if not policy.redact_exception_strings:
        _restore_exception_text_fields(redacted, payload)
    _sanitize_redacted_export_metadata(redacted)
    redacted["export_mode"] = "redacted"
    redacted["export_redaction_profile"] = normalized_profile
    return redacted


def build_export_bundle_manifest(
    *,
    payload: dict[str, Any],
    export_format: str,
    artifact_filename: str,
    compare_payload: dict[str, Any] | None = None,
    compare_artifact_filename: str | None = None,
) -> dict[str, Any]:
    """Build a self-contained manifest for bundle exports.

    Args:
        payload: Primary export payload.
        export_format: Format of the bundled artifacts, such as ``"json"`` or
            ``"html"``.
        artifact_filename: File name of the primary artifact inside the bundle.
        compare_payload: Export payload of the comparison snapshot, if bundled.
        compare_artifact_filename: File name of the comparison artifact, if bundled.

    Returns:
        A ``llmdebug.export_bundle`` manifest with export metadata, the primary (and
        optional comparison) snapshot locators, bundle artifact names, the retry
        summary count, and RCA presence counts.
    """
    source = cast(dict[str, Any], payload.get("source") or {})
    retry_summaries = payload.get("retry_summaries")
    retry_summary_count = (
        len(cast(JsonList, retry_summaries)) if isinstance(retry_summaries, list) else 0
    )
    rca = cast(dict[str, Any], payload.get("rca") or {})
    rca_history = cast(JsonList, rca.get("history")) if isinstance(rca.get("history"), list) else []
    manifest: dict[str, Any] = {
        "kind": "llmdebug.export_bundle",
        "format_version": _EXPORT_FORMAT_VERSION,
        "exported_at_utc": str(payload.get("exported_at_utc") or _utc_now()),
        "export_format": export_format,
        "export_mode": payload.get("export_mode"),
        "export_redaction_profile": payload.get("export_redaction_profile"),
        "session_id": payload.get("session_id"),
        "failure_signature": payload.get("failure_signature"),
        "primary_snapshot": {
            "ref": source.get("snapshot_ref"),
            "path": source.get("snapshot_path"),
            "file": source.get("snapshot_file"),
        },
        "artifacts": {
            "primary": artifact_filename,
            "compare": compare_artifact_filename,
            "manifest": "manifest.json",
            "index": "index.html",
        },
        "retry_summary_count": retry_summary_count,
        "rca": {
            "has_status": isinstance(rca.get("status"), dict),
            "history_count": len(rca_history),
        },
    }
    if compare_payload is not None:
        compare_source = cast(dict[str, Any], compare_payload.get("source") or {})
        manifest["compare_snapshot"] = {
            "ref": cast(dict[str, Any], payload.get("comparison") or {}).get("ref"),
            "path": compare_source.get("snapshot_path"),
            "file": compare_source.get("snapshot_file"),
        }
    return manifest


def _html_json_blob(payload: dict[str, Any]) -> str:
    return escape(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _render_html_card(title: str, body: str) -> str:
    return f'<section class="card"><h2>{escape(title)}</h2>{body}</section>'


def _render_key_value_list(items: list[tuple[str, str]]) -> str:
    rows = "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>" for label, value in items if value
    )
    return f'<dl class="meta">{rows}</dl>'


def _render_retry_summary_timeline(retry_summaries: list[dict[str, Any]]) -> str:
    items = [
        (
            '<article class="timeline-item">'
            f"<h3>Attempt {escape(str(summary.get('attempt') or '?'))}</h3>"
            f"<p><strong>Hypothesis:</strong> {escape(str(summary.get('hypothesis_tested') or ''))}</p>"
            f"<p><strong>Evidence:</strong> {escape(str(summary.get('evidence_used') or ''))}</p>"
            f"<p><strong>Outcome:</strong> {escape(str(summary.get('outcome') or ''))}</p>"
            f"<p><strong>Constraint:</strong> {escape(str(summary.get('new_constraint') or ''))}</p>"
            "</article>"
        )
        for summary in retry_summaries
    ]
    return f'<div class="timeline">{"".join(items)}</div>'


def _render_summary_card(
    payload: dict[str, Any],
    *,
    source: dict[str, Any],
    exc: dict[str, Any],
    crash_frame: dict[str, Any],
    pytest_block: dict[str, Any],
) -> str:
    return _render_html_card(
        "Summary",
        _render_key_value_list(
            [
                ("Snapshot", str(source.get("snapshot_file") or "")),
                ("Exported", str(payload.get("exported_at_utc") or "")),
                ("Session ID", str(payload.get("session_id") or "")),
                ("Failure Signature", str(payload.get("failure_signature") or "")),
                ("Mode", str(payload.get("export_mode") or "")),
                (
                    "Redaction Profile",
                    str(payload.get("export_redaction_profile") or "full-fidelity"),
                ),
                ("Exception", f"{exc.get('type', 'UnknownError')}: {exc.get('message', '')}"),
                (
                    "Crash Frame",
                    f"{frame_file_label(crash_frame)}:{crash_frame.get('line', '?')} "
                    f"in {crash_frame.get('function', '?')}()",
                ),
                ("Pytest Node", str(pytest_block.get("nodeid") or "")),
            ]
        ),
    )


def _render_repro_card(pytest_block: dict[str, Any]) -> str | None:
    repro = pytest_block.get("repro")
    if not isinstance(repro, list) or not repro:
        return None
    repro_parts = cast(list[Any], repro)
    return _render_html_card(
        "Repro Command",
        f"<pre>{escape(' '.join(str(part) for part in repro_parts))}</pre>",
    )


def _render_frames_card(frame_sections: list[dict[str, Any]]) -> str | None:
    if not frame_sections:
        return None
    frame_html: list[str] = []
    for frame in frame_sections:
        summary = (
            f"Frame #{int(frame.get('index', 0))} - "
            f"{frame.get('location', '?')} in {frame.get('function', '?')}()"
        )
        if bool(frame.get("is_crash")):
            summary += " [CRASH]"
        frame_html.append(
            "<details open>"
            f"<summary>{escape(summary)}</summary>"
            f"<pre>{escape(str(frame.get('text') or ''))}</pre>"
            "</details>"
        )
    return _render_html_card("Frames", "".join(frame_html))


def _render_json_card(title: str, value: Any) -> str:
    return _render_html_card(
        title,
        f"<pre>{escape(json.dumps(value, indent=2, ensure_ascii=False, default=str))}</pre>",
    )


def _render_rca_cards(rca: dict[str, Any]) -> list[str]:
    cards: list[str] = []
    status = rca.get("status")
    if isinstance(status, dict):
        cards.append(_render_json_card("RCA Status", status))
    history = rca.get("history")
    if isinstance(history, list) and history:
        cards.append(_render_json_card("RCA History", history))
    return cards


def _render_comparison_card(comparison: dict[str, Any]) -> str | None:
    if not comparison:
        return None
    return _render_html_card(
        "Comparison",
        f"<pre>{escape(str(comparison.get('text') or ''))}</pre>",
    )


def _build_export_html_cards(
    payload: dict[str, Any],
    *,
    source: dict[str, Any],
    exc: dict[str, Any],
    crash_frame: dict[str, Any],
    frame_sections: list[dict[str, Any]],
    retry_summaries: list[dict[str, Any]],
    rca: dict[str, Any],
    comparison: dict[str, Any],
    pytest_block: dict[str, Any],
) -> list[str]:
    cards = [
        _render_summary_card(
            payload,
            source=source,
            exc=exc,
            crash_frame=crash_frame,
            pytest_block=pytest_block,
        )
    ]
    repro_card = _render_repro_card(pytest_block)
    if repro_card is not None:
        cards.append(repro_card)
    if retry_summaries:
        cards.append(
            _render_html_card(
                "Retry Timeline",
                _render_retry_summary_timeline(retry_summaries),
            )
        )
    frames_card = _render_frames_card(frame_sections)
    if frames_card is not None:
        cards.append(frames_card)
    cards.extend(_render_rca_cards(rca))
    comparison_card = _render_comparison_card(comparison)
    if comparison_card is not None:
        cards.append(comparison_card)
    cards.append(
        _render_html_card(
            "Embedded JSON",
            f'<pre class="json">{_html_json_blob(payload)}</pre>',
        )
    )
    return cards


def _render_export_html_document(
    *,
    snapshot_file: str,
    exported_at_utc: str,
    cards: list[str],
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>llmdebug export - {escape(snapshot_file or "snapshot")}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe6;
      --panel: #fffdf8;
      --ink: #1f1a17;
      --muted: #6e6257;
      --line: #d8cdbd;
      --accent: #8f3b2e;
      --accent-soft: #f2d9d1;
      --mono: "SFMono-Regular", "Menlo", "Consolas", monospace;
      --sans: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, #f6e7cf 0%, transparent 28%),
        linear-gradient(180deg, #f3ede2 0%, #efe4d6 100%);
      color: var(--ink);
      font-family: var(--sans);
      line-height: 1.5;
    }}
    header {{
      padding: 2.5rem 1.5rem 1rem;
      border-bottom: 1px solid var(--line);
    }}
    header h1 {{
      margin: 0 0 0.4rem;
      font-size: clamp(2rem, 5vw, 3.6rem);
      line-height: 1;
    }}
    header p {{
      margin: 0;
      max-width: 60rem;
      color: var(--muted);
    }}
    main {{
      max-width: 72rem;
      margin: 0 auto;
      padding: 1.25rem;
      display: grid;
      gap: 1rem;
    }}
    .card {{
      background: color-mix(in srgb, var(--panel) 96%, white 4%);
      border: 1px solid var(--line);
      border-radius: 1rem;
      padding: 1rem 1.1rem;
      box-shadow: 0 16px 32px rgba(60, 41, 26, 0.06);
    }}
    .card h2 {{
      margin: 0 0 0.8rem;
      font-size: 1.1rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent);
    }}
    .meta {{
      display: grid;
      gap: 0.35rem 1rem;
      grid-template-columns: minmax(10rem, 14rem) 1fr;
      margin: 0;
    }}
    .meta dt {{
      font-weight: 600;
    }}
    .meta dd {{
      margin: 0;
      color: var(--muted);
    }}
    pre {{
      margin: 0;
      padding: 0.9rem;
      overflow: auto;
      border-radius: 0.9rem;
      background: #201d1a;
      color: #f7efe6;
      font-family: var(--mono);
      font-size: 0.92rem;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    details summary {{
      cursor: pointer;
      font-weight: 600;
      margin-bottom: 0.7rem;
    }}
    .json {{
      max-height: 30rem;
    }}
    .timeline {{
      display: grid;
      gap: 0.8rem;
    }}
    .timeline-item {{
      padding: 0.9rem;
      border-radius: 0.85rem;
      background: var(--accent-soft);
      border: 1px solid color-mix(in srgb, var(--accent) 24%, white 76%);
    }}
    .timeline-item h3 {{
      margin: 0 0 0.45rem;
      font-size: 1rem;
    }}
    .timeline-item p {{
      margin: 0.2rem 0;
    }}
    footer {{
      padding: 1rem 1.5rem 2rem;
      color: var(--muted);
      text-align: center;
    }}
    @media (max-width: 700px) {{
      .meta {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>llmdebug export</h1>
    <p>Self-contained debugging artifact for sharing failures, RCA state, and comparison context.</p>
  </header>
  <main>
    {"".join(cards)}
  </main>
  <footer>Generated by llmdebug export at {escape(exported_at_utc)}</footer>
</body>
</html>"""


def render_export_html(payload: dict[str, Any]) -> str:
    """Render a single-file HTML export artifact.

    Args:
        payload: Export payload, optionally redacted.

    Returns:
        A standalone HTML document with summary, repro, retry timeline, frames, RCA,
        comparison, and raw-JSON cards; all payload text is HTML-escaped.
    """
    source = cast(dict[str, Any], payload.get("source") or {})
    exc = cast(dict[str, Any], payload.get("exception") or {})
    crash_frame = cast(dict[str, Any], payload.get("crash_frame") or {})
    frames = payload.get("frames")
    frame_sections = cast(list[dict[str, Any]], frames) if isinstance(frames, list) else []
    retry_summaries_raw = payload.get("retry_summaries")
    retry_summaries = (
        cast(list[dict[str, Any]], retry_summaries_raw)
        if isinstance(retry_summaries_raw, list)
        else []
    )
    rca = cast(dict[str, Any], payload.get("rca") or {})
    comparison = cast(dict[str, Any], payload.get("comparison") or {})
    pytest_block = cast(
        dict[str, Any], cast(dict[str, Any], payload.get("snapshot") or {}).get("pytest") or {}
    )
    cards = _build_export_html_cards(
        payload,
        source=source,
        exc=exc,
        crash_frame=crash_frame,
        frame_sections=frame_sections,
        retry_summaries=retry_summaries,
        rca=rca,
        comparison=comparison,
        pytest_block=pytest_block,
    )
    return _render_export_html_document(
        snapshot_file=str(source.get("snapshot_file") or ""),
        exported_at_utc=str(payload.get("exported_at_utc") or ""),
        cards=cards,
    )


def render_bundle_index_html(manifest: dict[str, Any]) -> str:
    """Render a tiny index page for bundle exports.

    Args:
        manifest: Bundle manifest from :func:`build_export_bundle_manifest`.

    Returns:
        A standalone HTML page with the bundle metadata and links to the primary
        artifact, the optional comparison artifact, and ``manifest.json``.
    """
    artifacts = cast(dict[str, Any], manifest.get("artifacts") or {})
    compare_artifact = str(artifacts.get("compare") or "")
    links = [
        f'<li><a href="{escape(str(artifacts.get("primary") or ""))}">Primary artifact</a></li>',
        '<li><a href="manifest.json">Manifest JSON</a></li>',
    ]
    if compare_artifact:
        links.insert(
            1,
            f'<li><a href="{escape(compare_artifact)}">Compare artifact</a></li>',
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>llmdebug export bundle</title>
  <style>
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background: linear-gradient(180deg, #f3ede2 0%, #efe4d6 100%);
      color: #1f1a17;
    }}
    main {{
      max-width: 48rem;
      margin: 0 auto;
      padding: 2rem 1.25rem 3rem;
    }}
    .card {{
      background: rgba(255, 253, 248, 0.95);
      border: 1px solid #d8cdbd;
      border-radius: 1rem;
      padding: 1.1rem 1.2rem;
    }}
    h1 {{
      margin-top: 0;
      font-size: clamp(2rem, 5vw, 3rem);
    }}
    ul {{
      margin: 1rem 0 0;
      padding-left: 1.25rem;
    }}
    a {{
      color: #8f3b2e;
    }}
    dl {{
      display: grid;
      grid-template-columns: 12rem 1fr;
      gap: 0.35rem 1rem;
      margin: 1rem 0 0;
    }}
    dd {{
      margin: 0;
      color: #6e6257;
    }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>llmdebug export bundle</h1>
      <p>Bundle manifest for the exported snapshot artifact set.</p>
      <dl>
        <dt>Failure signature</dt>
        <dd>{escape(str(manifest.get("failure_signature") or ""))}</dd>
        <dt>Session ID</dt>
        <dd>{escape(str(manifest.get("session_id") or ""))}</dd>
        <dt>Mode</dt>
        <dd>{escape(str(manifest.get("export_mode") or ""))}</dd>
        <dt>Redaction profile</dt>
        <dd>{escape(str(manifest.get("export_redaction_profile") or "full-fidelity"))}</dd>
      </dl>
      <ul>
        {"".join(links)}
      </ul>
    </section>
  </main>
</body>
</html>"""
