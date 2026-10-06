"""Prompt rendering for the eval harness (stage A, B, retry, compaction).

Extracted from ``evals.run_eval`` to keep the main orchestrator lean.
Re-exported by ``evals.run_eval`` for backward compatibility.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

from . import _json as json
from .adaptive_online_policy import ACTION_INVOKE_HELPER, ACTION_RETRY_HELPER_STRICT_JSON
from .eval_types import (
    Condition,
    ContextCompactionBundle,
    PromptBudgetResult,
    _helper_action_allowed_request_types,
)
from .stage_a_contract import STAGE_A_REQUEST_SCHEMA_PRETTY


def _format_local_value(name: str, value: Any, meta: dict[str, Any] | None) -> str:
    """Format a single local variable with type/shape annotation."""
    meta = meta or {}
    type_str = str(meta.get("type", ""))
    # Strip 'builtins.' prefix for readability.
    type_str = type_str.removeprefix("builtins.")

    # Array-like: show shape and dtype prominently.
    if isinstance(value, dict) and "__array__" in value:
        shape = value.get("shape")
        dtype = value.get("dtype", "")
        arr_type = value.get("__array__", "ndarray")
        shape_str = f"({', '.join(str(s) for s in shape)})" if isinstance(shape, list) else "?"
        return f"    {name} = <{arr_type}, shape={shape_str}, dtype={dtype}>"

    # Scalar or small repr.
    try:
        repr_str = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        repr_str = repr(value)
    # Truncate long values.
    if len(repr_str) > 120:
        repr_str = repr_str[:117] + "..."

    # Add type annotation.
    annotation = ""
    length = meta.get("len")
    shape = meta.get("shape")
    if shape is not None and isinstance(shape, list):
        annotation = f"  ({type_str}, shape=({', '.join(str(s) for s in shape)}))"
    elif length is not None:
        annotation = f"  ({type_str}, len={length})"
    elif type_str:
        annotation = f"  ({type_str})"

    return f"    {name} = {repr_str}{annotation}"


def _project_snapshot_for_prompt(
    snapshot: dict[str, Any],
    *,
    max_frames: int = 3,
) -> dict[str, Any]:
    """Project a snapshot down to the Stage-B prompt contract.

    The legacy Stage-B snapshot section only exposed the exception header plus
    up to ``max_frames`` user-code frames with selected locals. TOON snapshot
    payloads must stay within that same evidence surface instead of serializing
    the full runtime snapshot object.
    """
    exception_obj = snapshot.get("exception")
    exception = exception_obj if isinstance(exception_obj, dict) else {}
    projected_exception = {
        "type": exception.get("type", "Unknown"),
        "message": exception.get("message", ""),
    }

    projected_frames: list[dict[str, Any]] = []
    frames_obj = snapshot.get("frames")
    frames = frames_obj if isinstance(frames_obj, list) else []
    for frame in frames:
        if len(projected_frames) >= max_frames:
            break
        if not isinstance(frame, dict):
            continue
        file_rel = frame.get("file_rel")
        if file_rel is None:
            continue

        projected_frame: dict[str, Any] = {
            "file_rel": file_rel,
            "line": frame.get("line", "?"),
            "function": frame.get("function", "?"),
        }
        code = frame.get("code", "")
        if code:
            projected_frame["code"] = code

        frame_locals_obj = frame.get("locals")
        frame_locals = frame_locals_obj if isinstance(frame_locals_obj, dict) else {}
        filtered_locals = {
            str(var_name): var_value
            for var_name, var_value in frame_locals.items()
            if not str(var_name).startswith("@")
        }
        if filtered_locals:
            projected_frame["locals"] = filtered_locals
            locals_meta_obj = frame.get("locals_meta")
            locals_meta = locals_meta_obj if isinstance(locals_meta_obj, dict) else {}
            filtered_meta = {
                str(var_name): var_meta
                for var_name, var_meta in locals_meta.items()
                if str(var_name) in filtered_locals
            }
            if filtered_meta:
                projected_frame["locals_meta"] = filtered_meta

        projected_frames.append(projected_frame)

    return {
        "exception": projected_exception,
        "frames": projected_frames,
    }


def _format_snapshot_section(snapshot: dict[str, Any], *, max_frames: int = 3) -> str:
    """Format a snapshot into readable text for the LLM prompt."""
    lines: list[str] = []
    lines.append("=== llmdebug Debug Snapshot ===")
    lines.append("")

    # Exception header.
    exc = snapshot.get("exception", {})
    exc_type = exc.get("type", "Unknown")
    exc_msg = exc.get("message", "")
    lines.append(f"Exception: {exc_type} — {exc_msg}")

    lines.append("")

    # Frames — only user code (file_rel is not None), up to max_frames.
    frames = snapshot.get("frames", [])
    shown = 0
    for i, frame in enumerate(frames):
        if shown >= max_frames:
            break
        # Skip framework internals (file_rel is None for site-packages).
        file_rel = frame.get("file_rel")
        if file_rel is None:
            continue

        label = "crash site" if i == 0 else "caller"
        func = frame.get("function", "?")
        line_no = frame.get("line", "?")
        code = frame.get("code", "")
        lines.append(f"Frame {shown} ({label}): {file_rel}:{line_no} in {func}()")
        if code:
            lines.append(f"  Code: {code}")

        # Locals with metadata.
        frame_locals = frame.get("locals", {})
        locals_meta = frame.get("locals_meta", {})
        if frame_locals:
            lines.append("  Locals:")
            for var_name, var_value in frame_locals.items():
                # Skip pytest-internal variables.
                if var_name.startswith("@"):
                    continue
                var_meta = locals_meta.get(var_name)
                lines.append(_format_local_value(var_name, var_value, var_meta))
        lines.append("")
        shown += 1

    lines.append("===")
    return "\n".join(lines)


def _format_snapshot_payload_section(snapshot_prompt_text: str) -> str:
    """Format a compact machine-oriented snapshot payload for the prompt."""
    payload = str(snapshot_prompt_text or "").strip()
    if not payload:
        return ""
    return (
        "=== llmdebug Debug Snapshot ===\n\n"
        "Structured payload (compact encoding; keys may be abbreviated):\n"
        f"{payload}\n\n"
        "==="
    )


def prune_snapshot_for_prompt(
    snapshot: dict[str, Any],
    *,
    mode: str = "off",
    pregate_score: float = 0.0,
    failure_signature_repeat: bool = False,
    max_user_frames: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prune snapshot frames for Stage-B/TOON prompt assembly.

    The heuristic keeps the crash frame, preserves user-code-adjacent frames,
    then ranks remaining user-code frames by closeness to the crash plus a
    small boost from pregate/failure-repeat signals.
    """
    frames_obj = snapshot.get("frames")
    frames = frames_obj if isinstance(frames_obj, list) else []
    if mode != "heuristic" or not frames:
        return snapshot, {
            "mode": str(mode or "off"),
            "applied": False,
            "frames_before": len(frames),
            "frames_after": len(frames),
            "kept_frame_indices": list(range(len(frames))),
        }

    crash_index_raw = snapshot.get("crash_frame_index")
    if isinstance(crash_index_raw, int):
        crash_index = max(0, min(len(frames) - 1, int(crash_index_raw)))
    else:
        crash_index = 0

    crash_frame_obj = frames[crash_index]
    crash_frame = crash_frame_obj if isinstance(crash_frame_obj, dict) else {}
    crash_file_rel = str(crash_frame.get("file_rel") or "")
    bounded_pregate_score = max(0.0, min(1.0, float(pregate_score)))

    keep_indices = {crash_index}
    for neighbor_index in (crash_index - 1, crash_index + 1):
        if not 0 <= neighbor_index < len(frames):
            continue
        neighbor_obj = frames[neighbor_index]
        neighbor = neighbor_obj if isinstance(neighbor_obj, dict) else {}
        if neighbor.get("file_rel") is not None:
            keep_indices.add(neighbor_index)

    scored_user_frames: list[tuple[float, int]] = []
    for index, frame_obj in enumerate(frames):
        if index in keep_indices or not isinstance(frame_obj, dict):
            continue
        if frame_obj.get("file_rel") is None:
            continue
        distance = abs(index - crash_index)
        score = max(0.0, 1.0 - 0.2 * float(distance))
        if str(frame_obj.get("file_rel") or "") == crash_file_rel:
            score += 0.5
        if failure_signature_repeat:
            score += 0.25
        score += 0.25 * bounded_pregate_score
        scored_user_frames.append((score, index))

    kept_user_frames = sum(
        1
        for index in keep_indices
        if isinstance(frames[index], dict) and frames[index].get("file_rel") is not None
    )
    remaining_budget = max(0, int(max_user_frames) - kept_user_frames)
    for _score, index in sorted(
        scored_user_frames,
        key=lambda item: (item[0], -abs(item[1] - crash_index), -item[1]),
        reverse=True,
    )[:remaining_budget]:
        keep_indices.add(index)

    ordered_keep_indices = sorted(keep_indices)
    pruned_frames = [frames[index] for index in ordered_keep_indices]
    crash_index_after = ordered_keep_indices.index(crash_index)
    pruned_snapshot = dict(snapshot)
    pruned_snapshot["frames"] = pruned_frames
    pruned_snapshot["crash_frame_index"] = crash_index_after
    return pruned_snapshot, {
        "mode": "heuristic",
        "applied": True,
        "frames_before": len(frames),
        "frames_after": len(pruned_frames),
        "kept_frame_indices": ordered_keep_indices,
        "crash_frame_index_before": crash_index,
        "crash_frame_index_after": crash_index_after,
    }


def _build_snapshot_prompt_text(
    snapshot: dict[str, Any],
    *,
    use_toon: bool = True,
    use_compact_keys: bool = True,
) -> str:
    """Return a compact snapshot payload only when it beats the legacy section.

    This keeps TOON snapshot prompts within the same evidence surface as the
    old human-readable snapshot section and avoids regressions where the
    replacement payload becomes longer than the baseline prompt path.
    """
    from evals.toon_llm_encoder import encode_for_llm

    projected_snapshot = _project_snapshot_for_prompt(snapshot)
    encoded_payload = encode_for_llm(
        projected_snapshot,
        use_toon=use_toon,
        use_compact_keys=use_compact_keys,
    )
    if not encoded_payload.strip():
        return ""

    legacy_section = _format_snapshot_section(snapshot)
    encoded_section = _format_snapshot_payload_section(encoded_payload)
    return encoded_payload if len(encoded_section) < len(legacy_section) else ""


def _format_snapshot_index_section(snapshot: dict[str, Any], *, max_frames: int = 6) -> str:
    """Format a snapshot index for helper decision prompts.

    This intentionally excludes locals/values to avoid biasing helper decisions and to keep the
    payload compact. The full snapshot is available in Stage-B when explicitly requested.
    """
    lines: list[str] = []
    lines.append("=== llmdebug Snapshot Index ===")

    exc = snapshot.get("exception", {})
    exc_type = exc.get("type", "Unknown")
    exc_msg = exc.get("message", "")
    lines.append(f"Exception: {exc_type} — {exc_msg}")
    lines.append("")

    frames_obj = snapshot.get("frames")
    frames = frames_obj if isinstance(frames_obj, list) else []
    shown = 0
    for i, frame in enumerate(frames):
        if shown >= max(0, int(max_frames)):
            break
        if not isinstance(frame, dict):
            continue
        file_rel = frame.get("file_rel")
        if file_rel is None:
            continue
        label = "crash site" if i == 0 else "caller"
        func = frame.get("function", "?")
        line_no = frame.get("line", "?")
        lines.append(f"Frame {shown} ({label}): {file_rel}:{line_no} in {func}()")
        shown += 1

    lines.append("===")
    return "\n".join(lines)


def _format_frame_locals_section(snapshot: dict[str, Any], *, max_locals: int = 5) -> str:
    """Format crash frame with code and locals for ``frame_locals_lite`` action.

    Richer than ``_format_snapshot_index_section`` (includes code + locals for the
    crash frame) but cheaper than a full snapshot (single frame only).
    """
    lines: list[str] = []
    lines.append("=== llmdebug Crash Frame ===")

    exc = snapshot.get("exception", {})
    exc_type = exc.get("type", "Unknown")
    exc_msg = exc.get("message", "")
    lines.append(f"Exception: {exc_type} — {exc_msg}")
    lines.append("")

    frames_obj = snapshot.get("frames")
    frames = frames_obj if isinstance(frames_obj, list) else []
    # Find the first user-code frame (crash site).
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        file_rel = frame.get("file_rel")
        if file_rel is None:
            continue
        func = frame.get("function", "?")
        line_no = frame.get("line", "?")
        code = frame.get("code", "")
        lines.append(f"Crash site: {file_rel}:{line_no} in {func}()")
        if code:
            lines.append(f"  Code: {code}")
        frame_locals = frame.get("locals", {})
        locals_meta = frame.get("locals_meta", {})
        shown_locals = 0
        if frame_locals:
            lines.append("  Locals:")
            for var_name, var_value in frame_locals.items():
                if var_name.startswith("@"):
                    continue
                if shown_locals >= max(0, int(max_locals)):
                    break
                var_meta = locals_meta.get(var_name)
                lines.append(_format_local_value(var_name, var_value, var_meta))
                shown_locals += 1
        break  # Only the crash frame.

    lines.append("===")
    return "\n".join(lines)


def _format_snapshot_full_section(snapshot: dict[str, Any], *, max_frames: int = 3) -> str:
    """Format full projected snapshot for ``snapshot_full_lite`` action.

    Delegates to ``_format_snapshot_section`` — the same content that a successful
    ``invoke_helper`` Stage-A would have surfaced, but pre-computed without an LLM call.
    """
    return _format_snapshot_section(snapshot, max_frames=max_frames)


_GENERIC_SNAPSHOT_JUSTIFICATIONS = {
    "need more context",
    "insufficient context",
    "not enough info",
    "not enough information",
    "need snapshot",
    "need the snapshot",
    "need full snapshot",
    "need the full snapshot",
    "require snapshot",
    "require the snapshot",
    "require full snapshot",
    "require the full snapshot",
    "requires snapshot",
    "requires the snapshot",
    "requires full snapshot",
    "requires the full snapshot",
    "more context",
}


def _is_generic_snapshot_justification(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()
    if not normalized:
        return True
    if normalized in _GENERIC_SNAPSHOT_JUSTIFICATIONS:
        return True
    if re.fullmatch(
        r"(need|require|requires)\s+(more\s+)?(context|info|information)(\s+please)?", normalized
    ):
        return True
    if re.fullmatch(
        r"(need|require|requires)\s+(the\s+)?(full\s+)?snapshot(\s+please)?",
        normalized,
    ):
        return True
    return bool(
        re.fullmatch(
            r"(insufficient|not\s+enough)\s+(context|info|information)",
            normalized,
        )
    )


def _prompt_block_override(
    overrides: Mapping[str, str] | None,
    *,
    key: str,
    default_text: str,
) -> str:
    if overrides is None:
        return default_text
    candidate = overrides.get(key)
    if isinstance(candidate, str) and candidate.strip():
        return candidate
    return default_text


_STAGE_A_CALIBRATION_EXAMPLES = (
    "Examples of correct decisions:\n"
    "\n"
    "Example 1 — traceback IS sufficient:\n"
    "  Exception: IndexError: list index out of range\n"
    "  Crash: utils.py:42 in get_last() → return items[length]\n"
    '  Diagnosis: "Off-by-one: items[length] should be items[length - 1]"\n'
    "  Missing fact: null → need_more_context=false\n"
    "\n"
    "Example 2 — traceback IS sufficient:\n"
    "  Exception: KeyError: 'user_id'\n"
    '  Crash: handlers.py:15 in process() → uid = data["user_id"]\n'
    "  Diagnosis: \"Dict key mismatch: key is likely 'userId' or 'id'\"\n"
    "  Missing fact: null → need_more_context=false\n"
    "\n"
    "Example 3 — traceback NOT sufficient:\n"
    "  Exception: AssertionError (no message)\n"
    "  Crash: test_parser.py:30 in test_parse() → assert result == expected\n"
    '  Diagnosis: "Cannot determine root cause: assertion has no message and crash is in test"\n'
    '  Missing fact: "parser.py source to see parse() implementation" → need_more_context=true\n'
)


def render_stage_a_request_prompt(
    *,
    evidence: dict[str, Any],
    snapshot: dict[str, Any] | None,
    helper_budget_total: int = 0,
    helper_budget_remaining: int = 0,
    helper_action: str = ACTION_INVOKE_HELPER,
    tep_actor_overrides: Mapping[str, str] | None = None,
) -> str:
    """Build the Stage-A prompt that asks the LLM to select evidence.

    Includes traceback, pytest output, file index, and optional snapshot
    context.  Returns a single prompt string for the context-request step.
    """
    repro = evidence.get("repro")
    stdout = evidence.get("stdout", "")
    stderr = evidence.get("stderr", "")
    failure_signature = evidence.get("failure_signature", "")
    previous_failure_signature = evidence.get("previous_failure_signature")
    failure_signature_repeat = bool(evidence.get("failure_signature_repeat"))
    previous_attempt_diff = evidence.get("previous_attempt_diff")
    file_index = evidence.get("file_index", "")

    allowed_request_types = _helper_action_allowed_request_types(helper_action)
    allowed_request_types_text = ", ".join(allowed_request_types)
    request_mode_instructions = (
        f"Allowed request types: {allowed_request_types_text}."
        if helper_action == ACTION_INVOKE_HELPER
        else (
            "Action mode: request at most one targeted "
            f"{allowed_request_types_text} item. Do not request any other request type. "
            "Set need_snapshot=false and need_compact_context=false."
        )
    )
    strict_json_instructions = ""
    if helper_action == ACTION_RETRY_HELPER_STRICT_JSON:
        strict_json_instructions = (
            "\nSTRICT JSON mode:\n"
            "Return JSON only with no prose, markdown, comments, or trailing explanation.\n"
            "If traceback evidence is already sufficient, set need_more_context=false and requests=[].\n"
        )

    instructions_block = _prompt_block_override(
        tep_actor_overrides,
        key="instructions",
        default_text=(
            "You are selecting additional debugging evidence before proposing a patch.\n"
            "\n"
            "STEP 1 — DIAGNOSE from traceback alone:\n"
            "Read the exception type, message, crash file/line, and test output below.\n"
            "Write a one-sentence root-cause hypothesis.\n"
            "Then name a specific fact that is MISSING from the traceback and would change your diagnosis,\n"
            "or write null if no fact is missing.\n"
            "\n"
            "STEP 2 — DECIDE:\n"
            "If missing_fact is null → set need_more_context=false, requests=[].\n"
            "If you named a missing fact → request ONLY that item (1 request, max 2).\n"
            "Only set need_snapshot=true if you require the full llmdebug snapshot (locals/state); "
            "prefer targeted frame requests.\n"
            "Set need_compact_context=true only when the current evidence is too large/noisy and "
            "a compacted evidence bundle is needed before patching.\n"
            + request_mode_instructions
            + strict_json_instructions
        ),
    )
    calibration_block = _prompt_block_override(
        tep_actor_overrides,
        key="calibration_examples",
        default_text=_STAGE_A_CALIBRATION_EXAMPLES,
    )
    response_contract_block = _prompt_block_override(
        tep_actor_overrides,
        key="response_contract",
        default_text="Return STRICT JSON only with this schema:\n" + STAGE_A_REQUEST_SCHEMA_PRETTY,
    )

    parts: list[str] = []
    parts.append(
        "\n".join(
            (
                instructions_block.rstrip(),
                calibration_block.rstrip(),
                "",
                response_contract_block.rstrip(),
                "",
            )
        )
    )
    if helper_budget_total > 0:
        parts.append(
            _prompt_block_override(
                tep_actor_overrides,
                key="budget",
                default_text=(
                    f"Budget: You have {helper_budget_remaining} helper call(s) remaining "
                    f"(of {helper_budget_total} total). "
                    "If you use it on low-value context, no further calls are available."
                ),
            )
        )
    parts.append(f"Repro command:\n{repro}\n")
    parts.append("Pytest output (stdout):\n" + (stdout or "").rstrip() + "\n")
    if stderr:
        parts.append("Pytest output (stderr):\n" + stderr.rstrip() + "\n")
    if snapshot is not None:
        condition = str(evidence.get("condition") or "")
        is_adaptive = condition in {
            Condition.LAYERED_ADAPTIVE_GATED.value,
            Condition.LAYERED_ADAPTIVE_LLM.value,
        }
        parts.append(
            _format_snapshot_index_section(snapshot)
            if is_adaptive
            else _format_snapshot_section(snapshot)
        )
    if failure_signature:
        parts.append(f"Current failure signature:\n{failure_signature}\n")
    if previous_failure_signature is not None:
        parts.append(f"Previous failure signature:\n{previous_failure_signature}\n")
        parts.append(f"Failure signature repeated:\n{failure_signature_repeat}\n")
    if previous_attempt_diff is not None:
        prompt_safe_diff = _prompt_safe_previous_attempt_diff(previous_attempt_diff)
        parts.append(
            "Structured diff vs previous failed attempt:\n"
            + json.dumps(prompt_safe_diff, indent=2, ensure_ascii=False)
            + "\n"
        )
    if file_index:
        parts.append(
            "Available files (use exact relative paths):\n" + str(file_index).rstrip() + "\n"
        )
    parts.append(
        _prompt_block_override(
            tep_actor_overrides,
            key="closing",
            default_text=(
                "Do NOT propose code edits in this step. "
                "Return JSON only. Use need_more_context=false with requests=[] when evidence is sufficient."
            ),
        )
    )
    return "\n".join(parts).rstrip() + "\n"


def render_stage_b_patch_prompt(
    *,
    evidence: dict[str, Any],
    snapshot: dict[str, Any] | None,
    snapshot_prompt_text: str = "",
    stage_a_context: str = "",
    snapshot_lite_context: str = "",
    include_project_files: bool = True,
    retry_packet: str = "",
    retry_history: str = "",
    retry_prompt_mode: str = "full",
    attempt: int = 1,
    known_evidence_refs: list[str] | None = None,
    new_evidence_ids: list[str] | None = None,
    stage_a_evidence_index: str = "",
    include_stage_a_evidence_index: bool = True,
    compacted_context: str = "",
    include_snapshot: bool = True,
    include_pytest_output: bool = True,
    include_stage_a_context: bool = True,
    prompt_retry_reflection: bool = False,
    prompt_adaptive_stage_b_cot: bool = False,
    prompt_repair_method_scaffold: bool = False,
    retry_repair_plan: str = "",
    prompt_order_mode: str = "current",
) -> str:
    """Build the Stage-B prompt that asks the LLM to propose a patch.

    Assembles all evidence sections (snapshot, stage-A context, retry
    history, project files) and applies per-section budgets.
    """
    sections = _build_stage_b_prompt_sections(
        evidence=evidence,
        snapshot=snapshot,
        snapshot_prompt_text=snapshot_prompt_text,
        stage_a_context=stage_a_context,
        snapshot_lite_context=snapshot_lite_context,
        stage_a_evidence_index=stage_a_evidence_index,
        include_stage_a_evidence_index=include_stage_a_evidence_index,
        compacted_context=compacted_context,
        include_snapshot=include_snapshot,
        include_pytest_output=include_pytest_output,
        include_stage_a_context=include_stage_a_context,
        include_project_files=include_project_files,
        retry_packet=retry_packet,
        retry_history=retry_history,
        retry_prompt_mode=retry_prompt_mode,
        attempt=attempt,
        known_evidence_refs=known_evidence_refs,
        new_evidence_ids=new_evidence_ids,
        prompt_retry_reflection=prompt_retry_reflection,
        prompt_adaptive_stage_b_cot=prompt_adaptive_stage_b_cot,
        prompt_repair_method_scaffold=prompt_repair_method_scaffold,
        retry_repair_plan=retry_repair_plan,
        prompt_order_mode=prompt_order_mode,
    )
    return _join_prompt_sections(sections)


def render_retry_plan_prompt(
    *,
    evidence: dict[str, Any],
    snapshot: dict[str, Any] | None,
    snapshot_prompt_text: str = "",
    stage_a_context: str = "",
    snapshot_lite_context: str = "",
    include_project_files: bool = True,
    retry_packet: str = "",
    retry_history: str = "",
    retry_prompt_mode: str = "full",
    attempt: int = 1,
    known_evidence_refs: list[str] | None = None,
    new_evidence_ids: list[str] | None = None,
    stage_a_evidence_index: str = "",
    include_stage_a_evidence_index: bool = True,
    compacted_context: str = "",
    include_snapshot: bool = True,
    include_pytest_output: bool = True,
    include_stage_a_context: bool = True,
    prompt_retry_reflection: bool = False,
    prompt_order_mode: str = "current",
) -> str:
    """Build the retry-only planning prompt used by plan-then-patch mode."""
    sections = _build_stage_b_prompt_sections(
        evidence=evidence,
        snapshot=snapshot,
        snapshot_prompt_text=snapshot_prompt_text,
        stage_a_context=stage_a_context,
        snapshot_lite_context=snapshot_lite_context,
        stage_a_evidence_index=stage_a_evidence_index,
        include_stage_a_evidence_index=include_stage_a_evidence_index,
        compacted_context=compacted_context,
        include_snapshot=include_snapshot,
        include_pytest_output=include_pytest_output,
        include_stage_a_context=include_stage_a_context,
        include_project_files=include_project_files,
        retry_packet=retry_packet,
        retry_history=retry_history,
        retry_prompt_mode=retry_prompt_mode,
        attempt=attempt,
        known_evidence_refs=known_evidence_refs,
        new_evidence_ids=new_evidence_ids,
        prompt_retry_reflection=prompt_retry_reflection,
        planning_mode=True,
        prompt_order_mode=prompt_order_mode,
    )
    return _join_prompt_sections(sections)


def _build_stage_b_prompt_sections(
    *,
    evidence: dict[str, Any],
    snapshot: dict[str, Any] | None,
    snapshot_prompt_text: str,
    stage_a_context: str,
    stage_a_evidence_index: str,
    include_stage_a_evidence_index: bool,
    compacted_context: str,
    include_snapshot: bool,
    include_pytest_output: bool,
    include_stage_a_context: bool,
    include_project_files: bool,
    retry_packet: str,
    retry_history: str,
    retry_prompt_mode: str,
    attempt: int,
    known_evidence_refs: list[str] | None,
    new_evidence_ids: list[str] | None,
    snapshot_lite_context: str = "",
    prompt_retry_reflection: bool = False,
    prompt_adaptive_stage_b_cot: bool = False,
    prompt_repair_method_scaffold: bool = False,
    retry_repair_plan: str = "",
    planning_mode: bool = False,
    prompt_order_mode: str = "current",
    enable_tep: bool = False,
    stagnation_streak: int = 0,
    tep_actor_overrides: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the ordered list of prompt sections for Stage-B.

    Each section is a dict with ``key`` and ``text``. ``_apply_prompt_budget`` trims
    sections in its own fixed ``drop_order``; sections not listed there, such as
    ``diagnosis_refinement``, are never trimmed (LLM-20). The unread per-section
    ``priority`` field was removed.
    """
    repro = evidence.get("repro")
    stdout = evidence.get("stdout", "")
    stderr = evidence.get("stderr", "")
    failure_signature = evidence.get("failure_signature", "")
    previous_failure_signature = evidence.get("previous_failure_signature")
    failure_signature_repeat = bool(evidence.get("failure_signature_repeat"))
    previous_attempt_diff = evidence.get("previous_attempt_diff")
    timed_out = bool(evidence.get("timed_out"))
    timeout_sec_raw = evidence.get("timeout_sec")
    timeout_sec = int(timeout_sec_raw) if isinstance(timeout_sec_raw, int) else None
    previous_note = str(evidence.get("previous_note") or "")
    previous_patch_verifier_rejected = bool(evidence.get("previous_patch_verifier_rejected"))
    previous_patch_verifier_reasons_raw = evidence.get("previous_patch_verifier_reasons")
    previous_patch_verifier_reasons = (
        [str(item) for item in previous_patch_verifier_reasons_raw]
        if isinstance(previous_patch_verifier_reasons_raw, list)
        else []
    )
    previous_patch_verifier_scope = str(evidence.get("previous_patch_verifier_scope") or "")
    file_index = evidence.get("file_index", "")
    files = evidence.get("files") or ""
    has_previous = previous_failure_signature is not None
    is_delta_retry = retry_prompt_mode == "delta" and attempt > 1
    condition_name = str(evidence.get("condition") or "")
    is_adaptive_condition = condition_name in {
        Condition.LAYERED_ADAPTIVE_GATED.value,
        Condition.LAYERED_ADAPTIVE_LLM.value,
    }
    is_cot_repair = condition_name == Condition.COT_REPAIR.value or (
        prompt_adaptive_stage_b_cot and is_adaptive_condition
    )

    sections: list[dict[str, Any]] = []
    if is_delta_retry:
        sections.append(
            {
                "key": "instructions",
                "text": (
                    "You are a coding assistant. This is a retry with delta-first protocol.\n"
                    "Use the retry packet and new evidence blocks first.\n"
                    "Prioritize non-test source files; edit tests only when evidence shows a test is wrong.\n"
                    "Ignore cosmetic CLI/TUI wording or formatting tweaks unless evidence requires them.\n"
                    "Avoid equivalent edits unless evidence changed.\n"
                    "Goal: produce the minimal edit that resolves the failing behavior."
                ),
            }
        )
    else:
        sections.append(
            {
                "key": "instructions",
                "text": (
                    "You are a coding assistant. Fix the failing behavior shown by the tests.\n"
                    "Prioritize non-test source files; edit tests only when evidence shows a test is wrong.\n"
                    "Ignore cosmetic CLI/TUI wording or formatting tweaks unless evidence requires them.\n"
                    "Make the smallest behavior-changing edit (avoid extra refactors or added features).\n"
                    "Do not submit observability-only patches (logging/comments/assertions) unless required "
                    "for the fix.\n"
                    "Do not repeat an equivalent patch when the failure signature is unchanged."
                ),
            }
        )

    rca_lines = [
        "RCA checklist — write each section explicitly before proposing your patch:",
        "1) Observed failure: quote the concrete evidence (exception, assertion, output)",
        "2) Root-cause diagnosis: one falsifiable sentence explaining why the code fails",
        "3) Supporting evidence: why the diagnosis is the best explanation of the data",
        "4) Minimal fix plan: what to change and why it addresses the root cause",
        "5) Verification plan: how to confirm the fix resolves the failure",
    ]
    if has_previous:
        rca_lines.append("6) Diff vs previous failed attempt: what changed and why")
    sections.append({"key": "rca_checklist", "text": "\n".join(rca_lines)})

    if prompt_repair_method_scaffold and not planning_mode:
        sections.append(
            {
                "key": "repair_method_scaffold",
                "text": (
                    "Modification method — write each item explicitly before your patch:\n"
                    "- Code region: identify the exact file and code region to edit.\n"
                    "- Modification method: state the minimum behavior-changing modification.\n"
                    "- Invariant: state one invariant or constraint the edit must preserve.\n"
                    "- Verification: state how the edit will be verified against the failing "
                    "evidence."
                ),
            }
        )

    if is_cot_repair:
        sections.append(
            {
                "key": "cot_preamble",
                "text": (
                    "Before proposing a fix, think through the problem step by step:\n"
                    "Step 1: Identify the failing behavior from the test output.\n"
                    "Step 2: Trace the exception to its root cause in the source code.\n"
                    "Step 3: Formulate a hypothesis for why the code fails.\n"
                    "Step 4: Determine the minimal code change that fixes the root cause.\n"
                    "\n"
                    "Write out your reasoning for each step, then provide the patch."
                ),
            }
        )

    if retry_packet:
        sections.append({"key": "retry_packet", "text": retry_packet.rstrip() + "\n"})
    if retry_history:
        sections.append(
            {
                "key": "retry_history",
                "text": "Retry history from previous attempts:\n" + retry_history.rstrip() + "\n",
            }
        )

    sections.append({"key": "repro", "text": f"Repro command:\n{repro}\n"})

    if not is_delta_retry and include_pytest_output:
        pytest_block = "Pytest output (stdout):\n" + (stdout or "").rstrip() + "\n"
        if stderr:
            pytest_block += "\nPytest output (stderr):\n" + stderr.rstrip() + "\n"
        sections.append({"key": "pytest_output", "text": pytest_block})

    if not is_delta_retry and include_snapshot:
        snapshot_section_text = ""
        if snapshot_prompt_text.strip():
            snapshot_section_text = _format_snapshot_payload_section(snapshot_prompt_text)
        elif snapshot is not None:
            snapshot_section_text = _format_snapshot_section(snapshot)
        if snapshot_section_text:
            sections.append({"key": "snapshot", "text": snapshot_section_text})
    if (
        not is_delta_retry
        and include_snapshot
        and (snapshot is not None or snapshot_prompt_text.strip())
    ):
        sections.append(
            {
                "key": "distractor_mitigation",
                "text": (
                    "The snapshot above contains many locals. Focus ONLY on variables mentioned "
                    "in the exception message, the crash line, or the test assertion. Other locals "
                    "are likely irrelevant distractors — ignore them."
                ),
            }
        )

    if failure_signature:
        sections.append(
            {
                "key": "failure_signatures",
                "text": f"Current failure signature:\n{failure_signature}\n",
            }
        )
    if has_previous:
        sections.append(
            {
                "key": "failure_signatures",
                "text": (
                    f"Previous failure signature:\n{previous_failure_signature}\n"
                    f"Failure signature repeated:\n{failure_signature_repeat}\n"
                ),
            }
        )

    if (not is_delta_retry) and previous_attempt_diff is not None:
        prompt_safe_diff = _prompt_safe_previous_attempt_diff(previous_attempt_diff)
        sections.append(
            {
                "key": "previous_attempt_diff",
                "text": (
                    "Structured diff vs previous failed attempt:\n"
                    + json.dumps(prompt_safe_diff, indent=2, ensure_ascii=False)
                    + "\n"
                ),
            }
        )
    if failure_signature_repeat:
        if enable_tep and stagnation_streak > 0:
            # TEP: Escalate nudge intensity with stagnation depth.
            if stagnation_streak >= 3:
                nudge_text = (
                    f"CRITICAL STAGNATION ({stagnation_streak} consecutive identical failures): "
                    "Abandon your current approach entirely. Consider a fundamentally different "
                    "root cause or repair strategy.\n"
                )
            elif stagnation_streak >= 2:
                nudge_text = (
                    f"Stagnation warning ({stagnation_streak} identical failures): "
                    "Your hypothesis is likely wrong. Articulate a DIFFERENT diagnosis "
                    "before attempting another repair.\n"
                )
            else:
                nudge_text = (
                    "Loop breaker: The failure signature repeated. "
                    "Change either diagnosis or evidence basis before retrying.\n"
                )
            sections.append(
                {
                    "key": "loop_breaker",
                    "text": nudge_text,
                }
            )
            if stagnation_streak >= 2:
                sections.append(
                    {
                        "key": "diagnosis_refinement",
                        "text": (
                            "REQUIRED: Before proposing a fix, write one sentence: "
                            "'My new diagnosis is [X] because [reasoning different from previous].'\n"
                        ),
                    }
                )
        else:
            sections.append(
                {
                    "key": "loop_breaker",
                    "text": (
                        "Loop breaker: The signature repeated. "
                        "Change either diagnosis or evidence basis before retrying.\n"
                    ),
                }
            )
    if prompt_retry_reflection and has_previous:
        sections.append(
            {
                "key": "retry_reflection",
                "text": (
                    "Retry reflection — write each item explicitly before your patch:\n"
                    "- Previous failure: explain why the previous attempt failed or remained "
                    "insufficient.\n"
                    "- Learned constraint: state one concrete constraint learned from the latest "
                    "failure evidence.\n"
                    "- Changed basis: state how this retry materially changes the diagnosis or "
                    "evidence basis.\n"
                    "- Non-equivalence: state why the next edit is not equivalent to the previous "
                    "failed patch.\n"
                ),
            }
        )
    if timed_out:
        timeout_label = (
            f"{timeout_sec}s" if isinstance(timeout_sec, int) and timeout_sec > 0 else "timeout"
        )
        sections.append(
            {
                "key": "timeout_risk",
                "text": (
                    "Timeout risk signal: the latest failing run timed out "
                    f"(budget={timeout_label}).\n"
                    "Prefer low-risk fixes that reduce pathological loops/retries, avoid added "
                    "blocking I/O, and preserve bounded behavior.\n"
                ),
            }
        )
    if previous_note == "patch_verifier_rejected" or previous_patch_verifier_rejected:
        reasons_text = (
            ", ".join(previous_patch_verifier_reasons[:6])
            if previous_patch_verifier_reasons
            else "insufficient source relevance"
        )
        scope_text = previous_patch_verifier_scope or "adaptive_only"
        sections.append(
            {
                "key": "verifier_retry_guidance",
                "text": (
                    "Verifier-aware retry guidance: the previous candidate was rejected by patch "
                    f"verifier (scope={scope_text}; reasons={reasons_text}).\n"
                    "Prioritize traceback-linked non-test source edits first; treat test edits as "
                    "last resort unless evidence proves the test is wrong.\n"
                ),
            }
        )

    if not is_delta_retry and file_index:
        sections.append(
            {
                "key": "file_index",
                "text": "Available files (use exact relative paths):\n"
                + str(file_index).rstrip()
                + "\n",
            }
        )
    if compacted_context:
        sections.append(
            {
                "key": "compacted_context",
                "text": "Compacted context summary:\n" + compacted_context.rstrip() + "\n",
            }
        )
    if snapshot_lite_context:
        sections.append(
            {
                "key": "snapshot_lite",
                "text": "Compact snapshot index:\n" + snapshot_lite_context.rstrip() + "\n",
            }
        )
    if include_stage_a_context and stage_a_context:
        sections.append(
            {
                "key": "stage_a_context",
                "text": "Stage-A requested context:\n" + stage_a_context.rstrip() + "\n",
            }
        )
    if include_stage_a_evidence_index and stage_a_evidence_index:
        sections.append(
            {
                "key": "stage_a_evidence_index",
                "text": "Stage-A evidence index:\n" + stage_a_evidence_index.rstrip() + "\n",
            }
        )
    if retry_repair_plan.strip():
        sections.append(
            {
                "key": "retry_repair_plan",
                "text": "Structured repair plan from planning step:\n"
                + retry_repair_plan.strip()
                + "\n",
            }
        )

    known_refs = [item for item in (known_evidence_refs or []) if item]
    if is_delta_retry and known_refs:
        sections.append(
            {
                "key": "known_evidence_refs",
                "text": "Known evidence references:\n"
                + "\n".join(f"- {item}" for item in known_refs),
            }
        )
    if is_delta_retry and new_evidence_ids:
        sections.append(
            {
                "key": "new_evidence_ids",
                "text": "New evidence IDs in this retry:\n"
                + "\n".join(f"- {item}" for item in new_evidence_ids),
            }
        )
    if include_project_files and files and not is_delta_retry:
        sections.append(
            {
                "key": "project_files",
                "text": "Project files:\n" + str(files).rstrip() + "\n",
            }
        )

    sections.append(
        {
            "key": "closing",
            "text": (
                "CRITICAL — RETURN ONLY THE REPAIR PLAN:\n"
                "- Do NOT return code, a diff, file contents, or markdown fences.\n"
                "- Use these headings exactly: Diagnosis, Why previous attempt failed, "
                "New constraint, Minimal edit plan, Verification plan.\n"
                "- Keep the plan concise and actionable.\n"
                "- Base the plan on the traceback-linked root cause, not superficial symptoms."
                if planning_mode
                else "CRITICAL — YOUR PATCH MUST:\n"
                "- Target the ROOT CAUSE visible in the traceback, not the symptom.\n"
                "- If the traceback alone identifies the bug, ignore supplementary context.\n"
                "- Be the smallest correct edit (no refactoring, no added features).\n"
                "- Not repeat a previously failed equivalent patch."
            ),
        }
    )
    if tep_actor_overrides:
        overridden_sections: list[dict[str, Any]] = []
        for section in sections:
            key = str(section.get("key") or "")
            if key in tep_actor_overrides:
                overridden = dict(section)
                overridden["text"] = tep_actor_overrides[key]
                overridden_sections.append(overridden)
            else:
                overridden_sections.append(section)
        sections = overridden_sections
    return _reorder_stage_b_sections(sections, prompt_order_mode=prompt_order_mode)


def _reorder_stage_b_sections(
    sections: list[dict[str, Any]],
    *,
    prompt_order_mode: str,
) -> list[dict[str, Any]]:
    """Apply stable Stage-B section ordering without changing membership."""
    normalized_mode = str(prompt_order_mode or "current").strip().lower()
    if normalized_mode == "current":
        return sections

    order_map: dict[str, int]
    if normalized_mode == "crash_first":
        order_map = {
            "instructions": 0,
            "rca_checklist": 1,
            "repair_method_scaffold": 2,
            "cot_preamble": 2,
            "failure_signatures": 3,
            "timeout_risk": 4,
            "previous_attempt_diff": 4,
            "snapshot": 5,
            "snapshot_lite": 5,
            "distractor_mitigation": 6,
            "pytest_output": 7,
            "stage_a_context": 8,
            "compacted_context": 8,
            "retry_packet": 9,
            "retry_history": 10,
            "loop_breaker": 11,
            "retry_reflection": 11,
            "verifier_retry_guidance": 11,
            "retry_repair_plan": 12,
            "repro": 13,
            "stage_a_evidence_index": 14,
            "file_index": 15,
            "known_evidence_refs": 16,
            "new_evidence_ids": 16,
            "project_files": 17,
            "closing": 18,
        }
    elif normalized_mode == "evidence_last":
        order_map = {
            "instructions": 0,
            "rca_checklist": 1,
            "repair_method_scaffold": 2,
            "cot_preamble": 2,
            "retry_packet": 3,
            "retry_history": 4,
            "failure_signatures": 5,
            "loop_breaker": 6,
            "previous_attempt_diff": 6,
            "retry_reflection": 6,
            "timeout_risk": 6,
            "verifier_retry_guidance": 6,
            "retry_repair_plan": 7,
            "repro": 8,
            "pytest_output": 9,
            "snapshot": 10,
            "snapshot_lite": 10,
            "distractor_mitigation": 11,
            "stage_a_context": 12,
            "compacted_context": 12,
            "stage_a_evidence_index": 13,
            "file_index": 14,
            "known_evidence_refs": 15,
            "new_evidence_ids": 15,
            "project_files": 16,
            "closing": 17,
        }
    else:
        return sections

    default_rank = max(order_map.values(), default=0) + 1
    indexed_sections = list(enumerate(sections))
    return [
        section
        for _, section in sorted(
            indexed_sections,
            key=lambda item: (order_map.get(str(item[1].get("key") or ""), default_rank), item[0]),
        )
    ]


def _join_prompt_sections(sections: list[dict[str, Any]]) -> str:
    parts = [
        str(section.get("text") or "").strip("\n") for section in sections if section.get("text")
    ]
    if not parts:
        return ""
    return "\n\n".join(parts).rstrip() + "\n"


def _truncate_text_for_section(*, key: str, text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 64:
        return text[:max_chars]

    marker = "\n...[truncated]...\n"
    if len(marker) >= max_chars:
        return text[:max_chars]

    if key in {"pytest_output", "retry_packet"}:
        tail_chars = max_chars // 2
        head_chars = max_chars - tail_chars - len(marker)
        if head_chars < 0:
            return text[-max_chars:]
        return text[:head_chars] + marker + text[-tail_chars:]

    head_chars = max_chars - len(marker)
    return text[:head_chars] + marker


def _section_text_by_key(sections: list[dict[str, Any]]) -> dict[str, str]:
    by_key: dict[str, str] = {}
    for section in sections:
        key = str(section.get("key") or "")
        if not key:
            continue
        text = str(section.get("text") or "")
        if key in by_key:
            by_key[key] = by_key[key] + text
        else:
            by_key[key] = text
    return by_key


def _apply_prompt_budget(
    *,
    sections: list[dict[str, Any]],
    budget: dict[str, int],
    is_delta_retry: bool,
) -> PromptBudgetResult:
    """Enforce per-section and total-char limits on prompt sections.

    Returns a ``PromptBudgetResult`` with the final prompt, overflow
    metrics, and a list of dropped section keys.
    """
    capped_sections = [dict(section) for section in sections]
    original_by_key = _section_text_by_key(sections)
    per_section_limits = {
        "retry_packet": int(budget.get("retry_delta_chars", 0)),
        "pytest_output": int(budget.get("stdout_chars", 0)),
        "snapshot": int(budget.get("snapshot_chars", 0)),
        "snapshot_lite": int(budget.get("snapshot_chars", 0)),
        "compacted_context": int(budget.get("evidence_chars", 0)),
        "stage_a_context": int(budget.get("evidence_chars", 0)),
        "stage_a_evidence_index": int(budget.get("evidence_chars", 0)),
        "retry_history": int(budget.get("retry_history_chars", 0)),
    }

    for section in capped_sections:
        key = str(section.get("key") or "")
        text = str(section.get("text") or "")
        limit = per_section_limits.get(key, 0)
        if limit > 0:
            section["text"] = _truncate_text_for_section(key=key, text=text, max_chars=limit)

    prompt = _join_prompt_sections(capped_sections)
    total_budget = int(budget.get("total_chars", 0))
    if total_budget <= 0 or len(prompt) <= total_budget:
        final_by_key = _section_text_by_key(capped_sections)
        return PromptBudgetResult(
            prompt=prompt,
            used_chars=len(prompt),
            overflow_chars=0,
            dropped_sections=[],
            section_status={
                key: ("full" if final_by_key.get(key, "") == original else "truncated")
                for key, original in original_by_key.items()
                if original
            },
            section_chars={
                key: len(final_by_key.get(key, ""))
                for key, original in original_by_key.items()
                if original
            },
            section_original_chars={
                key: len(original) for key, original in original_by_key.items() if original
            },
        )

    overflow = len(prompt) - total_budget
    dropped_sections: list[str] = []
    drop_order = [
        "project_files",
        "file_index",
        "stage_a_evidence_index",
        "known_evidence_refs",
        "retry_history",
        "snapshot",
        "snapshot_lite",
        "pytest_output",
        "compacted_context",
        "stage_a_context",
        "previous_attempt_diff",
    ]
    min_keep_by_key = {
        "snapshot": 240,
        "snapshot_lite": 240,
        "pytest_output": 240,
        "previous_attempt_diff": 240,
        "retry_history": 160,
        "compacted_context": 512 if is_delta_retry else 240,
        "stage_a_context": 512 if is_delta_retry else 240,
    }

    for key in drop_order:
        if overflow <= 0:
            break
        for section in capped_sections:
            if str(section.get("key") or "") != key:
                continue
            text = str(section.get("text") or "")
            if not text:
                continue
            min_keep = min(min_keep_by_key.get(key, 0), len(text))
            target = max(min_keep, len(text) - overflow)
            next_text = _truncate_text_for_section(key=key, text=text, max_chars=target)
            overflow -= max(0, len(text) - len(next_text))
            section["text"] = next_text
            if not next_text.strip():
                dropped_sections.append(key)
            break

    prompt = _join_prompt_sections(capped_sections)
    if len(prompt) > total_budget:
        prompt = _truncate_text_for_section(key="final", text=prompt, max_chars=total_budget)
    overflow_chars = max(0, len(prompt) - total_budget)
    final_by_key = _section_text_by_key(capped_sections)
    section_status: dict[str, str] = {}
    section_chars: dict[str, int] = {}
    for key, original_text in original_by_key.items():
        if not original_text:
            continue
        final_text = final_by_key.get(key, "")
        if not final_text.strip():
            section_status[key] = "dropped"
            section_chars[key] = 0
            continue
        section_chars[key] = len(final_text)
        section_status[key] = "full" if final_text == original_text else "truncated"
    return PromptBudgetResult(
        prompt=prompt,
        used_chars=len(prompt),
        overflow_chars=overflow_chars,
        dropped_sections=sorted(set(dropped_sections)),
        section_status=section_status,
        section_chars=section_chars,
        section_original_chars={
            key: len(original) for key, original in original_by_key.items() if original
        },
    )


def _build_context_compaction_bundle(
    *,
    scope: set[str],
    snapshot: dict[str, Any] | None,
    stdout: str,
    stderr: str,
    stage_a_context: str,
    project_files_context: str,
    max_input_chars: int,
) -> ContextCompactionBundle:
    """Gather evidence sections into a ``ContextCompactionBundle``.

    Selects sections based on *scope* (pytest, snapshot, files) and
    truncates the combined input to *max_input_chars*.
    """
    parts: list[str] = []
    source_sections: list[str] = []
    source_chars: dict[str, int] = {}
    replace_snapshot = False
    replace_pytest_output = False
    replace_stage_a_context = False
    replace_project_files = False

    if "pytest" in scope:
        pytest_parts: list[str] = []
        if stdout:
            pytest_parts.append("Pytest output (stdout):\n" + stdout.rstrip())
        if stderr:
            pytest_parts.append("Pytest output (stderr):\n" + stderr.rstrip())
        if pytest_parts:
            pytest_text = "\n\n".join(pytest_parts).rstrip() + "\n"
            parts.append("=== pytest_output ===\n" + pytest_text)
            source_sections.append("pytest_output")
            source_chars["pytest_output"] = len(pytest_text)
            replace_pytest_output = True

    if "snapshot" in scope and snapshot is not None:
        snapshot_text = _format_snapshot_section(snapshot).rstrip() + "\n"
        parts.append("=== snapshot ===\n" + snapshot_text)
        source_sections.append("snapshot")
        source_chars["snapshot"] = len(snapshot_text)
        replace_snapshot = True

    if "files" in scope:
        if stage_a_context:
            stage_a_text = stage_a_context.rstrip() + "\n"
            parts.append("=== stage_a_context ===\n" + stage_a_text)
            source_sections.append("stage_a_context")
            source_chars["stage_a_context"] = len(stage_a_text)
            replace_stage_a_context = True
        if project_files_context:
            project_text = project_files_context.rstrip() + "\n"
            parts.append("=== project_files ===\n" + project_text)
            source_sections.append("project_files")
            source_chars["project_files"] = len(project_text)
            replace_project_files = True

    bundle_text = "\n\n".join(parts).rstrip()
    if bundle_text:
        bundle_text += "\n"
    if max_input_chars > 0 and len(bundle_text) > max_input_chars:
        bundle_text = _truncate_text_for_section(
            key="compaction_input",
            text=bundle_text,
            max_chars=max_input_chars,
        )
    return ContextCompactionBundle(
        text=bundle_text,
        source_sections=source_sections,
        source_chars=source_chars,
        replace_snapshot=replace_snapshot,
        replace_pytest_output=replace_pytest_output,
        replace_stage_a_context=replace_stage_a_context,
        replace_project_files=replace_project_files,
    )


def _should_invoke_context_compaction(
    *,
    mode: str,
    supports_context_compaction: bool,
    stage_a_requested: bool,
    policy_action_requested: bool,
    auto_overflow_triggered: bool,
    calls_used: int,
    max_calls: int,
    bundle_input_chars: int,
) -> tuple[bool, str]:
    """Decide whether context compaction should run.

    Returns (should_invoke, reason_token) after checking patcher
    support, call budget, input size, and compaction mode.
    """
    if not supports_context_compaction:
        return False, "patcher_unsupported"
    max_calls_safe = max(0, int(max_calls))
    if max_calls_safe <= 0:
        return False, "max_calls_disabled"
    if max(0, int(calls_used)) >= max_calls_safe:
        return False, "max_calls_reached"
    if max(0, int(bundle_input_chars)) <= 0:
        return False, "empty_input"
    if mode == "off":
        return False, "mode_off"
    if policy_action_requested:
        return True, "policy_action"
    if stage_a_requested:
        return True, "stage_a_requested"
    if mode == "always":
        return True, "mode_always"
    if mode == "auto_overflow" and auto_overflow_triggered:
        return True, "prompt_overflow"
    return False, "auto_no_overflow"


def render_context_compaction_prompt(
    *,
    evidence: dict[str, Any],
    attempt: int,
    reason: str,
    source_sections: list[str],
    target_chars: int,
    tep_actor_overrides: Mapping[str, str] | None = None,
) -> str:
    """Build a prompt asking the LLM to compact debugging evidence.

    The compacted output is used in subsequent patch-proposal prompts
    when the raw evidence exceeds the prompt budget.
    """
    case_id = str(evidence.get("case_id") or "")
    condition = str(evidence.get("condition") or "")
    failure_signature = str(evidence.get("failure_signature") or "")
    repro = evidence.get("repro")
    source_list = ", ".join(source_sections) if source_sections else "none"
    reason_line = str(reason or "").strip()
    if not reason_line:
        reason_line = "prompt overflow or noisy evidence"
    instructions_block = _prompt_block_override(
        tep_actor_overrides,
        key="instructions",
        default_text=(
            "You compact debugging evidence for a downstream patching model.\n"
            "Return a compact, loss-aware summary with exact literals for key facts.\n"
            "Preserve crash file/line, assertion text, failing values, and critical paths.\n"
            "Do not suggest edits.\n"
        ),
    )
    closing_block = _prompt_block_override(
        tep_actor_overrides,
        key="closing",
        default_text="",
    )
    body = (
        f"Case: {case_id}\n"
        f"Condition: {condition}\n"
        f"Attempt: {attempt}\n"
        f"Failure signature: {failure_signature}\n"
        f"Repro: {repro}\n"
        f"Reason: {reason_line}\n"
        f"Sources: {source_list}\n"
        f"Target chars: {max(1, int(target_chars))}\n"
    )
    suffix = f"\n{closing_block.rstrip()}\n" if closing_block.strip() else ""
    return f"{instructions_block.rstrip()}\n\n{body}{suffix}"


def render_prompt(*, evidence: dict[str, Any], snapshot: dict[str, Any] | None) -> str:
    """Backward-compatible prompt renderer (monolithic context protocol)."""
    return render_stage_b_patch_prompt(
        evidence=evidence,
        snapshot=snapshot,
        stage_a_context="",
        include_project_files=True,
    )


def _build_retry_history_section(
    *,
    prior_turns: list[str],
    max_chars: int,
) -> tuple[str, int, int, bool]:
    """Assemble a truncated retry-history section from prior turns.

    Returns (text, turns_included, used_chars, truncated).
    """
    if not prior_turns:
        return "", 0, 0, False
    if max_chars <= 0:
        return "", 0, 0, True

    cleaned_turns = [str(turn).strip() for turn in prior_turns if str(turn).strip()]
    if not cleaned_turns:
        return "", 0, 0, False

    selected_turns: list[str] = []
    used_chars = 0
    truncated = False
    for turn in reversed(cleaned_turns):
        candidate = turn if not selected_turns else "\n\n" + turn
        if used_chars + len(candidate) <= max_chars:
            selected_turns.append(turn)
            used_chars += len(candidate)
            continue

        truncated = True
        if not selected_turns:
            clipped = turn[:max_chars].rstrip()
            if clipped:
                selected_turns.append(clipped)
                used_chars = len(clipped)
        break

    if not selected_turns:
        return "", 0, 0, truncated

    rendered = "\n\n".join(selected_turns).rstrip()
    return rendered, len(rendered), len(selected_turns), truncated


def _render_retry_history_turn(
    *,
    attempt: int,
    note: str,
    failure_signature: str,
    patch_meta: dict[str, Any] | None,
) -> str:
    meta = patch_meta if isinstance(patch_meta, dict) else {}
    persisted_summary = meta.get("retry_summary")
    if isinstance(persisted_summary, dict):
        summary = cast(dict[str, Any], persisted_summary)
    else:
        summary = _build_retry_summary(
            attempt=attempt,
            note=note,
            failure_signature=failure_signature,
            patch_meta=meta,
        )
    lines = [
        f"Attempt {int(summary.get('attempt', attempt) or attempt)}",
        f"Hypothesis tested: {summary.get('hypothesis_tested', '')}",
        f"Evidence used: {summary.get('evidence_used', '')}",
        f"Outcome: {summary.get('outcome', note)}",
        f"New constraint: {summary.get('new_constraint', '')}",
    ]
    summary_failure_signature = str(summary.get("failure_signature") or "").strip()
    if summary_failure_signature:
        lines.append(f"Failure signature: {summary_failure_signature}")

    return "\n".join(lines).strip()


def _build_retry_summary(
    *,
    attempt: int,
    note: str,
    failure_signature: str,
    patch_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    def _compact(value: Any, *, max_chars: int = 220) -> str:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if len(text) <= max_chars:
            return text
        suffix = "...[truncated]"
        keep = max(0, max_chars - len(suffix))
        return text[:keep].rstrip() + suffix

    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _hypothesis_tested(meta: dict[str, Any]) -> str:
        diagnosis = str(meta.get("stage_a_diagnosis") or "").strip()
        if diagnosis:
            return _compact(diagnosis, max_chars=140)
        missing_fact = meta.get("stage_a_missing_fact")
        if isinstance(missing_fact, str) and missing_fact.strip():
            return _compact(f"traceback insufficient; probe {missing_fact.strip()}", max_chars=140)
        patcher_reason = str(meta.get("reason") or "").strip()
        if patcher_reason:
            return _compact(f"traceback-guided repair via {patcher_reason}", max_chars=140)
        return "traceback-guided minimal source fix"

    def _evidence_used(meta: dict[str, Any]) -> str:
        requests_count = _safe_int(meta.get("stage_a_requests_count"))
        need_snapshot = bool(meta.get("stage_a_need_snapshot"))
        need_compact_context = bool(meta.get("stage_a_need_compact_context"))
        new_count = _safe_int(meta.get("evidence_ids_new_count"))
        reused_count = _safe_int(meta.get("evidence_ids_reused_count"))
        parts: list[str] = []
        if requests_count > 0:
            parts.append(f"stage_a_requests={requests_count}")
        if need_snapshot:
            parts.append("full_snapshot")
        if need_compact_context:
            parts.append("compact_context")
        if new_count or reused_count:
            parts.append(f"evidence_ids(new={new_count}, reused={reused_count})")
        if bool(meta.get("stage_a_used_fallback")):
            parts.append(f"fallback:{meta.get('stage_a_reason') or 'unknown'!s}")
        if not parts:
            return "base_repair"
        return ", ".join(parts)

    def _new_constraint(note_value: str, meta: dict[str, Any]) -> str:
        if note_value == "syntax_preflight_failed":
            error_type = str(meta.get("syntax_preflight_last_error_type") or "").strip()
            error_msg = str(meta.get("syntax_preflight_last_error_msg") or "").strip()
            detail = error_msg or error_type or "syntax preflight failed"
            return _compact(detail, max_chars=140)
        if note_value == "patch_verifier_rejected":
            reasons = meta.get("patch_verifier_reasons")
            if isinstance(reasons, list) and reasons:
                rendered = ", ".join(str(item) for item in reasons[:3])
                return _compact(f"must satisfy verifier: {rendered}", max_chars=140)
            return "must satisfy patch verifier before apply"
        if bool(meta.get("patcher_request_failed")):
            kind = str(meta.get("patcher_request_failure_kind") or "transport").strip()
            error = str(meta.get("patcher_request_failure_error") or "").strip()
            return _compact(f"patch request failed: {kind} {error}".strip(), max_chars=140)
        if bool(meta.get("retry_gate_blocked")):
            reason = str(meta.get("retry_gate_reason") or "need new evidence").strip()
            return _compact(reason, max_chars=140)
        if note_value == "success":
            return "verified passing"
        if note_value:
            return _compact(note_value.replace("_", " "), max_chars=140)
        return "unknown"

    meta = patch_meta if isinstance(patch_meta, dict) else {}
    return {
        "attempt": int(attempt),
        "hypothesis_tested": _hypothesis_tested(meta),
        "evidence_used": _evidence_used(meta),
        "outcome": str(note or ""),
        "new_constraint": _new_constraint(note, meta),
        "failure_signature": _compact(failure_signature, max_chars=120)
        if failure_signature
        else "",
    }


def _render_retry_packet(
    *,
    failure_signature: str,
    previous_failure_signature: Any,
    failure_signature_repeat: bool,
    previous_attempt_diff: dict[str, Any] | None,
    timed_out: bool = False,
    timeout_sec: int | None = None,
    previous_note: str = "",
    previous_patch_verifier_rejected: bool = False,
    previous_patch_verifier_reasons: list[str] | None = None,
    previous_patch_verifier_scope: str = "",
    toon_encoding_active: bool = False,
) -> str:
    prompt_safe_diff = _prompt_safe_previous_attempt_diff(previous_attempt_diff)
    packet: dict[str, Any] = {
        "failure_signature": failure_signature,
        "previous_failure_signature": previous_failure_signature,
        "failure_signature_repeat": failure_signature_repeat,
        "previous_attempt_diff": prompt_safe_diff,
        "timed_out": bool(timed_out),
        "timeout_sec": int(timeout_sec) if isinstance(timeout_sec, int) else None,
        "previous_note": str(previous_note or ""),
        "previous_patch_verifier_rejected": bool(previous_patch_verifier_rejected),
        "previous_patch_verifier_reasons": list(previous_patch_verifier_reasons or []),
        "previous_patch_verifier_scope": str(previous_patch_verifier_scope or ""),
    }
    if toon_encoding_active:
        from evals.toon_llm_encoder import encode_for_llm

        encoded = encode_for_llm(packet, use_toon=True, use_compact_keys=False)
    else:
        encoded = json.dumps(packet, indent=2, ensure_ascii=False, default=str)
    return "Retry packet (delta-first, structured):\n" + encoded + "\n"


def _prompt_safe_previous_attempt_diff(
    previous_attempt_diff: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(previous_attempt_diff, dict):
        return None

    def _rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, child in value.items():
                mapped_key = "diagnosis_delta" if str(key) == "hypothesis_delta" else str(key)
                out[mapped_key] = _rewrite(child)
            return out
        if isinstance(value, list):
            return [_rewrite(item) for item in value]
        return value

    rewritten = _rewrite(previous_attempt_diff)
    if isinstance(rewritten, dict):
        return rewritten
    return None
