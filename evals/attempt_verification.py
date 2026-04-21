"""Patch application and syntax-preflight helpers for eval attempts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .eval_patches import (
    _backup_diff_target_files,
    _build_runner_forced_attempt_diff,
    _compute_patch_path_metrics,
    _render_patch_sanitize_retry_feedback,
    _render_syntax_preflight_retry_feedback,
    _restore_diff_target_files,
    _run_patch_sanitization,
    _run_syntax_preflight,
    apply_diff,
)


@dataclass
class SyntaxPreflightState:
    """Mutable state for patch-sanitization and syntax-preflight retry loops."""

    enabled: bool
    max_cycles: int
    cycles_used: int = 0
    failures: int = 0
    last_error_type: str = ""
    last_error_msg: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    retry_feedback: str = ""

    def apply_to_metadata(
        self,
        *,
        attempt_metadata: dict[str, Any],
        patch_meta: dict[str, Any],
    ) -> None:
        patch_meta["syntax_preflight_enabled"] = bool(self.enabled)
        patch_meta["syntax_preflight_cycles_used"] = int(self.cycles_used)
        patch_meta["syntax_preflight_failures"] = int(self.failures)
        patch_meta["syntax_preflight_last_error_type"] = str(self.last_error_type or "")
        patch_meta["syntax_preflight_last_error_msg"] = str(self.last_error_msg or "")
        patch_meta["syntax_preflight_events"] = list(self.events)
        attempt_metadata["syntax_preflight_enabled"] = bool(self.enabled)
        attempt_metadata["syntax_preflight_cycles_used"] = int(self.cycles_used)
        attempt_metadata["syntax_preflight_failures"] = int(self.failures)
        attempt_metadata["syntax_preflight_last_error_type"] = str(self.last_error_type or "")
        attempt_metadata["syntax_preflight_last_error_msg"] = str(self.last_error_msg or "")


@dataclass(frozen=True)
class PatchApplicationResult:
    """Result of applying a candidate diff before syntax preflight."""

    apply_ok: bool
    apply_msg: str
    applied_diff_for_preflight: str
    file_backups: dict[str, bytes]


@dataclass(frozen=True)
class SyntaxPreflightDecision:
    """Decision returned after sanitize/preflight evaluation.

    Status values distinguish retry with feedback, ready-to-continue,
    fallthrough to later verification, and terminal failure.
    """

    status: str
    state: SyntaxPreflightState
    meta: dict[str, Any] = field(default_factory=dict)


def make_syntax_preflight_state(*, enabled: bool, max_cycles: int) -> SyntaxPreflightState:
    """Construct normalized syntax-preflight state with a minimum of one cycle."""
    return SyntaxPreflightState(
        enabled=bool(enabled),
        max_cycles=max(1, int(max_cycles)),
    )


def apply_diff_with_optional_recovery(
    *,
    workdir: Path,
    attempt_dir: Path,
    diff_to_apply: str,
    file_index: str,
    attempt: int,
    runner_force_patch_attempt_enabled: bool,
    patch_meta: dict[str, Any],
    apply_diff_fn=None,
) -> PatchApplicationResult:
    """Apply a diff and optionally try runner-owned recovery when it fails.

    When ``runner_force_patch_attempt_enabled`` is on, a failed initial apply
    may trigger a synthetic recovery diff that records its own metadata and can
    replace the original diff for later syntax-preflight checks.
    """
    if apply_diff_fn is None:
        apply_diff_fn = apply_diff
    file_backups = _backup_diff_target_files(cwd=workdir, diff=diff_to_apply)
    applied_diff_for_preflight = diff_to_apply
    apply_ok, apply_msg = apply_diff_fn(workdir, diff_to_apply)
    if not apply_ok and runner_force_patch_attempt_enabled:
        patch_meta["initial_apply_error"] = apply_msg
        recovery_patch = _build_runner_forced_attempt_diff(
            cwd=workdir,
            file_index=file_index,
            attempt=attempt,
        )
        if recovery_patch is not None:
            recovery_target, recovery_diff = recovery_patch
            (attempt_dir / "patch_recovery.diff").write_text(recovery_diff, encoding="utf-8")
            recovery_metrics = _compute_patch_path_metrics(recovery_diff)
            recovery_backups = _backup_diff_target_files(cwd=workdir, diff=recovery_diff)
            for rel, payload in recovery_backups.items():
                file_backups.setdefault(rel, payload)
            recovery_ok, recovery_msg = apply_diff_fn(workdir, recovery_diff)
            patch_meta["runner_forced_patch_recovery"] = True
            patch_meta["runner_forced_patch_recovery_target"] = recovery_target
            patch_meta["runner_forced_patch_recovery_reason"] = "patch_apply_failed"
            patch_meta["runner_forced_patch_recovery_touches_tests"] = bool(
                recovery_metrics["patch_touches_tests"]
            )
            patch_meta["runner_forced_patch_recovery_file_count"] = int(
                recovery_metrics["patch_file_count"]
            )
            patch_meta["runner_forced_patch_recovery_test_file_count"] = int(
                recovery_metrics["patch_test_file_count"]
            )
            patch_meta["runner_forced_patch_recovery_file_paths"] = list(
                recovery_metrics["patch_file_paths"]
            )
            if recovery_ok:
                apply_ok = True
                apply_msg = recovery_msg
                applied_diff_for_preflight = recovery_diff
            else:
                patch_meta["runner_forced_patch_recovery_error"] = recovery_msg
    return PatchApplicationResult(
        apply_ok=apply_ok,
        apply_msg=apply_msg,
        applied_diff_for_preflight=applied_diff_for_preflight,
        file_backups=file_backups,
    )


def evaluate_patch_sanitize(
    *,
    state: SyntaxPreflightState,
    syntax_cycle: int,
    diff_to_apply: str,
    patch_sanitize_enabled: bool,
) -> SyntaxPreflightDecision:
    """Run patch-sanitization checks and decide continue, retry, or fail."""
    if not patch_sanitize_enabled:
        return SyntaxPreflightDecision(status="continue", state=state)
    sanitize_ok, sanitize_meta = _run_patch_sanitization(diff=diff_to_apply)
    sanitize_reasons_raw = sanitize_meta.get("patch_sanitize_reasons")
    sanitize_reasons = (
        [str(item) for item in sanitize_reasons_raw]
        if isinstance(sanitize_reasons_raw, list)
        else []
    )
    sanitize_error_msg = str(sanitize_meta.get("patch_sanitize_error_msg") or "")
    if sanitize_ok:
        return SyntaxPreflightDecision(
            status="continue",
            state=state,
            meta={
                "patch_sanitize_passed": True,
                "patch_sanitize_rejected": False,
                "patch_sanitize_reasons": sanitize_reasons,
                "patch_sanitize_error_msg": sanitize_error_msg,
            },
        )
    sanitize_checked_files_raw = sanitize_meta.get("patch_sanitize_checked_files")
    sanitize_checked_files = (
        [str(item) for item in sanitize_checked_files_raw]
        if isinstance(sanitize_checked_files_raw, list)
        else []
    )
    sanitize_error_file = str(sanitize_meta.get("patch_sanitize_error_file") or "")
    state.events.append(
        {
            "cycle": syntax_cycle,
            "passed": False,
            "syntax_preflight_checked_files": sanitize_checked_files,
            "syntax_preflight_checked_count": int(
                sanitize_meta.get("patch_sanitize_checked_count") or len(sanitize_checked_files)
            ),
            "syntax_preflight_error_file": sanitize_error_file,
            "syntax_preflight_error_type": "PatchSanitizationError",
            "syntax_preflight_error_msg": sanitize_error_msg,
        }
    )
    state.failures += 1
    state.last_error_type = "PatchSanitizationError"
    state.last_error_msg = sanitize_error_msg
    if syntax_cycle < state.max_cycles:
        state.retry_feedback = _render_patch_sanitize_retry_feedback(
            cycle=syntax_cycle,
            max_cycles=state.max_cycles,
            sanitize_error=sanitize_error_msg,
        )
        return SyntaxPreflightDecision(
            status="retry",
            state=state,
            meta={
                "patch_sanitize_passed": False,
                "patch_sanitize_rejected": True,
                "patch_sanitize_reasons": sanitize_reasons,
                "patch_sanitize_error_msg": sanitize_error_msg,
                "patch_sanitize_checked_files": sanitize_checked_files,
                "patch_sanitize_error_file": sanitize_error_file,
            },
        )
    return SyntaxPreflightDecision(
        status="terminal_failed",
        state=state,
        meta={
            "patch_sanitize_passed": False,
            "patch_sanitize_rejected": True,
            "patch_sanitize_reasons": sanitize_reasons,
            "patch_sanitize_error_msg": sanitize_error_msg,
            "patch_sanitize_checked_files": sanitize_checked_files,
            "patch_sanitize_error_file": sanitize_error_file,
        },
    )


def evaluate_syntax_preflight(
    *,
    state: SyntaxPreflightState,
    syntax_cycle: int,
    workdir: Path,
    diff: str,
    file_backups: dict[str, bytes],
    syntax_preflight_fallthrough: bool,
) -> SyntaxPreflightDecision:
    """Run syntax preflight on the applied diff and choose the next action.

    On retryable failures the original files are restored before returning
    ``retry``. Exhausted failures return ``terminal_failed`` unless
    ``syntax_preflight_fallthrough`` keeps the attempt alive for downstream
    verification.
    """
    if not state.enabled:
        state.events.append(
            {
                "cycle": syntax_cycle,
                "passed": True,
                "syntax_preflight_checked_files": [],
                "syntax_preflight_checked_count": 0,
                "syntax_preflight_error_file": "",
                "syntax_preflight_error_type": "",
                "syntax_preflight_error_msg": "",
            }
        )
        return SyntaxPreflightDecision(status="ready", state=state)

    syntax_preflight_passed, syntax_preflight_meta = _run_syntax_preflight(
        cwd=workdir,
        diff=diff,
    )
    state.events.append(
        {
            "cycle": syntax_cycle,
            "passed": bool(syntax_preflight_passed),
            **syntax_preflight_meta,
        }
    )
    if syntax_preflight_passed:
        return SyntaxPreflightDecision(status="ready", state=state)

    state.failures += 1
    state.last_error_type = str(syntax_preflight_meta.get("syntax_preflight_error_type") or "")
    state.last_error_msg = str(syntax_preflight_meta.get("syntax_preflight_error_msg") or "")
    if syntax_cycle < state.max_cycles:
        _restore_diff_target_files(cwd=workdir, backups=file_backups)
        state.retry_feedback = _render_syntax_preflight_retry_feedback(
            cycle=syntax_cycle,
            max_cycles=state.max_cycles,
            error_file=str(syntax_preflight_meta.get("syntax_preflight_error_file") or ""),
            error_type=state.last_error_type,
            error_msg=state.last_error_msg,
        )
        return SyntaxPreflightDecision(status="retry", state=state)
    if syntax_preflight_fallthrough:
        return SyntaxPreflightDecision(status="fallthrough", state=state)
    _restore_diff_target_files(cwd=workdir, backups=file_backups)
    return SyntaxPreflightDecision(status="terminal_failed", state=state)


__all__ = [
    "PatchApplicationResult",
    "SyntaxPreflightDecision",
    "SyntaxPreflightState",
    "apply_diff_with_optional_recovery",
    "evaluate_patch_sanitize",
    "evaluate_syntax_preflight",
    "make_syntax_preflight_state",
]
