from __future__ import annotations

from pathlib import Path

from evals import attempt_verification as verification


def test_evaluate_patch_sanitize_retries_with_feedback(monkeypatch) -> None:
    monkeypatch.setattr(
        verification,
        "_run_patch_sanitization",
        lambda diff: (
            False,
            {
                "patch_sanitize_reasons": ["escaped newline marker"],
                "patch_sanitize_error_msg": "escaped newline marker",
                "patch_sanitize_checked_files": ["buggy.py"],
                "patch_sanitize_error_file": "buggy.py",
            },
        ),
    )

    state = verification.make_syntax_preflight_state(enabled=True, max_cycles=2)
    decision = verification.evaluate_patch_sanitize(
        state=state,
        syntax_cycle=1,
        diff_to_apply="diff --git a/buggy.py b/buggy.py",
        patch_sanitize_enabled=True,
    )

    assert decision.status == "retry"
    assert decision.meta["patch_sanitize_rejected"] is True
    assert state.failures == 1
    assert state.last_error_type == "PatchSanitizationError"
    assert "malformed" in state.retry_feedback


def test_apply_diff_with_optional_recovery_uses_recovery_patch(tmp_path: Path, monkeypatch) -> None:
    apply_calls = {"count": 0}

    def _fake_apply_diff(_workdir, diff: str):  # type: ignore[no-untyped-def]
        apply_calls["count"] += 1
        return (False, "initial failure") if "recovery" not in diff else (True, "applied recovery")

    monkeypatch.setattr(verification, "apply_diff", _fake_apply_diff)
    monkeypatch.setattr(
        verification,
        "_backup_diff_target_files",
        lambda cwd, diff: {"buggy.py": b"before"},
    )
    monkeypatch.setattr(
        verification,
        "_build_runner_forced_attempt_diff",
        lambda cwd, file_index, attempt: (
            "buggy.py",
            "diff --git a/buggy.py b/buggy.py\n+recovery\n",
        ),
    )
    monkeypatch.setattr(
        verification,
        "_compute_patch_path_metrics",
        lambda diff: {
            "patch_touches_tests": False,
            "patch_file_count": 1,
            "patch_test_file_count": 0,
            "patch_file_paths": ["buggy.py"],
        },
    )

    patch_meta: dict[str, object] = {}
    result = verification.apply_diff_with_optional_recovery(
        workdir=tmp_path,
        attempt_dir=tmp_path,
        diff_to_apply="diff --git a/buggy.py b/buggy.py\n-bad\n+still bad\n",
        file_index="buggy.py",
        attempt=2,
        runner_force_patch_attempt_enabled=True,
        patch_meta=patch_meta,
    )

    assert apply_calls["count"] == 2
    assert result.apply_ok is True
    assert result.applied_diff_for_preflight.endswith("+recovery\n")
    assert patch_meta["runner_forced_patch_recovery"] is True
    assert (tmp_path / "patch_recovery.diff").is_file()


def test_evaluate_syntax_preflight_fallthrough_skips_restore(
    tmp_path: Path,
    monkeypatch,
) -> None:
    restore_calls = {"count": 0}

    monkeypatch.setattr(
        verification,
        "_run_syntax_preflight",
        lambda cwd, diff: (
            False,
            {
                "syntax_preflight_error_file": "buggy.py",
                "syntax_preflight_error_type": "SyntaxError",
                "syntax_preflight_error_msg": "invalid syntax",
                "syntax_preflight_checked_files": ["buggy.py"],
                "syntax_preflight_checked_count": 1,
            },
        ),
    )
    monkeypatch.setattr(
        verification,
        "_restore_diff_target_files",
        lambda cwd, backups: restore_calls.__setitem__("count", restore_calls["count"] + 1),
    )

    state = verification.make_syntax_preflight_state(enabled=True, max_cycles=1)
    decision = verification.evaluate_syntax_preflight(
        state=state,
        syntax_cycle=1,
        workdir=tmp_path,
        diff="diff --git a/buggy.py b/buggy.py",
        file_backups={"buggy.py": b"before"},
        syntax_preflight_fallthrough=True,
    )

    assert decision.status == "fallthrough"
    assert state.failures == 1
    assert state.last_error_type == "SyntaxError"
    assert restore_calls["count"] == 0


def test_make_syntax_preflight_state_clamps_max_cycles() -> None:
    state = verification.make_syntax_preflight_state(enabled=False, max_cycles=0)

    assert state.enabled is False
    assert state.max_cycles == 1


def test_syntax_preflight_state_apply_to_metadata_copies_fields() -> None:
    state = verification.SyntaxPreflightState(
        enabled=True,
        max_cycles=2,
        cycles_used=1,
        failures=1,
        last_error_type="SyntaxError",
        last_error_msg="bad syntax",
        events=[{"cycle": 1, "passed": False}],
    )
    attempt_metadata: dict[str, object] = {}
    patch_meta: dict[str, object] = {}

    state.apply_to_metadata(attempt_metadata=attempt_metadata, patch_meta=patch_meta)

    assert attempt_metadata["syntax_preflight_enabled"] is True
    assert attempt_metadata["syntax_preflight_cycles_used"] == 1
    assert attempt_metadata["syntax_preflight_failures"] == 1
    assert patch_meta["syntax_preflight_last_error_type"] == "SyntaxError"
    assert patch_meta["syntax_preflight_last_error_msg"] == "bad syntax"
    assert patch_meta["syntax_preflight_events"] == [{"cycle": 1, "passed": False}]


def test_apply_diff_with_optional_recovery_skips_recovery_when_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    apply_calls = {"count": 0}

    def _fake_apply_diff(_workdir, diff: str):  # type: ignore[no-untyped-def]
        apply_calls["count"] += 1
        return False, f"failed:{diff}"

    monkeypatch.setattr(verification, "apply_diff", _fake_apply_diff)
    monkeypatch.setattr(
        verification,
        "_backup_diff_target_files",
        lambda cwd, diff: {"buggy.py": b"before"},
    )

    patch_meta: dict[str, object] = {}
    result = verification.apply_diff_with_optional_recovery(
        workdir=tmp_path,
        attempt_dir=tmp_path,
        diff_to_apply="diff --git a/buggy.py b/buggy.py\n-bad\n+still bad\n",
        file_index="buggy.py",
        attempt=1,
        runner_force_patch_attempt_enabled=False,
        patch_meta=patch_meta,
    )

    assert apply_calls["count"] == 1
    assert result.apply_ok is False
    assert result.apply_msg.startswith("failed:")
    assert "runner_forced_patch_recovery" not in patch_meta


def test_apply_diff_with_optional_recovery_handles_missing_recovery_patch(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        verification, "apply_diff", lambda *_args, **_kwargs: (False, "initial fail")
    )
    monkeypatch.setattr(
        verification,
        "_backup_diff_target_files",
        lambda cwd, diff: {"buggy.py": b"before"},
    )
    monkeypatch.setattr(
        verification,
        "_build_runner_forced_attempt_diff",
        lambda cwd, file_index, attempt: None,
    )

    patch_meta: dict[str, object] = {}
    result = verification.apply_diff_with_optional_recovery(
        workdir=tmp_path,
        attempt_dir=tmp_path,
        diff_to_apply="diff --git a/buggy.py b/buggy.py\n-bad\n+still bad\n",
        file_index="buggy.py",
        attempt=1,
        runner_force_patch_attempt_enabled=True,
        patch_meta=patch_meta,
    )

    assert result.apply_ok is False
    assert result.apply_msg == "initial fail"
    assert patch_meta["initial_apply_error"] == "initial fail"
    assert "runner_forced_patch_recovery" not in patch_meta


def test_apply_diff_with_optional_recovery_records_failed_recovery_patch(
    tmp_path: Path, monkeypatch
) -> None:
    apply_calls = {"count": 0}

    def _fake_apply_diff(_workdir, diff: str):  # type: ignore[no-untyped-def]
        apply_calls["count"] += 1
        return False, "still failing"

    monkeypatch.setattr(verification, "apply_diff", _fake_apply_diff)
    monkeypatch.setattr(
        verification,
        "_backup_diff_target_files",
        lambda cwd, diff: {"buggy.py": b"before"},
    )
    monkeypatch.setattr(
        verification,
        "_build_runner_forced_attempt_diff",
        lambda cwd, file_index, attempt: (
            "buggy.py",
            "diff --git a/buggy.py b/buggy.py\n+recovery\n",
        ),
    )
    monkeypatch.setattr(
        verification,
        "_compute_patch_path_metrics",
        lambda diff: {
            "patch_touches_tests": False,
            "patch_file_count": 1,
            "patch_test_file_count": 0,
            "patch_file_paths": ["buggy.py"],
        },
    )

    patch_meta: dict[str, object] = {}
    result = verification.apply_diff_with_optional_recovery(
        workdir=tmp_path,
        attempt_dir=tmp_path,
        diff_to_apply="diff --git a/buggy.py b/buggy.py\n-bad\n+still bad\n",
        file_index="buggy.py",
        attempt=3,
        runner_force_patch_attempt_enabled=True,
        patch_meta=patch_meta,
    )

    assert apply_calls["count"] == 2
    assert result.apply_ok is False
    assert patch_meta["runner_forced_patch_recovery"] is True
    assert patch_meta["runner_forced_patch_recovery_error"] == "still failing"


def test_evaluate_patch_sanitize_disabled_returns_continue() -> None:
    state = verification.make_syntax_preflight_state(enabled=True, max_cycles=2)

    decision = verification.evaluate_patch_sanitize(
        state=state,
        syntax_cycle=1,
        diff_to_apply="diff --git a/buggy.py b/buggy.py",
        patch_sanitize_enabled=False,
    )

    assert decision.status == "continue"
    assert decision.meta == {}


def test_evaluate_patch_sanitize_passes_without_mutating_state(monkeypatch) -> None:
    monkeypatch.setattr(
        verification,
        "_run_patch_sanitization",
        lambda diff: (
            True,
            {
                "patch_sanitize_reasons": [],
                "patch_sanitize_error_msg": "",
            },
        ),
    )
    state = verification.make_syntax_preflight_state(enabled=True, max_cycles=2)

    decision = verification.evaluate_patch_sanitize(
        state=state,
        syntax_cycle=1,
        diff_to_apply="diff --git a/buggy.py b/buggy.py",
        patch_sanitize_enabled=True,
    )

    assert decision.status == "continue"
    assert decision.meta["patch_sanitize_passed"] is True
    assert state.failures == 0


def test_evaluate_patch_sanitize_terminal_failure_without_retry(monkeypatch) -> None:
    monkeypatch.setattr(
        verification,
        "_run_patch_sanitization",
        lambda diff: (
            False,
            {
                "patch_sanitize_reasons": ["escaped newline marker"],
                "patch_sanitize_error_msg": "escaped newline marker",
                "patch_sanitize_checked_files": ["buggy.py"],
                "patch_sanitize_error_file": "buggy.py",
            },
        ),
    )
    state = verification.make_syntax_preflight_state(enabled=True, max_cycles=1)

    decision = verification.evaluate_patch_sanitize(
        state=state,
        syntax_cycle=1,
        diff_to_apply="diff --git a/buggy.py b/buggy.py",
        patch_sanitize_enabled=True,
    )

    assert decision.status == "terminal_failed"
    assert state.failures == 1
    assert state.retry_feedback == ""


def test_evaluate_syntax_preflight_disabled_returns_ready(tmp_path: Path) -> None:
    state = verification.make_syntax_preflight_state(enabled=False, max_cycles=2)

    decision = verification.evaluate_syntax_preflight(
        state=state,
        syntax_cycle=1,
        workdir=tmp_path,
        diff="diff --git a/buggy.py b/buggy.py",
        file_backups={},
        syntax_preflight_fallthrough=False,
    )

    assert decision.status == "ready"
    assert state.events == [
        {
            "cycle": 1,
            "passed": True,
            "syntax_preflight_checked_files": [],
            "syntax_preflight_checked_count": 0,
            "syntax_preflight_error_file": "",
            "syntax_preflight_error_type": "",
            "syntax_preflight_error_msg": "",
        }
    ]


def test_evaluate_syntax_preflight_passes_without_restore(tmp_path: Path, monkeypatch) -> None:
    restore_calls = {"count": 0}
    monkeypatch.setattr(
        verification,
        "_run_syntax_preflight",
        lambda cwd, diff: (
            True,
            {
                "syntax_preflight_checked_files": ["buggy.py"],
                "syntax_preflight_checked_count": 1,
                "syntax_preflight_error_file": "",
                "syntax_preflight_error_type": "",
                "syntax_preflight_error_msg": "",
            },
        ),
    )
    monkeypatch.setattr(
        verification,
        "_restore_diff_target_files",
        lambda cwd, backups: restore_calls.__setitem__("count", restore_calls["count"] + 1),
    )
    state = verification.make_syntax_preflight_state(enabled=True, max_cycles=2)

    decision = verification.evaluate_syntax_preflight(
        state=state,
        syntax_cycle=1,
        workdir=tmp_path,
        diff="diff --git a/buggy.py b/buggy.py",
        file_backups={"buggy.py": b"before"},
        syntax_preflight_fallthrough=False,
    )

    assert decision.status == "ready"
    assert state.failures == 0
    assert restore_calls["count"] == 0


def test_evaluate_syntax_preflight_retries_and_restores(tmp_path: Path, monkeypatch) -> None:
    restore_calls = {"count": 0}
    monkeypatch.setattr(
        verification,
        "_run_syntax_preflight",
        lambda cwd, diff: (
            False,
            {
                "syntax_preflight_error_file": "buggy.py",
                "syntax_preflight_error_type": "SyntaxError",
                "syntax_preflight_error_msg": "invalid syntax",
                "syntax_preflight_checked_files": ["buggy.py"],
                "syntax_preflight_checked_count": 1,
            },
        ),
    )
    monkeypatch.setattr(
        verification,
        "_restore_diff_target_files",
        lambda cwd, backups: restore_calls.__setitem__("count", restore_calls["count"] + 1),
    )
    state = verification.make_syntax_preflight_state(enabled=True, max_cycles=2)

    decision = verification.evaluate_syntax_preflight(
        state=state,
        syntax_cycle=1,
        workdir=tmp_path,
        diff="diff --git a/buggy.py b/buggy.py",
        file_backups={"buggy.py": b"before"},
        syntax_preflight_fallthrough=False,
    )

    assert decision.status == "retry"
    assert state.failures == 1
    assert state.last_error_type == "SyntaxError"
    assert "Retry request" in state.retry_feedback
    assert restore_calls["count"] == 1


def test_evaluate_syntax_preflight_terminal_failure_restores(tmp_path: Path, monkeypatch) -> None:
    restore_calls = {"count": 0}
    monkeypatch.setattr(
        verification,
        "_run_syntax_preflight",
        lambda cwd, diff: (
            False,
            {
                "syntax_preflight_error_file": "buggy.py",
                "syntax_preflight_error_type": "SyntaxError",
                "syntax_preflight_error_msg": "invalid syntax",
                "syntax_preflight_checked_files": ["buggy.py"],
                "syntax_preflight_checked_count": 1,
            },
        ),
    )
    monkeypatch.setattr(
        verification,
        "_restore_diff_target_files",
        lambda cwd, backups: restore_calls.__setitem__("count", restore_calls["count"] + 1),
    )
    state = verification.make_syntax_preflight_state(enabled=True, max_cycles=1)

    decision = verification.evaluate_syntax_preflight(
        state=state,
        syntax_cycle=1,
        workdir=tmp_path,
        diff="diff --git a/buggy.py b/buggy.py",
        file_backups={"buggy.py": b"before"},
        syntax_preflight_fallthrough=False,
    )

    assert decision.status == "terminal_failed"
    assert state.failures == 1
    assert restore_calls["count"] == 1
