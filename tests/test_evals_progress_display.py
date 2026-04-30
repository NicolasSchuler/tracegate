from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console as RichConsole

from evals import progress_display


class _NonTtyStream:
    def isatty(self) -> bool:
        return False


class _TtyStream:
    def isatty(self) -> bool:
        return True


class _FakeLive:
    def __init__(
        self,
        renderable=None,
        *,
        console=None,
        screen: bool = False,
        auto_refresh: bool = True,
        refresh_per_second: float = 4,
        transient: bool = False,
        redirect_stdout: bool = True,
        redirect_stderr: bool = True,
        vertical_overflow: str = "ellipsis",
        get_renderable=None,
    ) -> None:
        del screen, auto_refresh, transient, redirect_stdout, redirect_stderr, vertical_overflow
        self.renderable = renderable
        self.console = console
        self.refresh_per_second = refresh_per_second
        self.get_renderable = get_renderable
        self.start_calls = 0
        self.stop_calls = 0
        self.refresh_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def refresh(self) -> None:
        self.refresh_calls += 1


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        run_id="demo-run",
        config=["evals/configs/openai_adaptive.toml"],
        conditions="base_repair,layered_adaptive_gated",
        executor="local",
        jobs=4,
        k=2,
        keep_workdirs=False,
        patcher="openai",
        openai_api_key="secret-token",
        openai_model="gpt-5.4-mini",
        openai_base_url="http://127.0.0.1:12000/v1",
        openai_http2=True,
        openai_temperature=0.0,
        openai_max_tokens=2048,
        openai_response_mode="edits",
        openai_timeout_sec=120.0,
        openai_transport_max_retries=6,
        openai_transport_recovery=True,
        openai_transport_retry_backoff_base_sec=1.0,
        openai_transport_retry_backoff_max_sec=8.0,
        openai_transport_recovery_max_wait_sec=60.0,
        openai_transport_recovery_poll_sec=2.0,
        openai_response_mode_autoswitch="retry_to_diff",
        openai_auto_context_window=True,
        openai_max_propose_sec=300.0,
        context_protocol="staged",
        max_context_chars=200_000,
        no_files_context=False,
        context_stage_a_max_requests=6,
        context_stage_a_max_chars=120_000,
        context_compaction_mode="auto_overflow",
        context_compaction_scope="snapshot,pytest,files",
        context_compaction_max_input_chars=120_000,
        context_compaction_target_chars=18_000,
        context_compaction_max_calls=1,
        retry_prompt_mode="auto",
        retry_context_mode="history",
        retry_history_max_chars=48_000,
        prompt_retry_reflection=True,
        prompt_adaptive_stage_b_cot=True,
        prompt_repair_method_scaffold=False,
        prompt_retry_plan_then_patch=False,
        prompt_budget_total_chars=180_000,
        prompt_budget_retry_delta_chars=16_000,
        prompt_budget_stdout_chars=32_000,
        prompt_budget_snapshot_chars=32_000,
        prompt_budget_evidence_chars=64_000,
        prompt_budget_retry_history_chars=12_000,
        adaptive_gate_policy="aggressive",
        adaptive_max_helper_calls=2,
        adaptive_informativeness_pregate=True,
        adaptive_budget_policy_mode="on",
        adaptive_budget_policy_architecture="hierarchical_v1",
        adaptive_budget_policy_tiers="1.0,1.1,1.2,1.35,1.5",
        adaptive_online_policy_mode="shadow",
        adaptive_online_action_score_biases="search_only=-1.00,frame_only=-0.50",
        adaptive_online_state_path="evals/artifacts/policies/demo-run.json",
        record_prompt="hash",
        record_evidence="hash",
        record_deps="minimal",
        record_env=["OPENAI_BASE_URL"],
        output_format="json_compact",
        pytest_output_tail_lines=40,
        progress_heartbeat_sec=5.0,
        progress_verbosity="basic",
        baseline_name="",
        transport_fail_fast_streak=5,
        case_fail_fast_on_request_budget_exceeded=True,
        force_patch_attempt=False,
        syntax_preflight_enabled=True,
        syntax_preflight_max_cycles=2,
        syntax_preflight_fallthrough=True,
        patch_sanitize_enabled=True,
        patch_verifier_scope="adaptive_only",
        patch_verifier_min_score=2,
        patch_verifier_infer_test_source_scope=True,
        container_image="llmdebug-eval-runner:latest",
        container_network="none",
        container_cpus=1.0,
        container_memory="1g",
        container_pids_limit=256,
        container_user="1000:1000",
        container_tmpfs_size_mb=128,
        container_mount_repo_src=True,
        container_env=["FOO=bar"],
        deterministic=False,
        seed=None,
        cases_dir="evals/cases",
        case=[],
        case_list_file=[],
        category=["logic_error"],
        difficulty=["easy", "hard"],
        snapshot_dependency=["helpful"],
        bug_source=[],
        max_cases=25,
    )


def _cases() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            case_id="case_a",
            category="hidden_state",
            difficulty="easy",
            snapshot_dependency="required",
        ),
        SimpleNamespace(
            case_id="case_b",
            category="logic_error",
            difficulty="hard",
            snapshot_dependency="helpful",
        ),
    ]


def test_build_startup_sections_groups_expected_content() -> None:
    sections = progress_display.build_startup_sections(
        args=_args(),
        cases=_cases(),
        conditions=[
            SimpleNamespace(value="base_repair"),
            SimpleNamespace(value="layered_adaptive_gated"),
        ],
        has_adaptive_condition=True,
        run_id="demo-run",
        results_path=Path("evals/results/demo-run.jsonl"),
        config_paths=["evals/configs/openai_adaptive.toml"],
        resolved_config_lock_path=Path("evals/artifacts/demo-run.lock.json"),
        total_units=4,
        skipped_units=1,
        remaining_units=3,
        execution_config=SimpleNamespace(executor="local"),
    )

    assert [section.title for section in sections] == [
        "Run",
        "Selection",
        "Execution",
        "Model / Patcher",
        "Prompting / Adaptive",
        "Artifacts",
        "Resolved Config / General",
        "Resolved Config / Selection Filters",
        "Resolved Config / Execution Backend",
        "Resolved Config / OpenAI",
        "Resolved Config / Prompting & Budgets",
        "Resolved Config / Adaptive",
        "Resolved Config / Patch & Verification",
        "Resolved Config / Runtime Controls",
        "Resolved Config / Recording & Progress",
    ]
    run_rows = dict(sections[0].rows)
    assert run_rows["Run ID"] == "demo-run"
    assert run_rows["Resume"] == "1 complete, 3 remaining, 4 total"
    selection_rows = dict(sections[1].rows)
    assert selection_rows["Categories"] == "hidden_state, logic_error"
    assert selection_rows["Snapshot dep"] == "helpful, required"
    execution_backend_rows = dict(sections[8].rows)
    assert execution_backend_rows["container_image"] == "llmdebug-eval-runner:latest"
    assert execution_backend_rows["container_env"] == "[FOO=bar]"
    openai_rows = dict(sections[9].rows)
    assert openai_rows["openai_api_key"] == "***REDACTED***"
    assert openai_rows["openai_http2"] == "true"
    prompting_rows = dict(sections[10].rows)
    assert prompting_rows["prompt_retry_reflection"] == "true"
    assert prompting_rows["context_compaction_mode"] == "auto_overflow"
    adaptive_rows = dict(sections[11].rows)
    assert adaptive_rows["adaptive_budget_policy_architecture"] == "hierarchical_v1"
    assert adaptive_rows["adaptive_online_state_path"] == "evals/artifacts/policies/demo-run.json"


def test_create_run_display_uses_plain_when_not_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    display = progress_display.create_run_display(
        total_units=3,
        verbosity="basic",
        stream=_NonTtyStream(),
    )
    assert isinstance(display, progress_display.PlainRunDisplay)


def test_create_run_display_respects_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    display = progress_display.create_run_display(
        total_units=3,
        verbosity="basic",
        stream=_TtyStream(),
    )
    assert isinstance(display, progress_display.PlainRunDisplay)


def test_display_unit_state_notes_include_wait_target_and_operation() -> None:
    state = progress_display.DisplayUnitState(case_id="case_a", condition="base_repair")
    state.start()
    state.apply_progress(
        "llm_request",
        {
            "attempt": 1,
            "max_attempts": 2,
            "phase_label": "llm: propose patch",
            "waiting_for": "openai_server",
            "external_operation": "propose_patch",
            "external_provider": "openai_server",
            "note": "propose patch [gpt-5.4-mini]",
        },
    )

    assert state.phase == "llm: propose patch"
    assert state.notes_text() == (
        "recs=0 | wait=openai_server | op=propose_patch | propose patch [gpt-5.4-mini]"
    )


def test_display_unit_state_clears_wait_metadata_on_new_phase() -> None:
    state = progress_display.DisplayUnitState(case_id="case_a", condition="base_repair")
    state.start()
    state.apply_progress(
        "llm_request",
        {
            "phase_label": "llm: propose patch",
            "owner": "openai",
            "source": "phase",
            "waiting_for": "openai_server",
            "external_operation": "propose_patch",
            "note": "propose patch [gpt-5.4-mini]",
        },
    )
    state.apply_progress(
        "patch_apply",
        {
            "phase_label": "harness: patch apply",
            "owner": "harness",
            "source": "phase",
            "note": "applying diff",
        },
    )

    assert state.phase == "harness: patch apply"
    assert state.notes_text() == "recs=0 | owner=harness | source=phase | applying diff"


def test_display_unit_state_same_phase_progress_preserves_phase_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr(progress_display.time, "monotonic", lambda: clock["now"])

    state = progress_display.DisplayUnitState(case_id="case_a", condition="base_repair")
    state.start()
    clock["now"] = 101.0
    state.apply_progress(
        "baseline_progress",
        {
            "phase_label": "baseline: tracecoder subprocess",
            "owner": "baseline",
            "source": "stdout",
            "activity_excerpt": "line 1",
        },
    )
    phase_started = state.phase_started_monotonic

    clock["now"] = 106.0
    state.apply_progress(
        "baseline_progress",
        {
            "phase_label": "baseline: tracecoder subprocess",
            "owner": "baseline",
            "source": "stdout",
            "activity_excerpt": "line 2",
        },
    )

    assert state.phase_started_monotonic == phase_started
    assert state.phase_elapsed_text() == "5s"
    assert state.activity_excerpt == "line 2"


def test_display_unit_state_heartbeat_does_not_reset_quiet_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr(progress_display.time, "monotonic", lambda: clock["now"])

    state = progress_display.DisplayUnitState(case_id="case_a", condition="base_repair")
    state.start()
    clock["now"] = 101.0
    state.apply_progress(
        "baseline_progress",
        {
            "phase_label": "baseline: tracecoder subprocess",
            "owner": "baseline",
            "source": "sidecar",
            "note": "status=running",
        },
    )

    clock["now"] = 106.0
    state.apply_progress(
        "heartbeat",
        {
            "phase": "baseline_progress",
            "phase_elapsed_sec": 5.0,
            "note": "status=running",
        },
    )

    assert state.activity_status_text() == "quiet 5s"


def test_display_unit_state_notes_render_owner_source_counters_and_excerpt_in_order() -> None:
    state = progress_display.DisplayUnitState(case_id="case_a", condition="base_repair")
    state.start()
    state.apply_progress(
        "baseline_progress",
        {
            "attempt": 1,
            "max_attempts": 2,
            "phase_label": "baseline: tracecoder subprocess",
            "owner": "baseline",
            "source": "sidecar",
            "waiting_for": "tracecoder_subprocess",
            "external_operation": "baseline_repair",
            "attempts_completed": 2,
            "llm_calls_observed": 3,
            "eval_duration_sec_observed": 1.25,
            "note": "status=running stopped=Debug cycle completed",
            "activity_excerpt": "TRACECODER step 2 finished",
        },
    )

    assert state.notes_text() == (
        "recs=0 | owner=baseline | source=sidecar | wait=tracecoder_subprocess | "
        "op=baseline_repair | iters=2 | llm_calls=3 | eval=1.2s | "
        "status=running stopped=Debug cycle completed | "
        "activity=TRACECODER step 2 finished"
    )


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_create_run_display_uses_rich_on_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    display = progress_display.create_run_display(
        total_units=3,
        verbosity="basic",
        stream=_TtyStream(),
    )
    assert isinstance(display, progress_display.RichRunDisplay)
    display.close()


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_uses_dynamic_live_renderable_and_spinner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    display = progress_display.RichRunDisplay(total_units=3, verbosity="basic")
    assert isinstance(display.live, _FakeLive)
    assert callable(display.live.get_renderable)
    assert isinstance(display.progress.columns[0], progress_display.SpinnerColumn)


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_advances_elapsed_and_eta_between_summary_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}

    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    monkeypatch.setattr(progress_display.time, "monotonic", lambda: clock["now"])

    display = progress_display.RichRunDisplay(total_units=4, verbosity="basic")
    display.start_runtime()
    display.update_summary(
        progress_display.DisplayRunSummary(
            total_units=4,
            completed_units=1,
            skipped_units=0,
            success_units=1,
            failed_units=0,
            runtime_error_units=0,
            elapsed_sec=10.0,
            eta_sec=50.0,
        )
    )

    clock["now"] = 103.5
    live_summary = display._render_summary()

    assert live_summary.elapsed_sec == pytest.approx(13.5)
    assert live_summary.eta_sec == pytest.approx(46.5)


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_keeps_final_summary_static_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 200.0}

    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    monkeypatch.setattr(progress_display.time, "monotonic", lambda: clock["now"])

    display = progress_display.RichRunDisplay(total_units=1, verbosity="basic")
    display.start_runtime()
    display.update_summary(
        progress_display.DisplayRunSummary(
            total_units=1,
            completed_units=1,
            skipped_units=0,
            success_units=1,
            failed_units=0,
            runtime_error_units=0,
            elapsed_sec=12.0,
            eta_sec=0.0,
        )
    )

    clock["now"] = 207.0
    live_summary = display._render_summary()

    assert live_summary.elapsed_sec == pytest.approx(12.0)
    assert live_summary.eta_sec == pytest.approx(0.0)


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_renders_activity_status_and_record_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 300.0}
    console = RichConsole(width=160)

    monkeypatch.setattr(progress_display, "Console", lambda **kwargs: console)
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    monkeypatch.setattr(progress_display.time, "monotonic", lambda: clock["now"])

    display = progress_display.RichRunDisplay(total_units=2, verbosity="basic")
    display.start_runtime()
    display.start_unit(case_id="case_a", condition="base_repair")
    display.update_unit_progress(
        case_id="case_a",
        condition="base_repair",
        event="attempt_propose_patch",
        payload={"attempt": 1, "max_attempts": 2, "note": "warming"},
    )
    display.update_unit_progress(
        case_id="case_a",
        condition="base_repair",
        event="heartbeat",
        payload={
            "attempt": 1,
            "max_attempts": 2,
            "phase": "attempt_propose_patch",
            "note": "warming",
            "records_observed": 2,
            "phase_elapsed_sec": 4.0,
        },
    )

    with console.capture() as capture:
        console.print(display._build_dashboard())
    text_running = capture.get()

    assert "Status" in text_running
    assert "running" in text_running
    assert "recs=2 | warming" in text_running

    clock["now"] = 307.0
    with console.capture() as capture:
        console.print(display._build_dashboard())
    text_quiet = capture.get()

    assert "quiet 7s" in text_quiet


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_renders_sidecar_progress_and_quiet_after_heartbeat_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 400.0}
    console = RichConsole(width=180)

    monkeypatch.setattr(progress_display, "Console", lambda **kwargs: console)
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    monkeypatch.setattr(progress_display.time, "monotonic", lambda: clock["now"])

    display = progress_display.RichRunDisplay(total_units=1, verbosity="detailed")
    display.start_runtime()
    display.start_unit(case_id="case_a", condition="tracecoder")
    display.update_unit_progress(
        case_id="case_a",
        condition="tracecoder",
        event="baseline_progress",
        payload={
            "attempt": 1,
            "max_attempts": 2,
            "phase_label": "baseline: tracecoder subprocess",
            "owner": "baseline",
            "source": "sidecar",
            "waiting_for": "tracecoder_subprocess",
            "external_operation": "baseline_repair",
            "attempts_completed": 1,
            "llm_calls_observed": 2,
            "eval_duration_sec_observed": 0.5,
            "note": "status=running",
            "activity_excerpt": "debug cycle checkpoint",
        },
    )

    with console.capture() as capture:
        console.print(display._build_dashboard())
    text_running = capture.get()

    assert "owner=baseline" in text_running
    assert "source=sidecar" in text_running
    assert "iters=1" in text_running
    assert "llm_calls=2" in text_running
    assert "activity=debug cycle checkpoint" in text_running

    clock["now"] = 406.0
    display.update_unit_progress(
        case_id="case_a",
        condition="tracecoder",
        event="heartbeat",
        payload={
            "phase": "baseline_progress",
            "phase_elapsed_sec": 6.0,
            "records_observed": 1,
        },
    )

    with console.capture() as capture:
        console.print(display._build_dashboard())
    text_quiet = capture.get()

    assert "quiet 6s" in text_quiet


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_releases_state_lock_before_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    display = progress_display.RichRunDisplay(total_units=2, verbosity="basic")
    display.start_runtime()

    original_refresh = display.live.refresh

    def _assert_refresh_without_state_lock() -> None:
        assert not display._state_lock.locked()
        original_refresh()

    display.live.refresh = _assert_refresh_without_state_lock

    display.update_summary(
        progress_display.DisplayRunSummary(
            total_units=2,
            completed_units=0,
            skipped_units=0,
            success_units=0,
            failed_units=0,
            runtime_error_units=0,
        )
    )
    display.start_unit(case_id="case_a", condition="base_repair")
    display.update_unit_progress(
        case_id="case_a",
        condition="base_repair",
        event="attempt_start",
        payload={"attempt": 1, "max_attempts": 2},
    )
    display.finish_unit(case_id="case_a", condition="base_repair", status="success")
    display.skip_unit(case_id="case_b", condition="base_repair")
    display.add_event(message="done", style="success")


# ---------------------------------------------------------------------------
# Lines 26-27: HAS_RICH = False when rich is not installed
# ---------------------------------------------------------------------------


def test_has_rich_false_makes_rich_display_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_display, "HAS_RICH", False)
    with pytest.raises(RuntimeError, match="RichRunDisplay requires rich"):
        progress_display.RichRunDisplay(total_units=1, verbosity="basic")


# ---------------------------------------------------------------------------
# Lines 34, 37-38: _stdout_is_tty - non-callable isatty / isatty raises
# ---------------------------------------------------------------------------


def test_stdout_is_tty_non_callable_isatty() -> None:
    stream = SimpleNamespace(isatty="not callable")
    assert progress_display._stdout_is_tty(stream) is False  # type: ignore[arg-type]


def test_stdout_is_tty_isatty_raises() -> None:
    class _RaisingStream:
        def isatty(self) -> bool:
            raise OSError("broken fd")

    assert progress_display._stdout_is_tty(_RaisingStream()) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Lines 48, 50: _format_duration - hours and minutes branches
# ---------------------------------------------------------------------------


def test_format_duration_hours() -> None:
    assert progress_display._format_duration(3661.0) == "1h01m01s"


def test_format_duration_minutes() -> None:
    assert progress_display._format_duration(125.0) == "2m05s"


# ---------------------------------------------------------------------------
# Lines 81, 85: _top_failure_reasons_text - non-empty Counter
# ---------------------------------------------------------------------------


def test_top_failure_reasons_text_non_empty() -> None:
    from collections import Counter

    reasons: Counter[str] = Counter({"timeout": 3, "crash": 1})
    result = progress_display._top_failure_reasons_text(reasons, top_n=2)
    # timeout has highest count, comes first
    assert "3" in result
    assert "1" in result


# ---------------------------------------------------------------------------
# Lines 99-100, 107: _format_resolved_value - dict and custom type
# ---------------------------------------------------------------------------


def test_format_resolved_value_dict() -> None:
    result = progress_display._format_resolved_value({"b": 2, "a": 1})
    assert result == "{a=1, b=2}"


def test_format_resolved_value_custom_type() -> None:
    class _Custom:
        def __str__(self) -> str:
            return "custom-repr"

    assert progress_display._format_resolved_value(_Custom()) == "custom-repr"


# ---------------------------------------------------------------------------
# Line 163: _resolved_config_section_title_for_key - unrecognized key
# ---------------------------------------------------------------------------


def test_resolved_config_section_title_unrecognized_key() -> None:
    assert (
        progress_display._resolved_config_section_title_for_key("zzz_unknown")
        == "Resolved Config / Other"
    )


# ---------------------------------------------------------------------------
# Line 219: DisplayRunSummary.progress_pct - total_units <= 0
# ---------------------------------------------------------------------------


def test_display_run_summary_progress_pct_zero_total() -> None:
    summary = progress_display.DisplayRunSummary(
        total_units=0,
        completed_units=0,
        skipped_units=0,
        success_units=0,
        failed_units=0,
        runtime_error_units=0,
    )
    assert summary.progress_pct == 100.0


# ---------------------------------------------------------------------------
# Line 316: DisplayUnitState.apply_progress - payload is None early return
# ---------------------------------------------------------------------------


def test_display_unit_state_apply_progress_none_payload() -> None:
    state = progress_display.DisplayUnitState(case_id="c", condition="cond")
    state.start()
    state.apply_progress("some_event", payload=None)
    # attempt should remain at its initial value after start (0)
    assert state.attempt == 0


# ---------------------------------------------------------------------------
# Line 349: record_emitted without records_observed → increment
# ---------------------------------------------------------------------------


def test_display_unit_state_record_emitted_auto_increment() -> None:
    state = progress_display.DisplayUnitState(case_id="c", condition="cond")
    state.start()
    state.apply_progress("record_emitted", payload={})
    assert state.observed_records == 1
    state.apply_progress("record_emitted", payload={})
    assert state.observed_records == 2


# ---------------------------------------------------------------------------
# Line 353: phase_elapsed_text when phase_started_monotonic is None → "-"
# ---------------------------------------------------------------------------


def test_display_unit_state_phase_elapsed_text_none() -> None:
    state = progress_display.DisplayUnitState(case_id="c", condition="cond")
    # phase_started_monotonic is None before start()
    assert state.phase_elapsed_text() == "-"


# ---------------------------------------------------------------------------
# Lines 358, 360: activity_status_text - not running; running no progress
# ---------------------------------------------------------------------------


def test_display_unit_state_activity_status_text_not_running() -> None:
    state = progress_display.DisplayUnitState(case_id="c", condition="cond")
    state.status = "queued"
    assert state.activity_status_text() == "queued"


def test_display_unit_state_activity_status_text_running_no_progress() -> None:
    state = progress_display.DisplayUnitState(case_id="c", condition="cond")
    state.status = "running"
    state.last_meaningful_progress_monotonic = None
    assert state.activity_status_text() == "starting"


# ---------------------------------------------------------------------------
# Line 380: notes_text - provider set, wait_for empty → shows provider
# ---------------------------------------------------------------------------


def test_display_unit_state_notes_text_provider_without_wait() -> None:
    state = progress_display.DisplayUnitState(case_id="c", condition="cond")
    state.start()
    state.external_provider = "openai"
    state.waiting_for = ""
    notes = state.notes_text()
    assert "provider=openai" in notes
    assert "wait=" not in notes


# ---------------------------------------------------------------------------
# Lines 584-585, 591-592: RichRunDisplay._event_text - styled text
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_event_text_styles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    display = progress_display.RichRunDisplay(total_units=1, verbosity="basic")
    from rich.text import Text

    for style_name, expected_rich_style in (
        ("success", "bold green"),
        ("error", "bold red"),
        ("warning", "bold yellow"),
        ("skip", "bold cyan"),
        ("other", "bold white"),
    ):
        text = display._event_text("msg", style_name)
        assert isinstance(text, Text)
        assert str(text) == "msg"
        span = text._spans[0]
        assert span.style == expected_rich_style


# ---------------------------------------------------------------------------
# Lines 622-623: _build_active_units_panel(units=None) reads from state lock
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_build_active_units_panel_reads_from_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = RichConsole(width=160)
    monkeypatch.setattr(progress_display, "Console", lambda **kwargs: console)
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    display = progress_display.RichRunDisplay(total_units=1, verbosity="basic")
    display.start_unit(case_id="u1", condition="cond")

    with console.capture() as capture:
        console.print(display._build_active_units_panel())  # units=None
    rendered = capture.get()
    assert "u1" in rendered


# ---------------------------------------------------------------------------
# Lines 661-662, 666: _build_events_panel - reads from lock, non-empty events
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_build_events_panel_reads_from_state_and_renders_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = RichConsole(width=120)
    monkeypatch.setattr(progress_display, "Console", lambda **kwargs: console)
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    display = progress_display.RichRunDisplay(total_units=1, verbosity="basic")
    display.add_event(message="test event", style="info")

    # Call with events=None so it reads from internal deque under lock
    with console.capture() as capture:
        console.print(display._build_events_panel())  # events=None → lock path
    rendered = capture.get()
    assert "test event" in rendered

    # Also exercise the explicit non-empty events path (line 666)
    with console.capture() as capture:
        console.print(display._build_events_panel(events=[("explicit", "success")]))
    rendered2 = capture.get()
    assert "explicit" in rendered2


# ---------------------------------------------------------------------------
# Lines 705-725: render_startup - empty sections, featured + remaining
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_render_startup_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = RichConsole(width=120)
    monkeypatch.setattr(progress_display, "Console", lambda **kwargs: console)
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    display = progress_display.RichRunDisplay(total_units=1, verbosity="basic")

    # Empty sections → early return, no crash
    with console.capture() as capture:
        display.render_startup([])

    # The rendered output should be empty (early return for no sections)
    assert capture.get() == ""


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_render_startup_featured_and_remaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = RichConsole(width=120)
    monkeypatch.setattr(progress_display, "Console", lambda **kwargs: console)
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    display = progress_display.RichRunDisplay(total_units=1, verbosity="basic")

    sections = [
        progress_display.StartupSection(title="Featured", rows=(("k1", "v1"),)),
        progress_display.StartupSection(title="Sec2", rows=(("k2", "v2"),)),
        progress_display.StartupSection(title="Sec3", rows=(("k3", "v3"),)),
    ]
    with console.capture() as capture:
        display.render_startup(sections)
    rendered = capture.get()
    assert "Featured" in rendered
    assert "Sec2" in rendered
    assert "Sec3" in rendered


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_render_startup_treats_paths_as_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = RichConsole(width=120)
    monkeypatch.setattr(progress_display, "Console", lambda **kwargs: console)
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    display = progress_display.RichRunDisplay(total_units=1, verbosity="basic")

    sections = [
        progress_display.StartupSection(
            title="Resolved Config / Selection Filters",
            rows=(
                (
                    "case_list_file",
                    "[/repo/evals/artifacts/case_list_mounts/20260430T123633/001_cases.txt]",
                ),
            ),
        ),
    ]
    with console.capture() as capture:
        display.render_startup(sections)

    assert "001_cases.txt" in capture.get()


# ---------------------------------------------------------------------------
# Line 736: start_runtime called twice - second call returns early
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_start_runtime_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    display = progress_display.RichRunDisplay(total_units=1, verbosity="basic")
    display.start_runtime()
    assert isinstance(display.live, _FakeLive)
    first_start_calls = display.live.start_calls
    display.start_runtime()  # second call - should be a no-op
    assert display.live.start_calls == first_start_calls


# ---------------------------------------------------------------------------
# Lines 788-789: render_final_lines - prints each line
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_render_final_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = RichConsole(width=120)
    monkeypatch.setattr(progress_display, "Console", lambda **kwargs: console)
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    display = progress_display.RichRunDisplay(total_units=1, verbosity="basic")

    with console.capture() as capture:
        display.render_final_lines(["line-alpha", "line-beta"])
    rendered = capture.get()
    assert "line-alpha" in rendered
    assert "line-beta" in rendered


# ---------------------------------------------------------------------------
# Line 795: close() - live.stop() called when runtime was started
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_close_stops_live_when_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    display = progress_display.RichRunDisplay(total_units=1, verbosity="basic")
    display.start_runtime()
    assert display.live.stop_calls == 0
    display.close()
    assert display.live.stop_calls == 1


@pytest.mark.skipif(not progress_display.HAS_RICH, reason="rich not installed")
def test_rich_run_display_close_skips_stop_when_not_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_display, "Live", _FakeLive)
    display = progress_display.RichRunDisplay(total_units=1, verbosity="basic")
    display.close()
    assert display.live.stop_calls == 0
