"""Tests for the bounded, leak-free decision state."""

from __future__ import annotations

import json
from dataclasses import replace

from evals.decision_model import schema, state
from evals.eval_types import PreGateResult

CASE_ID = "async_lock_reentrant_medium"
CONDITION = "layered_adaptive_llm"
RUN_ID = "run-2026-10-04-dm-shadow"
WORKDIR = f"/abs/evals/artifacts/{RUN_ID}/workdirs/{CASE_ID}/{CONDITION}"


def _snapshot() -> dict[str, object]:
    return {
        "exception": {"type": "KeyError", "message": "'user_id'"},
        "frames": [
            {
                "file": f"{WORKDIR}/service.py",
                "file_rel": "service.py",
                "line": 42,
                "function": "lookup",
                "code": "return users[key]",
                "locals": {"key": "'user_id'", "users": "{}"},
                "locals_meta": {"key": {"type": "str"}, "users": {"type": "dict"}},
            },
            {"file": "/usr/lib/python3/site.py", "file_rel": None, "line": 7, "function": "x"},
            {"file_rel": "test_service.py", "line": 9, "function": "test_lookup"},
        ],
        "pytest": {"nodeid": "test_service.py::test_lookup"},
    }


def _inputs(**overrides: object) -> state.DecisionStateInputs:
    paths, tokens = state.redact_terms_for_unit(
        workdirs=(WORKDIR, f"evals/artifacts/{RUN_ID}/workdirs/{CASE_ID}/{CONDITION}"),
        case_id=CASE_ID,
        condition=CONDITION,
        run_id=RUN_ID,
    )
    values: dict[str, object] = {
        "stdout": (
            f"{WORKDIR}/test_service.py:9: in test_lookup\n"
            "E   KeyError: 'user_id'\n"
            f"FAILED test_service.py::test_lookup - KeyError (see /workdirs/{CASE_ID}/{CONDITION})\n"
            "1 failed, 3 passed in 0.12s\n"
        ),
        "stderr": "",
        "snapshot": _snapshot(),
        "attempt": 1,
        "max_attempts": 2,
        "pregate": PreGateResult(
            should_skip=False,
            score=0.62,
            exception_type="KeyError",
            reason="KeyError_tierB",
            f_type_specificity=0.7,
            f_message_diagnostic=0.3,
            f_crash_in_user_code=1.0,
            f_assertion_quality=0.5,
            f_frame_depth=1.0,
        ),
        "helper_calls_remaining": 1,
        "redact_paths": paths,
        "redact_tokens": tokens,
    }
    values.update(overrides)
    return state.DecisionStateInputs(**values)  # type: ignore[arg-type]


def _build(inputs: state.DecisionStateInputs, profile: str = "evidence_v1") -> state.DecisionState:
    return state.build_decision_state(inputs, profile=schema.get_profile(profile))


def test_state_sections_are_ordered_and_describe_the_failure() -> None:
    built = _build(_inputs())
    payload = built.payload

    assert list(payload) == sorted(payload)  # numeric prefixes survive sort_keys
    assert payload["0_note"] == state.STATE_NOTE
    assert payload["1_failure"] == {
        "exception": "KeyError",
        "message": "'user_id'",
        "failing_test": "test_service.py::test_lookup",
        "failed_or_errored_tests": 1,
        "timed_out": False,
    }
    assert payload["2_crash_site"]["location"] == "service.py:42"
    assert payload["4_call_path"] == [
        "service.py:42 in lookup()",
        "test_service.py:9 in test_lookup()",
    ]
    assert payload["3_assertion"] == ["E   KeyError: 'user_id'"]
    assert payload["8_budget"]["helper_calls_left"] == 1
    assert payload["6_hints"]["traceback_informativeness"] == {
        "score": 0.62,
        "tier": "KeyError_tierB",
    }
    assert "7_history" not in payload  # attempt 1 has no history


def test_state_never_contains_case_identifiers_or_workdirs() -> None:
    built = _build(_inputs())

    for forbidden in (CASE_ID, CONDITION, RUN_ID, "/abs/evals", "/workdirs/"):
        assert forbidden not in built.rendered
    assert "./test_service.py:9" in built.payload["9_test_output_tail"]


def test_rendered_state_matches_clef_render_and_hash_is_stable() -> None:
    first = _build(_inputs())
    second = _build(_inputs())

    expected = json.dumps(first.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert first.rendered == expected
    assert first.sha256 == second.sha256
    assert first.chars == len(first.rendered)


def test_locals_only_shown_under_full_reveal() -> None:
    full = _build(_inputs(reveal_mode="full"))
    traceback_first = _build(_inputs(reveal_mode="traceback_first"))

    assert any("key (str)" in line for line in full.payload["5_locals"])
    assert "5_locals" not in traceback_first.payload


def test_hints_report_disabled_pregate_and_can_be_omitted() -> None:
    no_pregate = _build(_inputs(pregate=None))
    no_hints = _build(_inputs(), profile="evidence_v1_nohints")

    assert no_pregate.payload["6_hints"]["traceback_informativeness"] == "disabled"
    assert "6_hints" not in no_hints.payload


def test_history_section_on_later_attempts() -> None:
    built = _build(
        _inputs(
            attempt=2,
            previous_failure_context={
                "note": "tests_still_failing",
                "adaptive_progress_reason": "no_progress_v2",
                "adaptive_helper_outcome": "materialized",
                "patch_verifier_reasons": ["touched tests"],
            },
            retry_history_turns=("turn one", "turn two", "turn three"),
            failure_signature_repeat=True,
            last_helper_action="frame_only",
        )
    )

    history = built.payload["7_history"]
    assert history["previous_outcome"] == "tests_still_failing"
    assert history["same_failure_as_before"] is True
    assert history["recent_attempts"] == ["turn two", "turn three"]
    assert history["last_helper_action"] == "frame_only"


def test_state_without_snapshot_falls_back_to_test_output() -> None:
    built = _build(_inputs(snapshot=None))

    assert built.payload["1_failure"]["exception"] == "KeyError"
    assert built.payload["1_failure"]["failing_test"] == "test_service.py::test_lookup"
    assert "2_crash_site" not in built.payload
    assert built.payload["8_budget"]["debug_snapshot_available"] is False


def test_long_test_output_keeps_only_its_tail() -> None:
    built = _build(_inputs(stdout="x" * 50_000 + "\nE   KeyError: 'user_id'\n"))
    tail = built.payload["9_test_output_tail"]

    assert "earlier characters]…" in tail
    assert tail.rstrip().endswith("E   KeyError: 'user_id'")
    assert built.chars <= schema.get_profile("evidence_v1").max_state_chars


def test_oversized_state_is_trimmed_tail_first_and_keeps_core_sections() -> None:
    profile = replace(schema.get_profile("evidence_v1"), max_state_chars=1800)
    built = state.build_decision_state(
        _inputs(stdout="x" * 50_000 + "\nE   KeyError: 'user_id'\n"), profile=profile
    )

    assert built.chars <= 1800
    assert built.truncated_sections[0] == "9_test_output_tail"
    for core in ("0_note", "1_failure", "2_crash_site", "8_budget"):
        assert core in built.payload


def test_text_profile_renders_sections_as_plain_text() -> None:
    built = _build(_inputs(), profile="evidence_v1_text")

    assert isinstance(built.payload, str)
    assert built.rendered.startswith("## NOTE")
    assert "## FAILURE" in built.rendered
    assert CASE_ID not in built.rendered


def test_short_tokens_are_not_redacted() -> None:
    built = _build(_inputs(redact_tokens=("test",)))
    assert "test_service.py" in built.rendered


def test_machine_specific_paths_are_normalized() -> None:
    import sys
    from pathlib import Path

    home = str(Path.home())
    noisy = f"E   ImportError: {sys.base_prefix}/lib/x.py and {home}/projects/y.py\n"
    built = _build(_inputs(stdout=noisy, snapshot=None))

    assert home not in built.rendered
    assert "<python>/lib/x.py" in built.rendered or "<venv>/lib/x.py" in built.rendered
    assert "~/projects/y.py" in built.rendered


def test_hints_keep_one_hypothesis_per_pattern(monkeypatch) -> None:
    """CAP-16 changed llmdebug's default; the decision state keeps the earlier inputs."""
    seen: dict[str, object] = {}

    def _fake(snapshot, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(state, "generate_hypotheses", _fake)
    _build(_inputs())

    assert seen == {"max_per_pattern": 1}


def test_oversized_parameterized_node_id_stays_within_the_budget() -> None:
    """Parameterized node ids are data from the run and must not defeat the size bound."""
    node_id = "test_service.py::test_lookup[" + ("p" * 12_000) + "]"
    snapshot = _snapshot()
    snapshot["pytest"] = {"nodeid": node_id}
    profile = schema.get_profile("evidence_v1")
    built = _build(_inputs(snapshot=snapshot))

    failing_test = built.payload["1_failure"]["failing_test"]
    assert len(failing_test) < 300
    assert failing_test.startswith("test_service.py::test_lookup[p")
    assert "more characters]" in failing_test
    assert built.chars <= profile.max_state_chars


def test_a_tight_budget_clips_the_mandatory_failure_section_last() -> None:
    """Once every optional section is gone, the copied failure text is cut too."""
    snapshot = _snapshot()
    snapshot["exception"] = {"type": "KeyError", "message": "u" * 4_000}
    snapshot["pytest"] = {"nodeid": "test_service.py::test_lookup[" + "p" * 4_000 + "]"}
    profile = replace(schema.get_profile("evidence_v1"), max_state_chars=800)
    built = state.build_decision_state(
        _inputs(snapshot=snapshot, stdout="x" * 5_000 + "\nE   KeyError: 'user_id'\n"),
        profile=profile,
    )

    assert built.chars <= 800
    assert "1_failure" in built.truncated_sections
    assert "1_failure" in built.payload
