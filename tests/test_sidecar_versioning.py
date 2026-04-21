"""Unit tests for sidecar schema versioning in adaptive_controller_packet."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from evals.adaptive_controller_packet import (
    _SCHEMA_VERSION,
    parse_controller_packet,
    write_adaptive_controller_sidecars,
)


class TestWriteSidecarsInjectsVersion:
    def test_sidecar_has_version_key(self, tmp_path: Path) -> None:
        sidecars = {"test_sidecar.json": {"key": "value"}}
        written = write_adaptive_controller_sidecars(context_dir=tmp_path, sidecars=sidecars)
        assert len(written) == 1
        content = json.loads(written[0].read_text(encoding="utf-8"))
        assert content["_sidecar_schema_version"] == _SCHEMA_VERSION
        assert content["key"] == "value"

    def test_version_does_not_overwrite_existing_keys(self, tmp_path: Path) -> None:
        sidecars = {"s.json": {"_sidecar_schema_version": "old", "x": 1}}
        written = write_adaptive_controller_sidecars(context_dir=tmp_path, sidecars=sidecars)
        content = json.loads(written[0].read_text(encoding="utf-8"))
        # The injected version should take precedence (it's first in dict merge).
        assert content["_sidecar_schema_version"] == _SCHEMA_VERSION


class TestParseControllerPacketVersionWarning:
    def _make_minimal_packet(self, *, schema_version: str = "v1") -> dict[str, Any]:
        """Build a minimal valid controller packet dict."""
        return {
            "schema_version": schema_version,
            "case_id": "test",
            "condition": "cond",
            "attempt": 1,
            "controller_state": {
                "schema_version": "v2",
                "helper_action_selected": "",
                "helper_action_executed": "",
                "helper_decision_reason": "",
                "helper_outcome": "",
                "helper_materialized_context": False,
                "helper_calls_used": 0,
                "helper_budget_max_calls": 0,
                "no_progress_streak": 0,
                "retry_prompt_mode": "",
                "retry_context_mode": "",
                "context_compaction_requested": False,
                "context_compaction_invoked": False,
                "context_compaction_success": False,
                "context_compaction_trigger": "",
                "prompt_budget": {},
            },
            "evidence_bundle": {
                "schema_version": "v1",
                "case_id": "test",
                "condition": "cond",
                "attempt": 1,
                "failure_signature": "",
                "selected_ids": [],
                "new_ids": [],
                "reused_ids": [],
            },
            "history_bundle": {
                "schema_version": "v1",
                "retry_context_mode": "",
                "attempt_turns": [],
                "retry_history_text": "",
                "retry_history_turns_included": 0,
                "retry_history_truncated": False,
                "previous_note": "",
                "previous_failure_signature": "",
                "previous_patch_verifier_rejected": False,
                "previous_patch_verifier_reasons": [],
            },
            "verifier_report": {
                "schema_version": "v1",
                "status": "",
                "scope": "",
                "reasons": [],
                "previous_timed_out": False,
            },
        }

    def test_no_warning_on_matching_version(self, caplog: pytest.LogCaptureFixture) -> None:
        packet = self._make_minimal_packet(schema_version=_SCHEMA_VERSION)
        with caplog.at_level(logging.WARNING, logger="evals.adaptive_controller_packet"):
            result = parse_controller_packet(packet)
        assert result is not None
        assert "best-effort" not in caplog.text

    def test_warning_on_mismatched_version(self, caplog: pytest.LogCaptureFixture) -> None:
        packet = self._make_minimal_packet(schema_version="v999")
        with caplog.at_level(logging.WARNING, logger="evals.adaptive_controller_packet"):
            result = parse_controller_packet(packet)
        # Should still parse (best-effort) but emit a warning.
        assert result is not None
        assert "best-effort" in caplog.text
        assert "v999" in caplog.text

    def test_no_warning_when_version_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        packet = self._make_minimal_packet(schema_version="")
        with caplog.at_level(logging.WARNING, logger="evals.adaptive_controller_packet"):
            result = parse_controller_packet(packet)
        # Empty version is not a mismatch — it's simply absent.
        assert result is not None
        assert "best-effort" not in caplog.text
