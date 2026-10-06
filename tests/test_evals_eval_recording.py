"""Tests for evals.eval_recording - focused on uncovered lines.

Covers:
  - _RunResultsLock.acquire / release edge-cases (fcntl missing, OSError, double-release)
  - _resolve_llmdebug_version (import failures, missing __version__)
  - _collect_dependency_versions (runtime mode, missing metadata, exception handling)
  - _build_patcher_config (openai / baseline / command branches, execution ValueError)
  - _build_run_metadata (empty env_capture path)
  - _materialize_attempt_metadata (timing/token breakdown merging, evidence/prompt full)
"""

from __future__ import annotations

import argparse
import builtins
import errno
import sys
import types
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evals import oracle_integrity
from evals.attempt_metadata import (
    AdaptiveAttemptMetadata,
    AttemptMetadataRequest,
    ContextCompactionMetadata,
    OpenAIAttemptMetadata,
    PromptBudgetMetadata,
    RetryAttemptMetadata,
    StageAAttemptMetadata,
    SyntaxPreflightMetadata,
    build_attempt_metadata_request_from_legacy_kwargs,
)
from evals.eval_recording import (
    _build_base_record,
    _build_model_metadata,
    _build_patcher_config,
    _build_run_metadata,
    _collect_allowed_env,
    _collect_dependency_versions,
    _default_run_metadata,
    _materialize_attempt_metadata,
    _materialize_attempt_metadata_request,
    _normalize_openai_base_url,
    _resolve_llmdebug_snapshot_schema_version,
    _resolve_llmdebug_version,
    _RunResultsLock,
    _SynchronizedJsonlWriter,
)
from evals.eval_types import BASELINE_CONDITION_BY_NAME, CaseSpec, Condition
from evals.oracle_integrity import oracle_policy_metadata
from evals.run_state import _build_run_settings_fingerprint

# ── Helpers ──────────────────────────────────────────────────────────────


def _minimal_patcher_args(**overrides: Any) -> argparse.Namespace:
    """Return a Namespace with all fields _build_patcher_config requires."""
    defaults: dict[str, Any] = {
        "patcher": "openai",
        "k": 3,
        "transport_fail_fast_streak": 5,
        "case_fail_fast_on_request_budget_exceeded": True,
        "output_format": "json",
        "force_patch_attempt": False,
        "deterministic": False,
        "seed": 42,
        "no_files_context": False,
        "max_context_chars": 10000,
        "context_protocol": "monolithic",
        "context_stage_a_max_requests": 1,
        "context_stage_a_max_chars": 5000,
        "retry_prompt_mode": "full",
        "retry_context_mode": "history",
        "retry_history_max_chars": 5000,
        "prompt_budget_total_chars": 50000,
        "prompt_budget_retry_delta_chars": 10000,
        "prompt_budget_stdout_chars": 5000,
        "prompt_budget_snapshot_chars": 5000,
        "prompt_budget_evidence_chars": 5000,
        "prompt_budget_retry_history_chars": 5000,
        "context_compaction_mode": "off",
        "context_compaction_scope": "all",
        "context_compaction_max_input_chars": 50000,
        "context_compaction_target_chars": 10000,
        "context_compaction_max_calls": 3,
        # openai-specific defaults
        "openai_base_url": "https://api.openai.com/v1",
        "openai_model": "gpt-4",
        "openai_http2": True,
        "openai_timeout_sec": 60.0,
        "openai_transport_max_retries": 3,
        "openai_transport_retry_backoff_base_sec": 1.0,
        "openai_transport_retry_backoff_max_sec": 30.0,
        "openai_max_propose_sec": 120.0,
        "openai_max_tokens": 4096,
        "openai_context_window_tokens": 128000,
        "openai_auto_context_window": False,
        "openai_temperature": 0.0,
        "openai_response_mode": "text",
        "openai_response_mode_autoswitch": "off",
        "openai_force_patch_attempt": False,
        # record fields
        "record_prompt": "hash",
        "record_evidence": "hash",
        "record_env": [],
        "record_deps": "none",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _run_metadata_args(**overrides: Any) -> argparse.Namespace:
    """Namespace suitable for _build_run_metadata."""
    return _minimal_patcher_args(**overrides)


# ── _RunResultsLock.acquire ──────────────────────────────────────────────


class TestRunResultsLockAcquire:
    """Lines 77-79, 98, 113-118."""

    def test_acquire_fcntl_not_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When fcntl is not importable, acquire returns (None, None, warning)."""
        results = tmp_path / "results.jsonl"
        results.touch()

        # Make `import fcntl` raise ModuleNotFoundError inside acquire
        real_import = builtins.__import__

        def _fake_import(name: str, *a: Any, **kw: Any) -> Any:
            if name == "fcntl":
                raise ModuleNotFoundError("No module named 'fcntl'")
            return real_import(name, *a, **kw)

        monkeypatch.setattr("builtins.__import__", _fake_import)

        lock, error, warning = _RunResultsLock.acquire(results_path=results, run_id="test-run")

        assert lock is None
        assert error is None
        assert warning is not None
        assert "missing fcntl" in warning

    def test_acquire_oserror_non_eagain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OSError with errno not in {11, 35} → (None, error_string, None)."""
        results = tmp_path / "results.jsonl"
        results.touch()

        fake_fcntl = types.ModuleType("fcntl")
        fake_fcntl.LOCK_EX = 2  # type: ignore[attr-defined]
        fake_fcntl.LOCK_NB = 4  # type: ignore[attr-defined]
        fake_fcntl.LOCK_UN = 8  # type: ignore[attr-defined]

        def _raise_oserror(*_a: Any, **_kw: Any) -> None:
            exc = OSError("bad fd")
            exc.errno = errno.EBADF  # 9, not in {11, 35}
            raise exc

        fake_fcntl.flock = _raise_oserror  # type: ignore[attr-defined]

        real_import = builtins.__import__

        def _fake_import(name: str, *a: Any, **kw: Any) -> Any:
            if name == "fcntl":
                return fake_fcntl
            return real_import(name, *a, **kw)

        monkeypatch.setattr("builtins.__import__", _fake_import)

        lock, error, warning = _RunResultsLock.acquire(results_path=results, run_id="test-run")

        assert lock is None
        assert error is not None
        assert "failed to acquire run lock" in error
        assert warning is None

    def test_acquire_oserror_eagain(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """OSError with errno 11 (EAGAIN) → (None, 'already held…', None)."""
        results = tmp_path / "results.jsonl"
        results.touch()

        fake_fcntl = types.ModuleType("fcntl")
        fake_fcntl.LOCK_EX = 2  # type: ignore[attr-defined]
        fake_fcntl.LOCK_NB = 4  # type: ignore[attr-defined]
        fake_fcntl.LOCK_UN = 8  # type: ignore[attr-defined]

        def _raise_eagain(*_a: Any, **_kw: Any) -> None:
            exc = OSError("resource temporarily unavailable")
            exc.errno = 11  # EAGAIN
            raise exc

        fake_fcntl.flock = _raise_eagain  # type: ignore[attr-defined]

        real_import = builtins.__import__

        def _fake_import(name: str, *a: Any, **kw: Any) -> Any:
            if name == "fcntl":
                return fake_fcntl
            return real_import(name, *a, **kw)

        monkeypatch.setattr("builtins.__import__", _fake_import)

        lock, error, warning = _RunResultsLock.acquire(results_path=results, run_id="run-x")

        assert lock is None
        assert error is not None
        assert "already held" in error
        assert warning is None

    def test_acquire_general_exception_reraises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """General exception after fcntl import → handle closed, exception re-raised."""
        results = tmp_path / "results.jsonl"
        results.touch()

        fake_fcntl = types.ModuleType("fcntl")
        fake_fcntl.LOCK_EX = 2  # type: ignore[attr-defined]
        fake_fcntl.LOCK_NB = 4  # type: ignore[attr-defined]
        fake_fcntl.LOCK_UN = 8  # type: ignore[attr-defined]

        def _raise_runtime(*_a: Any, **_kw: Any) -> None:
            raise RuntimeError("unexpected")

        fake_fcntl.flock = _raise_runtime  # type: ignore[attr-defined]

        real_import = builtins.__import__

        def _fake_import(name: str, *a: Any, **kw: Any) -> Any:
            if name == "fcntl":
                return fake_fcntl
            return real_import(name, *a, **kw)

        monkeypatch.setattr("builtins.__import__", _fake_import)

        with pytest.raises(RuntimeError, match="unexpected"):
            _RunResultsLock.acquire(results_path=results, run_id="test-run")

    def test_acquire_success(self, tmp_path: Path) -> None:
        """Successful acquire returns (lock, None, None)."""
        results = tmp_path / "results.jsonl"
        results.touch()

        lock, error, warning = _RunResultsLock.acquire(results_path=results, run_id="run-ok")

        # On macOS/Linux fcntl is available
        if lock is not None:
            assert error is None
            assert warning is None
            lock.release()
        else:
            # Windows-like: either warning (no fcntl) or error
            assert error is not None or warning is not None


# ── _RunResultsLock.release ──────────────────────────────────────────────


class TestRunResultsLockRelease:
    """Lines 123, 130-131, 136-137."""

    def test_double_release_is_noop(self, tmp_path: Path) -> None:
        """Calling release() twice doesn't error (line 123)."""
        lock_path = tmp_path / "results.jsonl.lock"
        handle = lock_path.open("w")
        lock = _RunResultsLock(path=lock_path, handle=handle)

        lock.release()
        # Second call should be a no-op
        lock.release()
        assert lock._released is True

    def test_release_swallows_flock_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exception from fcntl.flock(LOCK_UN) is swallowed (lines 130-131)."""
        lock_path = tmp_path / "results.jsonl.lock"
        handle = lock_path.open("w")
        lock = _RunResultsLock(path=lock_path, handle=handle)

        fake_fcntl = types.ModuleType("fcntl")
        fake_fcntl.LOCK_UN = 8  # type: ignore[attr-defined]

        def _raise_on_unlock(*_a: Any, **_kw: Any) -> None:
            raise OSError("unlock failed")

        fake_fcntl.flock = _raise_on_unlock  # type: ignore[attr-defined]

        real_import = builtins.__import__

        def _fake_import(name: str, *a: Any, **kw: Any) -> Any:
            if name == "fcntl":
                return fake_fcntl
            return real_import(name, *a, **kw)

        monkeypatch.setattr("builtins.__import__", _fake_import)

        # Should not raise
        lock.release()
        assert lock._released is True

    def test_release_swallows_unlink_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OSError from path.unlink() is swallowed (lines 136-137)."""
        lock_path = tmp_path / "results.jsonl.lock"
        handle = lock_path.open("w")
        lock = _RunResultsLock(path=lock_path, handle=handle)

        # Make unlink always raise
        monkeypatch.setattr(
            type(lock_path),
            "unlink",
            lambda self, **kw: (_ for _ in ()).throw(OSError("permission denied")),
        )

        lock.release()
        assert lock._released is True


# ── _resolve_llmdebug_version ────────────────────────────────────────────


class TestResolveLlmdebugVersion:
    """Lines 460-462."""

    def test_import_fails_returns_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When `import llmdebug` fails → 'unknown'."""
        real_import = builtins.__import__

        def _block_llmdebug(name: str, *a: Any, **kw: Any) -> Any:
            if name == "llmdebug":
                raise ImportError("no such module")
            return real_import(name, *a, **kw)

        monkeypatch.setattr("builtins.__import__", _block_llmdebug)
        assert _resolve_llmdebug_version() == "unknown"

    def test_version_is_none_returns_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When __version__ is None → 'unknown'."""
        fake = types.ModuleType("llmdebug")
        fake.__version__ = None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "llmdebug", fake)
        assert _resolve_llmdebug_version() == "unknown"

    def test_version_is_empty_returns_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When __version__ is '' → 'unknown'."""
        fake = types.ModuleType("llmdebug")
        fake.__version__ = ""  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "llmdebug", fake)
        assert _resolve_llmdebug_version() == "unknown"

    def test_version_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When __version__ is a real string → returns it."""
        fake = types.ModuleType("llmdebug")
        fake.__version__ = "1.2.3"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "llmdebug", fake)
        assert _resolve_llmdebug_version() == "1.2.3"


class TestResolveLlmdebugSnapshotSchemaVersion:
    """CAP-20: run metadata records the snapshot format the prompts were built from."""

    def test_matches_the_capture_schema(self) -> None:
        from llmdebug.capture import SCHEMA_VERSION

        assert _resolve_llmdebug_snapshot_schema_version() == SCHEMA_VERSION

    def test_import_fails_returns_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "llmdebug.capture", None)
        assert _resolve_llmdebug_snapshot_schema_version() == "unknown"

    def test_recorded_in_both_metadata_builders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "evals.eval_recording._resolve_llmdebug_snapshot_schema_version", lambda: "9.9"
        )
        monkeypatch.setattr("evals.eval_recording._collect_dependency_versions", lambda mode: {})
        built = _build_run_metadata(
            args=_run_metadata_args(record_env=[], record_deps="none"),
            patcher_name="openai",
            run_id="run-1",
            started_at_utc="2024-01-01T00:00:00Z",
            run_settings_fingerprint="abc123",
        )
        default = _default_run_metadata(run_id="r1", patcher_name="openai")

        assert built["llmdebug_snapshot_schema_version"] == "9.9"
        assert default["llmdebug_snapshot_schema_version"] == "9.9"


# ── _collect_dependency_versions ─────────────────────────────────────────


class TestCollectDependencyVersions:
    """Lines 484, 488-502, 520-521."""

    def test_mode_none_returns_empty(self) -> None:
        """mode='none' → {} (line 484)."""
        assert _collect_dependency_versions(mode="none") == {}

    def test_mode_runtime_collects_packages(self) -> None:
        """mode='runtime' → returns sorted dict of installed packages."""
        result = _collect_dependency_versions(mode="runtime")
        assert isinstance(result, dict)
        # pytest itself must appear
        assert "pytest" in result
        # sorted check
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_mode_runtime_skips_missing_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dist with missing name is skipped (lines 499-500)."""
        from importlib import metadata as importlib_metadata

        dist_good = MagicMock()
        dist_good.metadata.get.return_value = "fakepkg"
        dist_good.version = "1.0.0"

        dist_no_name = MagicMock()
        dist_no_name.metadata.get.return_value = None
        dist_no_name.version = "2.0.0"

        dist_no_version = MagicMock()
        dist_no_version.metadata.get.return_value = "nover"
        dist_no_version.version = ""

        monkeypatch.setattr(
            importlib_metadata,
            "distributions",
            lambda: [dist_good, dist_no_name, dist_no_version],
        )

        result = _collect_dependency_versions(mode="runtime")
        assert "fakepkg" in result
        assert result["fakepkg"] == "1.0.0"
        assert "nover" not in result

    def test_mode_runtime_skips_exception_in_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exception during dist.metadata access → skipped (line 497)."""
        from importlib import metadata as importlib_metadata

        dist_bad = MagicMock()
        dist_bad.metadata.get.side_effect = RuntimeError("broken")

        dist_ok = MagicMock()
        dist_ok.metadata.get.return_value = "okpkg"
        dist_ok.version = "3.0.0"

        monkeypatch.setattr(
            importlib_metadata,
            "distributions",
            lambda: [dist_bad, dist_ok],
        )

        result = _collect_dependency_versions(mode="runtime")
        assert "okpkg" in result

    def test_mode_default_general_exception_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default mode: general Exception (not PackageNotFoundError) → skipped."""
        from importlib import metadata as importlib_metadata

        def _boom(name: str) -> str:
            raise RuntimeError("metadata corrupt")

        monkeypatch.setattr(importlib_metadata, "version", _boom)
        result = _collect_dependency_versions(mode="default")
        assert result == {}


# ── _build_patcher_config ────────────────────────────────────────────────


class TestBuildPatcherConfig:
    """Lines 395-396, 437-438, 443-447."""

    def test_openai_patcher_includes_openai_config(self) -> None:
        """patcher='openai' → config has openai-specific keys."""
        args = _minimal_patcher_args(
            patcher="openai",
            adaptive_reward_success_hint=1.25,
            adaptive_reward_progress_scale=1.4,
            adaptive_reward_stagnation_scale=0.8,
            adaptive_reward_materialization_scale=0.6,
        )
        config = _build_patcher_config(args=args, patcher_name="openai")

        assert config["patcher"] == "openai"
        assert "openai_model" in config
        assert config["openai_model"] == "gpt-4"
        assert "openai_base_url" in config
        assert "openai_temperature" in config
        assert config["native_baseline_mode"] == ""
        assert config["adaptive_reward_success_hint"] == pytest.approx(1.25)
        assert config["adaptive_reward_progress_scale"] == pytest.approx(1.4)
        assert config["adaptive_reward_stagnation_scale"] == pytest.approx(0.8)
        assert config["adaptive_reward_materialization_scale"] == pytest.approx(0.6)

    def test_openai_patcher_includes_native_baseline_mode_when_present(self) -> None:
        args = _minimal_patcher_args(
            patcher="openai",
            native_baseline_mode="cot_repair",
        )

        config = _build_patcher_config(args=args, patcher_name="openai")

        assert config["patcher"] == "openai"
        assert config["native_baseline_mode"] == "cot_repair"

    def test_baseline_patcher_includes_baseline_config(self) -> None:
        """patcher='baseline' → config has baseline_name, baseline_repo_dir."""
        args = _minimal_patcher_args(
            patcher="baseline",
            baseline_name="random",
            baseline_repo_dir="/repo",
            baseline_python="python3",
        )
        config = _build_patcher_config(args=args, patcher_name="baseline")

        assert config["patcher"] == "baseline"
        assert config["baseline_name"] == "random"
        assert config["baseline_repo_dir"] == "/repo"
        assert config["baseline_python"] == "python3"
        assert "openai_model" not in config

    def test_command_patcher_includes_command_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """patcher='command' → config has patch_cmd, patch_cmd_shell, timeout."""
        monkeypatch.delenv("LLMDEBUG_EVAL_PATCH_CMD", raising=False)
        monkeypatch.delenv("LLMDEBUG_EVAL_PATCH_CMD_SHELL", raising=False)
        monkeypatch.delenv("LLMDEBUG_EVAL_PATCH_CMD_TIMEOUT_SEC", raising=False)

        args = _minimal_patcher_args(
            patcher="command",
            patch_cmd="echo hello",
            patch_cmd_shell=True,
            patch_cmd_timeout_sec=60.0,
        )
        config = _build_patcher_config(args=args, patcher_name="command")

        assert config["patcher"] == "command"
        assert config["patch_cmd"] == "echo hello"
        assert config["patch_cmd_shell"] is True
        assert config["patch_cmd_timeout_sec"] == 60.0
        assert "openai_model" not in config
        assert "baseline_name" not in config

    def test_command_patcher_env_timeout_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid LLMDEBUG_EVAL_PATCH_CMD_TIMEOUT_SEC → falls back to 30.0."""
        monkeypatch.setenv("LLMDEBUG_EVAL_PATCH_CMD_TIMEOUT_SEC", "not-a-number")
        monkeypatch.delenv("LLMDEBUG_EVAL_PATCH_CMD", raising=False)
        monkeypatch.delenv("LLMDEBUG_EVAL_PATCH_CMD_SHELL", raising=False)

        args = _minimal_patcher_args(
            patcher="command",
            patch_cmd="",
            patch_cmd_shell=False,
            patch_cmd_timeout_sec=0.0,
        )
        config = _build_patcher_config(args=args, patcher_name="command")
        assert config["patch_cmd_timeout_sec"] == 30.0

    def test_oracle_policy_is_recorded_and_fingerprinted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ATT-01: the oracle policy has no CLI knob, so the records carry its version.

        Two runs whose success criteria differ would otherwise share a settings
        fingerprint and be indistinguishable.
        """
        args = _minimal_patcher_args(patcher="openai", cases_dir="cases")
        config = _build_patcher_config(args=args, patcher_name="openai")
        assert config["oracle_policy"] == oracle_policy_metadata()
        assert config["oracle_policy"]["verdict_source"] == "junit_counts"
        assert config["oracle_policy"]["restore_test_surface_before_verification"] is True
        assert "conftest.py" in config["oracle_policy"]["test_surface_filenames"]

        fingerprint = _build_run_settings_fingerprint(args=args, cases=[], conditions=[])
        monkeypatch.setattr(oracle_integrity, "ORACLE_POLICY_VERSION", 2)
        assert _build_run_settings_fingerprint(args=args, cases=[], conditions=[]) != fingerprint

    def test_execution_config_valueerror_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ValueError from build_execution_config_from_args → swallowed."""
        args = _minimal_patcher_args(patcher="openai")
        monkeypatch.setattr(
            "evals.eval_recording.build_execution_config_from_args",
            lambda a: (_ for _ in ()).throw(ValueError("bad")),
        )
        config = _build_patcher_config(args=args, patcher_name="openai")
        assert "execution" not in config


# ── _build_run_metadata ──────────────────────────────────────────────────


class TestBuildRunMetadata:
    """Line 198: env_capture is empty → metadata['env'] not set."""

    def test_empty_env_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """record_env=[] → no 'env' key in metadata."""
        # Stub out version/deps to keep it simple
        monkeypatch.setattr(
            "evals.eval_recording._resolve_llmdebug_version",
            lambda: "0.0.0",
        )
        monkeypatch.setattr(
            "evals.eval_recording._collect_dependency_versions",
            lambda mode: {},
        )
        args = _run_metadata_args(record_env=[], record_deps="none")
        metadata = _build_run_metadata(
            args=args,
            patcher_name="openai",
            run_id="run-1",
            started_at_utc="2024-01-01T00:00:00Z",
            run_settings_fingerprint="abc123",
        )
        assert "env" not in metadata

    def test_nonempty_env_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """record_env=['FOO'] with FOO set → metadata['env'] populated."""
        monkeypatch.setenv("FOO", "bar")
        monkeypatch.setattr(
            "evals.eval_recording._resolve_llmdebug_version",
            lambda: "0.0.0",
        )
        monkeypatch.setattr(
            "evals.eval_recording._collect_dependency_versions",
            lambda mode: {},
        )
        args = _run_metadata_args(record_env=["FOO"], record_deps="none")
        metadata = _build_run_metadata(
            args=args,
            patcher_name="openai",
            run_id="run-2",
            started_at_utc="2024-01-01T00:00:00Z",
            run_settings_fingerprint="abc123",
        )
        assert metadata["env"] == {"FOO": "bar"}


# ── _materialize_attempt_metadata ────────────────────────────────────────


def test_run_eval_preserves_private_materializer_alias() -> None:
    from evals import run_eval

    assert run_eval._materialize_attempt_metadata is _materialize_attempt_metadata


class TestMaterializeAttemptMetadata:
    """Lines 1222-1233 (breakdown merge) and 1237-1249 (evidence/prompt full)."""

    @staticmethod
    def _call(
        tmp_path: Path,
        *,
        timing_breakdown: dict[str, float] | None = None,
        token_breakdown: dict[str, int] | None = None,
        record_evidence: str = "none",
        record_prompt: str = "none",
        evidence: dict[str, Any] | None = None,
        prompt: str = "",
    ) -> dict[str, Any]:
        """Thin wrapper calling _materialize_attempt_metadata with defaults."""
        attempt_dir = tmp_path / "attempt_1"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        return _materialize_attempt_metadata(
            attempt_dir=attempt_dir,
            attempt=1,
            max_attempts=3,
            evidence=evidence or {},
            prompt=prompt,
            record_prompt=record_prompt,
            record_evidence=record_evidence,
            context_protocol="monolithic",
            timing_breakdown=timing_breakdown,
            token_breakdown=token_breakdown,
        )

    def test_timing_breakdown_merge(self, tmp_path: Path) -> None:
        """timing_breakdown not None → merges into existing (lines 1222-1227)."""
        meta = self._call(
            tmp_path,
            timing_breakdown={
                "stage_b_llm_sec": 2.5,
                "nonexistent_key": 99.9,
            },
        )
        tb = meta["timing_breakdown"]
        assert tb["stage_b_llm_sec"] == 2.5
        # Non-matching key is NOT added
        assert "nonexistent_key" not in tb

    def test_timing_breakdown_none_leaves_defaults(self, tmp_path: Path) -> None:
        """timing_breakdown=None → defaults unchanged."""
        meta = self._call(tmp_path, timing_breakdown=None)
        assert meta["timing_breakdown"]["stage_b_llm_sec"] == 0.0

    def test_token_breakdown_merge(self, tmp_path: Path) -> None:
        """token_breakdown not None → merges into existing (lines 1228-1233)."""
        meta = self._call(
            tmp_path,
            token_breakdown={
                "stage_b_tokens_prompt": 500,
                "bogus_key": 999,
            },
        )
        tkb = meta["token_breakdown"]
        assert tkb["stage_b_tokens_prompt"] == 500
        # Non-matching key is NOT added
        assert "bogus_key" not in tkb

    def test_token_breakdown_none_leaves_defaults(self, tmp_path: Path) -> None:
        """token_breakdown=None → defaults unchanged."""
        meta = self._call(tmp_path, token_breakdown=None)
        assert meta["token_breakdown"]["stage_b_tokens_prompt"] == 0

    def test_record_evidence_full_writes_file(self, tmp_path: Path) -> None:
        """record_evidence='full' → evidence.json is written (lines 1237-1242)."""
        evidence = {"ids": ["e1", "e2"], "data": {"key": "value"}}
        meta = self._call(
            tmp_path,
            record_evidence="full",
            evidence=evidence,
        )
        epath = Path(meta["evidence_path"])
        assert epath.exists()
        assert "evidence_xxh3_128" in meta
        content = epath.read_text(encoding="utf-8")
        assert '"e1"' in content
        assert '"e2"' in content

    def test_record_evidence_hash_no_file(self, tmp_path: Path) -> None:
        """record_evidence='hash' → hash present but no file."""
        meta = self._call(
            tmp_path,
            record_evidence="hash",
            evidence={"x": 1},
        )
        assert "evidence_xxh3_128" in meta
        assert meta.get("evidence_path") is None

    def test_record_prompt_full_writes_file(self, tmp_path: Path) -> None:
        """record_prompt='full' → prompt.txt is written (lines 1246-1249)."""
        meta = self._call(
            tmp_path,
            record_prompt="full",
            prompt="Hello, this is a test prompt",
        )
        ppath = Path(meta["prompt_path"])
        assert ppath.exists()
        assert ppath.read_text(encoding="utf-8") == "Hello, this is a test prompt"
        assert "prompt_xxh3_128" in meta

    def test_record_prompt_hash_no_file(self, tmp_path: Path) -> None:
        """record_prompt='hash' → hash present but no file."""
        meta = self._call(
            tmp_path,
            record_prompt="hash",
            prompt="test",
        )
        assert "prompt_xxh3_128" in meta
        assert meta.get("prompt_path") is None

    def test_record_prompt_none_no_hash(self, tmp_path: Path) -> None:
        """record_prompt='none' → no hash, no file."""
        meta = self._call(tmp_path, record_prompt="none", prompt="test")
        assert meta.get("prompt_xxh3_128") is None
        assert meta.get("prompt_path") is None

    def test_timing_breakdown_negative_clamped(self, tmp_path: Path) -> None:
        """Negative timing values are clamped to 0.0."""
        meta = self._call(
            tmp_path,
            timing_breakdown={"stage_b_llm_sec": -1.0},
        )
        assert meta["timing_breakdown"]["stage_b_llm_sec"] == 0.0

    def test_token_breakdown_negative_clamped(self, tmp_path: Path) -> None:
        """Negative token values are clamped to 0."""
        meta = self._call(
            tmp_path,
            token_breakdown={"stage_b_tokens_prompt": -10},
        )
        assert meta["token_breakdown"]["stage_b_tokens_prompt"] == 0

    def test_stage_a_prompt_hashed(self, tmp_path: Path) -> None:
        """stage_a_prompt provided → xxh3_128 hash recorded (line 1252)."""
        attempt_dir = tmp_path / "attempt_s"
        attempt_dir.mkdir()
        meta = _materialize_attempt_metadata(
            attempt_dir=attempt_dir,
            attempt=1,
            max_attempts=3,
            evidence={},
            prompt="p",
            record_prompt="none",
            record_evidence="none",
            context_protocol="monolithic",
            stage_a_prompt="some stage-a prompt text",
        )
        assert meta.get("stage_a_prompt_xxh3_128") is not None

    def test_stage_a_raw_response_hashed(self, tmp_path: Path) -> None:
        """stage_a_raw_response → stage_a_response_xxh3_128 (line 1255)."""
        attempt_dir = tmp_path / "attempt_r"
        attempt_dir.mkdir()
        meta = _materialize_attempt_metadata(
            attempt_dir=attempt_dir,
            attempt=1,
            max_attempts=1,
            evidence={},
            prompt="p",
            record_prompt="none",
            record_evidence="none",
            context_protocol="monolithic",
            stage_a_raw_response="raw response data",
        )
        assert meta.get("stage_a_response_xxh3_128") is not None

    def test_stage_a_request_payload_hashed(self, tmp_path: Path) -> None:
        """stage_a_request_payload (no raw_response) → hash (line 1257)."""
        attempt_dir = tmp_path / "attempt_p"
        attempt_dir.mkdir()
        meta = _materialize_attempt_metadata(
            attempt_dir=attempt_dir,
            attempt=1,
            max_attempts=1,
            evidence={},
            prompt="p",
            record_prompt="none",
            record_evidence="none",
            context_protocol="monolithic",
            stage_a_request_payload={"files": ["a.py"]},  # type: ignore[arg-type]
        )
        assert meta.get("stage_a_response_xxh3_128") is not None

    def test_legacy_adapter_defaults_match_typed_request(self, tmp_path: Path) -> None:
        attempt_dir = tmp_path / "defaults"
        attempt_dir.mkdir()
        required: dict[str, Any] = {
            "attempt_dir": attempt_dir,
            "attempt": 1,
            "max_attempts": 2,
            "evidence": {},
            "prompt": "repair",
            "record_prompt": "none",
            "record_evidence": "none",
            "context_protocol": "monolithic",
        }

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            legacy_meta = _materialize_attempt_metadata(**required)
        with pytest.warns(DeprecationWarning, match="legacy_kwargs") as captured:
            legacy_request = build_attempt_metadata_request_from_legacy_kwargs(**required)
        assert Path(captured[0].filename).resolve() == Path(__file__).resolve()
        grouped_meta = _materialize_attempt_metadata_request(
            request=AttemptMetadataRequest(**required)
        )

        assert legacy_meta == grouped_meta
        assert _materialize_attempt_metadata_request(request=legacy_request) == grouped_meta

    def test_legacy_adapter_rejects_unknown_keyword(self, tmp_path: Path) -> None:
        attempt_dir = tmp_path / "unknown"
        attempt_dir.mkdir()

        with pytest.raises(
            TypeError,
            match=r"unexpected keyword argument 'adaptive_polciy'",
        ):
            _materialize_attempt_metadata(
                attempt_dir=attempt_dir,
                attempt=1,
                max_attempts=2,
                evidence={},
                prompt="repair",
                record_prompt="none",
                record_evidence="none",
                context_protocol="monolithic",
                adaptive_polciy="balanced",
            )

    def test_legacy_adapter_rejects_missing_required_keyword(self, tmp_path: Path) -> None:
        attempt_dir = tmp_path / "missing"
        attempt_dir.mkdir()

        with pytest.raises(TypeError, match=r"required keyword argument: 'context_protocol'"):
            _materialize_attempt_metadata(
                attempt_dir=attempt_dir,
                attempt=1,
                max_attempts=2,
                evidence={},
                prompt="repair",
                record_prompt="none",
                record_evidence="none",
            )

    @pytest.mark.parametrize(
        ("name", "case"),
        [
            (
                "baseline",
                {
                    "llm_call_count": 2,
                    "prompt_chars_total": 120,
                    "completion_chars_total": 40,
                    "tokens_prompt": 50,
                    "tokens_completion": 10,
                    "tokens_total": 60,
                    "stage_a": {
                        "stage_a_prompt": "inspect traceback",
                        "stage_a_prompt_chars": 17,
                        "stage_a_requests_count": 1,
                        "stage_a_chars_returned": 23,
                        "stage_a_request_success": True,
                    },
                    "prompt_budget": {
                        "prompt_budget_used_chars": 120,
                        "prompt_budget_overflow_chars": 0,
                        "prompt_budget_dropped_sections": [],
                    },
                },
            ),
            (
                "adaptive_helper",
                {
                    "adaptive": {
                        "adaptive_mode": "helper",
                        "adaptive_policy": "balanced",
                        "adaptive_helper_invoked": True,
                        "adaptive_helper_action_selected": "invoke_helper",
                        "adaptive_helper_action_executed": "invoke_helper",
                        "adaptive_helper_materialized_context": True,
                        "adaptive_helper_outcome": "materialized",
                        "adaptive_helper_available_actions": ["invoke_helper", "skip_helper"],
                        "adaptive_helper_unavailable_action_reasons": {"snapshot": "budget"},
                        "adaptive_progress_made": True,
                        "adaptive_progress_reason": "new_evidence",
                        "adaptive_informativeness_pregate_fired": True,
                        "adaptive_informativeness_pregate_score": 0.75,
                        "adaptive_informativeness_pregate_features": {"frame_depth": 1.0},
                    },
                    "openai": {
                        "openai_preflight_prompt_cap_chars": 4096,
                        "openai_preflight_trimmed": True,
                        "openai_context_window_source": "provider_probe",
                    },
                },
            ),
            (
                "retry_plan",
                {
                    "retry": {
                        "retry_prompt_mode": "delta",
                        "retry_context_mode": "history",
                        "retry_history_chars": 512,
                        "retry_history_truncated": True,
                        "retry_history_turns_included": 3,
                        "retry_history_budget_chars": 256,
                        "retry_history_budget_applied": True,
                        "retry_delta_prompt": True,
                        "retry_delta_chars": 144,
                    },
                    "stage_a": {
                        "stage_a_index_included": True,
                        "stage_a_index_reason": "retry_plan",
                    },
                },
            ),
            (
                "context_compaction",
                {
                    "context_compaction": {
                        "context_compaction_mode": "adaptive",
                        "context_compaction_scope": ["snapshot", "stdout"],
                        "context_compaction_supported": True,
                        "context_compaction_requested": True,
                        "context_compaction_request_reason": "overflow",
                        "context_compaction_trigger": "budget_overflow",
                        "context_compaction_auto_overflow_triggered": True,
                        "context_compaction_invoked": True,
                        "context_compaction_success": True,
                        "context_compaction_calls_used": 1,
                        "context_compaction_max_calls": 2,
                        "context_compaction_source_sections": ["snapshot"],
                        "context_compaction_source_chars": {"snapshot": 1200},
                        "context_compaction_replaced_sections": ["snapshot"],
                        "context_compaction_input_chars": 1200,
                        "context_compaction_target_chars": 600,
                        "context_compaction_output_chars": 550,
                        "context_compaction_prompt_chars": 200,
                    },
                    "syntax_preflight": {
                        "syntax_preflight_enabled": True,
                        "syntax_preflight_cycles_used": 1,
                        "syntax_preflight_failures": 0,
                    },
                },
            ),
        ],
    )
    def test_grouped_request_matches_legacy_flat_kwargs(
        self,
        tmp_path: Path,
        name: str,
        case: dict[str, Any],
    ) -> None:
        attempt_dir = tmp_path / name
        attempt_dir.mkdir(parents=True, exist_ok=True)
        base_kwargs: dict[str, Any] = {
            "attempt_dir": attempt_dir,
            "attempt": 1,
            "max_attempts": 3,
            "evidence": {"items": ["e1", "e2"]},
            "prompt": "repair prompt",
            "record_prompt": "hash",
            "record_evidence": "hash",
            "context_protocol": "staged",
        }
        for group_name in (
            "stage_a",
            "retry",
            "adaptive",
            "prompt_budget",
            "context_compaction",
            "openai",
            "syntax_preflight",
        ):
            base_kwargs.update(case.get(group_name, {}))
        for scalar_name in (
            "llm_call_count",
            "prompt_chars_total",
            "completion_chars_total",
            "tokens_prompt",
            "tokens_completion",
            "tokens_total",
            "evidence_ids_new_count",
            "evidence_ids_reused_count",
            "evidence_ids_sent_count",
        ):
            if scalar_name in case:
                base_kwargs[scalar_name] = case[scalar_name]

        legacy_meta = _materialize_attempt_metadata(**base_kwargs)
        grouped_meta = _materialize_attempt_metadata_request(
            request=AttemptMetadataRequest(
                attempt_dir=attempt_dir,
                attempt=1,
                max_attempts=3,
                evidence={"items": ["e1", "e2"]},
                prompt="repair prompt",
                record_prompt="hash",
                record_evidence="hash",
                context_protocol="staged",
                llm_call_count=int(case.get("llm_call_count", 0)),
                prompt_chars_total=int(case.get("prompt_chars_total", 0)),
                completion_chars_total=int(case.get("completion_chars_total", 0)),
                tokens_prompt=int(case.get("tokens_prompt", 0)),
                tokens_completion=int(case.get("tokens_completion", 0)),
                tokens_total=int(case.get("tokens_total", 0)),
                evidence_ids_new_count=int(case.get("evidence_ids_new_count", 0)),
                evidence_ids_reused_count=int(case.get("evidence_ids_reused_count", 0)),
                evidence_ids_sent_count=int(case.get("evidence_ids_sent_count", 0)),
                stage_a=StageAAttemptMetadata(**case.get("stage_a", {})),
                retry=RetryAttemptMetadata(**case.get("retry", {})),
                adaptive=AdaptiveAttemptMetadata(**case.get("adaptive", {})),
                prompt_budget=PromptBudgetMetadata(**case.get("prompt_budget", {})),
                context_compaction=ContextCompactionMetadata(**case.get("context_compaction", {})),
                openai=OpenAIAttemptMetadata(**case.get("openai", {})),
                syntax_preflight=SyntaxPreflightMetadata(**case.get("syntax_preflight", {})),
            )
        )

        assert grouped_meta == legacy_meta


# ── _SynchronizedJsonlWriter ─────────────────────────────────────────────


class TestSynchronizedJsonlWriter:
    """Lines 143-166."""

    def test_append_one_creates_file(self, tmp_path: Path) -> None:
        """First append_one lazily opens the file."""
        p = tmp_path / "out.jsonl"
        writer = _SynchronizedJsonlWriter(p)
        writer.append_one({"key": "value"})
        writer.close()
        assert p.exists()
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert '"key"' in lines[0]

    def test_append_multiple_records(self, tmp_path: Path) -> None:
        """Multiple appends produce multiple lines."""
        p = tmp_path / "out2.jsonl"
        writer = _SynchronizedJsonlWriter(p)
        writer.append_one({"a": 1})
        writer.append_one({"b": 2})
        writer.close()
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_close_without_open_is_noop(self, tmp_path: Path) -> None:
        """close() on a writer that was never opened does not error."""
        p = tmp_path / "never.jsonl"
        writer = _SynchronizedJsonlWriter(p)
        writer.close()  # should not raise
        assert not p.exists()


# ── _collect_allowed_env ─────────────────────────────────────────────────


class TestCollectAllowedEnv:
    """Line 470: empty-key branch."""

    def test_empty_key_skipped(self) -> None:
        """Keys that are blank after strip are skipped."""
        result = _collect_allowed_env(["", "  ", "   "])
        assert result == {}

    def test_missing_env_var_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env vars not present in os.environ are skipped."""
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        result = _collect_allowed_env(["NONEXISTENT_VAR_XYZ"])
        assert result == {}


# ── _normalize_openai_base_url ───────────────────────────────────────────


class TestNormalizeOpenaiBaseUrl:
    """Lines 525-529."""

    def test_already_has_v1(self) -> None:
        assert _normalize_openai_base_url("https://api.openai.com/v1") == (
            "https://api.openai.com/v1"
        )

    def test_trailing_slash_with_v1(self) -> None:
        assert _normalize_openai_base_url("https://api.openai.com/v1/") == (
            "https://api.openai.com/v1"
        )

    def test_no_version_suffix(self) -> None:
        assert _normalize_openai_base_url("https://api.openai.com") == ("https://api.openai.com/v1")

    def test_v2_suffix_kept(self) -> None:
        assert _normalize_openai_base_url("https://custom.api/v2") == ("https://custom.api/v2")


# ── _default_run_metadata ────────────────────────────────────────────────


class TestDefaultRunMetadata:
    """Line 533."""

    def test_returns_expected_keys(self) -> None:
        meta = _default_run_metadata(run_id="r1", patcher_name="openai")
        assert meta["run_id"] == "r1"
        assert meta["patcher_name"] == "openai"
        assert meta["deterministic"] is False
        assert meta["seed"] is None
        assert "llmdebug_version" in meta
        assert meta["record_policy"] == {"prompt": "hash", "evidence": "hash"}


# ── _build_base_record ───────────────────────────────────────────────────


class TestBuildBaseRecord:
    """Line 565."""

    def test_returns_skeleton(self) -> None:
        case = CaseSpec(
            case_id="case-1",
            path=Path("/fake"),
            category="unit",
            difficulty="easy",
            snapshot_dependency="none",
            expected_exception="AssertionError",
            tags=["tag1"],
            python_args=[],
            timeout_sec=30,
        )
        rec = _build_base_record(
            case=case,
            condition=Condition.BASE_REPAIR,
            run_id="r1",
            attempt=1,
            patcher_name="openai",
            run_metadata={"run_id": "r1"},
            attempt_metadata={"attempt": 1},
            model_metadata={"backend": "openai"},
        )
        assert rec["case_id"] == "case-1"
        assert rec["condition"] == "base_repair"
        assert rec["attempt"] == 1
        assert rec["patcher"] == "openai"
        assert "schema_version" in rec


# ── _build_model_metadata ────────────────────────────────────────────────


class TestBuildModelMetadata:
    """Lines 1264-1308."""

    def test_openai_patcher(self) -> None:
        """openai patcher → full openai metadata."""
        patcher = MagicMock()
        patcher.name = "openai"
        patcher.base_url = "https://api.openai.com/v1"
        patcher.model = "gpt-4"
        patcher.temperature = 0.0
        patcher.max_tokens = 4096
        patcher.response_mode = "text"
        patcher.seed = 42

        meta = _build_model_metadata(
            patcher=patcher,
            patch_meta={"model": "gpt-4o"},
        )
        assert meta["backend"] == "openai"
        assert meta["model"] == "gpt-4o"
        assert "api_compat_seed_requested" in meta

    def test_baseline_patcher(self) -> None:
        """Baseline patcher in BASELINE_CONDITION_BY_NAME → baseline metadata."""
        # Pick a known baseline name
        baseline_name = next(iter(BASELINE_CONDITION_BY_NAME))
        patcher = MagicMock()
        patcher.name = baseline_name

        meta = _build_model_metadata(
            patcher=patcher,
            patch_meta={"baseline": baseline_name, "model": "m1"},
        )
        assert meta["baseline_name"] == baseline_name
        assert meta["model"] == "m1"

    def test_unknown_patcher(self) -> None:
        """Unknown patcher → minimal metadata."""
        patcher = MagicMock()
        patcher.name = "custom_patcher"

        meta = _build_model_metadata(patcher=patcher, patch_meta=None)
        assert meta["backend"] == "custom_patcher"
        assert meta["name"] == "custom_patcher"
        assert "model" not in meta

    def test_patch_meta_none_for_openai(self) -> None:
        """patch_meta=None with openai → uses patcher attrs."""
        patcher = MagicMock()
        patcher.name = "openai"
        patcher.base_url = "https://api.openai.com"
        patcher.model = "gpt-4"
        patcher.temperature = 0.5
        patcher.max_tokens = 2048
        patcher.response_mode = "json"
        patcher.seed = None

        meta = _build_model_metadata(patcher=patcher, patch_meta=None)
        assert meta["backend"] == "openai"
        assert meta["model"] == "gpt-4"


# ── _build_run_metadata with dependency_versions ─────────────────────────


class TestBuildRunMetadataDeps:
    """Line 201: dependency_versions populated."""

    def test_deps_populated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """record_deps != 'none' and versions found → metadata has key."""
        monkeypatch.setattr(
            "evals.eval_recording._resolve_llmdebug_version",
            lambda: "0.0.0",
        )
        monkeypatch.setattr(
            "evals.eval_recording._collect_dependency_versions",
            lambda mode: {"pytest": "8.0.0"},
        )
        args = _run_metadata_args(record_env=[], record_deps="default")
        metadata = _build_run_metadata(
            args=args,
            patcher_name="openai",
            run_id="run-d",
            started_at_utc="2024-01-01T00:00:00Z",
            run_settings_fingerprint="fp",
        )
        assert metadata["dependency_versions"] == {"pytest": "8.0.0"}


# ── _collect_dependency_versions default mode ────────────────────────────


class TestCollectDependencyVersionsDefault:
    """Line 519: PackageNotFoundError path in default mode."""

    def test_default_mode_returns_found_packages(self) -> None:
        """Default mode returns at least pytest (since it's installed)."""
        result = _collect_dependency_versions(mode="default")
        assert "pytest" in result


def test_release_keeps_the_lock_file_so_its_inode_stays_shared(tmp_path: Path) -> None:
    """ORC-14: unlinking a held lock file let two processes lock different inodes."""
    from evals.eval_recording import _RunResultsLock

    results = tmp_path / "run.jsonl"
    lock, error, _warning = _RunResultsLock.acquire(results_path=results, run_id="r1")
    assert lock is not None and error is None
    lock_path = lock.path
    inode = lock_path.stat().st_ino
    lock.release()
    assert lock_path.exists()

    again, error, _warning = _RunResultsLock.acquire(results_path=results, run_id="r2")
    assert again is not None and error is None
    assert lock_path.stat().st_ino == inode
    assert '"run_id": "r2"' in lock_path.read_text(encoding="utf-8")
    again.release()


def test_jsonl_writer_fsyncs_on_its_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals import eval_recording

    synced: list[int] = []
    monkeypatch.setattr(eval_recording.os, "fsync", lambda fd: synced.append(fd))
    writer = eval_recording._SynchronizedJsonlWriter(tmp_path / "r.jsonl", fsync_every=2)
    for index in range(5):
        writer.append_one({"i": index})
    writer.close()
    assert len(synced) == 2

    off = eval_recording._SynchronizedJsonlWriter(tmp_path / "off.jsonl", fsync_every=0)
    off.append_one({"i": 0})
    off.close()
    assert len(synced) == 2
