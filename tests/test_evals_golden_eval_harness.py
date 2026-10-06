"""The golden-run harness itself: normalization, isolation, pinned environment, tokenizer.

These guard ``tests/golden_eval/runner.py``, which the golden fixture runs
(``tests/test_evals_golden_runs.py``) depend on. They are fast and run no scenario.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from tests.golden_eval import runner


def _own_run_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    run_root = tmp_path / "cache" / "golden"
    monkeypatch.setattr(runner, "RUN_ROOT", run_root)
    return run_root


def test_prepare_run_root_tightens_an_existing_permissive_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """mkdir(exist_ok=True) leaves an existing group/world-writable run root as it is."""
    run_root = _own_run_root(monkeypatch, tmp_path)
    run_root.mkdir(parents=True)
    run_root.chmod(0o777)

    runner.prepare_run_root()

    assert stat.S_IMODE(run_root.stat().st_mode) == 0o700


def test_prepare_run_root_refuses_a_world_writable_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Another account could replace the run root between the check and the run."""
    run_root = _own_run_root(monkeypatch, tmp_path)
    run_root.parent.mkdir(parents=True)
    run_root.parent.chmod(0o777)
    try:
        with pytest.raises(RuntimeError, match="group- or world-writable"):
            runner.prepare_run_root()
    finally:
        run_root.parent.chmod(0o755)


def test_fresh_run_root_creates_a_private_scenario_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = _own_run_root(monkeypatch, tmp_path)
    root = runner._fresh_run_root("native_default")

    assert root == run_root / "native_default"
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_scenario_lock_refuses_a_second_holder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The second invocation must fail before it resets the first one's run root."""
    _own_run_root(monkeypatch, tmp_path)
    with runner.scenario_lock("native_default"):
        with (
            pytest.raises(RuntimeError, match="already running"),
            runner.scenario_lock("native_default"),
        ):
            pass
        # A different scenario is independent.
        with runner.scenario_lock("native_full_file"):
            pass


def test_scenario_env_drops_proxies_and_pins_capture_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported proxy would send the loopback stub's requests to a real host."""
    for name, value in {
        "HTTP_PROXY": "http://proxy.invalid:3128",
        "HTTPS_PROXY": "http://proxy.invalid:3128",
        "ALL_PROXY": "socks5://proxy.invalid:1080",
        "LLMDEBUG_REDACTION_PROFILE": "strict",
        "LLMDEBUG_OUTPUT_FORMAT": "toon",
        "PYTEST_ADDOPTS": "-p no:randomly",
        "PYTHONHASHSEED": "99",
    }.items():
        monkeypatch.setenv(name, value)

    env = runner.scenario_env()

    assert not [key for key in env if "PROXY" in key.upper() and key.upper() != "NO_PROXY"]
    assert env["no_proxy"] == "*"
    assert not [key for key in env if key.startswith("LLMDEBUG_")]
    assert env["PYTEST_ADDOPTS"] == ""
    assert "PYTHONHASHSEED" not in env
    assert env["TIKTOKEN_CACHE_DIR"] == str(runner.TOKENIZER_CACHE)
    # Enough of the caller's environment to run python, git and patch.
    assert env["PATH"] == os.environ["PATH"]
    assert env["HOME"] == os.environ["HOME"]


def test_ensure_offline_tokenizer_seeds_the_cache_from_the_temporary_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cold persistent cache is filled from local data, not from a download."""
    cache = tmp_path / "tokenizer"
    monkeypatch.setattr(runner, "TOKENIZER_CACHE", cache)
    default_cache = tmp_path / "tmp" / "data-gym-cache"
    default_cache.mkdir(parents=True)
    (default_cache / "deadbeef").write_bytes(b"vocab")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "tmp"))
    monkeypatch.setattr(
        "evals.token_counting.estimate_text_token_count",
        lambda text, *, model="": {"available": True, "token_count": 1},
    )
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)

    runner.ensure_offline_tokenizer()

    assert (cache / "deadbeef").read_bytes() == b"vocab"
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(cache)


def test_ensure_offline_tokenizer_fails_loudly_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Otherwise a failed download silently changes the token estimates in every record."""
    monkeypatch.setattr(runner, "TOKENIZER_CACHE", tmp_path / "tokenizer")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "empty"))
    monkeypatch.setattr(
        "evals.token_counting.estimate_text_token_count",
        lambda text, *, model="": {"available": False, "error": "tokenizer_resolution_failed"},
    )

    with pytest.raises(RuntimeError, match="tokenizer_resolution_failed"):
        runner.ensure_offline_tokenizer()


def test_run_scenarios_runs_a_repeated_scenario_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two subprocesses for one scenario would share its run root and stub port."""
    started: list[str] = []
    monkeypatch.setattr(runner, "ensure_offline_tokenizer", lambda: None)

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(name: str, out: Path) -> _Completed:
        started.append(name)
        return _Completed()

    monkeypatch.setattr(runner, "run_in_subprocess", _fake_run)

    assert runner.run_scenarios(["native_default", "native_default"], tmp_path) == {}
    assert started == ["native_default"]


# ---------------------------------------------------------------------------
# Normalization: only measurements may be replaced by placeholders.
# ---------------------------------------------------------------------------


def _normalized(record: dict[str, object], tmp_path: Path) -> dict[str, object]:
    normalizer = runner.Normalizer(tmp_path, 47000)
    result = normalizer.value(record)
    assert isinstance(result, dict)
    return result


def test_normalization_distinguishes_budgets_coefficients_and_counts(tmp_path: Path) -> None:
    """A refactor that miscomputes a deadline must not produce identical golden output."""
    baseline: dict[str, object] = {
        "attempt_shared_propose_budget_sec": 330.0,
        "propose_budget_sec": 300.0,
        "effective_timeout_sec": 120.0,
        "openai_transport_retry_backoff_max_sec": 8.0,
        "adaptive_reward_mu_latency": 0.0,
        "adaptive_budget_reward_latency_scale": 10.0,
        "adaptive_budget_selected_time_attempts": 1,
        "adaptive_budget_selected_time_rate": 1.0,
        "budget_can_raise_time": 1.0,
    }
    for key, altered in (
        ("attempt_shared_propose_budget_sec", 60.0),
        ("propose_budget_sec", 30.0),
        ("effective_timeout_sec", 5.0),
        ("openai_transport_retry_backoff_max_sec", 0.0),
        ("adaptive_reward_mu_latency", 0.25),
        ("adaptive_budget_reward_latency_scale", 1.0),
        ("adaptive_budget_selected_time_attempts", 3),
        ("adaptive_budget_selected_time_rate", 0.5),
        ("budget_can_raise_time", 0.0),
    ):
        changed = {**baseline, key: altered}
        assert _normalized(baseline, tmp_path) != _normalized(changed, tmp_path), key
        assert _normalized(changed, tmp_path)[key] == altered


def test_normalization_replaces_measured_times(tmp_path: Path) -> None:
    """Wall-clock measurements differ on every run, so they carry no information."""
    measured: dict[str, object] = {
        "propose_duration_sec": 1.23,
        "attempt_wall_clock_sec": 4.56,
        "max_propose_sec": 329.998,
        "attempt_shared_propose_call_budget_sec": 329.998,
        "started_at_utc": "2026-10-06T07:54:03Z",
        "timestamp_utc": "2026-10-06T07:54:03.716292Z",
        "adaptive_online_dm_latency_ms": 12.5,
    }
    normalized = _normalized(measured, tmp_path)
    assert set(normalized.values()) == {"<T>"}
    # Missing measurements stay missing.
    assert _normalized({"propose_duration_sec": None}, tmp_path) == {"propose_duration_sec": None}


def test_fixed_times_only_rewrites_measured_times() -> None:
    """The analysis tools see stable inputs without losing the configured values."""
    record = {
        "propose_duration_sec": 1.23,
        "started_at_utc": "2026-10-06T07:54:03Z",
        "propose_budget_sec": 300.0,
        "adaptive_budget_selected_time_attempts": 2,
    }
    assert runner._fixed_times(record) == {
        "propose_duration_sec": 1.0,
        "started_at_utc": "2026-01-01T00:00:00Z",
        "propose_budget_sec": 300.0,
        "adaptive_budget_selected_time_attempts": 2,
    }


def test_token_estimates_distinguish_numeric_from_missing(tmp_path: Path) -> None:
    """A regression that drops the estimates must not match the golden output."""
    numeric = _normalized(
        {
            "adaptive_toon_encoded_tokens_estimate": 1200,
            "adaptive_toon_json_tokens_estimate": 1500,
            "adaptive_toon_savings_tokens_estimate": 300,
            "adaptive_toon_savings_tokens_frac": 0.2,
        },
        tmp_path,
    )
    missing = _normalized(
        {
            "adaptive_toon_encoded_tokens_estimate": None,
            "adaptive_toon_json_tokens_estimate": None,
            "adaptive_toon_savings_tokens_estimate": None,
            "adaptive_toon_savings_tokens_frac": None,
        },
        tmp_path,
    )
    assert set(numeric.values()) == {runner._SIZE_ESTIMATE_PLACEHOLDER}
    assert set(missing.values()) == {None}
    assert numeric != missing


def test_measured_and_configured_time_keys_are_disjoint() -> None:
    assert not runner._MEASURED_TIME_KEYS & runner._CONFIGURED_TIME_KEYS


def test_every_time_like_key_in_the_golden_files_is_classified() -> None:
    """An unclassified time-like key is compared, so it would make the goldens flaky."""
    unclassified: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if (
                    runner._TIME_KEY.search(lowered)
                    and lowered not in runner._MEASURED_TIME_KEYS
                    and lowered not in runner._CONFIGURED_TIME_KEYS
                ):
                    unclassified.add(str(key))
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for name in sorted(runner.SCENARIOS):
        walk(json.loads(runner.read_golden(name)))

    assert not unclassified, (
        "classify these in tests/golden_eval/runner.py as measurements "
        f"or as configured values: {sorted(unclassified)}"
    )
