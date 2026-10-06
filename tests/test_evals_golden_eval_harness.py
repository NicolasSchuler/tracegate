"""The golden-run harness itself: isolation, pinned environment and offline tokenizer.

These guard ``tests/golden_eval/runner.py``, which the golden fixture runs
(``tests/test_evals_golden_runs.py``) depend on. They are fast and run no scenario.
"""

from __future__ import annotations

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
