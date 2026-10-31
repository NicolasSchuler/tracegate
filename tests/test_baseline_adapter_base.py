"""Direct contract tests for shared baseline adapter helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from baselines.adapter_base import (
    BaselineAdapter,
    BaselineConfig,
    BaselineFileEdit,
    BaselineResult,
    _make_file_diff,
    _normalize_result_path,
    _read_case_text,
    _render_empty_file_diff_header,
    _synthesize_multifile_diff,
)
from evals.eval_patches import apply_diff


class _DummyAdapter(BaselineAdapter):
    @property
    def name(self) -> str:
        return "dummy"

    def setup(self, config: BaselineConfig) -> None:
        del config

    def convert_case(self, case_dir: Path, *, case_context=None) -> dict[str, Any]:
        del case_context
        return {"case_dir": str(case_dir)}

    def run_repair(self, case_input: dict[str, Any], config: BaselineConfig) -> BaselineResult:
        del case_input, config
        raise AssertionError("run_repair is not used in these tests")


def test_baseline_config_defaults_timeout_when_all_timeouts_unset() -> None:
    config = BaselineConfig(api_timeout_sec=None, process_timeout_sec=None, timeout_sec=None)

    assert config.api_timeout_sec == 300.0
    assert config.process_timeout_sec == 300.0
    assert config.timeout_sec == 300.0


@pytest.mark.parametrize(
    "raw_path",
    [
        "",
        "/abs/path.py",
        "../escape.py",
        "pkg/../buggy.py",
    ],
)
def test_normalize_result_path_rejects_invalid_values(raw_path: str) -> None:
    with pytest.raises(ValueError):
        _normalize_result_path(raw_path)


def test_read_case_text_rejects_non_utf8_file(tmp_path: Path) -> None:
    (tmp_path / "buggy.py").write_bytes(b"\xff\xfe\xfd")

    with pytest.raises(ValueError, match="UTF-8"):
        _read_case_text(case_dir=tmp_path, relative_path="buggy.py")


def test_render_empty_file_diff_header_for_new_empty_file() -> None:
    diff = _render_empty_file_diff_header(
        relative_path="empty.py",
        original_text=None,
        updated_text="",
    )

    assert diff == (
        "diff --git a/empty.py b/empty.py\nnew file mode 100644\n--- /dev/null\n+++ b/empty.py\n"
    )


def test_render_empty_file_diff_header_for_deleted_empty_file() -> None:
    diff = _render_empty_file_diff_header(
        relative_path="empty.py",
        original_text="",
        updated_text=None,
    )

    assert diff == (
        "diff --git a/empty.py b/empty.py\n"
        "deleted file mode 100644\n"
        "--- a/empty.py\n"
        "+++ /dev/null\n"
    )


def test_render_empty_file_diff_header_returns_none_for_non_empty_transition() -> None:
    diff = _render_empty_file_diff_header(
        relative_path="buggy.py",
        original_text="x = 1\n",
        updated_text="x = 2\n",
    )

    assert diff is None


def test_synthesize_multifile_diff_rejects_duplicate_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate baseline file edit"):
        _synthesize_multifile_diff(
            case_dir=tmp_path,
            file_edits=[
                BaselineFileEdit(path="buggy.py", content="x = 1\n"),
                BaselineFileEdit(path="buggy.py", content="x = 2\n"),
            ],
        )


def test_synthesize_multifile_diff_returns_none_for_noop(tmp_path: Path) -> None:
    (tmp_path / "buggy.py").write_text("x = 1\n", encoding="utf-8")

    diff = _synthesize_multifile_diff(
        case_dir=tmp_path,
        file_edits=[BaselineFileEdit(path="buggy.py", content="x = 1\n")],
    )

    assert diff is None


def test_extract_diff_round_trips_multifile_add_update_delete(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text("value = 1\n", encoding="utf-8")
    (case_dir / "helper.py").write_text("obsolete = True\n", encoding="utf-8")

    adapter = _DummyAdapter()
    diff = adapter.extract_diff(
        case_dir,
        BaselineResult(
            file_edits=[
                BaselineFileEdit(path="buggy.py", content="value = 2\n"),
                BaselineFileEdit(path="extra.py", content="created = True\n"),
                BaselineFileEdit(path="helper.py", content=None),
            ],
            success=True,
        ),
    )

    assert diff is not None

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "buggy.py").write_text("value = 1\n", encoding="utf-8")
    (workdir / "helper.py").write_text("obsolete = True\n", encoding="utf-8")
    ok, msg = apply_diff(workdir, diff)

    assert ok is True, msg
    assert (workdir / "buggy.py").read_text(encoding="utf-8") == "value = 2\n"
    assert (workdir / "extra.py").read_text(encoding="utf-8") == "created = True\n"
    assert not (workdir / "helper.py").exists()


def test_extract_diff_keeps_empty_file_edits_as_empty_files(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "empty_me.py").write_text("to_be_removed = True\n", encoding="utf-8")

    adapter = _DummyAdapter()
    diff = adapter.extract_diff(
        case_dir,
        BaselineResult(
            file_edits=[BaselineFileEdit(path="empty_me.py", content="")],
            success=True,
        ),
    )

    assert diff is not None

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "empty_me.py").write_text("to_be_removed = True\n", encoding="utf-8")
    ok, msg = apply_diff(workdir, diff)

    assert ok is True, msg
    assert (workdir / "empty_me.py").exists()
    assert (workdir / "empty_me.py").read_text(encoding="utf-8") == ""


def test_extract_diff_prefers_file_edits_over_legacy_repaired_code(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text("value = 1\n", encoding="utf-8")
    (case_dir / "helper.py").write_text("value = 10\n", encoding="utf-8")

    adapter = _DummyAdapter()
    diff = adapter.extract_diff(
        case_dir,
        BaselineResult(
            repaired_code="value = 999\n",
            file_edits=[BaselineFileEdit(path="helper.py", content="value = 11\n")],
            success=True,
        ),
    )

    assert diff is not None
    assert "diff --git a/helper.py b/helper.py" in diff
    assert "diff --git a/buggy.py b/buggy.py" not in diff


def test_emit_progress_forwards_copy_of_payload() -> None:
    adapter = _DummyAdapter()
    seen: list[tuple[str, dict[str, Any] | None]] = []

    def _callback(event: str, payload: dict[str, Any] | None) -> None:
        seen.append((event, payload))
        assert payload is not None
        payload["mutated"] = True

    adapter.set_progress_callback(_callback)
    original_payload = {"step": 1}

    adapter._emit_progress("phase_start", original_payload)

    assert seen == [("phase_start", {"step": 1, "mutated": True})]
    assert original_payload == {"step": 1}


def test_emit_progress_swallows_callback_errors() -> None:
    adapter = _DummyAdapter()
    invoked = False

    def _callback(_event: str, _payload: dict[str, Any] | None) -> None:
        nonlocal invoked
        invoked = True
        raise RuntimeError("boom")

    adapter.set_progress_callback(_callback)

    adapter._emit_progress("phase_start", {"step": 1})

    # Verify the callback was actually invoked (even though its error was swallowed)
    assert invoked


def test_make_file_diff_returns_none_when_both_versions_missing() -> None:
    assert _make_file_diff(relative_path="buggy.py", original_text=None, updated_text=None) is None


def test_make_file_diff_renders_new_empty_file_header() -> None:
    diff = _make_file_diff(relative_path="new_empty.py", original_text=None, updated_text="")

    assert diff == (
        "diff --git a/new_empty.py b/new_empty.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/new_empty.py\n"
    )


def _git(repo: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_observed_repo_state_reports_commit_dirtiness_and_patch_hash(tmp_path: Path) -> None:
    """BAS-05/BAS-14: what is checked out, not just the pin."""
    from baselines.adapter_base import observed_repo_state

    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")

    clean = observed_repo_state(tmp_path)
    assert len(clean["observed_commit"]) == 40
    assert clean["dirty"] is False

    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")  # an applied upstream patch
    patched = observed_repo_state(tmp_path)
    assert patched["dirty"] is True
    assert patched["observed_commit"] == clean["observed_commit"]
    assert patched["tree_patch_sha256"] != clean["tree_patch_sha256"]


def test_observed_repo_state_without_a_git_checkout(tmp_path: Path) -> None:
    from baselines.adapter_base import observed_repo_state

    assert observed_repo_state(tmp_path / "missing") == {
        "observed_commit": "",
        "dirty": None,
        "tree_patch_sha256": None,
    }


def test_require_pinned_commit_rejects_another_checkout() -> None:
    from baselines.adapter_base import require_pinned_commit

    require_pinned_commit(name="ldb", observed_commit="abc", pinned_commit="abc")
    require_pinned_commit(name="ldb", observed_commit="", pinned_commit="abc")  # unknown: skip
    with pytest.raises(RuntimeError, match="ldb repo is at def, not the pinned commit abc"):
        require_pinned_commit(name="ldb", observed_commit="def", pinned_commit="abc")


def test_generation_settings_env_carries_only_set_values() -> None:
    from baselines.adapter_base import BaselineGenerationSettings

    assert BaselineGenerationSettings().to_env() == {}
    assert BaselineGenerationSettings(
        temperature=0.2, seed=7, max_completion_tokens=512
    ).to_env() == {
        "TRACEGATE_BASELINE_TEMPERATURE": "0.2",
        "TRACEGATE_BASELINE_SEED": "7",
        "TRACEGATE_BASELINE_MAX_TOKENS": "512",
    }


def test_untracked_tree_digest_marks_missing_oversized_and_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex 1e#10: content for normal files, stable markers for the awkward ones."""
    from baselines import adapter_base
    from baselines.adapter_base import _untracked_tree_digest

    (tmp_path / "small.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "big.bin").write_bytes(b"0" * 64)
    listing = b"small.py\n\nmissing.py\nbig.bin\n"

    monkeypatch.setattr(adapter_base, "_UNTRACKED_CONTENT_HASH_MAX_BYTES", 8)
    digest = _untracked_tree_digest(tmp_path, listing).decode("utf-8")

    assert "missing.py\0absent" in digest
    assert "big.bin\0size:64" in digest
    assert "small.py\0" in digest
    assert "\0absent" in digest

    def _boom(self: Path) -> bool:
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "is_file", _boom)
    assert "small.py\0unreadable" in _untracked_tree_digest(tmp_path, b"small.py\n").decode("utf-8")


def test_repair_deadline_clamps_to_the_remaining_budget() -> None:
    """Codex 1e#3: one deadline per run_repair bounds every blocking step."""
    import time

    from baselines.adapter_base import RepairDeadline

    unbounded = RepairDeadline.from_config(BaselineConfig(process_timeout_sec=0, timeout_sec=0))
    assert unbounded.bounded is False
    assert unbounded.expired is False
    assert unbounded.remaining_sec() == float("inf")
    assert unbounded.clamp_sec(12.0) == 12.0
    assert unbounded.clamp_int_sec(12) == 12

    bounded = RepairDeadline.from_config(BaselineConfig(process_timeout_sec=30))
    assert bounded.bounded is True
    assert bounded.expired is False
    assert bounded.clamp_sec(5.0) == 5.0
    assert bounded.clamp_int_sec(5) == 5
    # No explicit request: take whatever is left.
    assert 0.0 < bounded.clamp_sec(None) <= 30.0

    spent = RepairDeadline(started_at=time.perf_counter() - 60.0, budget_sec=30.0)
    assert spent.expired is True
    assert spent.remaining_sec() == 0.0
    assert spent.clamp_sec(10.0) == 0.1
    assert spent.clamp_int_sec(10) == 1


def test_effective_generation_reports_adapter_overrides() -> None:
    """Codex 1e#7: the record follows the knob that actually won."""
    from baselines.adapter_base import BaselineGenerationSettings

    class _OverridingAdapter(_DummyAdapter):
        honours_generation_contract = True
        generation_override_keys = {
            "stub_temperature": "temperature",
            "stub_max_tokens": "max_completion_tokens",
        }

    adapter = _OverridingAdapter()
    shared = BaselineGenerationSettings(temperature=0.0, max_completion_tokens=2048)
    plain = adapter.effective_generation(BaselineConfig(generation=shared))
    assert plain["max_completion_tokens"] == 2048
    assert "overrides" not in plain

    overridden = adapter.effective_generation(
        BaselineConfig(generation=shared, extra={"stub_temperature": 0.7, "stub_max_tokens": 8192})
    )
    assert overridden["temperature"] == pytest.approx(0.7)
    assert overridden["max_completion_tokens"] == 8192
    assert overridden["overrides"] == {
        "temperature": "stub_temperature",
        "max_completion_tokens": "stub_max_tokens",
    }
