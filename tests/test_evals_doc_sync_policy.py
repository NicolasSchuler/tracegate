"""POL-25 / X-08: the policy defaults tables are generated from the code."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals import doc_sync_policy as dsp
from evals import eval_types as et
from evals.adaptive_online_policy import EVENT_SCHEMA_VERSION, ONLINE_POLICY_ACTIONS


def test_block_reflects_the_runtime_constants() -> None:
    block = dsp.render_policy_defaults_block()
    assert block.startswith(dsp.BEGIN_POLICY_DEFAULTS_MARKER)
    assert block.endswith(dsp.END_POLICY_DEFAULTS_MARKER)
    # The docs said 8 actions, "linucb only", alpha 0.8 and event schema 3.
    assert f"| Actions | {len(ONLINE_POLICY_ACTIONS)}: " in block
    assert "`linucb`, `thompson`" in block
    assert f"| `alpha` (CLI / Python API) | {et.DEFAULT_ADAPTIVE_ONLINE_ALPHA} / " in block
    assert f"| Event / state schema | {EVENT_SCHEMA_VERSION} / " in block
    assert f"| Success hint scale | {et.ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT} |" in block


def test_shipped_docs_are_in_sync() -> None:
    assert dsp.stale_policy_defaults_docs() == []


def test_stale_and_missing_blocks_are_detected_and_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dsp, "REPO_ROOT", tmp_path)
    stale = tmp_path / "stale.md"
    stale.write_text(
        f"intro\n{dsp.BEGIN_POLICY_DEFAULTS_MARKER}\nalpha 0.8\n{dsp.END_POLICY_DEFAULTS_MARKER}\n",
        encoding="utf-8",
    )
    missing = tmp_path / "missing.md"
    missing.write_text("no markers\n", encoding="utf-8")

    assert dsp.stale_policy_defaults_docs((stale, missing)) == ["stale.md", "missing.md"]
    assert dsp.write_policy_defaults_blocks((stale,)) == ["stale.md"]
    assert dsp.stale_policy_defaults_docs((stale,)) == []
    assert stale.read_text(encoding="utf-8").startswith("intro\n")
    assert dsp.write_policy_defaults_blocks((stale,)) == []
    with pytest.raises(ValueError, match="missing"):
        dsp.write_policy_defaults_blocks((missing,))
