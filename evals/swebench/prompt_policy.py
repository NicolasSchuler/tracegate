"""Prompt information policy shared by SWE-Bench Lite and Pro generation.

Review findings DAT-01, DAT-02 and DAT-03 (decision D2): by default the prompt carries
only what the official evaluation gives a model. ``hints_text`` and the
FAIL_TO_PASS / PASS_TO_PASS test IDs are opt-in, and the ``llmdebug`` condition
refuses to run without a snapshot directory, because without one its prompt is the
``baseline`` prompt plus a placeholder sentence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import _json as json
from .types import CONDITION_LLMDEBUG

_SNAPSHOT_SUFFIXES = (".txt", ".md", ".json")
_MAX_RECORDED_MISSING_IDS = 50

PROMPT_POLICY_FINGERPRINT_KEY = "prompt_policy_fingerprint"
"""``run_metadata.json`` key holding the fingerprint of the effective prompt policy."""


@dataclass(frozen=True)
class PromptInputs:
    """What a generation prompt may contain beyond the issue text."""

    snapshot_dir: Path | None = None
    include_hints: bool = False
    include_test_ids: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "include_hints": bool(self.include_hints),
            "include_test_ids": bool(self.include_test_ids),
            "snapshot_dir": str(self.snapshot_dir) if self.snapshot_dir is not None else None,
        }

    def fingerprint(self) -> str:
        """A stable digest of the effective prompt policy, for resume validation."""
        return prompt_policy_fingerprint(self.to_metadata())


def prompt_policy_fingerprint(policy: dict[str, Any]) -> str:
    """The short digest of a ``prompt_policy`` metadata block."""
    payload = json.dumps(policy, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def assert_resumable_prompt_policy(*, metadata_file: Path, prompt_inputs: PromptInputs) -> None:
    """Refuse to resume a run whose stored prompt policy differs from this one.

    Resuming keeps the existing predictions but rewrites ``run_metadata.json``, so a
    resume with different flags would relabel leaked-test or hinted predictions as
    official-task-only (DAT-01/02 bookkeeping). Legacy metadata without a recorded
    policy cannot be verified and is rejected explicitly.
    """
    if not metadata_file.exists():
        return
    try:
        stored_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {metadata_file} to validate --resume: {exc}") from exc
    if not isinstance(stored_metadata, dict):
        raise ValueError(f"cannot validate --resume: {metadata_file} is not a JSON object")

    stored_policy = stored_metadata.get("prompt_policy")
    if not isinstance(stored_policy, dict):
        raise ValueError(
            "cannot resume: the existing run predates prompt-policy bookkeeping "
            f"({metadata_file} has no prompt_policy). Generate into a fresh run id."
        )

    current_policy = prompt_inputs.to_metadata()
    if prompt_policy_fingerprint(stored_policy) == prompt_inputs.fingerprint():
        return
    differences = sorted(
        f"{key}: stored={stored_policy.get(key)!r} requested={current_policy[key]!r}"
        for key in current_policy
        if stored_policy.get(key) != current_policy[key]
    )
    raise ValueError(
        "cannot resume with a different prompt policy: the existing predictions were "
        "generated under another information policy. "
        f"Differences: {'; '.join(differences) or 'policy key set changed'}. "
        "Re-run with the original flags, or generate into a fresh run id."
    )


def require_snapshot_dir(conditions: Sequence[str], snapshot_dir: Path | None) -> None:
    """Fail fast when the ``llmdebug`` condition cannot receive snapshot evidence."""
    if CONDITION_LLMDEBUG not in conditions:
        return
    if snapshot_dir is None:
        raise ValueError(
            "The llmdebug condition needs --snapshot-dir: without snapshots its prompt "
            "equals the baseline prompt."
        )
    if not snapshot_dir.is_dir():
        raise ValueError(f"--snapshot-dir is not a directory: {snapshot_dir}")


def load_snapshot_text(*, snapshot_dir: Path | None, instance_id: str) -> str | None:
    """Return the instance's snapshot text, or ``None`` when no artifact exists."""
    if snapshot_dir is None:
        return None
    for suffix in _SNAPSHOT_SUFFIXES:
        path = snapshot_dir / f"{instance_id}{suffix}"
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text
    return None


def render_snapshot_section(*, snapshot_dir: Path | None, instance_id: str) -> str:
    text = load_snapshot_text(snapshot_dir=snapshot_dir, instance_id=instance_id)
    if text is None:
        text = "No instance-specific snapshot artifact was found."
    return "llmdebug runtime evidence:\n" + text


def snapshot_coverage(*, snapshot_dir: Path | None, instance_ids: Sequence[str]) -> dict[str, Any]:
    """Count instances with and without a snapshot artifact, for run metadata."""
    missing = [
        instance_id
        for instance_id in instance_ids
        if load_snapshot_text(snapshot_dir=snapshot_dir, instance_id=instance_id) is None
    ]
    return {
        "snapshot_dir": str(snapshot_dir) if snapshot_dir is not None else None,
        "snapshot_hits": len(instance_ids) - len(missing),
        "snapshot_misses": len(missing),
        "snapshot_missing_instance_ids": missing[:_MAX_RECORDED_MISSING_IDS],
    }
