"""Rule-based patcher that applies pattern-matching heuristics rather than an LLM.

Instead of querying a language model, ``HeuristicSnapshotPatcher`` inspects the
runtime snapshot produced by the debugger (``snapshot.json`` in *context_dir*)
and applies a hard-coded fix when a known bug pattern is detected.

Currently the only recognised pattern is the ``off_by_one`` toy case, where the
crash frame reports a list index equal to the list length.  The patcher rewrites
``return xs[i]`` → ``return xs[i - 1]`` and returns the resulting unified diff.

This patcher is primarily used to smoke-test the evaluation harness end-to-end
without a live LLM backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import _json as json
from ..eval_patches import _build_diff_from_changes
from .base import Patcher, PatchResult


@dataclass(frozen=True)
class HeuristicSnapshotPatcher(Patcher):
    """Demo patcher that only fixes the toy `off_by_one` case using snapshot locals.

    This is intentionally narrow: it exists to smoke-test the evaluation harness end-to-end
    without wiring a real LLM client.
    """

    name: str = "heuristic"

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        """Inspect the crash snapshot and return a diff if a known bug pattern matches.

        Reads ``snapshot.json`` from *context_dir*, examines the top crash frame's
        local variables, and—if the off-by-one pattern is detected in ``buggy.py``—
        produces a minimal unified diff that corrects the list index.

        All early-exit conditions (missing snapshot, unrecognised file, unexpected
        locals, pattern not found in source) surface in ``PatchResult.meta`` under
        the ``"reason"`` key rather than raising exceptions.

        Args:
            prompt: The natural-language debugging prompt (unused by this patcher).
            context_dir: Directory containing ``snapshot.json`` and other context
                files written by the eval harness.
            cwd: Working directory of the case under evaluation; used to resolve
                relative file paths recorded in the snapshot.

        Returns:
            A :class:`~evals.patchers.base.PatchResult` whose ``diff`` field holds
            a unified diff string on success, or ``None`` when no patch could be
            produced.  ``meta["reason"]`` is always populated with a diagnostic key.
        """
        snapshot_path = context_dir / "snapshot.json"
        if not snapshot_path.exists():
            return PatchResult(diff=None, meta={"reason": "no_snapshot"})

        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            return PatchResult(diff=None, meta={"reason": "bad_snapshot_json"})

        frames = snapshot.get("frames") or []
        if not isinstance(frames, list) or not frames:
            return PatchResult(diff=None, meta={"reason": "no_frames"})

        crash = frames[0]
        if not isinstance(crash, dict):
            return PatchResult(diff=None, meta={"reason": "bad_crash_frame"})

        file_rel = crash.get("file_rel") or crash.get("file")
        if not isinstance(file_rel, str) or not file_rel:
            return PatchResult(diff=None, meta={"reason": "no_crash_file"})

        # The toy case crashes in evals/cases/off_by_one/buggy.py
        if not file_rel.endswith("buggy.py"):
            return PatchResult(diff=None, meta={"reason": "not_toy_case", "file": file_rel})

        locals_ = crash.get("locals") or {}
        if not isinstance(locals_, dict):
            return PatchResult(diff=None, meta={"reason": "no_locals"})

        i_val = locals_.get("i")
        xs_val = locals_.get("xs")
        if not isinstance(i_val, int) or not isinstance(xs_val, list):
            return PatchResult(diff=None, meta={"reason": "unexpected_locals"})

        if i_val != len(xs_val):
            return PatchResult(diff=None, meta={"reason": "not_off_by_one"})

        # Fix: clamp index to last element (minimal patch).
        target = Path(file_rel)
        if not target.is_absolute():
            target = cwd / target

        try:
            src = target.read_text(encoding="utf-8").splitlines(True)
        except Exception:
            return PatchResult(
                diff=None, meta={"reason": "cannot_read_target", "file": str(target)}
            )

        # Replace `return xs[i]` with a guarded version.
        old = "    return xs[i]\n"
        new = "    return xs[i - 1]\n"
        if old not in src:
            return PatchResult(diff=None, meta={"reason": "pattern_not_found"})

        updated = "".join([new if line == old else line for line in src])
        before = "".join(src)

        # LLM-08: the private builder joined keepends lines with "\n", so every body line was
        # followed by a blank line and the diff never applied.
        diff = _build_diff_from_changes([(_relpath_from_cwd(target, cwd), before, updated)])
        return PatchResult(diff=diff, meta={"reason": "patched"})


def _relpath_from_cwd(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except Exception:
        return str(path)
