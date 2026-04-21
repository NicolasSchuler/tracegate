"""No-op patcher used as a control/baseline in evaluations.

``NullPatcher`` always returns an empty patch result (``diff=None``), making it
useful as a lower-bound control condition when measuring how much value the
actual patching logic adds.  It satisfies the :class:`~evals.patchers.base.Patcher`
protocol with the minimum possible implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import Patcher, PatchResult


@dataclass(frozen=True)
class NullPatcher(Patcher):
    """No-op patcher that unconditionally returns an empty patch.

    Acts as a control/baseline condition in A/B evaluation runs.  Every call to
    :meth:`propose_patch` returns a :class:`~evals.patchers.base.PatchResult`
    with ``diff=None``, so downstream scoring treats it as a failed repair
    attempt.

    Attributes:
        name: Identifier string used in evaluation result metadata.
    """

    name: str = "null"

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        """Return an empty patch result without performing any repair.

        Args:
            prompt: The natural-language debugging prompt (ignored).
            context_dir: Context directory (ignored).
            cwd: Working directory of the case (ignored).

        Returns:
            A :class:`~evals.patchers.base.PatchResult` with ``diff=None`` and
            ``meta["reason"]`` set to ``"null_patcher"``.
        """
        return PatchResult(diff=None, meta={"reason": "null_patcher"})
