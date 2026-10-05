"""Per-run execution state, passed explicitly from the main flow to the pytest runner.

Replaces the ``run_eval`` module globals that the extracted flows used to write
through forwarding shims, where the writes never reached the readers (X-02,
ORC-01, ORC-02).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from evals.eval_types import ExecutionConfig, _ActiveRunSettings

if TYPE_CHECKING:
    from evals.eval_containers import _EvalContainerPool


@dataclass(frozen=True)
class RunContext:
    """What every pytest run of one eval run shares.

    ``settings`` carries ``--deterministic``/``--seed`` (``PYTHONHASHSEED`` for the
    pytest subprocesses); ``container_pool`` is set only with ``--container-pool``.
    """

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    settings: _ActiveRunSettings = field(default_factory=_ActiveRunSettings)
    container_pool: _EvalContainerPool | None = None
