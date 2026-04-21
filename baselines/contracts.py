"""Machine-readable contract metadata for external baseline adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BASELINE_CONTRACT_VERSION = "v1"


@dataclass(frozen=True)
class BaselineContract:
    """Explicit contract metadata for one baseline integration."""

    baseline_name: str
    contract_version: str = BASELINE_CONTRACT_VERSION
    upstream_commit: str = ""
    fidelity_level: str = "adapter_defined"
    repair_scope_policy: str = "adapter_defined"
    prompt_contract_kind: str = "adapter_defined"
    output_contract: str = "adapter_defined"
    unsupported_case_policy: str = "explicit_failure"
    accepted_case_shapes: tuple[str, ...] = ()
    patch_scope_summary: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "contract_version": self.contract_version,
            "upstream_commit": self.upstream_commit,
            "fidelity_level": self.fidelity_level,
            "repair_scope_policy": self.repair_scope_policy,
            "prompt_contract_kind": self.prompt_contract_kind,
            "output_contract": self.output_contract,
            "unsupported_case_policy": self.unsupported_case_policy,
            "accepted_case_shapes": list(self.accepted_case_shapes),
            "patch_scope_summary": list(self.patch_scope_summary),
            "notes": list(self.notes),
        }
