"""Typed case-input schemas for each baseline adapter.

Each schema enforces the contract between ``convert_case()`` and
``run_repair()`` with runtime validation.  Unknown keys raise
``ValueError`` to catch typos early; harness-injected keys
(``llmdebug_*``) are explicitly allowed through.

The ``BaseCaseInput`` base class provides common fields shared by all
baselines.  Each adapter-specific subclass adds its own required and
optional fields.
"""

from __future__ import annotations

import logging
from dataclasses import MISSING, dataclass, fields
from typing import Any, ClassVar, TypeVar

logger = logging.getLogger(__name__)

# Keys injected by BaselinePatcher *after* convert_case() returns.
# These are always allowed and never part of the adapter schema.
_HARNESS_INJECTED_PREFIXES = ("llmdebug_",)


def _is_harness_key(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in _HARNESS_INJECTED_PREFIXES)


def _required_input_fields(cls: type[BaseCaseInput]) -> set[str]:
    required = {
        f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
    }
    declared_required = set(getattr(cls, "REQUIRED_INPUT_FIELDS", frozenset()))
    unknown_required = declared_required - {f.name for f in fields(cls)}
    if unknown_required:
        rendered = ", ".join(sorted(unknown_required))
        raise TypeError(
            f"{cls.__name__}: REQUIRED_INPUT_FIELDS references unknown field(s): {rendered}"
        )
    return required | declared_required


_CaseInputT = TypeVar("_CaseInputT", bound="BaseCaseInput")


@dataclass(frozen=True)
class BaseCaseInput:
    """Common fields shared across all baseline case inputs.

    Subclasses define adapter-specific fields.  The ``from_dict``
    classmethod validates that all required fields are present and that
    no unexpected keys appear (harness-injected ``llmdebug_*`` keys are
    silently allowed).
    """

    task_id: str
    entry_point: str
    test: str = ""
    python_args: list[str] | None = None
    timeout_sec: int = 0
    files: list[dict[str, str]] | None = None
    case_information: dict[str, Any] | None = None
    REQUIRED_INPUT_FIELDS: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def from_dict(cls: type[_CaseInputT], data: dict[str, Any]) -> _CaseInputT:
        """Validate and construct from a raw dict.

        Raises ``ValueError`` on missing required input fields or
        unexpected keys (excluding harness-injected ``llmdebug_*`` keys).
        """
        known_fields = {f.name for f in fields(cls)}
        unknown = {k for k in data if k not in known_fields and not _is_harness_key(k)}
        if unknown:
            raise ValueError(
                f"{cls.__name__}: unexpected keys {sorted(unknown)}. "
                f"Known fields: {sorted(known_fields)}"
            )
        required_fields = _required_input_fields(cls)
        missing = sorted(field_name for field_name in required_fields if field_name not in data)
        if missing:
            raise ValueError(f"{cls.__name__}: missing required fields {missing}")
        init_kwargs = {k: v for k, v in data.items() if k in known_fields}
        try:
            return cls(**init_kwargs)
        except TypeError as exc:
            raise ValueError(f"{cls.__name__}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Convert back to a plain dict for backward compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class LDBCaseInput(BaseCaseInput):
    """Case input for the LDB (LLMDebugger) baseline.

    LDB expects HumanEval-like rows with a callable prompt, seed code,
    and check-function-style tests.
    """

    REQUIRED_INPUT_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"prompt", "test", "seed", "given_tests"}
    )
    prompt: str = ""
    seed: str = ""
    given_tests: list[str] | None = None
    dataset_type: str = "llmdebug"


@dataclass(frozen=True)
class IntervenorCaseInput(BaseCaseInput):
    """Case input for the INTERVENOR teacher-learner baseline."""

    REQUIRED_INPUT_FIELDS: ClassVar[frozenset[str]] = frozenset({"prompt", "buggy_code"})
    prompt: str = ""
    buggy_code: str = ""


@dataclass(frozen=True)
class TraceCoderCaseInput(BaseCaseInput):
    """Case input for the TraceCoder baseline."""

    REQUIRED_INPUT_FIELDS: ClassVar[frozenset[str]] = frozenset({"complete_prompt", "buggy_code"})
    complete_prompt: str = ""
    buggy_code: str = ""
    repair_seeded: bool = True
    execution_mode: str = "llmdebug_native"


@dataclass(frozen=True)
class LambdaRLMCaseInput(BaseCaseInput):
    """Case input for the lambda-RLM baseline.

    Also used by the standalone RLM adapter (which delegates case
    conversion to lambda-RLM's converter).
    """

    REQUIRED_INPUT_FIELDS: ClassVar[frozenset[str]] = frozenset({"buggy_code", "prompt_context"})
    buggy_code: str = ""
    prompt_context: str = ""


# RLM reuses exactly the same case schema as lambda-RLM.
RLMCaseInput = LambdaRLMCaseInput
