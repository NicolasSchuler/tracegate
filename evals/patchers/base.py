"""Patcher interface for eval harness.

A patcher receives a prompt (plus a context directory path) and returns a single
unified diff that can be applied with `git apply`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .payload_types import ContextRequestPayload, PatcherMeta


def _method_is_inherited_noop(method: Any, *, base_method: Callable[..., Any]) -> bool:
    """Whether *method* resolves to the unchanged base-class implementation."""
    return getattr(method, "__func__", method) is base_method


@dataclass(frozen=True)
class PatchResult:
    """Output of a single patch-proposal attempt.

    Attributes:
        diff: Unified diff string ready to be applied with ``git apply``,
            or ``None`` if the patcher failed to produce a valid diff.
        meta: Optional metadata dict (timing, token usage, error details,
            etc.) recorded by the patcher during the attempt.
    """

    diff: str | None
    meta: PatcherMeta | None = None


@dataclass(frozen=True)
class ContextRequestResult:
    """Result of a context-fetching step.

    Holds the structured JSON payload that the model decided to request as
    well as the raw model response from which it was parsed.

    Attributes:
        request_json: Parsed ``ContextRequestPayload`` describing which
            files, symbols, or frames the model needs, or ``None`` when no
            parseable request was produced.
        raw_response: The unprocessed text returned by the LLM before JSON
            extraction, or ``None`` if the step was skipped.
        meta: Optional metadata dict capturing timing, token usage, and any
            error information for this step.
    """

    request_json: ContextRequestPayload | None
    raw_response: str | None = None
    meta: PatcherMeta | None = None


@dataclass(frozen=True)
class ContextCompactionResult:
    """Result of a context-compaction step.

    Compaction distils a large context blob down to a shorter summary that
    fits within the model's prompt budget.

    Attributes:
        compacted_text: The summarised context text produced by the model,
            or ``None`` if compaction was not performed or failed.
        raw_response: The raw text returned by the LLM before any
            post-processing, or ``None`` if the step was skipped.
        meta: Optional metadata dict with timing, token usage, and error
            details for this step.
    """

    compacted_text: str | None
    raw_response: str | None = None
    meta: PatcherMeta | None = None


@dataclass(frozen=True)
class PatchPlanResult:
    """Result of a patch-planning step.

    The planning step asks the model to reason about *how* to fix the bug
    before it writes the actual diff, producing a free-text plan.

    Attributes:
        plan_text: The model's patch plan as free-form text, or ``None`` if
            planning was skipped or produced no output.
        raw_response: The raw text returned by the LLM before any
            post-processing, or ``None`` if the step was skipped.
        meta: Optional metadata dict capturing timing, token usage, and
            error details for this step.
    """

    plan_text: str | None
    raw_response: str | None = None
    meta: PatcherMeta | None = None


class Patcher:
    """Abstract base class defining the patcher interface for the eval harness.

    A ``Patcher`` receives a natural-language bug report (``prompt``) and
    the path to a working directory (``cwd``) that contains the failing code,
    then returns a ``PatchResult`` holding a unified diff ready to be applied
    with ``git apply``.

    The interface is intentionally staged: subclasses may implement any
    combination of the optional steps — context request, context compaction,
    and patch planning — before producing the final diff via
    ``propose_patch``, which *must* be overridden.  Each optional step has
    both a synchronous and an ``async`` counterpart; the default async
    versions simply delegate to the synchronous ones.

    Attributes:
        name: Human-readable identifier for this patcher, used in result
            metadata and log messages.
    """

    name: str

    def _progress_callback_var(
        self,
    ) -> ContextVar[Callable[[str, dict[str, Any] | None], None] | None]:
        callback_var = getattr(self, "_llmdebug_progress_callback_var", None)
        if isinstance(callback_var, ContextVar):
            return callback_var
        new_var: ContextVar[Callable[[str, dict[str, Any] | None], None] | None] = ContextVar(
            f"llmdebug_patcher_progress_callback_{id(self)}", default=None
        )
        object.__setattr__(self, "_llmdebug_progress_callback_var", new_var)
        return new_var

    def set_progress_callback(
        self,
        callback: Callable[[str, dict[str, Any] | None], None] | None,
    ) -> None:
        self._progress_callback_var().set(callback)

    def _current_progress_callback(
        self,
    ) -> Callable[[str, dict[str, Any] | None], None] | None:
        return self._progress_callback_var().get()

    def _emit_progress(self, event: str, payload: dict[str, Any] | None = None) -> None:
        callback = self._current_progress_callback()
        if callback is None:
            return
        try:
            forwarded = dict(payload) if isinstance(payload, dict) else None
            callback(event, forwarded)
        except Exception:
            return

    def probe_server_ready(self) -> tuple[bool, PatcherMeta]:
        """Probe whether the patcher's backing service is ready for a fresh request.

        Patchers without a remote readiness concept can keep the default
        implementation, which reports the probe as unsupported.
        """
        return False, {
            "error": "server_ready_probe_unsupported",
            "reason": "patcher does not implement server readiness probing",
        }

    def supports_request_context(self) -> bool:
        """Whether this patcher provides a real Stage-A request-context backend.

        The default implementation preserves the legacy contract: subclasses
        that override :meth:`request_context` are treated as supporting Stage A
        even if they have not yet been updated to override this capability
        hook explicitly.
        """
        request_context_fn = getattr(self, "request_context", None)
        if not callable(request_context_fn):
            return False
        return not _method_is_inherited_noop(
            request_context_fn,
            base_method=Patcher.request_context,
        )

    def supports_context_compaction(self) -> bool:
        """Whether this patcher provides a real delegated context-compaction backend.

        The default implementation preserves the legacy contract: subclasses
        that override :meth:`compact_context` are treated as supporting
        compaction even if they have not yet been updated to override this
        capability hook explicitly.
        """
        compact_context_fn = getattr(self, "compact_context", None)
        if not callable(compact_context_fn):
            return False
        return not _method_is_inherited_noop(
            compact_context_fn,
            base_method=Patcher.compact_context,
        )

    def supports_strict_json_first(self) -> bool:
        """Whether this patcher supports strict-JSON-first Stage-A retries.

        The default implementation preserves the legacy contract by inferring
        support from the overridden :meth:`request_context` signature. Patchers
        that accept ``strict_json_first`` (or ``**kwargs``) continue to opt
        into strict-JSON retries without having to override this hook.
        """
        if not self.supports_request_context():
            return False
        request_context_fn = getattr(self, "request_context", None)
        if not callable(request_context_fn):
            return False
        try:
            signature = inspect.signature(request_context_fn)
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ) or ("strict_json_first" in signature.parameters)

    def request_context(
        self,
        *,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        strict_json_first: bool = False,
    ) -> ContextRequestResult | None:
        """Ask the model which additional context it needs before patching.

        The model inspects the prompt and any artefacts already present in
        *context_dir*, then returns a structured ``ContextRequestPayload``
        describing which files, symbols, stack frames, or snapshots should be
        fetched and passed back on the next turn.

        Args:
            prompt: The natural-language bug report or task description.
            context_dir: Directory where previously gathered context artefacts
                are stored and where new ones should be written.
            cwd: Root of the repository / working tree being debugged.
            strict_json_first: When ``True``, instruct the model to respond
                with valid JSON on the first attempt before falling back to
                free-form text.

        Returns:
            A ``ContextRequestResult`` containing the parsed request payload
            and raw model response, or ``None`` if this patcher does not
            support a context-request step. Callers must gate execution via
            :meth:`supports_request_context` rather than inferring support
            from the mere presence of this fallback method.
        """
        del strict_json_first
        return None

    async def arequest_context(
        self,
        *,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        strict_json_first: bool = False,
    ) -> ContextRequestResult | None:
        """Async variant of ``request_context``; delegates to the sync implementation by default."""
        return self.request_context(
            prompt=prompt,
            context_dir=context_dir,
            cwd=cwd,
            strict_json_first=strict_json_first,
        )

    def compact_context(
        self,
        *,
        prompt: str,
        context_text: str,
        context_dir: Path,
        cwd: Path,
        target_chars: int,
    ) -> ContextCompactionResult | None:
        """Summarise a large context blob to fit within the model's prompt budget.

        When the accumulated context exceeds the token limit, this step asks
        the model to distil *context_text* into a shorter summary while
        preserving the information needed to produce a correct patch.

        Args:
            prompt: The natural-language bug report or task description.
            context_text: The full context string to be compacted.
            context_dir: Directory where context artefacts are stored.
            cwd: Root of the repository / working tree being debugged.
            target_chars: Approximate character budget for the compacted
                output.

        Returns:
            A ``ContextCompactionResult`` containing the compacted text and
            raw model response, or ``None`` if this patcher does not support
            a compaction step. Callers must gate execution via
            :meth:`supports_context_compaction` rather than inferring support
            from the mere presence of this fallback method.
        """
        return None

    async def acompact_context(
        self,
        *,
        prompt: str,
        context_text: str,
        context_dir: Path,
        cwd: Path,
        target_chars: int,
    ) -> ContextCompactionResult | None:
        """Async variant of ``compact_context``; delegates to the sync implementation by default."""
        return self.compact_context(
            prompt=prompt,
            context_text=context_text,
            context_dir=context_dir,
            cwd=cwd,
            target_chars=target_chars,
        )

    def plan_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchPlanResult | None:
        """Generate a free-text repair plan before writing the diff.

        The planning step encourages the model to reason about the root cause
        and sketch a high-level fix strategy.  The resulting plan is typically
        injected into the patch-proposal prompt on the next turn.

        Args:
            prompt: The natural-language bug report or task description.
            context_dir: Directory where previously gathered context artefacts
                are stored.
            cwd: Root of the repository / working tree being debugged.

        Returns:
            A ``PatchPlanResult`` containing the plan text and raw model
            response, or ``None`` if this patcher does not support a planning
            step.
        """
        return None

    async def aplan_patch(
        self, *, prompt: str, context_dir: Path, cwd: Path
    ) -> PatchPlanResult | None:
        """Async variant of ``plan_patch``; delegates to the sync implementation by default."""
        return self.plan_patch(prompt=prompt, context_dir=context_dir, cwd=cwd)

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        """Produce a unified diff that fixes the reported bug.

        This is the only method that *must* be overridden by every concrete
        ``Patcher`` subclass.  It should incorporate any context or plan
        already written to *context_dir* and return a ``PatchResult`` whose
        ``diff`` field contains a valid unified diff, or ``None`` if the
        patcher was unable to generate one.

        Args:
            prompt: The natural-language bug report or task description.
            context_dir: Directory where previously gathered context artefacts
                and planning outputs are stored.
            cwd: Root of the repository / working tree being debugged.

        Returns:
            A ``PatchResult`` with a unified diff string (applicable via
            ``git apply``) in ``diff``, or ``diff=None`` on failure.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError

    async def apropose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        """Async variant of ``propose_patch``; delegates to the sync implementation by default."""
        return self.propose_patch(prompt=prompt, context_dir=context_dir, cwd=cwd)
