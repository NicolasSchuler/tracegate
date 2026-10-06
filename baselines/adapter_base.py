"""Abstract base class for external baseline adapters.

Each baseline (LDB, INTERVENOR, TraceCoder, lambda-RLM, RLM) subclasses
``BaselineAdapter`` to integrate with the llmdebug eval harness.  The
adapter is responsible for:

1. Converting llmdebug case directories into the baseline's input format.
2. Running the baseline's repair loop (calling the external tool).
3. Extracting a unified diff from the baseline's output.

The ``BaselinePatcher`` in ``evals.patchers.baseline_patcher`` wraps any
adapter into the standard ``Patcher`` protocol used by ``run_eval.py``.

Sidecar keys
~~~~~~~~~~~~
The harness injects up to four ``llmdebug_*`` keys into ``case_input``
after ``convert_case()`` returns:

- ``llmdebug_controller_packet`` -- full adaptive controller state
- ``llmdebug_evidence`` -- evidence bundle for the current attempt
- ``llmdebug_history`` -- retry history bundle
- ``llmdebug_verifier_report`` -- patch verifier report

These keys are additive: baseline-only runs never set them, while
adaptive or controller-backed runs may set any subset.  Adapters may
read them but should treat missing keys as normal.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import math
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from baselines.case_information import CASE_INFO_POLICY
from baselines.contracts import BASELINE_CONTRACT_VERSION, BaselineContract

if TYPE_CHECKING:
    from baselines.case_schemas import BaseCaseInput

logger = logging.getLogger(__name__)


def _empty_int_dict() -> dict[str, int]:
    return {}


def _empty_any_dict() -> dict[str, Any]:
    return {}


def _empty_file_edits() -> list[BaselineFileEdit]:
    return []


@dataclass(frozen=True)
class BaselineFileEdit:
    """Single text-file edit emitted by a multifile-capable baseline."""

    path: str
    content: str | None = None


@dataclass(frozen=True)
class BaselineResult:
    """Result from a single baseline repair attempt.

    Attributes:
        repaired_code: Legacy single-file repaired source code (full file), or
            *None* on failure.
        target_path: Relative path paired with ``repaired_code``. Defaults to
            ``buggy.py`` for the current single-file baselines.
        file_edits: Additive multifile-ready contract. Each edit contains a
            relative workspace path and replacement text (or *None* to delete
            the file).
        patch_diff: Optional adapter-supplied authoritative unified diff.
            When present it takes precedence over synthesized diffs.
        success: Whether the baseline's own internal evaluation (if any)
            considers the repair successful.  The eval harness re-runs pytest
            independently, so this is informational only.
        iterations_used: Number of debug/repair iterations the baseline
            actually executed (may be less than the budget).
        token_usage: Optional token-usage dict (prompt_tokens,
            completion_tokens, total_tokens) for cost tracking.
        meta: Arbitrary metadata from the baseline run (timings, errors,
            intermediate states, etc.).
    """

    repaired_code: str | None = None
    target_path: str = "buggy.py"
    file_edits: list[BaselineFileEdit] = field(default_factory=_empty_file_edits)
    patch_diff: str | None = None
    success: bool = False
    iterations_used: int = 0
    token_usage: dict[str, int] = field(default_factory=_empty_int_dict)
    meta: dict[str, Any] = field(default_factory=_empty_any_dict)


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration passed to every adapter.

    These mirror the eval-harness settings so that all conditions —
    llmdebug's own and external baselines — run under identical
    constraints.

    Attributes:
        model: Model name/alias for the OpenAI-compatible API.
        api_base_url: Base URL of the OpenAI-compatible endpoint.
        api_key: API key (may be empty for local servers).
        max_attempts: Number of internal iterations per adapter invocation.
            Controls the adapter's debug/repair loop count (e.g. TraceCoder
            ``--max-debug-attempts``, LDB ``--max_iters``, Intervenor/RLM
            loop bound).  Independent of the eval-harness retry count ``k``.
        api_timeout_sec: Timeout for one OpenAI-compatible request or adapter
            env-client call.
        process_timeout_sec: Timeout for one adapter-owned subprocess attempt.
        timeout_sec: Deprecated alias for ``process_timeout_sec`` retained for
            source compatibility.
        repo_dir: Path to the cloned baseline repository.
        python_executable: Optional interpreter for baseline-specific runtime
            dependencies. When unset, adapters use the current interpreter.
        extra: Baseline-specific key-value overrides.
    """

    model: str = ""
    api_base_url: str = "http://127.0.0.1:1234"
    api_key: str = ""
    max_attempts: int = 2
    api_timeout_sec: float | None = None
    process_timeout_sec: float | None = None
    timeout_sec: float | None = 300.0
    repo_dir: Path = Path(".")
    python_executable: Path | None = None
    extra: dict[str, Any] = field(default_factory=_empty_any_dict)
    generation: BaselineGenerationSettings = field(
        default_factory=lambda: BaselineGenerationSettings()
    )

    def __post_init__(self) -> None:
        process_timeout = self.process_timeout_sec
        legacy_timeout = self.timeout_sec
        if process_timeout is None and legacy_timeout is not None:
            process_timeout = float(legacy_timeout)
        if process_timeout is None:
            process_timeout = 300.0

        api_timeout = self.api_timeout_sec
        if api_timeout is None:
            api_timeout = float(process_timeout)

        object.__setattr__(self, "api_timeout_sec", max(0.0, float(api_timeout)))
        object.__setattr__(self, "process_timeout_sec", max(0.0, float(process_timeout)))
        object.__setattr__(self, "timeout_sec", max(0.0, float(process_timeout)))


def _git_output(repo: Path, *args: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, timeout=10
        )
    except Exception:
        return None
    return result.stdout


#: Above this size an untracked file contributes its length instead of its content hash.
_UNTRACKED_CONTENT_HASH_MAX_BYTES = 4 * 1024 * 1024


def _untracked_tree_digest(repo: Path, untracked: bytes) -> bytes:
    """Deterministic path + content digest for the untracked files in *repo* (BAS-14).

    The upstream patches add untracked modules (TraceCoder's ``src/exact_test_context.py``,
    for example). Editing one changes executable behaviour without changing either
    ``git diff HEAD`` or the file list, so the content has to be part of the identity.
    """
    lines: list[bytes] = []
    for raw in sorted(untracked.decode("utf-8", "replace").splitlines()):
        name = raw.strip()
        if not name:
            continue
        path = repo / name
        try:
            if not path.is_file():
                marker = "absent"
            elif path.stat().st_size > _UNTRACKED_CONTENT_HASH_MAX_BYTES:
                marker = f"size:{path.stat().st_size}"
            else:
                marker = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            marker = "unreadable"
        lines.append(f"{name}\0{marker}".encode())
    return b"\n".join(lines)


def observed_repo_state(repo: Path) -> dict[str, Any]:
    """Record what is actually checked out (BAS-05, BAS-14).

    ``tree_patch_sha256`` hashes the tracked diff against ``HEAD`` plus a path/content
    digest of the untracked files, i.e. the applied upstream patches and normalizations,
    so two runs can be compared even though the pinned commit alone doesn't identify the
    patched tree.
    """
    commit = (_git_output(repo, "rev-parse", "HEAD") or b"").decode("utf-8", "replace").strip()
    if not commit:
        return {"observed_commit": "", "dirty": None, "tree_patch_sha256": None}
    diff = _git_output(repo, "diff", "--binary", "HEAD") or b""
    untracked = _git_output(repo, "ls-files", "--others", "--exclude-standard") or b""
    untracked_digest = _untracked_tree_digest(repo, untracked)
    return {
        "observed_commit": commit,
        "dirty": bool(diff.strip() or untracked.strip()),
        "tree_patch_sha256": hashlib.sha256(diff + b"\0" + untracked_digest).hexdigest(),
    }


@dataclass(frozen=True)
class RepairDeadline:
    """One wall-clock deadline for a whole ``run_repair`` call (BAS-08, decision D4).

    ``BaselineConfig.process_timeout_sec`` is the budget for everything an adapter does for
    one case, so every blocking operation — an LLM request, an env-client subprocess, a
    pytest run — is clamped to what is left of it instead of starting with its own full
    timeout. A budget of ``0`` means "no deadline" (ORC-18).
    """

    started_at: float
    budget_sec: float

    @classmethod
    def from_config(
        cls, config: BaselineConfig, *, started_at: float | None = None
    ) -> RepairDeadline:
        """The deadline for one ``run_repair`` call, measured from *started_at*."""
        return cls(
            started_at=time.perf_counter() if started_at is None else float(started_at),
            budget_sec=max(0.0, float(config.process_timeout_sec or 0.0)),
        )

    @property
    def bounded(self) -> bool:
        """Whether a deadline applies at all."""
        return self.budget_sec > 0.0

    def remaining_sec(self) -> float:
        """Seconds left in the budget; ``inf`` when unbounded."""
        if not self.bounded:
            return math.inf
        return max(0.0, self.budget_sec - (time.perf_counter() - self.started_at))

    @property
    def expired(self) -> bool:
        """Whether the budget is used up, so no further operation may start."""
        return self.bounded and self.remaining_sec() <= 0.0

    def clamp_sec(self, requested: float | None) -> float:
        """*requested* seconds, never more than the remaining budget (never below 0.1)."""
        candidate = float(requested or 0.0)
        if candidate <= 0.0:
            candidate = self.remaining_sec()
        return max(0.1, min(candidate, self.remaining_sec()))

    def clamp_int_sec(self, requested: int | float | None) -> int:
        """Like :meth:`clamp_sec` for subprocess timeouts, in whole seconds (minimum 1).

        Rounded up, so a call that starts with the full budget still gets the timeout it
        asked for rather than one second less.
        """
        return max(1, math.ceil(self.clamp_sec(requested)))


def require_pinned_commit(*, name: str, observed_commit: str, pinned_commit: str) -> None:
    """Fail setup when the checkout is not the pinned upstream commit (BAS-05)."""
    if observed_commit and observed_commit != pinned_commit:
        raise RuntimeError(
            f"{name} repo is at {observed_commit}, not the pinned commit {pinned_commit}. "
            "Re-run baselines/setup_repos.sh or update baselines/pins.py."
        )


@dataclass(frozen=True)
class BaselineGenerationSettings:
    """Generation contract shared with the native conditions (decision D4, BAS-03/04/11).

    ``None`` fields keep an adapter's own default, which ``enforced_by`` reports. Adapters
    that run upstream code with hard-coded sampling (LDB, TraceCoder) receive the values
    as ``TRACEGATE_BASELINE_*`` environment variables but can't honour them until their
    upstream patches read those.
    """

    temperature: float | None = None
    seed: int | None = None
    max_completion_tokens: int | None = None
    transport_max_retries: int = 0
    transport_retry_backoff_sec: float = 1.0

    def to_env(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.temperature is not None:
            out["TRACEGATE_BASELINE_TEMPERATURE"] = repr(float(self.temperature))
        if self.seed is not None:
            out["TRACEGATE_BASELINE_SEED"] = str(int(self.seed))
        if self.max_completion_tokens is not None:
            out["TRACEGATE_BASELINE_MAX_TOKENS"] = str(int(self.max_completion_tokens))
        return out

    def record(self, *, enforced: bool) -> dict[str, Any]:
        """Per-record description of the effective contract."""
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "max_completion_tokens": self.max_completion_tokens,
            "transport_max_retries": int(self.transport_max_retries),
            "enforced": bool(enforced),
        }


@dataclass(frozen=True)
class BaselineCaseContext:
    """Runner-provided case metadata that is stripped from the patcher workdir.

    Dataset-derived labels (description, category, difficulty, snapshot dependency)
    are deliberately absent: see ``baselines.case_information``.
    """

    case_id: str
    python_args: tuple[str, ...]
    timeout_sec: int
    schema_version: str = BASELINE_CONTRACT_VERSION
    case_info_policy: str = CASE_INFO_POLICY


class BaselineCaseIncompatibleError(RuntimeError):
    """Raised when a case cannot be represented faithfully for a baseline."""

    def __init__(self, *, reason: str, detail: str = "") -> None:
        self.reason = str(reason or "unknown")
        self.detail = str(detail or "")
        message = self.reason if not self.detail else f"{self.reason}: {self.detail}"
        super().__init__(message)


class BaselineAdapter(ABC):
    """Abstract adapter for an external debugging/repair baseline.

    Subclass contract
    -----------------
    * ``name`` must be a short, unique identifier (``"ldb"``,
      ``"intervenor"``, ``"tracecoder"``).
    * ``default_model`` is the model this baseline was originally designed
      for.  The harness falls back to this when no model is configured
      explicitly, eliminating the hardcoded lookup table in
      ``BaselinePatcher``.
    * ``setup()`` is called once before any cases are processed.  Use it
      to validate the repo clone and check that dependencies are present.
    * ``convert_case()`` translates an llmdebug case directory into a
      typed ``BaseCaseInput`` subclass that ``run_repair()`` expects.
    * ``validate_extra_config()`` is called by the harness after
      construction to warn about unrecognised ``config.extra`` keys.
      The default implementation is a no-op; adapters that consume
      ``config.extra`` should override it.
    * ``run_repair()`` executes the baseline's debugging/repair loop on a
      single case and returns a ``BaselineResult``.
    * ``extract_diff()`` is a convenience that diffs ``BaselineResult.repaired_code``
      against ``buggy.py``.  The default implementation uses ``difflib``
      and should work for most baselines; override only if necessary.
    """

    honours_generation_contract: bool = False
    """Whether ``run_repair`` applies ``BaselineConfig.generation`` itself (decision D4)."""

    generation_override_keys: Mapping[str, str] = MappingProxyType({})
    """``config.extra`` keys that beat the shared contract, mapped to the recorded field.

    Adapters whose baseline-specific knob wins over ``BaselineConfig.generation`` declare it
    here so the record reports the value the request actually used.
    """

    retry_context_prompt_field: str | None = None
    """Free-text ``case_input`` field the harness may extend with retry context (BAS-02)."""

    repo_state: dict[str, Any] | None = None
    """Observed upstream checkout, set by ``setup`` (``observed_repo_state``; BAS-05/14)."""

    #: Fallback model used when the harness does not configure one.
    default_model: str = ""

    def effective_generation(self, config: BaselineConfig) -> dict[str, Any]:
        """The generation values this adapter really requests, for the record (D4).

        Resolves ``generation_override_keys`` on top of the shared contract so a
        baseline-specific knob is reported instead of the shared value it replaced.
        """
        record = config.generation.record(enforced=bool(self.honours_generation_contract))
        applied: dict[str, Any] = {}
        for extra_key, field_name in self.generation_override_keys.items():
            value = config.extra.get(extra_key)
            if value is None:
                continue
            resolved = int(value) if field_name.endswith("tokens") else float(value)
            record[field_name] = resolved
            applied[field_name] = extra_key
        if applied:
            record["overrides"] = applied
        return record

    def contract(self) -> BaselineContract:
        """Return machine-readable metadata for this adapter contract."""

        return BaselineContract(
            baseline_name=self.name,
            repair_scope_policy="adapter_defined",
            prompt_contract_kind="adapter_defined",
            output_contract="adapter_defined",
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier for this baseline (e.g. ``"ldb"``)."""

    @abstractmethod
    def setup(self, config: BaselineConfig) -> None:
        """One-time initialisation.

        Validate that ``config.repo_dir`` contains the expected baseline
        code, that required dependencies are importable, and that the LLM
        endpoint is reachable.

        Raises:
            RuntimeError: If the baseline cannot be initialised.
        """

    @abstractmethod
    def convert_case(
        self,
        case_dir: Path,
        *,
        case_context: BaselineCaseContext | None = None,
    ) -> BaseCaseInput:
        """Convert an llmdebug case directory into the baseline's input format.

        Parameters:
            case_dir: Path containing ``buggy.py``, ``test_buggy.py``,
                ``case.json``, and ``oracle.patch``.

        Returns:
            A typed ``BaseCaseInput`` subclass validated at construction
            time.  The harness calls ``.to_dict()`` on the result before
            injecting harness-specific keys and passing the merged dict
            to ``run_repair()``.
        """

    def validate_extra_config(self, config: BaselineConfig) -> None:  # noqa: B027
        """Warn about unrecognised keys in ``config.extra``.

        Called by the harness after ``setup()``.  The default
        implementation is a no-op.  Adapters that consume keys from
        ``config.extra`` should override this to validate and warn about
        unknown keys.
        """

    @abstractmethod
    def run_repair(
        self,
        case_input: dict[str, Any],
        config: BaselineConfig,
    ) -> BaselineResult:
        """Run the baseline's repair loop on a single case.

        Parameters:
            case_input: Output of ``convert_case().to_dict()`` with
                harness-injected ``llmdebug_*`` keys merged in.
            config: Shared run configuration (model, API, budget, etc.).

        Returns:
            A ``BaselineResult`` with repaired code (or *None* on failure).
        """

    # ------------------------------------------------------------------
    # Default diff extraction — override only if needed.
    # ------------------------------------------------------------------

    def extract_diff(
        self,
        case_dir: Path,
        result: BaselineResult,
    ) -> str | None:
        """Produce a unified diff from the baseline result.

        Precedence is:
        1. ``result.patch_diff``
        2. synthesized diff from ``result.file_edits``
        3. legacy ``result.repaired_code`` against ``result.target_path``

        Note that this is the *result-shape* precedence for diff extraction.
        ``BaselinePatcher`` records reporting metadata separately and may prefer
        explicit adapter metadata (``repair_scope`` / ``repair_targets``) when
        present.
        """
        if isinstance(result.patch_diff, str) and result.patch_diff.strip():
            return result.patch_diff

        if result.file_edits:
            return _synthesize_multifile_diff(case_dir=case_dir, file_edits=result.file_edits)

        if result.repaired_code is None:
            return None

        target_path = _normalize_result_path(result.target_path)
        original = _read_case_text(case_dir=case_dir, relative_path=target_path)
        return _make_file_diff(
            relative_path=target_path,
            original_text=original,
            updated_text=result.repaired_code,
        )

    def _progress_callback_var(
        self,
    ) -> ContextVar[Callable[[str, dict[str, Any] | None], None] | None]:
        callback_var = getattr(self, "_llmdebug_progress_callback_var", None)
        if isinstance(callback_var, ContextVar):
            return callback_var
        new_var: ContextVar[Callable[[str, dict[str, Any] | None], None] | None] = ContextVar(
            f"llmdebug_baseline_progress_callback_{id(self)}", default=None
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


def _normalize_result_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    if not normalized:
        raise ValueError("baseline result path is required")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute():
        raise ValueError(f"baseline result path must be relative: {path!r}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"baseline result path must not contain '.' or '..': {path!r}")
    return candidate.as_posix()


def _read_case_text(*, case_dir: Path, relative_path: str) -> str | None:
    target = case_dir.joinpath(*PurePosixPath(relative_path).parts)
    if not target.exists():
        return None
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"baseline diff only supports UTF-8 text files: {relative_path}") from exc


def _render_empty_file_diff_header(
    *,
    relative_path: str,
    original_text: str | None,
    updated_text: str | None,
) -> str | None:
    if original_text is None and updated_text == "":
        return (
            f"diff --git a/{relative_path} b/{relative_path}\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{relative_path}\n"
        )
    if original_text == "" and updated_text is None:
        return (
            f"diff --git a/{relative_path} b/{relative_path}\n"
            "deleted file mode 100644\n"
            f"--- a/{relative_path}\n"
            "+++ /dev/null\n"
        )
    return None


def _render_unified_diff_text(
    *,
    fromfile: str,
    tofile: str,
    original_text: str | None,
    updated_text: str | None,
) -> str:
    original_lines = [] if original_text is None else original_text.splitlines(keepends=True)
    updated_lines = [] if updated_text is None else updated_text.splitlines(keepends=True)
    diff_lines = difflib.unified_diff(
        original_lines,
        updated_lines,
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
    )

    rendered: list[str] = []
    for line in diff_lines:
        rendered.append(line if line.endswith("\n") else f"{line}\n")
        if _line_requires_no_newline_marker(line):
            rendered.append("\\ No newline at end of file\n")
    return "".join(rendered)


def _line_requires_no_newline_marker(line: str) -> bool:
    if line.endswith("\n"):
        return False
    if line.startswith(("--- ", "+++ ", "@@")):
        return False
    return line[:1] in {" ", "+", "-"}


def _make_file_diff(
    *,
    relative_path: str,
    original_text: str | None,
    updated_text: str | None,
) -> str | None:
    if original_text == updated_text:
        return None

    header = f"diff --git a/{relative_path} b/{relative_path}\n"
    if original_text is None:
        header += "new file mode 100644\n"
    elif updated_text is None:
        header += "deleted file mode 100644\n"

    diff_text = _render_unified_diff_text(
        fromfile="/dev/null" if original_text is None else f"a/{relative_path}",
        tofile="/dev/null" if updated_text is None else f"b/{relative_path}",
        original_text=original_text,
        updated_text=updated_text,
    )
    if diff_text:
        return header + diff_text
    return _render_empty_file_diff_header(
        relative_path=relative_path,
        original_text=original_text,
        updated_text=updated_text,
    )


def _synthesize_multifile_diff(
    *,
    case_dir: Path,
    file_edits: list[BaselineFileEdit],
) -> str | None:
    normalized_edits: dict[str, str | None] = {}
    for edit in file_edits:
        relative_path = _normalize_result_path(edit.path)
        if relative_path in normalized_edits:
            raise ValueError(f"duplicate baseline file edit for {relative_path!r}")
        normalized_edits[relative_path] = edit.content

    rendered_diffs: list[str] = []
    for relative_path in sorted(normalized_edits):
        rendered = _make_file_diff(
            relative_path=relative_path,
            original_text=_read_case_text(case_dir=case_dir, relative_path=relative_path),
            updated_text=normalized_edits[relative_path],
        )
        if rendered:
            rendered_diffs.append(rendered)
    return "".join(rendered_diffs) or None
