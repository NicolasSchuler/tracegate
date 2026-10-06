"""Offline probe: how well does a System-1 decision model judge our failures?

Runs each case's failing test once, builds the attempt-1 decision state exactly like the live
harness, asks the sidecar, and scores the diagnostic answers against the labels in ``case.json``.
No patch model and no repair budget are involved, so wording, thresholds and profiles can be
tuned cheaply before spending a comparison run.

The labels (``category``, ``difficulty``, ``snapshot_dependency``, ``bug_source``) are used for
scoring only. They never enter :class:`~evals.decision_model.state.DecisionStateInputs`, and the
temp workdir deliberately carries a neutral prefix so the case id cannot leak through a path.

Usage::

    python -m evals.decision_model.probe --limit 20 --report
    python -m evals.decision_model.probe --dry-run --limit 3   # build states, no sidecar
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .. import _json as json
from ..eval_config import (
    add_execution_arguments,
    build_execution_config_from_args,
    set_active_execution_config,
)
from .client import SYSTEMONE_PATH, SystemOneClient
from .config import (
    DEFAULT_DECISION_MODEL_NAME,
    DEFAULT_DECISION_MODEL_TIMEOUT_SEC,
    DEFAULT_LITE_SNAPSHOT_MODE,
    ROLE_SHADOW,
    DecisionModelClientConfig,
    DecisionModelPolicyConfig,
    DecisionModelSettings,
    resolve_decision_model_url,
)
from .policy import (
    ADVICE_STATUS_OK,
    DecisionModelAdvice,
    DecisionModelAdvisor,
    distribution_summary,
)
from .runtime import AdviceRequest, SystemOneAdvisor
from .schema import (
    COMPACT_THEN_PATCH,
    DEFAULT_PROFILE_NAME,
    FILE_ONLY,
    FRAME_LOCALS_LITE,
    FRAME_ONLY,
    HELPER_ACTIONS,
    INVOKE_HELPER,
    LITE_ACTIONS,
    LITE_SNAPSHOT_REPLACE,
    MISSING_CALL_PATH,
    MISSING_RUNTIME_VALUES,
    PROFILE_NAMES,
    QUESTION_NEXT_ACTION,
    RETRY_HELPER_STRICT_JSON,
    REVEAL_MODE_TRACEBACK_FIRST,
    SEARCH_ONLY,
    SKIP_HELPER,
    SNAPSHOT_FULL_LITE,
    SNAPSHOT_LITE,
    DecisionModelProfile,
    get_profile,
)
from .state import DecisionStateInputs, build_decision_state, redact_terms_for_unit

if TYPE_CHECKING:  # pragma: no cover - typing only, these imports pull in the harness
    from ..eval_types import CaseSpec, PreGateResult

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Neutral on purpose: ``validate_cases`` names its temp dir after the case id, which would put
# the category and difficulty into every path inside the test output.
_TEMP_DIR_PREFIX = "llmdebug-dmprobe-"
_DEFAULT_OUT_DIR = Path("evals/artifacts/decision_model_probe")
_COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")

DEFAULT_PREGATE_THRESHOLD = 0.7
DEFAULT_REVEAL_MODE = "full"
DEFAULT_OUTPUT_FORMAT = "json_compact"
# The probe always asks the attempt-1 question with the shipped k=2 / one-helper-call budget.
PROBE_ATTEMPT = 1
PROBE_MAX_ATTEMPTS = 2
PROBE_HELPER_CALLS_REMAINING = 1

STATUS_DRY_RUN = "dry_run"
SPLIT_DEV = "dev"
SPLIT_TEST = "test"
SPLIT_ALL = "all"
SPLIT_TEST_EVERY = 5
_DIFFICULTY_SUFFIXES = ("_easy", "_medium", "_hard")

DIFFICULTY_ORDINALS = {"easy": 0.0, "medium": 1.0, "hard": 2.0}
SNAPSHOT_DEPENDENCY_REQUIRED = "required"
SNAPSHOT_DEPENDENCY_NONE = "none"
SNAPSHOT_DEPENDENT = frozenset({SNAPSHOT_DEPENDENCY_REQUIRED, "helpful"})
CALIBRATION_BIN_COUNT = 5

_JsonRewrite = Callable[[Mapping[str, Any]], dict[str, Any]]
_BASE_ACTIONS = (SKIP_HELPER, INVOKE_HELPER, FRAME_ONLY, FILE_ONLY, SEARCH_ONLY)
# Clef sorts option ids alphabetically, so a fixed prefix permutation lets the probe measure how
# much of an answer is option-order bias rather than content.
_OPTION_KEY_PREFIXES = {
    SKIP_HELPER: "z",
    INVOKE_HELPER: "y",
    FRAME_ONLY: "a",
    FILE_ONLY: "b",
    SEARCH_ONLY: "c",
    SNAPSHOT_LITE: "x",
    FRAME_LOCALS_LITE: "d",
    SNAPSHOT_FULL_LITE: "w",
    COMPACT_THEN_PATCH: "e",
    RETRY_HELPER_STRICT_JSON: "v",
}


# --------------------------------------------------------------------------------------------
# Split assignment
# --------------------------------------------------------------------------------------------


def case_family(case_id: str) -> str:
    """Return the case family: the case id without its trailing difficulty suffix.

    Difficulty variants of one bug share a family so that a dev/test split cannot put two
    near-duplicates on both sides.
    """
    name = str(case_id)
    for suffix in _DIFFICULTY_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def split_for_case(case_id: str) -> str:
    """Assign ``dev`` or ``test`` deterministically from a stable hash of the case family."""
    digest = hashlib.sha256(case_family(case_id).encode("utf-8")).hexdigest()
    return SPLIT_TEST if int(digest[:16], 16) % SPLIT_TEST_EVERY == 0 else SPLIT_DEV


# --------------------------------------------------------------------------------------------
# Statistics (no new dependencies: the probe runs wherever the eval harness runs)
# --------------------------------------------------------------------------------------------


def average_ranks(values: Sequence[float]) -> list[float]:
    """Return 1-based ranks of *values* with tied values sharing their average rank."""
    numbers = [float(value) for value in values]
    order = sorted(range(len(numbers)), key=lambda index: numbers[index])
    ranks = [0.0] * len(numbers)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and numbers[order[end + 1]] == numbers[order[start]]:
            end += 1
        shared = (start + end) / 2.0 + 1.0
        for index in order[start : end + 1]:
            ranks[index] = shared
        start = end + 1
    return ranks


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Rank-based AUROC (ties averaged); ``None`` when one class is missing."""
    if len(scores) != len(labels) or not scores:
        return None
    flags = [bool(label) for label in labels]
    positives = sum(1 for flag in flags if flag)
    negatives = len(flags) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    positive_rank_sum = sum(rank for rank, flag in zip(ranks, flags, strict=True) if flag)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation; ``None`` for fewer than two points or a constant input."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rank_x = average_ranks(xs)
    rank_y = average_ranks(ys)
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)
    deltas = [(a - mean_x, b - mean_y) for a, b in zip(rank_x, rank_y, strict=True)]
    covariance = sum(dx * dy for dx, dy in deltas)
    variance_x = sum(dx * dx for dx, _ in deltas)
    variance_y = sum(dy * dy for _, dy in deltas)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def calibration_bins(scores: Sequence[float], labels: Sequence[bool]) -> list[dict[str, Any]]:
    """Equal-width reliability bins over ``[0, 1]`` (empty bins are kept for readability)."""
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(CALIBRATION_BIN_COUNT)]
    for score, label in zip(scores, labels, strict=True):
        value = min(1.0, max(0.0, float(score)))
        index = min(CALIBRATION_BIN_COUNT - 1, int(value * CALIBRATION_BIN_COUNT))
        buckets[index].append((value, bool(label)))
    bins: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        count = len(bucket)
        bins.append(
            {
                "lower": index / CALIBRATION_BIN_COUNT,
                "upper": (index + 1) / CALIBRATION_BIN_COUNT,
                "count": count,
                "mean_score": (sum(value for value, _ in bucket) / count) if count else None,
                "observed_rate": (
                    (sum(1.0 for _, label in bucket if label) / count) if count else None
                ),
            }
        )
    return bins


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linear-interpolated percentile (``fraction`` in ``[0, 1]``); ``None`` when empty."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, float(fraction))) * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# --------------------------------------------------------------------------------------------
# Option-key permutation (option-order bias check)
# --------------------------------------------------------------------------------------------


def permuted_option_key(action: str) -> str:
    """Return the renamed option id used by the option-order check."""
    return f"{_OPTION_KEY_PREFIXES.get(str(action), 'm')}_{action}"


def _original_option_keys() -> dict[str, str]:
    return {permuted_option_key(action): action for action in HELPER_ACTIONS}


def permute_request_options(body: Mapping[str, Any]) -> dict[str, Any]:
    """Rename the main question's option ids with the deterministic prefix permutation."""
    payload = dict(body)
    questions = payload.get("questions")
    if not isinstance(questions, Mapping):
        return payload
    question = questions.get(QUESTION_NEXT_ACTION)
    criteria = question.get("criteria") if isinstance(question, Mapping) else None
    if not isinstance(question, Mapping) or not isinstance(criteria, Mapping):
        return payload
    renamed = {permuted_option_key(str(key)): value for key, value in criteria.items()}
    updated = dict(questions)
    updated[QUESTION_NEXT_ACTION] = {**dict(question), "criteria": renamed}
    payload["questions"] = updated
    return payload


def restore_response_options(body: Mapping[str, Any]) -> dict[str, Any]:
    """Map renamed option ids in the main question's answer back to the original actions."""
    payload = dict(body)
    answers = payload.get("answers")
    answer = answers.get(QUESTION_NEXT_ACTION) if isinstance(answers, Mapping) else None
    if not isinstance(answers, Mapping) or not isinstance(answer, Mapping):
        return payload
    originals = _original_option_keys()
    restored: dict[str, Any] = {}
    for key, value in answer.items():
        if key == "probabilities" and isinstance(value, Mapping):
            restored[key] = {
                originals.get(str(name), str(name)): item for name, item in value.items()
            }
        elif isinstance(value, str) and value in originals:
            restored[key] = originals[value]
        else:
            restored[key] = value
    payload["answers"] = {**dict(answers), QUESTION_NEXT_ACTION: restored}
    return payload


def _rewrite_json(raw: bytes, rewrite: _JsonRewrite) -> bytes:
    try:
        body = json.loads(raw)
    except Exception:
        return raw
    if not isinstance(body, dict):
        return raw
    return json.dumps(rewrite(body), ensure_ascii=False).encode("utf-8")


def _headers_without_length(headers: httpx.Headers) -> list[tuple[bytes, bytes]]:
    dropped = {b"content-length", b"transfer-encoding"}
    return [(key, value) for key, value in headers.raw if key.lower() not in dropped]


class OptionKeyPermutingTransport(httpx.BaseTransport):
    """Renames the main question's option ids on the wire and maps the answer back.

    Wrapping the transport keeps the permutation local to the probe: neither ``schema`` nor
    ``runtime`` has to know that the option ids were scrambled.
    """

    def __init__(self, inner: httpx.BaseTransport | None = None) -> None:
        self._inner = inner if inner is not None else httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Permute the outgoing options, forward, and restore the incoming option ids."""
        if request.method != "POST" or not request.url.path.endswith(SYSTEMONE_PATH):
            return self._inner.handle_request(request)
        outbound = httpx.Request(
            method=request.method,
            url=request.url,
            headers=_headers_without_length(request.headers),
            content=_rewrite_json(request.content, permute_request_options),
            extensions=request.extensions,
        )
        response = self._inner.handle_request(outbound)
        try:
            raw = response.read()
        finally:
            response.close()
        return httpx.Response(
            status_code=response.status_code,
            headers=_headers_without_length(response.headers),
            content=_rewrite_json(raw, restore_response_options),
            request=request,
        )

    def close(self) -> None:
        """Release the wrapped transport."""
        self._inner.close()


# --------------------------------------------------------------------------------------------
# Probe execution
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeOptions:
    """Everything one probe run needs besides the case list and the advisor."""

    profile: DecisionModelProfile
    reveal_mode: str = DEFAULT_REVEAL_MODE
    lite_snapshot_mode: str = DEFAULT_LITE_SNAPSHOT_MODE
    output_format: str = DEFAULT_OUTPUT_FORMAT
    run_id: str = ""
    pregate_threshold: float = DEFAULT_PREGATE_THRESHOLD
    dry_run: bool = False


@dataclass(frozen=True)
class CaseObservation:
    """What the probe observed for one case before any decision-model query."""

    returncode: int
    timed_out: bool
    duration_sec: float
    snapshot_available: bool
    snapshot_error: str
    pregate: PreGateResult
    state_inputs: DecisionStateInputs
    available_actions: tuple[str, ...]


def available_actions_for_attempt_one(
    *,
    snapshot_available: bool,
    reveal_mode: str,
    lite_snapshot_mode: str = DEFAULT_LITE_SNAPSHOT_MODE,
) -> tuple[str, ...]:
    """Actions an OpenAI-compatible patcher would have on attempt 1 of ``layered_adaptive_llm``.

    ``compact_then_patch`` needs compaction enabled and ``retry_helper_strict_json`` needs a
    previous malformed helper answer, so neither is feasible here. The lite actions only change
    the patch prompt when their compact section replaces the full snapshot section.
    """
    actions: list[str] = list(_BASE_ACTIONS)
    lite_changes_evidence = (
        str(reveal_mode) == REVEAL_MODE_TRACEBACK_FIRST
        or str(lite_snapshot_mode) == LITE_SNAPSHOT_REPLACE
    )
    if snapshot_available and lite_changes_evidence:
        actions.extend(action for action in HELPER_ACTIONS if action in LITE_ACTIONS)
    return tuple(actions)


def observe_case(case: CaseSpec, *, options: ProbeOptions) -> CaseObservation:
    """Run the failing test once in a neutral temp dir and build the attempt-1 state inputs."""
    from ..adaptive_progress import _traceback_informativeness_pre_gate
    from ..eval_evidence import _strip_eval_only_case_artifacts
    from ..eval_types import Condition
    from ..pytest_runner import read_latest_snapshot, run_pytest

    with tempfile.TemporaryDirectory(prefix=_TEMP_DIR_PREFIX) as tmp:
        workdir = Path(tmp)
        shutil.copytree(case.path, workdir, dirs_exist_ok=True, ignore=_COPY_IGNORE)
        _strip_eval_only_case_artifacts(workdir)
        # LAYERED_ADAPTIVE_LLM (not BASE_REPAIR as in validate_cases): only snapshot conditions
        # load the llmdebug pytest plugin, and without a snapshot there is nothing to decide on.
        condition = Condition.LAYERED_ADAPTIVE_LLM
        run = run_pytest(
            workdir,
            list(case.python_args),
            condition,
            int(case.timeout_sec),
            options.output_format,
        )
        snapshot, snapshot_error = read_latest_snapshot(workdir / ".llmdebug")
        pregate = _traceback_informativeness_pre_gate(
            snapshot=snapshot,
            stdout=run.stdout,
            stderr=run.stderr,
            threshold=float(options.pregate_threshold),
        )
        redact_paths, redact_tokens = redact_terms_for_unit(
            workdirs=(str(workdir.resolve()), str(workdir)),
            case_id=case.case_id,
            condition=condition.value,
            run_id=options.run_id,
        )
        state_inputs = DecisionStateInputs(
            stdout=run.stdout,
            stderr=run.stderr,
            timed_out=bool(run.timed_out),
            snapshot=snapshot,
            attempt=PROBE_ATTEMPT,
            max_attempts=PROBE_MAX_ATTEMPTS,
            previous_failure_context=None,
            retry_history_turns=(),
            pregate=pregate,
            helper_calls_remaining=PROBE_HELPER_CALLS_REMAINING,
            compaction_calls_remaining=0,
            failure_signature_repeat=False,
            last_helper_action="",
            reveal_mode=options.reveal_mode,
            redact_paths=redact_paths,
            redact_tokens=redact_tokens,
        )
        return CaseObservation(
            returncode=int(run.returncode),
            timed_out=bool(run.timed_out),
            duration_sec=float(run.duration_sec),
            snapshot_available=snapshot is not None,
            snapshot_error=str(snapshot_error or ""),
            pregate=pregate,
            state_inputs=state_inputs,
            available_actions=available_actions_for_attempt_one(
                snapshot_available=snapshot is not None,
                reveal_mode=options.reveal_mode,
                lite_snapshot_mode=options.lite_snapshot_mode,
            ),
        )


def _case_labels(case: CaseSpec) -> dict[str, str]:
    return {
        "category": str(case.category),
        "difficulty": str(case.difficulty),
        "snapshot_dependency": str(case.snapshot_dependency),
        "bug_source": str(case.bug_source or "hand_crafted"),
    }


def _advice_fields(advice: DecisionModelAdvice) -> dict[str, Any]:
    answers = advice.answers
    top_action, top_prob, margin, entropy = distribution_summary(answers.action_probs)
    return {
        "status": advice.status,
        "model": advice.model,
        "profile": advice.profile,
        "action_probs": dict(answers.action_probs),
        "top_action": top_action,
        "top_prob": top_prob,
        "margin": margin,
        "entropy": entropy,
        "atomic_action": advice.atomic_action,
        "atomic_reason": advice.atomic_reason,
        "traceback_sufficient": answers.traceback_sufficient,
        "missing_information": dict(answers.missing_information),
        "fix_confidence": answers.fix_confidence,
        "latency_ms": float(advice.latency_ms),
        "input_tokens": int(advice.input_tokens),
        "state_sha256": advice.state_sha256,
        "state_chars": int(advice.state_chars),
        "truncated_sections": list(advice.truncated_sections),
        "error": advice.error,
    }


def _base_record(case: CaseSpec, observation: CaseObservation) -> dict[str, Any]:
    pregate = observation.pregate
    return {
        "case_id": case.case_id,
        "family": case_family(case.case_id),
        "split": split_for_case(case.case_id),
        "labels": _case_labels(case),
        "pytest": {
            "returncode": observation.returncode,
            "timed_out": observation.timed_out,
            "duration_sec": round(observation.duration_sec, 3),
        },
        "snapshot_available": observation.snapshot_available,
        "snapshot_error": observation.snapshot_error,
        "pregate": {
            "score": round(float(pregate.score), 4),
            "should_skip": bool(pregate.should_skip),
            "reason": str(pregate.reason),
        },
        "available_actions": list(observation.available_actions),
    }


def _dry_run_record(
    record: dict[str, Any], observation: CaseObservation, *, options: ProbeOptions
) -> dict[str, Any]:
    state = build_decision_state(observation.state_inputs, profile=options.profile)
    record["advice"] = {
        "status": STATUS_DRY_RUN,
        "profile": state.profile,
        "state_sha256": state.sha256,
        "state_chars": state.chars,
        "truncated_sections": list(state.truncated_sections),
    }
    record["state"] = state.rendered
    return record


def probe_case(
    case: CaseSpec, *, advisor: DecisionModelAdvisor | None, options: ProbeOptions
) -> dict[str, Any]:
    """Probe one case; execution or transport failures become a record, never an exception."""
    try:
        observation = observe_case(case, options=options)
    except Exception as exc:  # one broken case must not abort the probe
        return {
            "case_id": case.case_id,
            "family": case_family(case.case_id),
            "split": split_for_case(case.case_id),
            "labels": _case_labels(case),
            "advice": {"status": f"error:{type(exc).__name__}", "error": str(exc)[:400]},
        }
    record = _base_record(case, observation)
    if options.dry_run or advisor is None:
        return _dry_run_record(record, observation, options=options)
    advice = advisor.advise(
        AdviceRequest(
            state_inputs=observation.state_inputs,
            available_actions=observation.available_actions,
            lite_snapshot_mode=options.lite_snapshot_mode,
        )
    )
    record["advice"] = _advice_fields(advice)
    return record


def run_probe(
    cases: Sequence[CaseSpec],
    *,
    advisor: DecisionModelAdvisor | None,
    options: ProbeOptions,
    jobs: int = 1,
    writer: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Probe every case (optionally concurrently) and return the records in completion order."""
    records: list[dict[str, Any]] = []

    def _finish(record: Mapping[str, Any]) -> None:
        records.append(dict(record))
        if writer is not None:
            writer(dict(record))

    if max(1, int(jobs)) == 1:
        for case in cases:
            _finish(probe_case(case, advisor=advisor, options=options))
        return records
    with ThreadPoolExecutor(max_workers=int(jobs)) as executor:
        futures = {
            executor.submit(probe_case, case, advisor=advisor, options=options): case
            for case in cases
        }
        for future in as_completed(futures):
            _finish(future.result())
    return records


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredCase:
    """One probe record reduced to the numbers the report needs."""

    case_id: str
    split: str
    category: str
    difficulty: str
    snapshot_dependency: str
    top_action: str
    traceback_sufficient: float
    need_runtime_values: float
    pregate_score: float
    fix_confidence: float | None
    latency_ms: float


def _scored_case(record: Mapping[str, Any]) -> ScoredCase | None:
    advice = record.get("advice")
    if not isinstance(advice, Mapping) or advice.get("status") != ADVICE_STATUS_OK:
        return None
    sufficient = advice.get("traceback_sufficient")
    missing = advice.get("missing_information")
    if sufficient is None or not isinstance(missing, Mapping):
        return None
    labels = record.get("labels")
    labels = labels if isinstance(labels, Mapping) else {}
    pregate = record.get("pregate")
    pregate = pregate if isinstance(pregate, Mapping) else {}
    confidence = advice.get("fix_confidence")
    return ScoredCase(
        case_id=str(record.get("case_id") or ""),
        split=str(record.get("split") or ""),
        category=str(labels.get("category") or ""),
        difficulty=str(labels.get("difficulty") or ""),
        snapshot_dependency=str(labels.get("snapshot_dependency") or ""),
        top_action=str(advice.get("top_action") or advice.get("atomic_action") or ""),
        traceback_sufficient=float(sufficient),
        need_runtime_values=float(missing.get(MISSING_RUNTIME_VALUES, 0.0))
        + float(missing.get(MISSING_CALL_PATH, 0.0)),
        pregate_score=float(pregate.get("score", 0.0)),
        fix_confidence=None if confidence is None else float(confidence),
        latency_ms=float(advice.get("latency_ms", 0.0)),
    )


def _auroc_block(scored: Sequence[ScoredCase]) -> dict[str, dict[str, float | None]]:
    """AUROC of each uninformativeness signal against the snapshot-dependency labels."""
    sources = {
        # The decision model's two diagnostic signals, oriented so that higher means
        # "the traceback alone is not enough".
        "dm_not_traceback_sufficient": [1.0 - case.traceback_sufficient for case in scored],
        "dm_need_runtime_values": [case.need_runtime_values for case in scored],
        # The shipped heuristic on exactly the same cases, for comparison.
        "pregate_uninformative": [1.0 - case.pregate_score for case in scored],
    }
    targets = {
        "snapshot_dependency_required_or_helpful": [
            case.snapshot_dependency in SNAPSHOT_DEPENDENT for case in scored
        ],
        "snapshot_dependency_required": [
            case.snapshot_dependency == SNAPSHOT_DEPENDENCY_REQUIRED for case in scored
        ],
    }
    return {
        target: {source: auroc(values, labels) for source, values in sources.items()}
        for target, labels in targets.items()
    }


def _action_mix(scored: Sequence[ScoredCase]) -> dict[str, dict[str, int]]:
    mix: dict[str, Counter[str]] = {}
    for case in scored:
        mix.setdefault(case.category or "unknown", Counter())[case.top_action or "none"] += 1
    return {category: dict(sorted(counts.items())) for category, counts in sorted(mix.items())}


def _confidence_vs_difficulty(scored: Sequence[ScoredCase]) -> dict[str, Any]:
    pairs = [
        (case.fix_confidence, DIFFICULTY_ORDINALS[case.difficulty])
        for case in scored
        if case.fix_confidence is not None and case.difficulty in DIFFICULTY_ORDINALS
    ]
    return {
        "n": len(pairs),
        "rho": spearman([value for value, _ in pairs], [ordinal for _, ordinal in pairs]),
    }


def _metrics_for(scored: Sequence[ScoredCase]) -> dict[str, Any]:
    latencies = [case.latency_ms for case in scored if case.latency_ms > 0.0]
    return {
        "scored_cases": len(scored),
        "auroc": _auroc_block(scored),
        "spearman": {"fix_confidence_vs_difficulty": _confidence_vs_difficulty(scored)},
        "action_mix_by_category": _action_mix(scored),
        "calibration": {
            "traceback_sufficient_vs_snapshot_independent": calibration_bins(
                [case.traceback_sufficient for case in scored],
                [case.snapshot_dependency == SNAPSHOT_DEPENDENCY_NONE for case in scored],
            )
        },
        "latency_ms": {
            "n": len(latencies),
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
    }


def _state_chars_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sizes: list[float] = []
    for record in records:
        advice = record.get("advice")
        size = advice.get("state_chars") if isinstance(advice, Mapping) else None
        if isinstance(size, (int, float)) and size > 0:
            sizes.append(float(size))
    return {
        "n": len(sizes),
        "p50": percentile(sizes, 0.5),
        "p95": percentile(sizes, 0.95),
        "max": max(sizes) if sizes else None,
    }


def build_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate probe records into the JSON summary (pure; safe on partial records)."""
    statuses: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    for record in records:
        advice = record.get("advice")
        status = advice.get("status") if isinstance(advice, Mapping) else None
        statuses[str(status or "missing")] += 1
        splits[str(record.get("split") or "unknown")] += 1
    scored = [case for case in (_scored_case(record) for record in records) if case is not None]
    by_split = {
        split: _metrics_for([case for case in scored if case.split == split])
        for split in sorted({case.split for case in scored})
    }
    return {
        "cases": len(records),
        "counts": {
            "by_status": dict(sorted(statuses.items())),
            "by_split": dict(sorted(splits.items())),
        },
        "state_chars": _state_chars_summary(records),
        **_metrics_for(scored),
        "by_split": by_split,
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cases-dir", default="evals/cases", help="Cases directory.")
    parser.add_argument(
        "--case-id",
        "--case",
        action="append",
        default=[],
        dest="case_id",
        help="Case id to probe (repeatable).",
    )
    parser.add_argument("--category", action="append", default=[], help="Category filter.")
    parser.add_argument("--difficulty", action="append", default=[], help="Difficulty filter.")
    parser.add_argument(
        "--snapshot-dependency",
        action="append",
        default=[],
        help="Snapshot dependency filter (repeatable): required|helpful|none.",
    )
    parser.add_argument("--bug-source", action="append", default=[], help="Bug source filter.")
    parser.add_argument(
        "--split",
        choices=[SPLIT_DEV, SPLIT_TEST, SPLIT_ALL],
        default=SPLIT_DEV,
        help="Deterministic case-family split to probe (default: dev; tune on dev only).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max cases to probe (0 = all).")
    parser.add_argument("--jobs", type=int, default=1, help="Cases to probe concurrently.")


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=list(PROFILE_NAMES),
        default=DEFAULT_PROFILE_NAME,
        help=f"Versioned question profile (default: {DEFAULT_PROFILE_NAME}).",
    )
    parser.add_argument(
        "--url",
        default="",
        help="Sidecar base URL; defaults to $TRACEGATE_DECISION_MODEL_URL or the local default.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_DECISION_MODEL_NAME,
        help=f"Model name sent in the request body (default: {DEFAULT_DECISION_MODEL_NAME}).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=DEFAULT_DECISION_MODEL_TIMEOUT_SEC,
        help=f"Read timeout per query (default: {DEFAULT_DECISION_MODEL_TIMEOUT_SEC:g}).",
    )
    parser.add_argument(
        "--reveal-mode",
        choices=[DEFAULT_REVEAL_MODE, REVEAL_MODE_TRACEBACK_FIRST],
        default=DEFAULT_REVEAL_MODE,
        help="Reveal mode the state and option descriptions assume (default: full).",
    )
    parser.add_argument(
        "--rename-option-keys",
        action="store_true",
        help=(
            "Option-order check: send the action options under permuted ids and map the "
            "probabilities back, to see how much of the answer is option-order bias."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and record the decision states without querying the sidecar.",
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        default="",
        help=f"JSONL output path (default: {_DEFAULT_OUT_DIR}/<timestamp>.jsonl).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a JSON summary and write it next to the JSONL as <out>.report.json.",
    )
    parser.add_argument(
        "--output-format",
        default=DEFAULT_OUTPUT_FORMAT,
        choices=["json", "json_compact", "toon"],
        help="Snapshot output format env for the pytest run.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the probe CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m evals.decision_model.probe",
        description=(
            "Offline probe of a System-1 decision model against the eval case labels. "
            "Runs each failing test once, builds the attempt-1 decision state, queries the "
            "sidecar, and scores the diagnostic answers. No patch model, no repair budget."
        ),
    )
    _add_selection_arguments(parser)
    _add_model_arguments(parser)
    _add_output_arguments(parser)
    add_execution_arguments(parser)
    return parser


def select_cases(args: argparse.Namespace) -> list[CaseSpec]:
    """Load the cases directory and apply the selection filters, including the split."""
    from ..case_loading import load_cases

    cases_dir = Path(str(args.cases_dir))
    if not cases_dir.is_absolute():
        cases_dir = _REPO_ROOT / cases_dir
    cases = load_cases(cases_dir)
    filters = (
        (set(args.case_id), lambda case: case.case_id),
        ({item.strip() for item in args.category if item.strip()}, lambda case: case.category),
        ({item.strip() for item in args.difficulty if item.strip()}, lambda case: case.difficulty),
        (
            {item.strip() for item in args.snapshot_dependency if item.strip()},
            lambda case: case.snapshot_dependency,
        ),
        (
            {item.strip() for item in args.bug_source if item.strip()},
            lambda case: case.bug_source or "hand_crafted",
        ),
    )
    for wanted, key in filters:
        if wanted:
            cases = [case for case in cases if key(case) in wanted]
    if str(args.split) != SPLIT_ALL:
        cases = [case for case in cases if split_for_case(case.case_id) == str(args.split)]
    if int(args.limit) > 0:
        cases = cases[: int(args.limit)]
    return cases


def options_from_args(args: argparse.Namespace, *, run_id: str) -> ProbeOptions:
    """Build :class:`ProbeOptions` from parsed arguments."""
    return ProbeOptions(
        profile=get_profile(str(args.profile)),
        reveal_mode=str(args.reveal_mode),
        lite_snapshot_mode=LITE_SNAPSHOT_REPLACE,
        output_format=str(args.output_format),
        run_id=run_id,
        pregate_threshold=DEFAULT_PREGATE_THRESHOLD,
        dry_run=bool(args.dry_run),
    )


def build_advisor(args: argparse.Namespace) -> SystemOneAdvisor:
    """Create the advisor, optionally behind the option-key permuting transport."""
    settings = DecisionModelSettings(
        policy=DecisionModelPolicyConfig(role=ROLE_SHADOW, profile=str(args.profile)),
        client=DecisionModelClientConfig(
            url=resolve_decision_model_url(str(args.url)),
            model=str(args.model),
            timeout_sec=float(args.timeout_sec),
        ),
    )
    transport = OptionKeyPermutingTransport() if bool(args.rename_option_keys) else None
    return SystemOneAdvisor(settings, client=SystemOneClient(settings.client, transport=transport))


def _preflight(executor: str) -> tuple[bool, str]:
    if str(executor) == "local":
        from ..validate_cases import check_pytest_available

        return check_pytest_available()
    from ..eval_containers import check_testcontainers_runtime_available

    return check_testcontainers_runtime_available()


def _resolve_out_path(raw: str, *, timestamp: str) -> Path:
    path = Path(raw) if str(raw).strip() else _DEFAULT_OUT_DIR / f"{timestamp}.jsonl"
    if not path.is_absolute():
        path = _REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _progress_line(record: Mapping[str, Any]) -> str:
    advice = record.get("advice")
    advice = advice if isinstance(advice, Mapping) else {}
    parts = [
        f"[{record.get('split', '?')}]",
        str(record.get("case_id", "?")),
        f"status={advice.get('status', 'missing')}",
        f"state_chars={advice.get('state_chars', 0)}",
    ]
    if advice.get("top_action"):
        parts.append(f"top={advice['top_action']}")
    if advice.get("atomic_action"):
        parts.append(f"atomic={advice['atomic_action']}")
    if advice.get("error"):
        parts.append(f"error={advice['error']}")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: probe the decision model against the case labels."""
    args = build_parser().parse_args(argv)
    try:
        execution_config = build_execution_config_from_args(args)
    except ValueError as exc:
        print(f"Invalid arguments: {exc}")
        return 2
    if int(args.jobs) < 1:
        print("Invalid --jobs: must be >= 1")
        return 2
    try:
        cases = select_cases(args)
    except ValueError as exc:
        print(f"Invalid case metadata: {exc}")
        return 2
    if not cases:
        print("No cases selected for the probe.")
        return 2
    ok, message = _preflight(execution_config.executor)
    if not ok:
        print(f"Environment check failed: {message}")
        return 2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    options = options_from_args(args, run_id=f"dmprobe-{timestamp}")
    out_path = _resolve_out_path(str(args.out), timestamp=timestamp)
    advisor = None if options.dry_run else build_advisor(args)
    previous_execution_config = set_active_execution_config(execution_config)
    try:
        with out_path.open("w", encoding="utf-8") as handle:

            def _write(record: Mapping[str, Any]) -> None:
                handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
                handle.flush()
                print(_progress_line(record))

            records = run_probe(
                cases,
                advisor=advisor,
                options=options,
                jobs=int(args.jobs),
                writer=_write,
            )
    finally:
        set_active_execution_config(previous_execution_config)
        if advisor is not None:
            advisor.close()
    print(f"\nProbed {len(records)} case(s) -> {out_path}")
    if bool(args.report):
        report = build_report(records)
        rendered = json.dumps(report, indent=2, ensure_ascii=False)
        report_path = out_path.with_suffix(".report.json")
        report_path.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        print(f"Report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
