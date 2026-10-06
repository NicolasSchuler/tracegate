#!/usr/bin/env python3
"""Latency smoke test for the System-1 decision-model sidecar.

Sends canned SystemOne requests whose rendered states are roughly 1k, 2k and 4k tokens,
built through the same helpers the eval harness uses (``evals.decision_model.state`` and
``evals.decision_model.schema``), and reports p50/p95 latency per size.

Usage:
    just decision-model-smoke
    uv run python scripts/decision_model_smoke.py --repeats 5 --concurrency 3

Exits non-zero when the health check or any request fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.decision_model.client import SystemOneClient, SystemOneResult  # noqa: E402
from evals.decision_model.config import (  # noqa: E402
    DEFAULT_DECISION_MODEL_NAME,
    DecisionModelClientConfig,
    resolve_decision_model_api_key,
    resolve_decision_model_url,
)
from evals.decision_model.schema import (  # noqa: E402
    DEFAULT_PROFILE_NAME,
    HELPER_ACTIONS,
    PROFILE_NAMES,
    RETRY_HELPER_STRICT_JSON,
    build_questions,
    build_systemone_request,
    get_profile,
)
from evals.decision_model.state import DecisionStateInputs, build_decision_state  # noqa: E402

# The sidecar truncates by tokens; states are budgeted in characters. Clef serializes compact
# JSON of mostly-English traceback text, which runs close to 4 characters per token.
_CHARS_PER_TOKEN = 4
_SIZES: tuple[tuple[str, int], ...] = (("1k", 1000), ("2k", 2000), ("4k", 4000))
# The actions a probe run typically has available (no malformed helper answer to retry).
_AVAILABLE_ACTIONS = tuple(
    action for action in HELPER_ACTIONS if action != RETRY_HELPER_STRICT_JSON
)

_TRACEBACK_TEMPLATE = """\
_____________________ test_resolve_window[{index}] _____________________

    def test_resolve_window_{index}():
        session = Session(window="{index}s")
>       assert session.resolve_window() == {index}
E       AssertionError: assert None == {index}
E        +  where None = <Session window='{index}s'>.resolve_window()

tests/test_session_{index}.py:{line}: AssertionError
------------------------------ Captured log call -------------------------------
WARNING  app.session:session.py:{line} window '{index}s' did not match the duration pattern
Traceback (most recent call last):
  File "app/session.py", line {line}, in resolve_window
    return self._parse_duration(self.window)
  File "app/duration.py", line {other_line}, in _parse_duration
    return _UNITS[match.group("unit")] * int(match.group("value"))
TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'
"""


@dataclass(frozen=True)
class SmokeRequest:
    """One canned request, tagged with the size bucket it belongs to."""

    label: str
    body: dict[str, Any]
    state_chars: int


@dataclass(frozen=True)
class SmokeOutcome:
    """Result of one sent request."""

    label: str
    result: SystemOneResult


def _synthetic_output(target_chars: int) -> str:
    """Build a realistic pytest failure report of at least *target_chars* characters."""
    chunks: list[str] = []
    total = 0
    index = 1
    while total < target_chars:
        chunk = _TRACEBACK_TEMPLATE.format(
            index=index,
            line=40 + index * 7,
            other_line=112 + index * 3,
        )
        chunks.append(chunk)
        total += len(chunk)
        index += 1
    chunks.append("=========================== short test summary info ===========================")
    chunks.append("FAILED tests/test_session_1.py::test_resolve_window_1 - AssertionError")
    chunks.append(f"{index - 1} failed, 4 passed in 1.83s")
    return "\n".join(chunks)


def _synthetic_inputs(output: str) -> DecisionStateInputs:
    return DecisionStateInputs(
        stdout=output,
        timed_out=False,
        attempt=1,
        max_attempts=2,
        helper_calls_remaining=1,
        compaction_calls_remaining=1,
        reveal_mode="full",
    )


def _rendered_chars(payload: Any) -> int:
    """Length of the state exactly as Clef's ``render()`` serializes it."""
    if isinstance(payload, str):
        return len(payload)
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _grow_state(payload: Any, output: str, target_chars: int) -> Any:
    """Extend the state's test-output tail until the rendered state reaches *target_chars*.

    The shipped profiles bound production states at a few thousand characters, so the larger
    buckets exist only to measure prefill latency at a realistic 2k/4k-token context.
    """
    deficit = target_chars - _rendered_chars(payload)
    if deficit <= 0:
        return payload
    filler = (output * (deficit // max(1, len(output)) + 1))[:deficit]
    if isinstance(payload, str):
        return f"{payload}\n{filler}"
    grown = dict(payload)
    tail_key = "9_test_output_tail"
    grown[tail_key] = f"{grown.get(tail_key, '')}\n{filler}"
    return grown


def _build_requests(*, model: str, profile_name: str) -> list[SmokeRequest]:
    """One request per size bucket, rendered with the real state and question builders."""
    profile = get_profile(profile_name)
    questions = build_questions(
        profile,
        available_actions=_AVAILABLE_ACTIONS,
        reveal_mode="full",
        lite_snapshot_mode="replace",
    )
    requests: list[SmokeRequest] = []
    for label, tokens in _SIZES:
        budget = tokens * _CHARS_PER_TOKEN
        output = _synthetic_output(budget)
        state = build_decision_state(_synthetic_inputs(output), profile=profile)
        payload = _grow_state(state.payload, output, budget)
        body = build_systemone_request(
            model=model,
            state_payload=payload,
            questions=questions,
        )
        requests.append(SmokeRequest(label=label, body=body, state_chars=_rendered_chars(payload)))
    return requests


def _send(client: SystemOneClient, request: SmokeRequest) -> SmokeOutcome:
    return SmokeOutcome(label=request.label, result=client.query(request.body))


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _report(requests: Sequence[SmokeRequest], outcomes: Sequence[SmokeOutcome]) -> int:
    """Print per-size statistics and the first successful answer; return an exit code."""
    chars_by_label = {request.label: request.state_chars for request in requests}
    failures = [outcome for outcome in outcomes if not outcome.result.ok]
    print("\nsize  state_chars  requests  ok  p50_ms   p95_ms  input_tokens")
    for label, _tokens in _SIZES:
        selected = [outcome for outcome in outcomes if outcome.label == label]
        if not selected:
            continue
        ok = [outcome for outcome in selected if outcome.result.ok]
        latencies = [outcome.result.latency_ms for outcome in ok]
        tokens_seen = _input_tokens(ok)
        print(
            f"{label:<5} {chars_by_label.get(label, 0):>11} {len(selected):>9} "
            f"{len(ok):>3} {_percentile(latencies, 0.5):>7.0f} "
            f"{_percentile(latencies, 0.95):>8.0f} {tokens_seen:>13}"
        )
    for outcome in failures:
        print(
            f"FAILED {outcome.label}: status={outcome.result.status} "
            f"http={outcome.result.http_status} error={outcome.result.error}"
        )
    first_ok = next((outcome for outcome in outcomes if outcome.result.ok), None)
    if first_ok is not None and first_ok.result.body is not None:
        print(f"\nanswers for one {first_ok.label} request:")
        _print_answers(first_ok.result.body)
    if failures:
        print(f"\n{len(failures)} of {len(outcomes)} requests failed")
        return 1
    print(f"\nall {len(outcomes)} requests succeeded")
    return 0


def _input_tokens(outcomes: Sequence[SmokeOutcome]) -> int:
    for outcome in outcomes:
        body = outcome.result.body or {}
        usage = body.get("usage")
        if isinstance(usage, dict):
            try:
                return int(usage.get("input_tokens", 0))
            except (TypeError, ValueError):
                return 0
    return 0


def _print_answers(body: dict[str, Any]) -> None:
    answers = body.get("answers")
    if not isinstance(answers, dict):
        print("  (response carried no answers object)")
        return
    for question_id, answer in answers.items():
        print(f"  {question_id}: {answer}")


def _health(client: SystemOneClient) -> int:
    result = client.health()
    if not result.ok or result.body is None:
        print(
            f"health check failed: status={result.status} http={result.http_status} "
            f"error={result.error}"
        )
        return 1
    fields = ("status", "model", "revision", "device", "dtype", "max_length")
    rendered = " ".join(f"{name}={result.body.get(name)!r}" for name in fields)
    print(f"health ok in {result.latency_ms:.0f} ms: {rendered}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latency smoke test for the System-1 decision-model sidecar."
    )
    parser.add_argument(
        "--url",
        default="",
        help="Sidecar base URL (default: $TRACEGATE_DECISION_MODEL_URL or http://127.0.0.1:8090).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_DECISION_MODEL_NAME,
        help=f"Model name sent in the request body (default: {DEFAULT_DECISION_MODEL_NAME}).",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILE_NAMES),
        default=DEFAULT_PROFILE_NAME,
        help=f"Question profile to render (default: {DEFAULT_PROFILE_NAME}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Requests per size bucket (default: 5).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of client threads sending concurrently (default: 1).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
        help="Read timeout per request; a cold GPU is slow (default: 120).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke test and return a process exit code."""
    args = _parse_args(argv)
    repeats = max(1, int(args.repeats))
    concurrency = max(1, int(args.concurrency))
    url = resolve_decision_model_url(str(args.url))
    config = DecisionModelClientConfig(
        url=url,
        model=str(args.model),
        timeout_sec=float(args.timeout_sec),
        api_key=resolve_decision_model_api_key(),
    )
    requests = _build_requests(model=str(args.model), profile_name=str(args.profile))
    schedule = [request for request in requests for _ in range(repeats)]
    print(
        f"decision model smoke test: url={url} profile={args.profile} "
        f"repeats={repeats} concurrency={concurrency}"
    )
    client = SystemOneClient(config)
    try:
        if _health(client) != 0:
            return 1
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outcomes = list(pool.map(lambda request: _send(client, request), schedule))
    finally:
        client.close()
    return _report(requests, outcomes)


if __name__ == "__main__":
    raise SystemExit(main())
