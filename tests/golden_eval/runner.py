"""Run one golden eval scenario against a scripted stub model and normalize the output.

Usage::

    uv run python -m tests.golden_eval.runner            # compare every scenario
    uv run python -m tests.golden_eval.runner --update   # rewrite tests/golden/eval_runs/
    uv run python -m tests.golden_eval.runner --scenario native_default --out out.json

Each scenario runs ``run_eval.main`` in a fresh process (``--jobs 1``,
``--deterministic --seed 7``) on the synthetic cases in ``cases.py``. The
output keeps the result records, the policy event logs and state files, the
artifact tree and every request sent to the model, with timings, temporary
paths and ports replaced by placeholders. Two runs of the same code must
produce byte-identical output; a refactor is behaviour-preserving when its
output matches the committed golden file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import gzip
import hashlib
import io
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zlib
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llmdebug import _json as json

from .cases import GOLDEN_CASES, GoldenCase, PatchReply, render_diff, render_edits
from .scenarios import SCENARIOS

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "eval_runs"
RUN_ID = "golden"
# The base URL and the run root end up in settings fingerprints and prompt hashes,
# so both are fixed per scenario rather than random.
# User-private, so no other account can plant a symlink where the run root is deleted.
RUN_ROOT = Path.home() / ".cache" / "tracegate-golden-eval"


def stub_port(name: str) -> int:
    return 47000 + zlib.crc32(name.encode("utf-8")) % 900


assert len({stub_port(name) for name in SCENARIOS}) == len(SCENARIOS), "stub port collision"

_STAGE_A_PREFIX = "Return only JSON for staged context requests"
_EDITS_PREFIX = 'Return only JSON with this schema: {"edits"'
_DIFF_PREFIX = "You fix failing tests by returning a single unified diff"
_PLANNER_MARKER = "software repair planner"
_FULL_FILE_TARGET = re.compile(r"returning the complete repaired contents of (\S+?)\.\s")


# pytest prints "1 failed in 0.01s"; the duration would leak into prompt hashes and
# failure signatures, so each case freezes pytest's clock.
_FROZEN_CLOCK_CONFTEST = (
    "import _pytest.timing as _timing\n"
    "\n"
    "_timing.perf_counter = lambda: 0.0\n"
    "_timing.time = lambda: 0.0\n"
)


def _materialize_cases(root: Path) -> None:
    for case in GOLDEN_CASES:
        case_dir = root / "cases" / case.case_id
        case_dir.mkdir(parents=True)
        (case_dir / "case.json").write_text(json.dumps(case.metadata), encoding="utf-8")
        (case_dir / "conftest.py").write_text(_FROZEN_CLOCK_CONFTEST, encoding="utf-8")
        for name, text in case.files.items():
            (case_dir / name).write_text(text, encoding="utf-8")


def _message_text(body: dict[str, Any]) -> tuple[str, str]:
    system = ""
    parts: list[str] = []
    for message in body.get("messages") or []:
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        if message.get("role") == "system" and not system:
            system = text
        else:
            parts.append(text)
    return system, "\n".join(parts)


def _classify(system: str, user: str) -> str:
    if system.startswith(_STAGE_A_PREFIX):
        return "stage_a"
    if system.startswith(_EDITS_PREFIX):
        return "edits"
    if system.startswith(_DIFF_PREFIX):
        return "diff"
    if _PLANNER_MARKER in system:
        return "plan"
    if _FULL_FILE_TARGET.search(system):
        return "full_file"
    if "compact" in (system + user[:400]).lower():
        return "compact"
    return "other"


class ScriptedModel:
    """Answers chat completions from the case scripts; one patch counter per unit."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.patch_calls: dict[tuple[str, str], int] = defaultdict(int)
        self.kinds: list[str] = []

    def _active_unit(self, user: str) -> tuple[GoldenCase, str, dict[str, str]] | None:
        workdirs = self.root / "evals" / "artifacts" / RUN_ID / "workdirs"
        for case in GOLDEN_CASES:
            signature = next(iter(case.files.values())).splitlines()[0]
            if signature not in user and case.case_id not in user:
                continue
            case_dirs = sorted(p for p in (workdirs / case.case_id).glob("*") if p.is_dir())
            if not case_dirs:
                return case, "", dict(case.files)
            # --jobs 1: the most recently created workdir is the active unit.
            unit_dir = max(case_dirs, key=lambda p: p.stat().st_mtime_ns)
            files = {
                name: (unit_dir / name).read_text(encoding="utf-8")
                for name in case.files
                if (unit_dir / name).is_file()
            }
            return case, unit_dir.name, files
        return None

    def _patch_reply(self, kind: str, system: str, user: str) -> str:
        active = self._active_unit(user)
        if active is None:
            return "No change."
        case, unit, files = active
        self.patch_calls[(case.case_id, unit)] += 1
        reply: PatchReply = case.patch_script(self.patch_calls[(case.case_id, unit)], files)
        if kind == "edits":
            return render_edits(reply)
        if kind == "full_file":
            if reply.edit is None:
                return reply.raw or ""
            match = _FULL_FILE_TARGET.search(system)
            target = match.group(1) if match else reply.edit.path
            source = files.get(target, "")
            if target != reply.edit.path:
                return source
            return source.replace(reply.edit.find, reply.edit.replace, 1)
        return render_diff(reply, files)

    def _stage_a_reply(self, user: str) -> str:
        active = self._active_unit(user)
        requests = active[0].stage_a_requests if active is not None else []
        return json.dumps(
            {
                "diagnosis": "The failing line uses the wrong value.",
                "missing_fact": "the runtime value at the crash site" if requests else None,
                "need_more_context": bool(requests),
                "need_snapshot": bool(requests),
                "need_compact_context": False,
                "requests": requests,
                "snapshot_justification": "locals show the wrong value" if requests else "",
                "compact_context_reason": "",
                "sufficiency_rationale": "the traceback names the failing line",
            }
        )

    def respond(self, body: dict[str, Any]) -> str:
        system, user = _message_text(body)
        kind = _classify(system, user)
        self.kinds.append(kind)
        if kind == "stage_a":
            return self._stage_a_reply(user)
        if kind in {"edits", "diff", "full_file"}:
            return self._patch_reply(kind, system, user)
        if kind == "plan":
            return (
                "Diagnosis: the failing line uses the wrong value.\n"
                "Why previous attempt failed: it changed the wrong expression.\n"
                "New constraints: keep the edit minimal.\n"
                "Patch plan: fix the expression on the failing line.\n"
            )
        if body.get("response_format"):
            return json.dumps({"summary": "compacted context", "facts": []})
        return "OK"


def _completion(body: dict[str, Any], content: str) -> dict[str, Any]:
    system, user = _message_text(body)
    prompt_tokens = (len(system) + len(user)) // 4
    completion_tokens = max(1, len(content) // 4)
    return {
        "id": "chatcmpl-golden",
        "object": "chat.completion",
        "model": str(body.get("model") or "golden-model"),
        "seed": body.get("seed"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _systemone_answers(body: dict[str, Any]) -> dict[str, Any]:
    """Deterministic SystemOne answers: probability mass decreasing in option order."""
    answers: dict[str, Any] = {}
    for question_id, question in (body.get("questions") or {}).items():
        kind = question.get("type")
        if kind == "choice":
            options = list(question.get("criteria") or {})
            weights = [1.0 / (index + 1) for index in range(len(options))]
            total = sum(weights) or 1.0
            answers[question_id] = {
                "probabilities": {
                    option: weight / total for option, weight in zip(options, weights, strict=True)
                }
            }
        elif kind == "noul":
            answers[question_id] = {"noul": 0.25}
        elif kind == "score":
            answers[question_id] = {"score": 2}
    return {"answers": answers}


# --- normalization ----------------------------------------------------------------------

# Every record key that reads like a time. Each one must be classified below: the
# regex alone also matches configured budgets, reward coefficients and attempt counts,
# and normalizing those away made a refactor that miscomputes an effective deadline
# produce identical golden output.
_TIME_KEY = re.compile(
    r"(^|_)(sec|secs|seconds|ms|ns|ts|time|timestamp|duration|elapsed|latency|at|eta)"
    r"(_[a-z0-9]+)?$"
)
# Measurements: wall-clock durations, latencies and timestamps, replaced by a
# placeholder because they differ on every run.
_MEASURED_TIME_KEYS = frozenset(
    {
        "adaptive_budget_local_wall_sec",
        "adaptive_budget_mean_local_wall_sec",
        "adaptive_budget_reward_latency_penalty",
        "adaptive_controller_prepare_sec",
        "adaptive_controller_sidecar_build_sec",
        "adaptive_controller_sidecar_write_sec",
        "adaptive_controller_summarizer_sec",
        "adaptive_online_dm_latency_ms",
        "adaptive_online_local_wall_sec",
        "adaptive_online_mean_local_wall_sec",
        "adaptive_online_reward_latency_penalty",
        "after_duration_sec",
        "apply_duration_sec",
        "attempt_shared_propose_budget_remaining_sec",
        "attempt_shared_propose_call_budget_sec",
        "attempt_wall_clock_sec",
        "baseline_case_conversion_sec",
        "baseline_eval_sec",
        "baseline_internal_total_sec",
        "baseline_orchestration_sec",
        "baseline_sidecar_load_sec",
        "before_duration_sec",
        "compact_context_duration_sec",
        "context_compaction_llm_sec",
        "context_compaction_total_sec",
        "created_at_utc",
        "cumulative_runtime_sec",
        "duration_sec",
        "generated_at_utc",
        "json_generation_duration_sec",
        "latency_ms",
        "llm_total_sec",
        "llmdebug_total_sec",
        "local_wall_sec",
        "max_propose_sec",
        "mean_runtime_sec",
        "no_diff_recovery_duration_sec",
        "overhead_sec",
        "patch_apply_sec",
        "patch_verifier_recovery_duration_sec",
        "plan_patch_duration_sec",
        "prompt_assembly_sec",
        "propose_duration_sec",
        "recovery_llm_sec",
        "recovery_total_sec",
        "remaining_budget_sec",
        "request_context_duration_sec",
        "retry_gate_carried_wall_sec",
        "retry_plan_then_patch_duration_sec",
        "reward_latency_penalty",
        "runner_forced_patch_build_sec",
        "saved_at_utc",
        "server_ready_probe_duration_sec",
        "snapshot_read_sec",
        "stage_a_context_fulfillment_sec",
        "stage_a_llm_sec",
        "stage_a_total_sec",
        "stage_b_llm_sec",
        "started_at_utc",
        "syntax_preflight_sec",
        "tep_learning_llm_sec",
        "test_after_sec",
        "test_before_sec",
        "time",
        "timestamp_utc",
        "transport_recovery_wait_sec",
    }
)
# Configured budgets and deadlines, reward coefficients, attempt counts and the rates
# derived from them: deterministic, behaviour-defining, and therefore compared.
_CONFIGURED_TIME_KEYS = frozenset(
    {
        "adaptive_budget_applied_time_attempts",
        "adaptive_budget_applied_time_rate",
        "adaptive_budget_reward_eta_progress",
        "adaptive_budget_reward_latency_scale",
        "adaptive_budget_reward_mu_latency",
        "adaptive_budget_selected_time_attempts",
        "adaptive_budget_selected_time_rate",
        "adaptive_controller_lambda_rlm_timeout_sec",
        "adaptive_decision_model_timeout_sec",
        "adaptive_online_reward_eta_progress",
        "adaptive_online_reward_latency_scale",
        "adaptive_online_reward_mu_latency",
        "adaptive_reward_eta_progress",
        "adaptive_reward_latency_scale",
        "adaptive_reward_mu_latency",
        "attempt_shared_propose_budget_base_sec",
        "attempt_shared_propose_budget_sec",
        "base_propose_budget_sec",
        "budget_can_raise_time",
        "effective_timeout_sec",
        "openai_max_propose_sec",
        "openai_timeout_sec",
        "openai_transport_recovery_max_wait_sec",
        "openai_transport_recovery_poll_sec",
        "openai_transport_retry_backoff_base_sec",
        "openai_transport_retry_backoff_max_sec",
        "propose_budget_sec",
        "timeout_sec",
        "transport_retry_backoff_base_sec",
        "transport_retry_backoff_max_sec",
    }
)


def is_measured_time(key: str) -> bool:
    """Whether *key*'s value is a measurement that must be normalized away.

    An allowlist, not a pattern: an unclassified time-like key stays in the comparison,
    so a new budget or coefficient cannot silently escape the lock. A new measurement
    makes two runs differ instead, and ``test_every_time_like_key_is_classified`` names
    it before that happens.
    """
    return bool(key) and key.lower() in _MEASURED_TIME_KEYS


_PYTEST_TIMING = re.compile(r"\b(passed|failed|error|errors|skipped)( in) \d+(\.\d+)?s\b")
_ADDRESS = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")
_VERDICT_FILE = re.compile(r"(tracegate_pytest_verdict_)[0-9a-f]{32}")
# Host-specific values; the golden files are tied to the Python minor version instead.
_HOST_KEYS = frozenset({"platform", "hostname", "pid"})
# Random identifiers (uuid4); aliased in order of first appearance to keep the links.
_RANDOM_ID_KEY = re.compile(r"decision_ids?$")
# Snapshots of pytest's own frames hold object reprs and id()-keyed dicts; their lengths
# vary by a few characters per process, and these size estimates follow.
_OBJECT_ID_KEY = re.compile(r"^\d{9,}$")
_SIZE_ESTIMATE_KEY = re.compile(r"^adaptive_toon_(encoded_tokens|json_tokens|savings_tokens)")
_SIZE_ESTIMATE_PLACEHOLDER = "<SIZE>"


class Normalizer:
    def __init__(self, root: Path, port: int) -> None:
        self.replacements = [
            (str(root.resolve()), "<ROOT>"),
            (str(root), "<ROOT>"),
            (str(REPO_ROOT), "<REPO>"),
            (sys.executable, "<PYTHON>"),
            (f"127.0.0.1:{port}", "127.0.0.1:<PORT>"),
        ]
        self.replacements.sort(key=lambda item: len(item[0]), reverse=True)
        self.aliases: dict[str, str] = {}

    def alias(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.alias(item) for item in value]
        if not isinstance(value, str) or not value:
            return value
        return self.aliases.setdefault(value, f"<ID{len(self.aliases) + 1}>")

    def text(self, value: str) -> str:
        for old, new in self.replacements:
            value = value.replace(old, new)
        value = _PYTEST_TIMING.sub(r"\1\2 <T>s", value)
        value = _VERDICT_FILE.sub(r"\1<UUID>", value)
        return _ADDRESS.sub("0x<ADDR>", value)

    def value(self, value: Any, key: str = "") -> Any:
        if key in _HOST_KEYS:
            return "<HOST>"
        if _RANDOM_ID_KEY.search(key):
            return self.alias(value)
        if key == "python_version" and isinstance(value, str):
            return ".".join(value.split(".")[:2])
        if is_measured_time(key):
            if isinstance(value, bool) or value is None:
                return value
            return "<T>" if isinstance(value, (int, float, str)) else value
        if _SIZE_ESTIMATE_KEY.search(key):
            # Missingness is a recording regression, so None is kept; the magnitude is
            # not comparable because the encoded snapshot holds pytest's own object
            # reprs, whose length varies by a few characters per process.
            return value if value is None else _SIZE_ESTIMATE_PLACEHOLDER
        if isinstance(value, dict):
            return {
                (f"<OBJID{index}>" if _OBJECT_ID_KEY.match(str(k)) else str(k)): self.value(
                    v, str(k)
                )
                for index, (k, v) in enumerate(value.items())
            }
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value


def _fixed_times(value: Any, key: str = "") -> Any:
    """Replace measured times by constants so the analysis tools see stable inputs."""
    if is_measured_time(key):
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return type(value)(1)
        if isinstance(value, str):
            return "2026-01-01T00:00:00Z"
    if isinstance(value, dict):
        return {k: _fixed_times(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_fixed_times(item) for item in value]
    return value


def _run_analysis(root: Path, normalizer: Normalizer) -> dict[str, Any]:
    """Run the analysis and report tools over the run's records, with times fixed."""
    from evals import analyze_results, eval_summary, report_run_metrics

    records = _read_jsonl(root / "evals" / "results" / f"{RUN_ID}.jsonl")
    analysis_dir = root / "analysis"
    analysis_dir.mkdir()
    jsonl = analysis_dir / f"{RUN_ID}.jsonl"
    jsonl.write_text(
        "".join(json.dumps(_fixed_times(record)) + "\n" for record in records), encoding="utf-8"
    )
    summary_json = analysis_dir / "summary.json"
    outputs: dict[str, Any] = {}
    commands = {
        "analyze_results": (
            analyze_results.main,
            [str(jsonl), "--stats", "--aggregation", "any_success", "--out", str(summary_json)],
        ),
        "report_run_metrics": (report_run_metrics.main, ["--jsonl", str(jsonl)]),
        "eval_summary": (eval_summary.main, [RUN_ID, "--results-dir", str(analysis_dir)]),
    }
    for tool, (entry, argv) in commands.items():
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exit_code = entry(argv)
        outputs[tool] = {"exit_code": exit_code, "stdout": normalizer.text(buffer.getvalue())}
    outputs["analyze_results"]["summary"] = normalizer.value(
        json.loads(summary_json.read_text(encoding="utf-8"))
    )
    return outputs


def _read_jsonl(path: Path) -> list[Any]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _collect(root: Path, normalizer: Normalizer, requests: list[dict[str, Any]]) -> dict[str, Any]:
    results_dir = root / "evals" / "results"
    artifacts_dir = root / "evals" / "artifacts"
    output: dict[str, Any] = {"files": {}, "artifacts": {}, "requests": []}
    for path in sorted(results_dir.rglob("*")):
        # The run lock outlives the run (ORC-14) but only names its last holder's pid.
        if not path.is_file() or path.suffix == ".lock":
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix == ".jsonl":
            output["files"][rel] = normalizer.value(_read_jsonl(path))
        elif path.suffix == ".json":
            output["files"][rel] = normalizer.value(json.loads(path.read_text(encoding="utf-8")))
        else:
            output["files"][rel] = normalizer.text(path.read_text(encoding="utf-8"))
    for path in sorted(artifacts_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        if path.suffix in {".sqlite3", ".db", ".pyc"} or b"\x00" in raw[:1024]:
            output["artifacts"][rel] = "<binary>"
            continue
        if path.suffix in {".json", ".jsonl"}:
            try:
                normalized: Any = normalizer.value(
                    json.loads(raw.decode("utf-8")) if path.suffix == ".json" else _read_jsonl(path)
                )
            except ValueError:
                normalized = normalizer.text(raw.decode("utf-8", errors="replace"))
            digest_source = json.dumps(normalized, sort_keys=True)
        else:
            digest_source = normalizer.text(raw.decode("utf-8", errors="replace"))
        output["artifacts"][rel] = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
    for request in requests:
        output["requests"].append(
            {
                "method": request["method"],
                "path": request["path"],
                "json": normalizer.value(request.get("json")),
            }
        )
    return output


def _check_owned(directory: Path) -> os.stat_result:
    """Return *directory*'s lstat, refusing a symlink or another user's directory."""
    info = directory.lstat()
    if stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"refusing to use {directory}: a symlink or not owned by this user")
    return info


def prepare_run_root() -> Path:
    """Create the run root with private permissions, refusing an exposed parent.

    Ownership alone is not enough: in a group- or world-writable parent another account
    can replace a scenario directory or plant fixture content between the check and the
    run, and ``mkdir(exist_ok=True)`` does not tighten an existing permissive directory.
    """
    RUN_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_info = _check_owned(RUN_ROOT.parent)
    exposed = parent_info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if exposed and not parent_info.st_mode & stat.S_ISVTX:
        raise RuntimeError(
            f"refusing to use {RUN_ROOT.parent}: it is group- or world-writable; "
            "tighten it or point RUN_ROOT elsewhere"
        )
    # Owned, so tighten rather than refuse.
    if _check_owned(RUN_ROOT).st_mode & 0o077:
        RUN_ROOT.chmod(0o700)
    return RUN_ROOT


@contextlib.contextmanager
def scenario_lock(name: str) -> Iterator[None]:
    """Hold an exclusive cross-process lock on one scenario's run root and port.

    Two invocations of the same scenario share the directory and the stub port, so
    without the lock the second deletes the first's live work tree before failing on the
    occupied port, and its cleanup deletes it again.
    """
    prepare_run_root()
    lock_path = RUN_ROOT / f"{name}.lock"
    handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"scenario {name!r} is already running in {RUN_ROOT}; "
                "wait for it to finish instead of resetting its run root"
            ) from exc
        yield
    finally:
        os.close(handle)


def _fresh_run_root(name: str) -> Path:
    """Create an empty scenario root; call under :func:`scenario_lock`."""
    prepare_run_root()
    root = RUN_ROOT / name
    if root.is_symlink():
        raise RuntimeError(f"refusing to use {root}: it is a symlink")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(mode=0o700)
    return root


def run_scenario(name: str, keep_root: Path | None = None) -> dict[str, Any]:
    """Run one scenario under its cross-process lock and return its normalized output."""
    ensure_offline_tokenizer()
    with scenario_lock(name):
        return _run_scenario_locked(name, keep_root=keep_root)


def _run_scenario_locked(name: str, *, keep_root: Path | None = None) -> dict[str, Any]:
    from evals import run_eval
    from tests._openai_stub_server import OpenAIStubServer, StubbedResponse

    scenario = SCENARIOS[name]
    root = _fresh_run_root(name)
    try:
        _materialize_cases(root)
        model = ScriptedModel(root)

        chat_calls = 0

        def responder(method: str, path: str, body: Any) -> StubbedResponse | None:
            nonlocal chat_calls
            if method == "POST" and path.endswith("/chat/completions") and isinstance(body, dict):
                chat_calls += 1
                every = scenario.transport_fault_every
                if every and chat_calls % every == 0:
                    return StubbedResponse(503, {"error": "golden transport fault"}, {})
                content = model.respond(body)
                return StubbedResponse(200, _completion(body, content), {})
            if method == "POST" and path == "/v1/systemone" and isinstance(body, dict):
                return StubbedResponse(200, _systemone_answers(body), {})
            if method == "GET" and path == "/health":
                return StubbedResponse(200, {"status": "ok"}, {})
            if method == "GET" and path == "/v1/models":
                return StubbedResponse(
                    200, {"object": "list", "data": [{"id": "golden-model"}]}, {}
                )
            return None

        run_eval._REPO_ROOT = root  # outputs land under the temporary root
        with OpenAIStubServer(responder, port=stub_port(name)) as server:
            argv = [
                "--cases-dir",
                "cases",
                "--run-id",
                RUN_ID,
                "--patcher",
                "openai",
                "--openai-base-url",
                server.base_url_v1,
                "--openai-model",
                "golden-model",
                "--openai-api-key",
                "golden-key",
                "--executor",
                "local",
                "--jobs",
                "1",
                "--deterministic",
                "--seed",
                "7",
                "--openai-transport-retry-backoff-base-sec",
                "0",
                "--openai-transport-retry-backoff-max-sec",
                "0",
                *(arg.replace("{stub}", server.base_url) for arg in scenario.argv),
            ]
            exit_code = run_eval.main(argv)
            port = int(server.base_url.rsplit(":", 1)[1])
            normalizer = Normalizer(root, port)
            output = _collect(root, normalizer, server.requests)
        output["analysis"] = _run_analysis(root, normalizer)
        output["exit_code"] = exit_code
        if keep_root is not None:
            shutil.copytree(root, keep_root, dirs_exist_ok=True)
        output["model_call_kinds"] = model.kinds
        output["argv"] = normalizer.value(argv)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return output


def structural_diff(expected: Any, actual: Any, path: str = "", limit: int = 40) -> list[str]:
    """Return up to ``limit`` leaf paths where two outputs differ."""
    lines: list[str] = []

    def walk(left: Any, right: Any, where: str) -> None:
        if len(lines) >= limit or left == right:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left:
                    lines.append(f"+ {where}/{key}")
                elif key not in right:
                    lines.append(f"- {where}/{key}")
                else:
                    walk(left[key], right[key], f"{where}/{key}")
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                lines.append(f"~ {where}: length {len(left)} -> {len(right)}")
            for index, (lhs, rhs) in enumerate(zip(left, right, strict=False)):
                walk(lhs, rhs, f"{where}[{index}]")
            return
        lines.append(f"~ {where}: {str(left)[:120]!r} -> {str(right)[:120]!r}")

    walk(expected, actual, path)
    return lines


def _dump(output: dict[str, Any]) -> str:
    return json.dumps(output, indent=2, sort_keys=True) + "\n"


def interpreter() -> str:
    """The interpreter for scenario runs; its path appears in every repro command."""
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.is_file() else sys.executable


# Only these are inherited by a scenario subprocess; everything else the caller exports
# (proxies, LLMDEBUG_*, PYTEST_ADDOPTS, ...) would change the recorded output without
# changing the recorded environment fingerprint.
_INHERITED_ENV_KEYS = ("HOME", "PATH", "TMPDIR", "USER", "LOGNAME")
# tiktoken caches its vocabulary in $TMPDIR by default, which is wiped; with a cold cache
# the token estimates in the records come from a download instead of local data.
TOKENIZER_CACHE = Path.home() / ".cache" / "tracegate-golden-eval-tokenizer"


def scenario_env() -> dict[str, str]:
    """The pinned environment of a scenario subprocess.

    Proxies are off so the loopback stub cannot be routed to a real host, and the capture
    and pytest settings that reach the prompts are pinned instead of inherited.
    """
    env = {key: os.environ[key] for key in _INHERITED_ENV_KEYS if key in os.environ}
    env.update(
        {
            "no_proxy": "*",
            "NO_PROXY": "*",
            "PYTEST_ADDOPTS": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "COLUMNS": "80",
            "TIKTOKEN_CACHE_DIR": str(TOKENIZER_CACHE),
        }
    )
    return env


def ensure_offline_tokenizer() -> None:
    """Point tiktoken at a persistent cache and fail loudly if it is not usable.

    ``estimate_text_token_count`` records a token count per request, and its tokenizer
    downloads its vocabulary when the cache is cold. Resolving it once here keeps the
    scenario runs offline and turns a missing tokenizer into one explicit error instead
    of silently different telemetry in every record.
    """
    from evals.token_counting import estimate_text_token_count

    TOKENIZER_CACHE.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TOKENIZER_CACHE))
    default_cache = Path(tempfile.gettempdir()) / "data-gym-cache"
    if default_cache.is_dir():
        # Reuse vocabulary already on this machine rather than fetching it again.
        for blob in default_cache.iterdir():
            target = TOKENIZER_CACHE / blob.name
            if blob.is_file() and not target.exists():
                shutil.copy2(blob, target)
    probe = estimate_text_token_count("golden", model="golden-model")
    if not probe.get("available"):
        raise RuntimeError(
            "golden runs need a local tokenizer: "
            f"{probe.get('error') or 'unavailable'}. Warm {TOKENIZER_CACHE} once with "
            "network access (python -c \"import tiktoken; tiktoken.get_encoding('o200k_base')\")"
        )


def environment() -> dict[str, str]:
    """What the golden files depend on besides the code under test.

    Prompts carry the checkout and interpreter paths, and pytest's output depends on the
    locked versions, so golden files only compare within the same environment.
    """
    lock = REPO_ROOT / "uv.lock"
    return {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "system": platform.system(),
        "repo_root": str(REPO_ROOT),
        "run_root": str(RUN_ROOT),
        "executable": interpreter(),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else "",
    }


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json.gz"


def read_golden(name: str) -> str:
    return gzip.decompress(golden_path(name).read_bytes()).decode("utf-8")


def write_golden(name: str, text: str) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden_path(name).write_bytes(gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0))


def environment_path() -> Path:
    return GOLDEN_DIR / "environment.json"


def golden_environment_mismatch() -> str | None:
    """Return why the committed golden files can't be compared here, or None."""
    if not environment_path().is_file():
        return "no golden environment recorded"
    recorded = json.loads(environment_path().read_text(encoding="utf-8"))
    current = environment()
    changed = sorted(key for key in current if recorded.get(key) != current[key])
    if changed:
        return "golden files were generated in a different environment: " + ", ".join(changed)
    return None


def run_in_subprocess(name: str, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [interpreter(), "-m", "tests.golden_eval.runner", "--scenario", name, "--out", str(out)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=scenario_env(),
    )


def run_scenarios(names: list[str], out_dir: Path) -> dict[str, str]:
    """Run scenarios in parallel subprocesses; return the error output of failed runs."""
    # Two subprocesses for one scenario would fight over its run root and stub port.
    unique = sorted(set(names))
    ensure_offline_tokenizer()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(unique) or 1) as pool:
        futures = {
            name: pool.submit(run_in_subprocess, name, out_dir / f"{name}.json") for name in unique
        }
        results = {name: future.result() for name, future in futures.items()}
    return {
        name: (result.stderr or result.stdout)[-4000:]
        for name, result in results.items()
        if result.returncode != 0
    }


# Report differences in the records before the analysis derived from them.
_DIFF_ORDER = ("exit_code", "files", "requests", "model_call_kinds", "artifacts", "analysis")


def compare(name: str, actual: str) -> list[str]:
    """Return a structural diff of ``actual`` against the golden file (empty if identical)."""
    expected = read_golden(name)
    if actual == expected:
        return []
    old, new = json.loads(expected), json.loads(actual)
    keys = [key for key in _DIFF_ORDER if key in old or key in new]
    keys += sorted((set(old) | set(new)) - set(keys))
    lines: list[str] = []
    for key in keys:
        lines += structural_diff(old.get(key), new.get(key), f"/{key}", limit=40 - len(lines))
        if len(lines) >= 40:
            break
    return lines or ["(formatting differs)"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), action="append")
    parser.add_argument("--out", type=Path, help="Write one scenario's output here.")
    parser.add_argument("--update", action="store_true", help="Rewrite the golden files.")
    parser.add_argument("--keep-root", type=Path, help="Copy the run's root directory here.")
    args = parser.parse_args(argv)

    if args.out:
        if not args.scenario or len(args.scenario) != 1:
            parser.error("--out needs exactly one --scenario")
        output = run_scenario(args.scenario[0], keep_root=args.keep_root)
        args.out.write_text(_dump(output), encoding="utf-8")
        return 0

    names = sorted(set(args.scenario or SCENARIOS))
    if not args.update:
        mismatch = golden_environment_mismatch()
        if mismatch is not None:
            print(f"cannot compare: {mismatch}.")
            print("The golden files lock behaviour in the environment that recorded them;")
            print("compare there, or regenerate on a commit whose behaviour you trust with")
            print("`just golden-evals-update` and commit the new environment.json with them.")
            return 2
    with tempfile.TemporaryDirectory(prefix="golden-out-") as tmp:
        errors = run_scenarios(names, Path(tmp))
        for name, error in errors.items():
            print(f"{name}: RUN FAILED\n{error}")
        if errors:
            return 1
        failures = 0
        for name in names:
            actual = (Path(tmp) / f"{name}.json").read_text(encoding="utf-8")
            if args.update:
                write_golden(name, actual)
                print(f"updated {golden_path(name).relative_to(REPO_ROOT)}")
                continue
            differences = compare(name, actual)
            if not differences:
                print(f"{name}: identical")
                continue
            failures += 1
            diff_path = REPO_ROOT / ".golden-diff" / f"{name}.json"
            diff_path.parent.mkdir(exist_ok=True)
            diff_path.write_text(actual, encoding="utf-8")
            print(f"{name}: DIFFERS (full output in {diff_path.relative_to(REPO_ROOT)})")
            for line in differences:
                print(f"  {line}")
    if args.update:
        environment_path().write_text(
            json.dumps(environment(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
