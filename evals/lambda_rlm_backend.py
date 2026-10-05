"""Controller-owned lambda-RLM backend for decomposition and compaction."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from baselines.openai_compat import env_client_subprocess_env, scrub_secret

from . import _json as json
from .runtime_resolution import OPENAI_API_KEY_ENV_FALLBACKS, resolve_openai_api_key_from_env

if TYPE_CHECKING:
    from baselines.controller_packet import ControllerPacketV1

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_CLIENT_SCRIPT = _REPO_ROOT / "baselines" / "lambda_rlm" / "env_client.py"
DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS = 100_000
DEFAULT_LAMBDA_RLM_TIMEOUT_SEC = 120
_MAX_SUMMARY_CHARS = 220
_MAX_QUESTION_CHARS = 220


def _truncate(text: str, *, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max(0, int(max_chars)):
        return value
    if max_chars <= 3:
        return value[: max(0, int(max_chars))]
    return value[: max(0, int(max_chars)) - 3].rstrip() + "..."


def default_lambda_rlm_repo_dir() -> Path:
    return _REPO_ROOT / "baselines" / "repos" / "lambda-RLM"


def default_lambda_rlm_python() -> Path:
    return _REPO_ROOT / "baselines" / "envs" / "lambda_rlm" / "bin" / "python"


def _normalize_title(value: str) -> str:
    lowered = str(value or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "decompose_step"


class LambdaRLMDecompositionStepModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    question: str
    supporting_routes: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("title", mode="before")
    @classmethod
    def _coerce_title(cls, value: object) -> str:
        return _normalize_title(str(value or ""))

    @field_validator("question", mode="before")
    @classmethod
    def _coerce_question(cls, value: object) -> str:
        return _truncate(str(value or ""), max_chars=_MAX_QUESTION_CHARS)

    @field_validator("supporting_routes", "evidence_ids", mode="before")
    @classmethod
    def _coerce_list(cls, value: object) -> list[object]:
        return list(value) if isinstance(value, list) else []

    def to_packet_dict(
        self,
        *,
        allowed_routes: set[str],
        allowed_evidence_ids: set[str],
    ) -> dict[str, Any]:
        supporting_routes = [item for item in self.supporting_routes if item in allowed_routes]
        evidence_ids = [item for item in self.evidence_ids if item in allowed_evidence_ids]
        return {
            "title": self.title,
            "question": self.question,
            "supporting_routes": supporting_routes[:4],
            "evidence_ids": evidence_ids[:4],
        }


class LambdaRLMDecompositionPayloadModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str = ""
    steps: list[LambdaRLMDecompositionStepModel] = Field(default_factory=list)

    @field_validator("summary", mode="before")
    @classmethod
    def _coerce_summary(cls, value: object) -> str:
        return _truncate(str(value or ""), max_chars=_MAX_SUMMARY_CHARS)


@dataclass(frozen=True)
class LambdaRLMDecompositionResult:
    summary: str
    steps: list[dict[str, Any]]
    usage_summary: dict[str, Any]
    execution_time_sec: float
    raw_response: str
    root_model: str


def _json_object_candidates(text: str) -> list[str]:
    stripped = str(text or "").strip()
    if not stripped:
        return []
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])
    return candidates


def parse_lambda_rlm_decomposition_text(
    text: str,
    *,
    allowed_routes: set[str],
    allowed_evidence_ids: set[str],
) -> LambdaRLMDecompositionResult:
    """Parse lambda-RLM JSON output into controller packet steps."""
    errors: list[str] = []
    for candidate in _json_object_candidates(text):
        try:
            payload = json.loads(candidate)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"invalid_payload_type:{type(payload).__name__}")
            continue
        try:
            model = LambdaRLMDecompositionPayloadModel.model_validate(payload)
        except ValidationError as exc:
            errors.append(str(exc))
            continue
        steps = [
            row.to_packet_dict(
                allowed_routes=allowed_routes,
                allowed_evidence_ids=allowed_evidence_ids,
            )
            for row in model.steps[:3]
        ]
        steps = [
            step
            for step in steps
            if str(step.get("title") or "").strip() and str(step.get("question") or "").strip()
        ]
        return LambdaRLMDecompositionResult(
            summary=model.summary,
            steps=steps,
            usage_summary={},
            execution_time_sec=0.0,
            raw_response=text,
            root_model="",
        )
    raise ValueError("lambda-RLM decomposition parse failed: " + " | ".join(errors[:3]))


def build_lambda_rlm_decomposition_prompt(controller_packet: ControllerPacketV1) -> str:
    """Build a bounded structured prompt for lambda-RLM decomposition."""
    evidence_bundle = controller_packet.evidence_bundle
    history_bundle = controller_packet.history_bundle
    verifier_report = controller_packet.verifier_report
    solver_hints = controller_packet.solver_hints
    evidence_summary = evidence_bundle.summary
    ast_context = evidence_bundle.ast_context
    repo_history = history_bundle.repo_history
    lines = [
        "You are assisting an outer adaptive controller for program repair.",
        "Return JSON only.",
        "",
        "JSON schema:",
        "{",
        '  "summary": "one short sentence",',
        '  "steps": [',
        "    {",
        '      "title": "snake_case_step_name",',
        '      "question": "actionable question for the downstream solver",',
        '      "supporting_routes": ["route_name"],',
        '      "evidence_ids": ["e1", "e2"]',
        "    }",
        "  ]",
        "}",
        "",
        "Rules:",
        "- Emit 1 to 3 steps.",
        "- Keep questions concrete and solver-facing.",
        "- supporting_routes must come from the provided allowed routes.",
        "- evidence_ids must come from the provided evidence ids.",
        "- Do not include prose outside the JSON object.",
        "",
        f"Case: {controller_packet.case_id}",
        f"Condition: {controller_packet.condition}",
        f"Attempt: {controller_packet.attempt}",
    ]
    if evidence_bundle.failure_signature:
        lines.append(f"Failure signature: {evidence_bundle.failure_signature}")
    if evidence_summary is not None:
        if evidence_summary.crash_anchor:
            lines.append(f"Crash anchor: {evidence_summary.crash_anchor}")
        if evidence_summary.top_facts:
            lines.append("Evidence facts:")
            lines.extend(f"- {fact}" for fact in evidence_summary.top_facts[:5])
    if ast_context is not None and ast_context.snippet:
        lines.append("AST crash snippet:")
        lines.append(_truncate(ast_context.snippet, max_chars=900))
    if history_bundle.lessons:
        lines.append("History lessons:")
        lines.extend(f"- {lesson.lesson}" for lesson in history_bundle.lessons[:4])
    if repo_history is not None and repo_history.crash_file_blame_summary:
        lines.append(f"Repo history hint: {repo_history.crash_file_blame_summary}")
    if verifier_report.status and verifier_report.status != "none":
        lines.append(f"Verifier status: {verifier_report.status}")
    if verifier_report.reasons:
        lines.append("Verifier reasons:")
        lines.extend(f"- {reason}" for reason in verifier_report.reasons[:4])
    if solver_hints.retry_plan:
        lines.append(f"Retry plan: {solver_hints.retry_plan}")
    lines.append("Allowed routes: " + ", ".join(solver_hints.tool_routes))
    lines.append(
        "Available evidence ids: "
        + ", ".join(item.id for item in evidence_bundle.items[:6] if item.id)
    )
    return "\n".join(lines).strip()


def invoke_lambda_rlm_decomposition(
    *,
    controller_packet: ControllerPacketV1,
    model: str,
    base_url: str,
    api_key: str,
    repo_dir: Path,
    python_executable: Path,
    context_window_chars: int,
    timeout_sec: int,
) -> LambdaRLMDecompositionResult:
    """Run lambda-RLM in its isolated env and parse the decomposition output."""
    prompt = build_lambda_rlm_decomposition_prompt(controller_packet)
    resolved_api_key = resolve_openai_api_key_from_env()
    effective_api_key = str(api_key or "").strip() or resolved_api_key.value
    if not effective_api_key:
        raise RuntimeError(
            "lambda-RLM requires an OpenAI-compatible API key; set --openai-api-key or one of "
            f"{', '.join(OPENAI_API_KEY_ENV_FALLBACKS)}"
        )
    payload = {
        "repo_dir": str(repo_dir),
        "model": str(model or ""),
        "prompt": prompt,
        "base_url": str(base_url or ""),
        "context_window_chars": max(1, int(context_window_chars)),
    }
    with tempfile.TemporaryDirectory(prefix="lambda_rlm_") as tmpdir:
        tmp_path = Path(tmpdir)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        request_path.write_text(json.dumps(payload), encoding="utf-8")
        completed = subprocess.run(
            [
                str(python_executable),
                str(_ENV_CLIENT_SCRIPT),
                str(request_path),
                str(response_path),
            ],
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
            cwd=str(repo_dir),
            env=env_client_subprocess_env(effective_api_key),
        )
        if completed.returncode != 0:
            detail = completed.stderr[-1200:] if completed.stderr else completed.stdout[-1200:]
            detail = scrub_secret(detail, effective_api_key)
            raise RuntimeError(f"lambda-RLM env client failed: {detail}")
        payload_raw = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, dict):
        raise RuntimeError(f"lambda-RLM env client returned {type(payload_raw).__name__}")
    raw_response = str(payload_raw.get("response") or "")
    parsed = parse_lambda_rlm_decomposition_text(
        raw_response,
        allowed_routes=set(controller_packet.solver_hints.tool_routes),
        allowed_evidence_ids={
            item.id for item in controller_packet.evidence_bundle.items if item.id
        },
    )
    usage_summary_raw = payload_raw.get("usage_summary")
    return LambdaRLMDecompositionResult(
        summary=parsed.summary,
        steps=parsed.steps,
        usage_summary=usage_summary_raw if isinstance(usage_summary_raw, dict) else {},
        execution_time_sec=float(payload_raw.get("execution_time") or 0.0),
        raw_response=raw_response,
        root_model=str(payload_raw.get("root_model") or ""),
    )


__all__ = [
    "DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS",
    "DEFAULT_LAMBDA_RLM_TIMEOUT_SEC",
    "LambdaRLMDecompositionResult",
    "build_lambda_rlm_decomposition_prompt",
    "default_lambda_rlm_python",
    "default_lambda_rlm_repo_dir",
    "invoke_lambda_rlm_decomposition",
    "parse_lambda_rlm_decomposition_text",
]
