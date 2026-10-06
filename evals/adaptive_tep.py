"""Adaptive full-mode TEP helpers for the native OpenAI eval path.

This module keeps the full-mode TEP state, prompt-surface extraction, and
lightweight equilibrium/update heuristics isolated from the main attempt loop.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import pvariance
from typing import Any

from . import _json as json

TEP_STATE_SCHEMA_VERSION = 1

TEP_NODE_STAGE_A_REQUEST = "stage_a_request"
TEP_NODE_CONTEXT_COMPACTION = "context_compaction"
TEP_NODE_RETRY_PLANNER = "retry_planner"
TEP_NODE_PATCH_PROPOSER = "patch_proposer"

TEP_NODE_ORDER = (
    TEP_NODE_STAGE_A_REQUEST,
    TEP_NODE_CONTEXT_COMPACTION,
    TEP_NODE_RETRY_PLANNER,
    TEP_NODE_PATCH_PROPOSER,
)
TEP_REVERSE_NODE_ORDER = tuple(reversed(TEP_NODE_ORDER))

TEP_STAGE_B_MUTABLE_KEYS = (
    "instructions",
    "rca_checklist",
    "repair_method_scaffold",
    "cot_preamble",
    "loop_breaker",
    "diagnosis_refinement",
    "retry_reflection",
    "closing",
)

TEP_TASK_INDEPENDENT_CRITERIA = (
    "structural_clarity",
    "completeness",
    "consistency",
    "context_integration",
    "reasoning_transparency",
    "format_compliance",
)
TEP_IMPROVEMENT_KINDS = {"substantive", "stylistic", "stable", "degrading"}
TEP_CRITIC_SYSTEM_PROMPT = (
    "You are a local critic for one node in a staged software-repair graph. "
    "Score only the node-local prompt surface against the provided input, output, "
    "and downstream outcome packet. Return JSON only."
)
TEP_UPDATE_SYSTEM_PROMPT = (
    "You update mutable prompt parameters for a single node in a staged repair graph. "
    "Rewrite only the provided system prompts and actor blocks. "
    "Do not invent new block keys. Return JSON only."
)
TEP_CRITIC_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_score_10": {"type": "number"},
        "improvement_kind": {"type": "string"},
        "summary": {"type": "string"},
        "actionable_feedback": {"type": "array", "items": {"type": "string"}},
        "task_independent_scores": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "task_dependent_scores": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    },
    "required": ["overall_score_10", "improvement_kind", "actionable_feedback"],
    "additionalProperties": False,
}
TEP_UPDATE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "system_prompts": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "blocks": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "rationale": {"type": "string"},
    },
    "required": ["system_prompts", "blocks"],
    "additionalProperties": False,
}


def _default_stage_a_system_prompts() -> dict[str, str]:
    from .patchers.openai_compat_patcher import _request_context_system_prompt

    return {
        "standard": _request_context_system_prompt(strict_json=False),
        "strict_json": _request_context_system_prompt(strict_json=True),
    }


def _default_planner_system_prompts() -> dict[str, str]:
    from .patchers.openai_compat_patcher import _plan_patch_system_prompt

    return {"default": _plan_patch_system_prompt()}


def _default_compaction_system_prompts() -> dict[str, str]:
    return {
        "default": (
            "Return only JSON for context compaction. "
            'Use this schema exactly: {"compacted_context":"...","reduction_rationale":"..."}. '
            "The compacted_context must be <= {target_chars} characters. "
            "Preserve critical traceback facts, file paths, assertions, and state clues. "
            "No prose and no markdown."
        )
    }


def _default_patch_system_prompts() -> dict[str, str]:
    return {
        "edits": (
            "Return only JSON with this schema: "
            '{"edits":[{"path":"relative/path.py","find":"exact old text","replace":"new text","count":1}]}. '
            "No prose. No markdown. "
            "Use only exact existing paths from the prompt's available file list. "
            "Never use placeholder paths like relative/path.py."
        ),
        "diff": (
            "You fix failing tests by returning a single unified diff. "
            "Return ONLY the diff; no explanations. Do not use Markdown fences."
        ),
        "full_file": (
            "You fix failing tests by returning the complete repaired contents of {target_rel}. "
            "Return ONLY the full contents of {target_rel}. "
            "No prose. No markdown fences. Do not return a diff."
        ),
    }


def _default_system_prompts_for_node(node_id: str) -> dict[str, str]:
    if node_id == TEP_NODE_STAGE_A_REQUEST:
        return _default_stage_a_system_prompts()
    if node_id == TEP_NODE_CONTEXT_COMPACTION:
        return _default_compaction_system_prompts()
    if node_id == TEP_NODE_RETRY_PLANNER:
        return _default_planner_system_prompts()
    if node_id == TEP_NODE_PATCH_PROPOSER:
        return _default_patch_system_prompts()
    return {}


TEP_CRITIC_UNPARSED = "unparsed"
"""Improvement kind of a critic response without a usable score (POL-17)."""


def _parse_score(value: Any) -> float | None:
    """A finite numeric score, or ``None``; accepts numbers and strings like ``"7"`` or
    ``"7/10"`` (a bare ``float()`` raised on such strings)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str):
        text = value.strip().split("/", 1)[0].strip()
        try:
            score = float(text)
        except ValueError:
            return None
    else:
        return None
    return score if math.isfinite(score) else None


def _parse_score_map(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    parsed = {str(key): _parse_score(value) for key, value in raw.items()}
    return {key: value for key, value in parsed.items() if value is not None}


@dataclass
class TepCriticResult:
    overall_score_10: float
    actionable_feedback: list[str] = field(default_factory=list)
    improvement_kind: str = "stable"
    task_independent_scores: dict[str, float] = field(default_factory=dict)
    task_dependent_scores: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    critic_parsed: bool = True
    """False when the response had no usable ``overall_score_10`` (POL-17). A missing
    payload used to read as a ``stable`` critique with score 0, which counted as
    convergence."""

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> TepCriticResult:
        data = payload if isinstance(payload, dict) else {}
        feedback_raw = data.get("actionable_feedback")
        score = _parse_score(data.get("overall_score_10"))
        improvement_kind = str(data.get("improvement_kind") or "stable").strip().lower()
        if improvement_kind not in TEP_IMPROVEMENT_KINDS:
            improvement_kind = "stable"
        if score is None:
            improvement_kind = TEP_CRITIC_UNPARSED
        return cls(
            overall_score_10=score if score is not None else 0.0,
            actionable_feedback=[
                str(item).strip()
                for item in (feedback_raw if isinstance(feedback_raw, list) else [])
                if str(item).strip()
            ],
            improvement_kind=improvement_kind,
            task_independent_scores=_parse_score_map(data.get("task_independent_scores")),
            task_dependent_scores=_parse_score_map(data.get("task_dependent_scores")),
            summary=str(data.get("summary") or ""),
            critic_parsed=score is not None,
        )


@dataclass
class TepNodeTrace:
    node_id: str
    iterations: int = 0
    converged: bool = False
    convergence_reason: str = "not_run"
    scores: list[float] = field(default_factory=list)
    improvement_kinds: list[str] = field(default_factory=list)
    feedback_rounds: list[list[str]] = field(default_factory=list)
    input_summary: str = ""
    output_summary: str = ""
    critic_summary: str = ""

    @property
    def final_score(self) -> float:
        return float(self.scores[-1]) if self.scores else 0.0


@dataclass
class TepNodeValidationEntry:
    case_id: str
    score: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "score": float(self.score),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> TepNodeValidationEntry:
        data = payload if isinstance(payload, dict) else {}
        return cls(
            case_id=str(data.get("case_id") or ""),
            score=float(data.get("score") or 0.0),
            note=str(data.get("note") or ""),
        )


@dataclass
class TepNodeState:
    node_id: str
    actor_blocks: dict[str, str] = field(default_factory=dict)
    system_prompts: dict[str, str] = field(default_factory=dict)
    critic_prompt: str = ""
    task_criteria: list[str] = field(default_factory=list)
    temperature: float = 0.6
    update_count: int = 0
    accepted_updates: int = 0
    rejected_updates: int = 0
    validation_buffer: list[TepNodeValidationEntry] = field(default_factory=list)

    def ensure_default_system_prompts(self) -> None:
        defaults = _default_system_prompts_for_node(self.node_id)
        for key, value in defaults.items():
            self.system_prompts.setdefault(key, value)

    def append_validation_entry(
        self, *, case_id: str, score: float, note: str, max_size: int
    ) -> None:
        self.validation_buffer.append(
            TepNodeValidationEntry(case_id=case_id, score=float(score), note=str(note or ""))
        )
        if max_size > 0 and len(self.validation_buffer) > max_size:
            self.validation_buffer[:] = self.validation_buffer[-max_size:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "actor_blocks": dict(self.actor_blocks),
            "system_prompts": dict(self.system_prompts),
            "critic_prompt": self.critic_prompt,
            "task_criteria": list(self.task_criteria),
            "temperature": float(self.temperature),
            "update_count": int(self.update_count),
            "accepted_updates": int(self.accepted_updates),
            "rejected_updates": int(self.rejected_updates),
            "validation_buffer": [entry.to_dict() for entry in self.validation_buffer],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> TepNodeState:
        data = payload if isinstance(payload, dict) else {}
        task_criteria_raw = data.get("task_criteria")
        validation_buffer_raw = data.get("validation_buffer")
        state = cls(
            node_id=str(data.get("node_id") or ""),
            actor_blocks={
                str(key): str(value)
                for key, value in (
                    data.get("actor_blocks", {}).items()
                    if isinstance(data.get("actor_blocks"), dict)
                    else []
                )
            },
            system_prompts={
                str(key): str(value)
                for key, value in (
                    data.get("system_prompts", {}).items()
                    if isinstance(data.get("system_prompts"), dict)
                    else []
                )
            },
            critic_prompt=str(data.get("critic_prompt") or ""),
            task_criteria=[
                str(item)
                for item in (task_criteria_raw if isinstance(task_criteria_raw, list) else [])
            ],
            temperature=float(data.get("temperature") or 0.6),
            update_count=int(data.get("update_count") or 0),
            accepted_updates=int(data.get("accepted_updates") or 0),
            rejected_updates=int(data.get("rejected_updates") or 0),
            validation_buffer=[
                TepNodeValidationEntry.from_dict(item)
                for item in (
                    validation_buffer_raw if isinstance(validation_buffer_raw, list) else []
                )
            ],
        )
        state.ensure_default_system_prompts()
        return state


@dataclass
class AdaptiveTepState:
    schema_version: int = TEP_STATE_SCHEMA_VERSION
    beta: float = 1.0
    beta_decay: float = 0.9
    validation_buffer_size: int = 4
    nodes: dict[str, TepNodeState] = field(default_factory=dict)

    def ensure_nodes(self) -> None:
        for node_id in TEP_NODE_ORDER:
            if node_id not in self.nodes:
                node_state = TepNodeState(node_id=node_id)
                node_state.ensure_default_system_prompts()
                self.nodes[node_id] = node_state
            else:
                self.nodes[node_id].ensure_default_system_prompts()

    def node(self, node_id: str) -> TepNodeState:
        self.ensure_nodes()
        return self.nodes[node_id]

    def to_dict(self) -> dict[str, Any]:
        self.ensure_nodes()
        return {
            "schema_version": int(self.schema_version),
            "beta": float(self.beta),
            "beta_decay": float(self.beta_decay),
            "validation_buffer_size": int(self.validation_buffer_size),
            "nodes": {node_id: state.to_dict() for node_id, state in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> AdaptiveTepState:
        data = payload if isinstance(payload, dict) else {}
        state = cls(
            schema_version=int(data.get("schema_version") or TEP_STATE_SCHEMA_VERSION),
            beta=float(data.get("beta") or 1.0),
            beta_decay=float(data.get("beta_decay") or 0.9),
            validation_buffer_size=int(data.get("validation_buffer_size") or 4),
            nodes={
                str(node_id): TepNodeState.from_dict(node_payload)
                for node_id, node_payload in (
                    data.get("nodes", {}).items() if isinstance(data.get("nodes"), dict) else []
                )
            },
        )
        state.ensure_nodes()
        return state


def build_default_tep_state(
    *,
    beta_init: float = 1.0,
    beta_decay: float = 0.9,
    validation_buffer_size: int = 4,
) -> AdaptiveTepState:
    state = AdaptiveTepState(
        beta=float(beta_init),
        beta_decay=float(beta_decay),
        validation_buffer_size=max(0, int(validation_buffer_size)),
    )
    state.ensure_nodes()
    return state


def load_tep_state(
    *,
    state_path: str,
    init_state_path: str = "",
    beta_init: float = 1.0,
    beta_decay: float = 0.9,
    validation_buffer_size: int = 4,
) -> AdaptiveTepState:
    candidate_paths = [Path(state_path)] if str(state_path).strip() else []
    if str(init_state_path).strip():
        candidate_paths.append(Path(init_state_path))
    for path in candidate_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        except ValueError:
            continue
        state = AdaptiveTepState.from_dict(data)
        state.beta = float(state.beta or beta_init)
        state.beta_decay = float(state.beta_decay or beta_decay)
        if state.validation_buffer_size <= 0:
            state.validation_buffer_size = max(0, int(validation_buffer_size))
        state.ensure_nodes()
        return state
    return build_default_tep_state(
        beta_init=beta_init,
        beta_decay=beta_decay,
        validation_buffer_size=validation_buffer_size,
    )


def save_tep_state(*, state: AdaptiveTepState, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def extract_mutable_stage_b_blocks(sections: list[dict[str, Any]]) -> dict[str, str]:
    extracted: dict[str, str] = {}
    for section in sections:
        key = str(section.get("key") or "")
        text = section.get("text")
        if key in TEP_STAGE_B_MUTABLE_KEYS and isinstance(text, str):
            extracted.setdefault(key, text)
    return extracted


def resolve_equilibrium_status(
    *,
    scores: list[float],
    improvement_kinds: list[str],
    max_iterations: int,
    window: int,
    variance_threshold: float,
) -> tuple[bool, str]:
    if not scores:
        return False, "not_run"
    if max_iterations > 0 and len(scores) >= max_iterations:
        return True, "max_iterations"
    normalized_window = max(1, int(window))
    if len(scores) >= normalized_window:
        recent_scores = scores[-normalized_window:]
        if len(recent_scores) >= 2 and pvariance(recent_scores) <= float(variance_threshold):
            return True, "score_variance"
    if improvement_kinds:
        latest = str(improvement_kinds[-1] or "").strip().lower()
        if latest in {"stylistic", "stable"}:
            return True, latest
    return False, "continue"


def should_accept_candidate_update(
    *,
    free_score: float,
    nudged_score: float,
    validation_buffer: list[TepNodeValidationEntry],
) -> bool:
    """Accept when the candidate surface (*nudged_score*) beats the current one (*free_score*).

    Both scores come from the same critic prompt on two different surfaces, so a tie is
    no evidence and is rejected (POL-17: the two critiques used to score the same surface,
    so ``>=`` accepted about half of all updates on noise).
    """
    if float(nudged_score) <= float(free_score):
        return False
    if not validation_buffer:
        return True
    historical_floor = min(float(entry.score) for entry in validation_buffer)
    return float(nudged_score) >= historical_floor


def clamp_temperature(value: float) -> float:
    return max(0.3, min(0.9, float(value)))


def next_temperature(*, current: float, accepted: bool) -> float:
    factor = 0.95 if accepted else 1.05
    return clamp_temperature(float(current) * factor)


def next_beta(*, current: float, beta_decay: float, accepted: bool) -> float:
    if not accepted:
        return float(current)
    return max(0.0, float(current) * float(beta_decay))


def apply_update_payload(
    *,
    node_state: TepNodeState,
    payload: dict[str, Any] | None,
) -> None:
    data = payload if isinstance(payload, dict) else {}
    blocks_raw = data.get("blocks")
    if isinstance(blocks_raw, dict):
        for key, value in blocks_raw.items():
            text = str(value or "")
            if text:
                node_state.actor_blocks[str(key)] = text
    systems_raw = data.get("system_prompts")
    if isinstance(systems_raw, dict):
        for key, value in systems_raw.items():
            text = str(value or "")
            if text:
                node_state.system_prompts[str(key)] = text


def render_system_prompt_template(template: str, **replacements: str | int) -> str:
    rendered = str(template or "")
    for key, value in replacements.items():
        rendered = rendered.replace("{" + str(key) + "}", str(value))
    return rendered


def _truncate_text(value: str, *, max_chars: int = 2000) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _render_critic_prompt(
    *,
    node_id: str,
    phase: str,
    actor_surface: dict[str, Any],
    input_summary: str,
    output_summary: str,
    objective_packet: str,
    task_criteria: list[str],
) -> str:
    packet = {
        "node_id": node_id,
        "phase": phase,
        "task_independent_criteria": list(TEP_TASK_INDEPENDENT_CRITERIA),
        "task_dependent_criteria": list(task_criteria),
        "actor_surface": actor_surface,
        "input_summary": _truncate_text(input_summary),
        "output_summary": _truncate_text(output_summary),
        "objective_packet": _truncate_text(objective_packet),
    }
    return (
        "Score this node-local actor surface.\n"
        "Use the fixed task-independent criteria, the node-specific criteria, and the downstream objective packet.\n"
        "Mark `improvement_kind` as one of: substantive, stylistic, stable, degrading.\n"
        "Return JSON only.\n\n" + json.dumps(packet, indent=2, ensure_ascii=False)
    )


def _render_update_prompt(
    *,
    node_id: str,
    actor_surface: dict[str, Any],
    critic: TepCriticResult,
    objective_packet: str,
) -> str:
    packet = {
        "node_id": node_id,
        "actor_surface": actor_surface,
        "critic": {
            "overall_score_10": critic.overall_score_10,
            "improvement_kind": critic.improvement_kind,
            "summary": critic.summary,
            "actionable_feedback": list(critic.actionable_feedback),
        },
        "objective_packet": _truncate_text(objective_packet),
        "constraints": [
            "Rewrite only provided system prompts and actor blocks.",
            "Do not change graph membership or evidence sections.",
            "Keep edits local and behavior-focused.",
        ],
    }
    return (
        "Propose a candidate prompt update for this node.\n"
        "Use the critic's feedback to improve the actor surface.\n"
        "Return JSON only.\n\n" + json.dumps(packet, indent=2, ensure_ascii=False)
    )


class AdaptiveTepController:
    """Stateful helper for the native-only full-mode TEP path."""

    def __init__(
        self,
        *,
        state_path: str,
        init_state_path: str = "",
        beta_init: float = 1.0,
        beta_decay: float = 0.9,
        equilibrium_max_iters: int = 20,
        equilibrium_window: int = 3,
        equilibrium_variance_threshold: float = 0.5,
        validation_buffer_size: int = 4,
    ) -> None:
        self.state_path = str(state_path or "")
        self.state = load_tep_state(
            state_path=state_path,
            init_state_path=init_state_path,
            beta_init=beta_init,
            beta_decay=beta_decay,
            validation_buffer_size=validation_buffer_size,
        )
        self.equilibrium_max_iters = max(1, int(equilibrium_max_iters))
        self.equilibrium_window = max(1, int(equilibrium_window))
        self.equilibrium_variance_threshold = float(equilibrium_variance_threshold)
        self.latest_traces: dict[str, TepNodeTrace] = {}

    def save(self) -> str:
        """Write the state; returns ``""`` or the error (ATT-17: failures were swallowed,
        so later cases silently started from stale state)."""
        if not self.state_path:
            return ""
        try:
            save_tep_state(state=self.state, path=self.state_path)
        except (OSError, TypeError, ValueError) as exc:
            return f"{type(exc).__name__}: {exc}"
        return ""

    def node(self, node_id: str) -> TepNodeState:
        return self.state.node(node_id)

    def prime_stage_b_defaults(
        self, *, planning_mode: bool, sections: list[dict[str, Any]]
    ) -> None:
        node_id = TEP_NODE_RETRY_PLANNER if planning_mode else TEP_NODE_PATCH_PROPOSER
        node_state = self.node(node_id)
        for key, text in extract_mutable_stage_b_blocks(sections).items():
            node_state.actor_blocks.setdefault(key, text)

    def patch_actor_overrides(self) -> dict[str, str]:
        return dict(self.node(TEP_NODE_PATCH_PROPOSER).actor_blocks)

    def planner_actor_overrides(self) -> dict[str, str]:
        return dict(self.node(TEP_NODE_RETRY_PLANNER).actor_blocks)

    def stage_a_actor_overrides(self) -> dict[str, str]:
        return dict(self.node(TEP_NODE_STAGE_A_REQUEST).actor_blocks)

    def compaction_actor_overrides(self) -> dict[str, str]:
        return dict(self.node(TEP_NODE_CONTEXT_COMPACTION).actor_blocks)

    def stage_a_system_prompt(self, *, strict_json: bool) -> str:
        prompt_key = "strict_json" if strict_json else "standard"
        return self.node(TEP_NODE_STAGE_A_REQUEST).system_prompts[prompt_key]

    def planner_system_prompt(self) -> str:
        return self.node(TEP_NODE_RETRY_PLANNER).system_prompts["default"]

    def compaction_system_prompt(self, *, target_chars: int) -> str:
        template = self.node(TEP_NODE_CONTEXT_COMPACTION).system_prompts["default"]
        return render_system_prompt_template(template, target_chars=max(1, int(target_chars)))

    def patch_system_prompt(self, *, response_mode: str, target_rel: str = "target.py") -> str:
        template = self.node(TEP_NODE_PATCH_PROPOSER).system_prompts.get(
            str(response_mode or "edits"),
            self.node(TEP_NODE_PATCH_PROPOSER).system_prompts["edits"],
        )
        return render_system_prompt_template(template, target_rel=target_rel)

    def record_free_phase(
        self,
        *,
        node_id: str,
        case_id: str,
        input_summary: str,
        output_summary: str,
        critic_result: TepCriticResult,
        scores: list[float],
        improvement_kinds: list[str],
        feedback_rounds: list[list[str]],
    ) -> TepNodeTrace:
        converged, reason = resolve_equilibrium_status(
            scores=scores,
            improvement_kinds=improvement_kinds,
            max_iterations=self.equilibrium_max_iters,
            window=self.equilibrium_window,
            variance_threshold=self.equilibrium_variance_threshold,
        )
        trace = TepNodeTrace(
            node_id=node_id,
            iterations=len(scores),
            converged=converged,
            convergence_reason=reason,
            scores=list(scores),
            improvement_kinds=list(improvement_kinds),
            feedback_rounds=[list(round_feedback) for round_feedback in feedback_rounds],
            input_summary=input_summary,
            output_summary=output_summary,
            critic_summary=critic_result.summary,
        )
        self.latest_traces[node_id] = trace
        self.node(node_id).append_validation_entry(
            case_id=case_id,
            score=trace.final_score,
            note=f"free:{reason}",
            max_size=self.state.validation_buffer_size,
        )
        return trace

    def apply_candidate_update(
        self,
        *,
        node_id: str,
        case_id: str,
        free_score: float,
        nudged_score: float,
        update_payload: dict[str, Any] | None,
        note: str,
    ) -> dict[str, Any]:
        node_state = self.node(node_id)
        accepted = should_accept_candidate_update(
            free_score=free_score,
            nudged_score=nudged_score,
            validation_buffer=node_state.validation_buffer,
        )
        node_state.update_count += 1
        if accepted:
            apply_update_payload(node_state=node_state, payload=update_payload)
            node_state.accepted_updates += 1
        else:
            node_state.rejected_updates += 1
        node_state.temperature = next_temperature(current=node_state.temperature, accepted=accepted)
        self.state.beta = next_beta(
            current=self.state.beta,
            beta_decay=self.state.beta_decay,
            accepted=accepted,
        )
        node_state.append_validation_entry(
            case_id=case_id,
            score=float(nudged_score if accepted else free_score),
            note=note,
            max_size=self.state.validation_buffer_size,
        )
        return {
            "accepted": bool(accepted),
            "free_score": float(free_score),
            "nudged_score": float(nudged_score),
            "temperature": float(node_state.temperature),
            "beta": float(self.state.beta),
            "update_count": int(node_state.update_count),
            "accepted_updates": int(node_state.accepted_updates),
            "rejected_updates": int(node_state.rejected_updates),
        }


@dataclass
class _TepCalls:
    """The LLM endpoint of one post-attempt learning pass, and its call log."""

    patcher: Any
    context_dir: Path
    cwd: Path
    call_metas: list[dict[str, Any]] = field(default_factory=list)


def _critique_surface(
    calls: _TepCalls,
    *,
    node_state: TepNodeState,
    surface: dict[str, Any],
    packet: dict[str, str],
    phase: str,
) -> TepCriticResult:
    """Score *surface* with the node's critic; *phase* only labels the call."""
    result = calls.patcher.generate_json(
        system_prompt=TEP_CRITIC_SYSTEM_PROMPT,
        prompt=_render_critic_prompt(
            node_id=node_state.node_id,
            phase="evaluate",
            actor_surface=surface,
            input_summary=str(packet.get("input_summary") or ""),
            output_summary=str(packet.get("output_summary") or ""),
            objective_packet=str(packet.get("objective_packet") or ""),
            task_criteria=node_state.task_criteria,
        ),
        context_dir=calls.context_dir,
        cwd=calls.cwd,
        schema_name=f"{node_state.node_id}_critic",
        schema=TEP_CRITIC_JSON_SCHEMA,
        temperature=node_state.temperature,
        max_tokens=1024,
        operation_name=f"tep_{node_state.node_id}_critic_{phase}",
    )
    calls.call_metas.append(dict(getattr(result, "meta", None) or {}))
    return TepCriticResult.from_payload(result.payload)


def run_post_attempt_learning(
    *,
    controller: AdaptiveTepController,
    patcher: Any,
    context_dir: Path,
    cwd: Path,
    case_id: str,
    node_packets: list[dict[str, str]],
) -> dict[str, Any]:
    """Run a lightweight post-attempt critic/update pass over invoked nodes."""
    if not callable(getattr(patcher, "generate_json", None)):
        return {"enabled": False, "reason": "generate_json_unavailable"}

    calls = _TepCalls(patcher=patcher, context_dir=context_dir, cwd=cwd)
    updates: dict[str, Any] = {
        "enabled": True,
        "beta_before": float(controller.state.beta),
        "node_updates": {},
    }
    for packet in reversed(node_packets):
        node_id = str(packet.get("node_id") or "").strip()
        if not node_id:
            continue
        node_state = controller.node(node_id)
        current_surface = {
            "system_prompts": dict(node_state.system_prompts),
            "blocks": dict(node_state.actor_blocks),
        }
        objective_packet = str(packet.get("objective_packet") or "")
        # POL-17: a paired comparison. The critic scores the current surface, the updater
        # proposes a change from that critique, and the same critic prompt scores the
        # candidate surface. The two critiques used to score the same surface under a
        # "free"/"nudged" label, so accepting the higher one was a coin flip.
        free_critic = _critique_surface(
            calls, node_state=node_state, surface=current_surface, packet=packet, phase="current"
        )
        if not free_critic.critic_parsed:
            updates["node_updates"][node_id] = {
                "accepted": False,
                "reason": "critic_unparsed",
                "critic_parsed": False,
            }
            continue
        update_result = patcher.generate_json(
            system_prompt=TEP_UPDATE_SYSTEM_PROMPT,
            prompt=_render_update_prompt(
                node_id=node_id,
                actor_surface=current_surface,
                critic=free_critic,
                objective_packet=objective_packet,
            ),
            context_dir=context_dir,
            cwd=cwd,
            schema_name=f"{node_id}_update",
            schema=TEP_UPDATE_JSON_SCHEMA,
            temperature=node_state.temperature,
            max_tokens=1536,
            operation_name=f"tep_{node_id}_update",
        )
        calls.call_metas.append(dict(getattr(update_result, "meta", None) or {}))
        candidate_state = copy.deepcopy(node_state)
        apply_update_payload(node_state=candidate_state, payload=update_result.payload)
        candidate_surface = {
            "system_prompts": dict(candidate_state.system_prompts),
            "blocks": dict(candidate_state.actor_blocks),
        }
        if candidate_surface == current_surface:
            updates["node_updates"][node_id] = {
                "accepted": False,
                "reason": "no_candidate_change",
                "critic_parsed": True,
                "free_score": free_critic.overall_score_10,
            }
            continue
        nudged_critic = _critique_surface(
            calls,
            node_state=node_state,
            surface=candidate_surface,
            packet=packet,
            phase="candidate",
        )
        if not nudged_critic.critic_parsed:
            updates["node_updates"][node_id] = {
                "accepted": False,
                "reason": "critic_unparsed",
                "critic_parsed": False,
                "free_score": free_critic.overall_score_10,
            }
            continue
        decision = controller.apply_candidate_update(
            node_id=node_id,
            case_id=case_id,
            free_score=free_critic.overall_score_10,
            nudged_score=nudged_critic.overall_score_10,
            update_payload=update_result.payload,
            note=str(packet.get("note") or "post_attempt"),
        )
        updates["node_updates"][node_id] = {
            **decision,
            "critic_parsed": True,
            "free_improvement_kind": free_critic.improvement_kind,
            "nudged_improvement_kind": nudged_critic.improvement_kind,
            "free_summary": free_critic.summary,
            "nudged_summary": nudged_critic.summary,
        }
    updates["beta_after"] = float(controller.state.beta)
    # POL-02: every critic/update call is a real LLM call the attempt must pay for.
    updates["llm_call_metas"] = calls.call_metas
    return updates


__all__ = [
    "TEP_NODE_CONTEXT_COMPACTION",
    "TEP_NODE_ORDER",
    "TEP_NODE_PATCH_PROPOSER",
    "TEP_NODE_RETRY_PLANNER",
    "TEP_NODE_STAGE_A_REQUEST",
    "TEP_REVERSE_NODE_ORDER",
    "TEP_STAGE_B_MUTABLE_KEYS",
    "AdaptiveTepController",
    "AdaptiveTepState",
    "TepCriticResult",
    "TepNodeState",
    "TepNodeTrace",
    "apply_update_payload",
    "build_default_tep_state",
    "clamp_temperature",
    "extract_mutable_stage_b_blocks",
    "load_tep_state",
    "next_beta",
    "next_temperature",
    "resolve_equilibrium_status",
    "run_post_attempt_learning",
    "save_tep_state",
    "should_accept_candidate_update",
]
