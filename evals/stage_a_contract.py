"""Shared Stage-A context request contract.

Centralises the typed schema and request-type constraints so the shared
prompt-rendering helpers, patcher system prompts, docs, and tests stay aligned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import _json as json

if TYPE_CHECKING:
    from .patchers.payload_types import ContextRequestEntry, ContextRequestPayload
else:
    ContextRequestEntry = dict[str, Any]  # type: ignore[misc,assignment]
    ContextRequestPayload = dict[str, Any]  # type: ignore[misc,assignment]

STAGE_A_ALLOWED_REQUEST_TYPES: tuple[str, ...] = ("file", "search", "frame", "local_var")


def coerce_stage_a_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int):
        if value in {0, 1}:
            return bool(value)
        return False
    if isinstance(value, float):
        if value in {0.0, 1.0}:
            return bool(int(value))
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return False


def _coerce_bool(value: object) -> bool:
    return coerce_stage_a_bool(value)


class StageARequestEntryModel(BaseModel):
    """Typed request item for Stage-A evidence selection."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["file", "search", "frame", "local_var"]
    path: str = ""
    pattern: str = ""
    index: int | None = None
    frame: int | None = None
    name: str = ""

    @field_validator("path", "pattern", "name", mode="before")
    @classmethod
    def _coerce_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("index", "frame", mode="before")
    @classmethod
    def _coerce_int(cls, value: object) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, float):
            if value.is_integer() and value >= 0:
                return int(value)
            return None
        try:
            coerced = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return coerced if coerced >= 0 else None

    def is_valid_request(self) -> bool:
        if self.type == "frame":
            return self.index is not None
        if self.type == "local_var":
            return self.frame is not None and bool(self.name)
        return True

    def to_typed_dict(self) -> ContextRequestEntry:
        payload: ContextRequestEntry = {"type": self.type}
        if self.path:
            payload["path"] = self.path
        if self.pattern:
            payload["pattern"] = self.pattern
        if self.type == "frame":
            if self.index is None:
                raise ValueError("frame requests require a non-negative index")
            payload["index"] = self.index
        elif self.index is not None:
            payload["index"] = self.index
        if self.type == "local_var":
            if self.frame is None or not self.name:
                raise ValueError("local_var requests require frame and name")
            payload["frame"] = self.frame
        elif self.frame is not None:
            payload["frame"] = self.frame
        if self.name:
            payload["name"] = self.name
        return payload


class StageARequestPayloadModel(BaseModel):
    """Typed Stage-A request payload used for prompt rendering and validation."""

    model_config = ConfigDict(extra="ignore")

    diagnosis: str = ""
    missing_fact: str | None = ""
    need_more_context: bool = False
    need_snapshot: bool = False
    need_compact_context: bool = False
    requests: list[StageARequestEntryModel] = Field(default_factory=list)
    snapshot_justification: str = ""
    compact_context_reason: str = ""
    sufficiency_rationale: str = ""

    @field_validator(
        "diagnosis",
        "snapshot_justification",
        "compact_context_reason",
        "sufficiency_rationale",
        mode="before",
    )
    @classmethod
    def _coerce_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("missing_fact", mode="before")
    @classmethod
    def _coerce_missing_fact(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or ""

    @field_validator(
        "need_more_context",
        "need_snapshot",
        "need_compact_context",
        mode="before",
    )
    @classmethod
    def _coerce_bool(cls, value: object) -> bool:
        return coerce_stage_a_bool(value)

    @field_validator("requests", mode="before")
    @classmethod
    def _coerce_requests(cls, value: object) -> list[StageARequestEntryModel]:
        if not isinstance(value, list):
            return []
        out: list[StageARequestEntryModel] = []
        for entry in cast(list[object], value):
            if not isinstance(entry, dict):
                continue
            try:
                normalized_entry = StageARequestEntryModel.model_validate(entry)
            except ValidationError:
                continue
            if normalized_entry.is_valid_request():
                out.append(normalized_entry)
        return out

    def to_typed_dict(self) -> ContextRequestPayload:
        payload: ContextRequestPayload = {
            "need_more_context": bool(self.need_more_context),
            "need_snapshot": bool(self.need_snapshot),
            "need_compact_context": bool(self.need_compact_context),
            "snapshot_justification": self.snapshot_justification,
            "compact_context_reason": self.compact_context_reason,
            "sufficiency_rationale": self.sufficiency_rationale,
            "requests": [row.to_typed_dict() for row in self.requests],
        }
        if self.diagnosis:
            payload["diagnosis"] = self.diagnosis
        if self.missing_fact is None:
            payload["missing_fact"] = None
        elif self.missing_fact:
            payload["missing_fact"] = self.missing_fact
        return payload

    @classmethod
    def prompt_example(cls) -> dict[str, object]:
        return {
            "diagnosis": "one-sentence root-cause hypothesis",
            "missing_fact": "specific fact needed, or null",
            "need_more_context": True,
            "need_snapshot": False,
            "need_compact_context": False,
            "requests": [
                {"type": "file", "path": "relative/path.py"},
                {"type": "search", "path": "relative/path.py", "pattern": "literal_text"},
                {"type": "frame", "index": 1},
                {"type": "local_var", "frame": 0, "name": "user_payload"},
            ],
            "snapshot_justification": "one short sentence (only when need_snapshot=true)",
            "compact_context_reason": ("one short sentence (only when need_compact_context=true)"),
            "sufficiency_rationale": "one short sentence",
        }


STAGE_A_REQUEST_SCHEMA_COMPACT = json.dumps(
    StageARequestPayloadModel.prompt_example(),
    ensure_ascii=False,
    separators=(",", ":"),
)

STAGE_A_REQUEST_SCHEMA_PRETTY = json.dumps(
    StageARequestPayloadModel.prompt_example(),
    ensure_ascii=False,
    indent=2,
)


def stage_a_allowed_request_types_text() -> str:
    """Return the allowed Stage-A request types as a comma-separated string."""
    return ", ".join(STAGE_A_ALLOWED_REQUEST_TYPES)


def normalize_stage_a_request_payload(payload: dict[str, Any]) -> ContextRequestPayload | None:
    """Validate and normalize a raw Stage-A JSON object."""
    try:
        return StageARequestPayloadModel.model_validate(payload).to_typed_dict()
    except ValidationError:
        return None


__all__ = [
    "STAGE_A_ALLOWED_REQUEST_TYPES",
    "STAGE_A_REQUEST_SCHEMA_COMPACT",
    "STAGE_A_REQUEST_SCHEMA_PRETTY",
    "StageARequestEntryModel",
    "StageARequestPayloadModel",
    "coerce_stage_a_bool",
    "normalize_stage_a_request_payload",
    "stage_a_allowed_request_types_text",
]
