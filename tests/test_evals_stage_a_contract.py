"""Tests for Stage-A context request contract (evals/stage_a_contract.py)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evals.stage_a_contract import (
    STAGE_A_ALLOWED_REQUEST_TYPES,
    StageARequestEntryModel,
    StageARequestPayloadModel,
    _coerce_bool,
    normalize_stage_a_request_payload,
    stage_a_allowed_request_types_text,
)


# ---------------------------------------------------------------------------
# _coerce_bool (module-level helper) — lines 25-33
# ---------------------------------------------------------------------------
class TestCoerceBool:
    def test_bool_passthrough(self) -> None:
        assert _coerce_bool(True) is True
        assert _coerce_bool(False) is False

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"])
    def test_truthy_strings(self, value: str) -> None:
        assert _coerce_bool(value) is True

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF"])
    def test_falsy_strings(self, value: str) -> None:
        assert _coerce_bool(value) is False

    def test_non_standard_string_fails_closed(self) -> None:
        assert _coerce_bool("maybe") is False
        assert _coerce_bool("") is False

    def test_numeric_values_only_accept_zero_or_one(self) -> None:
        assert _coerce_bool(1) is True
        assert _coerce_bool(0) is False
        assert _coerce_bool(1.0) is True
        assert _coerce_bool(0.0) is False
        assert _coerce_bool(2) is False
        assert _coerce_bool(1.5) is False


# ---------------------------------------------------------------------------
# StageARequestEntryModel._coerce_text — line 51
# ---------------------------------------------------------------------------
class TestCoerceText:
    def test_none_coerced_to_empty(self) -> None:
        entry = StageARequestEntryModel(type="file", path=None)  # type: ignore[arg-type]
        assert entry.path == ""

    def test_int_coerced_to_string(self) -> None:
        entry = StageARequestEntryModel(type="search", pattern=123)  # type: ignore[arg-type]
        assert entry.pattern == "123"

    def test_whitespace_stripped(self) -> None:
        entry = StageARequestEntryModel(type="file", path="  foo.py  ")
        assert entry.path == "foo.py"


# ---------------------------------------------------------------------------
# StageARequestEntryModel._coerce_int — lines 56-63
# ---------------------------------------------------------------------------
class TestCoerceInt:
    def test_bool_true_rejected(self) -> None:
        entry = StageARequestEntryModel(type="frame", index=True)  # type: ignore[arg-type]
        assert entry.index is None

    def test_bool_false_rejected(self) -> None:
        entry = StageARequestEntryModel(type="frame", index=False)  # type: ignore[arg-type]
        assert entry.index is None

    def test_int_passthrough(self) -> None:
        entry = StageARequestEntryModel(type="frame", index=5)
        assert entry.index == 5

    def test_string_int_coerced(self) -> None:
        entry = StageARequestEntryModel(type="frame", index="3")  # type: ignore[arg-type]
        assert entry.index == 3

    def test_non_numeric_string_is_rejected(self) -> None:
        entry = StageARequestEntryModel(type="frame", index="abc")  # type: ignore[arg-type]
        assert entry.index is None

    def test_none_is_rejected(self) -> None:
        entry = StageARequestEntryModel(type="frame", index=None)  # type: ignore[arg-type]
        assert entry.index is None

    def test_negative_int_is_rejected(self) -> None:
        entry = StageARequestEntryModel(type="frame", index=-1)
        assert entry.index is None


# ---------------------------------------------------------------------------
# StageARequestEntryModel.to_typed_dict — lines 66-77
# ---------------------------------------------------------------------------
class TestEntryToTypedDict:
    def test_minimal_file_entry(self) -> None:
        entry = StageARequestEntryModel(type="file", path="a.py")
        d = entry.to_typed_dict()
        assert d["type"] == "file"
        assert d["path"] == "a.py"
        assert "pattern" not in d
        assert "index" not in d
        assert "name" not in d

    def test_search_entry_with_pattern(self) -> None:
        entry = StageARequestEntryModel(type="search", path="b.py", pattern="TODO")
        d = entry.to_typed_dict()
        assert d["pattern"] == "TODO"
        assert d["path"] == "b.py"

    def test_frame_entry_always_includes_index(self) -> None:
        entry = StageARequestEntryModel(type="frame", index=0)
        d = entry.to_typed_dict()
        assert "index" in d
        assert d["index"] == 0

    def test_local_var_entry_always_includes_frame(self) -> None:
        entry = StageARequestEntryModel(type="local_var", frame=0, name="x")
        d = entry.to_typed_dict()
        assert "frame" in d
        assert d["frame"] == 0
        assert d["name"] == "x"

    def test_invalid_frame_entry_raises(self) -> None:
        entry = StageARequestEntryModel(type="frame", index=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="index"):
            entry.to_typed_dict()

    def test_invalid_local_var_entry_raises(self) -> None:
        entry = StageARequestEntryModel(type="local_var", frame=0, name="")
        with pytest.raises(ValueError, match="frame and name"):
            entry.to_typed_dict()

    def test_nonzero_index_included_for_non_frame_type(self) -> None:
        entry = StageARequestEntryModel(type="file", index=2)
        d = entry.to_typed_dict()
        assert d["index"] == 2

    def test_nonzero_frame_included_for_non_local_var_type(self) -> None:
        entry = StageARequestEntryModel(type="file", frame=3)
        d = entry.to_typed_dict()
        assert d["frame"] == 3

    def test_empty_name_excluded(self) -> None:
        entry = StageARequestEntryModel(type="file")
        d = entry.to_typed_dict()
        assert "name" not in d


# ---------------------------------------------------------------------------
# StageARequestPayloadModel._coerce_text — line 104
# ---------------------------------------------------------------------------
class TestPayloadCoerceText:
    def test_none_coerced_to_empty(self) -> None:
        m = StageARequestPayloadModel(diagnosis=None)  # type: ignore[arg-type]
        assert m.diagnosis == ""

    def test_int_coerced_to_string(self) -> None:
        m = StageARequestPayloadModel(diagnosis=42)  # type: ignore[arg-type]
        assert m.diagnosis == "42"


# ---------------------------------------------------------------------------
# StageARequestPayloadModel._coerce_missing_fact — lines 109-112
# ---------------------------------------------------------------------------
class TestCoerceMissingFact:
    def test_none_stays_none(self) -> None:
        m = StageARequestPayloadModel(missing_fact=None)
        assert m.missing_fact is None

    def test_empty_string(self) -> None:
        m = StageARequestPayloadModel(missing_fact="")
        assert m.missing_fact == ""

    def test_non_empty(self) -> None:
        m = StageARequestPayloadModel(missing_fact="some fact")
        assert m.missing_fact == "some fact"


# ---------------------------------------------------------------------------
# StageARequestPayloadModel._coerce_bool — line 122
# ---------------------------------------------------------------------------
class TestPayloadCoerceBool:
    @pytest.mark.parametrize(
        "field", ["need_more_context", "need_snapshot", "need_compact_context"]
    )
    def test_string_true(self, field: str) -> None:
        m = StageARequestPayloadModel(**{field: "yes"})
        assert getattr(m, field) is True

    @pytest.mark.parametrize(
        "field", ["need_more_context", "need_snapshot", "need_compact_context"]
    )
    def test_string_false(self, field: str) -> None:
        m = StageARequestPayloadModel(**{field: "no"})
        assert getattr(m, field) is False

    def test_int_one_is_true(self) -> None:
        m = StageARequestPayloadModel(need_more_context=1)  # type: ignore[arg-type]
        assert m.need_more_context is True

    def test_unknown_string_fails_closed(self) -> None:
        m = StageARequestPayloadModel(need_snapshot="maybe")  # type: ignore[arg-type]
        assert m.need_snapshot is False

    def test_off_string_is_false(self) -> None:
        m = StageARequestPayloadModel(need_compact_context="off")  # type: ignore[arg-type]
        assert m.need_compact_context is False


# ---------------------------------------------------------------------------
# StageARequestPayloadModel._coerce_requests — lines 127-137
# ---------------------------------------------------------------------------
class TestCoerceRequests:
    def test_non_list_coerced_to_empty(self) -> None:
        m = StageARequestPayloadModel(requests="not a list")  # type: ignore[arg-type]
        assert m.requests == []

    def test_non_dict_entries_skipped(self) -> None:
        m = StageARequestPayloadModel(
            requests=["string_entry", 42, {"type": "file", "path": "ok.py"}]  # type: ignore[arg-type]
        )
        assert len(m.requests) == 1
        assert m.requests[0].path == "ok.py"

    def test_invalid_dict_entries_skipped(self) -> None:
        """A dict with an invalid type value is silently skipped."""
        m = StageARequestPayloadModel(
            requests=[{"type": "nonexistent_type"}, {"type": "file"}]  # type: ignore[arg-type]
        )
        assert len(m.requests) == 1
        assert m.requests[0].type == "file"

    def test_invalid_frame_and_local_var_entries_are_dropped(self) -> None:
        m = StageARequestPayloadModel(
            requests=[
                {"type": "frame", "index": "abc"},
                {"type": "local_var", "frame": True, "name": "x"},
                {"type": "local_var", "frame": 0, "name": "x"},
            ]  # type: ignore[arg-type]
        )
        assert len(m.requests) == 1
        assert m.requests[0].type == "local_var"


# ---------------------------------------------------------------------------
# StageARequestPayloadModel.to_typed_dict — lines 140-155
# ---------------------------------------------------------------------------
class TestPayloadToTypedDict:
    def test_all_fields_present(self) -> None:
        m = StageARequestPayloadModel(
            diagnosis="root cause",
            missing_fact="important info",
            need_more_context=True,
            need_snapshot=False,
            need_compact_context=True,
            requests=[{"type": "file", "path": "x.py"}],  # type: ignore[list-item]
            snapshot_justification="need snapshot",
            compact_context_reason="too large",
            sufficiency_rationale="not enough",
        )
        d = m.to_typed_dict()
        assert d["diagnosis"] == "root cause"
        assert d["missing_fact"] == "important info"
        assert d["need_more_context"] is True
        assert d["need_compact_context"] is True
        assert len(d["requests"]) == 1

    def test_missing_fact_none_included(self) -> None:
        m = StageARequestPayloadModel(missing_fact=None)
        d = m.to_typed_dict()
        assert d["missing_fact"] is None

    def test_missing_fact_empty_excluded(self) -> None:
        m = StageARequestPayloadModel(missing_fact="")
        d = m.to_typed_dict()
        assert "missing_fact" not in d

    def test_empty_diagnosis_excluded(self) -> None:
        m = StageARequestPayloadModel(diagnosis="")
        d = m.to_typed_dict()
        assert "diagnosis" not in d

    def test_nonempty_diagnosis_included(self) -> None:
        m = StageARequestPayloadModel(diagnosis="bug here")
        d = m.to_typed_dict()
        assert d["diagnosis"] == "bug here"


# ---------------------------------------------------------------------------
# normalize_stage_a_request_payload — lines 199-202
# ---------------------------------------------------------------------------
class TestNormalizePayload:
    def test_valid_payload_returns_dict(self) -> None:
        result = normalize_stage_a_request_payload(
            {"need_more_context": True, "requests": [{"type": "file", "path": "a.py"}]}
        )
        assert result is not None
        assert result["need_more_context"] is True

    def test_completely_invalid_payload_returns_none(self) -> None:
        """A payload that causes ValidationError returns None."""
        # Passing a non-dict value that Pydantic cannot validate at all
        result = normalize_stage_a_request_payload(None)  # type: ignore[arg-type]
        assert result is None

    def test_empty_payload_normalizes(self) -> None:
        result = normalize_stage_a_request_payload({})
        assert result is not None
        assert result["need_more_context"] is False

    def test_unknown_boolean_strings_fail_closed(self) -> None:
        result = normalize_stage_a_request_payload(
            {
                "need_more_context": "maybe",
                "need_snapshot": "on",
                "need_compact_context": "off",
            }
        )
        assert result is not None
        assert result["need_more_context"] is False
        assert result["need_snapshot"] is True
        assert result["need_compact_context"] is False

    def test_invalid_request_entries_are_dropped_not_repaired(self) -> None:
        result = normalize_stage_a_request_payload(
            {
                "requests": [
                    {"type": "frame", "index": "abc"},
                    {"type": "local_var", "frame": "bad", "name": "value"},
                    {"type": "frame", "index": 0},
                ]
            }
        )
        assert result is not None
        assert result["requests"] == [{"type": "frame", "index": 0}]


# ---------------------------------------------------------------------------
# stage_a_allowed_request_types_text — line 194
# ---------------------------------------------------------------------------
def test_stage_a_allowed_request_types_text() -> None:
    text = stage_a_allowed_request_types_text()
    for t in STAGE_A_ALLOWED_REQUEST_TYPES:
        assert t in text
    assert ", " in text
