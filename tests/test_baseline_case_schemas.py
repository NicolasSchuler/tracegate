"""Unit tests for baselines.case_schemas typed case-input validation."""

from __future__ import annotations

import pytest

from baselines.case_schemas import (
    BaseCaseInput,
    IntervenorCaseInput,
    LambdaRLMCaseInput,
    LDBCaseInput,
    TraceCoderCaseInput,
)

_VALID_SCHEMA_INPUTS: dict[type[BaseCaseInput], dict[str, object]] = {
    BaseCaseInput: {"task_id": "test", "entry_point": "solve"},
    LDBCaseInput: {
        "task_id": "llmdebug/test",
        "entry_point": "solve",
        "prompt": "def solve():",
        "test": "check(candidate)",
        "seed": "def solve(): pass",
        "given_tests": ["assert solve() == 42"],
        "description": "",
        "dataset_type": "llmdebug",
    },
    IntervenorCaseInput: {
        "task_id": "case1",
        "entry_point": "func",
        "prompt": "fix this",
        "buggy_code": "def func(): return 1",
        "test": "assert func() == 2",
        "description": "",
        "python_args": ["-m", "pytest"],
        "timeout_sec": 30,
        "files": [{"path": "buggy.py", "encoding": "utf-8", "content": "..."}],
    },
    TraceCoderCaseInput: {
        "task_id": "case1",
        "entry_point": "func",
        "complete_prompt": "Repair buggy.py",
        "buggy_code": "def func(): return 1",
        "test": "assert func() == 2",
        "files": [],
        "python_args": ["-m", "pytest", "-q"],
        "timeout_sec": 30,
        "repair_seeded": True,
        "execution_mode": "llmdebug_native",
        "description": "",
    },
    LambdaRLMCaseInput: {
        "task_id": "case1",
        "entry_point": "func",
        "buggy_code": "def func(): return 1",
        "test": "assert func() == 2",
        "description": "",
        "python_args": [],
        "timeout_sec": 0,
        "files": [],
        "prompt_context": "You are repairing buggy.py",
    },
}

_REQUIRED_SCHEMA_FIELDS: dict[type[BaseCaseInput], tuple[str, ...]] = {
    LDBCaseInput: ("prompt", "test", "seed", "given_tests"),
    IntervenorCaseInput: ("prompt", "buggy_code"),
    TraceCoderCaseInput: ("complete_prompt", "buggy_code"),
    LambdaRLMCaseInput: ("buggy_code", "prompt_context"),
}


class TestBaseCaseInput:
    def test_from_dict_valid(self) -> None:
        data = {"task_id": "case_001", "entry_point": "solve"}
        result = BaseCaseInput.from_dict(data)
        assert result.task_id == "case_001"
        assert result.entry_point == "solve"
        assert result.description == ""

    def test_from_dict_missing_required_field(self) -> None:
        with pytest.raises(ValueError, match="BaseCaseInput"):
            BaseCaseInput.from_dict({"entry_point": "solve"})

    def test_from_dict_unknown_keys_raise(self) -> None:
        data = {"task_id": "x", "entry_point": "y", "bogus_key": 42}
        with pytest.raises(ValueError, match=r"unexpected keys.*bogus_key"):
            BaseCaseInput.from_dict(data)

    def test_from_dict_allows_harness_injected_keys(self) -> None:
        data = {
            "task_id": "x",
            "entry_point": "y",
            "llmdebug_controller_packet": {"a": 1},
            "llmdebug_evidence": {"b": 2},
        }
        result = BaseCaseInput.from_dict(data)
        assert result.task_id == "x"

    def test_to_dict_roundtrip(self) -> None:
        original = {"task_id": "a", "entry_point": "b", "description": "c"}
        result = BaseCaseInput.from_dict(original)
        roundtripped = result.to_dict()
        assert roundtripped["task_id"] == "a"
        assert roundtripped["entry_point"] == "b"
        assert roundtripped["description"] == "c"

    def test_frozen(self) -> None:
        case = BaseCaseInput(task_id="x", entry_point="y")
        with pytest.raises(AttributeError):
            case.task_id = "z"  # type: ignore[misc]


class TestLDBCaseInput:
    def test_from_dict_with_all_fields(self) -> None:
        data = {
            "task_id": "llmdebug/case_001",
            "entry_point": "solve",
            "prompt": "def solve():",
            "test": "check(candidate)",
            "seed": "def solve(): pass",
            "given_tests": ["assert solve() == 42"],
            "description": "test desc",
            "dataset_type": "llmdebug",
        }
        result = LDBCaseInput.from_dict(data)
        assert result.task_id == "llmdebug/case_001"
        assert result.prompt == "def solve():"
        assert result.seed == "def solve(): pass"
        assert result.dataset_type == "llmdebug"

    @pytest.mark.parametrize("missing_field", _REQUIRED_SCHEMA_FIELDS[LDBCaseInput])
    def test_from_dict_requires_adapter_inputs(self, missing_field: str) -> None:
        data = dict(_VALID_SCHEMA_INPUTS[LDBCaseInput])
        data.pop(missing_field)
        with pytest.raises(ValueError, match=rf"missing required fields.*{missing_field}"):
            LDBCaseInput.from_dict(data)

    def test_to_dict_includes_subclass_fields(self) -> None:
        case = LDBCaseInput(task_id="t", entry_point="e", prompt="p")
        d = case.to_dict()
        assert "prompt" in d
        assert "seed" in d
        assert "dataset_type" in d


class TestIntervenorCaseInput:
    def test_from_dict_valid(self) -> None:
        data = {
            "task_id": "case1",
            "entry_point": "func",
            "prompt": "fix this",
            "buggy_code": "def func(): return 1",
            "test": "assert func() == 2",
            "description": "",
            "python_args": ["-m", "pytest"],
            "timeout_sec": 30,
            "files": [{"path": "buggy.py", "encoding": "utf-8", "content": "..."}],
        }
        result = IntervenorCaseInput.from_dict(data)
        assert result.task_id == "case1"
        assert result.buggy_code == "def func(): return 1"

    @pytest.mark.parametrize("missing_field", _REQUIRED_SCHEMA_FIELDS[IntervenorCaseInput])
    def test_from_dict_requires_adapter_inputs(self, missing_field: str) -> None:
        data = dict(_VALID_SCHEMA_INPUTS[IntervenorCaseInput])
        data.pop(missing_field)
        with pytest.raises(ValueError, match=rf"missing required fields.*{missing_field}"):
            IntervenorCaseInput.from_dict(data)


class TestTraceCoderCaseInput:
    def test_from_dict_valid(self) -> None:
        data = {
            "task_id": "case1",
            "entry_point": "func",
            "complete_prompt": "Repair buggy.py",
            "buggy_code": "def func(): return 1",
            "test": "assert func() == 2",
            "files": [],
            "python_args": ["-m", "pytest", "-q"],
            "timeout_sec": 30,
            "repair_seeded": True,
            "execution_mode": "llmdebug_native",
            "description": "",
        }
        result = TraceCoderCaseInput.from_dict(data)
        assert result.complete_prompt == "Repair buggy.py"
        assert result.repair_seeded is True
        assert result.execution_mode == "llmdebug_native"

    @pytest.mark.parametrize("missing_field", _REQUIRED_SCHEMA_FIELDS[TraceCoderCaseInput])
    def test_from_dict_requires_adapter_inputs(self, missing_field: str) -> None:
        data = dict(_VALID_SCHEMA_INPUTS[TraceCoderCaseInput])
        data.pop(missing_field)
        with pytest.raises(ValueError, match=rf"missing required fields.*{missing_field}"):
            TraceCoderCaseInput.from_dict(data)


class TestLambdaRLMCaseInput:
    def test_from_dict_valid(self) -> None:
        data = {
            "task_id": "case1",
            "entry_point": "func",
            "buggy_code": "def func(): return 1",
            "test": "assert func() == 2",
            "description": "",
            "python_args": [],
            "timeout_sec": 0,
            "files": [],
            "prompt_context": "You are repairing buggy.py",
        }
        result = LambdaRLMCaseInput.from_dict(data)
        assert result.prompt_context == "You are repairing buggy.py"

    @pytest.mark.parametrize("missing_field", _REQUIRED_SCHEMA_FIELDS[LambdaRLMCaseInput])
    def test_from_dict_requires_adapter_inputs(self, missing_field: str) -> None:
        data = dict(_VALID_SCHEMA_INPUTS[LambdaRLMCaseInput])
        data.pop(missing_field)
        with pytest.raises(ValueError, match=rf"missing required fields.*{missing_field}"):
            LambdaRLMCaseInput.from_dict(data)

    def test_rlm_alias(self) -> None:
        from baselines.case_schemas import RLMCaseInput

        assert RLMCaseInput is LambdaRLMCaseInput


class TestCrossSchemaConsistency:
    """Verify structural guarantees across all schemas."""

    @pytest.mark.parametrize(
        "cls",
        [
            BaseCaseInput,
            LDBCaseInput,
            IntervenorCaseInput,
            TraceCoderCaseInput,
            LambdaRLMCaseInput,
        ],
    )
    def test_all_schemas_have_task_id_and_entry_point(self, cls: type) -> None:
        from dataclasses import fields as dc_fields

        names = {f.name for f in dc_fields(cls)}
        assert "task_id" in names
        assert "entry_point" in names

    @pytest.mark.parametrize(
        "cls",
        [
            BaseCaseInput,
            LDBCaseInput,
            IntervenorCaseInput,
            TraceCoderCaseInput,
            LambdaRLMCaseInput,
        ],
    )
    def test_all_schemas_frozen(self, cls: type) -> None:
        assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "cls",
        [
            BaseCaseInput,
            LDBCaseInput,
            IntervenorCaseInput,
            TraceCoderCaseInput,
            LambdaRLMCaseInput,
        ],
    )
    def test_from_dict_to_dict_roundtrip(self, cls: type) -> None:
        data = dict(_VALID_SCHEMA_INPUTS[cls])
        instance = cls.from_dict(data)  # type: ignore[attr-defined]
        roundtripped = instance.to_dict()
        assert roundtripped["task_id"] == data["task_id"]
        assert roundtripped["entry_point"] == data["entry_point"]
