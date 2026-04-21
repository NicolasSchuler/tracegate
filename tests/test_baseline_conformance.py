"""Cross-adapter conformance test suite.

Verifies structural contracts shared by all five baseline adapters
without requiring external repos or LLM access.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields

import pytest

from baselines.adapter_base import BaselineAdapter, BaselineConfig
from baselines.case_schemas import (
    BaseCaseInput,
    IntervenorCaseInput,
    LambdaRLMCaseInput,
    LDBCaseInput,
    TraceCoderCaseInput,
)
from baselines.contracts import BASELINE_CONTRACT_VERSION, BaselineContract
from baselines.extra_configs import (
    IntervenorExtraConfig,
    LambdaRLMExtraConfig,
    LDBExtraConfig,
    RLMExtraConfig,
    TraceCoderExtraConfig,
)

# ── Schema / adapter pairs ────────────────────────────────────────────

_ADAPTER_SPECS: list[tuple[str, type, type]] = [
    ("ldb", LDBCaseInput, LDBExtraConfig),
    ("intervenor", IntervenorCaseInput, IntervenorExtraConfig),
    ("tracecoder", TraceCoderCaseInput, TraceCoderExtraConfig),
    ("lambda_rlm", LambdaRLMCaseInput, LambdaRLMExtraConfig),
    ("rlm", LambdaRLMCaseInput, RLMExtraConfig),
]

_SCHEMA_VALID_PAYLOADS: dict[str, dict[str, object]] = {
    "ldb": {
        "task_id": "llmdebug/test",
        "entry_point": "solve",
        "prompt": "def solve():",
        "test": "check(candidate)",
        "seed": "def solve(): pass",
        "given_tests": ["assert solve() == 42"],
        "dataset_type": "llmdebug",
    },
    "intervenor": {
        "task_id": "case1",
        "entry_point": "func",
        "prompt": "fix this",
        "buggy_code": "def func(): return 1",
        "test": "assert func() == 2",
        "python_args": ["-m", "pytest"],
        "timeout_sec": 30,
        "files": [{"path": "buggy.py", "encoding": "utf-8", "content": "..."}],
    },
    "tracecoder": {
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
    },
    "lambda_rlm": {
        "task_id": "case1",
        "entry_point": "func",
        "buggy_code": "def func(): return 1",
        "test": "assert func() == 2",
        "python_args": [],
        "timeout_sec": 0,
        "files": [],
        "prompt_context": "You are repairing buggy.py",
    },
    "rlm": {
        "task_id": "case1",
        "entry_point": "func",
        "buggy_code": "def func(): return 1",
        "test": "assert func() == 2",
        "python_args": [],
        "timeout_sec": 0,
        "files": [],
        "prompt_context": "You are repairing buggy.py",
    },
}

_SCHEMA_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "ldb": ("prompt", "test", "seed", "given_tests"),
    "intervenor": ("prompt", "buggy_code"),
    "tracecoder": ("complete_prompt", "buggy_code"),
    "lambda_rlm": ("buggy_code", "prompt_context"),
    "rlm": ("buggy_code", "prompt_context"),
}


# ── Typed case_input schema conformance ───────────────────────────────


class TestCaseSchemaConformance:
    """Every case schema must inherit BaseCaseInput and satisfy the interface."""

    @pytest.mark.parametrize(
        "name,schema_cls,_extra_cls",
        _ADAPTER_SPECS,
        ids=[s[0] for s in _ADAPTER_SPECS],
    )
    def test_inherits_base(self, name: str, schema_cls: type, _extra_cls: type) -> None:
        assert issubclass(schema_cls, BaseCaseInput), (
            f"{name}: {schema_cls.__name__} must inherit BaseCaseInput"
        )

    @pytest.mark.parametrize(
        "name,schema_cls,_extra_cls",
        _ADAPTER_SPECS,
        ids=[s[0] for s in _ADAPTER_SPECS],
    )
    def test_has_task_id_and_entry_point(
        self, name: str, schema_cls: type, _extra_cls: type
    ) -> None:
        field_names = {f.name for f in dc_fields(schema_cls)}
        assert "task_id" in field_names
        assert "entry_point" in field_names

    @pytest.mark.parametrize(
        "name,schema_cls,_extra_cls",
        _ADAPTER_SPECS,
        ids=[s[0] for s in _ADAPTER_SPECS],
    )
    def test_frozen_dataclass(self, name: str, schema_cls: type, _extra_cls: type) -> None:
        assert schema_cls.__dataclass_params__.frozen  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "name,schema_cls,_extra_cls",
        _ADAPTER_SPECS,
        ids=[s[0] for s in _ADAPTER_SPECS],
    )
    def test_has_from_dict_and_to_dict(self, name: str, schema_cls: type, _extra_cls: type) -> None:
        assert hasattr(schema_cls, "from_dict")
        assert hasattr(schema_cls, "to_dict")

    @pytest.mark.parametrize(
        "name,schema_cls,_extra_cls",
        _ADAPTER_SPECS,
        ids=[s[0] for s in _ADAPTER_SPECS],
    )
    def test_unknown_keys_rejected(self, name: str, schema_cls: type, _extra_cls: type) -> None:
        data = {"task_id": "t", "entry_point": "e", "definitely_not_real": True}
        with pytest.raises(ValueError, match="unexpected keys"):
            schema_cls.from_dict(data)  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "name,schema_cls,_extra_cls",
        _ADAPTER_SPECS,
        ids=[s[0] for s in _ADAPTER_SPECS],
    )
    def test_harness_keys_pass_through(self, name: str, schema_cls: type, _extra_cls: type) -> None:
        data = dict(_SCHEMA_VALID_PAYLOADS[name])
        data["llmdebug_controller_packet"] = {}
        data["llmdebug_evidence"] = {}
        # Should NOT raise.
        result = schema_cls.from_dict(data)  # type: ignore[attr-defined]
        assert result.task_id == str(_SCHEMA_VALID_PAYLOADS[name]["task_id"])

    @pytest.mark.parametrize(
        "name,schema_cls,_extra_cls",
        _ADAPTER_SPECS,
        ids=[s[0] for s in _ADAPTER_SPECS],
    )
    def test_missing_required_input_fields_raise(
        self, name: str, schema_cls: type, _extra_cls: type
    ) -> None:
        for field_name in _SCHEMA_REQUIRED_FIELDS[name]:
            data = dict(_SCHEMA_VALID_PAYLOADS[name])
            data.pop(field_name)
            with pytest.raises(ValueError, match=rf"missing required fields.*{field_name}"):
                schema_cls.from_dict(data)  # type: ignore[attr-defined]


# ── Extra config conformance ──────────────────────────────────────────


class TestExtraConfigConformance:
    """Every extra config must accept empty dicts and warn on unknown keys."""

    @pytest.mark.parametrize(
        "name,_schema_cls,extra_cls",
        _ADAPTER_SPECS,
        ids=[s[0] for s in _ADAPTER_SPECS],
    )
    def test_empty_extra_accepted(self, name: str, _schema_cls: type, extra_cls: type) -> None:
        result = extra_cls.validate({})  # type: ignore[attr-defined]
        assert result is not None

    @pytest.mark.parametrize(
        "name,_schema_cls,extra_cls",
        _ADAPTER_SPECS,
        ids=[s[0] for s in _ADAPTER_SPECS],
    )
    def test_unknown_extra_key_warns(
        self,
        name: str,
        _schema_cls: type,
        extra_cls: type,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            extra_cls.validate({"not_a_real_key": 42}, strict=False)  # type: ignore[attr-defined]
        assert "ignoring unrecognised" in caplog.text


# ── Adapter class-level contracts ─────────────────────────────────────


class TestAdapterClassContracts:
    """Structural checks on adapter classes (importable without repos)."""

    @pytest.mark.parametrize(
        "adapter_module,adapter_class,expected_name",
        [
            ("baselines.ldb.adapter", "LDBAdapter", "ldb"),
            ("baselines.intervenor.adapter", "IntervenorAdapter", "intervenor"),
            ("baselines.tracecoder.adapter", "TraceCoderAdapter", "tracecoder"),
            ("baselines.lambda_rlm.adapter", "LambdaRLMAdapter", "lambda_rlm"),
            ("baselines.rlm.adapter", "RLMAdapter", "rlm"),
        ],
    )
    def test_adapter_has_default_model(
        self, adapter_module: str, adapter_class: str, expected_name: str
    ) -> None:
        import importlib

        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        assert hasattr(cls, "default_model"), f"{adapter_class} must define default_model"
        assert isinstance(cls.default_model, str)
        assert cls.default_model.strip(), (
            f"{adapter_class}.default_model must be a non-empty string"
        )

    @pytest.mark.parametrize(
        "adapter_module,adapter_class,expected_name",
        [
            ("baselines.ldb.adapter", "LDBAdapter", "ldb"),
            ("baselines.intervenor.adapter", "IntervenorAdapter", "intervenor"),
            ("baselines.tracecoder.adapter", "TraceCoderAdapter", "tracecoder"),
            ("baselines.lambda_rlm.adapter", "LambdaRLMAdapter", "lambda_rlm"),
            ("baselines.rlm.adapter", "RLMAdapter", "rlm"),
        ],
    )
    def test_adapter_inherits_base(
        self, adapter_module: str, adapter_class: str, expected_name: str
    ) -> None:
        import importlib

        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        assert issubclass(cls, BaselineAdapter)

    @pytest.mark.parametrize(
        "adapter_module,adapter_class,expected_name",
        [
            ("baselines.ldb.adapter", "LDBAdapter", "ldb"),
            ("baselines.intervenor.adapter", "IntervenorAdapter", "intervenor"),
            ("baselines.tracecoder.adapter", "TraceCoderAdapter", "tracecoder"),
            ("baselines.lambda_rlm.adapter", "LambdaRLMAdapter", "lambda_rlm"),
            ("baselines.rlm.adapter", "RLMAdapter", "rlm"),
        ],
    )
    def test_adapter_has_validate_extra_config(
        self, adapter_module: str, adapter_class: str, expected_name: str
    ) -> None:
        import importlib

        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        assert hasattr(cls, "validate_extra_config"), (
            f"{adapter_class} must implement validate_extra_config"
        )

    @pytest.mark.parametrize(
        "adapter_module,adapter_class,expected_name",
        [
            ("baselines.ldb.adapter", "LDBAdapter", "ldb"),
            ("baselines.intervenor.adapter", "IntervenorAdapter", "intervenor"),
            ("baselines.tracecoder.adapter", "TraceCoderAdapter", "tracecoder"),
            ("baselines.lambda_rlm.adapter", "LambdaRLMAdapter", "lambda_rlm"),
            ("baselines.rlm.adapter", "RLMAdapter", "rlm"),
        ],
    )
    def test_adapter_exposes_contract(
        self, adapter_module: str, adapter_class: str, expected_name: str
    ) -> None:
        import importlib

        mod = importlib.import_module(adapter_module)
        cls = getattr(mod, adapter_class)
        adapter = cls()
        contract = adapter.contract()
        assert isinstance(contract, BaselineContract)
        assert contract.baseline_name == expected_name
        assert contract.contract_version == BASELINE_CONTRACT_VERSION
        assert contract.fidelity_level
        assert contract.repair_scope_policy
        assert contract.prompt_contract_kind
        assert contract.output_contract
