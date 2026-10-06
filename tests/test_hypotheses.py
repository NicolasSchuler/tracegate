"""Tests for hypothesis generation engine — all 10 detectors and public API."""

from __future__ import annotations

from typing import Any

import pytest

from llmdebug.hypotheses import Hypothesis, format_hypotheses_text, generate_hypotheses

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snap(
    exc_type: str = "Exception",
    exc_msg: str = "",
    locals_: dict[str, Any] | None = None,
    locals_meta: dict[str, Any] | None = None,
    crash_frame_index: int | str = 0,
) -> dict[str, Any]:
    """Build a minimal snapshot dict for testing detectors."""
    return {
        "exception": {"type": exc_type, "message": exc_msg},
        "frames": [
            {
                "file": "/code/app.py",
                "file_rel": "app.py",
                "line": 10,
                "function": "do_stuff",
                "locals": locals_ or {},
                "locals_meta": locals_meta or {},
            },
        ],
        "crash_frame_index": crash_frame_index,
    }


# ---------------------------------------------------------------------------
# 1. empty_array detector
# ---------------------------------------------------------------------------


class TestDetectEmptyArray:
    def test_shape_zero_triggers(self) -> None:
        snap = _snap(locals_={"data": {"__array__": True, "shape": [0, 5], "dtype": "float64"}})
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "empty_array"]
        assert len(matching) == 1
        h = matching[0]
        assert h.confidence == 0.85
        assert "data" in h.description
        assert "shape [0, 5]" in h.description
        assert "data.shape = [0, 5]" in h.evidence

    def test_non_empty_array_no_trigger(self) -> None:
        snap = _snap(locals_={"data": {"__array__": True, "shape": [10, 5], "dtype": "float64"}})
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "empty_array"]
        assert len(matching) == 0

    def test_no_shape_key_no_trigger(self) -> None:
        snap = _snap(locals_={"data": {"__array__": True}})
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "empty_array"]
        assert len(matching) == 0

    def test_scalar_locals_no_trigger(self) -> None:
        snap = _snap(locals_={"x": 42, "y": "hello"})
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "empty_array"]
        assert len(matching) == 0

    def test_no_frames_no_trigger(self) -> None:
        snap: dict[str, Any] = {
            "exception": {"type": "RuntimeError", "message": "fail"},
            "frames": [],
        }
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "empty_array"]
        assert len(matching) == 0

    def test_dataframe_summary_empty(self) -> None:
        snap = _snap(locals_={"df": {"__dataframe__": True, "shape": [0, 3]}})
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "empty_array"]
        assert len(matching) == 1

    def test_series_summary_empty(self) -> None:
        snap = _snap(locals_={"s": {"__series__": True, "shape": [0]}})
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "empty_array"]
        assert len(matching) == 1


# ---------------------------------------------------------------------------
# 2. shape_mismatch detector
# ---------------------------------------------------------------------------


class TestDetectShapeMismatch:
    def test_value_error_with_shape_keyword_and_two_arrays(self) -> None:
        snap = _snap(
            exc_type="ValueError",
            exc_msg="operands could not be broadcast together with shapes (3,4) and (5,6)",
            locals_={
                "a": {"__array__": True, "shape": [3, 4], "dtype": "float64"},
                "b": {"__array__": True, "shape": [5, 6], "dtype": "float64"},
            },
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "shape_mismatch"]
        assert len(matching) == 1
        h = matching[0]
        assert h.confidence == 0.9
        assert "a.shape = [3, 4]" in h.evidence
        assert "b.shape = [5, 6]" in h.evidence

    def test_runtime_error_with_dimension_keyword(self) -> None:
        snap = _snap(
            exc_type="RuntimeError",
            exc_msg="dimension mismatch at index 1",
            locals_={
                "x": {"__array__": True, "shape": [10, 3], "dtype": "float32"},
                "y": {"__array__": True, "shape": [10, 7], "dtype": "float32"},
            },
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "shape_mismatch"]
        assert len(matching) == 1

    def test_wrong_exc_type_no_trigger(self) -> None:
        snap = _snap(
            exc_type="TypeError",
            exc_msg="shape mismatch",
            locals_={
                "a": {"__array__": True, "shape": [3, 4], "dtype": "float64"},
                "b": {"__array__": True, "shape": [5, 6], "dtype": "float64"},
            },
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "shape_mismatch"]
        assert len(matching) == 0

    def test_no_shape_keyword_in_message_no_trigger(self) -> None:
        snap = _snap(
            exc_type="ValueError",
            exc_msg="something completely unrelated",
            locals_={
                "a": {"__array__": True, "shape": [3, 4], "dtype": "float64"},
                "b": {"__array__": True, "shape": [5, 6], "dtype": "float64"},
            },
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "shape_mismatch"]
        assert len(matching) == 0

    def test_only_one_array_no_trigger(self) -> None:
        snap = _snap(
            exc_type="ValueError",
            exc_msg="shape mismatch",
            locals_={
                "a": {"__array__": True, "shape": [3, 4], "dtype": "float64"},
                "x": 42,
            },
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "shape_mismatch"]
        assert len(matching) == 0

    def test_size_keyword_triggers(self) -> None:
        snap = _snap(
            exc_type="ValueError",
            exc_msg="size of tensor a (10) must match the size of tensor b (20)",
            locals_={
                "a": {"__array__": True, "shape": [10], "dtype": "float32"},
                "b": {"__array__": True, "shape": [20], "dtype": "float32"},
            },
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "shape_mismatch"]
        assert len(matching) == 1


# ---------------------------------------------------------------------------
# 3. none_dereference detector
# ---------------------------------------------------------------------------


class TestDetectNoneDereference:
    def test_attribute_error_with_none_local(self) -> None:
        snap = _snap(
            exc_type="AttributeError",
            exc_msg="'NoneType' object has no attribute 'strip'",
            locals_={"result": None, "x": 42},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "none_dereference"]
        assert len(matching) == 1
        h = matching[0]
        assert h.confidence == 0.9  # boosted because "NoneType" in msg
        assert "result" in h.description

    def test_type_error_with_none_local(self) -> None:
        snap = _snap(
            exc_type="TypeError",
            exc_msg="argument of type 'NoneType' is not iterable",
            locals_={"config": None},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "none_dereference"]
        assert len(matching) == 1
        assert matching[0].confidence == 0.9

    def test_lower_confidence_without_nonetype_in_message(self) -> None:
        snap = _snap(
            exc_type="AttributeError",
            exc_msg="object has no attribute 'foo'",
            locals_={"obj": None},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "none_dereference"]
        assert len(matching) == 1
        assert matching[0].confidence == 0.75

    def test_confidence_boost_with_quoted_none(self) -> None:
        snap = _snap(
            exc_type="TypeError",
            exc_msg="unsupported operand type(s): 'None' and 'int'",
            locals_={"val": None},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "none_dereference"]
        assert len(matching) == 1
        assert matching[0].confidence == 0.9  # "'None'" in msg

    def test_no_none_locals_no_trigger(self) -> None:
        snap = _snap(
            exc_type="AttributeError",
            exc_msg="'NoneType' object has no attribute 'strip'",
            locals_={"result": "hello", "x": 42},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "none_dereference"]
        assert len(matching) == 0

    def test_wrong_exc_type_no_trigger(self) -> None:
        snap = _snap(
            exc_type="ValueError",
            exc_msg="some error",
            locals_={"result": None},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "none_dereference"]
        assert len(matching) == 0

    def test_multiple_none_vars_listed(self) -> None:
        snap = _snap(
            exc_type="AttributeError",
            exc_msg="'NoneType' object has no attribute 'x'",
            locals_={"a": None, "b": None, "c": 99},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "none_dereference"]
        assert len(matching) == 1
        h = matching[0]
        assert "a" in h.description
        assert "b" in h.description
        assert len(h.evidence) == 2


# ---------------------------------------------------------------------------
# 4. off_by_one detector
# ---------------------------------------------------------------------------


class TestDetectOffByOne:
    def test_index_equals_list_length(self) -> None:
        snap = _snap(
            exc_type="IndexError",
            exc_msg="list index out of range",
            locals_={"i": 5, "items": [1, 2, 3, 4, 5]},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "off_by_one"]
        assert len(matching) == 1
        h = matching[0]
        assert h.confidence == 0.9
        assert "i" in h.description
        assert "items" in h.description
        assert "i = 5" in h.evidence
        assert "len(items) = 5" in h.evidence

    def test_index_equals_array_shape0(self) -> None:
        snap = _snap(
            exc_type="IndexError",
            exc_msg="index 10 is out of bounds for axis 0 with size 10",
            locals_={
                "idx": 10,
                "arr": {"__array__": True, "shape": [10, 3], "dtype": "float64"},
            },
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "off_by_one"]
        assert len(matching) == 1

    def test_index_less_than_length_no_trigger(self) -> None:
        snap = _snap(
            exc_type="IndexError",
            exc_msg="list index out of range",
            locals_={"i": 3, "items": [1, 2, 3, 4, 5]},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "off_by_one"]
        assert len(matching) == 0

    def test_wrong_exc_type_no_trigger(self) -> None:
        snap = _snap(
            exc_type="ValueError",
            exc_msg="index out of range",
            locals_={"i": 5, "items": [1, 2, 3, 4, 5]},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "off_by_one"]
        assert len(matching) == 0

    def test_no_int_vars_no_trigger(self) -> None:
        snap = _snap(
            exc_type="IndexError",
            exc_msg="list index out of range",
            locals_={"items": [1, 2, 3]},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "off_by_one"]
        assert len(matching) == 0

    def test_string_length_match(self) -> None:
        snap = _snap(
            exc_type="IndexError",
            exc_msg="string index out of range",
            locals_={"pos": 3, "text": "abc"},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "off_by_one"]
        assert len(matching) == 1
        assert "pos" in matching[0].description
        assert "text" in matching[0].description


# ---------------------------------------------------------------------------
# 5. nan_inf_anomaly detector
# ---------------------------------------------------------------------------


class TestDetectNanInf:
    def test_nan_in_anomalies(self) -> None:
        snap = _snap(
            locals_={
                "weights": {"__array__": True, "shape": [100], "anomalies": {"nan": 5, "inf": 0}},
            }
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "nan_inf_anomaly"]
        assert len(matching) == 1
        h = matching[0]
        assert h.confidence == 0.85
        assert "5 NaN" in h.description
        assert "Inf" not in h.description  # inf count is 0

    def test_inf_in_anomalies(self) -> None:
        snap = _snap(
            locals_={
                "loss": {"__array__": True, "shape": [1], "anomalies": {"nan": 0, "inf": 2}},
            }
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "nan_inf_anomaly"]
        assert len(matching) == 1
        assert "2 Inf" in matching[0].description

    def test_both_nan_and_inf(self) -> None:
        snap = _snap(
            locals_={
                "data": {"anomalies": {"nan": 3, "inf": 7}},
            }
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "nan_inf_anomaly"]
        assert len(matching) == 1
        h = matching[0]
        assert "3 NaN" in h.description
        assert "7 Inf" in h.description

    def test_zero_counts_no_trigger(self) -> None:
        snap = _snap(
            locals_={
                "data": {"anomalies": {"nan": 0, "inf": 0}},
            }
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "nan_inf_anomaly"]
        assert len(matching) == 0

    def test_no_anomalies_key_no_trigger(self) -> None:
        snap = _snap(
            locals_={
                "data": {"__array__": True, "shape": [10, 5]},
            }
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "nan_inf_anomaly"]
        assert len(matching) == 0

    def test_non_dict_value_no_trigger(self) -> None:
        snap = _snap(locals_={"x": 42, "y": "hello"})
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "nan_inf_anomaly"]
        assert len(matching) == 0


# ---------------------------------------------------------------------------
# 6. type_mismatch detector
# ---------------------------------------------------------------------------


class TestDetectTypeMismatch:
    def test_type_in_message_with_locals_meta(self) -> None:
        snap = _snap(
            exc_type="TypeError",
            exc_msg="unsupported operand type(s) for +: 'int' and 'str'",
            locals_={"name": "alice", "count": 5},
            locals_meta={"name": {"type": "str"}, "count": {"type": "int"}},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "type_mismatch"]
        # Both "int" and "str" appear in the message: both locals are named (CAP-16).
        assert len(matching) == 2
        assert all(h.confidence == 0.8 for h in matching)
        assert {"name", "count"} <= {word for h in matching for word in h.description.split("'")}
        assert len(generate_hypotheses(snap, max_per_pattern=1)) == 1

    def test_no_type_in_message_no_trigger(self) -> None:
        snap = _snap(
            exc_type="TypeError",
            exc_msg="missing 1 required positional argument",
            locals_={"x": 42},
            locals_meta={"x": {"type": "int"}},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "type_mismatch"]
        assert len(matching) == 0

    def test_wrong_exc_type_no_trigger(self) -> None:
        snap = _snap(
            exc_type="ValueError",
            exc_msg="int is not valid",
            locals_={"x": 42},
            locals_meta={"x": {"type": "int"}},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "type_mismatch"]
        assert len(matching) == 0

    def test_no_locals_meta_no_trigger(self) -> None:
        snap = _snap(
            exc_type="TypeError",
            exc_msg="unsupported operand type(s) for +: 'int' and 'str'",
            locals_={"name": "alice"},
            locals_meta={},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "type_mismatch"]
        assert len(matching) == 0

    def test_evidence_includes_message(self) -> None:
        snap = _snap(
            exc_type="TypeError",
            exc_msg="expected list got tuple",
            locals_={"items": (1, 2)},
            locals_meta={"items": {"type": "tuple"}},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "type_mismatch"]
        assert len(matching) == 1
        assert any("message:" in e for e in matching[0].evidence)


# ---------------------------------------------------------------------------
# 7. key_not_found detector
# ---------------------------------------------------------------------------


class TestDetectKeyNotFound:
    def test_key_error_with_dict_local(self) -> None:
        snap = _snap(
            exc_type="KeyError",
            exc_msg="'color'",
            locals_={"config": {"name": "test", "size": 10}},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "key_not_found"]
        assert len(matching) == 1
        h = matching[0]
        assert h.confidence == 0.85
        assert "'color'" in h.description
        assert "config" in h.description
        assert "name, size" in h.evidence[1]

    def test_no_dict_locals_no_trigger(self) -> None:
        snap = _snap(
            exc_type="KeyError",
            exc_msg="'missing'",
            locals_={"x": 42, "y": "text"},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "key_not_found"]
        assert len(matching) == 0

    def test_wrong_exc_type_no_trigger(self) -> None:
        snap = _snap(
            exc_type="ValueError",
            exc_msg="'missing'",
            locals_={"d": {"a": 1}},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "key_not_found"]
        assert len(matching) == 0

    def test_suggestion_mentions_get(self) -> None:
        snap = _snap(
            exc_type="KeyError",
            exc_msg="'k'",
            locals_={"d": {"a": 1}},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "key_not_found"]
        assert len(matching) == 1
        assert ".get(" in matching[0].suggestion

    def test_large_dict_truncates_keys(self) -> None:
        big_dict = {f"key_{i}": i for i in range(20)}
        snap = _snap(
            exc_type="KeyError",
            exc_msg="'missing'",
            locals_={"data": big_dict},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "key_not_found"]
        assert len(matching) == 1
        # Only first 10 keys shown
        keys_evidence = matching[0].evidence[1]
        assert keys_evidence.count(",") <= 10


# ---------------------------------------------------------------------------
# 8. import_failure detector
# ---------------------------------------------------------------------------


class TestDetectImportFailure:
    def test_module_not_found_error(self) -> None:
        snap = _snap(
            exc_type="ModuleNotFoundError",
            exc_msg="No module named 'torch'",
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "import_failure"]
        assert len(matching) == 1
        h = matching[0]
        assert h.confidence == 0.95
        assert "torch" in h.description
        assert "pip install torch" in h.suggestion

    def test_import_error(self) -> None:
        snap = _snap(
            exc_type="ImportError",
            exc_msg="No module named 'sklearn.ensemble'",
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "import_failure"]
        assert len(matching) == 1
        h = matching[0]
        assert "sklearn.ensemble" in h.description
        # Suggestion uses the top-level package
        assert "pip install sklearn" in h.suggestion

    def test_wrong_exc_type_no_trigger(self) -> None:
        snap = _snap(
            exc_type="RuntimeError",
            exc_msg="No module named 'foo'",
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "import_failure"]
        assert len(matching) == 0

    def test_non_standard_import_message(self) -> None:
        snap = _snap(
            exc_type="ImportError",
            exc_msg="cannot import name 'bar' from 'foo'",
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "import_failure"]
        assert len(matching) == 1
        # Falls back to exc_msg since regex doesn't match "No module named"
        assert matching[0].description


# ---------------------------------------------------------------------------
# 9. file_not_found detector
# ---------------------------------------------------------------------------


class TestDetectFileNotFound:
    def test_file_not_found_error_with_path(self) -> None:
        snap = _snap(
            exc_type="FileNotFoundError",
            exc_msg="[Errno 2] No such file or directory: '/data/input.csv'",
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "file_not_found"]
        assert len(matching) == 1
        h = matching[0]
        assert h.confidence == 0.95
        assert "/data/input.csv" in h.description
        assert "/data/input.csv" in h.suggestion

    def test_wrong_exc_type_no_trigger(self) -> None:
        snap = _snap(
            exc_type="IOError",
            exc_msg="file not found: '/data/input.csv'",
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "file_not_found"]
        assert len(matching) == 0

    def test_path_extraction_with_single_quotes(self) -> None:
        snap = _snap(
            exc_type="FileNotFoundError",
            exc_msg="No such file: 'config.yaml'",
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "file_not_found"]
        assert len(matching) == 1
        assert "config.yaml" in matching[0].description

    def test_path_extraction_with_double_quotes(self) -> None:
        snap = _snap(
            exc_type="FileNotFoundError",
            exc_msg='No such file: "settings.json"',
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "file_not_found"]
        assert len(matching) == 1
        assert "settings.json" in matching[0].description

    def test_no_quoted_path_fallback(self) -> None:
        snap = _snap(
            exc_type="FileNotFoundError",
            exc_msg="file missing",
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "file_not_found"]
        assert len(matching) == 1
        assert "unknown" in matching[0].description


# ---------------------------------------------------------------------------
# 10. assertion_failure detector
# ---------------------------------------------------------------------------


class TestDetectAssertionFailure:
    def test_assertion_error_with_locals(self) -> None:
        snap = _snap(
            exc_type="AssertionError",
            exc_msg="",
            locals_={"expected": 10, "actual": 7},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "assertion_failure"]
        assert len(matching) == 1
        h = matching[0]
        assert h.confidence == 0.8
        assert "check variable values" in h.description.lower()
        # Evidence should contain the locals
        evidence_str = " ".join(h.evidence)
        assert "expected" in evidence_str
        assert "actual" in evidence_str

    def test_assertion_error_without_locals(self) -> None:
        snap = _snap(
            exc_type="AssertionError",
            exc_msg="",
            locals_={},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "assertion_failure"]
        assert len(matching) == 1
        h = matching[0]
        assert h.confidence == 0.7
        assert "no locals" in h.description.lower()
        assert "assertion messages" in h.suggestion.lower()

    def test_wrong_exc_type_no_trigger(self) -> None:
        snap = _snap(
            exc_type="RuntimeError",
            exc_msg="assertion failed",
            locals_={"x": 1},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "assertion_failure"]
        assert len(matching) == 0

    def test_limits_evidence_to_five_locals(self) -> None:
        many_locals = {f"var_{i}": i for i in range(10)}
        snap = _snap(
            exc_type="AssertionError",
            exc_msg="",
            locals_=many_locals,
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "assertion_failure"]
        assert len(matching) == 1
        assert len(matching[0].evidence) <= 5

    def test_pytest_internals_filtered(self) -> None:
        snap = _snap(
            exc_type="AssertionError",
            exc_msg="",
            locals_={"@py_builtins": "<module>", "@py_assert0": True, "real_var": 42},
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "assertion_failure"]
        assert len(matching) == 1
        h = matching[0]
        # pytest internals filtered out, only real_var remains
        evidence_str = " ".join(h.evidence)
        assert "@py_" not in evidence_str
        assert "real_var" in evidence_str


# ---------------------------------------------------------------------------
# generate_hypotheses — integration / orchestration
# ---------------------------------------------------------------------------


class TestGenerateHypotheses:
    def test_sorted_by_confidence_descending(self) -> None:
        """When multiple detectors fire, results are sorted highest confidence first."""
        snap = _snap(
            exc_type="AttributeError",
            exc_msg="'NoneType' object has no attribute 'shape'",
            locals_={
                "result": None,
                "data": {"__array__": True, "shape": [0, 5], "dtype": "float64"},
            },
        )
        hyps = generate_hypotheses(snap)
        assert len(hyps) >= 2
        for i in range(len(hyps) - 1):
            assert hyps[i].confidence >= hyps[i + 1].confidence

    def test_empty_snapshot_returns_empty(self) -> None:
        hyps = generate_hypotheses({})
        assert hyps == []

    def test_snapshot_with_no_frames_returns_empty(self) -> None:
        snap: dict[str, Any] = {
            "exception": {"type": "RuntimeError", "message": "some error"},
            "frames": [],
        }
        hyps = generate_hypotheses(snap)
        assert hyps == []

    def test_multiple_patterns_fire_together(self) -> None:
        """empty_array and none_dereference can co-exist."""
        snap = _snap(
            exc_type="AttributeError",
            exc_msg="'NoneType' object has no attribute 'sum'",
            locals_={
                "model": None,
                "data": {"__array__": True, "shape": [0, 10], "dtype": "float32"},
            },
        )
        hyps = generate_hypotheses(snap)
        patterns = {h.pattern for h in hyps}
        assert "empty_array" in patterns
        assert "none_dereference" in patterns

    def test_broken_detector_silently_skipped(self) -> None:
        """A detector that raises is silently caught."""
        from llmdebug.hypotheses import _detectors

        def _boom(snapshot: dict[str, Any]) -> list[Hypothesis]:
            raise RuntimeError("detector bug")

        _detectors.append(_boom)
        try:
            snap = _snap(
                exc_type="FileNotFoundError",
                exc_msg="No such file: 'x.txt'",
            )
            # Should not raise, and file_not_found should still work
            hyps = generate_hypotheses(snap)
            matching = [h for h in hyps if h.pattern == "file_not_found"]
            assert len(matching) == 1
        finally:
            _detectors.remove(_boom)

    def test_deduplication_by_pattern_and_description(self) -> None:
        """CAP-16: distinct hypotheses of one pattern are kept (up to a cap); exact
        duplicates once; max_per_pattern=1 keeps only the first, as before."""
        from llmdebug.hypotheses import _detectors

        def _duplicate_file_detector(snapshot: dict[str, Any]) -> list[Hypothesis]:
            other = Hypothesis(
                confidence=0.5,
                pattern="file_not_found",
                description="duplicate",
                evidence=(),
                suggestion="none",
            )
            return [other, other]

        _detectors.append(_duplicate_file_detector)
        try:
            snap = _snap(
                exc_type="FileNotFoundError",
                exc_msg="No such file: 'x.txt'",
            )
            hyps = generate_hypotheses(snap)
            file_hyps = [h for h in hyps if h.pattern == "file_not_found"]
            assert [h.confidence for h in file_hyps] == [0.95, 0.5]  # exact duplicate once

            first_only = generate_hypotheses(snap, max_per_pattern=1)
            assert [h.confidence for h in first_only if h.pattern == "file_not_found"] == [0.95]
        finally:
            _detectors.remove(_duplicate_file_detector)

    def test_crash_frame_index_respected(self) -> None:
        """Detector reads the correct frame based on crash_frame_index."""
        snap: dict[str, Any] = {
            "exception": {"type": "AssertionError", "message": ""},
            "frames": [
                {"locals": {"x": 1}, "locals_meta": {}},
                {"locals": {"target": 99, "actual": 0}, "locals_meta": {}},
            ],
            "crash_frame_index": 1,
        }
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "assertion_failure"]
        assert len(matching) == 1
        evidence_str = " ".join(matching[0].evidence)
        assert "target" in evidence_str
        assert "actual" in evidence_str
        # Should NOT include x from frame 0
        assert "x = 1" not in evidence_str

    def test_crash_frame_index_string_value_supported(self) -> None:
        snap: dict[str, Any] = {
            "exception": {"type": "AssertionError", "message": ""},
            "frames": [
                {"locals": {"x": 1}, "locals_meta": {}},
                {"locals": {"target": 99, "actual": 0}, "locals_meta": {}},
            ],
            "crash_frame_index": "1",
        }
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "assertion_failure"]
        assert len(matching) == 1
        evidence_str = " ".join(matching[0].evidence)
        assert "target" in evidence_str
        assert "actual" in evidence_str
        assert "x = 1" not in evidence_str

    def test_invalid_crash_frame_index_falls_back_to_first_valid_frame(self) -> None:
        snap: dict[str, Any] = {
            "exception": {"type": "AssertionError", "message": ""},
            "frames": [
                "bad-frame",
                {"locals": {"target": 99, "actual": 0}, "locals_meta": {}},
            ],
            "crash_frame_index": "not-an-int",
        }
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "assertion_failure"]
        assert len(matching) == 1
        evidence_str = " ".join(matching[0].evidence)
        assert "target" in evidence_str


# ---------------------------------------------------------------------------
# format_hypotheses_text
# ---------------------------------------------------------------------------


class TestFormatHypothesesText:
    def test_empty_list_message(self) -> None:
        text = format_hypotheses_text([])
        assert "No hypotheses generated" in text

    def test_single_hypothesis_includes_all_fields(self) -> None:
        h = Hypothesis(
            confidence=0.85,
            pattern="empty_array",
            description="Variable 'data' is an empty array (shape [0, 5])",
            evidence=("data.shape = [0, 5]",),
            suggestion="Check why 'data' has no elements.",
        )
        text = format_hypotheses_text([h])
        assert "## Hypothesis #1 [85%]" in text
        assert "empty_array" in text
        assert "Variable 'data' is an empty array (shape [0, 5])" in text
        assert "Evidence:" in text
        assert "data.shape = [0, 5]" in text
        assert "Suggestion:" in text
        assert "Check why 'data' has no elements." in text

    def test_multiple_hypotheses_numbered(self) -> None:
        h1 = Hypothesis(
            confidence=0.95,
            pattern="import_failure",
            description="Cannot import 'torch'",
            evidence=("exception: ModuleNotFoundError",),
            suggestion="pip install torch",
        )
        h2 = Hypothesis(
            confidence=0.7,
            pattern="assertion_failure",
            description="Assertion failed",
            evidence=("x = 1",),
            suggestion="Check values.",
        )
        text = format_hypotheses_text([h1, h2])
        assert "## Hypothesis #1 [95%]" in text
        assert "## Hypothesis #2 [70%]" in text

    def test_confidence_is_integer_percent(self) -> None:
        h = Hypothesis(
            confidence=0.333,
            pattern="test",
            description="test",
            evidence=(),
            suggestion="test",
        )
        text = format_hypotheses_text([h])
        assert "[33%]" in text

    def test_multiple_evidence_items(self) -> None:
        h = Hypothesis(
            confidence=0.9,
            pattern="off_by_one",
            description="off by one",
            evidence=("i = 5", "len(items) = 5"),
            suggestion="fix index",
        )
        text = format_hypotheses_text([h])
        assert "  - i = 5" in text
        assert "  - len(items) = 5" in text


# ---------------------------------------------------------------------------
# Hypothesis dataclass
# ---------------------------------------------------------------------------


class TestHypothesisDataclass:
    def test_frozen(self) -> None:
        h = Hypothesis(
            confidence=0.9,
            pattern="test",
            description="desc",
            evidence=("e1",),
            suggestion="fix it",
        )
        with pytest.raises(AttributeError):
            h.confidence = 0.5  # type: ignore[misc]

    def test_fields(self) -> None:
        h = Hypothesis(
            confidence=0.75,
            pattern="none_dereference",
            description="Variable 'x' is None",
            evidence=("x = None",),
            suggestion="Check init",
        )
        assert h.confidence == 0.75
        assert h.pattern == "none_dereference"
        assert h.description == "Variable 'x' is None"
        assert h.evidence == ("x = None",)
        assert h.suggestion == "Check init"


# ---------------------------------------------------------------------------
# Coverage gap tests — lines 64-66, 73, 107, 115, 163, 198, 231, 252, 287, 320, 351, 427
# ---------------------------------------------------------------------------


class TestGetCrashFrameNonDictFallback:
    """Lines 64-66: _get_crash_frame returns first dict frame when crash_idx points to non-dict."""

    def test_crash_idx_points_to_non_dict(self) -> None:
        from llmdebug.hypotheses import _get_crash_frame

        snap: dict[str, Any] = {
            "exception": {"type": "Exception", "message": ""},
            "frames": ["bad-string", {"function": "good", "locals": {}}],
            "crash_frame_index": 0,
        }
        frame = _get_crash_frame(snap)
        assert frame is not None
        assert frame.get("function") == "good"

    def test_first_frame_fallback_when_idx_frame_not_dict(self) -> None:
        from llmdebug.hypotheses import _get_crash_frame

        snap: dict[str, Any] = {
            "exception": {"type": "Exception", "message": ""},
            "frames": [{"function": "first"}, "bad"],
            "crash_frame_index": 1,
        }
        frame = _get_crash_frame(snap)
        assert frame is not None
        assert frame.get("function") == "first"

    def test_no_dict_frames_at_all(self) -> None:
        from llmdebug.hypotheses import _get_crash_frame

        snap: dict[str, Any] = {
            "exception": {"type": "Exception", "message": ""},
            "frames": ["bad1", "bad2"],
            "crash_frame_index": 0,
        }
        frame = _get_crash_frame(snap)
        assert frame is None


class TestGetLocalsFiltersPyPrefix:
    """Line 73: _get_locals filters @py_ prefixed keys."""

    def test_py_prefix_filtered(self) -> None:
        from llmdebug.hypotheses import _get_locals

        frame: dict[str, Any] = {
            "locals": {
                "@py_builtins": "<module>",
                "@py_assert0": True,
                "real_var": 42,
            }
        }
        result = _get_locals(frame)
        assert "real_var" in result
        assert "@py_builtins" not in result
        assert "@py_assert0" not in result

    def test_non_dict_locals_returns_empty(self) -> None:
        from llmdebug.hypotheses import _get_locals

        frame: dict[str, Any] = {"locals": "not-a-dict"}
        assert _get_locals(frame) == {}


class TestExcTypeExcMessageNonDictException:
    """Lines 107, 115: _exc_type/_exc_message with non-dict exception → ''."""

    def test_exc_type_non_dict_returns_empty(self) -> None:
        from llmdebug.hypotheses import _exc_type

        assert _exc_type({"exception": "not-a-dict"}) == ""
        assert _exc_type({"exception": 42}) == ""
        assert _exc_type({}) == ""

    def test_exc_message_non_dict_returns_empty(self) -> None:
        from llmdebug.hypotheses import _exc_message

        assert _exc_message({"exception": "not-a-dict"}) == ""
        assert _exc_message({"exception": None}) == ""
        assert _exc_message({}) == ""


class TestDetectorsReturnEmptyWithoutCrashFrame:
    """Lines 198, 231, 351, 427: detectors return [] when no crash frame."""

    def _no_frame_snap(self, exc_type: str = "Exception", exc_msg: str = "") -> dict[str, Any]:
        return {
            "exception": {"type": exc_type, "message": exc_msg},
            "frames": [],
        }

    def test_none_dereference_no_frame(self) -> None:
        """Line 198: _detect_none_dereference returns [] with no crash frame."""
        snap = self._no_frame_snap("AttributeError", "'NoneType' object has no attribute 'x'")
        hyps = generate_hypotheses(snap)
        assert all(h.pattern != "none_dereference" for h in hyps)

    def test_off_by_one_no_frame(self) -> None:
        """Line 231: _detect_off_by_one returns [] with no crash frame."""
        snap = self._no_frame_snap("IndexError", "list index out of range")
        hyps = generate_hypotheses(snap)
        assert all(h.pattern != "off_by_one" for h in hyps)

    def test_key_not_found_no_frame(self) -> None:
        """Line 351: _detect_key_not_found returns [] with no crash frame."""
        snap = self._no_frame_snap("KeyError", "'missing_key'")
        hyps = generate_hypotheses(snap)
        assert all(h.pattern != "key_not_found" for h in hyps)

    def test_assertion_failure_no_frame(self) -> None:
        """Line 427: _detect_assertion_failure returns [] with no crash frame."""
        snap = self._no_frame_snap("AssertionError", "assertion failed")
        hyps = generate_hypotheses(snap)
        assert all(h.pattern != "assertion_failure" for h in hyps)

    def test_shape_mismatch_no_frame(self) -> None:
        """Line 163: _detect_shape_mismatch returns [] with no crash frame."""
        snap = self._no_frame_snap("ValueError", "shape mismatch")
        hyps = generate_hypotheses(snap)
        assert all(h.pattern != "shape_mismatch" for h in hyps)

    def test_nan_inf_no_frame(self) -> None:
        """Line 287: _detect_nan_inf returns [] with no crash frame."""
        snap = self._no_frame_snap()
        hyps = generate_hypotheses(snap)
        assert all(h.pattern != "nan_inf_anomaly" for h in hyps)

    def test_type_mismatch_no_frame(self) -> None:
        """Line 320: _detect_type_mismatch returns [] with no crash frame."""
        snap = self._no_frame_snap("TypeError", "expected int got str")
        hyps = generate_hypotheses(snap)
        assert all(h.pattern != "type_mismatch" for h in hyps)


class TestOffByOneWithArraySummary:
    """Line 252: off-by-one detector with array summary shapes."""

    def test_index_equals_array_shape_dim0(self) -> None:
        snap = _snap(
            exc_type="IndexError",
            exc_msg="index out of bounds",
            locals_={
                "idx": 10,
                "data": {"__array__": True, "shape": [10, 5], "dtype": "float64"},
            },
        )
        hyps = generate_hypotheses(snap)
        matching = [h for h in hyps if h.pattern == "off_by_one"]
        assert len(matching) == 1
        assert "idx" in matching[0].description
        assert "data" in matching[0].description
