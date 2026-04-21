"""Tests for error categorization."""

import re

import llmdebug.error_categories as error_categories
from llmdebug.error_categories import categorize_exception


def test_shape_mismatch_category():
    """Test that shape mismatch errors are categorized."""
    exc = ValueError("shape mismatch: expected (10,) got (20,)")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "shape_mismatch"
    assert "suggestion" in result
    assert "shape" in result["suggestion"].lower()


def test_broadcast_error_category():
    """Test that broadcast errors are categorized as shape mismatch."""
    exc = ValueError("cannot broadcast array of shape (3, 4) to (5, 6)")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "shape_mismatch"


def test_type_error_category():
    """Test that type errors are categorized."""
    exc = TypeError("expected str, not int")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "type_error"


def test_index_error_category():
    """Test that index errors are categorized."""
    exc = IndexError("list index out of range")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "index_error"


def test_key_error_category():
    """Test that key errors are categorized."""
    exc = KeyError("missing_key")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "key_error"


def test_attribute_error_category():
    """Test that attribute errors are categorized."""
    exc = AttributeError("'NoneType' object has no attribute 'foo'")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "attribute_error"
    assert "None" in result["suggestion"]


def test_file_not_found_category():
    """Test that file not found errors are categorized."""
    exc = FileNotFoundError("No such file or directory: '/path/to/file'")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "file_not_found"


def test_import_error_category():
    """Test that import errors are categorized."""
    exc = ModuleNotFoundError("No module named 'nonexistent'")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "import_error"


def test_memory_error_category():
    """Test that memory errors are categorized."""
    exc = MemoryError("out of memory")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "memory_error"


def test_cuda_oom_category():
    """Test that CUDA OOM is categorized as memory error."""
    exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "memory_error"


def test_permission_error_category():
    """Test that permission errors are categorized."""
    exc = PermissionError("Permission denied: '/path/to/file'")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "permission_error"


def test_value_error_generic_category():
    """Test that generic value errors are categorized."""
    exc = ValueError("invalid literal for int() with base 10: 'abc'")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "value_error"


def test_assertion_error_category():
    """Test that assertion errors are categorized."""
    exc = AssertionError("assertion failed")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "assertion_error"


def test_timeout_error_category():
    """Test that timeout errors are categorized."""
    exc = TimeoutError("Connection timed out")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "timeout_error"


def test_no_category_for_unknown_error():
    """Test that unknown errors return None."""

    class CustomError(Exception):
        pass

    exc = CustomError("some custom error message")
    result = categorize_exception(exc)

    # Unknown exceptions should return None
    assert result is None


def test_pattern_matching_case_insensitive():
    """Test that pattern matching is case insensitive."""
    exc = ValueError("SHAPE MISMATCH in tensor")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "shape_mismatch"


def test_exception_type_priority():
    """Test that exception type is prioritized over message patterns."""
    # IndexError with unrelated message should still be categorized by type
    exc = IndexError("something went wrong")
    result = categorize_exception(exc)

    assert result is not None
    assert result["category"] == "index_error"


def test_tie_break_uses_explicit_priority(monkeypatch):
    """Tie scores should resolve via CATEGORY_PRIORITY, not insertion order."""
    categories = {
        "cat_b": {
            "patterns": [r"does-not-match"],
            "exception_types": ["ValueError"],
            "suggestion": "b",
        },
        "cat_a": {
            "patterns": [r"does-not-match"],
            "exception_types": ["ValueError"],
            "suggestion": "a",
        },
    }
    priorities = {"cat_a": 10, "cat_b": 20}
    compiled = {
        category: [re.compile(pattern, re.IGNORECASE) for pattern in info["patterns"]]
        for category, info in categories.items()
    }

    monkeypatch.setattr(error_categories, "CATEGORIES", categories)
    monkeypatch.setattr(error_categories, "CATEGORY_PRIORITY", priorities)
    monkeypatch.setattr(error_categories, "_COMPILED_PATTERNS", compiled)

    result = error_categories.categorize_exception(ValueError("no pattern match"))
    assert result is not None
    assert result["category"] == "cat_a"
