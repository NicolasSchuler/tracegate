"""Error categorization with pattern-based classification and suggestions."""

from __future__ import annotations

import re
from typing import Any

# Category definitions with patterns, exception types, and suggestions
CATEGORIES: dict[str, dict[str, Any]] = {
    "shape_mismatch": {
        "patterns": [
            r"shape.*mismatch",
            r"cannot broadcast",
            r"incompatible.*shape",
            r"dimension.*mismatch",
            r"size mismatch",
            r"shapes.*not aligned",
            r"matmul.*shape",
            r"expected.*got.*tensor",
        ],
        "exception_types": ["ValueError", "RuntimeError"],
        "suggestion": "Check array/tensor shapes with .shape before operations. "
        "Use reshape(), squeeze(), or unsqueeze() to align dimensions.",
    },
    "type_error": {
        "patterns": [
            r"expected.*type",
            r"must be.*not",
            r"unsupported operand type",
            r"cannot convert",
            r"invalid type",
            r"type.*is not",
        ],
        "exception_types": ["TypeError"],
        "suggestion": "Check variable types with type() or isinstance(). "
        "Verify function arguments match expected types.",
    },
    "index_error": {
        "patterns": [
            r"index.*out of",
            r"out of bounds",
            r"list index",
            r"tuple index",
            r"string index",
        ],
        "exception_types": ["IndexError"],
        "suggestion": "Verify array/list length with len() before indexing. "
        "Check for off-by-one errors (0-based indexing).",
    },
    "key_error": {
        "patterns": [
            r"key.*not found",
            r"KeyError",
        ],
        "exception_types": ["KeyError"],
        "suggestion": "Use dict.get(key, default) or check 'key in dict' before access. "
        "Print available keys with dict.keys().",
    },
    "attribute_error": {
        "patterns": [
            r"has no attribute",
            r"object.*attribute",
            r"'NoneType'.*attribute",
        ],
        "exception_types": ["AttributeError"],
        "suggestion": "Check if object is None before attribute access. "
        "Verify the object type and available attributes with dir().",
    },
    "file_not_found": {
        "patterns": [
            r"No such file",
            r"FileNotFoundError",
            r"path.*not exist",
            r"cannot find.*file",
        ],
        "exception_types": ["FileNotFoundError", "OSError"],
        "suggestion": "Verify file path with os.path.exists(). "
        "Check working directory with os.getcwd().",
    },
    "import_error": {
        "patterns": [
            r"No module named",
            r"cannot import",
            r"ImportError",
            r"ModuleNotFoundError",
        ],
        "exception_types": ["ImportError", "ModuleNotFoundError"],
        "suggestion": "Install missing package with pip/uv. Check virtual environment activation.",
    },
    "memory_error": {
        "patterns": [
            r"out of memory",
            r"MemoryError",
            r"OOM",
            r"CUDA out of memory",
            r"allocation.*failed",
        ],
        "exception_types": ["MemoryError", "RuntimeError"],
        "suggestion": "Reduce batch size or array dimensions. "
        "Use del to free unused variables. Check for memory leaks with tracemalloc.",
    },
    "permission_error": {
        "patterns": [
            r"Permission denied",
            r"access denied",
            r"PermissionError",
        ],
        "exception_types": ["PermissionError", "OSError"],
        "suggestion": "Check file/directory permissions. "
        "Verify you have write access to the target location.",
    },
    "value_error": {
        "patterns": [
            r"invalid.*value",
            r"could not convert",
            r"invalid literal",
            r"ValueError",
        ],
        "exception_types": ["ValueError"],
        "suggestion": "Validate input values before processing. "
        "Check for empty strings, NaN, or unexpected formats.",
    },
    "assertion_error": {
        "patterns": [
            r"AssertionError",
            r"assertion failed",
        ],
        "exception_types": ["AssertionError"],
        "suggestion": "Review the assertion condition and the values that triggered it. "
        "Add print statements before the assertion to inspect values.",
    },
    "timeout_error": {
        "patterns": [
            r"timed? ?out",
            r"TimeoutError",
            r"deadline exceeded",
        ],
        # Note: asyncio.TimeoutError is an alias of TimeoutError in Python, so the
        # runtime type name is still "TimeoutError".
        "exception_types": ["TimeoutError"],
        "suggestion": "Increase timeout value or optimize slow operation. "
        "Check network connectivity for remote calls.",
    },
}

# Lower numeric value wins when multiple categories have identical score.
# Keep priorities explicit so tie behavior remains stable even if dict order changes.
CATEGORY_PRIORITY: dict[str, int] = {
    "shape_mismatch": 10,
    "type_error": 20,
    "index_error": 30,
    "key_error": 40,
    "attribute_error": 50,
    "file_not_found": 60,
    "import_error": 70,
    "memory_error": 80,
    "permission_error": 90,
    "value_error": 100,
    "assertion_error": 110,
    "timeout_error": 120,
}

assert CATEGORIES.keys() == CATEGORY_PRIORITY.keys(), (
    f"CATEGORIES and CATEGORY_PRIORITY keys out of sync: "
    f"missing from priority={CATEGORIES.keys() - CATEGORY_PRIORITY.keys()}, "
    f"extra in priority={CATEGORY_PRIORITY.keys() - CATEGORIES.keys()}"
)

# Pre-compile patterns for performance
_COMPILED_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in info["patterns"]]
    for cat, info in CATEGORIES.items()
}


def categorize_exception(exc: BaseException) -> dict[str, Any] | None:
    """Categorize an exception based on type and message patterns.

    Returns:
        Dict with category name and suggestion, or None if no match.

    Example output:
        {
            "category": "shape_mismatch",
            "suggestion": "Check array shapes with .shape before operations..."
        }
    """
    exc_type_name = type(exc).__name__
    message = str(exc)

    # Score each category
    best_match: tuple[str, int, int] | None = None

    for category, info in CATEGORIES.items():
        score = 0

        # Check exception type (strong signal)
        if exc_type_name in info["exception_types"]:
            score += 10

        # Check message patterns
        for pattern in _COMPILED_PATTERNS[category]:
            if pattern.search(message):
                score += 5
                break  # One pattern match is enough

        if score == 0:
            continue

        priority = CATEGORY_PRIORITY.get(category, 10_000)
        if best_match is None:
            best_match = (category, score, priority)
            continue

        _, best_score, best_priority = best_match
        if score > best_score or (score == best_score and priority < best_priority):
            best_match = (category, score, priority)

    if best_match is None:
        return None

    category = best_match[0]
    return {
        "category": category,
        "suggestion": CATEGORIES[category]["suggestion"],
    }
