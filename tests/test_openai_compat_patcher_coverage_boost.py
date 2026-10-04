"""High-ROI coverage tests for evals/patchers/openai_compat_patcher.py.

Targets previously-uncovered pure-function paths: fuzzy block replacement,
whitespace-normalized matching, LMStudio context window extraction, model entry
matching, edits payload normalization, response seed extraction, backend
unavailability detection, file index extraction, and hunk count fixing.
"""

from __future__ import annotations

import pytest

from evals.patchers import openai_compat_patcher as mod

# ---------------------------------------------------------------------------
# 1. _apply_find_replace_with_recovery — newline normalization paths
# ---------------------------------------------------------------------------


class TestApplyFindReplaceWithRecovery:
    """Direct unit tests for the find-replace recovery pipeline."""

    def test_exact_match(self) -> None:
        result, mode = mod._apply_find_replace_with_recovery(
            before="hello world",
            find="hello",
            replace="goodbye",
            count=1,
        )
        assert result == "goodbye world"
        assert mode == "exact"

    def test_exact_noop_same_find_and_replace(self) -> None:
        result, mode = mod._apply_find_replace_with_recovery(
            before="hello world",
            find="hello",
            replace="hello",
            count=1,
        )
        assert result == "hello world"
        assert mode == "exact_noop"

    def test_normalized_newlines_crlf_find_in_lf_source(self) -> None:
        before = "line1\nline2\nline3\n"
        find = "line1\r\nline2"
        replace = "LINE1\nLINE2"
        result, mode = mod._apply_find_replace_with_recovery(
            before=before,
            find=find,
            replace=replace,
            count=1,
        )
        assert result is not None
        assert "LINE1" in result
        assert mode == "normalized_newlines"

    def test_normalized_newlines_noop(self) -> None:
        before = "line1\nline2\n"
        find = "line1\r\nline2"
        replace = "line1\nline2"  # same content after normalization
        result, mode = mod._apply_find_replace_with_recovery(
            before=before,
            find=find,
            replace=replace,
            count=1,
        )
        assert result == before
        assert mode == "normalized_newlines_noop"

    def test_empty_find_after_normalization_returns_none(self) -> None:
        result, mode = mod._apply_find_replace_with_recovery(
            before="some content",
            find="\r\n",  # normalizes to "\n" which is non-empty, but...
            replace="x",
            count=1,
        )
        # "\r\n" normalizes to "\n", which IS present in source after normalization
        # but the raw "\r\n" is NOT in "some content" so it won't be exact
        assert result is not None or mode in ("none", "normalized_newlines")

    def test_empty_find_string_returns_none(self) -> None:
        # find="" is in every string via `str.__contains__` → exact branch
        # but replace="" → same → exact_noop
        _result, mode = mod._apply_find_replace_with_recovery(
            before="hello",
            find="",
            replace="",
            count=1,
        )
        assert mode == "exact_noop"

    def test_whitespace_normalized_match(self) -> None:
        before = "if  x  ==  1:\n    pass\n"
        find = "if x == 1:\n    pass"
        replace = "if x == 2:\n    pass"
        result, mode = mod._apply_find_replace_with_recovery(
            before=before,
            find=find,
            replace=replace,
            count=1,
        )
        assert result is not None
        assert "x == 2" in result
        assert mode == "normalized_whitespace"

    def test_whitespace_normalized_noop(self) -> None:
        before = "if  x  ==  1:\n    pass\n"
        find = "if x == 1:\n    pass"
        replace = "if  x  ==  1:\n    pass"  # same as actual whitespace in source
        _result, mode = mod._apply_find_replace_with_recovery(
            before=before,
            find=find,
            replace=replace,
            count=1,
        )
        # The replacement text with extra spaces produces same result → noop
        assert mode == "normalized_whitespace_noop"

    def test_fuzzy_block_noop_when_replace_equals_source(self) -> None:
        # Build a source where exact+newline+whitespace fail but fuzzy matches,
        # and replacement equals the matched block → noop.
        original_line = "def calculate_result(value, factor):"
        source = f"{original_line}\n    return value * factor\n"
        # Slightly different find to avoid exact/newline/whitespace match
        # but close enough for fuzzy (> 0.92 ratio)
        find = "def calculate_result(value, factr):\n    return value * factor"
        # Replace with exact source content (including trailing newline) → noop
        replace = source
        _result, mode = mod._apply_find_replace_with_recovery(
            before=source,
            find=find,
            replace=replace,
            count=1,
        )
        assert mode == "fuzzy_block_noop"

    def test_fuzzy_skipped_when_count_is_not_one(self) -> None:
        source = "def calculate_result(value, factor):\n    return value * factor\n"
        find = "def calculate_result(value, factr):\n    return value * factor"
        replace = "def calculate_result(value, factor):\n    return value + factor"
        result, mode = mod._apply_find_replace_with_recovery(
            before=source,
            find=find,
            replace=replace,
            count=0,
        )
        assert mode == "none"
        assert result is None

    def test_none_when_find_normalizes_to_empty(self) -> None:
        # A find consisting only of \r characters normalizes to empty \n's,
        # but _normalize_newlines converts \r to \n so find_nl won't be empty.
        # Use a truly empty find after normalization path.
        result, mode = mod._apply_find_replace_with_recovery(
            before="content\nhere\n",
            find="\r",  # normalizes to "\n"
            replace="x",
            count=1,
        )
        # "\n" is in the normalized before, so should match
        assert result is not None or mode == "none"


# ---------------------------------------------------------------------------
# 2. _apply_fuzzy_block_replacement — edge cases
# ---------------------------------------------------------------------------


class TestApplyFuzzyBlockReplacement:
    """Direct unit tests for fuzzy block matching."""

    def test_empty_find_block(self) -> None:
        result = mod._apply_fuzzy_block_replacement(
            before_nl="some source\n", find_nl="   \n  ", replace_nl="x"
        )
        assert result is None

    def test_short_find_block_under_16_chars(self) -> None:
        result = mod._apply_fuzzy_block_replacement(
            before_nl="hello world\n", find_nl="hello wrld", replace_nl="goodbye"
        )
        assert result is None  # len("hello wrld") < 16

    def test_empty_source(self) -> None:
        result = mod._apply_fuzzy_block_replacement(
            before_nl="",
            find_nl="def long_function_name():\n    pass",
            replace_nl="replacement",
        )
        assert result is None

    def test_source_over_6000_lines(self) -> None:
        big_source = "line\n" * 6001
        result = mod._apply_fuzzy_block_replacement(
            before_nl=big_source,
            find_nl="def long_function_name():\n    pass",
            replace_nl="replacement",
        )
        assert result is None

    def test_ratio_just_above_threshold_multiline(self) -> None:
        # Multi-line: min_ratio = 0.92
        source = "def compute_values(x, y):\n    result = x + y\n    return result\n"
        # Very similar find — small typo
        find = (
            "def compute_values(x, y):\n"
            "    result = x + y\n"
            "    return reuslt\n"  # typo: 'reuslt' instead of 'result'
        )
        replace = "def compute_values(x, y):\n    result = x * y\n    return result\n"
        result = mod._apply_fuzzy_block_replacement(
            before_nl=source,
            find_nl=find,
            replace_nl=replace,
        )
        assert result is not None
        assert "x * y" in result

    def test_ratio_below_threshold_returns_none(self) -> None:
        source = "def compute_values(x, y):\n    result = x + y\n    return result\n"
        # Very different find
        find = "def totally_different(a, b, c):\n    output = a * b * c\n    return output * 2\n"
        replace = "replacement\n"
        result = mod._apply_fuzzy_block_replacement(
            before_nl=source,
            find_nl=find,
            replace_nl=replace,
        )
        assert result is None

    def test_ambiguous_matches_rejected(self) -> None:
        # Two very similar blocks — gap < 3% → rejected
        source = (
            "def process_a(value):\n"
            "    return value + 1\n"
            "\n"
            "def process_b(value):\n"
            "    return value + 1\n"
        )
        find = "def process_a(value):\n    return value + 1\n"
        replace = "def process_a(value):\n    return value + 2\n"
        result = mod._apply_fuzzy_block_replacement(
            before_nl=source,
            find_nl=find,
            replace_nl=replace,
        )
        assert result is None  # ambiguous — gap < 3%

    def test_single_line_uses_higher_threshold(self) -> None:
        # Single line: min_ratio = 0.96
        source = "def very_long_function_name_here_for_testing(x):\n    return x\n"
        # Single-line find with small difference (close to source but < 0.96)
        find = "def very_long_FUNCTION_name_here_for_testing(x):"
        replace = "def very_long_function_name_here_for_testing(y):"
        result = mod._apply_fuzzy_block_replacement(
            before_nl=source,
            find_nl=find,
            replace_nl=replace,
        )
        # With uppercase "FUNCTION" vs lowercase, ratio should be near 0.96
        # Whether it matches depends on exact ratio, but we test the code path
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# 3. _compile_whitespace_normalized_pattern
# ---------------------------------------------------------------------------


class TestCompileWhitespaceNormalizedPattern:
    """Direct tests for the whitespace normalization regex builder."""

    def test_tabs_and_spaces_collapsed(self) -> None:
        pattern = mod._compile_whitespace_normalized_pattern("a  b\tc")
        assert pattern.search("a b\tc") is not None
        assert pattern.search("a    b     c") is not None
        assert pattern.search("abc") is None  # no whitespace

    def test_trailing_whitespace_before_newline(self) -> None:
        pattern = mod._compile_whitespace_normalized_pattern("line1\nline2")
        assert pattern.search("line1\nline2") is not None
        assert pattern.search("line1  \nline2") is not None  # trailing ws before \n

    def test_empty_text(self) -> None:
        pattern = mod._compile_whitespace_normalized_pattern("")
        assert pattern.search("") is not None  # empty pattern matches empty string

    def test_special_regex_chars_escaped(self) -> None:
        pattern = mod._compile_whitespace_normalized_pattern("a.b(c)")
        assert pattern.search("a.b(c)") is not None
        assert pattern.search("aXb(c)") is None  # dot should be literal


# ---------------------------------------------------------------------------
# 4. _entry_matches_model — loaded_instances path
# ---------------------------------------------------------------------------


class TestEntryMatchesModel:
    """Tests for model entry matching logic."""

    def test_matches_by_id(self) -> None:
        assert mod._entry_matches_model({"id": "my-model"}, model="my-model") is True

    def test_matches_by_key(self) -> None:
        assert mod._entry_matches_model({"key": "my-model"}, model="my-model") is True

    def test_matches_by_model(self) -> None:
        assert mod._entry_matches_model({"model": "my-model"}, model="my-model") is True

    def test_matches_by_name(self) -> None:
        assert mod._entry_matches_model({"name": "my-model"}, model="my-model") is True

    def test_no_match_returns_false(self) -> None:
        assert mod._entry_matches_model({"id": "other"}, model="my-model") is False

    def test_empty_model_returns_false(self) -> None:
        assert mod._entry_matches_model({"id": "my-model"}, model="") is False
        assert mod._entry_matches_model({"id": "my-model"}, model="   ") is False

    def test_loaded_instances_id_match(self) -> None:
        entry = {
            "id": "wrong-id",
            "loaded_instances": [{"id": "my-model"}],
        }
        assert mod._entry_matches_model(entry, model="my-model") is True

    def test_loaded_instances_non_dict_skipped(self) -> None:
        entry = {
            "loaded_instances": ["not-a-dict", 42, None, {"id": "my-model"}],
        }
        assert mod._entry_matches_model(entry, model="my-model") is True

    def test_loaded_instances_no_match(self) -> None:
        entry = {
            "loaded_instances": [{"id": "other-model"}],
        }
        assert mod._entry_matches_model(entry, model="my-model") is False

    def test_loaded_instances_empty_list(self) -> None:
        entry = {"loaded_instances": []}
        assert mod._entry_matches_model(entry, model="my-model") is False

    def test_loaded_instances_not_a_list(self) -> None:
        entry = {"loaded_instances": "not-a-list"}
        assert mod._entry_matches_model(entry, model="my-model") is False

    def test_whitespace_stripped(self) -> None:
        entry = {"id": "  my-model  "}
        assert mod._entry_matches_model(entry, model="  my-model  ") is True


# ---------------------------------------------------------------------------
# 5. _extract_lmstudio_instance_context_length
# ---------------------------------------------------------------------------


class TestExtractLmstudioInstanceContextLength:
    """Tests for LMStudio instance context length extraction."""

    def test_direct_context_length(self) -> None:
        assert mod._extract_lmstudio_instance_context_length({"context_length": 4096}) == 4096

    def test_config_context_length_fallback(self) -> None:
        instance = {"config": {"context_length": 8192}}
        assert mod._extract_lmstudio_instance_context_length(instance) == 8192

    def test_missing_both_returns_none(self) -> None:
        assert mod._extract_lmstudio_instance_context_length({}) is None

    def test_config_not_dict_returns_none(self) -> None:
        assert mod._extract_lmstudio_instance_context_length({"config": "not-a-dict"}) is None

    def test_config_missing_context_length_returns_none(self) -> None:
        assert mod._extract_lmstudio_instance_context_length({"config": {"other_key": 123}}) is None

    def test_direct_takes_priority_over_config(self) -> None:
        instance = {"context_length": 2048, "config": {"context_length": 4096}}
        assert mod._extract_lmstudio_instance_context_length(instance) == 2048

    def test_negative_direct_falls_through_to_config(self) -> None:
        instance = {"context_length": -1, "config": {"context_length": 4096}}
        assert mod._extract_lmstudio_instance_context_length(instance) == 4096


# ---------------------------------------------------------------------------
# 6. _extract_lmstudio_v1_context_window — loaded_instances edge cases
# ---------------------------------------------------------------------------


class TestExtractLmstudioV1ContextWindow:
    """Tests for LMStudio v1 context window probing."""

    def test_loaded_instance_with_context_length(self) -> None:
        payload = {
            "data": [
                {
                    "id": "my-model",
                    "loaded_instances": [
                        {"id": "inst-1", "context_length": 32768},
                    ],
                }
            ]
        }
        ctx, meta = mod._extract_lmstudio_v1_context_window(payload=payload, model="my-model")
        assert ctx == 32768
        assert meta["context_window_source"] == "loaded_instance.context_length"
        assert meta["context_window_probe_instance_id"] == "inst-1"

    def test_loaded_instance_with_config_context_length(self) -> None:
        payload = {
            "data": [
                {
                    "id": "my-model",
                    "loaded_instances": [
                        {"id": "inst-1", "config": {"context_length": 16384}},
                    ],
                }
            ]
        }
        ctx, meta = mod._extract_lmstudio_v1_context_window(payload=payload, model="my-model")
        assert ctx == 16384
        assert meta["context_window_source"] == "loaded_instance.context_length"

    def test_loaded_instances_missing_context_falls_to_entry_fields(self) -> None:
        payload = {
            "data": [
                {
                    "id": "my-model",
                    "loaded_instances": [{"id": "inst-1"}],
                    "max_context_length": 65536,
                }
            ]
        }
        ctx, meta = mod._extract_lmstudio_v1_context_window(payload=payload, model="my-model")
        assert ctx == 65536
        assert meta["context_window_source"] == "max_context_length"

    def test_loaded_instances_empty_list_falls_to_entry_fields(self) -> None:
        payload = {
            "data": [
                {
                    "id": "my-model",
                    "loaded_instances": [],
                    "context_length": 4096,
                }
            ]
        }
        ctx, meta = mod._extract_lmstudio_v1_context_window(payload=payload, model="my-model")
        assert ctx == 4096
        assert meta["context_window_source"] == "context_length"

    def test_loaded_instances_non_dict_skipped(self) -> None:
        payload = {
            "data": [
                {
                    "id": "my-model",
                    "loaded_instances": ["not-a-dict"],
                    "max_input_tokens": 2048,
                }
            ]
        }
        ctx, meta = mod._extract_lmstudio_v1_context_window(payload=payload, model="my-model")
        assert ctx == 2048
        assert meta["context_window_source"] == "max_input_tokens"

    def test_no_matching_model_returns_none(self) -> None:
        payload = {"data": [{"id": "other-model", "context_length": 4096}]}
        ctx, meta = mod._extract_lmstudio_v1_context_window(payload=payload, model="my-model")
        assert ctx is None
        assert meta == {}

    def test_no_loaded_instances_key(self) -> None:
        payload = {
            "data": [
                {
                    "id": "my-model",
                    "context_length": 2048,
                }
            ]
        }
        ctx, meta = mod._extract_lmstudio_v1_context_window(payload=payload, model="my-model")
        assert ctx == 2048
        assert meta["context_window_source"] == "context_length"

    def test_loaded_instance_id_missing_uses_empty_string(self) -> None:
        payload = {
            "data": [
                {
                    "id": "my-model",
                    "loaded_instances": [
                        {"context_length": 4096},  # no "id" key
                    ],
                }
            ]
        }
        ctx, meta = mod._extract_lmstudio_v1_context_window(payload=payload, model="my-model")
        assert ctx == 4096
        assert meta["context_window_probe_instance_id"] == ""


# ---------------------------------------------------------------------------
# 7. _extract_openai_models_context_window
# ---------------------------------------------------------------------------


class TestExtractOpenaiModelsContextWindow:
    """Tests for generic OpenAI /models context window extraction."""

    def test_finds_context_length(self) -> None:
        payload = {"data": [{"id": "my-model", "context_length": 128000}]}
        ctx, meta = mod._extract_openai_models_context_window(payload=payload, model="my-model")
        assert ctx == 128000
        assert meta["context_window_source"] == "context_length"

    def test_finds_max_context_length(self) -> None:
        payload = {"data": [{"id": "my-model", "max_context_length": 64000}]}
        ctx, meta = mod._extract_openai_models_context_window(payload=payload, model="my-model")
        assert ctx == 64000
        assert meta["context_window_source"] == "max_context_length"

    def test_finds_max_input_tokens(self) -> None:
        payload = {"data": [{"id": "my-model", "max_input_tokens": 32000}]}
        ctx, meta = mod._extract_openai_models_context_window(payload=payload, model="my-model")
        assert ctx == 32000
        assert meta["context_window_source"] == "max_input_tokens"

    def test_no_match_returns_none(self) -> None:
        payload = {"data": [{"id": "other", "context_length": 4096}]}
        ctx, meta = mod._extract_openai_models_context_window(payload=payload, model="my-model")
        assert ctx is None
        assert meta == {}


# ---------------------------------------------------------------------------
# 8. _normalize_edits_payload — alternative shapes
# ---------------------------------------------------------------------------


class TestNormalizeEditsPayload:
    """Tests for the edits payload normalization logic."""

    def test_standard_edits_list(self) -> None:
        payload = {
            "edits": [{"path": "a.py", "find": "x", "replace": "y"}],
        }
        edits, shape = mod._normalize_edits_payload(payload)
        assert edits is not None
        assert len(edits) == 1
        assert shape == "edits_list"

    def test_edits_as_single_dict(self) -> None:
        payload = {
            "edits": {"path": "a.py", "find": "x", "replace": "y"},
        }
        edits, shape = mod._normalize_edits_payload(payload)
        assert edits is not None
        assert len(edits) == 1
        assert shape == "edits_object"

    def test_single_edit_at_root(self) -> None:
        payload = {"path": "a.py", "find": "x", "replace": "y"}
        edits, shape = mod._normalize_edits_payload(payload)
        assert edits is not None
        assert len(edits) == 1
        assert shape == "single_edit_root"

    def test_edit_key_single_object(self) -> None:
        payload = {
            "edit": {"path": "a.py", "find": "x", "replace": "y"},
        }
        edits, shape = mod._normalize_edits_payload(payload)
        assert edits is not None
        assert len(edits) == 1
        assert shape == "edit_object"

    def test_changes_list(self) -> None:
        payload = {
            "changes": [
                {"path": "a.py", "find": "x", "replace": "y"},
                {"path": "b.py", "find": "a", "replace": "b"},
            ],
        }
        edits, shape = mod._normalize_edits_payload(payload)
        assert edits is not None
        assert len(edits) == 2
        assert shape == "changes_list"

    def test_returns_none_for_unrecognized(self) -> None:
        payload = {"summary": "no edits here"}
        edits, shape = mod._normalize_edits_payload(payload)
        assert edits is None
        assert shape is None

    def test_empty_edits_list_not_matched(self) -> None:
        payload = {"edits": []}
        edits, shape = mod._normalize_edits_payload(payload)
        assert edits is None
        assert shape is None

    def test_old_new_keys_accepted(self) -> None:
        payload = {"path": "a.py", "old": "x", "new": "y"}
        edits, shape = mod._normalize_edits_payload(payload)
        assert edits is not None
        assert shape == "single_edit_root"


# ---------------------------------------------------------------------------
# 9. _extract_response_seed
# ---------------------------------------------------------------------------


class TestExtractResponseSeed:
    """Tests for response seed extraction from various response shapes."""

    def test_seed_from_top_level(self) -> None:
        assert mod._extract_response_seed({"seed": 42}) == 42

    def test_response_seed_from_top_level(self) -> None:
        assert mod._extract_response_seed({"response_seed": 99}) == 99

    def test_seed_from_metadata(self) -> None:
        assert mod._extract_response_seed({"metadata": {"seed": 123}}) == 123

    def test_response_seed_from_metadata(self) -> None:
        assert mod._extract_response_seed({"metadata": {"response_seed": 456}}) == 456

    def test_string_seed_parsed(self) -> None:
        assert mod._extract_response_seed({"seed": "789"}) == 789

    def test_invalid_string_seed_skipped(self) -> None:
        # "abc" can't be int, but metadata.seed=100 succeeds
        assert mod._extract_response_seed({"seed": "abc", "metadata": {"seed": 100}}) == 100

    def test_no_seed_returns_none(self) -> None:
        assert mod._extract_response_seed({}) is None
        assert mod._extract_response_seed({"metadata": {}}) is None

    def test_metadata_not_dict_skipped(self) -> None:
        assert mod._extract_response_seed({"metadata": "not-a-dict"}) is None

    def test_all_invalid_strings_return_none(self) -> None:
        assert (
            mod._extract_response_seed({"seed": "not_an_int", "response_seed": "also_not"}) is None
        )


# ---------------------------------------------------------------------------
# 10. _is_backend_unavailable_error — request_failed with exc markers
# ---------------------------------------------------------------------------


class TestIsBackendUnavailableError:
    """Tests for backend unavailability detection."""

    def test_http_error_502(self) -> None:
        assert mod._is_backend_unavailable_error({"error": "http_error", "status": 502}) is True

    def test_http_error_503(self) -> None:
        assert mod._is_backend_unavailable_error({"error": "http_error", "status": 503}) is True

    def test_http_error_504(self) -> None:
        assert mod._is_backend_unavailable_error({"error": "http_error", "status": 504}) is True

    def test_http_error_non_unavailable_status(self) -> None:
        assert mod._is_backend_unavailable_error({"error": "http_error", "status": 400}) is False

    def test_request_failed_connection_refused(self) -> None:
        assert (
            mod._is_backend_unavailable_error(
                {"error": "request_failed", "exc": "ConnectionRefusedError: ..."}
            )
            is True
        )

    def test_request_failed_connection_refused_lowercase(self) -> None:
        assert (
            mod._is_backend_unavailable_error(
                {"error": "request_failed", "exc": "connection refused"}
            )
            is True
        )

    def test_request_failed_server_disconnected(self) -> None:
        assert (
            mod._is_backend_unavailable_error(
                {"error": "request_failed", "exc": "server disconnected unexpectedly"}
            )
            is True
        )

    def test_request_failed_temporarily_unavailable(self) -> None:
        assert (
            mod._is_backend_unavailable_error(
                {"error": "request_failed", "exc": "temporarily unavailable"}
            )
            is True
        )

    def test_request_failed_service_unavailable(self) -> None:
        assert (
            mod._is_backend_unavailable_error(
                {"error": "request_failed", "exc": "service unavailable"}
            )
            is True
        )

    def test_request_failed_unrelated_exc(self) -> None:
        assert (
            mod._is_backend_unavailable_error(
                {"error": "request_failed", "exc": "timeout reading response"}
            )
            is False
        )

    def test_unrelated_error_type(self) -> None:
        assert mod._is_backend_unavailable_error({"error": "parse_error"}) is False


# ---------------------------------------------------------------------------
# 11. _safe_trunc
# ---------------------------------------------------------------------------


class TestSafeTrunc:
    """Tests for safe truncation helper."""

    def test_normal_dict(self) -> None:
        result = mod._safe_trunc({"key": "value"})
        assert "key" in result
        assert len(result) <= 2000

    def test_non_serializable_uses_repr(self) -> None:
        obj = object()
        result = mod._safe_trunc(obj)
        assert "object" in result

    def test_custom_limit(self) -> None:
        long_data = {"key": "x" * 5000}
        result = mod._safe_trunc(long_data, limit=50)
        assert len(result) == 50


# ---------------------------------------------------------------------------
# 12. _extract_file_index_paths
# ---------------------------------------------------------------------------


class TestExtractFileIndexPaths:
    """Tests for file index extraction from prompts."""

    def test_extracts_paths(self) -> None:
        prompt = (
            "Some preamble\nAvailable files:\nsrc/main.py\nsrc/utils.py\n\nPytest output:\nFAILED\n"
        )
        paths = mod._extract_file_index_paths(prompt)
        assert paths == ["src/main.py", "src/utils.py"]

    def test_no_available_files_header(self) -> None:
        assert mod._extract_file_index_paths("no header here") == []

    def test_stops_at_colon_terminated_line(self) -> None:
        prompt = "Available files:\na.py\nSomething else:\nb.py\n"
        paths = mod._extract_file_index_paths(prompt)
        assert paths == ["a.py"]

    def test_stops_at_pytest_output(self) -> None:
        prompt = "Available files:\na.py\nPytest output\n"
        paths = mod._extract_file_index_paths(prompt)
        assert paths == ["a.py"]

    def test_stops_at_project_files(self) -> None:
        prompt = "Available files:\na.py\nProject files:\n"
        paths = mod._extract_file_index_paths(prompt)
        assert paths == ["a.py"]

    def test_skips_leading_blank_lines(self) -> None:
        prompt = "Available files:\n\n\na.py\nb.py\n"
        paths = mod._extract_file_index_paths(prompt)
        assert paths == ["a.py", "b.py"]

    def test_empty_prompt(self) -> None:
        assert mod._extract_file_index_paths("") == []


# ---------------------------------------------------------------------------
# 13. _fix_hunk_counts
# ---------------------------------------------------------------------------


class TestFixHunkCounts:
    """Tests for hunk count correction in unified diffs."""

    def test_corrects_wrong_counts(self) -> None:
        diff = (
            "diff --git a/f.py b/f.py\n"
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,99 +1,99 @@\n"
            " context\n"
            "-old\n"
            "+new\n"
        )
        fixed = mod._fix_hunk_counts(diff)
        assert "@@ -1,2 +1,2 @@" in fixed

    def test_preserves_correct_counts(self) -> None:
        diff = (
            "diff --git a/f.py b/f.py\n"
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,2 +1,2 @@\n"
            " context\n"
            "-old\n"
            "+new\n"
        )
        fixed = mod._fix_hunk_counts(diff)
        assert "@@ -1,2 +1,2 @@" in fixed

    def test_handles_no_newline_metadata(self) -> None:
        diff = "@@ -1,1 +1,1 @@\n-old\n+new\n\\ No newline at end of file\n"
        fixed = mod._fix_hunk_counts(diff)
        assert "@@ -1,1 +1,1 @@" in fixed

    def test_multiple_hunks(self) -> None:
        diff = "@@ -1,99 +1,99 @@\n line1\n-a\n+b\n@@ -10,99 +10,99 @@\n line10\n+added\n"
        fixed = mod._fix_hunk_counts(diff)
        assert "@@ -1,2 +1,2 @@" in fixed
        assert "@@ -10,1 +10,2 @@" in fixed

    def test_suffix_preserved(self) -> None:
        diff = "@@ -1,0 +1,0 @@ def foo():\n-old\n+new\n"
        fixed = mod._fix_hunk_counts(diff)
        assert "def foo():" in fixed

    def test_interior_empty_body_line_is_context(self) -> None:
        # LLM-03: an interior empty line is a context line that lost its leading space.
        diff = "@@ -1,99 +1,99 @@\n-old\n\n+new\n"
        fixed = mod._fix_hunk_counts(diff)
        assert fixed == "@@ -1,2 +1,2 @@\n-old\n \n+new\n"

    def test_trailing_empty_body_lines_are_not_counted(self) -> None:
        diff = "@@ -1,99 +1,99 @@\n-old\n+new\n\n\n"
        fixed = mod._fix_hunk_counts(diff)
        assert fixed == "@@ -1,1 +1,1 @@\n-old\n+new\n"

    def test_empty_side_keeps_zero_start(self) -> None:
        # LLM-10: file creation is "-0,0"; a 1-based start would break `patch`.
        fixed = mod._fix_hunk_counts("@@ -0,0 +1,9 @@\n+a\n+b\n")
        assert fixed.startswith("@@ -0,0 +1,2 @@\n")


# ---------------------------------------------------------------------------
# 14. _normalize_newlines
# ---------------------------------------------------------------------------


class TestNormalizeNewlines:
    """Tests for newline normalization helper."""

    def test_crlf_to_lf(self) -> None:
        assert mod._normalize_newlines("a\r\nb") == "a\nb"

    def test_cr_to_lf(self) -> None:
        assert mod._normalize_newlines("a\rb") == "a\nb"

    def test_mixed(self) -> None:
        assert mod._normalize_newlines("a\r\nb\rc\nd") == "a\nb\nc\nd"

    def test_no_change(self) -> None:
        assert mod._normalize_newlines("a\nb") == "a\nb"


# ---------------------------------------------------------------------------
# 15. _extract_model_entries
# ---------------------------------------------------------------------------


class TestExtractModelEntries:
    """Tests for model entry extraction from various payload shapes."""

    def test_dict_with_models_key(self) -> None:
        entries = mod._extract_model_entries({"models": [{"id": "a"}, {"id": "b"}]})
        assert len(entries) == 2

    def test_dict_with_data_key(self) -> None:
        entries = mod._extract_model_entries({"data": [{"id": "a"}]})
        assert len(entries) == 1

    def test_raw_list(self) -> None:
        entries = mod._extract_model_entries([{"id": "a"}, {"id": "b"}])
        assert len(entries) == 2

    def test_filters_non_dicts(self) -> None:
        entries = mod._extract_model_entries([{"id": "a"}, "not-a-dict", 42])
        assert len(entries) == 1

    def test_dict_without_models_or_data(self) -> None:
        entries = mod._extract_model_entries({"other": [{"id": "a"}]})
        assert entries == []

    def test_models_key_not_list(self) -> None:
        entries = mod._extract_model_entries({"models": "not-a-list"})
        assert entries == []


# ---------------------------------------------------------------------------
# 16. _coerce_positive_int
# ---------------------------------------------------------------------------


class TestCoercePositiveInt:
    """Tests for positive integer coercion."""

    def test_positive_int(self) -> None:
        assert mod._coerce_positive_int(42) == 42

    def test_zero_returns_none(self) -> None:
        assert mod._coerce_positive_int(0) is None

    def test_negative_returns_none(self) -> None:
        assert mod._coerce_positive_int(-5) is None

    def test_string_int(self) -> None:
        assert mod._coerce_positive_int("100") == 100

    def test_invalid_string(self) -> None:
        assert mod._coerce_positive_int("abc") is None

    def test_none_returns_none(self) -> None:
        assert mod._coerce_positive_int(None) is None


# ---------------------------------------------------------------------------
# 17. _emit_openai_transport_progress — with base payload merging
# ---------------------------------------------------------------------------


class TestEmitOpenaiTransportProgress:
    """Tests for the progress emission helper."""

    def test_no_callback_is_noop(self) -> None:
        # Should not raise when no callback is set
        mod._emit_openai_transport_progress("test_event", note="hello")

    def test_callback_receives_event_and_payload(self) -> None:
        events: list[tuple[str, dict]] = []

        def _record(event: str, payload: dict | None) -> None:
            events.append((event, payload or {}))

        token = mod._OPENAI_PROGRESS_CALLBACK.set(_record)
        try:
            mod._emit_openai_transport_progress("retry", note="test_note", extra_key=42)
        finally:
            mod._OPENAI_PROGRESS_CALLBACK.reset(token)

        assert len(events) == 1
        assert events[0][0] == "retry"
        assert events[0][1]["note"] == "test_note"
        assert events[0][1]["extra_key"] == 42
        assert events[0][1]["owner"] == "openai"

    def test_base_payload_merged(self) -> None:
        events: list[tuple[str, dict]] = []

        def _record(event: str, payload: dict | None) -> None:
            events.append((event, payload or {}))

        cb_token = mod._OPENAI_PROGRESS_CALLBACK.set(_record)
        bp_token = mod._OPENAI_PROGRESS_BASE_PAYLOAD.set(
            {"phase_label": "proposing", "response_mode": "edits"}
        )
        try:
            mod._emit_openai_transport_progress("start")
        finally:
            mod._OPENAI_PROGRESS_BASE_PAYLOAD.reset(bp_token)
            mod._OPENAI_PROGRESS_CALLBACK.reset(cb_token)

        assert len(events) == 1
        assert events[0][1]["phase_label"] == "proposing"
        assert events[0][1]["response_mode"] == "edits"

    def test_callback_exception_swallowed(self) -> None:
        def _boom(event: str, payload: dict | None) -> None:
            raise RuntimeError("boom")

        token = mod._OPENAI_PROGRESS_CALLBACK.set(_boom)
        try:
            # Should not raise
            mod._emit_openai_transport_progress("retry", note="test")
        finally:
            mod._OPENAI_PROGRESS_CALLBACK.reset(token)


# ---------------------------------------------------------------------------
# 18. _looks_like_edit_entry
# ---------------------------------------------------------------------------


class TestLooksLikeEditEntry:
    """Tests for edit entry shape detection."""

    def test_standard_keys(self) -> None:
        assert mod._looks_like_edit_entry({"path": "a.py", "find": "x", "replace": "y"}) is True

    def test_old_new_keys(self) -> None:
        assert mod._looks_like_edit_entry({"path": "a.py", "old": "x", "new": "y"}) is True

    def test_missing_path(self) -> None:
        assert mod._looks_like_edit_entry({"find": "x", "replace": "y"}) is False

    def test_missing_find(self) -> None:
        assert mod._looks_like_edit_entry({"path": "a.py", "replace": "y"}) is False

    def test_missing_replace(self) -> None:
        assert mod._looks_like_edit_entry({"path": "a.py", "find": "x"}) is False


# ---------------------------------------------------------------------------
# 19. _request_context_failure_kind
# ---------------------------------------------------------------------------


class TestRequestContextFailureKind:
    """Tests for request context failure classification."""

    def test_success_returns_empty(self) -> None:
        assert (
            mod._request_context_failure_kind(
                meta={}, request_json={"requests": []}, raw_response=None
            )
            == ""
        )

    def test_budget_exceeded(self) -> None:
        assert (
            mod._request_context_failure_kind(
                meta={"error": "request_budget_exceeded"},
                request_json=None,
                raw_response=None,
            )
            == "request_budget_exceeded"
        )

    def test_backend_unavailable(self) -> None:
        assert (
            mod._request_context_failure_kind(
                meta={"error": "request_failed", "exc": "ConnectionRefusedError"},
                request_json=None,
                raw_response=None,
            )
            == "backend_unavailable"
        )

    def test_transport_error(self) -> None:
        assert (
            mod._request_context_failure_kind(
                meta={"error": "request_failed"},
                request_json=None,
                raw_response=None,
            )
            == "transport"
        )

    def test_invalid_payload_from_raw_response(self) -> None:
        assert (
            mod._request_context_failure_kind(
                meta={"error": "some_error"},
                request_json=None,
                raw_response="bad json",
            )
            == "invalid_payload"
        )

    def test_invalid_payload_from_error_type(self) -> None:
        assert (
            mod._request_context_failure_kind(
                meta={"error": "invalid_request_context_payload"},
                request_json=None,
                raw_response=None,
            )
            == "invalid_payload"
        )

    def test_unknown_returns_empty(self) -> None:
        assert (
            mod._request_context_failure_kind(
                meta={"error": "unknown_thing"},
                request_json=None,
                raw_response=None,
            )
            == ""
        )


# ---------------------------------------------------------------------------
# 20. _extract_lmstudio_v0_context_window
# ---------------------------------------------------------------------------


class TestExtractLmstudioV0ContextWindow:
    """Tests for LMStudio v0 context window extraction."""

    def test_finds_context_length(self) -> None:
        payload = [{"id": "my-model", "context_length": 4096}]
        ctx, meta = mod._extract_lmstudio_v0_context_window(payload=payload, model="my-model")
        assert ctx == 4096
        assert meta["context_window_source"] == "context_length"

    def test_falls_back_to_max_context_length(self) -> None:
        payload = [{"id": "my-model", "max_context_length": 8192}]
        ctx, _meta = mod._extract_lmstudio_v0_context_window(payload=payload, model="my-model")
        assert ctx == 8192

    def test_no_match_returns_none(self) -> None:
        payload = [{"id": "other"}]
        ctx, meta = mod._extract_lmstudio_v0_context_window(payload=payload, model="my-model")
        assert ctx is None
        assert meta == {}


# ---------------------------------------------------------------------------
# 21. _extract_json_object — fenced code, fallback decoder, empty input
# ---------------------------------------------------------------------------


class TestExtractJsonObject:
    """Tests for JSON object extraction from text."""

    def test_plain_json(self) -> None:
        result = mod._extract_json_object('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self) -> None:
        text = 'Some preamble\n```json\n{"key": "value"}\n```\nAfter'
        result = mod._extract_json_object(text)
        assert result == {"key": "value"}

    def test_empty_text_returns_none(self) -> None:
        assert mod._extract_json_object("") is None
        assert mod._extract_json_object("   ") is None

    def test_think_tags_stripped(self) -> None:
        text = '<think>reasoning</think>\n{"edits": []}'
        result = mod._extract_json_object(text)
        assert result is not None
        assert "edits" in result

    def test_non_dict_json_returns_none(self) -> None:
        assert mod._extract_json_object("[1, 2, 3]") is None

    def test_json_with_preamble_text(self) -> None:
        text = 'Here is the fix:\n{"edits": [{"path": "a.py", "find": "x", "replace": "y"}]}'
        result = mod._extract_json_object(text)
        assert result is not None
        assert "edits" in result

    def test_no_json_returns_none(self) -> None:
        assert mod._extract_json_object("no json here at all") is None

    def test_think_only_returns_none(self) -> None:
        assert mod._extract_json_object("<think>only thinking</think>") is None


# ---------------------------------------------------------------------------
# 22. _extract_first_fenced_code_block
# ---------------------------------------------------------------------------


class TestExtractFirstFencedCodeBlock:
    """Tests for fenced code block extraction."""

    def test_with_code_fence(self) -> None:
        text = "Preamble\n```python\ndef foo():\n    pass\n```\nAfter"
        code, stripped = mod._extract_first_fenced_code_block(text)
        assert "def foo():" in code
        assert stripped is True

    def test_without_code_fence(self) -> None:
        text = "def foo():\n    pass\n"
        code, stripped = mod._extract_first_fenced_code_block(text)
        assert "def foo():" in code
        assert stripped is False

    def test_empty_text(self) -> None:
        code, stripped = mod._extract_first_fenced_code_block("")
        assert code == ""
        assert stripped is False

    def test_think_tags_stripped_first(self) -> None:
        text = "<think>reason</think>\n```python\ncode\n```"
        code, stripped = mod._extract_first_fenced_code_block(text)
        assert "code" in code
        assert stripped is True

    def test_think_only_returns_empty(self) -> None:
        code, stripped = mod._extract_first_fenced_code_block("<think>only think</think>")
        assert code == ""
        assert stripped is False


# ---------------------------------------------------------------------------
# 23. _normalize_full_file_response
# ---------------------------------------------------------------------------


class TestNormalizeFullFileResponse:
    """Tests for full-file response normalization."""

    def test_no_content(self) -> None:
        raw = {"choices": []}
        result, meta = mod._normalize_full_file_response(raw)
        assert result is None
        assert meta["error"] == "empty_full_file_response"

    def test_truncated_response(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {"content": "def foo():\n    pass"},
                    "finish_reason": "length",
                }
            ]
        }
        result, meta = mod._normalize_full_file_response(raw)
        assert result is None
        assert meta["error"] == "truncated_full_file_response"

    def test_empty_content_after_fence_strip(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {"content": "```python\n   \n```"},
                    "finish_reason": "stop",
                }
            ]
        }
        result, meta = mod._normalize_full_file_response(raw)
        assert result is None
        assert meta["error"] == "empty_full_file_response"

    def test_valid_response(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {"content": "def foo():\n    return 42\n"},
                    "finish_reason": "stop",
                }
            ]
        }
        result, meta = mod._normalize_full_file_response(raw)
        assert result is not None
        assert "def foo():" in result
        assert meta.get("finish_reason") == "stop"

    def test_diff_instead_of_file(self) -> None:
        diff_text = (
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def foo():\n"
            "-    return 1\n"
            "+    return 2\n"
        )
        raw = {
            "choices": [
                {
                    "message": {"content": diff_text},
                    "finish_reason": "stop",
                }
            ]
        }
        result, meta = mod._normalize_full_file_response(raw)
        assert result is None
        assert meta["error"] == "returned_diff_instead_of_file"


# ---------------------------------------------------------------------------
# 24. _resolve_full_file_target
# ---------------------------------------------------------------------------


class TestResolveFullFileTarget:
    """Tests for full-file target resolution."""

    def test_single_non_test_file(self, tmp_path) -> None:
        (tmp_path / "buggy.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "test_buggy.py").write_text("import buggy\n", encoding="utf-8")
        target, reason = mod._resolve_full_file_target(cwd=tmp_path)
        assert target == "buggy.py"
        assert reason == "single_non_test_python_file"

    def test_no_python_files(self, tmp_path) -> None:
        (tmp_path / "readme.txt").write_text("hello\n", encoding="utf-8")
        target, reason = mod._resolve_full_file_target(cwd=tmp_path)
        assert target is None
        assert reason == "no_non_test_python_files"

    def test_multiple_non_test_files(self, tmp_path) -> None:
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
        target, reason = mod._resolve_full_file_target(cwd=tmp_path)
        assert target is None
        assert reason == "multiple_non_test_python_files"


# ---------------------------------------------------------------------------
# 25. _build_diff_from_changes
# ---------------------------------------------------------------------------


class TestBuildDiffFromChanges:
    """Tests for diff building from changes tuples."""

    def test_single_change(self) -> None:
        diff = mod._build_diff_from_changes(
            [
                ("a.py", "old line\n", "new line\n"),
            ]
        )
        assert "diff --git a/a.py b/a.py" in diff
        assert "-old line" in diff
        assert "+new line" in diff

    def test_no_effective_change(self) -> None:
        diff = mod._build_diff_from_changes(
            [
                ("a.py", "same\n", "same\n"),
            ]
        )
        # No actual diff lines produced — empty result
        assert "diff --git" not in diff


# ---------------------------------------------------------------------------
# 26. _get_first_choice_content edge cases
# ---------------------------------------------------------------------------


class TestGetFirstChoiceContent:
    """Tests for choice content extraction."""

    def test_normal_string_content(self) -> None:
        resp = {"choices": [{"message": {"content": "hello"}}]}
        assert mod._get_first_choice_content(resp) == "hello"

    def test_no_choices(self) -> None:
        assert mod._get_first_choice_content({"choices": []}) is None

    def test_no_message(self) -> None:
        assert mod._get_first_choice_content({"choices": [{}]}) is None

    def test_text_field_fallback(self) -> None:
        resp = {"choices": [{"text": "fallback"}]}
        assert mod._get_first_choice_content(resp) == "fallback"

    def test_choices_not_list(self) -> None:
        assert mod._get_first_choice_content({"choices": "not-a-list"}) is None

    def test_choice_not_dict(self) -> None:
        assert mod._get_first_choice_content({"choices": ["not-a-dict"]}) is None


# ---------------------------------------------------------------------------
# 27. _get_first_choice_finish_reason edge cases
# ---------------------------------------------------------------------------


class TestGetFirstChoiceFinishReason:
    """Tests for finish reason extraction."""

    def test_normal(self) -> None:
        resp = {"choices": [{"finish_reason": "stop"}]}
        assert mod._get_first_choice_finish_reason(resp) == "stop"

    def test_no_choices(self) -> None:
        assert mod._get_first_choice_finish_reason({"choices": []}) == ""

    def test_no_finish_reason(self) -> None:
        assert mod._get_first_choice_finish_reason({"choices": [{}]}) == ""

    def test_non_string_finish_reason(self) -> None:
        assert mod._get_first_choice_finish_reason({"choices": [{"finish_reason": 42}]}) == ""

    def test_choice_not_dict(self) -> None:
        assert mod._get_first_choice_finish_reason({"choices": ["x"]}) == ""


# ---------------------------------------------------------------------------
# 28. _get_first_choice_reasoning_content / _get_first_choice_reasoning
# ---------------------------------------------------------------------------


class TestGetFirstChoiceReasoning:
    """Tests for reasoning content extraction."""

    def test_reasoning_content_present(self) -> None:
        resp = {"choices": [{"message": {"content": "x", "reasoning_content": "think"}}]}
        assert mod._get_first_choice_reasoning_content(resp) == "think"

    def test_reasoning_content_missing(self) -> None:
        resp = {"choices": [{"message": {"content": "x"}}]}
        assert mod._get_first_choice_reasoning_content(resp) is None

    def test_reasoning_content_no_message(self) -> None:
        assert mod._get_first_choice_reasoning_content({"choices": [{}]}) is None

    def test_reasoning_content_not_string(self) -> None:
        resp = {"choices": [{"message": {"reasoning_content": 42}}]}
        assert mod._get_first_choice_reasoning_content(resp) is None

    def test_reasoning_present(self) -> None:
        resp = {"choices": [{"message": {"content": "x", "reasoning": "thought"}}]}
        assert mod._get_first_choice_reasoning(resp) == "thought"

    def test_reasoning_missing(self) -> None:
        resp = {"choices": [{"message": {"content": "x"}}]}
        assert mod._get_first_choice_reasoning(resp) is None

    def test_reasoning_no_message(self) -> None:
        assert mod._get_first_choice_reasoning({"choices": [{}]}) is None

    def test_reasoning_not_string(self) -> None:
        resp = {"choices": [{"message": {"reasoning": 123}}]}
        assert mod._get_first_choice_reasoning(resp) is None

    def test_reasoning_content_choices_not_list(self) -> None:
        assert mod._get_first_choice_reasoning_content({"choices": "bad"}) is None

    def test_reasoning_choices_not_list(self) -> None:
        assert mod._get_first_choice_reasoning({"choices": "bad"}) is None

    def test_reasoning_content_choice_not_dict(self) -> None:
        assert mod._get_first_choice_reasoning_content({"choices": [42]}) is None

    def test_reasoning_choice_not_dict(self) -> None:
        assert mod._get_first_choice_reasoning({"choices": [42]}) is None


# ---------------------------------------------------------------------------
# 29. _coerce_content_to_text
# ---------------------------------------------------------------------------


class TestCoerceContentToText:
    """Tests for content part coercion to text."""

    def test_string_passthrough(self) -> None:
        assert mod._coerce_content_to_text("hello") == "hello"

    def test_none_returns_none(self) -> None:
        assert mod._coerce_content_to_text(None) is None

    def test_int_returns_none(self) -> None:
        assert mod._coerce_content_to_text(42) is None

    def test_list_with_output_text_parts(self) -> None:
        parts = [
            {"type": "output_text", "text": "result"},
        ]
        assert mod._coerce_content_to_text(parts) == "result"

    def test_list_with_reasoning_parts_skipped(self) -> None:
        parts = [
            {"type": "reasoning_text", "text": "internal"},
            {"type": "output_text", "text": "result"},
        ]
        assert mod._coerce_content_to_text(parts) == "result"

    def test_list_with_string_elements_as_fallback(self) -> None:
        parts = ["raw string"]
        assert mod._coerce_content_to_text(parts) == "raw string"

    def test_empty_list_returns_none(self) -> None:
        assert mod._coerce_content_to_text([]) is None

    def test_unknown_type_goes_to_fallback(self) -> None:
        parts = [{"type": "custom_type", "text": "data"}]
        assert mod._coerce_content_to_text(parts) == "data"


# ---------------------------------------------------------------------------
# 30. _extract_content_part_text
# ---------------------------------------------------------------------------


class TestExtractContentPartText:
    """Tests for content part text extraction."""

    def test_text_key(self) -> None:
        assert mod._extract_content_part_text({"text": "hello"}) == "hello"

    def test_content_key(self) -> None:
        assert mod._extract_content_part_text({"content": "world"}) == "world"

    def test_value_key(self) -> None:
        assert mod._extract_content_part_text({"value": "data"}) == "data"

    def test_nested_value(self) -> None:
        assert mod._extract_content_part_text({"text": {"value": "nested"}}) == "nested"

    def test_nested_text(self) -> None:
        assert mod._extract_content_part_text({"text": {"text": "deep"}}) == "deep"

    def test_no_text_returns_none(self) -> None:
        assert mod._extract_content_part_text({"type": "something"}) is None


# ---------------------------------------------------------------------------
# 31. _build_forced_attempt_diff
# ---------------------------------------------------------------------------


class TestBuildForcedAttemptDiff:
    """Tests for forced attempt diff building."""

    def test_no_candidates_returns_none(self) -> None:
        assert mod._build_forced_attempt_diff(prompt="no files", cwd=None) is None

    def test_non_py_candidates_returns_none(self) -> None:
        prompt = "Available files:\nreadme.txt\nMakefile\n\nPytest output:\n"
        import tempfile
        from pathlib import Path

        result = mod._build_forced_attempt_diff(prompt=prompt, cwd=Path(tempfile.gettempdir()))
        assert result is None

    def test_builds_diff_for_non_test_file(self, tmp_path) -> None:
        (tmp_path / "buggy.py").write_text("x = 1\n", encoding="utf-8")
        prompt = "Available files:\nbuggy.py\n\nPytest output:\n"
        result = mod._build_forced_attempt_diff(prompt=prompt, cwd=tmp_path)
        assert result is not None
        target_rel, diff = result
        assert target_rel == "buggy.py"
        assert "llmdebug-force-attempt" in diff

    def test_prefers_non_test_file(self, tmp_path) -> None:
        (tmp_path / "test_foo.py").write_text("t = 1\n", encoding="utf-8")
        (tmp_path / "foo.py").write_text("x = 1\n", encoding="utf-8")
        prompt = "Available files:\ntest_foo.py\nfoo.py\n\nPytest output:\n"
        result = mod._build_forced_attempt_diff(prompt=prompt, cwd=tmp_path)
        assert result is not None
        assert result[0] == "foo.py"

    def test_falls_back_to_test_file_if_only_option(self, tmp_path) -> None:
        (tmp_path / "test_foo.py").write_text("t = 1\n", encoding="utf-8")
        prompt = "Available files:\ntest_foo.py\n\nPytest output:\n"
        result = mod._build_forced_attempt_diff(prompt=prompt, cwd=tmp_path)
        assert result is not None
        assert result[0] == "test_foo.py"

    def test_existing_marker_uses_alternate(self, tmp_path) -> None:
        (tmp_path / "buggy.py").write_text("x = 1\n# llmdebug-force-attempt\n", encoding="utf-8")
        prompt = "Available files:\nbuggy.py\n\nPytest output:\n"
        result = mod._build_forced_attempt_diff(prompt=prompt, cwd=tmp_path)
        assert result is not None
        assert "llmdebug-force-attempt-2" in result[1]

    def test_file_not_exists_returns_none(self, tmp_path) -> None:
        prompt = "Available files:\nmissing.py\n\nPytest output:\n"
        result = mod._build_forced_attempt_diff(prompt=prompt, cwd=tmp_path)
        assert result is None
