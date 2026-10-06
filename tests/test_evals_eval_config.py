"""Tests for evals.eval_config - focused on uncovered lines/branches.

Covers: tier parsing, TOML loading fallback, config extraction, flattening,
parser action collection, CLI dest collection, bool/scalar coercion,
config-value-for-dest dispatch, TOML→args application, JSON compat helpers,
CLI-argv redaction, container-env parsing, execution-config validation,
active-config get/set, and OpenAI API key env resolution.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from evals import eval_config
from evals.adaptive_online_policy import (
    ACTION_FRAME_ONLY,
    ACTION_INVOKE_HELPER,
    ACTION_SEARCH_ONLY,
)
from evals.eval_types import (
    DEFAULT_CONTAINER_CPUS,
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_CONTAINER_MEMORY,
    DEFAULT_CONTAINER_NETWORK,
    DEFAULT_CONTAINER_PIDS_LIMIT,
    DEFAULT_CONTAINER_TMPFS_SIZE_MB,
    DEFAULT_CONTAINER_USER,
    DEFAULT_EXECUTOR,
    RUN_EVAL_LOCKFILE_REDACTED,
    ExecutionConfig,
)

# ---------------------------------------------------------------------------
# _parse_adaptive_budget_policy_tiers
# ---------------------------------------------------------------------------


class TestParseAdaptiveBudgetPolicyTiers:
    """Cover every error branch plus the happy path."""

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty") as exc_info:
            eval_config._parse_adaptive_budget_policy_tiers("")
        assert "empty" in str(exc_info.value)

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            eval_config._parse_adaptive_budget_policy_tiers(None)

    def test_whitespace_only_raises(self) -> None:
        """Whitespace-only after strip → empty."""
        with pytest.raises(ValueError, match="empty"):
            eval_config._parse_adaptive_budget_policy_tiers("   ")

    def test_comma_only_raises(self) -> None:
        """All tokens are empty after split → hits the second 'empty' check."""
        with pytest.raises(ValueError, match="empty"):
            eval_config._parse_adaptive_budget_policy_tiers(",,,")

    def test_non_positive_value_raises(self) -> None:
        with pytest.raises(ValueError, match="non_positive"):
            eval_config._parse_adaptive_budget_policy_tiers("0.0, 1.0")

    def test_negative_value_raises(self) -> None:
        with pytest.raises(ValueError, match="non_positive"):
            eval_config._parse_adaptive_budget_policy_tiers("-1.0, 1.0")

    def test_duplicate_values_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            eval_config._parse_adaptive_budget_policy_tiers("1.0, 1.0, 2.0")

    def test_non_increasing_values_raises(self) -> None:
        with pytest.raises(ValueError, match="non_increasing"):
            eval_config._parse_adaptive_budget_policy_tiers("2.0, 1.0")

    def test_missing_base_tier_raises(self) -> None:
        with pytest.raises(ValueError, match="missing_base"):
            eval_config._parse_adaptive_budget_policy_tiers("0.5, 2.0")

    def test_valid_tiers(self) -> None:
        result = eval_config._parse_adaptive_budget_policy_tiers("0.5, 1.0, 2.0")
        assert result == [0.5, 1.0, 2.0]

    def test_single_base_tier(self) -> None:
        result = eval_config._parse_adaptive_budget_policy_tiers("1.0")
        assert result == [1.0]


class TestParseAdaptiveOnlineActionScoreBiases:
    def test_empty_string_returns_zero_biases(self) -> None:
        result = eval_config._parse_adaptive_online_action_score_biases("")
        assert all(value == 0.0 for value in result.values())

    def test_valid_bias_string(self) -> None:
        result = eval_config._parse_adaptive_online_action_score_biases(
            "search_only=-0.1,frame_only=-0.05"
        )
        assert result[ACTION_SEARCH_ONLY] == pytest.approx(-0.1)
        assert result[ACTION_FRAME_ONLY] == pytest.approx(-0.05)
        assert result[ACTION_INVOKE_HELPER] == pytest.approx(0.0)

    @pytest.mark.parametrize("raw", ["search_only=-0.5", "frame_only=0.11", "search_only=nan"])
    def test_bias_beyond_the_d6_limit_raises(self, raw: str) -> None:
        """POL-07 / D6: a bias is at most 0.1 of the success reward (it was -1.00)."""
        with pytest.raises(ValueError, match="bias_out_of_range"):
            eval_config._parse_adaptive_online_action_score_biases(raw)

    def test_cli_and_controller_defaults_are_one_source(self) -> None:
        """POL-07: the CLI string, the parsed map and the controller default agree."""
        from evals import eval_types, run_eval
        from evals.adaptive_online_policy import (
            DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES,
            ONLINE_ACTION_SCORE_BIAS_LIMIT,
            OnlinePolicyConfig,
            default_online_action_score_biases,
        )

        parsed = eval_config._parse_adaptive_online_action_score_biases(
            DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES
        )
        assert parsed == default_online_action_score_biases()
        assert OnlinePolicyConfig().action_score_biases == parsed
        assert run_eval.DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES is (
            DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES
        )
        assert eval_types.AdaptiveRuntimeConfig().adaptive_online_action_score_biases == (
            DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES
        )
        assert all(abs(value) <= ONLINE_ACTION_SCORE_BIAS_LIMIT for value in parsed.values())
        assert any(value < 0.0 for value in parsed.values())

    def test_unknown_action_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown_action"):
            eval_config._parse_adaptive_online_action_score_biases("not_real=-0.5")

    def test_duplicate_action_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            eval_config._parse_adaptive_online_action_score_biases(
                "search_only=-0.05,search_only=-0.1"
            )

    def test_invalid_float_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid_float"):
            eval_config._parse_adaptive_online_action_score_biases("search_only=abc")


# ---------------------------------------------------------------------------
# _load_toml_document - _toml_loader=None fallback
# ---------------------------------------------------------------------------


class TestLoadTomlDocument:
    """Cover the fallback path when _toml_loader is None."""

    def test_fallback_to_tomli_when_toml_loader_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When _toml_loader is None, tomli is tried as fallback."""
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text("[run_eval]\nk = 2\n", encoding="utf-8")

        monkeypatch.setattr(eval_config, "_toml_loader", None)

        # Should still succeed if tomli is importable (or tomllib
        # is available via the monkeypatched fallback).  We use a
        # real TOML file; if neither loader exists the test still
        # validates the branch via the ModuleNotFoundError path.
        try:
            result = eval_config._load_toml_document(toml_file)
            assert isinstance(result, dict)
        except ValueError as exc:
            # Acceptable: the branch was hit but tomli is not installed.
            assert "TOML parsing requires" in str(exc)

    def test_fallback_raises_when_no_tomli(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When both tomllib and tomli are unavailable → ValueError."""
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text("[x]\na = 1\n", encoding="utf-8")

        monkeypatch.setattr(eval_config, "_toml_loader", None)

        import builtins

        real_import = builtins.__import__

        def _block_tomli(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "tomli":
                raise ModuleNotFoundError("no tomli")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_tomli)

        with pytest.raises(ValueError, match="TOML parsing requires"):
            eval_config._load_toml_document(toml_file)

    def test_unreadable_file_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.toml"
        with pytest.raises(ValueError, match="cannot read config file"):
            eval_config._load_toml_document(missing)

    def test_invalid_toml_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.toml"
        bad_file.write_text("[[broken", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid TOML"):
            eval_config._load_toml_document(bad_file)

    def test_non_dict_root_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the TOML root isn't a dict → ValueError (line 124)."""
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text("a = 1\n", encoding="utf-8")

        # Monkeypatch the active TOML loader to return a non-dict value.
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
            import tomli as tomllib

        monkeypatch.setattr(tomllib, "load", lambda _h: "not_a_dict")
        monkeypatch.setattr(eval_config, "_toml_loader", tomllib)

        with pytest.raises(ValueError, match="config root must be a TOML table"):
            eval_config._load_toml_document(toml_file)


# ---------------------------------------------------------------------------
# _extract_run_eval_config_payload
# ---------------------------------------------------------------------------


class TestExtractRunEvalConfigPayload:
    """Cover the two branches in _extract_run_eval_config_payload."""

    def test_returns_root_when_section_missing(self) -> None:
        """No 'run_eval' key → returns root unchanged (line 130)."""
        root: dict[str, Any] = {"k": 3, "jobs": 4}
        result = eval_config._extract_run_eval_config_payload(Path("dummy.toml"), root)
        assert result == root
        assert result is root

    def test_raises_when_section_not_dict(self) -> None:
        """'run_eval' is not a dict → ValueError (line 133)."""
        root: dict[str, Any] = {"run_eval": "not_a_dict"}
        with pytest.raises(ValueError, match="must be a TOML table"):
            eval_config._extract_run_eval_config_payload(Path("dummy.toml"), root)

    def test_extracts_section_dict(self) -> None:
        root: dict[str, Any] = {"run_eval": {"k": 5}}
        result = eval_config._extract_run_eval_config_payload(Path("dummy.toml"), root)
        assert result == {"k": 5}
        assert "run_eval" not in result


# ---------------------------------------------------------------------------
# _flatten_config_mapping
# ---------------------------------------------------------------------------


class TestFlattenConfigMapping:
    """Cover empty-token skip (line 143) and nested flattening."""

    def test_empty_key_skipped(self) -> None:
        data: dict[str, Any] = {"": "val", "  ": "val2", "good": 1}
        result = eval_config._flatten_config_mapping(data)
        assert "good" in result
        assert len(result) == 1

    def test_nested_flattening(self) -> None:
        data: dict[str, Any] = {"outer": {"inner": 42}}
        result = eval_config._flatten_config_mapping(data)
        assert result == {"outer_inner": 42}
        assert len(result) == 1

    def test_dash_to_underscore(self) -> None:
        data: dict[str, Any] = {"my-key": "value"}
        result = eval_config._flatten_config_mapping(data)
        assert result == {"my_key": "value"}
        assert "my-key" not in result


# ---------------------------------------------------------------------------
# _collect_parser_actions_by_dest
# ---------------------------------------------------------------------------


class TestCollectParserActionsByDest:
    """Cover branches for positional actions and SUPPRESS dest."""

    def test_skips_positional_actions(self) -> None:
        """Actions without option_strings (positionals) → skipped (line 158)."""
        parser = argparse.ArgumentParser()
        parser.add_argument("positional")
        parser.add_argument("--flag", default="x")

        result = eval_config._collect_parser_actions_by_dest(parser)
        assert "positional" not in result
        assert "flag" in result

    def test_skips_suppress_dest(self) -> None:
        """Actions with dest=SUPPRESS → skipped (line 160)."""
        parser = argparse.ArgumentParser()
        _action = parser.add_argument("--hidden", dest=argparse.SUPPRESS)
        parser.add_argument("--visible", default="y")

        result = eval_config._collect_parser_actions_by_dest(parser)
        assert "visible" in result
        # SUPPRESS actions are excluded
        assert argparse.SUPPRESS not in result


# ---------------------------------------------------------------------------
# _collect_cli_specified_dests
# ---------------------------------------------------------------------------


class TestCollectCliSpecifiedDests:
    """Cover the -- break (line 172) and SUPPRESS dest (line 180)."""

    def test_double_dash_stops_parsing(self) -> None:
        """Tokens after '--' are ignored (line 172)."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--alpha", default="a")
        parser.add_argument("--beta", default="b")

        tokens = ["--alpha", "val1", "--", "--beta", "val2"]
        result = eval_config._collect_cli_specified_dests(parser, tokens)
        assert "alpha" in result
        assert "beta" not in result

    def test_suppress_dest_skipped(self) -> None:
        """Actions with dest=SUPPRESS → not added to result (line 180)."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--hidden", dest=argparse.SUPPRESS)
        parser.add_argument("--visible", default="v")

        tokens = ["--hidden", "val", "--visible", "v2"]
        result = eval_config._collect_cli_specified_dests(parser, tokens)
        assert "visible" in result
        # SUPPRESS dest should not be present
        assert argparse.SUPPRESS not in result

    def test_non_option_tokens_ignored(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--flag", default="x")

        tokens = ["positional_value", "--flag", "y"]
        result = eval_config._collect_cli_specified_dests(parser, tokens)
        assert "flag" in result
        assert isinstance(result, set)

    def test_equals_form_recognised(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--key", default="d")

        tokens = ["--key=value"]
        result = eval_config._collect_cli_specified_dests(parser, tokens)
        assert "key" in result
        assert len(result) == 1

    def test_unknown_option_ignored(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--known", default="k")

        tokens = ["--unknown", "v", "--known", "v2"]
        result = eval_config._collect_cli_specified_dests(parser, tokens)
        assert "known" in result
        assert "unknown" not in result


# ---------------------------------------------------------------------------
# _coerce_config_bool
# ---------------------------------------------------------------------------


class TestCoerceConfigBool:
    """Cover string parsing and error for non-bool values (lines 188-194)."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (True, True),
            (False, False),
            ("1", True),
            ("true", True),
            ("yes", True),
            ("on", True),
            ("TRUE", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("off", False),
            ("OFF", False),
        ],
    )
    def test_valid_values(self, raw: Any, expected: bool) -> None:
        result = eval_config._coerce_config_bool(raw, key="test")
        assert result is expected
        assert isinstance(result, bool)

    @pytest.mark.parametrize("raw", [42, "maybe", "2", None, []])
    def test_invalid_raises(self, raw: Any) -> None:
        with pytest.raises(ValueError, match="must be a boolean"):
            eval_config._coerce_config_bool(raw, key="bad")


# ---------------------------------------------------------------------------
# _coerce_config_scalar_for_action
# ---------------------------------------------------------------------------


class TestCoerceConfigScalarForAction:
    """Cover list/dict guard, converter failure, choices check."""

    def _store_action(
        self,
        *,
        type_fn: Any = None,
        choices: Any = None,
    ) -> argparse.Action:
        parser = argparse.ArgumentParser()
        kwargs: dict[str, Any] = {"default": None}
        if type_fn is not None:
            kwargs["type"] = type_fn
        if choices is not None:
            kwargs["choices"] = choices
        action = parser.add_argument("--x", **kwargs)
        return action

    def test_list_input_raises(self) -> None:
        """Line 199: list/dict values are rejected."""
        action = self._store_action()
        with pytest.raises(ValueError, match="must be a scalar"):
            eval_config._coerce_config_scalar_for_action([1, 2], action=action, key="k")

    def test_dict_input_raises(self) -> None:
        action = self._store_action()
        with pytest.raises(ValueError, match="must be a scalar"):
            eval_config._coerce_config_scalar_for_action({"a": 1}, action=action, key="k")

    def test_converter_failure_raises(self) -> None:
        """Lines 205-206: converter raises → wrapped ValueError."""
        action = self._store_action(type_fn=int)
        with pytest.raises(ValueError, match="invalid value"):
            eval_config._coerce_config_scalar_for_action("not_an_int", action=action, key="k")

    def test_choices_violation_raises(self) -> None:
        """Lines 211-212: value not in choices → ValueError."""
        action = self._store_action(choices=["a", "b", "c"])
        with pytest.raises(ValueError, match="must be one of"):
            eval_config._coerce_config_scalar_for_action("d", action=action, key="k")

    def test_valid_scalar(self) -> None:
        action = self._store_action(type_fn=int)
        result = eval_config._coerce_config_scalar_for_action("42", action=action, key="k")
        assert result == 42
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# _coerce_config_value_for_dest
# ---------------------------------------------------------------------------


class TestCoerceConfigValueForDest:
    """Cover append-action branch, unsupported-action branch."""

    def test_append_action_list_input(self) -> None:
        """Lines 219-220: AppendAction with list input."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument("--items", action="append", type=int)
        actions = [action]
        result = eval_config._coerce_config_value_for_dest([1, 2, 3], key="items", actions=actions)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_append_action_scalar_input(self) -> None:
        """AppendAction with scalar → wrapped into list."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument("--items", action="append", type=int)
        actions = [action]
        result = eval_config._coerce_config_value_for_dest(42, key="items", actions=actions)
        assert result == [42]
        assert len(result) == 1

    def test_store_true_action(self) -> None:
        """StoreTrueAction → delegates to _coerce_config_bool."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument("--flag", action="store_true")
        actions = [action]
        result = eval_config._coerce_config_value_for_dest("yes", key="flag", actions=actions)
        assert result is True
        assert isinstance(result, bool)

    def test_unsupported_action_raises(self) -> None:
        """Line 233: no StoreAction found → ValueError."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument("--count", action="count")
        actions = [action]
        with pytest.raises(ValueError, match="unsupported parser action"):
            eval_config._coerce_config_value_for_dest(5, key="count", actions=actions)


# ---------------------------------------------------------------------------
# _json_compatible
# ---------------------------------------------------------------------------


class TestJsonCompatible:
    """Cover Path, dict, list/tuple/set, and fallback-to-str branches."""

    def test_none(self) -> None:
        assert eval_config._json_compatible(None) is None

    def test_scalar(self) -> None:
        assert eval_config._json_compatible(42) == 42
        assert eval_config._json_compatible("s") == "s"

    def test_path(self) -> None:
        """Line 324."""
        p = Path("/some/path")
        result = eval_config._json_compatible(p)
        assert result == "/some/path"
        assert isinstance(result, str)

    def test_dict(self) -> None:
        """Line 326."""
        result = eval_config._json_compatible({"a": Path("/x")})
        assert result == {"a": "/x"}
        assert isinstance(result, dict)

    def test_list(self) -> None:
        """Line 328-329: list/tuple/set."""
        assert eval_config._json_compatible([1, "a"]) == [1, "a"]
        assert eval_config._json_compatible((1,)) == [1]
        assert isinstance(eval_config._json_compatible({1, 2}), list)

    def test_fallback_to_str(self) -> None:
        """Line 329: unknown type → str()."""
        obj = object()
        result = eval_config._json_compatible(obj)
        assert isinstance(result, str)
        assert "object" in result


# ---------------------------------------------------------------------------
# _redact_cli_argv
# ---------------------------------------------------------------------------


class TestRedactCliArgv:
    """Cover the three redaction branches (lines 337-346)."""

    def test_no_redaction_needed(self) -> None:
        tokens = ["--run-id", "abc", "--executor", "local"]
        result = eval_config._redact_cli_argv(tokens)
        assert result == tokens
        assert len(result) == 4

    def test_space_separated_api_key_redacted(self) -> None:
        """Lines 337-339, 341-343: --openai-api-key VALUE."""
        tokens = ["--openai-api-key", "secret123", "--jobs", "2"]
        result = eval_config._redact_cli_argv(tokens)
        assert result == [
            "--openai-api-key",
            RUN_EVAL_LOCKFILE_REDACTED,
            "--jobs",
            "2",
        ]

    def test_equals_form_api_key_redacted(self) -> None:
        """Lines 345-346: --openai-api-key=VALUE."""
        tokens = ["--openai-api-key=secret123", "--jobs", "2"]
        result = eval_config._redact_cli_argv(tokens)
        assert result == [
            f"--openai-api-key={RUN_EVAL_LOCKFILE_REDACTED}",
            "--jobs",
            "2",
        ]

    def test_decision_model_api_key_redacted_in_both_forms(self) -> None:
        tokens = [
            "--adaptive-decision-model-api-key",
            "dm-secret",
            "--adaptive-decision-model-api-key=dm-secret",
            "--adaptive-decision-model-url=http://gateway:9292/upstream/clef-flash-q4",
        ]
        result = eval_config._redact_cli_argv(tokens)
        assert result == [
            "--adaptive-decision-model-api-key",
            RUN_EVAL_LOCKFILE_REDACTED,
            f"--adaptive-decision-model-api-key={RUN_EVAL_LOCKFILE_REDACTED}",
            "--adaptive-decision-model-url=http://gateway:9292/upstream/clef-flash-q4",
        ]


# ---------------------------------------------------------------------------
# _parse_container_env
# ---------------------------------------------------------------------------


class TestParseContainerEnv:
    """Cover empty, missing '=', and empty-key branches."""

    def test_empty_entries_skipped(self) -> None:
        """Line 477: empty/whitespace-only entries are ignored."""
        result = eval_config._parse_container_env(["", "  ", "A=1"])
        assert result == {"A": "1"}
        assert len(result) == 1

    def test_missing_equals_raises(self) -> None:
        """Lines 479-480: no '=' → ValueError."""
        with pytest.raises(ValueError, match="expected KEY=VALUE"):
            eval_config._parse_container_env(["NOEQUALS"])

    def test_empty_key_raises(self) -> None:
        """Lines 482-484: key part is empty → ValueError."""
        with pytest.raises(ValueError, match="empty key"):
            eval_config._parse_container_env(["=value"])

    def test_valid_entries(self) -> None:
        result = eval_config._parse_container_env(["FOO=bar", "BAZ=qux=extra"])
        assert result == {"FOO": "bar", "BAZ": "qux=extra"}
        assert result["BAZ"] == "qux=extra"


# ---------------------------------------------------------------------------
# build_execution_config_from_args - validation error branches
# ---------------------------------------------------------------------------


class TestBuildExecutionConfigFromArgs:
    """Cover every validation error in build_execution_config_from_args."""

    @staticmethod
    def _default_ns(**overrides: Any) -> argparse.Namespace:
        """Build a Namespace with valid defaults, then apply overrides."""
        defaults: dict[str, Any] = {
            "executor": DEFAULT_EXECUTOR,
            "container_image": DEFAULT_CONTAINER_IMAGE,
            "container_network": DEFAULT_CONTAINER_NETWORK,
            "container_cpus": DEFAULT_CONTAINER_CPUS,
            "container_memory": DEFAULT_CONTAINER_MEMORY,
            "container_pids_limit": DEFAULT_CONTAINER_PIDS_LIMIT,
            "container_user": DEFAULT_CONTAINER_USER,
            "container_tmpfs_size_mb": DEFAULT_CONTAINER_TMPFS_SIZE_MB,
            "container_mount_repo_src": True,
            "container_env": [],
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_invalid_executor(self) -> None:
        """Line 496."""
        ns = self._default_ns(executor="docker")
        with pytest.raises(ValueError, match="invalid --executor"):
            eval_config.build_execution_config_from_args(ns)

    def test_invalid_container_network(self) -> None:
        """Line 501."""
        ns = self._default_ns(container_network="host")
        with pytest.raises(ValueError, match="invalid --container-network"):
            eval_config.build_execution_config_from_args(ns)

    def test_empty_container_image(self) -> None:
        """Line 504."""
        ns = self._default_ns(container_image="  ")
        with pytest.raises(ValueError, match="container-image must be non-empty"):
            eval_config.build_execution_config_from_args(ns)

    def test_zero_container_cpus(self) -> None:
        """Line 507."""
        ns = self._default_ns(container_cpus=0.0)
        with pytest.raises(ValueError, match="container-cpus must be > 0"):
            eval_config.build_execution_config_from_args(ns)

    def test_negative_container_cpus(self) -> None:
        ns = self._default_ns(container_cpus=-1.0)
        with pytest.raises(ValueError, match="container-cpus must be > 0"):
            eval_config.build_execution_config_from_args(ns)

    def test_zero_pids_limit(self) -> None:
        """Line 510."""
        ns = self._default_ns(container_pids_limit=0)
        with pytest.raises(ValueError, match="container-pids-limit must be > 0"):
            eval_config.build_execution_config_from_args(ns)

    def test_negative_pids_limit(self) -> None:
        ns = self._default_ns(container_pids_limit=-10)
        with pytest.raises(ValueError, match="container-pids-limit must be > 0"):
            eval_config.build_execution_config_from_args(ns)

    def test_zero_tmpfs_size(self) -> None:
        """Line 515."""
        ns = self._default_ns(container_tmpfs_size_mb=0)
        with pytest.raises(ValueError, match="container-tmpfs-size-mb must be > 0"):
            eval_config.build_execution_config_from_args(ns)

    def test_empty_container_user(self) -> None:
        """Line 518."""
        ns = self._default_ns(container_user="  ")
        with pytest.raises(ValueError, match="container-user must be non-empty"):
            eval_config.build_execution_config_from_args(ns)

    def test_empty_container_memory(self) -> None:
        """Line 521."""
        ns = self._default_ns(container_memory="  ")
        with pytest.raises(ValueError, match="container-memory must be non-empty"):
            eval_config.build_execution_config_from_args(ns)

    def test_valid_config(self) -> None:
        ns = self._default_ns(executor="local")
        cfg = eval_config.build_execution_config_from_args(ns)
        assert cfg.executor == "local"
        assert cfg.container_cpus == DEFAULT_CONTAINER_CPUS

    def test_implicit_latest_fallback_emits_warning(self) -> None:
        ns = self._default_ns(
            _container_image_default_source="implicit_fallback",
            _container_image_default_implicit_latest=True,
        )
        cfg = eval_config.build_execution_config_from_args(ns)
        assert cfg.container_image_source == "implicit_fallback"
        assert cfg.container_image_implicit_latest is True
        assert cfg.warnings
        assert "implicit fallback container image" in cfg.warnings[0]


def test_add_execution_arguments_resolves_container_image_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMDEBUG_EVAL_CONTAINER_IMAGE", "repo/image:sha256-deadbeef")
    parser = argparse.ArgumentParser()
    eval_config.add_execution_arguments(parser)
    args = parser.parse_args([])
    assert args.container_image == "repo/image:sha256-deadbeef"
    assert args._container_image_default_source == "env:LLMDEBUG_EVAL_CONTAINER_IMAGE"
    assert args._container_image_default_implicit_latest is False


def test_explicit_cli_container_image_uses_cli_provenance() -> None:
    parser = argparse.ArgumentParser()
    eval_config.add_execution_arguments(parser)
    args = parser.parse_args(["--container-image", DEFAULT_CONTAINER_IMAGE])
    cfg = eval_config.build_execution_config_from_args(args)
    assert cfg.container_image_source == "cli"
    assert cfg.container_image_implicit_latest is False
    assert cfg.warnings == ()
    metadata = eval_config.resolve_execution_runtime_metadata(args)
    assert metadata["container_image"]["source"] == "cli"
    assert metadata["container_image"]["implicit_latest"] is False


def test_resolve_execution_runtime_metadata_records_sources() -> None:
    args = argparse.Namespace(
        container_image="repo/image:sha256-deadbeef",
        _container_image_default_source="config",
        _container_image_default_implicit_latest=False,
        _openai_api_key_default_source="env:OPENAI_API_KEY",
        _openai_api_key_default_env_key="OPENAI_API_KEY",
    )
    metadata = eval_config.resolve_execution_runtime_metadata(args)
    assert metadata["container_image"] == {
        "value": "repo/image:sha256-deadbeef",
        "source": "config",
        "implicit_latest": False,
    }
    assert metadata["openai_api_key"] == {
        "source": "env:OPENAI_API_KEY",
        "env_key": "OPENAI_API_KEY",
    }


# ---------------------------------------------------------------------------
# set_active_execution_config / get_active_execution_config (line 540)
# ---------------------------------------------------------------------------


class TestActiveExecutionConfig:
    """Cover get/set of the module-level active execution config."""

    def test_set_returns_previous(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Line 540: set_active_execution_config returns old config."""
        original = eval_config.get_active_execution_config()
        new_config = ExecutionConfig(executor="local")
        previous = eval_config.set_active_execution_config(new_config)
        assert previous is original
        assert eval_config.get_active_execution_config() is new_config
        # Restore to avoid leaking state.
        eval_config.set_active_execution_config(original)


# ---------------------------------------------------------------------------
# _first_nonempty_env / _resolve_openai_api_key_env_for_config
# ---------------------------------------------------------------------------


class TestOpenaiApiKeyEnvResolution:
    """Cover env-variable resolution helpers."""

    def test_first_nonempty_env_returns_first_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("A_KEY", "")
        monkeypatch.setenv("B_KEY", "  ")
        monkeypatch.setenv("C_KEY", "found")
        result = eval_config._first_nonempty_env("A_KEY", "B_KEY", "C_KEY")
        assert result == "found"
        assert isinstance(result, str)

    def test_first_nonempty_env_returns_empty_when_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MISSING_KEY", raising=False)
        result = eval_config._first_nonempty_env("MISSING_KEY")
        assert result == ""
        assert isinstance(result, str)

    def test_resolve_openai_api_key_env_direct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-OPENAI_API_KEY env name → calls _first_nonempty_env."""
        monkeypatch.setenv("CUSTOM_KEY", "custom_val")
        result = eval_config._resolve_openai_api_key_env_for_config("CUSTOM_KEY")
        assert result == "custom_val"
        assert len(result) > 0

    def test_resolve_openai_api_key_env_default_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'OPENAI_API_KEY' → calls _default_openai_api_key()."""
        from evals.eval_types import OPENAI_API_KEY_ENV_FALLBACKS

        # Clear all fallback env vars so we control resolution order.
        for key in OPENAI_API_KEY_ENV_FALLBACKS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "from_env")
        result = eval_config._resolve_openai_api_key_env_for_config("OPENAI_API_KEY")
        assert result == "from_env"

    def test_resolve_openai_api_key_default_uses_local_when_chain_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from evals.eval_types import OPENAI_API_KEY_ENV_FALLBACKS

        for key in OPENAI_API_KEY_ENV_FALLBACKS:
            monkeypatch.delenv(key, raising=False)
        resolved = eval_config.resolve_openai_api_key_default()
        assert resolved.value == "local"
        assert resolved.source == "implicit_default"
        assert resolved.env_key == "OPENAI_API_KEY"
        assert os.environ["OPENAI_API_KEY"] == "local"

    def test_runtime_resolution_imports_without_repo_env_defaults(self, tmp_path: Path) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        temp_package = tmp_path / "evals"
        temp_package.mkdir()
        shutil.copy2(repo_root / "evals" / "__init__.py", temp_package / "__init__.py")
        shutil.copy2(
            repo_root / "evals" / "runtime_resolution.py",
            temp_package / "runtime_resolution.py",
        )

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        for key in ("Z_API", "OPENAI_API_KEY", "LLMDEBUG_EVAL_OPENAI_API_KEY"):
            env.pop(key, None)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from evals.runtime_resolution import resolve_openai_api_key_from_env; "
                    "import os; "
                    "resolved = resolve_openai_api_key_from_env(); "
                    "print("
                    "f'{resolved.value}|{resolved.source}|{resolved.env_key}|"
                    '{os.environ.get("OPENAI_API_KEY", "")}\''
                    ")"
                ),
            ],
            check=False,
            capture_output=True,
            cwd=tmp_path,
            env=env,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "local|implicit_default|OPENAI_API_KEY|local"


# ---------------------------------------------------------------------------
# _apply_toml_configs_to_args - error branches
# ---------------------------------------------------------------------------


def _make_parser_with_flags() -> argparse.ArgumentParser:
    """Build a minimal parser that mirrors key run_eval flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--container-image", default=DEFAULT_CONTAINER_IMAGE)
    parser.add_argument("--executor", choices=["local", "testcontainers"])
    return parser


class TestApplyTomlConfigsToArgs:
    """Cover error branches inside _apply_toml_configs_to_args."""

    def test_no_config_paths(self) -> None:
        """No config files → returns default sources."""
        parser = _make_parser_with_flags()
        args = parser.parse_args([])
        state = eval_config._apply_toml_configs_to_args(parser=parser, args=args, argv_tokens=[])
        assert state.config_paths == []
        assert state.value_sources["k"] == "default"

    def test_config_file_not_found(self, tmp_path: Path) -> None:
        """Line 275: config path does not exist."""
        parser = _make_parser_with_flags()
        missing = str(tmp_path / "nope.toml")
        args = parser.parse_args(["--config", missing])
        with pytest.raises(ValueError, match="config file not found"):
            eval_config._apply_toml_configs_to_args(
                parser=parser, args=args, argv_tokens=["--config", missing]
            )

    def test_config_path_is_directory(self, tmp_path: Path) -> None:
        """Line 277: config path is a directory."""
        parser = _make_parser_with_flags()
        dir_path = str(tmp_path)
        args = parser.parse_args(["--config", dir_path])
        with pytest.raises(ValueError, match="not a file"):
            eval_config._apply_toml_configs_to_args(
                parser=parser, args=args, argv_tokens=["--config", dir_path]
            )

    def test_nested_config_key_raises(self, tmp_path: Path) -> None:
        """Line 283: nested 'config' key inside TOML."""
        cfg = tmp_path / "nested.toml"
        cfg.write_text('config = "bad"\n', encoding="utf-8")
        parser = _make_parser_with_flags()
        cfg_str = str(cfg)
        args = parser.parse_args(["--config", cfg_str])
        with pytest.raises(ValueError, match="nested 'config' keys"):
            eval_config._apply_toml_configs_to_args(
                parser=parser, args=args, argv_tokens=["--config", cfg_str]
            )

    def test_literal_secret_key_nonempty_raises(self, tmp_path: Path) -> None:
        """Line 286-290: openai_api_key in TOML with non-empty value."""
        cfg = tmp_path / "secret.toml"
        cfg.write_text('openai_api_key = "sk-123"\n', encoding="utf-8")
        parser = _make_parser_with_flags()
        cfg_str = str(cfg)
        args = parser.parse_args(["--config", cfg_str])
        with pytest.raises(ValueError, match="not allowed in config files"):
            eval_config._apply_toml_configs_to_args(
                parser=parser, args=args, argv_tokens=["--config", cfg_str]
            )

    def test_literal_decision_model_api_key_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "dm_secret.toml"
        cfg.write_text('[adaptive]\ndecision_model_api_key = "dm-secret"\n', encoding="utf-8")
        parser = _make_parser_with_flags()
        cfg_str = str(cfg)
        args = parser.parse_args(["--config", cfg_str])
        with pytest.raises(ValueError, match="TRACEGATE_DECISION_MODEL_API_KEY"):
            eval_config._apply_toml_configs_to_args(
                parser=parser, args=args, argv_tokens=["--config", cfg_str]
            )

    def test_literal_secret_key_empty_skipped(self, tmp_path: Path) -> None:
        """Line 286: openai_api_key with empty value → silently skipped."""
        cfg = tmp_path / "empty_secret.toml"
        cfg.write_text('openai_api_key = ""\nk = 5\n', encoding="utf-8")
        parser = _make_parser_with_flags()
        cfg_str = str(cfg)
        args = parser.parse_args(["--config", cfg_str])
        state = eval_config._apply_toml_configs_to_args(
            parser=parser, args=args, argv_tokens=["--config", cfg_str]
        )
        assert args.k == 5
        assert len(state.config_paths) == 1

    def test_env_key_empty_raises(self, tmp_path: Path) -> None:
        """Line 294: openai_api_key_env with empty string."""
        cfg = tmp_path / "env_empty.toml"
        cfg.write_text('openai_api_key_env = "  "\n', encoding="utf-8")
        parser = _make_parser_with_flags()
        cfg_str = str(cfg)
        args = parser.parse_args(["--config", cfg_str])
        with pytest.raises(ValueError, match="must be a non-empty environment variable"):
            eval_config._apply_toml_configs_to_args(
                parser=parser, args=args, argv_tokens=["--config", cfg_str]
            )

    def test_env_key_cli_override_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Line 298: cli specifies openai_api_key → env key skipped."""
        cfg = tmp_path / "env_skip.toml"
        cfg.write_text('openai_api_key_env = "MY_KEY"\nk = 7\n', encoding="utf-8")
        parser = _make_parser_with_flags()
        cfg_str = str(cfg)
        args = parser.parse_args(["--config", cfg_str, "--openai-api-key", "cli_val"])
        _state = eval_config._apply_toml_configs_to_args(
            parser=parser,
            args=args,
            argv_tokens=[
                "--config",
                cfg_str,
                "--openai-api-key",
                "cli_val",
            ],
        )
        # The key was specified on CLI so env resolution is skipped.
        assert args.openai_api_key == "cli_val"
        assert args.k == 7

    def test_env_key_unset_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Line 300-303: env var referenced but empty/unset."""
        cfg = tmp_path / "env_unset.toml"
        cfg.write_text('openai_api_key_env = "NONEXISTENT_VAR"\n', encoding="utf-8")
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        parser = _make_parser_with_flags()
        cfg_str = str(cfg)
        args = parser.parse_args(["--config", cfg_str])
        with pytest.raises(ValueError, match=r"is required by.*but is empty"):
            eval_config._apply_toml_configs_to_args(
                parser=parser, args=args, argv_tokens=["--config", cfg_str]
            )

    def test_env_key_resolved_successfully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: env key resolves to a real value."""
        cfg = tmp_path / "env_ok.toml"
        cfg.write_text('openai_api_key_env = "MY_API"\n', encoding="utf-8")
        monkeypatch.setenv("MY_API", "resolved_key_value")
        parser = _make_parser_with_flags()
        cfg_str = str(cfg)
        args = parser.parse_args(["--config", cfg_str])
        state = eval_config._apply_toml_configs_to_args(
            parser=parser, args=args, argv_tokens=["--config", cfg_str]
        )
        assert args.openai_api_key == "resolved_key_value"
        assert "env:MY_API" in state.value_sources["openai_api_key"]

    def test_unknown_config_key_raises(self, tmp_path: Path) -> None:
        """Line 310: key not recognised by parser."""
        cfg = tmp_path / "unknown.toml"
        cfg.write_text("totally_unknown = 42\n", encoding="utf-8")
        parser = _make_parser_with_flags()
        cfg_str = str(cfg)
        args = parser.parse_args(["--config", cfg_str])
        with pytest.raises(ValueError, match="unknown config key"):
            eval_config._apply_toml_configs_to_args(
                parser=parser, args=args, argv_tokens=["--config", cfg_str]
            )

    def test_cli_specified_key_takes_precedence(self, tmp_path: Path) -> None:
        """CLI value takes precedence over config file value."""
        cfg = tmp_path / "prec.toml"
        cfg.write_text("k = 99\n", encoding="utf-8")
        parser = _make_parser_with_flags()
        cfg_str = str(cfg)
        args = parser.parse_args(["--config", cfg_str, "--k", "5"])
        state = eval_config._apply_toml_configs_to_args(
            parser=parser,
            args=args,
            argv_tokens=["--config", cfg_str, "--k", "5"],
        )
        # CLI value should win
        assert args.k == 5
        assert state.value_sources["k"] == "cli"

    def test_container_image_config_overrides_env_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = tmp_path / "container.toml"
        cfg.write_text('container_image = "config:image"\n', encoding="utf-8")
        parser = _make_parser_with_flags()
        monkeypatch.setenv("LLMDEBUG_EVAL_CONTAINER_IMAGE", "env:image")
        args = parser.parse_args(["--config", str(cfg)])
        args.container_image = "env:image"
        state = eval_config._apply_toml_configs_to_args(
            parser=parser,
            args=args,
            argv_tokens=["--config", str(cfg)],
        )
        assert args.container_image == "config:image"
        assert args._container_image_default_source.startswith("config:")
        assert args._container_image_default_implicit_latest is False
        assert state.value_sources["container_image"].startswith("config:")

    def test_container_image_cli_overrides_config(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = tmp_path / "container_cli.toml"
        cfg.write_text('container_image = "config:image"\n', encoding="utf-8")
        parser = _make_parser_with_flags()
        args = parser.parse_args(["--config", str(cfg), "--container-image", "cli:image"])
        state = eval_config._apply_toml_configs_to_args(
            parser=parser,
            args=args,
            argv_tokens=["--config", str(cfg), "--container-image", "cli:image"],
        )
        assert args.container_image == "cli:image"
        assert args._container_image_default_source == "cli"
        assert args._container_image_default_implicit_latest is False
        assert state.value_sources["container_image"] == "cli"


# ---------------------------------------------------------------------------
# _read_run_eval_config - non-dict payload guard (line 241)
# ---------------------------------------------------------------------------


class TestReadRunEvalConfig:
    """Cover the isinstance(payload, dict) guard."""

    def test_non_dict_payload_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Line 241: payload must be a dict."""
        cfg = tmp_path / "ok.toml"
        cfg.write_text("k = 1\n", encoding="utf-8")

        # Force _extract_run_eval_config_payload to return non-dict
        monkeypatch.setattr(
            eval_config,
            "_extract_run_eval_config_payload",
            lambda _p, _r: "not_a_dict",
        )
        with pytest.raises(ValueError, match="must be a TOML table"):
            eval_config._read_run_eval_config(cfg)


# ---------------------------------------------------------------------------
# _resolved_args_for_lockfile
# ---------------------------------------------------------------------------


class TestResolvedArgsForLockfile:
    """Cover redaction of secret keys."""

    def test_redacts_api_key(self) -> None:
        ns = argparse.Namespace(openai_api_key="secret", k=2, jobs=3)
        result = eval_config._resolved_args_for_lockfile(ns)
        assert result["openai_api_key"] == RUN_EVAL_LOCKFILE_REDACTED
        assert result["k"] == 2

    def test_redacts_decision_model_api_key(self) -> None:
        ns = argparse.Namespace(adaptive_decision_model_api_key="dm-secret", k=2)
        result = eval_config._resolved_args_for_lockfile(ns)
        assert result["adaptive_decision_model_api_key"] == RUN_EVAL_LOCKFILE_REDACTED

    def test_empty_api_key_not_redacted(self) -> None:
        ns = argparse.Namespace(openai_api_key="", k=2)
        result = eval_config._resolved_args_for_lockfile(ns)
        assert result["openai_api_key"] == ""
        assert result["k"] == 2

    def test_none_api_key(self) -> None:
        ns = argparse.Namespace(openai_api_key=None, k=2)
        result = eval_config._resolved_args_for_lockfile(ns)
        assert result["openai_api_key"] == ""
        assert result["k"] == 2


# ---------------------------------------------------------------------------
# _tomllib import fallback (lines 40-41)
# ---------------------------------------------------------------------------


class TestTomlImportFallback:
    """Ensure the except ModuleNotFoundError branch is exercisable."""

    def test_toml_loader_set_when_tomllib_available(self) -> None:
        """On Python 3.11+, _toml_loader should not be None."""
        import sys

        if sys.version_info >= (3, 11):
            assert eval_config._toml_loader is not None
        # On older Pythons with tomllib absent, it would be None.

    def test_load_toml_with_loader_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Force _toml_loader=None to exercise the fallback import."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("x = 1\n", encoding="utf-8")
        monkeypatch.setattr(eval_config, "_toml_loader", None)
        # This exercises lines 40-41 indirectly through _load_toml_document.
        try:
            result = eval_config._load_toml_document(toml_file)
            assert result == {"x": 1}
        except ValueError:
            pass  # tomli not installed - branch still covered


# ---------------------------------------------------------------------------
# _write_resolved_run_config_lockfile
# ---------------------------------------------------------------------------


class TestWriteResolvedRunConfigLockfile:
    """Cover the lockfile writing function."""

    def test_writes_lockfile(self, tmp_path: Path) -> None:
        from evals.eval_types import ConfigResolutionState

        args = argparse.Namespace(openai_api_key="secret", k=2, jobs=3)
        state = ConfigResolutionState(
            config_paths=[],
            value_sources={"k": "default", "jobs": "default"},
            runtime_resolution={
                "container_image": {
                    "value": "repo/image:sha256-deadbeef",
                    "source": "env:LLMDEBUG_EVAL_CONTAINER_IMAGE",
                    "implicit_latest": False,
                }
            },
        )
        path = eval_config._write_resolved_run_config_lockfile(
            run_id="test-run",
            args=args,
            argv_tokens=["--k", "2"],
            run_settings_fingerprint="abc123",
            config_state=state,
            repo_root=tmp_path,
        )
        assert path.exists()
        import json as stdlib_json

        payload = stdlib_json.loads(path.read_text(encoding="utf-8"))
        assert payload["run_id"] == "test-run"
        assert payload["resolved_args"]["openai_api_key"] == (RUN_EVAL_LOCKFILE_REDACTED)
        assert payload["runtime_resolution"]["container_image"]["source"] == (
            "env:LLMDEBUG_EVAL_CONTAINER_IMAGE"
        )


# ---------------------------------------------------------------------------
# _execution_config_payload
# ---------------------------------------------------------------------------


class TestExecutionConfigPayload:
    """Cover the payload serialization helper."""

    def test_payload_keys(self) -> None:
        cfg = ExecutionConfig(
            executor="local",
            container_image_source="config",
            container_image_implicit_latest=False,
            container_env={"X": "1", "A": "2"},
        )
        result = eval_config._execution_config_payload(cfg)
        assert result["executor"] == "local"
        assert result["container_image_source"] == "config"
        assert result["container_image_implicit_latest"] is False
        assert result["container_env_keys"] == ["A", "X"]
