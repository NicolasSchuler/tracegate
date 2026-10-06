"""Config/args parsing, TOML loading, and config resolution for the eval harness.

Extracted from ``evals.run_eval`` to keep the main orchestrator lean.
Re-exported by ``evals.run_eval`` for backward compatibility.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import _json as json
from .decision_model.config import DECISION_MODEL_API_KEY_ENV
from .eval_types import (
    CONTAINER_NETWORK_MODES,
    DEFAULT_CONTAINER_CPUS,
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_CONTAINER_MEMORY,
    DEFAULT_CONTAINER_NETWORK,
    DEFAULT_CONTAINER_PIDS_LIMIT,
    DEFAULT_CONTAINER_TMPFS_SIZE_MB,
    DEFAULT_CONTAINER_USER,
    DEFAULT_EXECUTOR,
    ONLINE_POLICY_ACTIONS,
    RUN_EVAL_CONFIG_LOCK_SCHEMA_VERSION,
    RUN_EVAL_CONFIG_SECTION,
    RUN_EVAL_LOCKFILE_REDACTED,
    RUN_EVAL_LOCKFILE_REDACTED_KEYS,
    RUN_EVAL_SECRET_CONFIG_ENV_KEYS,
    RUN_EVAL_SECRET_CONFIG_LITERAL_KEYS,
    VALID_EXECUTORS,
    ConfigResolutionState,
    ExecutionConfig,
)
from .runtime_resolution import (
    ResolvedApiKey,
    first_nonempty_env,
    resolve_default_container_image,
    resolve_openai_api_key_from_env,
)

try:
    import tomllib as _toml_loader
except ModuleNotFoundError:
    _toml_loader = None

if TYPE_CHECKING:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[1]

# CLI flags whose values never reach lockfiles, and how to pass each secret instead of a config
# literal.
_SECRET_CLI_FLAGS = frozenset({"--openai-api-key", "--adaptive-decision-model-api-key"})
_SECRET_CONFIG_HINTS = {
    "openai_api_key": "Use 'openai_api_key_env' to reference an environment variable.",
    "adaptive_decision_model_api_key": (
        f"Set {DECISION_MODEL_API_KEY_ENV} in the environment instead."
    ),
}

# Module-level reference to the active execution config (set by set_active_execution_config).
_ACTIVE_EXECUTION_CONFIG = ExecutionConfig()


class _StoreWithExplicitFlagAction(argparse.Action):
    """Store a CLI value and mark that the destination was set explicitly."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_explicit", True)


def _first_nonempty_env(*keys: str) -> str:
    return first_nonempty_env(*keys).value


def _default_openai_api_key() -> str:
    return resolve_openai_api_key_default().value


def resolve_openai_api_key_default() -> ResolvedApiKey:
    """Resolve the default OpenAI-compatible API key with provenance."""
    return resolve_openai_api_key_from_env()


def resolve_execution_runtime_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Collect runtime-resolution provenance for lockfiles and diagnostics."""
    container_image_source, container_image_implicit_latest = (
        _effective_container_image_runtime_source(args)
    )
    return {
        "container_image": {
            "value": str(getattr(args, "container_image", "") or ""),
            "source": container_image_source,
            "implicit_latest": container_image_implicit_latest,
        },
        "openai_api_key": {
            "source": str(getattr(args, "_openai_api_key_default_source", "unknown")),
            "env_key": str(getattr(args, "_openai_api_key_default_env_key", "")),
        },
    }


def _effective_container_image_runtime_source(args: argparse.Namespace) -> tuple[str, bool]:
    source = str(getattr(args, "_container_image_default_source", "unknown"))
    implicit_latest = bool(getattr(args, "_container_image_default_implicit_latest", False))
    if bool(getattr(args, "_container_image_explicit", False)):
        return "cli", False
    return source, implicit_latest


def _sync_runtime_resolution_overrides(
    *,
    args: argparse.Namespace,
    value_sources: dict[str, str],
) -> None:
    container_image_source = value_sources.get("container_image", "default")
    if container_image_source == "cli":
        args._container_image_default_source = "cli"
        args._container_image_default_implicit_latest = False
        args._container_image_explicit = True
    elif container_image_source.startswith("config:"):
        args._container_image_default_source = container_image_source
        args._container_image_default_implicit_latest = False
        args._container_image_explicit = False

    openai_api_key_source = value_sources.get("openai_api_key", "default")
    if openai_api_key_source == "cli":
        args._openai_api_key_default_source = "cli"
        args._openai_api_key_default_env_key = ""
    elif openai_api_key_source.startswith("config:"):
        args._openai_api_key_default_source = openai_api_key_source
        env_fragment = openai_api_key_source.partition(" env:")[2]
        args._openai_api_key_default_env_key = env_fragment


def _resolve_openai_api_key_env_for_config(env_key: str) -> str:
    normalized = str(env_key).strip()
    if normalized == "OPENAI_API_KEY":
        return _default_openai_api_key()
    return _first_nonempty_env(normalized)


def _parse_adaptive_budget_policy_tiers(raw: Any) -> list[float]:
    """Parse and validate a comma-separated tier-multiplier string.

    Raises:
        ValueError: On empty, non-positive, duplicate, non-increasing
            values or when the mandatory 1.0 base tier is missing.
    """
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty")
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        raise ValueError("empty")
    tiers: list[float] = []
    for token in tokens:
        value = float(token)
        if value <= 0.0:
            raise ValueError("non_positive")
        tiers.append(value)
    if len(set(tiers)) != len(tiers):
        raise ValueError("duplicate")
    if any(tiers[idx] >= tiers[idx + 1] for idx in range(len(tiers) - 1)):
        raise ValueError("non_increasing")
    has_base = any(abs(value - 1.0) <= 1e-9 for value in tiers)
    if not has_base:
        raise ValueError("missing_base")
    return tiers


def _parse_adaptive_online_action_score_biases(raw: Any) -> dict[str, float]:
    """Parse comma-separated helper score-bias overrides into a full action map."""
    text = str(raw or "").strip()
    parsed: dict[str, float] = {str(action): 0.0 for action in ONLINE_POLICY_ACTIONS}
    if not text:
        return parsed
    seen: set[str] = set()
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        action_text, separator, bias_text = item.partition("=")
        if separator != "=":
            raise ValueError("malformed_pair")
        action = action_text.strip()
        if action not in ONLINE_POLICY_ACTIONS:
            raise ValueError("unknown_action")
        if action in seen:
            raise ValueError("duplicate")
        try:
            bias = float(bias_text.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_float") from exc
        parsed[action] = float(bias)
        seen.add(action)
    return parsed


def _load_toml_document(path: Path) -> dict[str, Any]:
    """Load and parse a TOML file, returning its root table.

    Raises:
        ValueError: If the file cannot be read or contains invalid TOML.
    """
    try:
        with path.open("rb") as handle:
            if _toml_loader is not None:
                raw = _toml_loader.load(handle)
            else:
                try:
                    import tomli  # type: ignore[import-not-found]
                except ModuleNotFoundError as exc:
                    raise ValueError(
                        "TOML parsing requires Python 3.11+ or an installed 'tomli' package."
                    ) from exc
                raw = tomli.load(handle)
    except OSError as exc:
        raise ValueError(f"cannot read config file {path}: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"invalid TOML in {path}: {type(exc).__name__}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config root must be a TOML table/object")
    return dict(raw)


def _extract_run_eval_config_payload(path: Path, root: dict[str, Any]) -> dict[str, Any]:
    if RUN_EVAL_CONFIG_SECTION not in root:
        return root
    section = root.get(RUN_EVAL_CONFIG_SECTION)
    if not isinstance(section, dict):
        raise ValueError(f"{path}: [{RUN_EVAL_CONFIG_SECTION}] must be a TOML table/object")
    return dict(section)


def _flatten_config_mapping(data: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict into underscore-joined keys."""
    flat: dict[str, Any] = {}
    for raw_key, raw_value in data.items():
        token = str(raw_key).strip().replace("-", "_").lower()
        if not token:
            continue
        full_key = f"{prefix}{token}" if not prefix else f"{prefix}_{token}"
        if isinstance(raw_value, dict):
            flat.update(_flatten_config_mapping(dict(raw_value), prefix=full_key))
            continue
        flat[full_key] = raw_value
    return flat


def _collect_parser_actions_by_dest(
    parser: argparse.ArgumentParser,
) -> dict[str, list[argparse.Action]]:
    actions: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        if action.dest == argparse.SUPPRESS:
            continue
        actions.setdefault(action.dest, []).append(action)
    return actions


def _collect_cli_specified_dests(
    parser: argparse.ArgumentParser, argv_tokens: list[str]
) -> set[str]:
    option_to_action = parser._option_string_actions
    out: set[str] = set()
    for token in argv_tokens:
        if token == "--":
            break
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        action = option_to_action.get(option)
        if action is None:
            continue
        if action.dest == argparse.SUPPRESS:
            continue
        out.add(action.dest)
    return out


def _coerce_config_bool(raw: Any, *, key: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"config key '{key}' must be a boolean")


def _coerce_config_scalar_for_action(raw: Any, *, action: argparse.Action, key: str) -> Any:
    if isinstance(raw, (list, dict)):
        raise ValueError(f"config key '{key}' must be a scalar value")
    value = raw
    converter = getattr(action, "type", None)
    if converter is not None:
        try:
            value = converter(raw)
        except Exception as exc:
            raise ValueError(
                f"config key '{key}' has invalid value {raw!r}: {type(exc).__name__}: {exc}"
            ) from exc
    choices = getattr(action, "choices", None)
    if choices is not None and value not in choices:
        allowed = ", ".join(str(item) for item in choices)
        raise ValueError(f"config key '{key}' must be one of: {allowed}")
    return value


def _coerce_config_value_for_dest(raw: Any, *, key: str, actions: list[argparse.Action]) -> Any:
    append_action = next((a for a in actions if isinstance(a, argparse._AppendAction)), None)
    if append_action is not None:
        raw_items = raw if isinstance(raw, list) else [raw]
        return [
            _coerce_config_scalar_for_action(item, action=append_action, key=key)
            for item in raw_items
        ]

    if any(
        isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
        for action in actions
    ):
        return _coerce_config_bool(raw, key=key)

    store_action = next((a for a in actions if isinstance(a, argparse._StoreAction)), None)
    if store_action is None:
        raise ValueError(f"config key '{key}' is mapped to unsupported parser action")
    return _coerce_config_scalar_for_action(raw, action=store_action, key=key)


def _read_run_eval_config(path: Path) -> dict[str, Any]:
    root = _load_toml_document(path)
    payload = _extract_run_eval_config_payload(path, root)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: run-eval config payload must be a TOML table/object")
    return _flatten_config_mapping(payload)


def _apply_toml_configs_to_args(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv_tokens: list[str],
) -> ConfigResolutionState:
    """Merge one or more TOML config files into *args*.

    CLI-specified values take precedence over config-file values.
    Returns a ``ConfigResolutionState`` tracking provenance per key.
    """
    config_paths_raw = getattr(args, "config", [])
    config_paths = [str(item).strip() for item in config_paths_raw if str(item).strip()]
    cli_specified = _collect_cli_specified_dests(parser, argv_tokens)
    value_sources = dict.fromkeys(vars(args), "default")
    for key in cli_specified:
        if key in value_sources:
            value_sources[key] = "cli"
    if not config_paths:
        _sync_runtime_resolution_overrides(args=args, value_sources=value_sources)
        return ConfigResolutionState(
            config_paths=[],
            value_sources=value_sources,
        )

    actions_by_dest = _collect_parser_actions_by_dest(parser)

    resolved_paths: list[str] = []
    for config_path_raw in config_paths:
        path = Path(config_path_raw).expanduser()
        if not path.exists():
            raise ValueError(f"config file not found: {path}")
        if not path.is_file():
            raise ValueError(f"config path is not a file: {path}")
        path = path.resolve()
        resolved_paths.append(str(path))
        payload = _read_run_eval_config(path)
        for key, raw_value in payload.items():
            if key == "config":
                raise ValueError(f"{path}: nested 'config' keys are not supported")
            if key in RUN_EVAL_SECRET_CONFIG_LITERAL_KEYS:
                if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
                    continue
                raise ValueError(
                    f"{path}: '{key}' is not allowed in config files. "
                    f"{_SECRET_CONFIG_HINTS.get(key, 'Pass it via the environment instead.')}"
                )
            if key in RUN_EVAL_SECRET_CONFIG_ENV_KEYS:
                env_key = str(raw_value).strip()
                if not env_key:
                    raise ValueError(
                        f"{path}: '{key}' must be a non-empty environment variable name"
                    )
                if "openai_api_key" in cli_specified:
                    continue
                env_value = _resolve_openai_api_key_env_for_config(env_key)
                if not env_value:
                    raise ValueError(
                        f"{path}: environment variable {env_key!r} is required by '{key}' but is empty or unset"
                    )
                args.openai_api_key = env_value
                value_sources["openai_api_key"] = f"config:{path} env:{env_key}"
                continue

            actions = actions_by_dest.get(key)
            if actions is None:
                raise ValueError(f"{path}: unknown config key '{key}'")
            if key in cli_specified:
                continue
            coerced = _coerce_config_value_for_dest(raw_value, key=key, actions=actions)
            setattr(args, key, coerced)
            value_sources[key] = f"config:{path}"

    _sync_runtime_resolution_overrides(args=args, value_sources=value_sources)
    return ConfigResolutionState(config_paths=resolved_paths, value_sources=value_sources)


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(raw) for raw in value]
    return str(value)


def _redact_cli_argv(argv_tokens: list[str]) -> list[str]:
    out: list[str] = []
    redact_next = False
    for token in argv_tokens:
        if redact_next:
            out.append(RUN_EVAL_LOCKFILE_REDACTED)
            redact_next = False
            continue
        if token in _SECRET_CLI_FLAGS:
            out.append(token)
            redact_next = True
            continue
        flag, separator, _value = token.partition("=")
        if separator and flag in _SECRET_CLI_FLAGS:
            out.append(f"{flag}={RUN_EVAL_LOCKFILE_REDACTED}")
            continue
        out.append(token)
    return out


def _resolved_args_for_lockfile(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, raw in vars(args).items():
        if key in RUN_EVAL_LOCKFILE_REDACTED_KEYS:
            text = str(raw).strip() if raw is not None else ""
            payload[key] = RUN_EVAL_LOCKFILE_REDACTED if text else ""
            continue
        payload[key] = _json_compatible(raw)
    return dict(sorted(payload.items()))


def _write_resolved_run_config_lockfile(
    *,
    run_id: str,
    args: argparse.Namespace,
    argv_tokens: list[str],
    run_settings_fingerprint: str,
    config_state: ConfigResolutionState,
    repo_root: Path | None = None,
) -> Path:
    """Persist a JSON lockfile capturing the fully-resolved run config."""
    root = repo_root if repo_root is not None else _REPO_ROOT
    path = root / "evals" / "artifacts" / "configs" / f"{run_id}.resolved.json"
    payload = {
        "schema_version": RUN_EVAL_CONFIG_LOCK_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "run_settings_fingerprint": run_settings_fingerprint,
        "config_files": list(config_state.config_paths),
        "value_sources": dict(sorted(config_state.value_sources.items())),
        "cli_argv": _redact_cli_argv(argv_tokens),
        "resolved_args": _resolved_args_for_lockfile(args),
        "runtime_resolution": dict(config_state.runtime_resolution),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    return path


def add_execution_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_executor: str = DEFAULT_EXECUTOR,
    include_defaults: bool = True,
) -> None:
    """Register ``--executor`` / ``--container-*`` CLI flags on *parser*."""

    def _default(value: Any) -> Any:
        return value if include_defaults else argparse.SUPPRESS

    resolved_container_image = resolve_default_container_image()
    parser.set_defaults(
        _container_image_default_source=resolved_container_image.source,
        _container_image_default_implicit_latest=resolved_container_image.implicit_latest,
        _container_image_explicit=False,
    )

    parser.add_argument(
        "--executor",
        choices=["local", "testcontainers"],
        default=_default(default_executor),
        help="Test execution backend.",
    )
    parser.add_argument(
        "--container-image",
        action=_StoreWithExplicitFlagAction,
        default=_default(resolved_container_image.value),
        help="Container image for --executor testcontainers.",
    )
    parser.add_argument(
        "--container-network",
        choices=["none", "bridge"],
        default=_default(DEFAULT_CONTAINER_NETWORK),
        help="Container network mode for --executor testcontainers.",
    )
    parser.add_argument(
        "--container-cpus",
        type=float,
        default=_default(DEFAULT_CONTAINER_CPUS),
        help="CPU limit for containerized pytest execution.",
    )
    parser.add_argument(
        "--container-memory",
        default=_default(DEFAULT_CONTAINER_MEMORY),
        help="Memory limit for containerized pytest execution (e.g. 1g).",
    )
    parser.add_argument(
        "--container-pids-limit",
        type=int,
        default=_default(DEFAULT_CONTAINER_PIDS_LIMIT),
        help="PID limit for containerized pytest execution.",
    )
    parser.add_argument(
        "--container-user",
        default=_default(DEFAULT_CONTAINER_USER),
        help="Container user for pytest execution (uid:gid).",
    )
    parser.add_argument(
        "--container-tmpfs-size-mb",
        type=int,
        default=_default(DEFAULT_CONTAINER_TMPFS_SIZE_MB),
        help="Size in MB for /tmp tmpfs in the container.",
    )
    parser.add_argument(
        "--container-mount-repo-src",
        dest="container_mount_repo_src",
        action="store_true",
        default=_default(True),
        help="Mount local src/ into container as read-only for llmdebug plugin imports.",
    )
    parser.add_argument(
        "--container-no-mount-repo-src",
        dest="container_mount_repo_src",
        action="store_false",
        default=_default(True),
        help="Do not mount local src/ into container.",
    )
    parser.add_argument(
        "--container-env",
        action="append",
        default=_default([]),
        help="Environment variable to inject in container (KEY=VALUE, repeatable).",
    )
    parser.add_argument(
        "--container-pool",
        dest="container_pool",
        action="store_true",
        default=_default(False),
        help=(
            "Reuse one long-lived container per job instead of a fresh container per pytest "
            "call (faster, weaker isolation between units)."
        ),
    )


def _parse_container_env(entries: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in entries:
        item = str(raw).strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid --container-env entry {raw!r}: expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --container-env entry {raw!r}: empty key")
        out[key] = value
    return out


def build_execution_config_from_args(args: argparse.Namespace) -> ExecutionConfig:
    """Build an ``ExecutionConfig`` from parsed CLI arguments.

    Raises:
        ValueError: If any execution flag has an invalid value.
    """
    executor = str(getattr(args, "executor", DEFAULT_EXECUTOR)).strip().lower()
    if executor not in VALID_EXECUTORS:
        raise ValueError(f"invalid --executor: {executor!r}")
    container_network = (
        str(getattr(args, "container_network", DEFAULT_CONTAINER_NETWORK)).strip().lower()
    )
    if container_network not in CONTAINER_NETWORK_MODES:
        raise ValueError(f"invalid --container-network: {container_network!r}")
    container_image = str(getattr(args, "container_image", DEFAULT_CONTAINER_IMAGE)).strip()
    if not container_image:
        raise ValueError("--container-image must be non-empty")
    container_cpus = float(getattr(args, "container_cpus", DEFAULT_CONTAINER_CPUS))
    if container_cpus <= 0:
        raise ValueError("--container-cpus must be > 0")
    container_pids_limit = int(getattr(args, "container_pids_limit", DEFAULT_CONTAINER_PIDS_LIMIT))
    if container_pids_limit <= 0:
        raise ValueError("--container-pids-limit must be > 0")
    container_tmpfs_size_mb = int(
        getattr(args, "container_tmpfs_size_mb", DEFAULT_CONTAINER_TMPFS_SIZE_MB)
    )
    if container_tmpfs_size_mb <= 0:
        raise ValueError("--container-tmpfs-size-mb must be > 0")
    container_user = str(getattr(args, "container_user", DEFAULT_CONTAINER_USER)).strip()
    if not container_user:
        raise ValueError("--container-user must be non-empty")
    container_memory = str(getattr(args, "container_memory", DEFAULT_CONTAINER_MEMORY)).strip()
    if not container_memory:
        raise ValueError("--container-memory must be non-empty")
    env_raw = getattr(args, "container_env", [])
    container_env = _parse_container_env(env_raw if isinstance(env_raw, list) else [])
    container_image_source, container_image_implicit_latest = (
        _effective_container_image_runtime_source(args)
    )
    warnings: tuple[str, ...] = ()
    if container_image_source == "implicit_fallback" and container_image == DEFAULT_CONTAINER_IMAGE:
        warnings = (
            "Using implicit fallback container image "
            f"{DEFAULT_CONTAINER_IMAGE!r}; set --container-image or "
            "LLMDEBUG_EVAL_CONTAINER_IMAGE for reproducible eval runs.",
        )
    return ExecutionConfig(
        executor=executor,
        container_image=container_image,
        container_image_source=container_image_source,
        container_image_implicit_latest=container_image_implicit_latest,
        container_network=container_network,
        container_cpus=container_cpus,
        container_memory=container_memory,
        container_pids_limit=container_pids_limit,
        container_user=container_user,
        container_tmpfs_size_mb=container_tmpfs_size_mb,
        container_mount_repo_src=bool(getattr(args, "container_mount_repo_src", True)),
        container_pool=bool(getattr(args, "container_pool", False)),
        container_env=container_env,
        warnings=warnings,
    )


def get_active_execution_config() -> ExecutionConfig:
    """Return the module-level active execution configuration."""
    return _ACTIVE_EXECUTION_CONFIG


def set_active_execution_config(config: ExecutionConfig) -> ExecutionConfig:
    """Replace the active execution config, returning the previous one."""
    global _ACTIVE_EXECUTION_CONFIG
    previous = _ACTIVE_EXECUTION_CONFIG
    _ACTIVE_EXECUTION_CONFIG = config
    return previous


def _execution_config_payload(config: ExecutionConfig) -> dict[str, Any]:
    return {
        "executor": config.executor,
        "container_image": config.container_image,
        "container_image_source": config.container_image_source,
        "container_image_implicit_latest": config.container_image_implicit_latest,
        "container_network": config.container_network,
        "container_cpus": config.container_cpus,
        "container_memory": config.container_memory,
        "container_pids_limit": config.container_pids_limit,
        "container_user": config.container_user,
        "container_tmpfs_size_mb": config.container_tmpfs_size_mb,
        "container_mount_repo_src": config.container_mount_repo_src,
        "container_pool": config.container_pool,
        "container_env_keys": sorted(config.container_env),
    }
