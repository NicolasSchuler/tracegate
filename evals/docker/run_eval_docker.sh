#!/usr/bin/env bash
# Ergonomic wrapper for running the eval pipeline inside a Docker container.
#
# Usage:
#   evals/docker/run_eval_docker.sh [run_eval.py args...]
#
# Examples:
#   # Smoke test (no network needed)
#   evals/docker/run_eval_docker.sh --patcher null --max-cases 5 --k 1
#
#   # OpenAI patcher against local vLLM
#   evals/docker/run_eval_docker.sh \
#     --patcher openai --openai-model mistral-7b \
#     --openai-base-url http://127.0.0.1:12000/v1 \
#     --category hidden_state --k 2
#
#   # Run from TOML config (paths are staged into the container automatically)
#   evals/docker/run_eval_docker.sh \
#     --patcher openai \
#     --config evals/configs/openai_adaptive.template.toml
#
#   # Force rebuild
#   LLMDEBUG_EVAL_BUILD=1 evals/docker/run_eval_docker.sh --patcher null --max-cases 1
#
# Environment variables:
#   LLMDEBUG_EVAL_BUILD=1          Force image rebuild
#   LLMDEBUG_EVAL_BUILD_NO_CACHE=1 Force docker build with --no-cache
#   LLMDEBUG_EVAL_DOCKER_TTY=1|0|auto
#                                  Control whether the wrapper allocates a container TTY.
#                                  Default `1` enables the Rich/live `run_eval` UI by
#                                  default. Use `0` for plain logs or `auto` to preserve
#                                  the host TTY behavior.
#   LLMDEBUG_EVAL_IMAGE=name:tag   Override image name (defaults to llmdebug-eval:latest
#                                  or llmdebug-eval-baselines:latest for baseline runs)
#   DOCKER_EXTRA_ARGS="..."        Extra docker run flags (e.g. --gpus all)
#   OPENAI_MODEL=...               Default model when an OpenAI-backed patcher is used
#   OPENAI_BASE_URL=...            Default base URL when an OpenAI-backed patcher is used
#   Z_API / OPENAI_API_KEY / LLMDEBUG_EVAL_OPENAI_API_KEY
#                                  Used to synthesize --openai-api-key when omitted
#                                  (falls back to `local` when all are blank)
#   TRACEGATE_DECISION_MODEL_API_KEY
#                                  Used to synthesize --adaptive-decision-model-api-key
#                                  when omitted (decision model behind llama-swap apiKeys)
#
# Implementation note:
#   The wrapper uses `docker create` + `docker start -a` instead of `docker run`
#   so it can explicitly stop/remove the container, including anonymous volumes,
#   and preserve signal-specific exit codes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_IMAGE="llmdebug-eval:latest"
DEFAULT_BASELINE_IMAGE="llmdebug-eval-baselines:latest"
# Mirrors DEFAULT_DECISION_MODEL_URL in evals/decision_model/config.py: the harness falls
# back to it when no URL is configured, and a test asserts both stay in sync.
DEFAULT_DECISION_MODEL_URL="http://127.0.0.1:8090"
IMAGE_BUILD_FINGERPRINT_LABEL="io.github.nicolasschuler.llmdebug.eval-build-fingerprint"
IMAGE=""
IMAGE_BASELINES_ENABLED=0
CLEANUP_PATHS=()
CONTAINER_ID=""

_register_cleanup_path() {
    local path="$1"
    CLEANUP_PATHS+=("$path")
}

_cleanup_container() {
    if [[ -n "${CONTAINER_ID:-}" ]]; then
        docker rm -f -v "$CONTAINER_ID" >/dev/null 2>&1 || true
    fi
}

_signal_exit_code() {
    case "$1" in
        INT)
            printf "130\n"
            ;;
        TERM)
            printf "143\n"
            ;;
        HUP)
            printf "129\n"
            ;;
        QUIT)
            printf "131\n"
            ;;
        *)
            printf "1\n"
            ;;
    esac
}

_cleanup_wrapper_artifacts() {
    local path

    _cleanup_container

    for path in "${CLEANUP_PATHS[@]}"; do
        rm -rf "$path"
    done
}

_handle_interrupt() {
    local signal_name="$1"
    local exit_code

    _cleanup_container
    exit_code="$(_signal_exit_code "$signal_name")"
    exit "$exit_code"
}

trap '_handle_interrupt INT' INT
trap '_handle_interrupt TERM' TERM
trap '_handle_interrupt HUP' HUP
trap '_handle_interrupt QUIT' QUIT
trap _cleanup_wrapper_artifacts EXIT

# ---------------------------------------------------------------------------
# Prepare docker run arguments
# ---------------------------------------------------------------------------
DOCKER_ARGS=(
    -v "$REPO_ROOT/evals/results:/repo/evals/results"
    -v "$REPO_ROOT/evals/artifacts:/repo/evals/artifacts"
)
DEFAULT_DOCKER_USER="$(id -u):$(id -g)"
DOCKER_USER="${LLMDEBUG_EVAL_DOCKER_USER:-$DEFAULT_DOCKER_USER}"

# Forward non-secret config env vars if set.
# OpenAI credentials are passed via CLI flags (only when needed) so that
# benchmarked test code cannot read them from the environment.
for var in LLMDEBUG_EVAL_PATCH_CMD LLMDEBUG_EVAL_PATCH_CMD_TIMEOUT_SEC \
           LLMDEBUG_EVAL_PATCH_CMD_SHELL HF_HOME TRACEGATE_DECISION_MODEL_URL; do
    if [[ -n "${!var:-}" ]]; then
        DOCKER_ARGS+=(-e "$var=${!var}")
    fi
done

# Extra user-specified docker flags
if [[ -n "${DOCKER_EXTRA_ARGS:-}" ]]; then
    read -ra _extra <<< "$DOCKER_EXTRA_ARGS"
    for arg in "${_extra[@]}"; do
        if [[ "$arg" == "--rm" ]]; then
            echo ">>> NOTE: Ignoring wrapper-managed --rm in DOCKER_EXTRA_ARGS." >&2
            continue
        fi
        DOCKER_ARGS+=("$arg")
    done
fi

# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------
_arg_value() {
    local key="$1"
    shift
    local next_is_value=0
    local found=0
    local value=""
    local arg
    for arg in "$@"; do
        if [[ $next_is_value -eq 1 ]]; then
            value="$arg"
            found=1
            next_is_value=0
            continue
        fi
        if [[ "$arg" == "$key" ]]; then
            next_is_value=1
            continue
        fi
        if [[ "$arg" == "$key="* ]]; then
            value="${arg#*=}"
            found=1
        fi
    done
    if [[ $found -eq 1 ]]; then
        printf "%s\n" "$value"
        return 0
    fi
    return 1
}

_arg_present() {
    local key="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$key" || "$arg" == "$key="* ]]; then
            return 0
        fi
    done
    return 1
}

_replace_or_append_arg() {
    local key="$1"
    local value="$2"
    local out=()
    local replaced=0
    local skip_next=0
    local arg

    for arg in "${EVAL_ARGS[@]}"; do
        if [[ $skip_next -eq 1 ]]; then
            skip_next=0
            continue
        fi
        if [[ "$arg" == "$key" ]]; then
            out+=("$key" "$value")
            replaced=1
            skip_next=1
            continue
        fi
        if [[ "$arg" == "$key="* ]]; then
            out+=("$key=$value")
            replaced=1
            continue
        fi
        out+=("$arg")
    done

    if [[ $replaced -eq 0 ]]; then
        out+=("$key" "$value")
    fi

    EVAL_ARGS=("${out[@]}")
}

_first_nonempty_env() {
    local var
    local value
    for var in "$@"; do
        value="${!var:-}"
        if [[ -n "$value" ]]; then
            printf "%s\n" "$value"
            return 0
        fi
    done
    return 1
}

_default_openai_api_key() {
    _first_nonempty_env Z_API OPENAI_API_KEY LLMDEBUG_EVAL_OPENAI_API_KEY || printf "local\n"
}

_env_value_or_empty() {
    local var="$1"
    if [[ -z "$var" ]]; then
        return 0
    fi
    printf "%s\n" "${!var:-}"
}

_wrapper_should_allocate_tty() {
    local override="${LLMDEBUG_EVAL_DOCKER_TTY:-1}"
    case "$override" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        0|false|FALSE|no|NO|off|OFF)
            return 1
            ;;
        auto|"")
            [[ -t 1 ]]
            return
            ;;
        *)
            [[ -t 1 ]]
            return
            ;;
    esac
}

_python_cmd() {
    if command -v python3 >/dev/null 2>&1; then
        printf "python3\n"
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        printf "python\n"
        return 0
    fi
    echo ">>> ERROR: python3 or python is required to compute the eval Docker image fingerprint." >&2
    exit 2
}

_compute_image_build_fingerprint() {
    local include_baselines="$1"
    local python_cmd
    python_cmd="$(_python_cmd)"
    "$python_cmd" - "$REPO_ROOT" "$include_baselines" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[1])
include_baselines = sys.argv[2] == "1"
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from evals.docker_image_fingerprint import compute_eval_image_build_fingerprint

print(compute_eval_image_build_fingerprint(repo_root, include_baselines=include_baselines))
PY
}

_docker_image_label_value() {
    local image="$1"
    local label="$2"
    docker image inspect --format "{{ index .Config.Labels \"$label\" }}" "$image" 2>/dev/null
}

_expand_config_path() {
    local raw="$1"
    local expanded="$raw"
    if [[ "$expanded" == "~/"* ]]; then
        expanded="$HOME/${expanded#~/}"
    fi
    if [[ "$expanded" != /* ]]; then
        expanded="$PWD/$expanded"
    fi
    printf "%s\n" "$expanded"
}

_rewrite_config_paths_for_container() {
    local config_stage_id
    config_stage_id="$(date +%Y%m%dT%H%M%S)-$$-$RANDOM"
    local host_mount_dir="$REPO_ROOT/evals/artifacts/config_mounts/$config_stage_id"
    local guest_mount_dir="/repo/evals/artifacts/config_mounts/$config_stage_id"
    local out=()
    local expect_value=0
    local config_index=0
    local arg

    for arg in "${EVAL_ARGS[@]}"; do
        if [[ $expect_value -eq 1 ]]; then
            local host_path
            host_path="$(_expand_config_path "$arg")"
            if [[ ! -f "$host_path" ]]; then
                echo ">>> ERROR: Config file not found: $arg" >&2
                exit 2
            fi
            mkdir -p "$host_mount_dir"
            _register_cleanup_path "$host_mount_dir"
            config_index=$((config_index + 1))
            local staged_name
            staged_name="$(printf "%03d_%s" "$config_index" "$(basename "$host_path")")"
            cp "$host_path" "$host_mount_dir/$staged_name"
            out+=(--config "$guest_mount_dir/$staged_name")
            expect_value=0
            continue
        fi

        if [[ "$arg" == "--config" ]]; then
            expect_value=1
            continue
        fi
        if [[ "$arg" == --config=* ]]; then
            local raw_path="${arg#--config=}"
            local host_path
            host_path="$(_expand_config_path "$raw_path")"
            if [[ ! -f "$host_path" ]]; then
                echo ">>> ERROR: Config file not found: $raw_path" >&2
                exit 2
            fi
            mkdir -p "$host_mount_dir"
            _register_cleanup_path "$host_mount_dir"
            config_index=$((config_index + 1))
            local staged_name
            staged_name="$(printf "%03d_%s" "$config_index" "$(basename "$host_path")")"
            cp "$host_path" "$host_mount_dir/$staged_name"
            out+=("--config=$guest_mount_dir/$staged_name")
            continue
        fi

        out+=("$arg")
    done

    if [[ $expect_value -eq 1 ]]; then
        echo ">>> ERROR: --config requires a path value" >&2
        exit 2
    fi

    EVAL_ARGS=("${out[@]}")
}

_rewrite_case_list_paths_for_container() {
    local case_list_stage_id
    case_list_stage_id="$(date +%Y%m%dT%H%M%S)-$$-$RANDOM"
    local host_mount_dir="$REPO_ROOT/evals/artifacts/case_list_mounts/$case_list_stage_id"
    local guest_mount_dir="/repo/evals/artifacts/case_list_mounts/$case_list_stage_id"
    local out=()
    local expect_value=0
    local case_list_index=0
    local arg

    for arg in "${EVAL_ARGS[@]}"; do
        if [[ $expect_value -eq 1 ]]; then
            local host_path
            host_path="$(_expand_config_path "$arg")"
            if [[ ! -f "$host_path" ]]; then
                echo ">>> ERROR: Case-list file not found: $arg" >&2
                exit 2
            fi
            mkdir -p "$host_mount_dir"
            _register_cleanup_path "$host_mount_dir"
            case_list_index=$((case_list_index + 1))
            local staged_name
            staged_name="$(printf "%03d_%s" "$case_list_index" "$(basename "$host_path")")"
            cp "$host_path" "$host_mount_dir/$staged_name"
            out+=(--case-list-file "$guest_mount_dir/$staged_name")
            expect_value=0
            continue
        fi

        if [[ "$arg" == "--case-list-file" ]]; then
            expect_value=1
            continue
        fi
        if [[ "$arg" == --case-list-file=* ]]; then
            local raw_path="${arg#--case-list-file=}"
            local host_path
            host_path="$(_expand_config_path "$raw_path")"
            if [[ ! -f "$host_path" ]]; then
                echo ">>> ERROR: Case-list file not found: $raw_path" >&2
                exit 2
            fi
            mkdir -p "$host_mount_dir"
            _register_cleanup_path "$host_mount_dir"
            case_list_index=$((case_list_index + 1))
            local staged_name
            staged_name="$(printf "%03d_%s" "$case_list_index" "$(basename "$host_path")")"
            cp "$host_path" "$host_mount_dir/$staged_name"
            out+=("--case-list-file=$guest_mount_dir/$staged_name")
            continue
        fi

        out+=("$arg")
    done

    if [[ $expect_value -eq 1 ]]; then
        echo ">>> ERROR: --case-list-file requires a path value" >&2
        exit 2
    fi

    EVAL_ARGS=("${out[@]}")
}

_collect_raw_config_paths() {
    local out=()
    local expect_value=0
    local arg

    for arg in "${EVAL_ARGS[@]}"; do
        if [[ $expect_value -eq 1 ]]; then
            local expanded
            expanded="$(_expand_config_path "$arg")"
            if [[ -f "$expanded" ]]; then
                out+=("$expanded")
            fi
            expect_value=0
            continue
        fi

        if [[ "$arg" == "--config" ]]; then
            expect_value=1
            continue
        fi
        if [[ "$arg" == --config=* ]]; then
            local expanded
            expanded="$(_expand_config_path "${arg#--config=}")"
            if [[ -f "$expanded" ]]; then
                out+=("$expanded")
            fi
        fi
    done

    printf "%s\n" "${out[@]}"
}

_config_value_from_files() {
    local key="$1"
    shift
    if [[ $# -eq 0 ]]; then
        return 1
    fi

    awk -v wanted="$key" '
        function trim(text) {
            sub(/^[[:space:]]+/, "", text)
            sub(/[[:space:]]+$/, "", text)
            return text
        }
        function strip_quotes(text) {
            if ((text ~ /^".*"$/) || (text ~ /^'\''.*'\''$/)) {
                return substr(text, 2, length(text) - 2)
            }
            return text
        }
        FNR == 1 {
            section = ""
        }
        {
            line = $0
            sub(/[[:space:]]*#.*/, "", line)
            line = trim(line)
            if (line == "") {
                next
            }
            if (line ~ /^\[/) {
                section = line
                gsub(/^\[/, "", section)
                gsub(/\]$/, "", section)
                section = trim(section)
                gsub(/-/, "_", section)
                section = tolower(section)
                next
            }
            split(line, parts, "=")
            key = trim(parts[1])
            gsub(/-/, "_", key)
            key = tolower(key)
            if (key != wanted) {
                next
            }
            if (section != "" && section != "run_eval") {
                next
            }
            value = trim(substr(line, index(line, "=") + 1))
            print strip_quotes(value)
        }
    ' "$@" | tail -n 1
}

_config_openai_value_from_files() {
    local leaf_key="$1"
    shift
    if [[ $# -eq 0 ]]; then
        return 1
    fi

    awk -v wanted="$leaf_key" '
        function trim(text) {
            sub(/^[[:space:]]+/, "", text)
            sub(/[[:space:]]+$/, "", text)
            return text
        }
        function strip_quotes(text) {
            if ((text ~ /^".*"$/) || (text ~ /^'\''.*'\''$/)) {
                return substr(text, 2, length(text) - 2)
            }
            return text
        }
        FNR == 1 {
            section = ""
        }
        {
            line = $0
            sub(/[[:space:]]*#.*/, "", line)
            line = trim(line)
            if (line == "") {
                next
            }
            if (line ~ /^\[/) {
                section = line
                gsub(/^\[/, "", section)
                gsub(/\]$/, "", section)
                section = trim(section)
                gsub(/-/, "_", section)
                section = tolower(section)
                next
            }
            split(line, parts, "=")
            key = trim(parts[1])
            gsub(/-/, "_", key)
            key = tolower(key)
            value = trim(substr(line, index(line, "=") + 1))

            if ((section == "run_eval.openai" || section == "openai") && key == wanted) {
                print strip_quotes(value)
                next
            }
            if ((section == "" || section == "run_eval") && key == ("openai_" wanted)) {
                print strip_quotes(value)
            }
        }
    ' "$@" | tail -n 1
}

# Same shape as _config_openai_value_from_files, for the [run_eval.adaptive] sub-table.
# _config_value_from_files deliberately ignores sub-tables (patcher/baseline keys must not be
# picked up from them), so sub-table lookups get their own reader.
_config_adaptive_value_from_files() {
    local leaf_key="$1"
    shift
    if [[ $# -eq 0 ]]; then
        return 1
    fi

    awk -v wanted="$leaf_key" '
        function trim(text) {
            sub(/^[[:space:]]+/, "", text)
            sub(/[[:space:]]+$/, "", text)
            return text
        }
        function strip_quotes(text) {
            if ((text ~ /^".*"$/) || (text ~ /^'\''.*'\''$/)) {
                return substr(text, 2, length(text) - 2)
            }
            return text
        }
        FNR == 1 {
            section = ""
        }
        {
            line = $0
            sub(/[[:space:]]*#.*/, "", line)
            line = trim(line)
            if (line == "") {
                next
            }
            if (line ~ /^\[/) {
                section = line
                gsub(/^\[/, "", section)
                gsub(/\]$/, "", section)
                section = trim(section)
                gsub(/-/, "_", section)
                section = tolower(section)
                next
            }
            split(line, parts, "=")
            key = trim(parts[1])
            gsub(/-/, "_", key)
            key = tolower(key)
            value = trim(substr(line, index(line, "=") + 1))

            if ((section == "run_eval.adaptive" || section == "adaptive") && key == wanted) {
                print strip_quotes(value)
                next
            }
            if ((section == "" || section == "run_eval") && key == ("adaptive_" wanted)) {
                print strip_quotes(value)
            }
        }
    ' "$@" | tail -n 1
}

_docker_has_network_flag() {
    local expect_value=0
    local arg
    for arg in "${DOCKER_ARGS[@]}"; do
        if [[ $expect_value -eq 1 ]]; then
            return 0
        fi
        case "$arg" in
            --network|--net)
                expect_value=1
                ;;
            --network=*|--net=*)
                return 0
                ;;
        esac
    done
    return 1
}

_docker_user_can_write_outputs() {
    local user="$1"
    docker run \
        --rm \
        --entrypoint /bin/sh \
        --user "$user" \
        -v "$REPO_ROOT/evals/results:/repo/evals/results" \
        -v "$REPO_ROOT/evals/artifacts:/repo/evals/artifacts" \
        "$IMAGE" \
        -lc 'touch /repo/evals/results/.perm_probe && rm -f /repo/evals/results/.perm_probe && touch /repo/evals/artifacts/.perm_probe && rm -f /repo/evals/artifacts/.perm_probe' \
        >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Filter and forward arguments to evals.run_eval
# ---------------------------------------------------------------------------
EVAL_ARGS=()
skip_next=0
for arg in "$@"; do
    if [[ $skip_next -eq 1 ]]; then
        skip_next=0
        continue
    fi

    case "$arg" in
        --executor)
            skip_next=1
            ;;
        --executor=*)
            ;;
        --container-image|--container-network|--container-cpus|--container-memory|\
        --container-pids-limit|--container-user|--container-tmpfs-size-mb|--container-env)
            skip_next=1
            ;;
        --container-image=*|--container-network=*|--container-cpus=*|--container-memory=*|\
        --container-pids-limit=*|--container-user=*|--container-tmpfs-size-mb=*|--container-env=*)
            ;;
        --container-mount-repo-src|--container-no-mount-repo-src)
            ;;
        --container-*)
            echo ">>> NOTE: Ignoring unsupported wrapper-only arg: $arg" >&2
            ;;
        *)
            EVAL_ARGS+=("$arg")
            ;;
    esac
done

# Inspect raw config files before staging them into the container so wrapper
# behavior can react to config-driven baseline runs.
RAW_CONFIG_PATHS=()
while IFS= read -r config_path; do
    if [[ -n "$config_path" ]]; then
        RAW_CONFIG_PATHS+=("$config_path")
    fi
done < <(_collect_raw_config_paths)

CLI_PATCHER="$(_arg_value --patcher "${EVAL_ARGS[@]}" || true)"
CLI_BASELINE="$(_arg_value --baseline "${EVAL_ARGS[@]}" || true)"
CLI_BASELINE_NAME="$(_arg_value --baseline-name "${EVAL_ARGS[@]}" || true)"
CLI_BASELINE_REPO_DIR="$(_arg_value --baseline-repo-dir "${EVAL_ARGS[@]}" || true)"
CLI_BASELINE_PYTHON="$(_arg_value --baseline-python "${EVAL_ARGS[@]}" || true)"

CONFIG_PATCHER="$(_config_value_from_files patcher "${RAW_CONFIG_PATHS[@]}" || true)"
CONFIG_BASELINE="$(_config_value_from_files baseline "${RAW_CONFIG_PATHS[@]}" || true)"
CONFIG_BASELINE_NAME="$(_config_value_from_files baseline_name "${RAW_CONFIG_PATHS[@]}" || true)"
CONFIG_BASELINE_REPO_DIR="$(_config_value_from_files baseline_repo_dir "${RAW_CONFIG_PATHS[@]}" || true)"
CONFIG_BASELINE_PYTHON="$(_config_value_from_files baseline_python "${RAW_CONFIG_PATHS[@]}" || true)"
CONFIG_OPENAI_API_KEY_ENV="$(_config_value_from_files openai_api_key_env "${RAW_CONFIG_PATHS[@]}" || true)"
CONFIG_OPENAI_BASE_URL="$(_config_openai_value_from_files base_url "${RAW_CONFIG_PATHS[@]}" || true)"
CONFIG_OPENAI_MODEL="$(_config_openai_value_from_files model "${RAW_CONFIG_PATHS[@]}" || true)"
CONFIG_DECISION_MODEL_URL="$(_config_adaptive_value_from_files decision_model_url "${RAW_CONFIG_PATHS[@]}" || true)"
CONFIG_DECISION_MODEL_ROLE="$(_config_adaptive_value_from_files decision_model_role "${RAW_CONFIG_PATHS[@]}" || true)"

PATCHER="${CLI_PATCHER:-$CONFIG_PATCHER}"
PATCHER="${PATCHER:-command}"

BASELINE_MODE=0
if [[ -n "${CLI_BASELINE:-}" || "$PATCHER" == "baseline" ]]; then
    BASELINE_MODE=1
elif [[ -n "${CONFIG_BASELINE:-}" ]]; then
    if [[ -z "${CLI_PATCHER:-}" || "$CLI_PATCHER" == "command" ]]; then
        BASELINE_MODE=1
    fi
fi

EFFECTIVE_BASELINE_REPO_DIR="${CLI_BASELINE_REPO_DIR:-$CONFIG_BASELINE_REPO_DIR}"
EFFECTIVE_BASELINE_PYTHON="${CLI_BASELINE_PYTHON:-$CONFIG_BASELINE_PYTHON}"

if [[ $BASELINE_MODE -eq 1 ]]; then
    if [[ -n "${EFFECTIVE_BASELINE_REPO_DIR:-}" ]]; then
        echo ">>> ERROR: Docker baseline runs do not support custom --baseline-repo-dir." >&2
        echo ">>>        Use the image-baked baseline repos instead." >&2
        exit 2
    fi
    if [[ -n "${EFFECTIVE_BASELINE_PYTHON:-}" ]]; then
        echo ">>> ERROR: Docker baseline runs do not support custom --baseline-python." >&2
        echo ">>>        Use the image-baked Linux baseline envs instead." >&2
        exit 2
    fi
fi

if [[ -n "${LLMDEBUG_EVAL_IMAGE:-}" ]]; then
    IMAGE="$LLMDEBUG_EVAL_IMAGE"
elif [[ $BASELINE_MODE -eq 1 ]]; then
    IMAGE="$DEFAULT_BASELINE_IMAGE"
else
    IMAGE="$DEFAULT_IMAGE"
fi

if [[ $BASELINE_MODE -eq 1 ]]; then
    IMAGE_BASELINES_ENABLED=1
fi

# Stage and rewrite --config file paths so they are readable inside the container.
_rewrite_config_paths_for_container
_rewrite_case_list_paths_for_container

# OpenAI defaults are resolved into CLI args so values are deterministic and
# localhost URL translation can safely rewrite the effective value.
if [[ "$PATCHER" == "openai" || $BASELINE_MODE -eq 1 ]]; then
    if ! _arg_present --openai-model "${EVAL_ARGS[@]}"; then
        if [[ -n "${CONFIG_OPENAI_MODEL:-}" ]]; then
            EVAL_ARGS+=(--openai-model "$CONFIG_OPENAI_MODEL")
        elif [[ -n "${OPENAI_MODEL:-}" ]]; then
            EVAL_ARGS+=(--openai-model "$OPENAI_MODEL")
        fi
    fi

    if ! _arg_present --openai-base-url "${EVAL_ARGS[@]}"; then
        if [[ -n "${CONFIG_OPENAI_BASE_URL:-}" ]]; then
            EVAL_ARGS+=(--openai-base-url "$CONFIG_OPENAI_BASE_URL")
        else
            EVAL_ARGS+=(--openai-base-url "${OPENAI_BASE_URL:-http://127.0.0.1:12000/v1}")
        fi
    fi

    if ! _arg_present --openai-api-key "${EVAL_ARGS[@]}"; then
        OPENAI_API_KEY_EFFECTIVE=""
        if [[ -n "${CONFIG_OPENAI_API_KEY_ENV:-}" ]]; then
            if [[ "$CONFIG_OPENAI_API_KEY_ENV" == "OPENAI_API_KEY" ]]; then
                OPENAI_API_KEY_EFFECTIVE="$(_default_openai_api_key)"
            else
                OPENAI_API_KEY_EFFECTIVE="$(_env_value_or_empty "$CONFIG_OPENAI_API_KEY_ENV")"
            fi
            if [[ -z "${OPENAI_API_KEY_EFFECTIVE:-}" ]]; then
                echo ">>> ERROR: environment variable '$CONFIG_OPENAI_API_KEY_ENV' is required by openai_api_key_env but is empty or unset." >&2
                exit 2
            fi
        else
            OPENAI_API_KEY_EFFECTIVE="$(_default_openai_api_key)"
        fi
        if [[ -n "${OPENAI_API_KEY_EFFECTIVE:-}" ]]; then
            EVAL_ARGS+=(--openai-api-key "$OPENAI_API_KEY_EFFECTIVE")
        fi
    fi
fi

# Lift the decision-model sidecar URL from [run_eval.adaptive] (or the environment) into a CLI
# arg so localhost translation can rewrite the effective value. The flag wins over the
# forwarded TRACEGATE_DECISION_MODEL_URL env var inside the container.
CLI_DECISION_MODEL_ROLE="$(_arg_value --adaptive-decision-model-role "${EVAL_ARGS[@]}" || true)"
EFFECTIVE_DECISION_MODEL_ROLE="${CLI_DECISION_MODEL_ROLE:-${CONFIG_DECISION_MODEL_ROLE:-off}}"
if ! _arg_present --adaptive-decision-model-url "${EVAL_ARGS[@]}"; then
    if [[ -n "${CONFIG_DECISION_MODEL_URL:-}" ]]; then
        EVAL_ARGS+=(--adaptive-decision-model-url "$CONFIG_DECISION_MODEL_URL")
    elif [[ -n "${TRACEGATE_DECISION_MODEL_URL:-}" ]]; then
        EVAL_ARGS+=(--adaptive-decision-model-url "$TRACEGATE_DECISION_MODEL_URL")
    elif [[ "$EFFECTIVE_DECISION_MODEL_ROLE" != "off" ]]; then
        # An enabled role without a configured URL uses the harness default, which points at
        # the container's own loopback once we run inside Docker. Materialize it so the
        # localhost translation below adds host networking (Linux) or rewrites the host
        # (Darwin) instead of letting the preflight fail against nothing.
        echo ">>> NOTE: Decision-model role '$EFFECTIVE_DECISION_MODEL_ROLE' without a URL; assuming $DEFAULT_DECISION_MODEL_URL" >&2
        EVAL_ARGS+=(--adaptive-decision-model-url "$DEFAULT_DECISION_MODEL_URL")
    fi
fi

# Like the OpenAI key, a decision-model API key (e.g. for llama-swap apiKeys) reaches the harness
# only as a CLI flag, never as an environment variable inside the container.
if ! _arg_present --adaptive-decision-model-api-key "${EVAL_ARGS[@]}"; then
    if [[ -n "${TRACEGATE_DECISION_MODEL_API_KEY:-}" ]]; then
        EVAL_ARGS+=(--adaptive-decision-model-api-key "$TRACEGATE_DECISION_MODEL_API_KEY")
    fi
fi

# ---------------------------------------------------------------------------
# Auto-build the selected image if it doesn't exist, is stale, or rebuild requested
# ---------------------------------------------------------------------------
CURRENT_IMAGE_BUILD_FINGERPRINT="$(_compute_image_build_fingerprint "$IMAGE_BASELINES_ENABLED")"
IMAGE_BUILD_REASON=""

if [[ "${LLMDEBUG_EVAL_BUILD:-}" == "1" ]]; then
    IMAGE_BUILD_REASON="forced rebuild requested"
elif [[ "${LLMDEBUG_EVAL_BUILD_NO_CACHE:-}" == "1" ]]; then
    IMAGE_BUILD_REASON="no-cache rebuild requested"
elif ! docker image inspect "$IMAGE" &>/dev/null; then
    IMAGE_BUILD_REASON="image missing"
else
    EXISTING_IMAGE_BUILD_FINGERPRINT="$(_docker_image_label_value "$IMAGE" "$IMAGE_BUILD_FINGERPRINT_LABEL" || true)"
    if [[ "${EXISTING_IMAGE_BUILD_FINGERPRINT:-}" == "<no value>" ]] \
        || [[ "${EXISTING_IMAGE_BUILD_FINGERPRINT:-}" == "<nil>" ]]; then
        EXISTING_IMAGE_BUILD_FINGERPRINT=""
    fi

    if [[ -z "${EXISTING_IMAGE_BUILD_FINGERPRINT:-}" ]]; then
        echo ">>> NOTE: Existing $IMAGE image is missing a build fingerprint label; rebuilding." >&2
        IMAGE_BUILD_REASON="image missing fingerprint label"
    elif [[ "$EXISTING_IMAGE_BUILD_FINGERPRINT" != "$CURRENT_IMAGE_BUILD_FINGERPRINT" ]]; then
        echo ">>> NOTE: Existing $IMAGE image does not match current repo contents; rebuilding." >&2
        IMAGE_BUILD_REASON="image fingerprint mismatch"
    fi
fi

if [[ -n "$IMAGE_BUILD_REASON" ]]; then
    echo ">>> Building $IMAGE ($IMAGE_BUILD_REASON) ..."
    DOCKER_BUILD_ARGS=()
    if [[ "${LLMDEBUG_EVAL_BUILD_NO_CACHE:-}" == "1" ]]; then
        DOCKER_BUILD_ARGS+=(--no-cache)
    fi
    docker build \
        "${DOCKER_BUILD_ARGS[@]}" \
        --build-arg "LLMDEBUG_EVAL_INCLUDE_BASELINES=$IMAGE_BASELINES_ENABLED" \
        --build-arg "LLMDEBUG_EVAL_BUILD_FINGERPRINT=$CURRENT_IMAGE_BUILD_FINGERPRINT" \
        -t "$IMAGE" \
        -f "$REPO_ROOT/evals/docker/eval.Dockerfile" \
        "$REPO_ROOT"
    echo ">>> Build complete."
fi

# Localhost API URL handling for Docker networking. Returns success when the caller has to add
# host networking (Linux); on Darwin the URL is rewritten in place instead.
_translate_localhost_url_arg() {
    local flag="$1"
    local current=""
    local translated=""

    current="$(_arg_value "$flag" "${EVAL_ARGS[@]}" || true)"
    if [[ -z "${current:-}" ]] \
        || ! printf "%s" "$current" | grep -qE '(127\.0\.0\.1|localhost)'; then
        return 1
    fi

    if [[ "$(uname -s)" != "Darwin" ]]; then
        return 0
    fi

    translated="$(printf "%s" "$current" | sed -E 's/(127\.0\.0\.1|localhost)/host.docker.internal/g')"
    if [[ "$translated" != "$current" ]]; then
        echo ">>> NOTE: Translating localhost URL for Docker access" >&2
        echo ">>>   $current -> $translated" >&2
        _replace_or_append_arg "$flag" "$translated"
    fi
    return 1
}

NEEDS_HOST_NETWORK=0
for url_flag in --openai-base-url --adaptive-decision-model-url; do
    if _translate_localhost_url_arg "$url_flag"; then
        NEEDS_HOST_NETWORK=1
    fi
done
if [[ $NEEDS_HOST_NETWORK -eq 1 ]] && ! _docker_has_network_flag; then
    echo ">>> NOTE: Detected localhost API URL, using --network host" >&2
    DOCKER_ARGS+=(--network host)
fi

# Force local executor (the whole point of this container). Pass the selected
# wrapper image into eval metadata too, so local-executor runs do not emit the
# unrelated implicit testcontainers image warning.
EVAL_ARGS+=(--executor local --container-image "$IMAGE")

# ---------------------------------------------------------------------------
# Ensure output directories exist on host (docker mount creates them as root otherwise)
# ---------------------------------------------------------------------------
mkdir -p "$REPO_ROOT/evals/results" "$REPO_ROOT/evals/artifacts"

# On some Docker Desktop mount setups, host-owned dirs are presented as
# root-owned 755 inside the container. In that case, forcing --user UID:GID
# causes PermissionError when creating run artifact folders.
if [[ -n "$DOCKER_USER" ]]; then
    if [[ -n "${LLMDEBUG_EVAL_DOCKER_USER:-}" ]]; then
        DOCKER_ARGS+=(--user "$DOCKER_USER")
    elif _docker_user_can_write_outputs "$DOCKER_USER"; then
        DOCKER_ARGS+=(--user "$DOCKER_USER")
    else
        echo ">>> WARNING: Docker user $DOCKER_USER cannot write mounted eval outputs; falling back to root." >&2
        echo ">>>          Set LLMDEBUG_EVAL_DOCKER_USER to override this behavior." >&2
    fi
fi

# ---------------------------------------------------------------------------
# Run — keep this shell alive so cleanup traps can reap the container and any
# staged config mounts even when the docker process exits non-zero or is
# interrupted.
# ---------------------------------------------------------------------------
CONTAINER_ID="$(
    docker create \
        "${DOCKER_ARGS[@]}" \
        $(_wrapper_should_allocate_tty && printf '%s' '--tty') \
        --entrypoint python \
        "$IMAGE" \
        -m evals.run_eval \
        "${EVAL_ARGS[@]}"
)"

docker start -a "$CONTAINER_ID"
exit_code=$?
exit "$exit_code"
