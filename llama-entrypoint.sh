#!/usr/bin/env bash
set -euo pipefail

# If the first argument is not an option, treat it as a full command override.
if [[ $# -gt 0 && "${1#-}" == "$1" ]]; then
    exec "$@"
fi

# Allow bypassing defaults for fully manual llama-server control.
if [[ "${LLAMA_DISABLE_DEFAULT_ARGS:-0}" == "1" ]]; then
    exec llama-server "$@"
fi

host="${LLAMA_HOST:-0.0.0.0}"
port="${LLAMA_PORT:-8080}"
preset="${LLAMA_PRESET:-latency}"
swa_full="${LLAMA_SWA_FULL:-1}"
ngl="${LLAMA_NGL:-all}"
flash_attn="${LLAMA_FLASH_ATTN:-auto}"
cache_type_k="${LLAMA_CACHE_TYPE_K:-q8_0}"
cache_type_v="${LLAMA_CACHE_TYPE_V:-q8_0}"
seed="${LLAMA_SEED:-}"
reasoning="${LLAMA_REASONING:-}"
reasoning_format="${LLAMA_REASONING_FORMAT:-}"
temp="${LLAMA_TEMP:-}"
top_p="${LLAMA_TOP_P:-}"
top_k="${LLAMA_TOP_K:-}"
min_p="${LLAMA_MIN_P:-}"
repeat_penalty="${LLAMA_REPEAT_PENALTY:-}"
presence_penalty="${LLAMA_PRESENCE_PENALTY:-}"
frequency_penalty="${LLAMA_FREQUENCY_PENALTY:-}"

detect_available_cores() {
    local cores

    if command -v nproc >/dev/null 2>&1; then
        cores="$(nproc)"
    elif command -v getconf >/dev/null 2>&1; then
        cores="$(getconf _NPROCESSORS_ONLN)"
    else
        cores=1
    fi

    if ! [[ "${cores}" =~ ^[0-9]+$ ]] || (( cores < 1 )); then
        cores=1
    fi

    printf '%s\n' "${cores}"
}

case "${preset}" in
    latency)
        parallel="${LLAMA_PARALLEL:-2}"
        ctx_size="${LLAMA_CTX_SIZE:-131072}"
        batch_size="${LLAMA_BATCH_SIZE:-2048}"
        ubatch_size="${LLAMA_UBATCH_SIZE:-1024}"
        ;;
    throughput)
        parallel="${LLAMA_PARALLEL:-4}"
        ctx_size="${LLAMA_CTX_SIZE:-262144}"
        batch_size="${LLAMA_BATCH_SIZE:-4096}"
        ubatch_size="${LLAMA_UBATCH_SIZE:-2048}"
        ;;
    qwen35-coding)
        # Match Qwen3.5 coding guidance while keeping >=128K context per slot
        # under llama.cpp's split-KV server mode.
        parallel="${LLAMA_PARALLEL:-2}"
        ctx_size="${LLAMA_CTX_SIZE:-262144}"
        batch_size="${LLAMA_BATCH_SIZE:-4096}"
        ubatch_size="${LLAMA_UBATCH_SIZE:-1024}"
        flash_attn="${LLAMA_FLASH_ATTN:-on}"
        reasoning="${LLAMA_REASONING:-on}"
        reasoning_format="${LLAMA_REASONING_FORMAT:-deepseek}"
        temp="${LLAMA_TEMP:-0.6}"
        top_p="${LLAMA_TOP_P:-0.95}"
        top_k="${LLAMA_TOP_K:-20}"
        min_p="${LLAMA_MIN_P:-0.00}"
        repeat_penalty="${LLAMA_REPEAT_PENALTY:-1.0}"
        presence_penalty="${LLAMA_PRESENCE_PENALTY:-0.0}"
        frequency_penalty="${LLAMA_FREQUENCY_PENALTY:-0.0}"
        ;;
    qwen35-coding-repro)
        # Reproducibility-oriented Qwen3.5 coding preset:
        # keep thinking enabled, default to greedy decoding and a fixed seed,
        # and reserve 256K context per slot across 4 parallel slots in split-KV mode.
        parallel="${LLAMA_PARALLEL:-4}"
        ctx_size="${LLAMA_CTX_SIZE:-1048576}"
        batch_size="${LLAMA_BATCH_SIZE:-4096}"
        ubatch_size="${LLAMA_UBATCH_SIZE:-1024}"
        flash_attn="${LLAMA_FLASH_ATTN:-on}"
        seed="${LLAMA_SEED:-42}"
        reasoning="${LLAMA_REASONING:-on}"
        reasoning_format="${LLAMA_REASONING_FORMAT:-deepseek}"
        temp="${LLAMA_TEMP:-0.0}"
        top_p="${LLAMA_TOP_P:-1.0}"
        top_k="${LLAMA_TOP_K:-1}"
        min_p="${LLAMA_MIN_P:-0.00}"
        repeat_penalty="${LLAMA_REPEAT_PENALTY:-1.0}"
        presence_penalty="${LLAMA_PRESENCE_PENALTY:-0.0}"
        frequency_penalty="${LLAMA_FREQUENCY_PENALTY:-0.0}"
        ;;
    qwen35-instruct)
        # Direct-response mode for faster tool-heavy loops with Qwen3.5.
        parallel="${LLAMA_PARALLEL:-2}"
        ctx_size="${LLAMA_CTX_SIZE:-262144}"
        batch_size="${LLAMA_BATCH_SIZE:-4096}"
        ubatch_size="${LLAMA_UBATCH_SIZE:-1024}"
        flash_attn="${LLAMA_FLASH_ATTN:-on}"
        reasoning="${LLAMA_REASONING:-off}"
        temp="${LLAMA_TEMP:-0.7}"
        top_p="${LLAMA_TOP_P:-0.8}"
        top_k="${LLAMA_TOP_K:-20}"
        min_p="${LLAMA_MIN_P:-0.00}"
        repeat_penalty="${LLAMA_REPEAT_PENALTY:-1.0}"
        presence_penalty="${LLAMA_PRESENCE_PENALTY:-1.5}"
        frequency_penalty="${LLAMA_FREQUENCY_PENALTY:-0.0}"
        ;;
    *)
        echo "Unknown LLAMA_PRESET='${preset}'. Use 'latency', 'throughput', 'qwen35-coding', 'qwen35-coding-repro', or 'qwen35-instruct'." >&2
        exit 2
        ;;
esac

available_cores="$(detect_available_cores)"
default_threads=$(( available_cores / parallel ))
if (( default_threads < 1 )); then
    default_threads=1
fi

threads="${LLAMA_THREADS:-${default_threads}}"
threads_batch="${LLAMA_THREADS_BATCH:-${available_cores}}"
threads_http="${LLAMA_THREADS_HTTP:-}"

default_args=(
    --host "${host}"
    --port "${port}"
    -ngl "${ngl}"
    --flash-attn "${flash_attn}"
    --parallel "${parallel}"
    --ctx-size "${ctx_size}"
    -b "${batch_size}"
    -ub "${ubatch_size}"
    -ctk "${cache_type_k}"
    -ctv "${cache_type_v}"
    --jinja
    --no-context-shift
    --mlock
    --metrics
)

if [[ -n "${seed:-}" ]]; then
    default_args+=(--seed "${seed}")
fi
if [[ -n "${reasoning:-}" ]]; then
    default_args+=(--reasoning "${reasoning}")
fi
if [[ -n "${reasoning_format:-}" ]]; then
    default_args+=(--reasoning-format "${reasoning_format}")
fi
if [[ -n "${temp:-}" ]]; then
    default_args+=(--temp "${temp}")
fi
if [[ -n "${top_p:-}" ]]; then
    default_args+=(--top-p "${top_p}")
fi
if [[ -n "${top_k:-}" ]]; then
    default_args+=(--top-k "${top_k}")
fi
if [[ -n "${min_p:-}" ]]; then
    default_args+=(--min-p "${min_p}")
fi
if [[ -n "${repeat_penalty:-}" ]]; then
    default_args+=(--repeat-penalty "${repeat_penalty}")
fi
if [[ -n "${presence_penalty:-}" ]]; then
    default_args+=(--presence-penalty "${presence_penalty}")
fi
if [[ -n "${frequency_penalty:-}" ]]; then
    default_args+=(--frequency-penalty "${frequency_penalty}")
fi

# Runtime thread defaults:
# - generation threads scale with visible CPU cores / parallel slots
# - batch threads use all visible CPU cores
# Explicit env vars still take precedence per host if needed.
default_args+=(-t "${threads}")
default_args+=(--threads-batch "${threads_batch}")
if [[ "${swa_full}" == "1" ]]; then
    default_args+=(--swa-full)
fi
if [[ -n "${threads_http}" ]]; then
    default_args+=(--threads-http "${threads_http}")
fi

# Optional compatibility switch for memory mapping behavior.
if [[ "${LLAMA_NO_MMAP:-0}" == "1" ]]; then
    default_args+=(--no-mmap)
fi

# User-provided args come last so they can override defaults.
exec llama-server "${default_args[@]}" "$@"
