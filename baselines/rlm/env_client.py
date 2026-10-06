"""RLM client shim executed inside an isolated env.

Monkey-patch contract
~~~~~~~~~~~~~~~~~~~~~
This shim patches upstream module globals to inject tracking clients.
The ``_PATCH_POINTS`` tuple enumerates every module attribute that is
replaced at runtime, and ``_COMPATIBLE_WITH`` records the pinned
upstream commit hash that these patches were last validated against.

Any change to the upstream module layout (renames, moved symbols) must
update both constants and re-run ``setup_repos.sh`` smoke checks.
"""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.openai_compat import (  # noqa: E402
    OpenAICompatRequestMeta,
    OpenAICompatResponseError,
    apply_compat_request_profile,
    build_request_validation_retry_plan,
    compat_config_from_extra,
    env_client_api_key,
    normalize_chat_completion_response,
    request_validation_compat_error_from_exception,
    require_final_content,
)

# ---------------------------------------------------------------------------
# Monkey-patch contract: these are the upstream symbols we replace.
# Tests assert that these constants stay in sync with the actual patches
# applied in ``_install_tracking()``.
# ---------------------------------------------------------------------------
_PATCH_POINTS: tuple[str, ...] = (
    "rlm.clients.get_client",
    "rlm.core.rlm.get_client",
    "RLM._default_answer",
)

#: Pinned upstream commit that these patches were last validated against.
#: Update this literal only after revalidating the shim against a new repo pin.
_COMPATIBLE_WITH: str = "rlm@4eba6ec7f2f0d5bec94d1f507ffcd79ad4ee7015"

_REPAIR_DEFAULT_ANSWER_PROMPT = (
    "Final turn. Return ONLY the full corrected contents of buggy.py as valid Python. "
    "Do not include markdown fences, FINAL(...), FINAL_VAR(...), or explanation."
)
_REPAIR_FINAL_ANSWER_OVERRIDE = ""
_TOOL_LEAKAGE_PATTERN = re.compile(
    r"\b(read_file|list_files|write_buggy_py|run_case_tests|llm_query|rlm_query)\s*\(",
)
_CONTEXT_LEAKAGE_PATTERN = re.compile(r"\bprint\s*\(\s*context\s*\)")


def _repair_default_answer(
    self: object,
    message_history: list[dict[str, Any]],
    lm_handler: Any,
) -> str:
    del self
    if _REPAIR_FINAL_ANSWER_OVERRIDE:
        return _REPAIR_FINAL_ANSWER_OVERRIDE
    return str(
        lm_handler.completion(
            [
                *list(message_history),
                {"role": "user", "content": _REPAIR_DEFAULT_ANSWER_PROMPT},
            ]
        )
        or ""
    )


def _prompt_char_count(prompt: str | list[dict[str, Any]]) -> int:
    if isinstance(prompt, str):
        return len(prompt)
    total = 0
    for item in prompt:
        content = item.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(str(part)) for part in content)
        else:
            total += len(str(content or ""))
    return total


def _truncate(text: str, *, max_chars: int) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."


def _resolve_workspace_path(workspace_dir: Path, requested_path: str) -> Path:
    normalized = str(requested_path or "").strip()
    if not normalized:
        raise ValueError("path is required")
    candidate = (workspace_dir / normalized).resolve()
    try:
        candidate.relative_to(workspace_dir)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {requested_path}") from exc
    return candidate


def _coerce_output(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        return ""
    split = urlsplit(normalized)
    if split.path not in {"", "/"}:
        return normalized
    return normalized + "/v1"


def _render_test_result(
    *, returncode: int, stdout: str, stderr: str, timed_out: bool, timeout_sec: int
) -> str:
    parts: list[str] = []
    if timed_out:
        parts.append(f"Timed out after {timeout_sec} seconds.")
    parts.append(f"Exit code: {returncode}")
    stdout_tail = _truncate(stdout.strip(), max_chars=4_000).strip()
    stderr_tail = _truncate(stderr.strip(), max_chars=4_000).strip()
    if stdout_tail:
        parts.append(f"stdout:\n{stdout_tail}")
    if stderr_tail:
        parts.append(f"stderr:\n{stderr_tail}")
    return "\n\n".join(part for part in parts if part)


def _rejected_write_reason(content: str) -> str:
    stripped = str(content or "").strip()
    if not stripped:
        return "empty_buggy_py"
    if _TOOL_LEAKAGE_PATTERN.search(stripped):
        return "tool_reference_leakage"
    if _CONTEXT_LEAKAGE_PATTERN.search(stripped):
        return "context_reference_leakage"
    return ""


def _import_smoke_failure(*, python_executable: str, workspace_dir: Path) -> str:
    try:
        completed = subprocess.run(
            [
                python_executable,
                "-c",
                (
                    "import importlib.util, pathlib\n"
                    "path = pathlib.Path('buggy.py')\n"
                    "spec = importlib.util.spec_from_file_location('buggy', str(path))\n"
                    "module = importlib.util.module_from_spec(spec)\n"
                    "assert spec is not None and spec.loader is not None\n"
                    "spec.loader.exec_module(module)\n"
                ),
            ],
            capture_output=True,
            text=True,
            cwd=str(workspace_dir),
            timeout=10,
            env=os.environ.copy(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "import smoke timed out"
    if completed.returncode == 0:
        return ""
    detail = "\n".join(
        part.strip()
        for part in (str(completed.stdout or ""), str(completed.stderr or ""))
        if str(part or "").strip()
    )
    return _truncate(detail or f"exit code {completed.returncode}", max_chars=800)


def _usage_summary_from_clients(created_clients: list[object]) -> dict[str, Any]:
    aggregated: dict[str, dict[str, float]] = {}
    total_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    any_cost = False
    for client in created_clients:
        get_usage_summary = getattr(client, "get_usage_summary", None)
        if not callable(get_usage_summary):
            continue
        summary = get_usage_summary()
        to_dict = getattr(summary, "to_dict", None)
        raw_payload = to_dict() if callable(to_dict) else summary
        if not isinstance(raw_payload, dict):
            continue
        payload = cast(dict[str, Any], raw_payload)
        model_rows = payload.get("model_usage_summaries")
        if not isinstance(model_rows, dict):
            continue
        for model, row in model_rows.items():
            if not isinstance(row, dict):
                continue
            model_key = str(model)
            target = aggregated.setdefault(
                model_key,
                {
                    "total_calls": 0.0,
                    "total_input_tokens": 0.0,
                    "total_output_tokens": 0.0,
                    "total_cost": 0.0,
                },
            )
            calls = int(row.get("total_calls", 0) or 0)
            input_tokens = int(row.get("total_input_tokens", 0) or 0)
            output_tokens = int(row.get("total_output_tokens", 0) or 0)
            target["total_calls"] += calls
            target["total_input_tokens"] += input_tokens
            target["total_output_tokens"] += output_tokens
            total_calls += calls
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            cost = row.get("total_cost")
            if isinstance(cost, (int, float)) and float(cost) > 0.0:
                target["total_cost"] += float(cost)
                total_cost += float(cost)
                any_cost = True

    model_usage = {
        model: {
            "total_calls": int(values["total_calls"]),
            "total_input_tokens": int(values["total_input_tokens"]),
            "total_output_tokens": int(values["total_output_tokens"]),
            **(
                {"total_cost": float(values["total_cost"])}
                if float(values["total_cost"]) > 0.0
                else {}
            ),
        }
        for model, values in aggregated.items()
    }
    payload: dict[str, Any] = {
        "model_usage_summaries": model_usage,
        "total_calls": total_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
    if any_cost:
        payload["total_cost"] = total_cost
    return payload


def _char_summary_from_clients(created_clients: list[object]) -> dict[str, Any]:
    model_char_summaries: dict[str, dict[str, int]] = {}
    total_calls = 0
    total_input_chars = 0
    total_output_chars = 0
    for client in created_clients:
        input_chars = getattr(client, "model_input_chars", {})
        output_chars = getattr(client, "model_output_chars", {})
        char_calls = getattr(client, "model_char_calls", {})
        if not isinstance(input_chars, dict) or not isinstance(output_chars, dict):
            continue
        for model in set(input_chars) | set(output_chars) | set(char_calls):
            calls = int(char_calls.get(model, 0) or 0)
            in_chars = int(input_chars.get(model, 0) or 0)
            out_chars = int(output_chars.get(model, 0) or 0)
            model_char_summaries[str(model)] = {
                "total_calls": calls,
                "total_input_chars": in_chars,
                "total_output_chars": out_chars,
            }
            total_calls += calls
            total_input_chars += in_chars
            total_output_chars += out_chars
    return {
        "model_char_summaries": model_char_summaries,
        "total_calls": total_calls,
        "total_input_chars": total_input_chars,
        "total_output_chars": total_output_chars,
    }


def _call_summary_from_clients(created_clients: list[object]) -> dict[str, Any]:
    call_records: list[dict[str, Any]] = []
    compat_recovery_events: list[dict[str, Any]] = []
    finish_reason_counts: dict[str, int] = defaultdict(int)
    length_finish_count = 0
    reasoning_fallback_call_count = 0
    empty_reasoning_only_call_count = 0
    for client in created_clients:
        raw_records = getattr(client, "call_records", None)
        if isinstance(raw_records, list):
            for raw_record in raw_records:
                if not isinstance(raw_record, dict):
                    continue
                record = {
                    "call_index": int(raw_record.get("call_index") or 0),
                    "model": str(raw_record.get("model") or ""),
                    "finish_reason": str(raw_record.get("finish_reason") or ""),
                    "input_chars": int(raw_record.get("input_chars") or 0),
                    "output_chars": int(raw_record.get("output_chars") or 0),
                    "compat_profile_applied": str(raw_record.get("compat_profile_applied") or ""),
                    "enable_thinking": bool(raw_record.get("enable_thinking", False)),
                    "content_present": bool(raw_record.get("content_present", False)),
                    "reasoning_present": bool(raw_record.get("reasoning_present", False)),
                    "empty_content_with_reasoning": bool(
                        raw_record.get("empty_content_with_reasoning", False)
                    ),
                    "used_reasoning_fallback": bool(
                        raw_record.get("used_reasoning_fallback", False)
                    ),
                    "request_validation_retry_applied": bool(
                        raw_record.get("request_validation_retry_applied", False)
                    ),
                }
                call_records.append(record)
                finish_reason = record["finish_reason"]
                if finish_reason:
                    finish_reason_counts[finish_reason] += 1
                    if finish_reason == "length":
                        length_finish_count += 1
                if record["used_reasoning_fallback"]:
                    reasoning_fallback_call_count += 1
                if record["empty_content_with_reasoning"]:
                    empty_reasoning_only_call_count += 1
        raw_events = getattr(client, "compat_recovery_events", None)
        if isinstance(raw_events, list):
            compat_recovery_events.extend(event for event in raw_events if isinstance(event, dict))
    return {
        "call_records": call_records,
        "call_finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "call_length_finish_count": length_finish_count,
        "reasoning_fallback_call_count": reasoning_fallback_call_count,
        "empty_reasoning_only_call_count": empty_reasoning_only_call_count,
        "compat_recovery_count": len(compat_recovery_events),
        "compat_recovery_events": compat_recovery_events,
    }


def _install_tracking(
    *,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    compat_config,
    base_url: str,
) -> tuple[type[object], type[object], list[object]]:
    import rlm.clients as clients_mod  # type: ignore[import-not-found]
    import rlm.clients.openai as openai_mod  # type: ignore[import-not-found]
    import rlm.core.rlm as rlm_core_mod  # type: ignore[import-not-found]
    from rlm.logger import RLMLogger  # type: ignore[import-not-found]

    created_clients: list[object] = []
    original_get_client = clients_mod.get_client
    original_openai_client = openai_mod.OpenAIClient
    prime_base_url = getattr(openai_mod, "DEFAULT_PRIME_INTELLECT_BASE_URL", "")

    class TrackingOpenAIClient(original_openai_client):  # type: ignore[misc,valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.model_input_chars: dict[str, int] = defaultdict(int)
            self.model_output_chars: dict[str, int] = defaultdict(int)
            self.model_char_calls: dict[str, int] = defaultdict(int)
            self.compat_recovery_events: list[dict[str, Any]] = []
            self.call_records: list[dict[str, Any]] = []
            self._logical_call_index = 0

        def _record_call(
            self,
            *,
            model: str,
            input_chars: int,
            output_chars: int,
            normalized,
            request_meta: OpenAICompatRequestMeta,
            retry_applied: bool,
        ) -> None:
            self._logical_call_index += 1
            self.call_records.append(
                {
                    "call_index": self._logical_call_index,
                    "model": model,
                    "finish_reason": normalized.finish_reason,
                    "input_chars": input_chars,
                    "output_chars": output_chars,
                    "compat_profile_applied": request_meta.compat_profile_applied,
                    "enable_thinking": request_meta.enable_thinking,
                    "content_present": normalized.content_present,
                    "reasoning_present": normalized.reasoning_present,
                    "empty_content_with_reasoning": normalized.empty_content_with_reasoning,
                    "used_reasoning_fallback": normalized.used_reasoning_fallback,
                    "request_validation_retry_applied": retry_applied,
                }
            )

        def _raise_known_request_validation_error(
            self,
            exc: BaseException,
            *,
            request_meta: OpenAICompatRequestMeta,
        ) -> None:
            compat_exc = self._known_request_validation_error(exc, request_meta=request_meta)
            if compat_exc is not None:
                raise compat_exc from exc

        def _known_request_validation_error(
            self,
            exc: BaseException,
            *,
            request_meta: OpenAICompatRequestMeta,
        ) -> OpenAICompatResponseError | None:
            return request_validation_compat_error_from_exception(
                exc,
                request_meta=request_meta,
                base_url=base_url,
            )

        def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

            resolved_model = model or self.model_name
            if not resolved_model:
                raise ValueError("Model name is required for OpenAI client.")

            base_request_kwargs: dict[str, Any] = {
                "model": resolved_model,
                "messages": messages,
            }
            if self.client.base_url == prime_base_url:
                base_request_kwargs["extra_body"] = {"usage": {"include": True}}
            if temperature is not None:
                base_request_kwargs["temperature"] = temperature
            if top_p is not None:
                base_request_kwargs["top_p"] = top_p
            if max_tokens is not None:
                base_request_kwargs["max_tokens"] = max_tokens
            request_kwargs, request_meta = apply_compat_request_profile(
                base_request_kwargs,
                compat_config=compat_config,
                base_url=base_url,
            )

            input_chars = _prompt_char_count(prompt)
            retry_applied = False
            try:
                response = self.client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                retry_plan = build_request_validation_retry_plan(
                    base_request_kwargs,
                    compat_config=compat_config,
                    request_meta=request_meta,
                    exc=exc,
                    base_url=base_url,
                )
                if retry_plan is not None:
                    original_compat_exc = self._known_request_validation_error(
                        exc,
                        request_meta=request_meta,
                    )
                    try:
                        response = self.client.chat.completions.create(**retry_plan.request_kwargs)
                    except Exception as retry_exc:
                        retry_compat_exc = self._known_request_validation_error(
                            retry_exc,
                            request_meta=retry_plan.request_meta,
                        )
                        if retry_compat_exc is not None:
                            raise retry_compat_exc from retry_exc
                        if original_compat_exc is not None:
                            raise original_compat_exc from exc
                        raise
                    else:
                        self.compat_recovery_events.append(
                            {
                                "reason": retry_plan.reason,
                                "model": resolved_model,
                                "compat_profile_before": request_meta.compat_profile_applied,
                                "compat_profile_after": retry_plan.request_meta.compat_profile_applied,
                                "enable_thinking_before": request_meta.enable_thinking,
                                "enable_thinking_after": retry_plan.request_meta.enable_thinking,
                            }
                        )
                        retry_applied = True
                        request_meta = retry_plan.request_meta
                else:
                    self._raise_known_request_validation_error(exc, request_meta=request_meta)
                    raise
            self._track_cost(response, resolved_model)
            normalized = normalize_chat_completion_response(
                response,
                compat_config=compat_config,
                base_url=base_url,
                request_meta=request_meta,
            )
            content = require_final_content(normalized, base_url=base_url)
            self.model_char_calls[resolved_model] += 1
            self.model_input_chars[resolved_model] += input_chars
            self.model_output_chars[resolved_model] += len(content)
            self._record_call(
                model=resolved_model,
                input_chars=input_chars,
                output_chars=len(content),
                normalized=normalized,
                request_meta=request_meta,
                retry_applied=retry_applied,
            )
            return content

        async def acompletion(
            self, prompt: str | list[dict[str, Any]], model: str | None = None
        ) -> str:
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

            resolved_model = model or self.model_name
            if not resolved_model:
                raise ValueError("Model name is required for OpenAI client.")

            base_request_kwargs: dict[str, Any] = {
                "model": resolved_model,
                "messages": messages,
            }
            if self.client.base_url == prime_base_url:
                base_request_kwargs["extra_body"] = {"usage": {"include": True}}
            if temperature is not None:
                base_request_kwargs["temperature"] = temperature
            if top_p is not None:
                base_request_kwargs["top_p"] = top_p
            if max_tokens is not None:
                base_request_kwargs["max_tokens"] = max_tokens
            request_kwargs, request_meta = apply_compat_request_profile(
                base_request_kwargs,
                compat_config=compat_config,
                base_url=base_url,
            )

            input_chars = _prompt_char_count(prompt)
            retry_applied = False
            try:
                response = await self.async_client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                retry_plan = build_request_validation_retry_plan(
                    base_request_kwargs,
                    compat_config=compat_config,
                    request_meta=request_meta,
                    exc=exc,
                    base_url=base_url,
                )
                if retry_plan is not None:
                    original_compat_exc = self._known_request_validation_error(
                        exc,
                        request_meta=request_meta,
                    )
                    try:
                        response = await self.async_client.chat.completions.create(
                            **retry_plan.request_kwargs
                        )
                    except Exception as retry_exc:
                        retry_compat_exc = self._known_request_validation_error(
                            retry_exc,
                            request_meta=retry_plan.request_meta,
                        )
                        if retry_compat_exc is not None:
                            raise retry_compat_exc from retry_exc
                        if original_compat_exc is not None:
                            raise original_compat_exc from exc
                        raise
                    else:
                        self.compat_recovery_events.append(
                            {
                                "reason": retry_plan.reason,
                                "model": resolved_model,
                                "compat_profile_before": request_meta.compat_profile_applied,
                                "compat_profile_after": retry_plan.request_meta.compat_profile_applied,
                                "enable_thinking_before": request_meta.enable_thinking,
                                "enable_thinking_after": retry_plan.request_meta.enable_thinking,
                            }
                        )
                        retry_applied = True
                        request_meta = retry_plan.request_meta
                else:
                    self._raise_known_request_validation_error(exc, request_meta=request_meta)
                    raise
            self._track_cost(response, resolved_model)
            normalized = normalize_chat_completion_response(
                response,
                compat_config=compat_config,
                base_url=base_url,
                request_meta=request_meta,
            )
            content = require_final_content(normalized, base_url=base_url)
            self.model_char_calls[resolved_model] += 1
            self.model_input_chars[resolved_model] += input_chars
            self.model_output_chars[resolved_model] += len(content)
            self._record_call(
                model=resolved_model,
                input_chars=input_chars,
                output_chars=len(content),
                normalized=normalized,
                request_meta=request_meta,
                retry_applied=retry_applied,
            )
            return content

    def tracked_get_client(backend: object, backend_kwargs: object) -> object:
        normalized_backend = str(backend or "")
        if normalized_backend in {"openai", "vllm", "openrouter", "vercel"}:
            client = TrackingOpenAIClient(**dict(cast(dict[str, object], backend_kwargs)))
            created_clients.append(client)
            return client
        client = original_get_client(backend, cast(dict[str, Any], backend_kwargs))
        created_clients.append(client)
        return client

    clients_mod.get_client = tracked_get_client
    rlm_core_mod.get_client = tracked_get_client
    return rlm_core_mod.RLM, RLMLogger, created_clients


def _write_buggy_candidate(
    *,
    workspace_dir: Path,
    content: str,
    test_python: str,
    rejections: list[dict[str, Any]],
    import_smoke_failures: list[dict[str, Any]],
) -> str:
    """Write a candidate ``buggy.py``; the import smoke test is telemetry only (BAS-13).

    RLM used to be the only arm whose candidates were rolled back when importing
    ``buggy.py`` failed or took more than 10 s, and 830 cases call code at import time.
    """
    target = workspace_dir / "buggy.py"
    text = str(content or "")
    previous_text = target.read_text(encoding="utf-8", errors="replace")
    rejection_reason = _rejected_write_reason(text)
    if rejection_reason:
        rejections.append({"reason": rejection_reason, "detail": _truncate(text, max_chars=200)})
        return f"Rejected write_buggy_py(content): {rejection_reason}"
    target.write_text(text, encoding="utf-8")
    import_failure = _import_smoke_failure(
        python_executable=test_python, workspace_dir=workspace_dir
    )
    if import_failure:
        import_smoke_failures.append(
            {"detail": import_failure, "previous_chars": len(previous_text)}
        )
        return (
            f"Wrote {len(text)} characters to buggy.py "
            f"(note: importing it failed: {_truncate(import_failure, max_chars=300)})"
        )
    return f"Wrote {len(text)} characters to buggy.py"


def main(argv: list[str]) -> int:
    global _REPAIR_FINAL_ANSWER_OVERRIDE
    if len(argv) != 3:
        raise SystemExit("usage: env_client.py <request.json> <response.json>")

    request_path = Path(argv[1])
    response_path = Path(argv[2])
    payload = json.loads(request_path.read_text(encoding="utf-8"))

    repo_dir = Path(str(payload.get("repo_dir") or "")).expanduser().resolve()
    if not repo_dir.is_dir():
        raise RuntimeError(f"rlm repo_dir not found: {repo_dir}")
    sys.path.insert(0, str(repo_dir))

    temperature = float(payload["temperature"]) if payload.get("temperature") is not None else None
    top_p = float(payload["top_p"]) if payload.get("top_p") is not None else None
    max_tokens = int(payload["max_tokens"]) if payload.get("max_tokens") is not None else None
    compat_config = compat_config_from_extra(cast(dict[str, Any], payload))
    compat_base_url = _normalize_openai_base_url(str(payload.get("base_url") or ""))
    RLM, RLMLogger, created_clients = _install_tracking(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        compat_config=compat_config,
        base_url=compat_base_url,
    )
    RLM._default_answer = _repair_default_answer
    from rlm.utils.exceptions import (  # type: ignore[import-not-found]
        BudgetExceededError,
        CancellationError,
        ErrorThresholdExceededError,
        TimeoutExceededError,
        TokenLimitExceededError,
    )

    workspace_dir = Path(str(payload.get("workspace_dir") or "")).expanduser().resolve()
    if not workspace_dir.is_dir():
        raise RuntimeError(f"workspace_dir not found: {workspace_dir}")
    original_buggy_code = (workspace_dir / "buggy.py").read_text(encoding="utf-8")
    _REPAIR_FINAL_ANSWER_OVERRIDE = ""
    test_python = str(payload.get("test_python") or sys.executable)
    python_args_raw = payload.get("python_args")
    python_args = (
        [str(item) for item in python_args_raw if isinstance(item, str)]
        if isinstance(python_args_raw, list)
        else []
    )
    test_timeout_sec = max(1, int(payload.get("test_timeout_sec") or 30))

    tool_call_counts: dict[str, int] = defaultdict(int)
    last_test_result: dict[str, Any] = {}
    subcall_count = 0
    max_depth_reached = 0
    workspace_write_rejections: list[dict[str, Any]] = []
    workspace_import_smoke_failures: list[dict[str, Any]] = []
    workspace_tests_passed = False

    def _record_tool(name: str) -> None:
        tool_call_counts[name] += 1

    def list_files() -> str:
        _record_tool("list_files")
        files = [
            path.relative_to(workspace_dir).as_posix()
            for path in sorted(workspace_dir.rglob("*"))
            if path.is_file()
        ]
        return "\n".join(files) if files else "(workspace is empty)"

    def read_file(path: str, max_chars: int = 20_000) -> str:
        _record_tool("read_file")
        target = _resolve_workspace_path(workspace_dir, path)
        if not target.is_file():
            raise FileNotFoundError(f"workspace file not found: {path}")
        return _truncate(target.read_text(encoding="utf-8", errors="replace"), max_chars=max_chars)

    def write_buggy_py(content: str) -> str:
        _record_tool("write_buggy_py")
        return _write_buggy_candidate(
            workspace_dir=workspace_dir,
            content=content,
            test_python=test_python,
            rejections=workspace_write_rejections,
            import_smoke_failures=workspace_import_smoke_failures,
        )

    def run_case_tests() -> str:
        global _REPAIR_FINAL_ANSWER_OVERRIDE
        nonlocal last_test_result, workspace_tests_passed
        _record_tool("run_case_tests")
        if not python_args:
            last_test_result = {
                "returncode": 2,
                "stdout": "",
                "stderr": "python_args missing for case tests",
                "timed_out": False,
                "timeout_sec": test_timeout_sec,
                "duration_sec": 0.0,
            }
            return _render_test_result(
                returncode=2,
                stdout="",
                stderr="python_args missing for case tests",
                timed_out=False,
                timeout_sec=test_timeout_sec,
            )
        started_at = time.perf_counter()
        try:
            completed = subprocess.run(
                [test_python, *python_args],
                capture_output=True,
                text=True,
                cwd=str(workspace_dir),
                timeout=test_timeout_sec,
                env=os.environ.copy(),
            )
            duration_sec = time.perf_counter() - started_at
            last_test_result = {
                "returncode": int(completed.returncode),
                "stdout": str(completed.stdout or ""),
                "stderr": str(completed.stderr or ""),
                "timed_out": False,
                "timeout_sec": test_timeout_sec,
                "duration_sec": duration_sec,
            }
            if completed.returncode == 0:
                workspace_tests_passed = True
                _REPAIR_FINAL_ANSWER_OVERRIDE = (workspace_dir / "buggy.py").read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            return _render_test_result(
                returncode=int(completed.returncode),
                stdout=str(completed.stdout or ""),
                stderr=str(completed.stderr or ""),
                timed_out=False,
                timeout_sec=test_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            duration_sec = time.perf_counter() - started_at
            stdout = _coerce_output(exc.stdout)
            stderr = _coerce_output(exc.stderr)
            last_test_result = {
                "returncode": 124,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": True,
                "timeout_sec": test_timeout_sec,
                "duration_sec": duration_sec,
            }
            return _render_test_result(
                returncode=124,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
                timeout_sec=test_timeout_sec,
            )

    custom_tools: dict[str, Any] = {
        "list_files": {
            "tool": list_files,
            "description": "List files in the copied case workspace. Read-only.",
        },
        "read_file": {
            "tool": read_file,
            "description": "Read a UTF-8 file from the copied case workspace by relative path.",
        },
        "write_buggy_py": {
            "tool": write_buggy_py,
            "description": "Overwrite buggy.py with the provided full-file contents.",
        },
        "run_case_tests": {
            "tool": run_case_tests,
            "description": "Run the case's test command in the copied workspace and return bounded output.",
        },
    }

    def _on_subcall_start(depth: int, model: str, prompt_preview: str) -> None:
        del model, prompt_preview
        nonlocal subcall_count, max_depth_reached
        subcall_count += 1
        max_depth_reached = max(max_depth_reached, int(depth))

    logger = RLMLogger()
    backend_kwargs: dict[str, Any] = {
        "model_name": str(payload.get("model") or ""),
        "api_key": env_client_api_key(payload),
        "base_url": compat_base_url or None,
    }
    api_timeout_sec = payload.get("api_timeout_sec")
    if isinstance(api_timeout_sec, (int, float)) and float(api_timeout_sec) > 0.0:
        backend_kwargs["timeout"] = float(api_timeout_sec)
    response_text = ""
    root_model = str(payload.get("model") or "")
    stop_reason = "exception"
    error = ""
    compat_message = ""
    compat_meta: dict[str, Any] = {}
    started_at = time.perf_counter()

    rlm = RLM(
        backend="openai",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=max(1, int(payload.get("max_depth") or 2)),
        max_iterations=max(1, int(payload.get("max_iterations") or 12)),
        max_timeout=float(payload.get("max_timeout_sec") or 120.0),
        max_tokens=(
            int(payload["max_total_tokens"])
            if payload.get("max_total_tokens") is not None
            else None
        ),
        max_errors=(int(payload["max_errors"]) if payload.get("max_errors") is not None else None),
        verbose=bool(payload.get("verbose", False)),
        compaction=bool(payload.get("compaction", False)),
        compaction_threshold_pct=float(payload.get("compaction_threshold_pct") or 0.85),
        logger=logger,
        custom_tools=custom_tools,
        custom_sub_tools={},
        on_subcall_start=_on_subcall_start,
    )

    try:
        completion = rlm.completion(
            str(payload.get("prompt_context") or ""),
            root_prompt=str(payload.get("root_prompt") or "") or None,
        )
        response_text = str(completion.response or "")
        root_model = str(completion.root_model or root_model)
        trajectory = (
            completion.metadata
            if isinstance(completion.metadata, dict)
            else logger.get_trajectory()
        )
        iterations = trajectory.get("iterations", []) if isinstance(trajectory, dict) else []
        if isinstance(iterations, list) and iterations:
            last_iteration = iterations[-1]
            if (
                isinstance(last_iteration, dict)
                and str(last_iteration.get("final_answer") or "").strip()
            ):
                stop_reason = "final_answer"
            elif len(iterations) >= max(1, int(payload.get("max_iterations") or 12)):
                stop_reason = "iteration_budget_exhausted"
            else:
                stop_reason = "completed"
        else:
            stop_reason = "completed"
    except TimeoutExceededError as exc:
        response_text = str(exc.partial_answer or "")
        stop_reason = "timeout_exceeded"
        error = str(exc)
        trajectory = logger.get_trajectory()
    except TokenLimitExceededError as exc:
        response_text = str(exc.partial_answer or "")
        stop_reason = "token_limit_exceeded"
        error = str(exc)
        trajectory = logger.get_trajectory()
    except ErrorThresholdExceededError as exc:
        response_text = str(exc.partial_answer or "")
        stop_reason = "error_threshold_exceeded"
        error = str(exc)
        trajectory = logger.get_trajectory()
    except BudgetExceededError as exc:
        response_text = ""
        stop_reason = "budget_exceeded"
        error = str(exc)
        trajectory = logger.get_trajectory()
    except CancellationError as exc:
        response_text = str(exc.partial_answer or "")
        stop_reason = "cancelled"
        error = str(exc)
        trajectory = logger.get_trajectory()
    except OpenAICompatResponseError as exc:
        response_text = ""
        stop_reason = "openai_compat_reasoning_only"
        error = "openai_compat_reasoning_only"
        trajectory = logger.get_trajectory()
        compat_message = str(exc)
        compat_meta = exc.meta
    except Exception as exc:
        response_text = ""
        stop_reason = "exception"
        error = f"{type(exc).__name__}: {exc}"
        trajectory = logger.get_trajectory()

    if not isinstance(trajectory, dict):
        trajectory = logger.get_trajectory()
    iterations = trajectory.get("iterations", []) if isinstance(trajectory, dict) else []
    iterations_completed = len(iterations) if isinstance(iterations, list) else 0

    usage_summary = _usage_summary_from_clients(created_clients)
    char_summary = _char_summary_from_clients(created_clients)
    call_summary = _call_summary_from_clients(created_clients)
    execution_time = max(0.0, time.perf_counter() - started_at)
    buggy_path = workspace_dir / "buggy.py"
    workspace_buggy_code = (
        buggy_path.read_text(encoding="utf-8", errors="replace") if buggy_path.is_file() else ""
    )
    response_path.write_text(
        json.dumps(
            {
                "root_model": root_model,
                "response": response_text,
                "execution_time": execution_time,
                "stop_reason": stop_reason,
                "error": error,
                "compat_message": compat_message,
                "compat_meta": compat_meta,
                "usage_summary": usage_summary,
                "char_summary": char_summary,
                "call_summary": call_summary,
                "trajectory_summary": {
                    "iterations_completed": iterations_completed,
                    "subcall_count": subcall_count,
                    "max_depth_reached": max_depth_reached,
                    "tool_call_counts": dict(tool_call_counts),
                    "test_run_count": int(tool_call_counts.get("run_case_tests", 0)),
                    "last_test_result": last_test_result,
                    "write_rejection_count": len(workspace_write_rejections),
                    "import_smoke_failure_count": len(workspace_import_smoke_failures),
                    "last_write_rejection": (
                        workspace_write_rejections[-1] if workspace_write_rejections else {}
                    ),
                    "workspace_tests_passed": workspace_tests_passed,
                },
                "workspace_buggy_code": workspace_buggy_code,
                "workspace_buggy_changed": workspace_buggy_code != original_buggy_code,
            }
        ),
        encoding="utf-8",
    )
    _REPAIR_FINAL_ANSWER_OVERRIDE = ""
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
