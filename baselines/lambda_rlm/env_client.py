"""Lambda-RLM client shim executed inside an isolated env.

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
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.openai_compat import (  # noqa: E402
    EnvClientGenerationSettings,
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
    "rlm.lambda_rlm.get_client",
    # Instance method patches on the LambdaRLM object:
    "LambdaRLM._parse_task_type",
    "LambdaRLM._plan",
)

#: Pinned upstream commit that these patches were last validated against.
#: Update this literal only after revalidating the shim against a new repo pin.
_COMPATIBLE_WITH: str = "lambda-RLM@c70c6db2cf8498eaf05fa6999dc41088827158f4"


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


def _install_tracking(
    repo_dir: Path,
    *,
    generation: EnvClientGenerationSettings | None = None,
    compat_config,
    base_url: str,
) -> tuple[type[object], list[object]]:
    import rlm.clients as clients_mod  # type: ignore[import-not-found]
    import rlm.clients.openai as openai_mod  # type: ignore[import-not-found]
    import rlm.lambda_rlm as lambda_mod  # type: ignore[import-not-found]

    settings = generation or EnvClientGenerationSettings()

    created_clients: list[object] = []
    original_get_client = clients_mod.get_client
    original_openai_client = openai_mod.OpenAIClient

    class TrackingOpenAIClient(original_openai_client):  # type: ignore[misc,valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            # D4 (BAS-11): the shared transport retry count, which upstream never sets.
            for attribute in ("client", "async_client"):
                existing = getattr(self, attribute, None)
                if existing is not None:
                    setattr(self, attribute, settings.apply_transport_retries(existing))
            self.model_input_chars: dict[str, int] = defaultdict(int)
            self.model_output_chars: dict[str, int] = defaultdict(int)
            self.model_char_calls: dict[str, int] = defaultdict(int)
            self.call_records: list[dict[str, Any]] = []
            self.compat_recovery_events: list[dict[str, Any]] = []
            self._logical_call_index = 0

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

        def _raise_known_request_validation_error(
            self,
            exc: BaseException,
            *,
            request_meta: OpenAICompatRequestMeta,
        ) -> None:
            compat_exc = self._known_request_validation_error(exc, request_meta=request_meta)
            if compat_exc is not None:
                raise compat_exc from exc

        def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
            resolved_model = model or self.model_name or "unknown"
            input_chars = _prompt_char_count(prompt)
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            logical_phase = "task_detection" if self._logical_call_index == 0 else "lambda_query"
            call_index = self._logical_call_index + 1
            self._logical_call_index = call_index
            base_request_kwargs = {
                "model": resolved_model,
                "messages": messages,
                **{
                    key: value
                    for key, value in getattr(self, "_completion_kwargs", {}).items()
                    if key != "stream"
                },
                # D4: the shared contract wins over whatever upstream kwargs carry.
                **settings.request_kwargs(),
            }
            request_kwargs, request_meta = apply_compat_request_profile(
                base_request_kwargs,
                compat_config=compat_config,
                base_url=base_url,
            )
            use_stream = bool(getattr(self, "_completion_kwargs", {}).get("stream", False))
            retry_applied = False
            if use_stream:
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                finish_reason = ""
                try:
                    stream = self.client.chat.completions.create(
                        **request_kwargs,
                        stream=True,
                    )
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
                            stream = self.client.chat.completions.create(
                                **retry_plan.request_kwargs,
                                stream=True,
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
                            retry_applied = True
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
                            request_meta = retry_plan.request_meta
                    else:
                        self._raise_known_request_validation_error(
                            exc,
                            request_meta=request_meta,
                        )
                        raise
                for chunk in stream:
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    choice0 = choices[0]
                    chunk_finish_reason = getattr(choice0, "finish_reason", None)
                    if isinstance(chunk_finish_reason, str) and chunk_finish_reason:
                        finish_reason = chunk_finish_reason
                    delta = getattr(choice0, "delta", None)
                    if delta is None:
                        continue
                    content = getattr(delta, "content", None)
                    if isinstance(content, str) and content:
                        content_parts.append(content)
                    reasoning = getattr(delta, "reasoning", None)
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_parts.append(reasoning)
                    reasoning_content = getattr(delta, "reasoning_content", None)
                    if isinstance(reasoning_content, str) and reasoning_content:
                        reasoning_parts.append(reasoning_content)
                response_text = "".join(content_parts).strip()
                reasoning_text = "".join(reasoning_parts).strip()
                content_present = bool(response_text)
                reasoning_present = bool(reasoning_text)
                empty_content_with_reasoning = bool(not content_present and reasoning_present)
                used_reasoning_fallback = bool(
                    empty_content_with_reasoning and compat_config.allow_reasoning_fallback
                )
                if empty_content_with_reasoning and not used_reasoning_fallback:
                    raise OpenAICompatResponseError(
                        "OpenAI-compatible streaming response contained reasoning but no final "
                        "answer in delta.content.",
                        meta={
                            **request_meta.to_dict(),
                            "content_present": False,
                            "reasoning_present": True,
                            "empty_content_with_reasoning": True,
                            "used_reasoning_fallback": False,
                            "reasoning_chars": len(reasoning_text),
                            "reasoning_content_chars": len(reasoning_text),
                            "final_text_chars": 0,
                            "finish_reason": finish_reason,
                        },
                    )
                response = reasoning_text if used_reasoning_fallback else response_text
            else:
                try:
                    raw_response = self.client.chat.completions.create(**request_kwargs)
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
                            raw_response = self.client.chat.completions.create(
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
                            retry_applied = True
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
                            request_meta = retry_plan.request_meta
                    else:
                        self._raise_known_request_validation_error(
                            exc,
                            request_meta=request_meta,
                        )
                        raise
                normalized = normalize_chat_completion_response(
                    raw_response,
                    compat_config=compat_config,
                    base_url=base_url,
                    request_meta=request_meta,
                )
                response = require_final_content(normalized, base_url=base_url)
                finish_reason = normalized.finish_reason
                content_present = normalized.content_present
                reasoning_present = normalized.reasoning_present
                empty_content_with_reasoning = normalized.empty_content_with_reasoning
                used_reasoning_fallback = normalized.used_reasoning_fallback
                track_cost = getattr(self, "_track_cost", None)
                if callable(track_cost):
                    track_cost(raw_response, resolved_model)
            self.model_char_calls[resolved_model] += 1
            self.model_input_chars[resolved_model] += input_chars
            self.model_output_chars[resolved_model] += len(str(response or ""))
            self.call_records.append(
                {
                    "call_index": call_index,
                    "logical_phase": logical_phase,
                    "model": resolved_model,
                    "finish_reason": finish_reason,
                    "input_chars": input_chars,
                    "output_chars": len(str(response or "")),
                    "compat_profile_applied": request_meta.compat_profile_applied,
                    "enable_thinking": request_meta.enable_thinking,
                    "content_present": content_present,
                    "reasoning_present": reasoning_present,
                    "empty_content_with_reasoning": empty_content_with_reasoning,
                    "used_reasoning_fallback": used_reasoning_fallback,
                    "request_validation_retry_applied": retry_applied,
                }
            )
            return response

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
    lambda_mod.get_client = tracked_get_client
    return lambda_mod.LambdaRLM, created_clients


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
    task_detection_call_count = 0
    repl_llm_query_call_count = 0
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
                    "logical_phase": str(raw_record.get("logical_phase") or ""),
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
                if record["logical_phase"] == "task_detection":
                    task_detection_call_count += 1
                elif record["logical_phase"] == "lambda_query":
                    repl_llm_query_call_count += 1
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
        "task_detection_call_count": task_detection_call_count,
        "repl_llm_query_call_count": repl_llm_query_call_count,
        "reasoning_fallback_call_count": reasoning_fallback_call_count,
        "empty_reasoning_only_call_count": empty_reasoning_only_call_count,
        "compat_recovery_count": len(compat_recovery_events),
        "compat_recovery_events": compat_recovery_events,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: env_client.py <request.json> <response.json>")

    request_path = Path(argv[1])
    response_path = Path(argv[2])
    payload = json.loads(request_path.read_text(encoding="utf-8"))

    repo_dir = Path(str(payload.get("repo_dir") or "")).expanduser().resolve()
    if not repo_dir.is_dir():
        raise RuntimeError(f"lambda-RLM repo_dir not found: {repo_dir}")
    sys.path.insert(0, str(repo_dir))
    compat_config = compat_config_from_extra(cast(dict[str, Any], payload))

    base_url = str(payload.get("base_url") or "")
    if base_url and not base_url.endswith("/v1"):
        base_url = f"{base_url.rstrip('/')}/v1"
    generation = EnvClientGenerationSettings.from_payload(cast(dict[str, Any], payload))
    LambdaRLM, created_clients = _install_tracking(
        repo_dir,
        generation=generation,
        compat_config=compat_config,
        base_url=base_url,
    )
    backend_kwargs: dict[str, object] = {
        "model_name": str(payload.get("model") or ""),
        "api_key": env_client_api_key(payload),
        "base_url": base_url or None,
    }
    api_timeout_sec = payload.get("api_timeout_sec")
    if isinstance(api_timeout_sec, (int, float)) and float(api_timeout_sec) > 0.0:
        backend_kwargs["timeout"] = float(api_timeout_sec)
    for key in ("temperature", "top_p", "max_tokens", "stream"):
        if key in payload and payload.get(key) is not None:
            backend_kwargs[key] = payload.get(key)
    rlm = LambdaRLM(
        backend="openai",
        backend_kwargs=backend_kwargs,
        context_window_chars=max(1, int(payload.get("context_window_chars") or 100_000)),
        accuracy_target=float(payload.get("accuracy_target") or 0.80),
        a_leaf=float(payload.get("a_leaf") or 0.95),
        a_compose=float(payload.get("a_compose") or 0.90),
        query=(str(payload.get("query") or "").strip() or None),
        verbose=bool(payload.get("verbose", False)),
    )
    original_parse_task_type = rlm._parse_task_type
    original_plan = rlm._plan
    tracked_task_type: str = ""
    tracked_plan: dict[str, Any] = {}

    # Optional forced task type — bypasses the LLM classification call.
    # Small models often misclassify code repair as "summarization" instead of
    # "qa", causing the pipeline to discard the repair question and produce
    # prose summaries instead of code.
    force_task_type_str = str(payload.get("force_task_type") or "").strip().lower()
    forced_task_type: object | None = None
    if force_task_type_str:
        from rlm.lambda_rlm import TaskType as _TaskType  # type: ignore[import-not-found]

        _name_map = {tt.value: tt for tt in _TaskType}
        forced_task_type = _name_map.get(force_task_type_str)

    def tracked_parse_task_type(response: str) -> object:
        nonlocal tracked_task_type
        if forced_task_type is not None:
            tracked_task_type = getattr(forced_task_type, "value", str(forced_task_type))
            return forced_task_type
        task_type = original_parse_task_type(response)
        tracked_task_type = getattr(task_type, "value", str(task_type))
        return task_type

    def tracked_plan_method(task_type: object, n: int) -> object:
        nonlocal tracked_plan
        plan = original_plan(task_type, n)
        tracked_plan = {
            "task_type": getattr(getattr(plan, "task_type", None), "value", tracked_task_type),
            "compose_op": getattr(getattr(plan, "compose_op", None), "value", ""),
            "k_star": int(getattr(plan, "k_star", 0) or 0),
            "tau_star": int(getattr(plan, "tau_star", 0) or 0),
            "depth": int(getattr(plan, "depth", 0) or 0),
            "cost_estimate": float(getattr(plan, "cost_estimate", 0.0) or 0.0),
            "n": int(getattr(plan, "n", 0) or 0),
        }
        return plan

    rlm._parse_task_type = tracked_parse_task_type  # type: ignore[method-assign]
    rlm._plan = tracked_plan_method  # type: ignore[method-assign]
    try:
        completion = rlm.completion(str(payload.get("prompt") or ""))
    except OpenAICompatResponseError as exc:
        call_summary = _call_summary_from_clients(created_clients)
        response_path.write_text(
            json.dumps(
                {
                    "error": "openai_compat_reasoning_only",
                    "compat_message": str(exc),
                    "compat_meta": exc.meta,
                    "plan_metadata": tracked_plan,
                    "char_summary": _char_summary_from_clients(created_clients),
                    "usage_summary": {
                        "model_usage_summaries": {},
                        "total_calls": 0,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                    },
                    "call_summary": call_summary,
                }
            ),
            encoding="utf-8",
        )
        return 0
    usage_summary = completion.usage_summary.to_dict()
    usage_summary["total_input_tokens"] = int(
        getattr(completion.usage_summary, "total_input_tokens", 0)
    )
    usage_summary["total_output_tokens"] = int(
        getattr(completion.usage_summary, "total_output_tokens", 0)
    )
    usage_summary["total_calls"] = sum(
        int((row or {}).get("total_calls", 0) or 0)
        for row in cast(
            dict[str, dict[str, Any]], usage_summary.get("model_usage_summaries", {})
        ).values()
    )
    char_summary = _char_summary_from_clients(created_clients)
    call_summary = _call_summary_from_clients(created_clients)
    planned_depth = int(tracked_plan.get("depth", 0) or 0)
    planned_decomposition = planned_depth > 0
    executed_decomposition = bool(
        planned_decomposition and int(call_summary.get("repl_llm_query_call_count", 0) or 0) > 1
    )
    call_summary["planned_decomposition"] = planned_decomposition
    call_summary["executed_decomposition"] = executed_decomposition
    call_summary["planned_depth"] = planned_depth
    call_summary["k_star"] = int(tracked_plan.get("k_star", 0) or 0)
    call_summary["tau_star"] = int(tracked_plan.get("tau_star", 0) or 0)
    call_summary["task_type"] = str(tracked_plan.get("task_type") or tracked_task_type or "")
    response_path.write_text(
        json.dumps(
            {
                "root_model": str(completion.root_model or ""),
                "response": str(completion.response or ""),
                "execution_time": float(completion.execution_time or 0.0),
                "usage_summary": usage_summary,
                "char_summary": char_summary,
                "plan_metadata": tracked_plan,
                "call_summary": call_summary,
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
