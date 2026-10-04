"""Generation adapter for SWE-Bench Lite predictions."""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.patchers import OpenAICompatPatcher

from .. import _json as json
from .io import (
    ensure_run_layout,
    instance_ids_path,
    metadata_path,
    parse_paired_conditions,
    prediction_path,
    run_dir_for_id,
    validate_run_id,
    write_instance_ids,
    write_json,
    write_predictions,
)
from .prompt_policy import (
    PromptInputs,
    render_snapshot_section,
    require_snapshot_dir,
    snapshot_coverage,
)
from .types import (
    CONDITION_LLMDEBUG,
    SWEBENCH_LITE_DATASET_NAME,
    SWEBENCH_LITE_DEFAULT_SPLIT,
    ConditionGenerationSummary,
    GenerationRunSummary,
    PredictionRecord,
)


def generate_lite_predictions(
    *,
    run_id: str,
    conditions: str | list[str],
    dataset_name: str = SWEBENCH_LITE_DATASET_NAME,
    split: str = SWEBENCH_LITE_DEFAULT_SPLIT,
    max_instances: int = 0,
    instance_ids_file: Path | None = None,
    openai_base_url: str,
    openai_model: str,
    openai_api_key: str | None,
    openai_http2: bool = True,
    openai_async: bool = False,
    openai_timeout_sec: float = 180.0,
    openai_max_tokens: int = 2048,
    openai_temperature: float = 0.0,
    openai_response_mode: str = "diff",
    prompt_inputs: PromptInputs | None = None,
    resume: bool = False,
    seed: int | None = None,
    jobs: int = 1,
    max_request_errors: int = 25,
    max_request_error_rate: float = 0.5,
    min_requests_for_error_rate_check: int = 10,
    root: Path | None = None,
) -> GenerationRunSummary:
    """Generate SWE-Bench Lite prediction files for one or more conditions.

    Loads instances from *dataset_name*/*split*, queries an OpenAI-compatible
    model for each instance, and writes per-condition JSONL prediction files
    under the run directory identified by *run_id*.  Supports resuming a
    partial run (``resume=True``) and parallel generation via ``jobs``.

    Returns a :class:`GenerationRunSummary` with per-condition counts and paths.
    """
    safe_run_id = validate_run_id(run_id)
    condition_list = parse_paired_conditions(conditions)
    if max_instances < 0:
        raise ValueError("max_instances must be >= 0")
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    if max_request_errors < 0:
        raise ValueError("max_request_errors must be >= 0")
    if not 0 <= max_request_error_rate <= 1:
        raise ValueError("max_request_error_rate must be between 0 and 1")
    if min_requests_for_error_rate_check < 1:
        raise ValueError("min_requests_for_error_rate_check must be >= 1")
    prompt_inputs = prompt_inputs or PromptInputs()
    require_snapshot_dir(condition_list, prompt_inputs.snapshot_dir)

    run_dir = run_dir_for_id(safe_run_id, root=root)
    ensure_run_layout(run_dir, conditions=condition_list)

    rows_by_id = _load_dataset_rows(dataset_name=dataset_name, split=split)
    selected_ids = _select_instance_ids(
        rows_by_id=rows_by_id,
        instance_ids_file=instance_ids_file,
        max_instances=max_instances,
    )
    write_instance_ids(instance_ids_path(run_dir), selected_ids)

    patcher = OpenAICompatPatcher(
        base_url=openai_base_url,
        model=openai_model,
        api_key=openai_api_key,
        http2=bool(openai_http2),
        timeout_sec=openai_timeout_sec,
        max_tokens=openai_max_tokens,
        temperature=openai_temperature,
        response_mode=openai_response_mode,
        seed=seed,
    )

    condition_summaries: list[ConditionGenerationSummary] = []
    for condition in condition_list:
        summary = _generate_for_condition(
            run_dir=run_dir,
            condition=condition,
            selected_ids=selected_ids,
            rows_by_id=rows_by_id,
            patcher=patcher,
            model_label=f"{openai_model}::{condition}",
            resume=resume,
            prompt_inputs=prompt_inputs,
            jobs=jobs,
            openai_async=openai_async,
            max_request_errors=max_request_errors,
            max_request_error_rate=max_request_error_rate,
            min_requests_for_error_rate_check=min_requests_for_error_rate_check,
        )
        condition_summaries.append(summary)

    metadata_payload: dict[str, Any] = {
        "schema_version": "swebench_run_v1",
        "run_id": safe_run_id,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_name": dataset_name,
        "split": split,
        "conditions": condition_list,
        "instance_count": len(selected_ids),
        "instance_ids_file": str(instance_ids_file) if instance_ids_file is not None else None,
        "max_instances": int(max_instances),
        "resume": bool(resume),
        "prompt_policy": prompt_inputs.to_metadata(),
        "snapshot_coverage": (
            snapshot_coverage(snapshot_dir=prompt_inputs.snapshot_dir, instance_ids=selected_ids)
            if CONDITION_LLMDEBUG in condition_list
            else None
        ),
        "openai": {
            "base_url": openai_base_url,
            "model": openai_model,
            "http2": bool(openai_http2),
            "async": bool(openai_async),
            "timeout_sec": float(openai_timeout_sec),
            "max_tokens": int(openai_max_tokens),
            "temperature": float(openai_temperature),
            "response_mode": openai_response_mode,
            "seed": seed,
            "jobs": int(jobs),
            "max_request_errors": int(max_request_errors),
            "max_request_error_rate": float(max_request_error_rate),
            "min_requests_for_error_rate_check": int(min_requests_for_error_rate_check),
        },
        "git_commit": _resolve_git_commit(),
        "condition_summaries": [
            {
                "condition": item.condition,
                "predictions_path": str(item.predictions_path),
                "total_instances": item.total_instances,
                "generated": item.generated,
                "resumed": item.resumed,
                "empty_patch_count": item.empty_patch_count,
                "request_error_count": item.request_error_count,
            }
            for item in condition_summaries
        ],
    }
    write_json(metadata_path(run_dir), metadata_payload)

    return GenerationRunSummary(
        run_id=safe_run_id,
        run_dir=run_dir,
        instance_count=len(selected_ids),
        condition_summaries=tuple(condition_summaries),
    )


def _load_dataset_rows(*, dataset_name: str, split: str) -> dict[str, dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for SWE-Bench generation. Install with: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    rows_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for row in dataset:
        payload = dict(row)
        instance_id = str(payload.get("instance_id") or "").strip()
        if not instance_id:
            continue
        if instance_id in rows_by_id:
            duplicate_ids.add(instance_id)
            continue
        rows_by_id[instance_id] = payload

    if duplicate_ids:
        sorted_dupes = sorted(duplicate_ids)
        preview = ", ".join(sorted_dupes[:5])
        if len(sorted_dupes) > 5:
            preview = f"{preview}, ..."
        raise RuntimeError(
            "Duplicate instance_id values found in dataset "
            f"{dataset_name!r} split {split!r}: {preview}"
        )

    if not rows_by_id:
        raise RuntimeError(
            f"No rows with instance_id found in dataset {dataset_name!r} split {split!r}"
        )
    return rows_by_id


def _select_instance_ids(
    *,
    rows_by_id: dict[str, dict[str, Any]],
    instance_ids_file: Path | None,
    max_instances: int,
) -> list[str]:
    if instance_ids_file is not None:
        requested: list[str] = []
        seen: set[str] = set()
        for raw in instance_ids_file.read_text(encoding="utf-8", errors="replace").splitlines():
            value = raw.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            requested.append(value)

        missing = [item for item in requested if item not in rows_by_id]
        if missing:
            raise RuntimeError(
                f"instance_ids_file contains ids not present in dataset: {missing[:5]}"
            )
        selected = requested
    else:
        selected = sorted(rows_by_id)

    if max_instances > 0:
        return selected[:max_instances]
    return selected


def _generate_for_condition(
    *,
    run_dir: Path,
    condition: str,
    selected_ids: list[str],
    rows_by_id: dict[str, dict[str, Any]],
    patcher: OpenAICompatPatcher,
    model_label: str,
    resume: bool,
    prompt_inputs: PromptInputs,
    jobs: int,
    openai_async: bool,
    max_request_errors: int,
    max_request_error_rate: float,
    min_requests_for_error_rate_check: int,
) -> ConditionGenerationSummary:
    path = prediction_path(run_dir, condition)
    existing_map: dict[str, PredictionRecord] = {}
    if resume and path.exists():
        from .io import read_predictions

        for record in read_predictions(path):
            existing_map[record.instance_id] = record

    resumed = sum(1 for item in selected_ids if item in existing_map)
    pending_ids = [item for item in selected_ids if item not in existing_map]
    generated = 0
    empty_patch_count = 0
    request_error_count = 0

    with tempfile.TemporaryDirectory(prefix=f"swebench-{condition}-") as tmp:
        tmp_root = Path(tmp)
        if openai_async:
            generated, empty_patch_count, request_error_count = asyncio.run(
                _generate_for_condition_async(
                    condition=condition,
                    pending_ids=pending_ids,
                    selected_ids=selected_ids,
                    rows_by_id=rows_by_id,
                    existing_map=existing_map,
                    prompt_inputs=prompt_inputs,
                    patcher=patcher,
                    model_label=model_label,
                    workspace_root=tmp_root,
                    generated=generated,
                    resumed=resumed,
                    empty_patch_count=empty_patch_count,
                    request_error_count=request_error_count,
                    jobs=jobs,
                    max_request_errors=max_request_errors,
                    max_request_error_rate=max_request_error_rate,
                    min_requests_for_error_rate_check=min_requests_for_error_rate_check,
                    prediction_path_value=path,
                )
            )
        elif jobs == 1:
            for instance_id in pending_ids:
                outcome = _propose_prediction(
                    instance_id=instance_id,
                    row=rows_by_id[instance_id],
                    condition=condition,
                    prompt_inputs=prompt_inputs,
                    patcher=patcher,
                    model_label=model_label,
                    workspace_root=tmp_root,
                )
                if outcome.request_error is not None:
                    # DAT-08: a failed request is not a prediction; --resume retries it.
                    request_error_count += 1
                else:
                    if outcome.is_empty:
                        empty_patch_count += 1
                    existing_map[instance_id] = outcome.record
                generated += 1
                _write_prediction_map(path=path, ordered_ids=selected_ids, record_map=existing_map)
                _assert_generation_health(
                    condition=condition,
                    attempted=generated,
                    total_pending=len(pending_ids),
                    request_error_count=request_error_count,
                    max_request_errors=max_request_errors,
                    max_request_error_rate=max_request_error_rate,
                    min_requests_for_error_rate_check=min_requests_for_error_rate_check,
                )
                if (generated + resumed) % 10 == 0 or generated == len(pending_ids):
                    print(
                        f"[{condition}] progress {generated + resumed}/{len(selected_ids)} "
                        f"generated={generated} resumed={resumed} empty={empty_patch_count} "
                        f"request_errors={request_error_count}"
                    )
        else:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                future_map: dict[Future[ProposalOutcome], str] = {}
                for instance_id in pending_ids:
                    future = pool.submit(
                        _propose_prediction,
                        instance_id=instance_id,
                        row=rows_by_id[instance_id],
                        condition=condition,
                        prompt_inputs=prompt_inputs,
                        patcher=patcher,
                        model_label=model_label,
                        workspace_root=tmp_root,
                    )
                    future_map[future] = instance_id

                for future in as_completed(future_map):
                    instance_id = future_map[future]
                    outcome = future.result()
                    if outcome.request_error is not None:
                        # DAT-08: a failed request is not a prediction; --resume retries it.
                        request_error_count += 1
                    else:
                        if outcome.is_empty:
                            empty_patch_count += 1
                        existing_map[instance_id] = outcome.record
                    generated += 1
                    _write_prediction_map(
                        path=path, ordered_ids=selected_ids, record_map=existing_map
                    )
                    _assert_generation_health(
                        condition=condition,
                        attempted=generated,
                        total_pending=len(pending_ids),
                        request_error_count=request_error_count,
                        max_request_errors=max_request_errors,
                        max_request_error_rate=max_request_error_rate,
                        min_requests_for_error_rate_check=min_requests_for_error_rate_check,
                    )
                    if (generated + resumed) % 10 == 0 or generated == len(pending_ids):
                        print(
                            f"[{condition}] progress {generated + resumed}/{len(selected_ids)} "
                            f"generated={generated} resumed={resumed} empty={empty_patch_count} "
                            f"request_errors={request_error_count}"
                        )

    _write_prediction_map(path=path, ordered_ids=selected_ids, record_map=existing_map)

    return ConditionGenerationSummary(
        condition=condition,
        predictions_path=path,
        total_instances=len(selected_ids),
        generated=generated,
        resumed=resumed,
        empty_patch_count=empty_patch_count,
        request_error_count=request_error_count,
    )


@dataclass(frozen=True)
class ProposalOutcome:
    record: PredictionRecord
    is_empty: bool
    request_error: str | None = None


def _propose_prediction(
    *,
    instance_id: str,
    row: dict[str, Any],
    condition: str,
    prompt_inputs: PromptInputs,
    patcher: OpenAICompatPatcher,
    model_label: str,
    workspace_root: Path,
) -> ProposalOutcome:
    prompt = _build_prompt(
        row=row,
        condition=condition,
        prompt_inputs=prompt_inputs,
    )
    patch = ""
    request_error: str | None = None
    with tempfile.TemporaryDirectory(
        prefix=f"instance-{instance_id[:24]}-",
        dir=str(workspace_root),
    ) as workspace:
        workspace_path = Path(workspace)
        scratch = workspace_path / "SCRATCH.txt"
        scratch.write_text("scratch workspace for patch generation\n", encoding="utf-8")
        try:
            result = patcher.propose_patch(
                prompt=prompt,
                context_dir=workspace_path,
                cwd=workspace_path,
            )
            patch = str(result.diff or "")
        except Exception as exc:
            patch = ""
            request_error = f"{type(exc).__name__}: {exc}"

    is_empty = not bool(patch.strip())
    if is_empty:
        patch = ""

    return ProposalOutcome(
        record=PredictionRecord(
            instance_id=instance_id,
            model_patch=patch,
            model_name_or_path=model_label,
        ),
        is_empty=is_empty,
        request_error=request_error,
    )


async def _propose_prediction_async(
    *,
    instance_id: str,
    row: dict[str, Any],
    condition: str,
    prompt_inputs: PromptInputs,
    patcher: OpenAICompatPatcher,
    model_label: str,
    workspace_root: Path,
) -> ProposalOutcome:
    prompt = _build_prompt(
        row=row,
        condition=condition,
        prompt_inputs=prompt_inputs,
    )
    patch = ""
    request_error: str | None = None
    with tempfile.TemporaryDirectory(
        prefix=f"instance-{instance_id[:24]}-",
        dir=str(workspace_root),
    ) as workspace:
        workspace_path = Path(workspace)
        scratch = workspace_path / "SCRATCH.txt"
        scratch.write_text("scratch workspace for patch generation\n", encoding="utf-8")
        try:
            result = await patcher.apropose_patch(
                prompt=prompt,
                context_dir=workspace_path,
                cwd=workspace_path,
            )
            patch = str(result.diff or "")
        except Exception as exc:
            patch = ""
            request_error = f"{type(exc).__name__}: {exc}"

    is_empty = not bool(patch.strip())
    if is_empty:
        patch = ""

    return ProposalOutcome(
        record=PredictionRecord(
            instance_id=instance_id,
            model_patch=patch,
            model_name_or_path=model_label,
        ),
        is_empty=is_empty,
        request_error=request_error,
    )


async def _generate_for_condition_async(
    *,
    condition: str,
    pending_ids: list[str],
    selected_ids: list[str],
    rows_by_id: dict[str, dict[str, Any]],
    existing_map: dict[str, PredictionRecord],
    prompt_inputs: PromptInputs,
    patcher: OpenAICompatPatcher,
    model_label: str,
    workspace_root: Path,
    generated: int,
    resumed: int,
    empty_patch_count: int,
    request_error_count: int,
    jobs: int,
    max_request_errors: int,
    max_request_error_rate: float,
    min_requests_for_error_rate_check: int,
    prediction_path_value: Path,
) -> tuple[int, int, int]:
    semaphore = asyncio.Semaphore(max(1, jobs))

    async def _run_instance(instance_id: str) -> tuple[str, ProposalOutcome]:
        async with semaphore:
            outcome = await _propose_prediction_async(
                instance_id=instance_id,
                row=rows_by_id[instance_id],
                condition=condition,
                prompt_inputs=prompt_inputs,
                patcher=patcher,
                model_label=model_label,
                workspace_root=workspace_root,
            )
        return instance_id, outcome

    tasks = [asyncio.create_task(_run_instance(instance_id)) for instance_id in pending_ids]
    try:
        for task in asyncio.as_completed(tasks):
            instance_id, outcome = await task
            if outcome.request_error is not None:
                # DAT-08: a failed request is not a prediction; --resume retries it.
                request_error_count += 1
            else:
                if outcome.is_empty:
                    empty_patch_count += 1
                existing_map[instance_id] = outcome.record
            generated += 1
            _write_prediction_map(
                path=prediction_path_value,
                ordered_ids=selected_ids,
                record_map=existing_map,
            )
            _assert_generation_health(
                condition=condition,
                attempted=generated,
                total_pending=len(pending_ids),
                request_error_count=request_error_count,
                max_request_errors=max_request_errors,
                max_request_error_rate=max_request_error_rate,
                min_requests_for_error_rate_check=min_requests_for_error_rate_check,
            )
            if (generated + resumed) % 10 == 0 or generated == len(pending_ids):
                print(
                    f"[{condition}] progress {generated + resumed}/{len(selected_ids)} "
                    f"generated={generated} resumed={resumed} empty={empty_patch_count} "
                    f"request_errors={request_error_count}"
                )
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return generated, empty_patch_count, request_error_count


def _assert_generation_health(
    *,
    condition: str,
    attempted: int,
    total_pending: int,
    request_error_count: int,
    max_request_errors: int,
    max_request_error_rate: float,
    min_requests_for_error_rate_check: int,
) -> None:
    if request_error_count == 0:
        return
    if total_pending > 0 and request_error_count >= total_pending:
        raise RuntimeError(
            f"{condition}: all patch requests failed "
            f"({request_error_count}/{total_pending}). Aborting run."
        )
    if request_error_count > max_request_errors:
        raise RuntimeError(
            f"{condition}: patch request errors exceeded threshold "
            f"({request_error_count} > {max_request_errors}). Aborting run."
        )
    if attempted >= min_requests_for_error_rate_check:
        error_rate = request_error_count / attempted
        if error_rate > max_request_error_rate:
            raise RuntimeError(
                f"{condition}: patch request error rate too high "
                f"({error_rate:.2%} > {max_request_error_rate:.2%}) after {attempted} attempts. "
                "Aborting run."
            )


def _write_prediction_map(
    *,
    path: Path,
    ordered_ids: list[str],
    record_map: dict[str, PredictionRecord],
) -> None:
    records = [record_map[item] for item in ordered_ids if item in record_map]
    write_predictions(path, records)


def _build_prompt(*, row: dict[str, Any], condition: str, prompt_inputs: PromptInputs) -> str:
    instance_id = str(row.get("instance_id") or "")
    repo = str(row.get("repo") or "")
    base_commit = str(row.get("base_commit") or "")
    problem_statement = str(row.get("problem_statement") or "").strip()

    extra_sections: list[str] = []
    # hints_text and the scorer's test IDs are not part of the official task (DAT-01/02).
    hints_text = str(row.get("hints_text") or "").strip()
    if prompt_inputs.include_hints and hints_text:
        extra_sections.append("Hints:\n" + hints_text)

    if prompt_inputs.include_test_ids:
        fail_to_pass = _stringify_field(row.get("FAIL_TO_PASS"))
        pass_to_pass = _stringify_field(row.get("PASS_TO_PASS"))
        if fail_to_pass:
            extra_sections.append("Fail-to-pass tests:\n" + fail_to_pass)
        if pass_to_pass:
            extra_sections.append("Pass-to-pass tests:\n" + pass_to_pass)

    if condition == CONDITION_LLMDEBUG:
        extra_sections.append(
            render_snapshot_section(
                snapshot_dir=prompt_inputs.snapshot_dir, instance_id=instance_id
            )
        )

    extra = "\n\n".join(extra_sections)
    return (
        "You are an autonomous software repair assistant.\n"
        "Return ONLY a unified git diff patch that addresses the issue.\n"
        "Do not include markdown fences or explanation.\n\n"
        f"Repository: {repo}\n"
        f"Instance ID: {instance_id}\n"
        f"Base commit: {base_commit}\n\n"
        "Issue:\n"
        f"{problem_statement}\n\n"
        f"{extra}\n"
    ).rstrip() + "\n"


def _stringify_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(cleaned)
    if isinstance(value, dict):
        return json.dumps(value, indent=2, ensure_ascii=False)
    return str(value).strip()


def _resolve_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    value = proc.stdout.strip()
    return value or "unknown"
