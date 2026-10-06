"""Evidence items, context requests, file index, and search utilities.

Extracted from ``evals.run_eval`` to keep the main orchestrator lean.
Re-exported by ``evals.run_eval`` for backward compatibility.
"""

from __future__ import annotations

import ast
import os
import re
import shlex
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

from llmdebug._hashing import xxh3_128_hexdigest_text

from . import _json as json
from .eval_patches import _is_safe_relative_path
from .eval_types import _EVAL_ONLY_CASE_ARTIFACTS, ContextRequestPayload, StageAContextData
from .stage_a_contract import coerce_stage_a_bool, normalize_stage_a_request_payload


def _xxh3_128_text(text: str) -> str:
    return xxh3_128_hexdigest_text(text)


def _xxh3_128_jsonlike(data: Any) -> str:
    payload = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return _xxh3_128_text(payload)


def _compute_evidence_fingerprint(
    *,
    failure_signature: str,
    stage_a_blocks: list[dict[str, Any]],
    stage_a_context: str,
    structured_seed_blocks: list[dict[str, Any]],
) -> str:
    payload = {
        "failure_signature": failure_signature,
        "stage_a_blocks": stage_a_blocks,
        "stage_a_block_ids": [
            str(block.get("id") or "")
            for block in stage_a_blocks
            if isinstance(block, dict) and block.get("id")
        ],
        "structured_seed_blocks": structured_seed_blocks,
        "stage_a_context_xxh3_128": _xxh3_128_text(stage_a_context or ""),
    }
    return _xxh3_128_jsonlike(payload)


def _build_evidence_item_id(*, kind: str, locator: str, text: str) -> str:
    payload = f"{kind}|{locator}|{_xxh3_128_text(text)}"
    return "e_" + _xxh3_128_text(payload)[:10]


def _annotate_evidence_text(*, evidence_id: str, kind: str, locator: str, text: str) -> str:
    header = f"[{evidence_id}] {kind}:{locator}".strip()
    body = str(text or "").rstrip()
    if not body:
        return header + "\n"
    return header + "\n" + body + "\n"


def _build_evidence_item(
    *,
    kind: str,
    locator: str,
    text: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    evidence_id = _build_evidence_item_id(kind=kind, locator=locator, text=text)
    annotated_text = _annotate_evidence_text(
        evidence_id=evidence_id,
        kind=kind,
        locator=locator,
        text=text,
    )
    return {
        "id": evidence_id,
        "type": kind,
        "locator": locator,
        "text": annotated_text,
        "chars": len(annotated_text),
        **meta,
    }


def _render_evidence_items(items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in items:
        text = str(item.get("text") or "")
        if text:
            parts.append(text.rstrip())
    if not parts:
        return ""
    return "\n\n".join(parts).rstrip() + "\n"


def _render_evidence_index(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("id") or "").strip()
        if not evidence_id:
            continue
        kind = str(item.get("type") or "evidence").strip() or "evidence"
        locator = str(item.get("locator") or item.get("path") or "").strip()
        chars_value = item.get("chars")
        chars_suffix = ""
        if isinstance(chars_value, int) and chars_value > 0:
            chars_suffix = f" ({chars_value} chars)"
        if locator:
            lines.append(f"- {evidence_id}: {kind}:{locator}{chars_suffix}")
        else:
            lines.append(f"- {evidence_id}: {kind}{chars_suffix}")
    if not lines:
        return ""
    return "\n".join(lines).rstrip() + "\n"


def _merge_evidence_items(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("id") or "")
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            out.append(item)
    return out


def _evidence_ids_from_items(items: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("id") or "") for item in items if isinstance(item, dict) and item.get("id")
    ]


def _evidence_ids_rendered_in_prompt(*, items: list[dict[str, Any]], prompt_text: str) -> list[str]:
    """IDs of *items* whose evidence block survived into *prompt_text* (POL-11).

    Evidence is selected before the prompt budget runs, and budgeting truncates or drops
    whole sections, so the selected IDs are not the delivered ones. Each block starts with
    the ``[<id>] kind:locator`` header written by :func:`_annotate_evidence_text`, so the
    header is the marker of a block that is still there. An empty *prompt_text* means no
    prompt was captured; then the selection is all there is to go on.
    """
    evidence_ids = _evidence_ids_from_items(items)
    if not prompt_text:
        return evidence_ids
    return [evidence_id for evidence_id in evidence_ids if f"[{evidence_id}]" in prompt_text]


def _coerce_payload_bool(value: Any) -> bool:
    return coerce_stage_a_bool(value)


def _coerce_context_request_payload(value: Any) -> ContextRequestPayload | None:
    if not isinstance(value, dict):
        return None
    requests_raw = value.get("requests")
    if requests_raw is not None and not isinstance(requests_raw, list):
        return None
    return normalize_stage_a_request_payload(cast(dict[str, Any], value))


def _normalize_context_rel_path(raw_path: str) -> str | None:
    candidate = raw_path.strip()
    if not candidate:
        return None
    candidate = candidate.replace("\\", "/")
    if candidate.startswith("/") or re.match(r"^[A-Za-z]:($|/)", candidate):
        return None

    normalized_parts: list[str] = []
    for part in candidate.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            return None
        normalized_parts.append(part)
    if not normalized_parts:
        return None
    return "/".join(normalized_parts)


def _iter_file_index_lines(file_index: str | Sequence[str]) -> Iterator[str]:
    if isinstance(file_index, str):
        yield from file_index.splitlines()
        return
    for line in file_index:
        yield str(line)


def _read_text_limited(path: Path, *, max_chars: int) -> str | None:
    """Read at most ``max_chars`` chars from a text file (+truncation marker)."""
    cap = max(1, int(max_chars))
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            text = handle.read(cap + 1)
    except Exception:
        return None

    if len(text) <= cap:
        return text
    return text[:cap] + "\n...[TRUNC]\n"


def _literal_search_matches_in_file(
    path: Path,
    pattern: str,
    *,
    max_matches: int,
    max_scan_chars: int,
) -> tuple[list[str], bool]:
    """Stream literal search over file content with bounded scan size."""
    if not pattern:
        return [], False

    out: list[str] = []
    scanned_chars = 0
    scan_cap = max(1, int(max_scan_chars))
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for idx, line in enumerate(handle, start=1):
                scanned_chars += len(line)
                if pattern in line:
                    rendered = line.rstrip()
                    if len(rendered) > 220:
                        rendered = rendered[:217] + "..."
                    out.append(f"{idx}: {rendered}")
                    if len(out) >= max_matches:
                        return out, scanned_chars >= scan_cap
                if scanned_chars >= scan_cap:
                    return out, True
    except Exception:
        return [], False
    return out, False


def _collect_file_index_lines(root: Path) -> list[str]:
    include_suffixes = {".py", ".txt", ".json", ".md"}
    exclude_dirs = {".llmdebug", "__pycache__", ".venv", ".git"}
    collected: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in exclude_dirs)
        dirpath_path = Path(dirpath)
        for filename in sorted(filenames):
            path = dirpath_path / filename
            if filename in _EVAL_ONLY_CASE_ARTIFACTS:
                continue
            if path.suffix not in include_suffixes:
                continue
            try:
                rel = str(path.relative_to(root))
            except Exception:
                rel = str(path)
            collected.append(rel)
    collected.sort()
    return collected


def execute_context_requests(
    *,
    request_payload: ContextRequestPayload,
    root: Path,
    snapshot: dict[str, Any] | None,
    file_index: str | Sequence[str],
    stdout: str,
    stderr: str,
    max_requests: int,
    max_total_chars: int,
) -> StageAContextData:
    """Fulfil Stage-A context requests from the model.

    Reads requested files, searches for literals, and extracts frames
    from the snapshot.  Falls back to ``build_fallback_context_bundle``
    when the payload is malformed or no context is needed.
    """
    file_index_lines = tuple(_iter_file_index_lines(file_index))
    requests_raw = request_payload.get("requests")
    if not isinstance(requests_raw, list):
        return build_fallback_context_bundle(
            root=root,
            snapshot=snapshot,
            stdout=stdout,
            stderr=stderr,
            max_total_chars=max_total_chars,
            reason="malformed_stage_a_requests",
            file_index_lines=file_index_lines,
        )

    need_more_context = _coerce_payload_bool(request_payload.get("need_more_context"))
    if not need_more_context:
        return StageAContextData(
            blocks=[],
            text="",
            used_fallback=False,
            requests_count=0,
            chars_returned=0,
            request_success=True,
            request_payload=request_payload,
            request_raw=None,
            reason="sufficient_context",
        )

    if max_requests <= 0:
        # ATT-21: a request budget of zero means zero requests.
        return StageAContextData(
            blocks=[],
            text="",
            used_fallback=False,
            requests_count=0,
            chars_returned=0,
            request_success=False,
            request_payload=request_payload,
            request_raw=None,
            reason="request_budget_zero",
        )

    allowed_paths: dict[str, str] = {}
    for line in file_index_lines:
        normalized = _normalize_context_rel_path(line)
        if normalized and normalized not in allowed_paths:
            allowed_paths[normalized] = normalized
    capped = requests_raw[:max_requests]
    blocks: list[dict[str, Any]] = []
    evidence_items: list[dict[str, Any]] = []
    seen_evidence_ids: set[str] = set()
    total_chars = 0
    accepted = 0
    max_per_block = 40_000

    for req in capped:
        if not isinstance(req, dict):
            continue
        kind = str(req.get("type") or "").strip()
        if kind == "file":
            normalized_rel_path = _normalize_context_rel_path(str(req.get("path") or ""))
            if normalized_rel_path is None:
                continue
            rel_path = allowed_paths.get(normalized_rel_path)
            if rel_path is None:
                continue
            path = root / rel_path
            if not path.exists() or not path.is_file():
                continue
            text = _read_text_limited(path, max_chars=max_per_block)
            if text is None:
                continue
            block_text = f"--- {rel_path} ---\n{text.rstrip()}\n"
            if total_chars + len(block_text) > max_total_chars:
                break
            evidence_item = _build_evidence_item(
                kind="file",
                locator=rel_path,
                text=block_text,
                meta={"path": rel_path},
            )
            evidence_id = str(evidence_item.get("id") or "")
            if evidence_id in seen_evidence_ids:
                continue
            seen_evidence_ids.add(evidence_id)
            blocks.append(
                {
                    "id": evidence_id,
                    "type": "file",
                    "path": rel_path,
                    "chars": len(block_text),
                }
            )
            evidence_items.append(evidence_item)
            total_chars += len(block_text)
            accepted += 1
            continue

        if kind == "search":
            normalized_rel_path = _normalize_context_rel_path(str(req.get("path") or ""))
            pattern = str(req.get("pattern") or "")
            if normalized_rel_path is None or not pattern:
                continue
            rel_path = allowed_paths.get(normalized_rel_path)
            if rel_path is None:
                continue
            path = root / rel_path
            if not path.exists() or not path.is_file():
                continue
            matches, search_truncated = _literal_search_matches_in_file(
                path,
                pattern,
                max_matches=20,
                max_scan_chars=max_per_block * 6,
            )
            lines = [f"--- search {rel_path!s} pattern={pattern!r} ---"]
            if matches:
                lines.extend(matches)
            else:
                lines.append("(no literal matches)")
            if search_truncated:
                lines.append("(search truncated)")
            block_text = "\n".join(lines).rstrip() + "\n"
            block_text = block_text[:max_per_block]
            if total_chars + len(block_text) > max_total_chars:
                break
            evidence_item = _build_evidence_item(
                kind="search",
                locator=f"{rel_path}:{pattern}",
                text=block_text,
                meta={"path": rel_path, "pattern": pattern, "matches": len(matches)},
            )
            evidence_id = str(evidence_item.get("id") or "")
            if evidence_id in seen_evidence_ids:
                continue
            seen_evidence_ids.add(evidence_id)
            blocks.append(
                {
                    "id": evidence_id,
                    "type": "search",
                    "path": rel_path,
                    "pattern": pattern,
                    "matches": len(matches),
                    "chars": len(block_text),
                }
            )
            evidence_items.append(evidence_item)
            total_chars += len(block_text)
            accepted += 1
            continue

        if kind == "frame":
            if not isinstance(snapshot, dict):
                continue
            idx = req.get("index")
            if not isinstance(idx, int):
                continue
            frames_obj = snapshot.get("frames")
            frames = frames_obj if isinstance(frames_obj, list) else []
            if idx < 0 or idx >= len(frames):
                continue
            frame = frames[idx]
            if not isinstance(frame, dict):
                continue
            file_rel = frame.get("file_rel", frame.get("file", "?"))
            line_no = frame.get("line", "?")
            func = frame.get("function", "?")
            code = frame.get("code", "")
            block_lines = [f"--- frame[{idx}] {file_rel}:{line_no} in {func}() ---"]
            if code:
                block_lines.append(f"code: {code}")
            locals_dict = frame.get("locals")
            if isinstance(locals_dict, dict) and locals_dict:
                block_lines.append("locals:")
                for key, value in list(locals_dict.items())[:20]:
                    if str(key).startswith("@"):
                        continue
                    try:
                        rendered_value = json.dumps(value, ensure_ascii=False, default=str)
                    except Exception:
                        rendered_value = repr(value)
                    if len(rendered_value) > 160:
                        rendered_value = rendered_value[:157] + "..."
                    block_lines.append(f"  {key} = {rendered_value}")
            block_text = "\n".join(block_lines).rstrip() + "\n"
            block_text = block_text[:max_per_block]
            if total_chars + len(block_text) > max_total_chars:
                break
            evidence_item = _build_evidence_item(
                kind="frame",
                locator=f"frame:{idx}",
                text=block_text,
                meta={"index": idx},
            )
            evidence_id = str(evidence_item.get("id") or "")
            if evidence_id in seen_evidence_ids:
                continue
            seen_evidence_ids.add(evidence_id)
            blocks.append(
                {"id": evidence_id, "type": "frame", "index": idx, "chars": len(block_text)}
            )
            evidence_items.append(evidence_item)
            total_chars += len(block_text)
            accepted += 1
            continue

        if kind == "local_var":
            if not isinstance(snapshot, dict):
                continue
            frame_idx = req.get("frame")
            name = str(req.get("name") or "").strip()
            if not isinstance(frame_idx, int) or not name:
                continue
            frames_obj = snapshot.get("frames")
            frames = frames_obj if isinstance(frames_obj, list) else []
            if frame_idx < 0 or frame_idx >= len(frames):
                continue
            frame = frames[frame_idx]
            if not isinstance(frame, dict):
                continue
            locals_dict = frame.get("locals")
            if not isinstance(locals_dict, dict) or name not in locals_dict:
                continue
            file_rel = frame.get("file_rel", frame.get("file", "?"))
            line_no = frame.get("line", "?")
            func = frame.get("function", "?")
            value = cast(dict[str, Any], locals_dict).get(name)
            try:
                rendered_value = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                rendered_value = repr(value)
            if len(rendered_value) > 4_000:
                rendered_value = rendered_value[:3997] + "..."
            block_text = (
                f"--- local_var frame[{frame_idx}] {file_rel}:{line_no} in {func}() ---\n"
                f"{name} = {rendered_value}\n"
            )
            block_text = block_text[:max_per_block]
            if total_chars + len(block_text) > max_total_chars:
                break
            evidence_item = _build_evidence_item(
                kind="local_var",
                locator=f"frame:{frame_idx}:{name}",
                text=block_text,
                meta={"frame": frame_idx, "name": name},
            )
            evidence_id = str(evidence_item.get("id") or "")
            if evidence_id in seen_evidence_ids:
                continue
            seen_evidence_ids.add(evidence_id)
            blocks.append(
                {
                    "id": evidence_id,
                    "type": "local_var",
                    "frame": frame_idx,
                    "name": name,
                    "chars": len(block_text),
                }
            )
            evidence_items.append(evidence_item)
            total_chars += len(block_text)
            accepted += 1
            continue

    if accepted == 0:
        return build_fallback_context_bundle(
            root=root,
            snapshot=snapshot,
            stdout=stdout,
            stderr=stderr,
            max_total_chars=max_total_chars,
            reason="no_valid_stage_a_requests",
            file_index_lines=file_index_lines,
            request_payload=request_payload,
        )

    return StageAContextData(
        blocks=blocks,
        text=_render_evidence_items(evidence_items),
        used_fallback=False,
        requests_count=accepted,
        chars_returned=total_chars,
        request_success=True,
        request_payload=request_payload,
        request_raw=None,
        reason="ok",
        evidence_items=evidence_items,
    )


EVIDENCE_CHAR_TYPES = ("snapshot", "snapshot_lite", "seed", "file", "search", "frame", "local_var")


def evidence_chars_by_type(
    *,
    snapshot_text: str,
    snapshot_lite_text: str,
    seed: StageAContextData | None,
    stage_a: StageAContextData,
) -> dict[str, int]:
    """Characters of evidence that reached the Stage-B prompt, per evidence type.

    ``seed`` is the whole-file seed bundle; Stage-A blocks are split by request type.
    Lets analysis check that restricted actions really deliver less (ATT-03/04/06).
    """
    out: dict[str, int] = dict.fromkeys(EVIDENCE_CHAR_TYPES, 0)
    out["snapshot"] = len(snapshot_text or "")
    out["snapshot_lite"] = len(snapshot_lite_text or "")
    out["seed"] = int(seed.chars_returned) if seed is not None else 0
    for block in stage_a.blocks:
        kind = str(block.get("type") or "")
        if kind in out:
            out[kind] += int(block.get("chars") or 0)
    return out


RESTRICTED_ACTION_NO_VALID_REQUESTS = "restricted_action_no_valid_requests"


def deny_fallback_for_restricted_action(
    data: StageAContextData, *, restricted: bool
) -> StageAContextData:
    """Drop a fallback bundle when the policy restricted the evidence types (ATT-03).

    The request filter (layer 3) may legitimately empty a restricted action's requests;
    falling back to whole files would hand back exactly the evidence the policy denied.
    """
    if not restricted or not data.used_fallback:
        return data
    return StageAContextData(
        blocks=[],
        text="",
        used_fallback=False,
        requests_count=0,
        chars_returned=0,
        request_success=False,
        request_payload=data.request_payload,
        request_raw=data.request_raw,
        reason=RESTRICTED_ACTION_NO_VALID_REQUESTS,
    )


def build_fallback_context_bundle(
    *,
    root: Path,
    snapshot: dict[str, Any] | None,
    stdout: str,
    stderr: str,
    max_total_chars: int,
    reason: str,
    file_index_lines: Sequence[str] | None = None,
    request_payload: ContextRequestPayload | None = None,
    request_raw: str | None = None,
) -> StageAContextData:
    """Build a best-effort context bundle when Stage-A requests fail.

    Heuristically selects crash-file, failing-test, and import-neighbor
    files up to *max_total_chars*.
    """
    allowed_paths = (
        {
            normalized
            for line in file_index_lines
            if (normalized := _normalize_context_rel_path(str(line))) is not None
        }
        if file_index_lines is not None
        else None
    )
    root_resolved = root.resolve()

    def _confined_rel_path(raw_path: str | None) -> str | None:
        """ATT-20: accept a candidate only if it stays inside *root* and the allow-list."""
        if not raw_path:
            return None
        rel = _normalize_context_rel_path(str(raw_path))
        if rel is None or (allowed_paths is not None and rel not in allowed_paths):
            return None
        if not (root / rel).resolve().is_relative_to(root_resolved):
            return None
        return rel

    candidate_paths: list[str] = []
    # ATT-20: validate the crash path before anything reads it, including neighbor discovery.
    crash_file = _confined_rel_path(_crash_file_from_snapshot(snapshot))
    if crash_file is not None:
        candidate_paths.append(crash_file)

    failing_test = _detect_failing_test_file(
        stdout=stdout,
        stderr=stderr,
        root=root,
        file_index_lines=file_index_lines,
    )
    if failing_test is not None and failing_test not in candidate_paths:
        candidate_paths.append(failing_test)

    neighbors = _import_neighbors(root=root, rel_path=crash_file, limit=8)
    for neighbor in neighbors:
        if neighbor not in candidate_paths:
            candidate_paths.append(neighbor)

    if not candidate_paths:
        lines = (
            file_index_lines if file_index_lines is not None else _collect_file_index_lines(root)
        )
        candidate_paths = [str(line).strip() for line in lines if str(line).strip()][:3]

    blocks: list[dict[str, Any]] = []
    evidence_items: list[dict[str, Any]] = []
    seen_evidence_ids: set[str] = set()
    total_chars = 0
    for rel_path in candidate_paths:
        rel = _confined_rel_path(rel_path)
        if rel is None:
            continue
        target = root / rel
        if not target.exists() or not target.is_file():
            continue
        remaining_chars = max_total_chars - total_chars
        if remaining_chars <= 0:
            break
        text = _read_text_limited(target, max_chars=min(40_000, remaining_chars))
        if text is None:
            continue
        block = f"--- {rel} ---\n{text.rstrip()}\n"
        if total_chars + len(block) > max_total_chars:
            break
        evidence_item = _build_evidence_item(
            kind="file",
            locator=rel,
            text=block,
            meta={"path": rel},
        )
        evidence_id = str(evidence_item.get("id") or "")
        if evidence_id in seen_evidence_ids:
            continue
        seen_evidence_ids.add(evidence_id)
        evidence_items.append(evidence_item)
        blocks.append({"id": evidence_id, "type": "file", "path": rel, "chars": len(block)})
        total_chars += len(block)

    return StageAContextData(
        blocks=blocks,
        text=_render_evidence_items(evidence_items),
        used_fallback=True,
        requests_count=len(blocks),
        chars_returned=total_chars,
        request_success=False,
        request_payload=request_payload,
        request_raw=request_raw,
        reason=reason,
        evidence_items=evidence_items,
    )


def _crash_file_from_snapshot(snapshot: dict[str, Any] | None) -> str | None:
    if not isinstance(snapshot, dict):
        return None
    frames_obj = snapshot.get("frames")
    frames = frames_obj if isinstance(frames_obj, list) else []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        rel = frame.get("file_rel")
        if isinstance(rel, str) and rel.strip():
            return rel.strip()
    return None


def _detect_failing_test_file(
    *,
    stdout: str,
    stderr: str,
    root: Path,
    file_index_lines: Sequence[str] | None = None,
) -> str | None:
    combined = f"{stdout}\n{stderr}"
    for match in re.finditer(r"([A-Za-z0-9_./\\-]*test[A-Za-z0-9_./\\-]*\.py)", combined):
        candidate = match.group(1).replace("\\", "/")
        if not candidate:
            continue
        candidate_path = root / candidate
        if candidate_path.exists() and candidate_path.is_file():
            return candidate
    # Fallback: first local test_*.py file.
    lines = file_index_lines if file_index_lines is not None else _collect_file_index_lines(root)
    for rel in lines:
        name = Path(rel).name
        if name.startswith("test_") and rel.endswith(".py"):
            return rel
    return None


def _import_neighbors(*, root: Path, rel_path: str | None, limit: int) -> list[str]:
    if not rel_path:
        return []
    path = root / rel_path
    if not path.exists() or not path.is_file():
        return []
    text = _read_text_limited(path, max_chars=120_000)
    if text is None:
        return []

    modules: list[str] = []
    for line in text.splitlines():
        m_from = re.match(r"\s*from\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+import\s+", line)
        if m_from:
            modules.append(m_from.group(1))
            continue
        m_import = re.match(r"\s*import\s+([A-Za-z_][A-Za-z0-9_\.]*)", line)
        if m_import:
            modules.append(m_import.group(1))

    out: list[str] = []
    for module in modules:
        candidate = module.replace(".", "/") + ".py"
        candidate_path = root / candidate
        if candidate_path.exists() and candidate_path.is_file() and candidate not in out:
            out.append(candidate)
        if len(out) >= limit:
            break
    return out


def _resolve_subprocess_script_path(
    *,
    root: Path,
    raw_path: str,
    file_index_lines: Sequence[str] | None = None,
) -> str | None:
    token = str(raw_path or "").strip().strip("'\"")
    if not token:
        return None
    normalized = token.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized.endswith(".py"):
        return None
    if normalized.startswith("-"):
        return None

    root_resolved = root.resolve()
    candidate_paths: list[Path] = []
    if Path(normalized).is_absolute():
        candidate_paths.append(Path(normalized))
    elif _is_safe_relative_path(normalized):
        candidate_paths.append(root / normalized)
    else:
        return None

    for candidate in candidate_paths:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root_resolved)
        except Exception:
            continue
        if resolved.exists() and resolved.is_file():
            return str(PurePosixPath(resolved.relative_to(root_resolved).as_posix()))

    # Basename-only fallback for subprocess calls like ["python", "buggy.py"].
    if "/" in normalized:
        return None
    lines = file_index_lines if file_index_lines is not None else _collect_file_index_lines(root)
    matches = [
        rel for rel in lines if rel.endswith(".py") and Path(rel).name == Path(normalized).name
    ]
    if len(matches) == 1:
        return str(PurePosixPath(matches[0].replace("\\", "/")))
    return None


def _extract_subprocess_script_neighbors(
    *,
    root: Path,
    rel_path: str | None,
    limit: int,
    file_index_lines: Sequence[str] | None = None,
) -> list[str]:
    if not rel_path:
        return []
    path = root / rel_path
    if not path.exists() or not path.is_file():
        return []
    text = _read_text_limited(path, max_chars=240_000)
    if text is None:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    call_names = {"run", "popen", "call", "check_call", "check_output"}
    subprocess_module_aliases: set[str] = {"subprocess"}
    subprocess_member_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    subprocess_module_aliases.add(alias.asname or alias.name)
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            for alias in node.names:
                imported = alias.name.lower()
                if imported in call_names:
                    subprocess_member_aliases.add(alias.asname or alias.name)

    def _candidate_script_tokens(raw: str) -> list[str]:
        text_value = str(raw or "").strip()
        if not text_value:
            return []
        out: list[str] = []
        try:
            split_tokens = shlex.split(text_value)
        except ValueError:
            split_tokens = []
        tokens = split_tokens if split_tokens else [text_value]
        for token in tokens:
            for match in re.findall(r"[A-Za-z0-9_./\\-]+\.py", token):
                if match not in out:
                    out.append(match)
        return out

    def _string_values(value: ast.AST | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return [value.value]
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            out_values: list[str] = []
            for entry in value.elts:
                out_values.extend(_string_values(entry))
            return out_values
        if isinstance(value, ast.JoinedStr):
            literal_parts = [
                entry.value
                for entry in value.values
                if isinstance(entry, ast.Constant) and isinstance(entry.value, str)
            ]
            rendered = "".join(literal_parts).strip()
            return [rendered] if rendered else []
        return []

    resolved_neighbors: list[str] = []
    seen_neighbors: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_subprocess_call = False
        if isinstance(node.func, ast.Attribute):
            if (
                node.func.attr.lower() in call_names
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in subprocess_module_aliases
            ):
                is_subprocess_call = True
        elif isinstance(node.func, ast.Name) and node.func.id in subprocess_member_aliases:
            is_subprocess_call = True
        if not is_subprocess_call:
            continue

        candidate_values: list[str] = []
        if node.args:
            candidate_values.extend(_string_values(node.args[0]))
        for keyword in node.keywords:
            if keyword.arg == "args":
                candidate_values.extend(_string_values(keyword.value))
        for candidate in candidate_values:
            for token in _candidate_script_tokens(candidate):
                resolved = _resolve_subprocess_script_path(
                    root=root,
                    raw_path=token,
                    file_index_lines=file_index_lines,
                )
                if not resolved:
                    continue
                if resolved in seen_neighbors:
                    continue
                seen_neighbors.add(resolved)
                resolved_neighbors.append(resolved)
                if len(resolved_neighbors) >= limit:
                    return resolved_neighbors
    return resolved_neighbors


def collect_text_context(
    root: Path,
    *,
    max_total_chars: int = 200_000,
    file_index_lines: Sequence[str] | None = None,
) -> str:
    """Collect small text files from a case directory for LLM context."""
    max_total_bytes = max_total_chars
    max_file_bytes = 50_000

    parts: list[str] = []
    total = 0

    lines = file_index_lines if file_index_lines is not None else _collect_file_index_lines(root)
    for rel_text in lines:
        rel = Path(str(rel_text))
        path = root / rel
        if not path.exists() or not path.is_file():
            continue
        try:
            rel_display = path.relative_to(root)
        except Exception:
            rel_display = path
        text = _read_text_limited(path, max_chars=max_file_bytes)
        if text is None:
            continue

        block = f"--- {rel_display} ---\n{text.rstrip()}\n"
        if total + len(block) > max_total_bytes:
            break
        parts.append(block)
        total += len(block)

    return "\n".join(parts).rstrip() + ("\n" if parts else "")


def collect_file_index(root: Path) -> str:
    """Return a newline-delimited index of eligible files under *root*."""
    return "\n".join(_collect_file_index_lines(root))


def _strip_eval_only_case_artifacts(root: Path) -> None:
    for relative in (Path(name) for name in _EVAL_ONLY_CASE_ARTIFACTS):
        candidate = root / relative
        try:
            if candidate.is_file():
                candidate.unlink()
        except OSError:
            continue
