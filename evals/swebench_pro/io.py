"""Filesystem and validation helpers for SWE-Bench Pro runs."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from .. import _json as json
from ..swebench.types import VALID_CONDITIONS, PredictionRecord

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def repo_root() -> Path:
    """Return the absolute path to the repository root.

    Returns:
        Path two levels above this file's package directory.
    """
    return Path(__file__).resolve().parents[2]


def swebench_pro_runs_root(*, root: Path | None = None) -> Path:
    """Return the root directory that holds all SWE-Bench Pro run subdirectories.

    Args:
        root: Optional override for the repository root.  Defaults to the
            value returned by :func:`repo_root`.

    Returns:
        ``<root>/evals/swebench_pro_runs``.
    """
    base = root or repo_root()
    return base / "evals" / "swebench_pro_runs"


def validate_run_id(run_id: str) -> str:
    """Validate and normalise a run ID string.

    Strips surrounding whitespace, then checks that the result is non-empty
    and contains only letters, digits, ``'.'``, ``'_'``, or ``'-'``.

    Args:
        run_id: The raw run identifier to validate.

    Returns:
        The stripped run ID.

    Raises:
        ValueError: If *run_id* is empty or contains unsupported characters.
    """
    value = run_id.strip()
    if not value:
        raise ValueError("run_id must be non-empty")
    if _RUN_ID_RE.fullmatch(value) is None:
        raise ValueError(
            "run_id contains unsupported characters; allowed: letters, digits, '.', '_', '-'"
        )
    return value


def run_dir_for_id(run_id: str, *, root: Path | None = None) -> Path:
    """Return the run directory path for a given run ID.

    Args:
        run_id: Identifier for the run; validated by :func:`validate_run_id`.
        root: Optional repository root override passed to
            :func:`swebench_pro_runs_root`.

    Returns:
        ``<swebench_pro_runs_root>/<run_id>``.

    Raises:
        ValueError: If *run_id* fails validation.
    """
    safe = validate_run_id(run_id)
    return swebench_pro_runs_root(root=root) / safe


def parse_conditions(raw: str | Iterable[str] | None) -> list[str]:
    """Parse a raw condition specifier into a validated, deduplicated list.

    Accepts a comma-separated string, an iterable of strings, or ``None``.
    When *raw* is ``None`` or resolves to an empty token list, all valid
    conditions are returned in their canonical order.

    Args:
        raw: Condition name(s) as a comma-separated string, an iterable of
            strings, or ``None`` to select all conditions.

    Returns:
        Ordered list of validated condition names.

    Raises:
        ValueError: If any token is not a member of ``VALID_CONDITIONS``.
    """
    if raw is None:
        return list(VALID_CONDITIONS)
    if isinstance(raw, str):
        tokens = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    else:
        tokens = [str(chunk).strip() for chunk in raw if str(chunk).strip()]
    if not tokens:
        return list(VALID_CONDITIONS)

    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in VALID_CONDITIONS:
            raise ValueError(
                f"unsupported condition {token!r}; expected one of {', '.join(VALID_CONDITIONS)}"
            )
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def parse_paired_conditions(raw: str | Iterable[str] | None) -> list[str]:
    """Validate that *raw* resolves to exactly the full paired condition set.

    Both ``baseline`` and ``llmdebug`` must be present — no more, no fewer.
    Returns conditions in their canonical order defined by ``VALID_CONDITIONS``.

    Args:
        raw: Condition specifier accepted by :func:`parse_conditions`.

    Returns:
        ``list(VALID_CONDITIONS)`` when validation passes.

    Raises:
        ValueError: If the resolved set does not exactly match
            ``{baseline, llmdebug}``.
    """
    parsed = parse_conditions(raw)
    expected = set(VALID_CONDITIONS)
    if set(parsed) != expected or len(parsed) != len(expected):
        raise ValueError(
            "conditions must include exactly baseline,llmdebug "
            "(both conditions are mandatory for SWE-Bench Pro runs)"
        )
    return list(VALID_CONDITIONS)


def prediction_path(run_dir: Path, condition: str) -> Path:
    """Return the predictions JSONL file path for a condition within a run.

    Args:
        run_dir: Root directory of the run.
        condition: Condition name; must be a member of ``VALID_CONDITIONS``.

    Returns:
        ``<run_dir>/<condition>/predictions.jsonl``.

    Raises:
        ValueError: If *condition* is not a valid condition name.
    """
    _assert_condition(condition)
    return run_dir / condition / "predictions.jsonl"


def metadata_path(run_dir: Path) -> Path:
    """Return the path to the run metadata JSON file.

    Args:
        run_dir: Root directory of the run.

    Returns:
        ``<run_dir>/run_metadata.json``.
    """
    return run_dir / "run_metadata.json"


def instance_ids_path(run_dir: Path) -> Path:
    """Return the path to the plain-text instance IDs file.

    Args:
        run_dir: Root directory of the run.

    Returns:
        ``<run_dir>/instance_ids.txt``.
    """
    return run_dir / "instance_ids.txt"


def copy_instructions_path(run_dir: Path) -> Path:
    """Return the path to the human-readable handoff instructions file.

    Args:
        run_dir: Root directory of the run.

    Returns:
        ``<run_dir>/COPY_INSTRUCTIONS.txt``.
    """
    return run_dir / "COPY_INSTRUCTIONS.txt"


def run_index_path(run_dir: Path) -> Path:
    """Return the path to the run index JSON manifest file.

    Args:
        run_dir: Root directory of the run.

    Returns:
        ``<run_dir>/run_index.json``.
    """
    return run_dir / "run_index.json"


def ensure_run_layout(run_dir: Path, *, conditions: Iterable[str]) -> None:
    """Create the standard on-disk layout for a SWE-Bench run.

    Creates *run_dir* and one subdirectory per condition if they do not
    already exist.  Existing directories are left unchanged.

    Args:
        run_dir: Root directory of the run to initialise.
        conditions: Condition names for which per-condition subdirectories
            should be created; each must be a member of ``VALID_CONDITIONS``.

    Raises:
        ValueError: If any condition name is invalid.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        _assert_condition(condition)
        (run_dir / condition).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Serialise *payload* to an indented UTF-8 JSON file.

    Parent directories are created automatically if they do not exist.
    The file is written with 2-space indentation and a trailing newline.

    Args:
        path: Destination file path.
        payload: JSON-serialisable mapping to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    """Read and parse a JSON object from *path*.

    Args:
        path: Path to a UTF-8 encoded JSON file whose top-level value is an
            object (``{…}``).

    Returns:
        The deserialised JSON object as a plain ``dict``.

    Raises:
        ValueError: If the file does not contain a JSON object at the top level.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def load_instance_ids(file_path: Path) -> list[str]:
    """Read a plain-text instance-ID file and return a deduplicated ordered list.

    Each non-empty line in the file is treated as one instance ID.  Blank
    lines and duplicate entries are silently skipped; first occurrence wins.

    Args:
        file_path: Path to a UTF-8 text file containing one instance ID per
            line.

    Returns:
        Ordered list of unique, non-empty instance ID strings.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for raw in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
        value = raw.strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        ids.append(value)
    return ids


def write_instance_ids(path: Path, instance_ids: Iterable[str]) -> None:
    """Write a deduplicated list of instance IDs to a plain-text file.

    Each unique, non-empty instance ID is written on its own line.
    Duplicates and blank entries are silently discarded; first occurrence
    wins.  The file is encoded as UTF-8.

    Args:
        path: Destination file path.
        instance_ids: Iterable of instance ID strings to persist.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in instance_ids:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    path.write_text("\n".join(cleaned) + ("\n" if cleaned else ""), encoding="utf-8")


def read_predictions(path: Path) -> list[PredictionRecord]:
    """Parse a JSONL predictions file into a list of ``PredictionRecord`` objects.

    Each non-empty line in *path* must be a JSON object with the required
    fields ``instance_id`` (non-empty string), ``model_patch`` (string), and
    ``model_name_or_path`` (non-empty string).  Returns an empty list if
    *path* does not exist.

    Args:
        path: Path to a UTF-8 JSONL file in the SWE-Bench predictions format.

    Returns:
        Ordered list of parsed prediction records.

    Raises:
        ValueError: If any line contains malformed JSON or is missing required
            fields.
    """
    out: list[PredictionRecord] = []
    if not path.exists():
        return out
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no}: prediction row must be object")

        instance_id = payload.get("instance_id")
        model_patch = payload.get("model_patch")
        model_name_or_path = payload.get("model_name_or_path")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"{path}:{line_no}: missing instance_id")
        if not isinstance(model_patch, str):
            raise ValueError(f"{path}:{line_no}: model_patch must be string")
        if not isinstance(model_name_or_path, str) or not model_name_or_path:
            raise ValueError(f"{path}:{line_no}: missing model_name_or_path")

        out.append(
            PredictionRecord(
                instance_id=instance_id,
                model_patch=model_patch,
                model_name_or_path=model_name_or_path,
            )
        )
    return out


def write_predictions(path: Path, records: Iterable[PredictionRecord]) -> None:
    """Atomically write prediction records to a JSONL file.

    Records are first written to a sibling temporary file, then renamed over
    *path* via :func:`os.replace` to avoid partial writes.  The output
    format is one JSON object per line (UTF-8, no BOM) matching the
    SWE-Bench predictions schema.

    Args:
        path: Destination JSONL file path.  Parent directories are created
            automatically if they do not exist.
        records: Prediction records to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record.as_json(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def collect_prediction_ids(path: Path) -> list[str]:
    """Return the ordered list of instance IDs from a predictions JSONL file.

    Args:
        path: Path to a JSONL predictions file; forwarded to
            :func:`read_predictions`.

    Returns:
        List of ``instance_id`` values in file order.
    """
    records = read_predictions(path)
    return [record.instance_id for record in records]


def validate_prediction_file(path: Path) -> tuple[int, int]:
    """Validate a predictions JSONL file and count duplicate instance IDs.

    Args:
        path: Path to a JSONL predictions file; forwarded to
            :func:`read_predictions`.

    Returns:
        A ``(total, duplicates)`` tuple where *total* is the number of parsed
        records and *duplicates* is the count of records whose
        ``instance_id`` appeared in a previous row.
    """
    records = read_predictions(path)
    seen: set[str] = set()
    duplicates = 0
    for record in records:
        if record.instance_id in seen:
            duplicates += 1
        seen.add(record.instance_id)
    return len(records), duplicates


def validate_paired_predictions(
    run_dir: Path,
    *,
    conditions: Iterable[str],
    expected_ids: Iterable[str] | None = None,
    require_files: bool = True,
    require_nonempty: bool = False,
) -> None:
    """Validate that prediction files across conditions are mutually consistent.

    Checks that:

    * Each condition's predictions file exists (when *require_files* is
      ``True``).
    * Each file is non-empty (when *require_nonempty* is ``True``).
    * Every condition contains no duplicate ``instance_id`` rows.
    * All conditions cover exactly the same set of instance IDs.
    * The instance IDs match *expected_ids* when supplied.

    Args:
        run_dir: Root directory of the run.
        conditions: Condition names to validate; forwarded to
            :func:`parse_conditions`.
        expected_ids: Optional reference set of instance IDs that every
            condition's predictions must exactly match.
        require_files: Raise ``ValueError`` when a predictions file is
            missing.
        require_nonempty: Raise ``ValueError`` when a predictions file
            contains zero records.

    Raises:
        ValueError: If any validation check fails.
    """
    condition_list = list(parse_conditions(conditions))
    id_sets: dict[str, set[str]] = {}
    for condition in condition_list:
        condition_path = prediction_path(run_dir, condition)
        if require_files and not condition_path.exists():
            raise ValueError(f"{condition}: missing predictions file ({condition_path})")
        records = read_predictions(condition_path)
        if require_nonempty and not records:
            raise ValueError(f"{condition}: predictions file is empty ({condition_path})")
        ids = [record.instance_id for record in records]
        id_sets[condition] = set(ids)
        if len(ids) != len(id_sets[condition]):
            raise ValueError(f"{condition}: prediction file contains duplicate instance_id rows")

    first = condition_list[0]
    first_ids = id_sets[first]
    for condition in condition_list[1:]:
        if id_sets[condition] != first_ids:
            missing = sorted(first_ids - id_sets[condition])
            extra = sorted(id_sets[condition] - first_ids)
            raise ValueError(
                f"paired prediction mismatch for {condition}: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )

    if expected_ids is not None:
        expected = {item.strip() for item in expected_ids if item.strip()}
        if first_ids != expected:
            missing = sorted(expected - first_ids)
            extra = sorted(first_ids - expected)
            raise ValueError(
                "prediction instance set differs from instance_ids.txt: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )


def copy_instructions_text(run_dir: Path, *, conditions: Iterable[str]) -> str:
    """Generate human-readable handoff instructions for the Docker scoring machine.

    Produces a multi-line text block listing the required files for the run
    and the shell command to execute on the Docker scoring machine.

    Args:
        run_dir: Root directory of the run; used to build absolute file paths
            displayed in the instructions.
        conditions: Conditions whose predictions files are included in the
            required-files list.

    Returns:
        UTF-8 string containing the formatted instructions, ending with a
        newline.
    """
    condition_list = list(parse_conditions(conditions))
    lines: list[str] = [
        "SWE-Bench Pro Handoff",
        "=====================",
        "",
        f"Run dir: {run_dir}",
        "",
        "Required files:",
        f"- {instance_ids_path(run_dir)}",
        f"- {metadata_path(run_dir)}",
    ]
    lines.extend(f"- {prediction_path(run_dir, condition)}" for condition in condition_list)

    lines.extend(
        [
            "",
            "On the Docker scoring machine, from repo root:",
            "1. Copy this entire run directory.",
            "2. Run:",
            "   ./scripts/score_swebench_pro_official.sh --run-dir <copied_run_dir>",
            "",
            "Example:",
            "   ./scripts/score_swebench_pro_official.sh \\",
            "     --run-dir /path/to/evals/swebench_pro_runs/<run_id>",
            "",
            "Outputs will be written under:",
            f"- {run_dir / 'docker_eval'}",
            "",
            f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_copy_instructions(run_dir: Path, *, conditions: Iterable[str]) -> Path:
    """Write the handoff instructions file to *run_dir*.

    Generates the instructions via :func:`copy_instructions_text` and writes
    them to :func:`copy_instructions_path`.

    Args:
        run_dir: Root directory of the run.
        conditions: Conditions forwarded to :func:`copy_instructions_text`.

    Returns:
        Path to the written ``COPY_INSTRUCTIONS.txt`` file.
    """
    text = copy_instructions_text(run_dir, conditions=conditions)
    path = copy_instructions_path(run_dir)
    path.write_text(text, encoding="utf-8")
    return path


def write_run_index(
    run_dir: Path,
    *,
    run_id: str,
    conditions: Iterable[str],
) -> Path:
    """Write a ``run_index.json`` manifest for the specified run.

    The manifest records the run ID, generation timestamp, and absolute paths
    to every artefact required for official SWE-Bench scoring, using schema
    version ``swebench_pro_handoff_index_v1``.

    Args:
        run_dir: Root directory of the run.
        run_id: Identifier for the run; embedded verbatim in the manifest.
        conditions: Conditions whose predictions paths are listed in the
            manifest.

    Returns:
        Path to the written ``run_index.json`` file.
    """
    condition_list = list(parse_conditions(conditions))
    payload: dict[str, object] = {
        "schema_version": "swebench_pro_handoff_index_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "instance_ids_file": str(instance_ids_path(run_dir)),
        "metadata_file": str(metadata_path(run_dir)),
        "predictions": {
            condition: str(prediction_path(run_dir, condition)) for condition in condition_list
        },
        "official_eval_root": str(run_dir / "docker_eval"),
        "required_files": [
            str(instance_ids_path(run_dir)),
            str(metadata_path(run_dir)),
            *(str(prediction_path(run_dir, condition)) for condition in condition_list),
        ],
    }
    path = run_index_path(run_dir)
    write_json(path, payload)
    return path


def _assert_condition(condition: str) -> None:
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"unsupported condition {condition!r}; expected one of {', '.join(VALID_CONDITIONS)}"
        )
