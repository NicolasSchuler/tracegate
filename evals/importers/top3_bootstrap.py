"""Deferred-download bootstrap for top-3 Python datasets.

Implements a strict two-step workflow:
- prepare-top3: lock manifest generation (metadata pinning only, no heavy downloads)
- materialize-top3: artifact materialization + importer orchestration
"""

from __future__ import annotations

import shutil
import stat
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.importers.types import ImportRunSummary
from llmdebug._hashing import CONTENT_HASH_ALGORITHM, new_xxh3_128

from .. import _json as json

DEFAULT_MANIFEST_PATH = Path("evals/manifests/top3.lock.json")
DEFAULT_IMPORT_REPORT_DIR = Path("evals/artifacts/import_reports")
MANIFEST_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1

BUGSINPY_GITHUB_OWNER = "soarsmu"
BUGSINPY_GITHUB_REPO = "BugsInPy"
PYBUGHIVE_GITHUB_OWNER = "pybughive"
PYBUGHIVE_GITHUB_REPO = "pybughive"
PYBUGHIVE_PART1_RECORD_ID = "12800595"
PYBUGHIVE_PART2_RECORD_ID = "12822633"
TRICKYBUGS_RECORD_ID = "10252289"


class ArchiveExtractionError(RuntimeError):
    """Raised when archive extraction violates safety constraints."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class ArtifactSpec:
    """Pinned dataset artifact metadata."""

    artifact_id: str
    url: str
    size_bytes: int | None = None
    filename: str | None = None
    note: str | None = None
    checksum: str | None = None
    required: bool = True

    def resolved_filename(self) -> str:
        if self.filename:
            return self.filename
        return _sanitize_filename_from_url(self.url)


@dataclass(frozen=True)
class DatasetLock:
    """Pinned metadata for one dataset source."""

    source: str
    version: str
    description: str
    artifacts: list[ArtifactSpec]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Top3LockManifest:
    """Top-level lock manifest."""

    schema_version: int
    generated_at_utc: str
    datasets: list[DatasetLock]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Top3LockManifest:
        datasets: list[DatasetLock] = []
        for raw_dataset in payload.get("datasets", []):
            raw_artifacts = raw_dataset.get("artifacts", [])
            artifacts = [
                ArtifactSpec(
                    artifact_id=str(raw_artifact["artifact_id"]),
                    url=str(raw_artifact["url"]),
                    size_bytes=(
                        int(raw_artifact["size_bytes"])
                        if raw_artifact.get("size_bytes") is not None
                        else None
                    ),
                    filename=(
                        str(raw_artifact["filename"])
                        if raw_artifact.get("filename") is not None
                        else None
                    ),
                    note=(
                        str(raw_artifact["note"]) if raw_artifact.get("note") is not None else None
                    ),
                    checksum=(
                        str(raw_artifact["checksum"])
                        if raw_artifact.get("checksum") is not None
                        else None
                    ),
                    required=bool(raw_artifact.get("required", True)),
                )
                for raw_artifact in raw_artifacts
            ]
            datasets.append(
                DatasetLock(
                    source=str(raw_dataset["source"]),
                    version=str(raw_dataset["version"]),
                    description=str(raw_dataset["description"]),
                    artifacts=artifacts,
                    metadata=dict(raw_dataset.get("metadata", {})),
                )
            )
        return cls(
            schema_version=int(payload["schema_version"]),
            generated_at_utc=str(payload["generated_at_utc"]),
            datasets=datasets,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceMaterializationReport:
    """Per-source materialization summary."""

    source: str
    attempted: int = 0
    imported: int = 0
    skipped: int = 0
    hard_failed: bool = False
    skip_reasons: dict[str, int] = field(default_factory=dict)
    imported_case_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class MaterializationReport:
    """Top-level materialization report."""

    schema_version: int
    started_at_utc: str
    finished_at_utc: str | None
    manifest_path: str
    data_root: str
    output_dir: str
    download_policy: str
    max_cases: int
    validate: bool
    dry_run: bool
    jobs: int
    sources: list[SourceMaterializationReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _sanitize_filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return "artifact.bin"
    if len(parts) >= 2 and parts[-1] == "content" and parts[-3:-2] == ["files"]:
        return parts[-2]
    return parts[-1]


def _slug(text: str) -> str:
    return "_".join(
        part for part in "".join(c if c.isalnum() else "_" for c in text).split("_") if part
    )


def _json_request(url: str, *, timeout_sec: int = 60) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "llmdebug-top3-bootstrap/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected JSON payload shape from {url}")
    return payload


def _fetch_zenodo_record(record_id: str) -> dict[str, Any]:
    return _json_request(f"https://zenodo.org/api/records/{record_id}")


def _fetch_github_default_branch_and_head_commit(owner: str, repo: str) -> tuple[str, str]:
    repo_payload = _json_request(f"https://api.github.com/repos/{owner}/{repo}")
    default_branch_raw = repo_payload.get("default_branch")
    if not isinstance(default_branch_raw, str) or not default_branch_raw.strip():
        raise ValueError(f"GitHub API did not return default branch for {owner}/{repo}")
    default_branch = default_branch_raw.strip()

    commit_payload = _json_request(
        f"https://api.github.com/repos/{owner}/{repo}/commits/{urllib.parse.quote(default_branch)}"
    )
    sha_raw = commit_payload.get("sha")
    if not isinstance(sha_raw, str) or not sha_raw.strip():
        raise ValueError(
            f"GitHub API did not return commit sha for {owner}/{repo}@{default_branch}"
        )
    commit_sha = sha_raw.strip().lower()
    if len(commit_sha) != 40 or any(ch not in "0123456789abcdef" for ch in commit_sha):
        raise ValueError(f"Invalid GitHub commit sha for {owner}/{repo}: {commit_sha!r}")
    return default_branch, commit_sha


def _codeload_zip_url_for_commit(owner: str, repo: str, commit_sha: str) -> str:
    return f"https://codeload.github.com/{owner}/{repo}/zip/{commit_sha}"


def _probe_remote_artifact_xxh3_128(url: str, *, timeout_sec: int = 180) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "llmdebug-top3-bootstrap/1.0"})
    total_size = 0
    hasher = new_xxh3_128()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with (
            urllib.request.urlopen(req, timeout=timeout_sec) as response,
            tmp_path.open("wb") as tmp_handle,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                tmp_handle.write(chunk)
                hasher.update(chunk)
                total_size += len(chunk)
    finally:
        tmp_path.unlink(missing_ok=True)
    return total_size, f"{CONTENT_HASH_ALGORITHM}:{hasher.hexdigest()}"


def _resolve_github_repo_archive_artifact(
    *,
    artifact_id: str,
    owner: str,
    repo: str,
    note: str,
) -> tuple[ArtifactSpec, dict[str, str]]:
    default_branch, commit_sha = _fetch_github_default_branch_and_head_commit(owner, repo)
    archive_url = _codeload_zip_url_for_commit(owner, repo, commit_sha)
    size_bytes, checksum = _probe_remote_artifact_xxh3_128(archive_url)
    artifact = ArtifactSpec(
        artifact_id=artifact_id,
        url=archive_url,
        size_bytes=size_bytes,
        filename=f"{repo}_{commit_sha}.zip",
        note=note,
        checksum=checksum,
        required=True,
    )
    repo_metadata = {
        "github_owner": owner,
        "github_repo": repo,
        "github_default_branch": default_branch,
        "github_commit_sha": commit_sha,
    }
    return artifact, repo_metadata


def _artifact_from_zenodo_file(
    *,
    source_prefix: str,
    record_id: str,
    file_entry: dict[str, Any],
) -> ArtifactSpec:
    key = str(file_entry.get("key", "")).strip()
    if not key:
        raise ValueError(f"Zenodo record {record_id} contains file entry without key")
    links = file_entry.get("links")
    if not isinstance(links, dict):
        raise ValueError(f"Zenodo record {record_id} file {key} is missing links")
    url = links.get("self")
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"Zenodo record {record_id} file {key} is missing download URL")

    size_raw = file_entry.get("size")
    size_bytes = int(size_raw) if size_raw is not None else None
    checksum = file_entry.get("checksum")
    checksum_value = str(checksum).strip() if checksum is not None else None

    return ArtifactSpec(
        artifact_id=f"{source_prefix}_{_slug(key.lower())}",
        url=url,
        size_bytes=size_bytes,
        filename=key,
        checksum=checksum_value,
        required=True,
    )


def _build_bugsinpy_lock(
    *,
    repo_archive: ArtifactSpec,
    repo_metadata: dict[str, str],
) -> DatasetLock:
    metadata: dict[str, Any] = {
        "language": "Python",
        "storage_guidance": "small",
    }
    metadata.update(repo_metadata)
    return DatasetLock(
        source="bugsinpy",
        version=f"github-{repo_metadata['github_commit_sha'][:12]}",
        description="BugsInPy metadata/framework repository archive.",
        artifacts=[repo_archive],
        metadata=metadata,
    )


def _build_pybughive_lock(
    part1: dict[str, Any],
    part2: dict[str, Any],
    *,
    repo_archive: ArtifactSpec,
    repo_metadata: dict[str, str],
) -> DatasetLock:
    part1_files = part1.get("files")
    part2_files = part2.get("files")
    if not isinstance(part1_files, list) or not isinstance(part2_files, list):
        raise ValueError("Unexpected Zenodo metadata: PyBugHive record files are missing")

    artifacts: list[ArtifactSpec] = [repo_archive]

    artifacts.extend(
        _artifact_from_zenodo_file(
            source_prefix="offline_part1",
            record_id=PYBUGHIVE_PART1_RECORD_ID,
            file_entry=file_entry,
        )
        for file_entry in sorted(part1_files, key=lambda item: str(item.get("key", "")))
    )
    artifacts.extend(
        _artifact_from_zenodo_file(
            source_prefix="offline_part2",
            record_id=PYBUGHIVE_PART2_RECORD_ID,
            file_entry=file_entry,
        )
        for file_entry in sorted(part2_files, key=lambda item: str(item.get("key", "")))
    )

    offline_total = sum(a.size_bytes or 0 for a in artifacts if a.artifact_id != "repo_zip")
    metadata: dict[str, Any] = {
        "language": "Python",
        "storage_guidance": "very_large",
        "offline_records": [
            f"zenodo:{PYBUGHIVE_PART1_RECORD_ID}",
            f"zenodo:{PYBUGHIVE_PART2_RECORD_ID}",
        ],
        "offline_file_count": len(artifacts) - 1,
        "offline_compressed_total_bytes": offline_total,
    }
    metadata.update(repo_metadata)

    return DatasetLock(
        source="pybughive",
        version=f"v2.0.0+offline-2024-07-pinned-{repo_metadata['github_commit_sha'][:12]}",
        description="PyBugHive metadata repository plus fully pinned offline dump archives.",
        artifacts=artifacts,
        metadata=metadata,
    )


def _build_trickybugs_lock(record: dict[str, Any]) -> DatasetLock:
    files = record.get("files")
    if not isinstance(files, list):
        raise ValueError("Unexpected Zenodo metadata: TrickyBugs files are missing")

    target_file: dict[str, Any] | None = None
    for file_entry in files:
        key = str(file_entry.get("key", "")).strip()
        if key == "TrickyBugs.zip":
            target_file = file_entry
            break
    if target_file is None:
        raise ValueError("TrickyBugs Zenodo record does not contain TrickyBugs.zip")

    artifact = _artifact_from_zenodo_file(
        source_prefix="archive",
        record_id=TRICKYBUGS_RECORD_ID,
        file_entry=target_file,
    )
    artifact = ArtifactSpec(
        artifact_id="archive_zip",
        url=artifact.url,
        size_bytes=artifact.size_bytes,
        filename=artifact.filename,
        note="Main TrickyBugs archive.",
        checksum=artifact.checksum,
        required=True,
    )

    return DatasetLock(
        source="trickybugs",
        version="zenodo-10.5281/zenodo.10252289",
        description="TrickyBugs archive with checksum pinning from Zenodo metadata.",
        artifacts=[artifact],
        metadata={
            "language": "C/C++|Java|Python",
            "storage_guidance": "large",
            "python_supported": True,
            "record": f"zenodo:{TRICKYBUGS_RECORD_ID}",
        },
    )


def default_top3_manifest() -> Top3LockManifest:
    """Create a manifest by pinning current metadata records (no heavy downloads)."""
    bugsinpy_repo_archive, bugsinpy_repo_metadata = _resolve_github_repo_archive_artifact(
        artifact_id="repo_zip",
        owner=BUGSINPY_GITHUB_OWNER,
        repo=BUGSINPY_GITHUB_REPO,
        note=(
            "Small metadata/framework archive; upstream buggy/fixed files are "
            "retrieved during materialization."
        ),
    )
    pybughive_repo_archive, pybughive_repo_metadata = _resolve_github_repo_archive_artifact(
        artifact_id="repo_zip",
        owner=PYBUGHIVE_GITHUB_OWNER,
        repo=PYBUGHIVE_GITHUB_REPO,
        note="PyBugHive metadata repository archive (dataset/pybughive_current.json).",
    )
    pybughive_part1 = _fetch_zenodo_record(PYBUGHIVE_PART1_RECORD_ID)
    pybughive_part2 = _fetch_zenodo_record(PYBUGHIVE_PART2_RECORD_ID)
    trickybugs_record = _fetch_zenodo_record(TRICKYBUGS_RECORD_ID)

    return Top3LockManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        generated_at_utc=_utc_now(),
        datasets=[
            _build_bugsinpy_lock(
                repo_archive=bugsinpy_repo_archive,
                repo_metadata=bugsinpy_repo_metadata,
            ),
            _build_pybughive_lock(
                pybughive_part1,
                pybughive_part2,
                repo_archive=pybughive_repo_archive,
                repo_metadata=pybughive_repo_metadata,
            ),
            _build_trickybugs_lock(trickybugs_record),
        ],
    )


def write_manifest(manifest: Top3LockManifest, path: Path) -> None:
    """Serialize lock manifest to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> Top3LockManifest:
    """Load lock manifest from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Top3LockManifest.from_dict(payload)


def prepare_top3_lock_manifest(manifest_path: Path) -> Top3LockManifest:
    """Generate and persist a lock manifest without downloading heavy payloads."""
    manifest = default_top3_manifest()
    write_manifest(manifest, manifest_path)
    return manifest


def _download_to_path(url: str, dest_path: Path, *, timeout_sec: int = 180) -> None:
    """Download `url` to `dest_path` atomically."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    req = urllib.request.Request(url, headers={"User-Agent": "llmdebug-top3-bootstrap/1.0"})
    with (
        urllib.request.urlopen(req, timeout=timeout_sec) as response,
        tmp_path.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle)
    tmp_path.replace(dest_path)


def _parse_checksum(checksum: str) -> tuple[str, str] | None:
    payload = checksum.strip().lower()
    if ":" not in payload:
        return None
    algorithm, digest = payload.split(":", 1)
    if algorithm not in {"md5", "sha256", CONTENT_HASH_ALGORITHM}:
        return None
    if not digest or any(ch not in "0123456789abcdef" for ch in digest):
        return None
    return algorithm, digest


def _verify_artifact(path: Path, artifact: ArtifactSpec) -> str | None:
    if artifact.size_bytes is not None and path.stat().st_size != artifact.size_bytes:
        return "artifact_size_mismatch"

    if artifact.checksum is None:
        return None
    parsed = _parse_checksum(artifact.checksum)
    if parsed is None:
        return "artifact_checksum_spec_invalid"

    algorithm, expected_digest = parsed
    if algorithm == CONTENT_HASH_ALGORITHM:
        hasher = new_xxh3_128()
    else:
        import hashlib

        hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != expected_digest:
        return "artifact_checksum_mismatch"
    return None


def _ensure_artifact(
    artifact: ArtifactSpec,
    downloads_dir: Path,
    *,
    download_policy: str,
) -> tuple[Path | None, str | None]:
    """Ensure one artifact exists and passes verification semantics."""
    if artifact.required and (artifact.size_bytes is None or artifact.checksum is None):
        return None, f"artifact_integrity_unpinned:{artifact.artifact_id}"

    filename = artifact.resolved_filename()
    dest = downloads_dir / filename

    if download_policy == "never":
        if not dest.exists():
            return None, f"artifact_missing:{artifact.artifact_id}"
        verify_error = _verify_artifact(dest, artifact)
        if verify_error is not None:
            return None, f"{verify_error}:{artifact.artifact_id}"
        return dest, None

    if download_policy == "missing":
        if dest.exists():
            verify_error = _verify_artifact(dest, artifact)
            if verify_error is None:
                return dest, None
            # Existing but corrupt/incomplete => one forced refresh attempt.
            try:
                _download_to_path(artifact.url, dest)
            except Exception as exc:  # pragma: no cover
                return None, f"download_failed:{artifact.artifact_id}:{type(exc).__name__}"
            verify_after = _verify_artifact(dest, artifact)
            if verify_after is not None:
                return None, f"{verify_after}:{artifact.artifact_id}"
            return dest, None

        try:
            _download_to_path(artifact.url, dest)
        except Exception as exc:  # pragma: no cover
            return None, f"download_failed:{artifact.artifact_id}:{type(exc).__name__}"
        verify_error = _verify_artifact(dest, artifact)
        if verify_error is not None:
            return None, f"{verify_error}:{artifact.artifact_id}"
        return dest, None

    if download_policy == "refresh":
        try:
            _download_to_path(artifact.url, dest)
        except Exception as exc:  # pragma: no cover
            return None, f"download_failed:{artifact.artifact_id}:{type(exc).__name__}"
        verify_error = _verify_artifact(dest, artifact)
        if verify_error is not None:
            return None, f"{verify_error}:{artifact.artifact_id}"
        return dest, None

    raise ValueError(f"Invalid download policy: {download_policy}")


def _safe_extract_zip(archive_path: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            member_name = member.filename
            target = _safe_extract_target(extract_dir, member_name)
            mode = (member.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ArchiveExtractionError("archive_symlink_member")

            if member.is_dir() or member_name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _safe_extract_tar(archive_path: Path, extract_dir: Path) -> None:
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            target = _safe_extract_target(extract_dir, member.name)
            if member.issym() or member.islnk():
                raise ArchiveExtractionError("archive_symlink_member")
            if member.ischr() or member.isblk() or member.isfifo():
                raise ArchiveExtractionError("archive_special_member")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ArchiveExtractionError("archive_unsupported_member")

            target.parent.mkdir(parents=True, exist_ok=True)
            fileobj = archive.extractfile(member)
            if fileobj is None:
                raise ArchiveExtractionError("archive_member_read_failed")
            with fileobj, target.open("wb") as dst:
                shutil.copyfileobj(fileobj, dst)


def _safe_extract_target(root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute():
        raise ArchiveExtractionError("archive_absolute_path")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ArchiveExtractionError("archive_path_traversal")
    return root.joinpath(*candidate.parts)


def _extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    """Safely extract a zip/tar archive and return dataset root."""
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    suffixes = "".join(archive_path.suffixes).lower()
    if suffixes.endswith(".zip"):
        _safe_extract_zip(archive_path, extract_dir)
    elif suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or suffixes.endswith(".tar"):
        _safe_extract_tar(archive_path, extract_dir)
    else:
        raise ArchiveExtractionError("archive_unsupported_format")

    children = [p for p in extract_dir.iterdir() if p.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extract_dir


def _find_dataset(manifest: Top3LockManifest, source: str) -> DatasetLock:
    for dataset in manifest.datasets:
        if dataset.source == source:
            return dataset
    raise KeyError(f"Dataset source not found in manifest: {source}")


def _inc_reason(report: SourceMaterializationReport, reason: str, *, count: int = 1) -> None:
    report.skip_reasons[reason] = report.skip_reasons.get(reason, 0) + count


def _sync_summary(report: SourceMaterializationReport, summary: ImportRunSummary) -> None:
    report.attempted = summary.attempted
    report.imported = summary.imported
    report.skipped = summary.skipped
    merged_skip_reasons = dict(report.skip_reasons)
    for reason, count in summary.skip_reasons.items():
        merged_skip_reasons[reason] = merged_skip_reasons.get(reason, 0) + int(count)
    report.skip_reasons = merged_skip_reasons
    report.imported_case_ids = list(summary.imported_case_ids)
    report.notes.extend(summary.notes)


def _materialize_artifacts(
    *,
    dataset: DatasetLock,
    downloads_dir: Path,
    download_policy: str,
    report: SourceMaterializationReport,
) -> tuple[dict[str, Path], bool]:
    local_artifacts: dict[str, Path] = {}
    hard_failed = False

    for artifact in dataset.artifacts:
        local_path, reason = _ensure_artifact(
            artifact,
            downloads_dir,
            download_policy=download_policy,
        )
        if reason is not None:
            _inc_reason(report, reason)
            report.notes.append(f"{artifact.artifact_id}:{reason}")
            if artifact.required:
                hard_failed = True
            continue
        if local_path is None:
            missing_reason = f"artifact_unavailable:{artifact.artifact_id}"
            _inc_reason(report, missing_reason)
            report.notes.append(missing_reason)
            if artifact.required:
                hard_failed = True
            continue
        local_artifacts[artifact.artifact_id] = local_path

    report.hard_failed = report.hard_failed or hard_failed
    return local_artifacts, hard_failed


def _materialize_bugsinpy(
    *,
    output_dir: Path,
    data_root: Path,
    manifest: Top3LockManifest,
    download_policy: str,
    max_cases: int,
    validate: bool,
    dry_run: bool,
    jobs: int,
) -> SourceMaterializationReport:
    from evals.importers.bugsinpy_importer import import_bugsinpy

    dataset = _find_dataset(manifest, "bugsinpy")
    downloads_dir = data_root / "downloads" / "bugsinpy"
    extract_dir = data_root / "extracted" / "bugsinpy"
    report = SourceMaterializationReport(source="bugsinpy")

    local_artifacts, hard_failed = _materialize_artifacts(
        dataset=dataset,
        downloads_dir=downloads_dir,
        download_policy=download_policy,
        report=report,
    )
    if hard_failed:
        report.skipped = max(1, report.skipped)
        return report

    if "repo_zip" not in local_artifacts:
        reason = "artifact_missing:repo_zip"
        _inc_reason(report, reason)
        report.skipped = max(1, report.skipped)
        report.hard_failed = True
        return report

    try:
        repo_root = _extract_archive(local_artifacts["repo_zip"], extract_dir)
    except ArchiveExtractionError as exc:
        reason = f"archive_extract_failed:{exc.reason_code}"
        _inc_reason(report, reason)
        report.notes.append(reason)
        report.hard_failed = True
        report.skipped = max(1, report.skipped)
        return report

    summary = import_bugsinpy(
        output_dir,
        bugsinpy_repo=repo_root,
        data_root=data_root / "workdirs" / "bugsinpy",
        max_cases=max_cases,
        validate=validate,
        dry_run=dry_run,
        jobs=jobs,
    )
    _sync_summary(report, summary)
    return report


def _materialize_pybughive(
    *,
    output_dir: Path,
    data_root: Path,
    manifest: Top3LockManifest,
    download_policy: str,
    max_cases: int,
    validate: bool,
    dry_run: bool,
    jobs: int,
) -> SourceMaterializationReport:
    from evals.importers.pybughive_importer import import_pybughive

    dataset = _find_dataset(manifest, "pybughive")
    downloads_dir = data_root / "downloads" / "pybughive"
    extract_dir = data_root / "extracted" / "pybughive"
    report = SourceMaterializationReport(source="pybughive")

    local_artifacts, hard_failed = _materialize_artifacts(
        dataset=dataset,
        downloads_dir=downloads_dir,
        download_policy=download_policy,
        report=report,
    )
    if hard_failed:
        # Mandatory and authoritative for strict reproducibility.
        report.hard_failed = True
        report.skipped = max(1, report.skipped)
        return report

    repo_zip = local_artifacts.get("repo_zip")
    if repo_zip is None:
        reason = "artifact_missing:repo_zip"
        _inc_reason(report, reason)
        report.notes.append(reason)
        report.hard_failed = True
        report.skipped = max(1, report.skipped)
        return report

    try:
        repo_root = _extract_archive(repo_zip, extract_dir)
    except ArchiveExtractionError as exc:
        reason = f"archive_extract_failed:{exc.reason_code}"
        _inc_reason(report, reason)
        report.notes.append(reason)
        report.hard_failed = True
        report.skipped = max(1, report.skipped)
        return report

    offline_archives = {
        artifact_id: path
        for artifact_id, path in local_artifacts.items()
        if artifact_id != "repo_zip"
    }

    summary = import_pybughive(
        output_dir,
        pybughive_repo=repo_root,
        offline_archives=offline_archives,
        data_root=data_root / "workdirs" / "pybughive",
        max_cases=max_cases,
        validate=validate,
        dry_run=dry_run,
        jobs=jobs,
    )
    _sync_summary(report, summary)
    return report


def _materialize_trickybugs(
    *,
    output_dir: Path,
    data_root: Path,
    manifest: Top3LockManifest,
    download_policy: str,
    max_cases: int,
    validate: bool,
    dry_run: bool,
    jobs: int,
) -> SourceMaterializationReport:
    from evals.importers.trickybugs_importer import import_trickybugs

    dataset = _find_dataset(manifest, "trickybugs")
    downloads_dir = data_root / "downloads" / "trickybugs"
    extract_dir = data_root / "extracted" / "trickybugs"
    report = SourceMaterializationReport(source="trickybugs")

    local_artifacts, hard_failed = _materialize_artifacts(
        dataset=dataset,
        downloads_dir=downloads_dir,
        download_policy=download_policy,
        report=report,
    )
    if hard_failed:
        report.skipped = max(1, report.skipped)
        return report

    archive_zip = local_artifacts.get("archive_zip")
    if archive_zip is None:
        reason = "artifact_missing:archive_zip"
        _inc_reason(report, reason)
        report.notes.append(reason)
        report.hard_failed = True
        report.skipped = max(1, report.skipped)
        return report

    try:
        root = _extract_archive(archive_zip, extract_dir)
    except ArchiveExtractionError as exc:
        reason = f"archive_extract_failed:{exc.reason_code}"
        _inc_reason(report, reason)
        report.notes.append(reason)
        report.hard_failed = True
        report.skipped = max(1, report.skipped)
        return report

    summary = import_trickybugs(
        output_dir,
        trickybugs_root=root,
        data_root=data_root / "workdirs" / "trickybugs",
        max_cases=max_cases,
        validate=validate,
        dry_run=dry_run,
        jobs=jobs,
    )
    _sync_summary(report, summary)
    return report


def materialize_top3(
    *,
    manifest_path: Path,
    output_dir: Path,
    data_root: Path,
    download_policy: str,
    jobs: int,
    max_cases: int,
    validate: bool,
    dry_run: bool,
) -> tuple[MaterializationReport, Path]:
    """Materialize top-3 dataset cases from a lock manifest."""
    if download_policy not in {"never", "missing", "refresh"}:
        raise ValueError(f"Invalid download policy: {download_policy}")

    manifest = load_manifest(manifest_path)
    started_at = _utc_now()
    report = MaterializationReport(
        schema_version=REPORT_SCHEMA_VERSION,
        started_at_utc=started_at,
        finished_at_utc=None,
        manifest_path=str(manifest_path),
        data_root=str(data_root),
        output_dir=str(output_dir),
        download_policy=download_policy,
        max_cases=int(max_cases),
        validate=bool(validate),
        dry_run=bool(dry_run),
        jobs=int(jobs),
        sources=[],
    )

    data_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    report.sources.append(
        _materialize_bugsinpy(
            output_dir=output_dir,
            data_root=data_root,
            manifest=manifest,
            download_policy=download_policy,
            max_cases=max_cases,
            validate=validate,
            dry_run=dry_run,
            jobs=jobs,
        )
    )
    report.sources.append(
        _materialize_pybughive(
            output_dir=output_dir,
            data_root=data_root,
            manifest=manifest,
            download_policy=download_policy,
            max_cases=max_cases,
            validate=validate,
            dry_run=dry_run,
            jobs=jobs,
        )
    )
    report.sources.append(
        _materialize_trickybugs(
            output_dir=output_dir,
            data_root=data_root,
            manifest=manifest,
            download_policy=download_policy,
            max_cases=max_cases,
            validate=validate,
            dry_run=dry_run,
            jobs=jobs,
        )
    )

    report.finished_at_utc = _utc_now()
    report_path = DEFAULT_IMPORT_REPORT_DIR / (
        f"top3-{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report, report_path


def report_has_hard_failures(report: MaterializationReport) -> bool:
    """Whether any source ended in a mandatory hard-failure state."""
    return any(source.hard_failed for source in report.sources)
