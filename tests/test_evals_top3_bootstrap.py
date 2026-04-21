"""Tests for Top-3 dataset bootstrap, artifact extraction, and lock validation."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import evals.importers.top3_bootstrap as top3_bootstrap
from evals.importers.top3_bootstrap import (
    ArchiveExtractionError,
    ArtifactSpec,
    DatasetLock,
    SourceMaterializationReport,
    Top3LockManifest,
    _ensure_artifact,
    _extract_archive,
    _materialize_pybughive,
    _verify_artifact,
    load_manifest,
    materialize_top3,
    prepare_top3_lock_manifest,
    report_has_hard_failures,
    write_manifest,
)
from evals.importers.types import ImportRunSummary
from llmdebug import _json as json
from llmdebug._hashing import xxh3_128_hexdigest_bytes


def _zenodo_record(*, record_id: str, files: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": int(record_id),
        "files": files,
        "metadata": {"version": "test"},
    }


def _file_entry(name: str, size: int, checksum: str, record_id: str) -> dict[str, object]:
    return {
        "key": name,
        "size": size,
        "checksum": checksum,
        "links": {"self": f"https://zenodo.org/api/records/{record_id}/files/{name}/content"},
    }


def _install_mock_zenodo(monkeypatch) -> None:
    responses = {
        top3_bootstrap.PYBUGHIVE_PART1_RECORD_ID: _zenodo_record(
            record_id=top3_bootstrap.PYBUGHIVE_PART1_RECORD_ID,
            files=[
                _file_entry(
                    "numpy.zip",
                    11,
                    "md5:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    top3_bootstrap.PYBUGHIVE_PART1_RECORD_ID,
                ),
                _file_entry(
                    "psf-black.zip",
                    12,
                    "md5:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    top3_bootstrap.PYBUGHIVE_PART1_RECORD_ID,
                ),
            ],
        ),
        top3_bootstrap.PYBUGHIVE_PART2_RECORD_ID: _zenodo_record(
            record_id=top3_bootstrap.PYBUGHIVE_PART2_RECORD_ID,
            files=[
                _file_entry(
                    "pandas_1.zip",
                    13,
                    "md5:cccccccccccccccccccccccccccccccc",
                    top3_bootstrap.PYBUGHIVE_PART2_RECORD_ID,
                )
            ],
        ),
        top3_bootstrap.TRICKYBUGS_RECORD_ID: _zenodo_record(
            record_id=top3_bootstrap.TRICKYBUGS_RECORD_ID,
            files=[
                _file_entry(
                    "TrickyBugs.zip",
                    21,
                    "md5:dddddddddddddddddddddddddddddddd",
                    top3_bootstrap.TRICKYBUGS_RECORD_ID,
                )
            ],
        ),
    }

    def _fake_fetch(record_id: str) -> dict[str, object]:
        return dict(responses[record_id])

    monkeypatch.setattr(top3_bootstrap, "_fetch_zenodo_record", _fake_fetch)

    def _fake_resolve_repo_archive(*, artifact_id: str, owner: str, repo: str, note: str):
        del note
        commit = "a" * 40 if owner == "soarsmu" else "b" * 40
        return (
            ArtifactSpec(
                artifact_id=artifact_id,
                url=f"https://codeload.github.com/{owner}/{repo}/zip/{commit}",
                size_bytes=1234 if owner == "soarsmu" else 2345,
                filename=f"{repo}_{commit}.zip",
                checksum=f"xxh3_128:{commit[:32]}",
                required=True,
            ),
            {
                "github_owner": owner,
                "github_repo": repo,
                "github_default_branch": "main",
                "github_commit_sha": commit,
            },
        )

    monkeypatch.setattr(
        top3_bootstrap,
        "_resolve_github_repo_archive_artifact",
        _fake_resolve_repo_archive,
    )


def test_prepare_top3_manifest_is_deterministic_with_mocked_records(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(top3_bootstrap, "_utc_now", lambda: "2026-01-01T00:00:00+00:00")
    _install_mock_zenodo(monkeypatch)

    manifest_path = tmp_path / "manifests" / "top3.lock.json"
    first = prepare_top3_lock_manifest(manifest_path)
    first_text = manifest_path.read_text(encoding="utf-8")
    second = prepare_top3_lock_manifest(manifest_path)
    second_text = manifest_path.read_text(encoding="utf-8")

    assert first.generated_at_utc == "2026-01-01T00:00:00+00:00"
    assert second.generated_at_utc == "2026-01-01T00:00:00+00:00"
    assert first_text == second_text

    loaded = load_manifest(manifest_path)
    assert loaded.schema_version == 1
    assert [dataset.source for dataset in loaded.datasets] == [
        "bugsinpy",
        "pybughive",
        "trickybugs",
    ]

    pybughive = next(dataset for dataset in loaded.datasets if dataset.source == "pybughive")
    artifact_by_id = {artifact.artifact_id: artifact for artifact in pybughive.artifacts}
    assert "repo_zip" in artifact_by_id
    assert "/zip/refs/heads/" not in artifact_by_id["repo_zip"].url
    assert artifact_by_id["repo_zip"].size_bytes == 2345
    assert artifact_by_id["repo_zip"].checksum == f"xxh3_128:{'b' * 32}"
    assert (
        artifact_by_id["offline_part1_numpy_zip"].checksum == "md5:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )
    assert artifact_by_id["offline_part1_psf_black_zip"].size_bytes == 12
    assert (
        artifact_by_id["offline_part2_pandas_1_zip"].checksum
        == "md5:cccccccccccccccccccccccccccccccc"
    )

    bugsinpy = next(dataset for dataset in loaded.datasets if dataset.source == "bugsinpy")
    bugsinpy_repo_zip = bugsinpy.artifacts[0]
    assert "/zip/refs/heads/" not in bugsinpy_repo_zip.url
    assert bugsinpy_repo_zip.checksum == f"xxh3_128:{'a' * 32}"
    assert bugsinpy.metadata["github_commit_sha"] == "a" * 40


def test_prepare_top3_does_not_download_heavy_artifacts(tmp_path, monkeypatch) -> None:
    called = {"download_called": False}

    def _fake_download(*args, **kwargs):
        called["download_called"] = True
        raise AssertionError("prepare_top3_lock_manifest must not download artifacts")

    _install_mock_zenodo(monkeypatch)
    monkeypatch.setattr(top3_bootstrap, "_download_to_path", _fake_download)

    manifest_path = tmp_path / "manifests" / "top3.lock.json"
    prepare_top3_lock_manifest(manifest_path)

    assert manifest_path.exists()
    assert called["download_called"] is False
    assert not (tmp_path / "downloads").exists()
    assert not (tmp_path / "extracted").exists()


def test_safe_extract_rejects_traversal_zip_entries(tmp_path) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("../evil.txt", "x")

    try:
        _extract_archive(archive_path, tmp_path / "extracted")
    except ArchiveExtractionError as exc:
        assert exc.reason_code == "archive_path_traversal"
    else:  # pragma: no cover
        raise AssertionError("Expected ArchiveExtractionError")


def test_ensure_artifact_missing_policy_redownloads_corrupt_existing(monkeypatch, tmp_path) -> None:
    good_bytes = b"expected payload"
    checksum = hashlib.md5(good_bytes).hexdigest()

    artifact = ArtifactSpec(
        artifact_id="demo",
        url="https://example.invalid/demo.bin",
        size_bytes=len(good_bytes),
        filename="demo.bin",
        checksum=f"md5:{checksum}",
    )
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    existing = downloads_dir / "demo.bin"
    existing.write_bytes(b"corrupt")

    def _fake_download(url: str, dest_path: Path, *, timeout_sec: int = 180) -> None:
        del url, timeout_sec
        dest_path.write_bytes(good_bytes)

    monkeypatch.setattr(top3_bootstrap, "_download_to_path", _fake_download)

    resolved, reason = _ensure_artifact(artifact, downloads_dir, download_policy="missing")
    assert reason is None
    assert resolved == existing
    assert existing.read_bytes() == good_bytes


def test_verify_artifact_accepts_xxh3_128_checksum(tmp_path) -> None:
    payload = b"expected payload"
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)
    artifact = ArtifactSpec(
        artifact_id="demo",
        url="https://example.invalid/demo.bin",
        filename="artifact.bin",
        size_bytes=len(payload),
        checksum=f"xxh3_128:{xxh3_128_hexdigest_bytes(payload)}",
    )

    assert _verify_artifact(path, artifact) is None


def test_verify_artifact_accepts_legacy_sha256_checksum(tmp_path) -> None:
    payload = b"expected payload"
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)
    artifact = ArtifactSpec(
        artifact_id="demo",
        url="https://example.invalid/demo.bin",
        filename="artifact.bin",
        size_bytes=len(payload),
        checksum=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )

    assert _verify_artifact(path, artifact) is None


def test_materialize_pybughive_hard_fails_when_mandatory_artifacts_missing(tmp_path) -> None:
    manifest = Top3LockManifest(
        schema_version=1,
        generated_at_utc="2026-01-01T00:00:00+00:00",
        datasets=[
            DatasetLock(
                source="pybughive",
                version="test",
                description="test",
                artifacts=[
                    ArtifactSpec(
                        artifact_id="repo_zip",
                        url="https://example.invalid/repo.zip",
                        filename="repo.zip",
                        size_bytes=10,
                        checksum="sha256:" + ("a" * 64),
                        required=True,
                    ),
                    ArtifactSpec(
                        artifact_id="offline_part1_numpy_zip",
                        url="https://example.invalid/numpy.zip",
                        filename="numpy.zip",
                        size_bytes=20,
                        checksum="sha256:" + ("b" * 64),
                        required=True,
                    ),
                ],
            )
        ],
    )

    report = _materialize_pybughive(
        output_dir=tmp_path / "cases",
        data_root=tmp_path / "data",
        manifest=manifest,
        download_policy="never",
        max_cases=0,
        validate=False,
        dry_run=True,
        jobs=1,
    )

    assert report.source == "pybughive"
    assert report.hard_failed is True
    assert report.skip_reasons["artifact_missing:repo_zip"] == 1
    assert report.skip_reasons["artifact_missing:offline_part1_numpy_zip"] == 1


def test_materialize_pybughive_hard_fails_when_required_artifact_is_unpinned(tmp_path) -> None:
    manifest = Top3LockManifest(
        schema_version=1,
        generated_at_utc="2026-01-01T00:00:00+00:00",
        datasets=[
            DatasetLock(
                source="pybughive",
                version="test",
                description="test",
                artifacts=[
                    ArtifactSpec(
                        artifact_id="repo_zip",
                        url="https://example.invalid/repo.zip",
                        filename="repo.zip",
                        required=True,
                        size_bytes=None,
                        checksum=None,
                    )
                ],
            )
        ],
    )

    report = _materialize_pybughive(
        output_dir=tmp_path / "cases",
        data_root=tmp_path / "data",
        manifest=manifest,
        download_policy="never",
        max_cases=0,
        validate=False,
        dry_run=True,
        jobs=1,
    )

    assert report.hard_failed is True
    assert report.skip_reasons == {"artifact_integrity_unpinned:repo_zip": 1}


def test_sync_summary_merges_existing_artifact_skip_reasons() -> None:
    report = SourceMaterializationReport(
        source="bugsinpy",
        skip_reasons={"artifact_missing:repo_zip": 1},
        notes=["artifact:download_failed"],
    )
    summary = ImportRunSummary(
        source="bugsinpy",
        attempted=5,
        imported=3,
        skipped=2,
        skip_reasons={"validation_failed": 2, "artifact_missing:repo_zip": 1},
        imported_case_ids=["case-1", "case-2", "case-3"],
        notes=["importer:validation_failed"],
    )

    top3_bootstrap._sync_summary(report, summary)

    assert report.attempted == 5
    assert report.imported == 3
    assert report.skipped == 2
    assert report.skip_reasons == {"artifact_missing:repo_zip": 2, "validation_failed": 2}
    assert report.notes == ["artifact:download_failed", "importer:validation_failed"]


def test_materialize_top3_writes_report_and_hard_failure_signal(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "top3.lock.json"
    write_manifest(
        Top3LockManifest(
            schema_version=1, generated_at_utc="2026-01-01T00:00:00+00:00", datasets=[]
        ),
        manifest_path,
    )

    monkeypatch.setattr(
        top3_bootstrap,
        "_materialize_bugsinpy",
        lambda **kwargs: SourceMaterializationReport(
            source="bugsinpy",
            attempted=2,
            imported=1,
            skipped=1,
            hard_failed=False,
            skip_reasons={"x": 1},
            imported_case_ids=["b1"],
            notes=[],
        ),
    )
    monkeypatch.setattr(
        top3_bootstrap,
        "_materialize_pybughive",
        lambda **kwargs: SourceMaterializationReport(
            source="pybughive",
            attempted=3,
            imported=0,
            skipped=3,
            hard_failed=True,
            skip_reasons={"artifact_missing:repo_zip": 1},
            imported_case_ids=[],
            notes=["mandatory failure"],
        ),
    )
    monkeypatch.setattr(
        top3_bootstrap,
        "_materialize_trickybugs",
        lambda **kwargs: SourceMaterializationReport(
            source="trickybugs",
            attempted=1,
            imported=1,
            skipped=0,
            hard_failed=False,
            skip_reasons={},
            imported_case_ids=["t1"],
            notes=[],
        ),
    )

    report, report_path = materialize_top3(
        manifest_path=manifest_path,
        output_dir=tmp_path / "cases",
        data_root=tmp_path / "data",
        download_policy="never",
        jobs=2,
        max_cases=7,
        validate=False,
        dry_run=True,
    )

    assert report_path.exists()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1
    assert [entry["source"] for entry in persisted["sources"]] == [
        "bugsinpy",
        "pybughive",
        "trickybugs",
    ]
    assert report_has_hard_failures(report) is True
