"""Additional tests for evals/importers/top3_bootstrap.py to increase branch coverage.

Focuses on uncovered code paths: URL sanitization, checksum parsing, archive
extraction (zip/tar/unsupported), _ensure_artifact policy branches, Zenodo
file validation errors, _find_dataset miss, _materialize_artifacts unavailable
branch, _build_* error paths, and materialize_top3 invalid policy.
"""

from __future__ import annotations

import hashlib
import io
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

import evals.importers.top3_bootstrap as top3_bootstrap
from evals.importers.top3_bootstrap import (
    ArchiveExtractionError,
    ArtifactSpec,
    DatasetLock,
    MaterializationReport,
    SourceMaterializationReport,
    Top3LockManifest,
    _artifact_from_zenodo_file,
    _build_pybughive_lock,
    _build_trickybugs_lock,
    _codeload_zip_url_for_commit,
    _ensure_artifact,
    _extract_archive,
    _find_dataset,
    _materialize_artifacts,
    _parse_checksum,
    _safe_extract_target,
    _safe_extract_zip,
    _sanitize_filename_from_url,
    _verify_artifact,
    materialize_top3,
    report_has_hard_failures,
    write_manifest,
)

# ---------------------------------------------------------------------------
# ArtifactSpec.resolved_filename fallback (line 64)
# ---------------------------------------------------------------------------


def test_resolved_filename_falls_back_to_url_sanitization() -> None:
    """resolved_filename uses URL sanitization when filename is None."""
    spec = ArtifactSpec(
        artifact_id="test",
        url="https://example.com/path/to/data.bin",
        filename=None,
    )
    assert spec.resolved_filename() == "data.bin"


# ---------------------------------------------------------------------------
# _sanitize_filename_from_url (lines 176-182)
# ---------------------------------------------------------------------------


class TestSanitizeFilenameFromUrl:
    """Tests for URL-to-filename sanitization logic."""

    def test_empty_path_returns_default(self) -> None:
        """URLs with empty path yield 'artifact.bin'."""
        assert _sanitize_filename_from_url("https://example.com") == "artifact.bin"

    def test_zenodo_content_url_extracts_filename(self) -> None:
        """Zenodo /files/<name>/content URLs return the file name."""
        url = "https://zenodo.org/api/records/123/files/data.zip/content"
        assert _sanitize_filename_from_url(url) == "data.zip"

    def test_simple_path_returns_last_segment(self) -> None:
        """Simple paths return the last segment."""
        url = "https://example.com/downloads/archive.tar.gz"
        assert _sanitize_filename_from_url(url) == "archive.tar.gz"


# ---------------------------------------------------------------------------
# _parse_checksum (lines 510-519)
# ---------------------------------------------------------------------------


class TestParseChecksum:
    """Tests for checksum string parsing."""

    def test_no_colon_returns_none(self) -> None:
        """Strings without ':' are rejected."""
        assert _parse_checksum("deadbeef") is None

    def test_unsupported_algorithm_returns_none(self) -> None:
        """Unknown hash algorithms are rejected."""
        assert _parse_checksum("blake2b:abcdef") is None

    def test_non_hex_digest_returns_none(self) -> None:
        """Digests with non-hex characters are rejected."""
        assert _parse_checksum("md5:zzzzzz") is None

    def test_empty_digest_returns_none(self) -> None:
        """Empty digest after colon is rejected."""
        assert _parse_checksum("md5:") is None

    def test_valid_md5_parsed(self) -> None:
        """Valid md5 checksum is parsed correctly."""
        result = _parse_checksum("md5:aabbccdd")
        assert result == ("md5", "aabbccdd")

    def test_valid_sha256_parsed(self) -> None:
        """Valid sha256 checksum is parsed correctly."""
        result = _parse_checksum("sha256:" + "ab" * 32)
        assert result == ("sha256", "ab" * 32)

    def test_valid_xxh3_128_parsed(self) -> None:
        """Valid xxh3_128 checksum is parsed correctly."""
        result = _parse_checksum("xxh3_128:aabbccdd11223344")
        assert result == ("xxh3_128", "aabbccdd11223344")


# ---------------------------------------------------------------------------
# _verify_artifact additional branches (lines 524-543)
# ---------------------------------------------------------------------------


class TestVerifyArtifact:
    """Tests for artifact verification edge cases."""

    def test_size_mismatch_detected(self, tmp_path: Path) -> None:
        """Size mismatch returns 'artifact_size_mismatch'."""
        path = tmp_path / "file.bin"
        path.write_bytes(b"short")
        spec = ArtifactSpec(artifact_id="a", url="http://x", size_bytes=9999)
        assert _verify_artifact(path, spec) == "artifact_size_mismatch"

    def test_none_checksum_passes(self, tmp_path: Path) -> None:
        """Artifacts with no checksum skip verification."""
        path = tmp_path / "file.bin"
        path.write_bytes(b"data")
        spec = ArtifactSpec(
            artifact_id="a",
            url="http://x",
            size_bytes=4,
            checksum=None,
        )
        assert _verify_artifact(path, spec) is None

    def test_invalid_checksum_spec(self, tmp_path: Path) -> None:
        """Unparseable checksum returns 'artifact_checksum_spec_invalid'."""
        path = tmp_path / "file.bin"
        path.write_bytes(b"data")
        spec = ArtifactSpec(
            artifact_id="a",
            url="http://x",
            size_bytes=4,
            checksum="bogus_no_colon",
        )
        assert _verify_artifact(path, spec) == "artifact_checksum_spec_invalid"

    def test_checksum_mismatch_detected(self, tmp_path: Path) -> None:
        """Wrong digest returns 'artifact_checksum_mismatch'."""
        path = tmp_path / "file.bin"
        path.write_bytes(b"data")
        spec = ArtifactSpec(
            artifact_id="a",
            url="http://x",
            size_bytes=4,
            checksum="md5:" + "0" * 32,
        )
        assert _verify_artifact(path, spec) == "artifact_checksum_mismatch"


# ---------------------------------------------------------------------------
# _safe_extract_target (lines 645-652)
# ---------------------------------------------------------------------------


class TestSafeExtractTarget:
    """Tests for path traversal and absolute path checks."""

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        """Absolute member paths raise ArchiveExtractionError."""
        with pytest.raises(ArchiveExtractionError, match="archive_absolute_path"):
            _safe_extract_target(tmp_path, "/etc/passwd")

    def test_normal_path_resolved(self, tmp_path: Path) -> None:
        """Normal paths resolve correctly."""
        result = _safe_extract_target(tmp_path, "subdir/file.txt")
        assert result == tmp_path / "subdir" / "file.txt"


# ---------------------------------------------------------------------------
# _safe_extract_zip: symlink member (lines 610-620)
# ---------------------------------------------------------------------------


def _make_zip_with_symlink(archive_path: Path) -> None:
    """Create a zip file containing a symlink entry."""
    with zipfile.ZipFile(archive_path, "w") as zf:
        info = zipfile.ZipInfo("link.txt")
        # Set the external_attr to indicate a symlink
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "target.txt")


def test_safe_extract_zip_rejects_symlink(tmp_path: Path) -> None:
    """Zip files with symlink members are rejected."""
    archive_path = tmp_path / "sym.zip"
    _make_zip_with_symlink(archive_path)
    with pytest.raises(ArchiveExtractionError, match="archive_symlink_member"):
        _safe_extract_zip(archive_path, tmp_path / "out")


def test_safe_extract_zip_extracts_directory_and_file(tmp_path: Path) -> None:
    """Normal zip extraction works for directories and files."""
    archive_path = tmp_path / "good.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("subdir/", "")
        zf.writestr("subdir/hello.txt", "world")
    out = tmp_path / "out"
    _safe_extract_zip(archive_path, out)
    assert (out / "subdir" / "hello.txt").read_text() == "world"


# ---------------------------------------------------------------------------
# _safe_extract_tar (lines 624-642)
# ---------------------------------------------------------------------------


def _make_tar_with_symlink(archive_path: Path) -> None:
    """Create a tar with a symlink member."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="link.txt")
        info.type = tarfile.SYMTYPE
        info.linkname = "target.txt"
        tf.addfile(info)
    archive_path.write_bytes(buf.getvalue())


def _make_tar_with_fifo(archive_path: Path) -> None:
    """Create a tar with a FIFO (special) member."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="my_fifo")
        info.type = tarfile.FIFOTYPE
        tf.addfile(info)
    archive_path.write_bytes(buf.getvalue())


def _make_tar_with_normal_file(archive_path: Path) -> None:
    """Create a tar with a normal file and a directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        # Add a directory
        dir_info = tarfile.TarInfo(name="mydir")
        dir_info.type = tarfile.DIRTYPE
        tf.addfile(dir_info)
        # Add a file
        content = b"hello tar"
        file_info = tarfile.TarInfo(name="mydir/file.txt")
        file_info.size = len(content)
        tf.addfile(file_info, io.BytesIO(content))
    archive_path.write_bytes(buf.getvalue())


def test_safe_extract_tar_rejects_symlink(tmp_path: Path) -> None:
    """Tar files with symlink members are rejected."""
    archive_path = tmp_path / "sym.tar.gz"
    _make_tar_with_symlink(archive_path)
    with pytest.raises(ArchiveExtractionError, match="archive_symlink_member"):
        from evals.importers.top3_bootstrap import _safe_extract_tar

        _safe_extract_tar(archive_path, tmp_path / "out")


def test_safe_extract_tar_rejects_fifo(tmp_path: Path) -> None:
    """Tar files with FIFO members are rejected."""
    archive_path = tmp_path / "fifo.tar.gz"
    _make_tar_with_fifo(archive_path)
    with pytest.raises(ArchiveExtractionError, match="archive_special_member"):
        from evals.importers.top3_bootstrap import _safe_extract_tar

        _safe_extract_tar(archive_path, tmp_path / "out")


def test_safe_extract_tar_extracts_normal_file(tmp_path: Path) -> None:
    """Normal tar extraction works for directories and files."""
    archive_path = tmp_path / "good.tar.gz"
    _make_tar_with_normal_file(archive_path)

    from evals.importers.top3_bootstrap import _safe_extract_tar

    out = tmp_path / "out"
    out.mkdir()
    _safe_extract_tar(archive_path, out)
    assert (out / "mydir" / "file.txt").read_text() == "hello tar"


# ---------------------------------------------------------------------------
# _extract_archive: format dispatch + children logic (lines 655-672)
# ---------------------------------------------------------------------------


class TestExtractArchive:
    """Tests for archive format dispatch and root detection."""

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        """Unsupported archive formats are rejected."""
        archive_path = tmp_path / "data.rar"
        archive_path.write_bytes(b"fake")
        with pytest.raises(ArchiveExtractionError, match="archive_unsupported_format"):
            _extract_archive(archive_path, tmp_path / "out")

    def test_existing_extract_dir_cleaned(self, tmp_path: Path) -> None:
        """Pre-existing extract directory is removed before extraction."""
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        (extract_dir / "stale.txt").write_text("old")

        archive_path = tmp_path / "good.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("root/file.txt", "new")

        result = _extract_archive(archive_path, extract_dir)
        assert result.name == "root"
        assert not (extract_dir / "stale.txt").exists()

    def test_single_child_dir_unwrapped(self, tmp_path: Path) -> None:
        """Single top-level directory is returned as dataset root."""
        archive_path = tmp_path / "wrapped.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("dataset-v1/data.csv", "a,b")

        result = _extract_archive(archive_path, tmp_path / "out")
        assert result.name == "dataset-v1"

    def test_multiple_children_returns_extract_dir(self, tmp_path: Path) -> None:
        """Multiple top-level items return the extract directory itself."""
        archive_path = tmp_path / "flat.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("file1.txt", "a")
            zf.writestr("file2.txt", "b")

        extract_dir = tmp_path / "out"
        result = _extract_archive(archive_path, extract_dir)
        assert result == extract_dir

    def test_tar_gz_extraction(self, tmp_path: Path) -> None:
        """tar.gz archives are dispatched to _safe_extract_tar."""
        archive_path = tmp_path / "data.tar.gz"
        _make_tar_with_normal_file(archive_path)

        result = _extract_archive(archive_path, tmp_path / "out")
        assert result.name == "mydir"

    def test_macosx_dir_ignored_in_children(self, tmp_path: Path) -> None:
        """__MACOSX directories are excluded from children enumeration."""
        archive_path = tmp_path / "mac.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("dataset/data.csv", "a,b")
            zf.writestr("__MACOSX/._data.csv", "")

        result = _extract_archive(archive_path, tmp_path / "out")
        assert result.name == "dataset"


# ---------------------------------------------------------------------------
# _find_dataset (line 679)
# ---------------------------------------------------------------------------


def test_find_dataset_raises_on_missing_source() -> None:
    """_find_dataset raises KeyError for unknown sources."""
    manifest = Top3LockManifest(schema_version=1, generated_at_utc="now", datasets=[])
    with pytest.raises(KeyError, match="nonexistent"):
        _find_dataset(manifest, "nonexistent")


# ---------------------------------------------------------------------------
# _artifact_from_zenodo_file error paths (lines 294, 297, 300)
# ---------------------------------------------------------------------------


class TestArtifactFromZenodoFile:
    """Tests for Zenodo file entry validation."""

    def test_missing_key_raises(self) -> None:
        """File entries without key are rejected."""
        with pytest.raises(ValueError, match="without key"):
            _artifact_from_zenodo_file(
                source_prefix="p",
                record_id="123",
                file_entry={"key": "", "links": {"self": "http://x"}},
            )

    def test_missing_links_raises(self) -> None:
        """File entries without links dict are rejected."""
        with pytest.raises(ValueError, match="missing links"):
            _artifact_from_zenodo_file(
                source_prefix="p",
                record_id="123",
                file_entry={"key": "data.zip"},
            )

    def test_missing_download_url_raises(self) -> None:
        """File entries with empty download URL are rejected."""
        with pytest.raises(ValueError, match="missing download URL"):
            _artifact_from_zenodo_file(
                source_prefix="p",
                record_id="123",
                file_entry={
                    "key": "data.zip",
                    "links": {"self": ""},
                },
            )

    def test_valid_entry_parsed(self) -> None:
        """Valid Zenodo file entries produce correct ArtifactSpec."""
        spec = _artifact_from_zenodo_file(
            source_prefix="part1",
            record_id="12345",
            file_entry={
                "key": "numpy.zip",
                "size": 1024,
                "checksum": "md5:aabb",
                "links": {"self": "https://zenodo.org/files/numpy.zip"},
            },
        )
        assert spec.artifact_id == "part1_numpy_zip"
        assert spec.size_bytes == 1024
        assert spec.checksum == "md5:aabb"


# ---------------------------------------------------------------------------
# _build_pybughive_lock error path (line 346)
# ---------------------------------------------------------------------------


def test_build_pybughive_lock_raises_on_missing_files() -> None:
    """_build_pybughive_lock raises when part files are not lists."""
    repo = ArtifactSpec(artifact_id="repo_zip", url="http://x")
    meta = {"github_commit_sha": "a" * 40}
    with pytest.raises(ValueError, match="PyBugHive record files"):
        _build_pybughive_lock(
            {"files": None},
            {"files": []},
            repo_archive=repo,
            repo_metadata=meta,
        )


# ---------------------------------------------------------------------------
# _build_trickybugs_lock error paths (lines 392, 401)
# ---------------------------------------------------------------------------


class TestBuildTrickybugsLock:
    """Tests for TrickyBugs lock builder validation."""

    def test_missing_files_key_raises(self) -> None:
        """Records without files list are rejected."""
        with pytest.raises(ValueError, match="TrickyBugs files are missing"):
            _build_trickybugs_lock({"files": "not_a_list"})

    def test_missing_target_zip_raises(self) -> None:
        """Records without TrickyBugs.zip are rejected."""
        entry = {
            "key": "other.zip",
            "size": 10,
            "checksum": "md5:aa",
            "links": {"self": "https://z.org/other.zip"},
        }
        with pytest.raises(ValueError, match=r"does not contain TrickyBugs\.zip"):
            _build_trickybugs_lock({"files": [entry]})

    def test_valid_record_builds_lock(self) -> None:
        """Valid record with TrickyBugs.zip builds DatasetLock."""
        entry = {
            "key": "TrickyBugs.zip",
            "size": 500,
            "checksum": "md5:ccdd",
            "links": {"self": "https://zenodo.org/files/TrickyBugs.zip"},
        }
        lock = _build_trickybugs_lock({"files": [entry]})
        assert lock.source == "trickybugs"
        assert lock.artifacts[0].artifact_id == "archive_zip"
        assert lock.artifacts[0].size_bytes == 500


# ---------------------------------------------------------------------------
# _codeload_zip_url_for_commit (line 232)
# ---------------------------------------------------------------------------


def test_codeload_zip_url_for_commit() -> None:
    """Generates correct codeload URL from owner/repo/sha."""
    url = _codeload_zip_url_for_commit("owner", "repo", "abc123")
    assert url == "https://codeload.github.com/owner/repo/zip/abc123"


# ---------------------------------------------------------------------------
# _ensure_artifact: "never" policy with verified file (lines 563-566)
# ---------------------------------------------------------------------------


class TestEnsureArtifactNeverPolicy:
    """Tests for 'never' download policy branches."""

    def test_never_policy_returns_verified_file(self, tmp_path: Path) -> None:
        """'never' policy returns existing, verified file."""
        payload = b"good data"
        checksum = hashlib.md5(payload).hexdigest()
        dest = tmp_path / "demo.bin"
        dest.write_bytes(payload)

        spec = ArtifactSpec(
            artifact_id="demo",
            url="http://x/demo.bin",
            size_bytes=len(payload),
            filename="demo.bin",
            checksum=f"md5:{checksum}",
        )
        path, reason = _ensure_artifact(spec, tmp_path, download_policy="never")
        assert reason is None
        assert path == dest

    def test_never_policy_verify_fail(self, tmp_path: Path) -> None:
        """'never' policy returns error when verification fails."""
        dest = tmp_path / "demo.bin"
        dest.write_bytes(b"corrupt")

        spec = ArtifactSpec(
            artifact_id="demo",
            url="http://x/demo.bin",
            size_bytes=99,
            filename="demo.bin",
            checksum="md5:" + "a" * 32,
        )
        path, reason = _ensure_artifact(spec, tmp_path, download_policy="never")
        assert path is None
        assert reason is not None
        assert "demo" in reason


# ---------------------------------------------------------------------------
# _ensure_artifact: "missing" policy - file exists and verifies (line 572)
# ---------------------------------------------------------------------------


def test_ensure_artifact_missing_policy_existing_valid(
    tmp_path: Path,
) -> None:
    """'missing' policy skips download when existing file is valid."""
    payload = b"ok"
    checksum = hashlib.md5(payload).hexdigest()
    dest = tmp_path / "file.bin"
    dest.write_bytes(payload)

    spec = ArtifactSpec(
        artifact_id="f",
        url="http://x/file.bin",
        size_bytes=len(payload),
        filename="file.bin",
        checksum=f"md5:{checksum}",
    )
    path, reason = _ensure_artifact(spec, tmp_path, download_policy="missing")
    assert reason is None
    assert path == dest


# ---------------------------------------------------------------------------
# _ensure_artifact: "missing" policy - redownload still fails (line 580)
# ---------------------------------------------------------------------------


def test_ensure_artifact_missing_policy_redownload_still_bad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """'missing' policy reports error when redownload still fails verification."""
    bad_payload = b"still bad"
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"corrupt initial")

    def _fake_download(url: str, dest_path: Path, *, timeout_sec: int = 180) -> None:
        dest_path.write_bytes(bad_payload)

    monkeypatch.setattr(top3_bootstrap, "_download_to_path", _fake_download)

    spec = ArtifactSpec(
        artifact_id="f",
        url="http://x/file.bin",
        size_bytes=99999,
        filename="file.bin",
        checksum="md5:" + "0" * 32,
    )
    path, reason = _ensure_artifact(spec, tmp_path, download_policy="missing")
    assert path is None
    assert reason is not None
    assert "f" in reason


# ---------------------------------------------------------------------------
# _ensure_artifact: "missing" policy - fresh download (lines 583-590)
# ---------------------------------------------------------------------------


def test_ensure_artifact_missing_policy_fresh_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """'missing' policy downloads when file does not exist yet."""
    payload = b"fresh data"
    checksum = hashlib.md5(payload).hexdigest()

    def _fake_download(url: str, dest_path: Path, *, timeout_sec: int = 180) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(payload)

    monkeypatch.setattr(top3_bootstrap, "_download_to_path", _fake_download)

    spec = ArtifactSpec(
        artifact_id="new",
        url="http://x/new.bin",
        size_bytes=len(payload),
        filename="new.bin",
        checksum=f"md5:{checksum}",
    )
    path, reason = _ensure_artifact(spec, tmp_path, download_policy="missing")
    assert reason is None
    assert path is not None
    assert path.read_bytes() == payload


def test_ensure_artifact_missing_fresh_verify_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """'missing' policy (fresh) reports error on verify failure."""

    def _fake_download(url: str, dest_path: Path, *, timeout_sec: int = 180) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"wrong bytes")

    monkeypatch.setattr(top3_bootstrap, "_download_to_path", _fake_download)

    spec = ArtifactSpec(
        artifact_id="bad",
        url="http://x/bad.bin",
        size_bytes=99,
        filename="bad.bin",
        checksum="md5:" + "0" * 32,
    )
    path, reason = _ensure_artifact(spec, tmp_path, download_policy="missing")
    assert path is None
    assert reason is not None


# ---------------------------------------------------------------------------
# _ensure_artifact: "refresh" policy (lines 592-602)
# ---------------------------------------------------------------------------


def test_ensure_artifact_refresh_policy_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """'refresh' policy always downloads and verifies."""
    payload = b"refreshed"
    checksum = hashlib.md5(payload).hexdigest()

    def _fake_download(url: str, dest_path: Path, *, timeout_sec: int = 180) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(payload)

    monkeypatch.setattr(top3_bootstrap, "_download_to_path", _fake_download)

    spec = ArtifactSpec(
        artifact_id="r",
        url="http://x/r.bin",
        size_bytes=len(payload),
        filename="r.bin",
        checksum=f"md5:{checksum}",
    )
    path, reason = _ensure_artifact(spec, tmp_path, download_policy="refresh")
    assert reason is None
    assert path is not None


def test_ensure_artifact_refresh_policy_verify_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """'refresh' policy reports error when verification fails."""

    def _fake_download(url: str, dest_path: Path, *, timeout_sec: int = 180) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"wrong")

    monkeypatch.setattr(top3_bootstrap, "_download_to_path", _fake_download)

    spec = ArtifactSpec(
        artifact_id="r",
        url="http://x/r.bin",
        size_bytes=99,
        filename="r.bin",
        checksum="md5:" + "0" * 32,
    )
    path, reason = _ensure_artifact(spec, tmp_path, download_policy="refresh")
    assert path is None
    assert reason is not None


# ---------------------------------------------------------------------------
# _ensure_artifact: invalid policy (line 602)
# ---------------------------------------------------------------------------


def test_ensure_artifact_invalid_policy_raises(tmp_path: Path) -> None:
    """Invalid download policy raises ValueError."""
    spec = ArtifactSpec(
        artifact_id="x",
        url="http://x",
        size_bytes=1,
        filename="x",
        checksum="md5:aa",
    )
    with pytest.raises(ValueError, match="Invalid download policy"):
        _ensure_artifact(spec, tmp_path, download_policy="bogus")


# ---------------------------------------------------------------------------
# _materialize_artifacts: local_path=None branch (lines 720-727)
# ---------------------------------------------------------------------------


def test_materialize_artifacts_unavailable_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_materialize_artifacts handles ensure returning (None, None)."""
    # Patch _ensure_artifact to return (None, None) => unavailable
    monkeypatch.setattr(
        top3_bootstrap,
        "_ensure_artifact",
        lambda artifact, dl_dir, *, download_policy: (None, None),
    )

    dataset = DatasetLock(
        source="test",
        version="v1",
        description="test",
        artifacts=[
            ArtifactSpec(
                artifact_id="demo",
                url="http://x",
                size_bytes=10,
                filename="demo.bin",
                checksum="md5:" + "a" * 32,
                required=True,
            )
        ],
    )
    report = SourceMaterializationReport(source="test")
    local, hard_failed = _materialize_artifacts(
        dataset=dataset,
        downloads_dir=tmp_path,
        download_policy="never",
        report=report,
    )
    assert hard_failed is True
    assert "artifact_unavailable:demo" in report.skip_reasons
    assert len(local) == 0


def test_materialize_artifacts_optional_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Optional artifacts that are unavailable don't cause hard failure."""
    monkeypatch.setattr(
        top3_bootstrap,
        "_ensure_artifact",
        lambda artifact, dl_dir, *, download_policy: (None, None),
    )

    dataset = DatasetLock(
        source="test",
        version="v1",
        description="test",
        artifacts=[
            ArtifactSpec(
                artifact_id="opt",
                url="http://x",
                size_bytes=10,
                filename="opt.bin",
                checksum="md5:" + "a" * 32,
                required=False,
            )
        ],
    )
    report = SourceMaterializationReport(source="test")
    _local, hard_failed = _materialize_artifacts(
        dataset=dataset,
        downloads_dir=tmp_path,
        download_policy="never",
        report=report,
    )
    assert hard_failed is False


# ---------------------------------------------------------------------------
# materialize_top3: invalid download policy (line 933)
# ---------------------------------------------------------------------------


def test_materialize_top3_invalid_policy_raises(tmp_path: Path) -> None:
    """materialize_top3 rejects invalid download_policy strings."""
    manifest_path = tmp_path / "top3.lock.json"
    write_manifest(
        Top3LockManifest(
            schema_version=1,
            generated_at_utc="2026-01-01T00:00:00+00:00",
            datasets=[],
        ),
        manifest_path,
    )
    with pytest.raises(ValueError, match="Invalid download policy"):
        materialize_top3(
            manifest_path=manifest_path,
            output_dir=tmp_path / "cases",
            data_root=tmp_path / "data",
            download_policy="invalid_policy",
            jobs=1,
            max_cases=0,
            validate=False,
            dry_run=True,
        )


# ---------------------------------------------------------------------------
# report_has_hard_failures: no failures case
# ---------------------------------------------------------------------------


def test_report_has_hard_failures_false() -> None:
    """report_has_hard_failures returns False when no sources failed."""
    report = MaterializationReport(
        schema_version=1,
        started_at_utc="t0",
        finished_at_utc="t1",
        manifest_path="m",
        data_root="d",
        output_dir="o",
        download_policy="never",
        max_cases=0,
        validate=False,
        dry_run=True,
        jobs=1,
        sources=[
            SourceMaterializationReport(source="a", hard_failed=False),
            SourceMaterializationReport(source="b", hard_failed=False),
        ],
    )
    assert report_has_hard_failures(report) is False


# ---------------------------------------------------------------------------
# Top3LockManifest.from_dict round-trip with optional fields
# ---------------------------------------------------------------------------


def test_manifest_from_dict_with_optional_nones() -> None:
    """from_dict handles None size_bytes, filename, note, checksum."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "datasets": [
            {
                "source": "test",
                "version": "v1",
                "description": "d",
                "artifacts": [
                    {
                        "artifact_id": "a",
                        "url": "http://x",
                        "size_bytes": None,
                        "filename": None,
                        "note": None,
                        "checksum": None,
                        "required": False,
                    }
                ],
            }
        ],
    }
    manifest = Top3LockManifest.from_dict(payload)
    art = manifest.datasets[0].artifacts[0]
    assert art.size_bytes is None
    assert art.filename is None
    assert art.note is None
    assert art.checksum is None
    assert art.required is False


# ---------------------------------------------------------------------------
# _slug helper via _artifact_from_zenodo_file
# ---------------------------------------------------------------------------


def test_slug_handles_special_characters() -> None:
    """_slug normalizes special characters in filenames."""
    spec = _artifact_from_zenodo_file(
        source_prefix="p",
        record_id="1",
        file_entry={
            "key": "my-file (2).zip",
            "size": 10,
            "links": {"self": "http://z.org/my-file%20(2).zip"},
        },
    )
    # _slug converts non-alnum to underscores and collapses them
    assert " " not in spec.artifact_id
    assert "-" not in spec.artifact_id
    assert "(" not in spec.artifact_id


# ---------------------------------------------------------------------------
# _safe_extract_tar: unsupported member type (line 635)
# ---------------------------------------------------------------------------


def test_safe_extract_tar_rejects_unsupported_member_type(
    tmp_path: Path,
) -> None:
    """Tar members that are not file/dir/sym/special are rejected."""
    from evals.importers.top3_bootstrap import _safe_extract_tar

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        # Create a member with an unusual type (GNU long name)
        info = tarfile.TarInfo(name="weird")
        info.type = b"9"  # arbitrary non-standard type
        tf.addfile(info)
    archive_path = tmp_path / "weird.tar.gz"
    archive_path.write_bytes(buf.getvalue())

    with pytest.raises(ArchiveExtractionError):
        _safe_extract_tar(archive_path, tmp_path / "out")


# ---------------------------------------------------------------------------
# _safe_extract_tar: extractfile returns None (line 640)
# ---------------------------------------------------------------------------


def test_safe_extract_tar_extractfile_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When extractfile returns None for a regular file, error is raised."""
    from evals.importers.top3_bootstrap import _safe_extract_tar

    # Create a normal tar first
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        content = b"hello"
        info = tarfile.TarInfo(name="file.txt")
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    archive_path = tmp_path / "null.tar.gz"
    archive_path.write_bytes(buf.getvalue())

    # Patch TarFile.extractfile to return None
    _orig_open = tarfile.open

    class _PatchedTarFile:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._tf = _orig_open(*args, **kwargs)

        def __enter__(self) -> _PatchedTarFile:
            self._tf.__enter__()
            return self

        def __exit__(self, *args: Any) -> None:
            self._tf.__exit__(*args)

        def getmembers(self) -> list[tarfile.TarInfo]:
            return self._tf.getmembers()

        def extractfile(self, member: tarfile.TarInfo) -> io.BufferedReader | None:
            return None

    monkeypatch.setattr(tarfile, "open", lambda *a, **kw: _PatchedTarFile(*a, **kw))

    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ArchiveExtractionError, match="archive_member_read_failed"):
        _safe_extract_tar(archive_path, out)


# ---------------------------------------------------------------------------
# _materialize_artifacts: success path (line 727)
# ---------------------------------------------------------------------------


def test_materialize_artifacts_success_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_materialize_artifacts records local paths on success."""
    fake_path = tmp_path / "demo.bin"
    fake_path.write_bytes(b"ok")

    monkeypatch.setattr(
        top3_bootstrap,
        "_ensure_artifact",
        lambda artifact, dl_dir, *, download_policy: (fake_path, None),
    )

    dataset = DatasetLock(
        source="test",
        version="v1",
        description="test",
        artifacts=[
            ArtifactSpec(
                artifact_id="demo",
                url="http://x",
                size_bytes=2,
                filename="demo.bin",
                checksum="md5:" + "a" * 32,
            )
        ],
    )
    report = SourceMaterializationReport(source="test")
    local, hard_failed = _materialize_artifacts(
        dataset=dataset,
        downloads_dir=tmp_path,
        download_policy="never",
        report=report,
    )
    assert hard_failed is False
    assert "demo" in local
    assert local["demo"] == fake_path


# ---------------------------------------------------------------------------
# Network helper stubs (lines 191-283)
# ---------------------------------------------------------------------------


class TestJsonRequest:
    """Cover _json_request (line 192)."""

    def test_returns_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import io
        import urllib.request

        fake_body = b'{"key": "value"}'

        class FakeResponse:
            def read(self) -> bytes:
                return fake_body

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *a: object) -> None:
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=60: FakeResponse())
        result = top3_bootstrap._json_request("https://example.com/api")
        assert result == {"key": "value"}

    def test_non_dict_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.request

        class FakeResponse:
            def read(self) -> bytes:
                return b"[1, 2, 3]"

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *a: object) -> None:
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=60: FakeResponse())
        with pytest.raises(ValueError, match="Unexpected JSON payload"):
            top3_bootstrap._json_request("https://example.com/api")


class TestFetchZenodoRecord:
    """Cover _fetch_zenodo_record (line 207)."""

    def test_delegates_to_json_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(top3_bootstrap, "_json_request", lambda url, **kw: {"id": "123"})
        result = top3_bootstrap._fetch_zenodo_record("123")
        assert result == {"id": "123"}


class TestFetchGitHubDefaultBranchAndHeadCommit:
    """Cover _fetch_github_default_branch_and_head_commit (lines 211-228)."""

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sha = "a" * 40

        def _fake_json(url: str, **kw: Any) -> dict[str, Any]:
            if "commits" in url:
                return {"sha": sha}
            return {"default_branch": "main"}

        monkeypatch.setattr(top3_bootstrap, "_json_request", _fake_json)
        branch, commit = top3_bootstrap._fetch_github_default_branch_and_head_commit(
            "owner", "repo"
        )
        assert branch == "main"
        assert commit == sha

    def test_missing_branch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            top3_bootstrap, "_json_request", lambda url, **kw: {"default_branch": ""}
        )
        with pytest.raises(ValueError, match="default branch"):
            top3_bootstrap._fetch_github_default_branch_and_head_commit("o", "r")

    def test_missing_sha_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake_json(url: str, **kw: Any) -> dict[str, Any]:
            if "commits" in url:
                return {"sha": ""}
            return {"default_branch": "main"}

        monkeypatch.setattr(top3_bootstrap, "_json_request", _fake_json)
        with pytest.raises(ValueError, match="commit sha"):
            top3_bootstrap._fetch_github_default_branch_and_head_commit("o", "r")

    def test_invalid_sha_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake_json(url: str, **kw: Any) -> dict[str, Any]:
            if "commits" in url:
                return {"sha": "not-a-valid-sha"}
            return {"default_branch": "main"}

        monkeypatch.setattr(top3_bootstrap, "_json_request", _fake_json)
        with pytest.raises(ValueError, match="Invalid GitHub commit sha"):
            top3_bootstrap._fetch_github_default_branch_and_head_commit("o", "r")


class TestProbeRemoteArtifactXxh3128:
    """Cover _probe_remote_artifact_xxh3_128 (lines 236-255)."""

    def test_computes_hash_and_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.request

        class FakeResponse:
            def __init__(self) -> None:
                self._chunks = [b"hello", b"world", b""]
                self._idx = 0

            def read(self, n: int) -> bytes:
                if self._idx >= len(self._chunks):
                    return b""
                chunk = self._chunks[self._idx]
                self._idx += 1
                return chunk

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *a: object) -> None:
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=180: FakeResponse())
        size, checksum = top3_bootstrap._probe_remote_artifact_xxh3_128("https://x")
        assert size == 10  # 5 + 5
        assert checksum.startswith(top3_bootstrap.CONTENT_HASH_ALGORITHM + ":")


class TestResolveGitHubRepoArchiveArtifact:
    """Cover _resolve_github_repo_archive_artifact (lines 265-283)."""

    def test_returns_artifact_and_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sha = "b" * 40
        monkeypatch.setattr(
            top3_bootstrap,
            "_fetch_github_default_branch_and_head_commit",
            lambda o, r: ("main", sha),
        )
        monkeypatch.setattr(
            top3_bootstrap,
            "_probe_remote_artifact_xxh3_128",
            lambda url, **kw: (12345, "xxh3_128:abcdef"),
        )
        artifact, meta = top3_bootstrap._resolve_github_repo_archive_artifact(
            artifact_id="repo_zip",
            owner="test",
            repo="testrepo",
            note="test note",
        )
        assert artifact.artifact_id == "repo_zip"
        assert artifact.size_bytes == 12345
        assert meta["github_commit_sha"] == sha
        assert meta["github_default_branch"] == "main"


# ---------------------------------------------------------------------------
# _download_to_path and _parse_checksum (lines 496-507)
# ---------------------------------------------------------------------------


class TestDownloadToPath:
    """Cover _download_to_path (lines 496-507)."""

    def test_atomic_download(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import urllib.request

        class FakeResponse:
            def read(self, n: int = -1) -> bytes:
                return b""

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *a: object) -> None:
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=180: FakeResponse())
        dest = tmp_path / "subdir" / "file.bin"
        top3_bootstrap._download_to_path("https://x", dest)
        assert dest.exists()

    def test_cleans_up_existing_tmp(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import urllib.request

        class FakeResponse:
            def read(self, n: int = -1) -> bytes:
                return b""

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *a: object) -> None:
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=180: FakeResponse())
        dest = tmp_path / "file.bin"
        tmp = dest.with_suffix(".bin.tmp")
        tmp.write_bytes(b"old")
        top3_bootstrap._download_to_path("https://x", dest)
        assert dest.exists()
        assert not tmp.exists()


# ---------------------------------------------------------------------------
# Materialization functions (lines 744-917)
# ---------------------------------------------------------------------------


def _make_manifest_with_dataset(source: str, artifact_ids: list[str]) -> Top3LockManifest:
    """Helper: create a manifest with one dataset."""
    artifacts = [
        ArtifactSpec(
            artifact_id=aid,
            url=f"https://example.com/{aid}",
            filename=f"{aid}.zip",
            required=True,
        )
        for aid in artifact_ids
    ]
    return Top3LockManifest(
        schema_version=1,
        generated_at_utc="2024-01-01T00:00:00+00:00",
        datasets=[
            DatasetLock(
                source=source,
                version="v1",
                description="test",
                artifacts=artifacts,
            )
        ],
    )


class TestMaterializeBugsinpy:
    """Cover _materialize_bugsinpy (lines 744-788)."""

    def test_hard_failed_returns_early(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Lines 757-759: hard_failed from _materialize_artifacts → early return."""
        manifest = _make_manifest_with_dataset("bugsinpy", ["repo_zip"])
        monkeypatch.setattr(
            top3_bootstrap,
            "_materialize_artifacts",
            lambda **kw: ({}, True),
        )
        report = top3_bootstrap._materialize_bugsinpy(
            output_dir=tmp_path / "out",
            data_root=tmp_path / "data",
            manifest=manifest,
            download_policy="never",
            max_cases=10,
            validate=False,
            dry_run=True,
            jobs=1,
        )
        assert report.skipped >= 1

    def test_missing_repo_zip(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Lines 761-766: repo_zip not in local_artifacts → hard_failed."""
        manifest = _make_manifest_with_dataset("bugsinpy", ["other"])
        monkeypatch.setattr(
            top3_bootstrap,
            "_materialize_artifacts",
            lambda **kw: ({"other": tmp_path / "other.zip"}, False),
        )
        report = top3_bootstrap._materialize_bugsinpy(
            output_dir=tmp_path / "out",
            data_root=tmp_path / "data",
            manifest=manifest,
            download_policy="never",
            max_cases=10,
            validate=False,
            dry_run=True,
            jobs=1,
        )
        assert report.hard_failed is True
        assert "artifact_missing:repo_zip" in report.skip_reasons

    def test_extraction_failure(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Lines 768-776: archive extraction fails → hard_failed."""
        manifest = _make_manifest_with_dataset("bugsinpy", ["repo_zip"])
        fake_zip = tmp_path / "fake.zip"
        fake_zip.write_bytes(b"not a zip")
        monkeypatch.setattr(
            top3_bootstrap,
            "_materialize_artifacts",
            lambda **kw: ({"repo_zip": fake_zip}, False),
        )
        monkeypatch.setattr(
            top3_bootstrap,
            "_extract_archive",
            _raise_extraction_error,
        )
        report = top3_bootstrap._materialize_bugsinpy(
            output_dir=tmp_path / "out",
            data_root=tmp_path / "data",
            manifest=manifest,
            download_policy="never",
            max_cases=10,
            validate=False,
            dry_run=True,
            jobs=1,
        )
        assert report.hard_failed is True

    def test_success_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Lines 778-788: successful import."""
        manifest = _make_manifest_with_dataset("bugsinpy", ["repo_zip"])
        fake_zip = tmp_path / "fake.zip"
        fake_zip.write_bytes(b"zip")
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        monkeypatch.setattr(
            top3_bootstrap,
            "_materialize_artifacts",
            lambda **kw: ({"repo_zip": fake_zip}, False),
        )
        monkeypatch.setattr(top3_bootstrap, "_extract_archive", lambda *a: repo_root)

        from evals.importers.types import ImportRunSummary

        fake_summary = ImportRunSummary(
            source="bugsinpy",
            attempted=5,
            imported=3,
            skipped=2,
            skip_reasons={"reason": 2},
            imported_case_ids=["case1"],
            notes=["note1"],
        )
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.import_bugsinpy",
            lambda *a, **kw: fake_summary,
        )
        report = top3_bootstrap._materialize_bugsinpy(
            output_dir=tmp_path / "out",
            data_root=tmp_path / "data",
            manifest=manifest,
            download_policy="never",
            max_cases=10,
            validate=False,
            dry_run=True,
            jobs=1,
        )
        assert report.imported == 3
        assert report.attempted == 5


class TestMaterializePybughive:
    """Cover _materialize_pybughive (lines 821-857)."""

    def test_hard_failed_returns_early(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Lines 815-819: hard_failed → early return."""
        manifest = _make_manifest_with_dataset("pybughive", ["repo_zip"])
        monkeypatch.setattr(
            top3_bootstrap,
            "_materialize_artifacts",
            lambda **kw: ({}, True),
        )
        report = top3_bootstrap._materialize_pybughive(
            output_dir=tmp_path / "out",
            data_root=tmp_path / "data",
            manifest=manifest,
            download_policy="never",
            max_cases=10,
            validate=False,
            dry_run=True,
            jobs=1,
        )
        assert report.hard_failed is True
        assert report.skipped >= 1

    def test_missing_repo_zip(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Lines 821-828: repo_zip missing → hard_failed."""
        manifest = _make_manifest_with_dataset("pybughive", ["other"])
        monkeypatch.setattr(
            top3_bootstrap,
            "_materialize_artifacts",
            lambda **kw: ({"other": tmp_path / "o.zip"}, False),
        )
        report = top3_bootstrap._materialize_pybughive(
            output_dir=tmp_path / "out",
            data_root=tmp_path / "data",
            manifest=manifest,
            download_policy="never",
            max_cases=10,
            validate=False,
            dry_run=True,
            jobs=1,
        )
        assert report.hard_failed is True

    def test_extraction_failure(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Lines 830-838: extraction error → hard_failed."""
        manifest = _make_manifest_with_dataset("pybughive", ["repo_zip"])
        fake_zip = tmp_path / "f.zip"
        fake_zip.write_bytes(b"bad")
        monkeypatch.setattr(
            top3_bootstrap,
            "_materialize_artifacts",
            lambda **kw: ({"repo_zip": fake_zip}, False),
        )
        monkeypatch.setattr(
            top3_bootstrap,
            "_extract_archive",
            _raise_extraction_error,
        )
        report = top3_bootstrap._materialize_pybughive(
            output_dir=tmp_path / "out",
            data_root=tmp_path / "data",
            manifest=manifest,
            download_policy="never",
            max_cases=10,
            validate=False,
            dry_run=True,
            jobs=1,
        )
        assert report.hard_failed is True

    def test_success_with_offline_archives(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Lines 840-857: success with offline archives."""
        manifest = _make_manifest_with_dataset("pybughive", ["repo_zip", "part1", "part2"])
        fake_zip = tmp_path / "repo.zip"
        fake_zip.write_bytes(b"zip")
        part1 = tmp_path / "part1.zip"
        part1.write_bytes(b"p1")
        part2 = tmp_path / "part2.zip"
        part2.write_bytes(b"p2")
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        monkeypatch.setattr(
            top3_bootstrap,
            "_materialize_artifacts",
            lambda **kw: (
                {"repo_zip": fake_zip, "part1": part1, "part2": part2},
                False,
            ),
        )
        monkeypatch.setattr(top3_bootstrap, "_extract_archive", lambda *a: repo_root)

        from evals.importers.types import ImportRunSummary

        fake_summary = ImportRunSummary(
            source="pybughive",
            attempted=4,
            imported=2,
            skipped=2,
            skip_reasons={},
            imported_case_ids=["c1", "c2"],
            notes=[],
        )
        monkeypatch.setattr(
            "evals.importers.pybughive_importer.import_pybughive",
            lambda *a, **kw: fake_summary,
        )
        report = top3_bootstrap._materialize_pybughive(
            output_dir=tmp_path / "out",
            data_root=tmp_path / "data",
            manifest=manifest,
            download_policy="never",
            max_cases=10,
            validate=False,
            dry_run=True,
            jobs=1,
        )
        assert report.imported == 2


class TestMaterializeTrickybugs:
    """Cover _materialize_trickybugs (lines 871-917)."""

    def test_hard_failed_returns_early(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Lines 884-886: hard_failed → early return."""
        manifest = _make_manifest_with_dataset("trickybugs", ["archive_zip"])
        monkeypatch.setattr(
            top3_bootstrap,
            "_materialize_artifacts",
            lambda **kw: ({}, True),
        )
        report = top3_bootstrap._materialize_trickybugs(
            output_dir=tmp_path / "out",
            data_root=tmp_path / "data",
            manifest=manifest,
            download_policy="never",
            max_cases=10,
            validate=False,
            dry_run=True,
            jobs=1,
        )
        assert report.skipped >= 1

    def test_missing_archive_zip(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Lines 888-895: archive_zip missing → hard_failed."""
        manifest = _make_manifest_with_dataset("trickybugs", ["other"])
        monkeypatch.setattr(
            top3_bootstrap,
            "_materialize_artifacts",
            lambda **kw: ({"other": tmp_path / "o.zip"}, False),
        )
        report = top3_bootstrap._materialize_trickybugs(
            output_dir=tmp_path / "out",
            data_root=tmp_path / "data",
            manifest=manifest,
            download_policy="never",
            max_cases=10,
            validate=False,
            dry_run=True,
            jobs=1,
        )
        assert report.hard_failed is True

    def test_extraction_failure(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Lines 897-905: extraction error → hard_failed."""
        manifest = _make_manifest_with_dataset("trickybugs", ["archive_zip"])
        fake = tmp_path / "f.zip"
        fake.write_bytes(b"bad")
        monkeypatch.setattr(
            top3_bootstrap,
            "_materialize_artifacts",
            lambda **kw: ({"archive_zip": fake}, False),
        )
        monkeypatch.setattr(
            top3_bootstrap,
            "_extract_archive",
            _raise_extraction_error,
        )
        report = top3_bootstrap._materialize_trickybugs(
            output_dir=tmp_path / "out",
            data_root=tmp_path / "data",
            manifest=manifest,
            download_policy="never",
            max_cases=10,
            validate=False,
            dry_run=True,
            jobs=1,
        )
        assert report.hard_failed is True

    def test_success_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Lines 907-917: successful import."""
        manifest = _make_manifest_with_dataset("trickybugs", ["archive_zip"])
        fake = tmp_path / "f.zip"
        fake.write_bytes(b"zip")
        root = tmp_path / "root"
        root.mkdir()
        monkeypatch.setattr(
            top3_bootstrap,
            "_materialize_artifacts",
            lambda **kw: ({"archive_zip": fake}, False),
        )
        monkeypatch.setattr(top3_bootstrap, "_extract_archive", lambda *a: root)

        from evals.importers.types import ImportRunSummary

        fake_summary = ImportRunSummary(
            source="trickybugs",
            attempted=3,
            imported=2,
            skipped=1,
            skip_reasons={},
            imported_case_ids=["t1", "t2"],
            notes=[],
        )
        monkeypatch.setattr(
            "evals.importers.trickybugs_importer.import_trickybugs",
            lambda *a, **kw: fake_summary,
        )
        report = top3_bootstrap._materialize_trickybugs(
            output_dir=tmp_path / "out",
            data_root=tmp_path / "data",
            manifest=manifest,
            download_policy="never",
            max_cases=10,
            validate=False,
            dry_run=True,
            jobs=1,
        )
        assert report.imported == 2


def _raise_extraction_error(*args: Any, **kwargs: Any) -> None:
    raise ArchiveExtractionError("test_failure")
