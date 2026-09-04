"""Sanity checks for scripts/build_ffmpeg.sh.

The build script is a release-gate artifact. These tests verify the
script structurally pins the source SHA, includes the §10 #7
configure flags, and calls the audit at the end. They do NOT execute
the build (a real build takes ~20 minutes and requires libvpx/libopus
headers on the target machine).
"""

from __future__ import annotations

from pathlib import Path

BUILD_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_ffmpeg.sh"


def _script_text() -> str:
    return BUILD_SCRIPT.read_text(encoding="utf-8")


class TestBuildScriptPins:
    def test_pins_ffmpeg_version(self) -> None:
        text = _script_text()
        assert "8.1.2" in text
        assert "VERSION=" in text

    def test_pins_source_sha256(self) -> None:
        text = _script_text()
        assert "464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c" in text

    def test_pins_source_url(self) -> None:
        text = _script_text()
        assert "ffmpeg.org" in text or "ffmpeg-8.1.2.tar.xz" in text

    def test_verifies_sha_before_extract(self) -> None:
        text = _script_text()
        sha_check_idx = text.find("sha256 mismatch")
        extract_idx = text.find("tar -xJf")
        assert sha_check_idx > 0
        assert extract_idx > 0
        assert sha_check_idx < extract_idx, "sha256 must be verified before extract"


class TestBuildScriptConfigureFlags:
    def test_disables_gpl(self) -> None:
        assert "--disable-gpl" in _script_text()

    def test_disables_nonfree(self) -> None:
        assert "--disable-nonfree" in _script_text()

    def test_enables_shared(self) -> None:
        assert "--enable-shared" in _script_text()

    def test_disables_static(self) -> None:
        assert "--disable-static" in _script_text()

    def test_enables_libvpx(self) -> None:
        assert "--enable-libvpx" in _script_text()

    def test_enables_libopus(self) -> None:
        assert "--enable-libopus" in _script_text()


class TestBuildScriptRunsAudit:
    def test_calls_audit_ffmpeg_at_end(self) -> None:
        text = _script_text()
        # Find the actual invocation, not any earlier comment text.
        copy_idx = text.find('cp "${FFMPEG_BIN}" "${DIST_DIR}/ffmpeg"')
        audit_idx = text.find('"${SAIMC_PYTHON}" scripts/audit_ffmpeg.py')
        assert copy_idx > 0, "build script must copy the binary before auditing"
        assert audit_idx > 0, "build script must run the audit"
        assert audit_idx > copy_idx, "audit must come after the binary is built"

    def test_writes_audit_json_artifact(self) -> None:
        text = _script_text()
        assert "audit.json" in text


class TestBuildScriptReleaseArtifacts:
    def test_writes_configure_log(self) -> None:
        assert "configure.log" in _script_text()

    def test_writes_build_log(self) -> None:
        assert "build.log" in _script_text()

    def test_writes_binary_sha256(self) -> None:
        text = _script_text()
        assert "ffmpeg.sha256" in text

    def test_does_not_install_globally(self) -> None:
        text = _script_text()
        assert "make install" in text
        assert "BUILD_ROOT}/install" in text
        assert "sudo" not in text


class TestBuildScriptErrorHandling:
    def test_uses_strict_mode(self) -> None:
        text = _script_text()
        assert "set -euo pipefail" in text

    def test_has_fail_function(self) -> None:
        text = _script_text()
        assert "fail()" in text
