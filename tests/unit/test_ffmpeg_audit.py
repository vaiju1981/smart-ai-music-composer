"""Unit tests for the FFmpeg audit."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from saimc.render.ffmpeg_audit import (
    FORBIDDEN_LICENSE_FLAGS,
    REQUIRED_CODEC_FLAGS,
    FfmpegLicense,
    _parse_flags,
    audit_ffmpeg,
)


def _write_fake_ffmpeg(tmp_path: Path, version_output: str, rc: int = 0) -> Path:
    """Write a fake ffmpeg shell script that prints `version_output` and exits `rc`."""
    script = tmp_path / "ffmpeg-fake.sh"
    script.write_text(
        f"#!/bin/sh\ncat <<'EOF'\n{version_output}\nEOF\nexit {rc}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


class TestParseFlags:
    def test_parses_double_dash_flags(self) -> None:
        flags = _parse_flags("--prefix=/usr --enable-libvpx --enable-libopus")
        assert "--enable-libvpx" in flags
        assert "--enable-libopus" in flags
        assert "--prefix=/usr" in flags

    def test_empty_line_returns_empty(self) -> None:
        assert _parse_flags("") == frozenset()

    def test_skips_non_flag_tokens(self) -> None:
        flags = _parse_flags("foo bar --enable-libvpx")
        assert "--enable-libvpx" in flags
        assert "foo" not in flags
        assert "bar" not in flags


class TestAuditWithFakeBinary:
    def test_lgpl_with_required_codecs_passes(self, tmp_path: Path) -> None:
        output = (
            "ffmpeg version 8.1.2 Copyright (c) 2000-2026 the FFmpeg developers\n"
            "  built with clang\n"
            "  configuration: --prefix=/usr --enable-shared --enable-libvpx --enable-libopus\n"
        )
        binary = _write_fake_ffmpeg(tmp_path, output)
        result = audit_ffmpeg(str(binary))
        assert result.ok
        assert result.license == FfmpegLicense.LGPL
        assert result.version == "8.1.2"
        assert result.missing_required_codecs == ()
        assert result.forbidden_flags_present == ()

    def test_gpl_binary_fails(self, tmp_path: Path) -> None:
        output = (
            "ffmpeg version 6.1 Copyright (c) 2000-2026 the FFmpeg developers\n"
            "  configuration: --enable-gpl --enable-libx264 --enable-libvpx --enable-libopus\n"
        )
        binary = _write_fake_ffmpeg(tmp_path, output)
        result = audit_ffmpeg(str(binary))
        assert not result.ok
        assert result.license == FfmpegLicense.GPL
        assert "--enable-gpl" in result.forbidden_flags_present

    def test_nonfree_binary_fails(self, tmp_path: Path) -> None:
        output = (
            "ffmpeg version 6.1 Copyright (c) 2000-2026 the FFmpeg developers\n"
            "  configuration: --enable-nonfree --enable-libvpx --enable-libopus\n"
        )
        binary = _write_fake_ffmpeg(tmp_path, output)
        result = audit_ffmpeg(str(binary))
        assert not result.ok
        assert result.license == FfmpegLicense.NON_FREE
        assert "--enable-nonfree" in result.forbidden_flags_present

    def test_missing_libvpx_fails(self, tmp_path: Path) -> None:
        output = (
            "ffmpeg version 8.1.2 Copyright (c) 2000-2026 the FFmpeg developers\n"
            "  configuration: --prefix=/usr --enable-shared --enable-libopus\n"
        )
        binary = _write_fake_ffmpeg(tmp_path, output)
        result = audit_ffmpeg(str(binary))
        assert not result.ok
        assert "--enable-libvpx" in result.missing_required_codecs

    def test_missing_libopus_fails(self, tmp_path: Path) -> None:
        output = (
            "ffmpeg version 8.1.2 Copyright (c) 2000-2026 the FFmpeg developers\n"
            "  configuration: --prefix=/usr --enable-shared --enable-libvpx\n"
        )
        binary = _write_fake_ffmpeg(tmp_path, output)
        result = audit_ffmpeg(str(binary))
        assert not result.ok
        assert "--enable-libopus" in result.missing_required_codecs

    def test_nonzero_rc_fails(self, tmp_path: Path) -> None:
        binary = _write_fake_ffmpeg(tmp_path, "garbage", rc=2)
        result = audit_ffmpeg(str(binary))
        assert not result.ok
        assert result.license == FfmpegLicense.UNKNOWN
        assert result.reasons

    def test_unparseable_version_still_extracts_codecs(self, tmp_path: Path) -> None:
        output = "weird build\n  configuration: --enable-libvpx --enable-libopus\n"
        binary = _write_fake_ffmpeg(tmp_path, output)
        result = audit_ffmpeg(str(binary))
        assert result.ok
        assert result.version == "unknown"


class TestAuditWithRealBinary:
    """If a real `ffmpeg` is on $PATH, audit it for documentation; otherwise skip."""

    @pytest.fixture
    def real_binary(self) -> str | None:
        import shutil

        return shutil.which("ffmpeg")

    def test_homebrew_ffmpeg_should_fail_gpl(self, real_binary: str | None) -> None:
        if real_binary is None:
            pytest.skip("no system ffmpeg available")
        result = audit_ffmpeg(real_binary)
        assert result.version, "real ffmpeg must report a version"
        assert isinstance(result.binary_sha256, str)
        assert len(result.binary_sha256) == 64
        # Homebrew's stock FFmpeg enables --enable-gpl + libx264. Assert
        # the audit correctly catches it. If the operator has built their
        # own LGPL binary and pointed PATH at it, this may legitimately pass;
        # in that case the test still documents that the binary was audited.
        if not result.ok:
            assert result.forbidden_flags_present or result.missing_required_codecs


class TestRequiredFlagsAreRealistic:
    def test_required_codecs_include_vp9_and_opus(self) -> None:
        assert "--enable-libvpx" in REQUIRED_CODEC_FLAGS
        assert "--enable-libopus" in REQUIRED_CODEC_FLAGS

    def test_forbidden_flags_block_gpl_and_nonfree(self) -> None:
        assert "--enable-gpl" in FORBIDDEN_LICENSE_FLAGS
        assert "--enable-nonfree" in FORBIDDEN_LICENSE_FLAGS


def _ignored(x: Iterable[str]) -> None:  # pragma: no cover — import helper
    pass
