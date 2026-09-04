#!/usr/bin/env bash
# Build the Phase 1 LGPL FFmpeg shared library.
#
# Per docs/roadmap.md §4 and §10 #7:
#   - Pin to FFmpeg 8.1.2 source archive.
#   - Verify SHA-256 of the archive before extracting.
#   - Configure with --disable-gpl --disable-nonfree --enable-shared
#     --disable-static --enable-libvpx --enable-libopus.
#   - Build, capture the resulting binary hash, and run scripts/audit_ffmpeg.py
#     against the build to record release artifacts.
#
# This script does NOT install FFmpeg globally. It builds into
# vendor/ffmpeg-build/ (gitignored) and writes release artifacts to
# dist/ffmpeg/8.1.2/:
#
#   dist/ffmpeg/8.1.2/
#     ffmpeg                       # the built binary
#     ffmpeg.sha256                # sha256 of the binary
#     configure.log                # complete ./configure output
#     build.log                    # complete make output
#     source.sha256                # sha256 of the upstream source archive
#     audit.json                   # the §10 #7 audit result
#
# Requirements: a POSIX shell, curl, sha256sum or shasum, nasm (or yasm),
# a C toolchain (Xcode CLT on macOS, gcc/clang on Linux), and the libvpx
# and libopus development headers available via the system package manager.
#
# The Phase 1 reference target is macOS Apple Silicon (darwin-arm64).
# Linux x86_64 and Linux aarch64 are release-gate additions that follow
# the same script with adjusted paths.

set -euo pipefail

readonly VERSION="8.1.2"
readonly SOURCE_SHA256="464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c"
readonly SOURCE_URL="https://ffmpeg.org/releases/ffmpeg-${VERSION}.tar.xz"
readonly BUILD_ROOT="${SAIMC_BUILD_ROOT:-vendor/ffmpeg-build}"
readonly SRC_DIR="${BUILD_ROOT}/ffmpeg-${VERSION}"
readonly DIST_DIR="${SAIMC_DIST_DIR:-dist/ffmpeg/${VERSION}}"
readonly JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

log() { printf '==> %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if ! command -v curl >/dev/null 2>&1; then
    fail "curl is required"
fi
if ! command -v make >/dev/null 2>&1; then
    fail "make is required"
fi
if ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
    fail "a C compiler (gcc or clang) is required"
fi

sha256_cmd() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256
    else
        fail "neither sha256sum nor shasum is available"
    fi
}

log "Phase 1 FFmpeg ${VERSION} build"
log "  source: ${SOURCE_URL}"
log "  expected sha256: ${SOURCE_SHA256}"
log "  jobs: ${JOBS}"
log "  build root: ${BUILD_ROOT}"
log "  dist: ${DIST_DIR}"

mkdir -p "${BUILD_ROOT}"
mkdir -p "${DIST_DIR}"

ARCHIVE="${BUILD_ROOT}/ffmpeg-${VERSION}.tar.xz"

if [[ -f "${ARCHIVE}" ]]; then
    log "archive already present; verifying sha256"
else
    log "downloading source archive"
    curl --fail --silent --show-error --location --output "${ARCHIVE}" "${SOURCE_URL}"
fi

log "verifying archive sha256"
ACTUAL_SHA256="$(sha256_cmd < "${ARCHIVE}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${SOURCE_SHA256}" ]]; then
    fail "sha256 mismatch: expected ${SOURCE_SHA256}, got ${ACTUAL_SHA256}"
fi
log "sha256 ok"
cp "${ARCHIVE}" "${DIST_DIR}/source.sha256"

if [[ ! -d "${SRC_DIR}" ]]; then
    log "extracting"
    tar -xJf "${ARCHIVE}" -C "${BUILD_ROOT}"
fi

cd "${SRC_DIR}"

log "configuring"
CONFIGURE_FLAGS=(
    --disable-gpl
    --disable-nonfree
    --enable-shared
    --disable-static
    --enable-libvpx
    --enable-libopus
    --enable-pic
    --prefix="${BUILD_ROOT}/install"
)

# shellcheck disable=SC2086
./configure "${CONFIGURE_FLAGS[@]}" 2>&1 | tee "${DIST_DIR}/configure.log"

log "building (${JOBS} parallel jobs)"
make -j"${JOBS}" 2>&1 | tee "${DIST_DIR}/build.log"

log "installing into build root"
make install 2>&1 | tee -a "${DIST_DIR}/build.log"

FFMPEG_BIN="${BUILD_ROOT}/install/bin/ffmpeg"
if [[ ! -x "${FFMPEG_BIN}" ]]; then
    fail "ffmpeg binary missing at ${FFMPEG_BIN}"
fi

log "capturing binary and sha256"
cp "${FFMPEG_BIN}" "${DIST_DIR}/ffmpeg"
chmod +x "${DIST_DIR}/ffmpeg"
BIN_SHA256="$(sha256_cmd < "${DIST_DIR}/ffmpeg" | awk '{print $1}')"
echo "${BIN_SHA256}  ffmpeg" > "${DIST_DIR}/ffmpeg.sha256"
log "binary sha256: ${BIN_SHA256}"

log "running release-gate audit"
SAIMC_PYTHON="${SAIMC_PYTHON:-python3}"
"${SAIMC_PYTHON}" scripts/audit_ffmpeg.py "${DIST_DIR}/ffmpeg" | tee "${DIST_DIR}/audit.json"

log "done"
log "release artifacts at: ${DIST_DIR}"
log "binary: ${DIST_DIR}/ffmpeg"
log "audit:  ${DIST_DIR}/audit.json"