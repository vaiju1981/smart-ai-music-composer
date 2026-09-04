#!/usr/bin/env python3
"""Download the soundfonts the instrument registry renders from.

Assets are gitignored (re-downloadable from documented sources, same
policy as the ffmpeg build). Sources and licenses:

- FluidR3_GM.sf2  — Frank Wen, MIT license (commercial use OK).
  Covers every instrument in INSTRUMENT_PROGRAMS, including the
  non-western ones GM provides (sitar, koto, shanai, taiko, kalimba).
  Downloaded from the Debian pool package `fluid-soundfont` version
  3.1 (the upstream orig tarball; Musical Artifacts blocks scripted
  downloads).

Salamander.sf2 (the Phase 1 piano, CC-BY 3.0) is NOT downloaded here —
it was fetched manually into ./assets/ (see THIRD_PARTY_NOTICES.md).

Usage:
    python scripts/download_soundfonts.py            # fetch anything missing
    python scripts/download_soundfonts.py --verify   # verify hashes only
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

SOUNDFONT_DIR = Path("./assets/soundfonts")

# URL -> (member-path inside the tarball, destination, pinned sha256 of
# the DOWNLOAD, pinned sha256 of the EXTRACTED FILE). Upgrade the pins
# together when bumping the version.
FLUIDR3 = {
    "url": "https://deb.debian.org/debian/pool/main/f/fluid-soundfont/fluid-soundfont_3.1.orig.tar.gz",
    "member": "fluid-soundfont-3.1/FluidR3_GM.sf2",
    "dest": SOUNDFONT_DIR / "FluidR3_GM.sf2",
    "download_sha256": "2621acaa1c78e4abdb24bdd163230cc577e61276936d6aa6e3180582142f0343",
    "file_sha256": "74594e8f4250680adf590507a306655a299935343583256f3b722c48a1bc1cb0",
    "license_file": ("fluid-soundfont-3.1/COPYING", SOUNDFONT_DIR / "FluidR3_COPYING.txt"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_fluidr3() -> None:
    dest = FLUIDR3["dest"]
    assert isinstance(dest, Path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "fluid-soundfont.tar.gz"
        print(f"downloading {FLUIDR3['url']}")
        urllib.request.urlretrieve(FLUIDR3["url"], tarball)
        actual = sha256_file(tarball)
        if actual != FLUIDR3["download_sha256"]:
            raise SystemExit(
                f"download sha256 mismatch:\n  expected {FLUIDR3['download_sha256']}\n"
                f"  actual   {actual}\n"
                "refusing to extract a tampered archive"
            )
        with tarfile.open(tarball) as tf:
            member = FLUIDR3["member"]
            license_member, license_dest = FLUIDR3["license_file"]
            tf.extract(member, tmp)
            shutil.move(str(Path(tmp) / member), dest)
            tf.extract(license_member, tmp)
            shutil.move(str(Path(tmp) / license_member), license_dest)
    extracted = sha256_file(dest)
    if extracted != FLUIDR3["file_sha256"]:
        dest.unlink(missing_ok=True)
        raise SystemExit(
            f"extracted file sha256 mismatch (deleted):\n  expected {FLUIDR3['file_sha256']}\n"
            f"  actual   {extracted}"
        )
    print(f"ok: {dest} ({dest.stat().st_size / 1e6:.0f} MB, sha256 verified)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="only verify existing files; do not download",
    )
    args = parser.parse_args()

    dest = FLUIDR3["dest"]
    assert isinstance(dest, Path)
    if dest.exists():
        actual = sha256_file(dest)
        if actual == FLUIDR3["file_sha256"]:
            print(f"ok: {dest} already present and verified")
            return 0
        print(
            f"{dest} exists but sha256 mismatch:\n  expected {FLUIDR3['file_sha256']}\n"
            f"  actual   {actual}\nre-downloading..."
        )
        dest.unlink()
    if args.verify:
        print(f"missing or invalid: {dest}; run without --verify to download")
        return 1
    download_fluidr3()
    return 0


if __name__ == "__main__":
    sys.exit(main())
