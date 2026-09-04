#!/usr/bin/env python3
"""Download and verify the soundfonts the instrument registry renders from.

Assets are gitignored (re-downloadable from documented sources, same
policy as the ffmpeg build). Sources and licenses — everything here is
commercial-use-safe:

Automated (pinned sha256, fetched from the Debian pool because Musical
Artifacts blocks scripted downloads):
- FluidR3_GM.sf2 — Frank Wen, MIT license (commercial use OK). Covers
  every instrument in INSTRUMENT_PROGRAMS, including the non-western
  voices GM provides (sitar, koto, shanai, taiko, kalimba).

Manual (browser download from Musical Artifacts, verified once placed —
see MANUAL_FONTS below):
- MFA Boston 1 (CC-BY 3.0, museum-sampled) — bansuri, sarangi, rudra
  veena, sarasvati veena, plus koto/shamisen/ud/qanoon/kora upgrades.
- 105-Sitar (public domain) — a better sitar than GM 104.
- Wetthasinghe's Harmonium (CC-BY 4.0).

Salamander.sf2 (the Phase 1 piano, CC-BY 3.0) is NOT downloaded here —
it was fetched manually into ./assets/ (see THIRD_PARTY_NOTICES.md).

Usage:
    python scripts/download_soundfonts.py             # fetch anything missing
    python scripts/download_soundfonts.py --verify    # verify hashes only
    python scripts/download_soundfonts.py --status    # what is installed
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
from typing import TypedDict

SOUNDFONT_DIR = Path("./assets/soundfonts")


class _FluidR3Spec(TypedDict):
    """Archive spec for FluidR3_GM. Upgrade the pins together."""

    url: str
    member: str
    dest: Path
    download_sha256: str
    file_sha256: str
    license_file: tuple[str, Path]


FLUIDR3: _FluidR3Spec = {
    "url": "https://deb.debian.org/debian/pool/main/f/fluid-soundfont/fluid-soundfont_3.1.orig.tar.gz",
    "member": "fluid-soundfont-3.1/FluidR3_GM.sf2",
    "dest": SOUNDFONT_DIR / "FluidR3_GM.sf2",
    "download_sha256": "2621acaa1c78e4abdb24bdd163230cc577e61276936d6aa6e3180582142f0343",
    "file_sha256": "74594e8f4250680adf590507a306655a299935343583256f3b722c48a1bc1cb0",
    "license_file": ("fluid-soundfont-3.1/COPYING", SOUNDFONT_DIR / "FluidR3_COPYING.txt"),
}

# Fonts that must be downloaded by hand (Musical Artifacts serves them
# only to browsers). Place the file at the given path under
# assets/soundfonts/, then run --verify: the script records the sha256
# of what you placed and reports it for THIRD_PARTY_NOTICES.md. No
# license file ships with these — the license URL is the attribution.
MANUAL_FONTS: dict[str, dict[str, str]] = {
    "mfa_boston_1.sf2": {
        "page": "https://musical-artifacts.com/artifacts/3593",
        "license": "CC-BY 3.0",
        "covers": "bansuri, sarangi, rudra veena, sarasvati veena, koto, shamisen, ud, qanoon, kora",
    },
    "105-sitar.sf2": {
        "page": "https://musical-artifacts.com/artifacts/3847",
        "license": "Public domain",
        "covers": "sitar (upgrades GM 104)",
    },
    "wetthasinghe_harmonium.sf2": {
        "page": "https://musical-artifacts.com/artifacts/1391",
        "license": "CC-BY 4.0",
        "covers": "harmonium",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_fluidr3() -> None:
    dest = FLUIDR3["dest"]
    member, license_dest = FLUIDR3["member"], FLUIDR3["license_file"][1]
    license_member = FLUIDR3["license_file"][0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "fluid-soundfont.tar.gz"
        print(f"downloading {FLUIDR3['url']}")
        urllib.request.urlretrieve(FLUIDR3["url"], tarball)
        actual = sha256_file(tarball)
        if actual != FLUIDR3["download_sha256"]:
            raise SystemExit(
                f"download sha256 mismatch:\n  expected {FLUIDR3['download_sha256']}\n"
                f"  actual   {actual}\nrefusing to extract a tampered archive"
            )
        with tarfile.open(tarball) as tf:
            tf.extract(member, tmp)
            shutil.move(str(Path(tmp) / member), dest)
            tf.extract(license_member, tmp)
            shutil.move(str(Path(tmp) / license_member), license_dest)
    extracted = sha256_file(dest)
    if extracted != FLUIDR3["file_sha256"]:
        dest.unlink(missing_ok=True)
        raise SystemExit(
            f"extracted file sha256 mismatch (deleted):\n"
            f"  expected {FLUIDR3['file_sha256']}\n  actual   {extracted}"
        )
    print(f"ok: {dest} ({dest.stat().st_size / 1e6:.0f} MB, sha256 verified)")


def status() -> None:
    dest = FLUIDR3["dest"]
    if dest.exists():
        actual = sha256_file(dest)
        mark = "ok" if actual == FLUIDR3["file_sha256"] else f"sha256 MISMATCH ({actual})"
        print(f"{dest}: {mark}")
    else:
        print(f"{dest}: missing (run this script without --verify to download)")

    if not MANUAL_FONTS:
        return
    print("\nManual fonts (browser-download, then place under assets/soundfonts/):")
    for name, info in MANUAL_FONTS.items():
        path = SOUNDFONT_DIR / name
        if path.exists():
            print(
                f"{path}: present ({path.stat().st_size / 1e6:.0f} MB, "
                f"sha256 {sha256_file(path)}; {info['license']} — {info['page']})"
            )
        else:
            print(
                f"{path}: missing — download from {info['page']} ({info['license']}; "
                f"covers: {info['covers']})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="only verify existing files; do not download",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="report what is installed and what is missing, then exit",
    )
    args = parser.parse_args()

    if args.status:
        status()
        return 0

    dest = FLUIDR3["dest"]
    if dest.exists():
        actual = sha256_file(dest)
        if actual == FLUIDR3["file_sha256"]:
            print(f"ok: {dest} already present and verified")
        else:
            print(
                f"{dest} exists but sha256 mismatch:\n"
                f"  expected {FLUIDR3['file_sha256']}\n  actual   {actual}\n"
                "re-downloading..."
            )
            dest.unlink()
    elif args.verify:
        print(f"missing or invalid: {dest}; run without --verify to download")
        return 1
    else:
        download_fluidr3()

    missing_manual = [n for n in MANUAL_FONTS if not (SOUNDFONT_DIR / n).exists()]
    if missing_manual:
        print(
            "note: manual fonts still missing: "
            + ", ".join(missing_manual)
            + " — see the MANUAL_FONTS section in this script"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
