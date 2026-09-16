"""Third-party attribution per `docs/roadmap.md` §10 #12.

Salamander Grand Piano (the Phase 1 piano font) is CC BY 3.0 —
attribution is a distribution condition, not a courtesy — and the
dedicated fonts in `FONT_PRESETS` carry their own licenses (FluidR3 is
MIT; MFA Boston 1 is CC BY 3.0; Wetthasinghe's Harmonium is CC BY 4.0;
105-Sitar is public domain). This module is the single source of truth
for that attribution: the manifest's `assets` entry, the OGG/Opus
metadata tags, and `THIRD_PARTY_NOTICES.md` all draw from here so they
cannot drift.

The Phase 1 sample pack ships **unmodified**; if that ever changes
(conversion, retuning, trimming), the modification must be stated
explicitly here and in the notice file per CC BY 3.0 §4(b).
"""

from __future__ import annotations

from dataclasses import dataclass

SOUNDFONT_NAME = "Salamander Grand Piano"
SOUNDFONT_AUTHOR = "Alexander Holm"
SOUNDFONT_VERSION = "V3+20200602"
SOUNDFONT_LICENSE = "CC BY 3.0"
SOUNDFONT_LICENSE_URL = "https://creativecommons.org/licenses/by/3.0/"
SOUNDFONT_SOURCE_URL = "https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html"
SOUNDFONT_MODIFICATION_STATUS = "unmodified"

# Repo-relative path of the canonical notice, referenced by
# THIRD_PARTY_NOTICES.md and by the manifest's assets block.
SOUNDFONT_NOTICE_PATH = "LICENSES/Salamander-Grand-Piano.txt"


@dataclass(frozen=True)
class SoundfontAttribution:
    """Attribution facts for one soundfont, keyed by its file name."""

    name: str
    author: str
    version: str
    license: str
    license_url: str
    source_url: str
    notice_path: str
    modification_status: str


_SALAMANDER = SoundfontAttribution(
    name=SOUNDFONT_NAME,
    author=SOUNDFONT_AUTHOR,
    version=SOUNDFONT_VERSION,
    license=SOUNDFONT_LICENSE,
    license_url=SOUNDFONT_LICENSE_URL,
    source_url=SOUNDFONT_SOURCE_URL,
    notice_path=SOUNDFONT_NOTICE_PATH,
    modification_status=SOUNDFONT_MODIFICATION_STATUS,
)

# Keyed by the SF2 file name as resolved by `soundfont_for_instrument`.
# Everything here is commercial-use-safe and documented in
# THIRD_PARTY_NOTICES.md; the notice_path anchor is that file's heading.
SOUNDFONT_ATTRIBUTION: dict[str, SoundfontAttribution] = {
    "Salamander.sf2": _SALAMANDER,
    "FluidR3_GM.sf2": SoundfontAttribution(
        name="FluidR3 General MIDI",
        author="Frank Wen",
        version="3.1",
        license="MIT",
        license_url="",
        source_url=(
            "https://deb.debian.org/debian/pool/main/f/fluid-soundfont/"
            "fluid-soundfont_3.1.orig.tar.gz"
        ),
        notice_path="assets/soundfonts/FluidR3_COPYING.txt",
        modification_status="unmodified",
    ),
    "105-Sitar.sf2": SoundfontAttribution(
        name="105-Sitar",
        author="public domain (Musical Artifacts artifact #3847)",
        version="",
        license="Public domain",
        license_url="",
        source_url="https://musical-artifacts.com/artifacts/3847",
        notice_path="THIRD_PARTY_NOTICES.md#105-sitar-soundfont",
        modification_status="unmodified",
    ),
    "Wetthasinghe_Harmonium.sf2": SoundfontAttribution(
        name="Wetthasinghe's Harmonium",
        author="Wetthasinghe",
        version="",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        source_url="https://musical-artifacts.com/artifacts/1391",
        notice_path="THIRD_PARTY_NOTICES.md#wetthasinghes-harmonium-soundfont",
        modification_status="unmodified",
    ),
    "MFA_Boston_1.sf2": SoundfontAttribution(
        name="MFA Boston 1",
        author="MFA Boston 1 (Musical Artifacts artifact #3593)",
        version="",
        license="CC BY 3.0",
        license_url="https://creativecommons.org/licenses/by/3.0/",
        source_url="https://musical-artifacts.com/artifacts/3593",
        notice_path="THIRD_PARTY_NOTICES.md#mfa-boston-1-soundfont",
        modification_status="unmodified",
    ),
}


def attribution_for_soundfont(soundfont_name: str) -> SoundfontAttribution:
    """Attribution for the font of the given file name.

    An unknown name (a font the registry does not know) yields an
    honest "unverified" record rather than someone else's attribution —
    a font whose license is not confirmed must never inherit another
    font's.
    """
    known = SOUNDFONT_ATTRIBUTION.get(soundfont_name)
    if known is not None:
        return known
    return SoundfontAttribution(
        name=soundfont_name,
        author="",
        version="",
        license="unverified",
        license_url="",
        source_url="",
        notice_path="THIRD_PARTY_NOTICES.md",
        modification_status="unknown",
    )


def audio_metadata_tags(soundfont_name: str | None = None) -> dict[str, str]:
    """Vorbis-comment tags embedded in every rendered OGG/Opus file.

    Per §10 #12: standalone audio metadata records the library, the
    author, and the license URL. ffmpeg writes these as Vorbis
    comments on the Opus stream, so any player's metadata view shows
    the attribution. `soundfont_name` picks the record for the font
    actually rendered (None keeps the Salamander default, the Phase 1
    behaviour).
    """
    attribution = (
        attribution_for_soundfont(soundfont_name) if soundfont_name is not None else _SALAMANDER
    )
    return {
        "LIBRARY": attribution.name,
        "AUTHOR": attribution.author,
        "LICENSE": attribution.license,
        "LICENSE_URL": attribution.license_url,
    }


def license_obligations() -> dict[str, str]:
    """Notice-file pointers for the LGPL/LGPL-conditional components.

    FluidSynth and the bundled FFmpeg are LGPL (the audit gate forbids
    GPL/nonfree builds); Salamander and MFA Boston 1 are the CC BY 3.0
    conditionals and Wetthasinghe's Harmonium the CC BY 4.0 one.
    """
    return {
        "fluidsynth": "THIRD_PARTY_NOTICES.md#fluidsynth",
        "ffmpeg": "THIRD_PARTY_NOTICES.md#ffmpeg",
        "salamander-grand-piano": SOUNDFONT_NOTICE_PATH,
        "mfa-boston-1": "THIRD_PARTY_NOTICES.md#mfa-boston-1-soundfont",
        "wetthasinghe-harmonium": "THIRD_PARTY_NOTICES.md#wetthasinghes-harmonium-soundfont",
        "105-sitar": "THIRD_PARTY_NOTICES.md#105-sitar-soundfont",
    }


__all__ = [
    "SOUNDFONT_ATTRIBUTION",
    "SOUNDFONT_AUTHOR",
    "SOUNDFONT_LICENSE",
    "SOUNDFONT_LICENSE_URL",
    "SOUNDFONT_NAME",
    "SOUNDFONT_NOTICE_PATH",
    "SOUNDFONT_SOURCE_URL",
    "SOUNDFONT_VERSION",
    "SoundfontAttribution",
    "attribution_for_soundfont",
    "audio_metadata_tags",
    "license_obligations",
]
