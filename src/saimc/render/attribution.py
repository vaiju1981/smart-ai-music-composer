"""Third-party attribution constants per `docs/roadmap.md` §10 #12.

The Salamander Grand Piano soundfont is CC BY 3.0 — attribution is a
distribution condition, not a courtesy. This module is the single
source of truth for that attribution: the notice file, the manifest's
`assets` entry, the OGG/Opus metadata tags, and
`THIRD_PARTY_NOTICES.md` all draw from here so they cannot drift.

Phase 1 ships the sample pack **unmodified**; if that ever changes
(conversion, retuning, trimming), the modification must be stated
explicitly here and in the notice file per CC BY 3.0 §4(b).
"""

from __future__ import annotations

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


def audio_metadata_tags() -> dict[str, str]:
    """Vorbis-comment tags embedded in every rendered OGG/Opus file.

    Per §10 #12: standalone audio metadata records the library, the
    author, and the license URL. ffmpeg writes these as Vorbis
    comments on the Opus stream, so any player's metadata view shows
    the attribution.
    """
    return {
        "LIBRARY": SOUNDFONT_NAME,
        "AUTHOR": SOUNDFONT_AUTHOR,
        "LICENSE": SOUNDFONT_LICENSE,
        "LICENSE_URL": SOUNDFONT_LICENSE_URL,
    }


def license_obligations() -> dict[str, str]:
    """Notice-file pointers for the LGPL/LGPL-conditional components.

    FluidSynth and the bundled FFmpeg are LGPL (the audit gate forbids
    GPL/nonfree builds); Salamander is the CC BY 3.0 conditional.
    """
    return {
        "fluidsynth": "THIRD_PARTY_NOTICES.md#fluidsynth",
        "ffmpeg": "THIRD_PARTY_NOTICES.md#ffmpeg",
        "salamander-grand-piano": SOUNDFONT_NOTICE_PATH,
    }


__all__ = [
    "SOUNDFONT_AUTHOR",
    "SOUNDFONT_LICENSE",
    "SOUNDFONT_LICENSE_URL",
    "SOUNDFONT_NAME",
    "SOUNDFONT_NOTICE_PATH",
    "SOUNDFONT_SOURCE_URL",
    "SOUNDFONT_VERSION",
    "audio_metadata_tags",
    "license_obligations",
]
