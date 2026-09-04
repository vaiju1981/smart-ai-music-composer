"""Instrument + voice registries — the seams Phase 2's orchestra needs.

Every place that would otherwise hardcode an instrument fact reads from
these tables, so adding an instrument is adding a row here (plus a
soundfont), not editing render call sites:

- `INSTRUMENT_PROGRAMS`: General MIDI program number (0-based) per
  instrument name; `build_smf` emits the program change so FluidSynth
  picks the right patch.
- `soundfont_for_instrument()`: per-instrument SF2 resolution. See its
  docstring for the full chain.

Licensing: everything registered here renders from commercial-use-safe
soundfonts. FluidR3_GM (MIT, Frank Wen) covers the whole palette,
including the non-western instruments GM provides (sitar, koto, shanai,
taiko, kalimba). `scripts/download_soundfonts.py` fetches it with a
pinned sha256.

Not yet renderable but planned: Indian classical instruments with no
GM equivalent (tabla, tanpura, bansuri, sarod, veena, mridangam,
ghatam, sarangi). They need dedicated soundfonts and — because they
follow raga/tala rather than western chord progressions — their own
composition engines via the composer registry
(`saimc.jobs.stages._composer_for_instrumentation`). Adding their rows
here waits for a commercial-safe font, so /meta never advertises an
instrument that cannot sound right.

Voice IDs live with the score format (`saimc.compose.score`); this
module maps a voice's instrument name to everything the audio stage
needs to render it.
"""

from __future__ import annotations

import os
from pathlib import Path

# General MIDI program numbers (0-based) per instrument name.
INSTRUMENT_PROGRAMS: dict[str, int] = {
    # Keyboard / core
    "piano": 0,  # acoustic grand (Salamander renders this one)
    "strings": 48,  # string ensemble
    # Strings
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "harp": 46,
    "timpani": 47,
    # Woodwinds
    "flute": 73,
    "oboe": 68,
    "clarinet": 71,
    "bassoon": 70,
    # Brass
    "french_horn": 60,
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    # World (GM-covered; quality upgrades need dedicated fonts)
    "sitar": 104,  # Indian classical
    "koto": 107,  # Japanese
    "shanai": 111,  # shehnai, Indian classical
    "taiko": 116,  # Japanese drum
    "kalimba": 108,  # African
}

INSTRUMENT_FAMILIES: dict[str, str] = {
    "piano": "western",
    "strings": "western",
    "violin": "western",
    "viola": "western",
    "cello": "western",
    "contrabass": "western",
    "harp": "western",
    "timpani": "western",
    "flute": "western",
    "oboe": "western",
    "clarinet": "western",
    "bassoon": "western",
    "french_horn": "western",
    "trumpet": "western",
    "trombone": "western",
    "tuba": "western",
    "sitar": "world",
    "koto": "world",
    "shanai": "world",
    "taiko": "world",
    "kalimba": "world",
}

# Where downloaded fonts live. `scripts/download_soundfonts.py` fills
# this directory with pinned-sha256 files (gitignored — re-downloadable
# from documented sources, like the ffmpeg build).
SOUNDFONT_DIR = Path("./assets/soundfonts")

# General-purpose font covering every registered instrument. MIT
# licensed (commercial use OK) — see assets/soundfonts/FluidR3_COPYING.txt.
GENERAL_SOUNDFONT = SOUNDFONT_DIR / "FluidR3_GM.sf2"

# Phase 1's piano font: a far better grand piano than FluidR3's.
# CC-BY 3.0 — the attribution Vorbis tags in the OGG cover it.
PIANO_SOUNDFONT = Path("./assets/Salamander.sf2")


def _env_override(instrument: str) -> Path | None:
    env_key = "SAIMC_SOUNDFONT_" + instrument.upper().replace("-", "_").replace(" ", "_")
    value = os.environ.get(env_key)
    return Path(value) if value else None


def soundfont_for_instrument(instrument: str) -> Path:
    """Resolve the SF2 path for an instrument name.

    Resolution order:
    1. `SAIMC_SOUNDFONT_<INSTRUMENT>` env var (e.g. `SAIMC_SOUNDFONT_PIANO`)
    2. a per-instrument font dropped into `assets/soundfonts/<name>.sf2`
    3. for piano: the Salamander grand (best-in-class), else the general font
    4. the general font (FluidR3_GM) — every registered instrument sounds
    5. the Salamander path anyway, so the missing-font error names the
       documented default location
    """
    override = _env_override(instrument)
    if override is not None:
        return override
    per_instrument = SOUNDFONT_DIR / f"{instrument}.sf2"
    if per_instrument.exists():
        return per_instrument
    if instrument == "piano" and PIANO_SOUNDFONT.exists():
        return PIANO_SOUNDFONT
    if GENERAL_SOUNDFONT.exists():
        return GENERAL_SOUNDFONT
    return PIANO_SOUNDFONT


__all__ = [
    "GENERAL_SOUNDFONT",
    "INSTRUMENT_FAMILIES",
    "INSTRUMENT_PROGRAMS",
    "PIANO_SOUNDFONT",
    "SOUNDFONT_DIR",
    "soundfont_for_instrument",
]
