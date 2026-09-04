"""Instrument + voice registries — what the audio stage can render.

Every place that would otherwise hardcode an instrument fact reads from
these tables, so adding an instrument is adding a row here (plus a
soundfont), not editing render call sites:

- `INSTRUMENT_PROGRAMS`: General MIDI program number (0-based) per
  instrument name; `build_smf` emits the program change so FluidSynth
  picks the right patch. Keys must match `saimc.spec.Instrument` 1:1 —
  the spec vocabulary and the render registry are drift-guard tested
  against each other.
- `soundfont_for_instrument()`: per-instrument SF2 resolution. See its
  docstring for the full chain.

Licensing: everything registered here renders from commercial-use-safe
soundfonts. FluidR3_GM (MIT, Frank Wen) covers the whole palette,
including the non-western voices GM provides (sitar, koto, shanai,
taiko, kalimba, bagpipe, shakuhachi). `scripts/download_soundfonts.py`
fetches it with a pinned sha256.

Planned dedicated fonts (commercial-safe, verified licenses — see
`scripts/download_soundfonts.py --status` for what is installed):
- MFA Boston 1 (CC-BY 3.0, museum-sampled): bansuri, sarangi, rudra
  veena, sarasvati veena, plus koto/shamisen/ud/qanoon/kora quality
  upgrades. Requires per-font preset mapping, since its preset layout
  is its own (not GM).
- 105-Sitar (public domain): better sitar than GM 104.
- Wetthasinghe's Harmonium (CC-BY 4.0): harmonium beyond the free-reed
  GM voice.
Instruments with no GM voice (tabla, tanpura, mridangam, ghatam) wait
for those fonts; /meta never advertises an instrument that cannot sound.

Voice IDs live with the score format (`saimc.compose.score`); this
module maps a voice's instrument name to everything the audio stage
needs to render it.
"""

from __future__ import annotations

import os
from pathlib import Path

# General MIDI program numbers (0-based) per instrument name. The keys
# are the `saimc.spec.Instrument` vocabulary.
INSTRUMENT_PROGRAMS: dict[str, int] = {
    # Keys
    "piano": 0,  # acoustic grand (Salamander renders this one)
    "harpsichord": 6,
    "celesta": 8,
    "music_box": 10,
    # Mallets and bells
    "glockenspiel": 9,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "tubular_bells": 14,
    "dulcimer": 15,
    # Organs and free reeds
    "pipe_organ": 20,
    "accordion": 21,
    "harmonica": 22,
    # Plucked strings
    "nylon_guitar": 24,
    "steel_guitar": 25,
    "banjo": 105,
    "shamisen": 106,
    "koto": 107,
    "sitar": 104,
    # Bowed strings and ensembles
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "tremolo_strings": 44,
    "pizzicato_strings": 45,
    "strings": 48,
    "fiddle": 110,
    # Harp and timpani
    "harp": 46,
    "timpani": 47,
    # Choir
    "choir": 52,
    # Brass
    "french_horn": 60,
    "brass_section": 61,
    "trumpet": 56,
    "muted_trumpet": 59,
    "trombone": 57,
    "tuba": 58,
    # Woodwinds
    "flute": 73,
    "piccolo": 72,
    "recorder": 74,
    "pan_flute": 75,
    "ocarina": 79,
    "oboe": 68,
    "english_horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    # Saxophone family
    "soprano_sax": 64,
    "alto_sax": 65,
    "tenor_sax": 66,
    "baritone_sax": 67,
    # World
    "bagpipe": 109,
    "shakuhachi": 77,
    "shanai": 111,  # shehnai
    "kalimba": 108,
    "steel_drums": 114,
    "agogo": 113,
    "woodblock": 115,
    "taiko": 116,
}

INSTRUMENT_FAMILIES: dict[str, str] = {
    "piano": "western",
    "harpsichord": "western",
    "celesta": "western",
    "music_box": "western",
    "glockenspiel": "western",
    "vibraphone": "western",
    "marimba": "western",
    "xylophone": "western",
    "tubular_bells": "western",
    "dulcimer": "western",
    "pipe_organ": "western",
    "accordion": "western",
    "harmonica": "western",
    "nylon_guitar": "western",
    "steel_guitar": "western",
    "banjo": "world",
    "shamisen": "world",
    "koto": "world",
    "sitar": "world",
    "violin": "western",
    "viola": "western",
    "cello": "western",
    "contrabass": "western",
    "tremolo_strings": "western",
    "pizzicato_strings": "western",
    "strings": "western",
    "fiddle": "western",
    "harp": "western",
    "timpani": "western",
    "choir": "western",
    "french_horn": "western",
    "brass_section": "western",
    "trumpet": "western",
    "muted_trumpet": "western",
    "trombone": "western",
    "tuba": "western",
    "flute": "western",
    "piccolo": "western",
    "recorder": "western",
    "pan_flute": "world",
    "ocarina": "western",
    "oboe": "western",
    "english_horn": "western",
    "bassoon": "western",
    "clarinet": "western",
    "soprano_sax": "western",
    "alto_sax": "western",
    "tenor_sax": "western",
    "baritone_sax": "western",
    "bagpipe": "world",
    "shakuhachi": "world",
    "shanai": "world",
    "kalimba": "world",
    "steel_drums": "world",
    "agogo": "world",
    "woodblock": "world",
    "taiko": "world",
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
