"""Instrument + voice registries — the seams Phase 2's orchestra needs.

Phase 1 is piano-only, but every place that would otherwise hardcode
that fact reads from these tables instead, so adding an instrument is
adding rows here (plus a soundfont), not editing render call sites:

- `INSTRUMENT_PROGRAMS`: General MIDI program number per instrument
  name; `build_smf` emits the program change so FluidSynth picks the
  right patch.
- `soundfont_for_instrument()`: per-instrument SF2 resolution. Each
  instrument can override its soundfont with
  `SAIMC_SOUNDFONT_<INSTRUMENT>` (e.g. `SAIMC_SOUNDFONT_PIANO`);
  the default maps to the Salamander grand piano.

Voice IDs live with the score format (`saimc.compose.score`); this
module maps a voice's instrument name to everything the audio stage
needs to render it.
"""

from __future__ import annotations

import os
from pathlib import Path

# General MIDI program numbers (0-based) per instrument name.
INSTRUMENT_PROGRAMS: dict[str, int] = {
    "piano": 0,
}

_DEFAULT_SOUNDFONT = Path("./assets/Salamander.sf2")


def soundfont_for_instrument(instrument: str) -> Path:
    """Resolve the SF2 path for an instrument name.

    Resolution: `SAIMC_SOUNDFONT_<INSTRUMENT>` env var (instrument
    upper-cased, non-alphanumerics to underscore) > the Phase 1
    Salamander default.
    """
    env_key = "SAIMC_SOUNDFONT_" + instrument.upper().replace("-", "_").replace(" ", "_")
    return Path(os.environ.get(env_key, str(_DEFAULT_SOUNDFONT)))


__all__ = ["INSTRUMENT_PROGRAMS", "soundfont_for_instrument"]
