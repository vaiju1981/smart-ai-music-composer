"""Ensemble resolution — what a spec's `instrumentation` means in voices.

The single place that owns the mood-driven default ensembles and the
spec-entries-to-voices mapping, so the engine, the job stages, and
`/meta` never re-derive it. `saimc.spec` calls `normalize_instrumentation`
lazily from its `mode="before"` validator (this module imports `saimc.spec`,
so a module-level import back would be a cycle).

Two kinds of defaulting, both collected here:

- **Scalar coercion** (legacy version-1/2 specs, and prompts like
  "electrifying drums"): a bare instrument becomes a full ensemble —
  melody = the instrument, harmony/bass from the mood's `SCALAR_*`
  tables. The `drum_set` special case preserves the Phase 2 drum-kit
  behavior exactly (piano accompaniment under the kit, no melodic
  harmony), and dedicated-font melodies keep their accompaniment inside
  their own font.
- **Role filling** (explicit role-tagged lists): only missing harmony
  and bass are filled from `DEFAULT_HARMONY`/`DEFAULT_BASS`. Percussion
  is never auto-added — a kit appearing unrequested is a texture
  surprise, and it keeps the deterministic fallback parser's output
  stable.

Dedicated-font melodies (no GM voice) keep their accompaniment inside
their own font: harmony doubles the melody instrument (separated by
channel gain/pan at render time) and bass stays silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from saimc.spec import (
    DEDICATED_FONT_INSTRUMENTS,
    ROLE_ORDER,
    CompositionSpec,
    Instrument,
    InstrumentationEntry,
    Mood,
    VoiceRole,
)

# Fill table for explicit role-tagged ensembles.
DEFAULT_HARMONY: dict[str, str] = {
    "electrifying": "strings",
    "calming": "strings",
    "sleep": "choir",
}
DEFAULT_BASS: dict[str, str] = {
    "electrifying": "contrabass",
    "calming": "cello",
    "sleep": "cello",
}

# Scalar-coercion table: the ensemble a bare instrument string expands
# to. The pad is softer than the fill table's (pizzicato/celesta rather
# than a sustained wash) so a coerced legacy spec stays close to the
# single-patch texture its version-2 render had.
SCALAR_HARMONY: dict[str, str] = {
    "electrifying": "strings",
    "calming": "pizzicato_strings",
    "sleep": "celesta",
}
SCALAR_BASS: dict[str, str] = {
    "electrifying": "contrabass",
    "calming": "cello",
    "sleep": "cello",
}

# A scalar drum_set spec means a drum-kit piece: the piano stays as the
# accompaniment and the kit plays the mood's rhythm pattern — the exact
# voice layout the Phase 2 drum-set branch composes.
SCALAR_DRUM_SET_HARMONY: str | None = None
SCALAR_DRUM_SET_BASS: str = "piano"


@dataclass(frozen=True)
class Ensemble:
    """The resolved voices a spec composes with.

    `None` means the role has no voice in this piece.
    """

    melody: str
    harmony: str | None
    bass: str | None
    percussion: str | None

    def voice_instruments(self) -> dict[int, str]:
        """Map engine voice ids to instrument names.

        Voice ids are the score's canonical ones (`VOICE_BASS=0`,
        `VOICE_MELODY=1`, `VOICE_PERCUSSION=2`, `VOICE_HARMONY=3`).
        """
        voices: dict[int, str] = {1: self.melody}
        if self.bass is not None:
            voices[0] = self.bass
        if self.percussion is not None:
            voices[2] = self.percussion
        if self.harmony is not None:
            voices[3] = self.harmony
        return voices


def resolve_ensemble(spec: CompositionSpec) -> Ensemble:
    """Read the spec's normalized instrumentation into an Ensemble."""
    by_role: dict[VoiceRole, list[str]] = {}
    for entry in spec.instrumentation:
        by_role.setdefault(entry.role, []).append(entry.instrument.value)
    harmony = by_role.get(VoiceRole.HARMONY, [])
    return Ensemble(
        melody=by_role[VoiceRole.MELODY][0],
        harmony=harmony[0] if harmony else None,
        bass=by_role[VoiceRole.BASS][0] if by_role.get(VoiceRole.BASS) else None,
        percussion=by_role[VoiceRole.PERCUSSION][0]
        if by_role.get(VoiceRole.PERCUSSION)
        else None,
    )


def _entry(role: VoiceRole, instrument: str) -> dict[str, str]:
    return {"role": role.value, "instrument": instrument}


def default_ensemble(melody: str, mood: str | None) -> list[dict[str, str]]:
    """The role-tagged ensemble a scalar melody instrument expands to.

    Shared by the spec's legacy coercion and by `/meta`'s
    `default_ensembles` documentation. Unknown moods fall back to the
    calming voicing (the spec's first mood).
    """
    mood_key = mood if mood in SCALAR_HARMONY else Mood.CALMING.value
    if melody == Instrument.DRUM_SET.value:
        return [
            _entry(VoiceRole.MELODY, "piano"),
            _entry(VoiceRole.BASS, SCALAR_DRUM_SET_BASS),
            _entry(VoiceRole.PERCUSSION, Instrument.DRUM_SET.value),
        ]
    if melody in {i.value for i in DEDICATED_FONT_INSTRUMENTS}:
        return [
            _entry(VoiceRole.MELODY, melody),
            _entry(VoiceRole.HARMONY, melody),
        ]
    entries = [_entry(VoiceRole.MELODY, melody)]
    # A scalar melody that IS the mood's pad/bass pick would double an
    # instrument the validator forbids duplicating — the melody voice
    # simply covers that role instead.
    if SCALAR_HARMONY[mood_key] != melody:
        entries.append(_entry(VoiceRole.HARMONY, SCALAR_HARMONY[mood_key]))
    if SCALAR_BASS[mood_key] != melody:
        entries.append(_entry(VoiceRole.BASS, SCALAR_BASS[mood_key]))
    return entries


def normalize_instrumentation(raw: Any, mood: Any) -> list[dict[str, str]]:
    """Normalize any accepted `instrumentation` input to entry dicts.

    Accepted shapes: a bare instrument name (legacy scalar), a list of
    `{role, instrument}` maps, and `None` (the field was omitted — the
    mood's default piano-led ensemble). Returns plain dicts sorted into
    `ROLE_ORDER` so pydantic builds `InstrumentationEntry` items from
    them. Raises `ValueError` for shapes the repair loop should fix in
    one step; everything else is left to field validation.

    `mood` may be `None` (the spec failed mood validation anyway) —
    fills then use the neutral default.
    """
    mood_key = str(mood) if mood is not None else None
    if raw is None:
        return default_ensemble(Instrument.PIANO.value, mood_key)
    if isinstance(raw, (str, Instrument)):
        return default_ensemble(str(raw), mood_key)
    if isinstance(raw, list):
        if raw and all(isinstance(item, str) for item in raw):
            names = ", ".join(raw)
            raise ValueError(
                "instrumentation must be role-tagged entries "
                '[{"role": "melody", "instrument": ...}]; got bare names: '
                f"{names}"
            )
        entries: list[dict[str, str]] = []
        roles_present: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError(
                    "instrumentation entries must be objects with role and instrument"
                )
            entry = dict(item)
            roles_present.add(str(entry.get("role", "")))
            entries.append(entry)
        melody = _melody_instrument(entries)
        if melody is not None:
            entries, roles_present = _fill_missing_roles(entries, roles_present, mood_key, melody)
        return _sorted_entries(entries)
    raise ValueError(
        "instrumentation must be an instrument name or a list of role-tagged entries"
    )


def _melody_instrument(entries: list[dict[str, str]]) -> str | None:
    for entry in entries:
        if str(entry.get("role", "")) == VoiceRole.MELODY.value:
            instrument = entry.get("instrument")
            if isinstance(instrument, str):
                return instrument
    return None


def _fill_missing_roles(
    entries: list[dict[str, str]],
    roles_present: set[str],
    mood_key: str | None,
    melody: str,
) -> tuple[list[dict[str, str]], set[str]]:
    """Add only the missing harmony/bass roles; percussion is never auto-added."""
    mood_fallback = Mood.CALMING.value if mood_key not in DEFAULT_HARMONY else mood_key
    dedicated = melody in {i.value for i in DEDICATED_FONT_INSTRUMENTS}
    if VoiceRole.HARMONY.value not in roles_present:
        if dedicated:
            entries.append(_entry(VoiceRole.HARMONY, melody))
        elif DEFAULT_HARMONY[mood_fallback] != melody:
            # Never fill a role with the melody's own instrument.
            entries.append(_entry(VoiceRole.HARMONY, DEFAULT_HARMONY[mood_fallback]))
            roles_present.add(VoiceRole.HARMONY.value)
    if VoiceRole.BASS.value not in roles_present and not dedicated:
        if DEFAULT_BASS[mood_fallback] != melody:
            entries.append(_entry(VoiceRole.BASS, DEFAULT_BASS[mood_fallback]))
            roles_present.add(VoiceRole.BASS.value)
    return entries, roles_present


def _sorted_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    order = {role.value: index for index, role in enumerate(ROLE_ORDER)}
    return sorted(
        entries,
        key=lambda e: (order.get(str(e.get("role", "")), len(order)), str(e.get("instrument", ""))),
    )


__all__ = [
    "DEFAULT_BASS",
    "DEFAULT_HARMONY",
    "Ensemble",
    "resolve_ensemble",
]