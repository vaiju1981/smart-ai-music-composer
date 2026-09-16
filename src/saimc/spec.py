"""CompositionSpec — the Phase 1 parser contract.

This is the canonical, versioned schema that the prompt parser (LLM +
fallback) must produce. Every downstream component consumes and emits
artifacts shaped against this schema. See `docs/roadmap.md` §6.

The schema is intentionally narrow in Phase 1: three moods, no
famous-piece catalog. `instrumentation` was widened from piano-only to
the `Instrument` enum for Phase 2 — an additive change (every
previously-valid spec value is still valid), so `SPEC_SCHEMA_VERSION`
stayed at 1. Version 2 widens `humanization` beyond "none" (the default
moves to "light"). Version 3 turns `instrumentation` from a single
instrument into a role-tagged ensemble list (`VoiceRole` + `Instrument`
entries): a scalar instrument value remains valid input and coerces to
the mood's default ensemble, so version-1/2 specs stay readable — every
value they could carry is still valid.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

# Type-level companion of SPEC_SCHEMA_VERSION so Pydantic / mypy accept the Literal.
SchemaVersion = Literal[1, 2, 3]

SPEC_SCHEMA_VERSION: SchemaVersion = 3
"""Bump on any breaking change to CompositionSpec."""

DURATION_SECONDS_MIN: int = 30
DURATION_SECONDS_MAX: int = 600
DURATION_SECONDS_DEFAULT: int = 180

TEMPO_BPM_MIN: int = 40
TEMPO_BPM_MAX: int = 240


class RequestKind(StrEnum):
    """Phase 1 only supports mood generation. `famous_piece` is reserved."""

    MOOD_GENERATION = "mood_generation"
    FAMOUS_PIECE = "famous_piece"


class Mood(StrEnum):
    """Bounded Phase 1 vocabulary. New moods require a new schema_version."""

    CALMING = "calming"
    ELECTRIFYING = "electrifying"
    SLEEP = "sleep"


class Instrument(StrEnum):
    """Bounded instrument vocabulary.

    Names match `INSTRUMENT_PROGRAMS` in `saimc.render.instruments` 1:1
    (a drift-guard test asserts the two stay in sync). Widening here is
    backward-compatible — old specs only ever contained values that
    remain members — so this does not bump `SPEC_SCHEMA_VERSION`.
    """

    # Keys
    PIANO = "piano"
    HARPSICHORD = "harpsichord"
    CELESTA = "celesta"
    MUSIC_BOX = "music_box"
    # Mallets and bells
    GLOCKENSPIEL = "glockenspiel"
    VIBRAPHONE = "vibraphone"
    MARIMBA = "marimba"
    XYLOPHONE = "xylophone"
    TUBULAR_BELLS = "tubular_bells"
    DULCIMER = "dulcimer"
    # Organs and free reeds
    PIPE_ORGAN = "pipe_organ"
    ACCORDION = "accordion"
    HARMONICA = "harmonica"
    # Plucked strings
    NYLON_GUITAR = "nylon_guitar"
    STEEL_GUITAR = "steel_guitar"
    BANJO = "banjo"
    SHAMISEN = "shamisen"
    KOTO = "koto"
    SITAR = "sitar"
    # Bowed strings and ensembles
    VIOLIN = "violin"
    VIOLA = "viola"
    CELLO = "cello"
    CONTRABASS = "contrabass"
    TREMOLO_STRINGS = "tremolo_strings"
    PIZZICATO_STRINGS = "pizzicato_strings"
    STRINGS = "strings"
    FIDDLE = "fiddle"
    # Harp and timpani
    HARP = "harp"
    TIMPANI = "timpani"
    # Choir
    CHOIR = "choir"
    # Brass
    FRENCH_HORN = "french_horn"
    BRASS_SECTION = "brass_section"
    TRUMPET = "trumpet"
    MUTED_TRUMPET = "muted_trumpet"
    TROMBONE = "trombone"
    TUBA = "tuba"
    # Woodwinds
    FLUTE = "flute"
    PICCOLO = "piccolo"
    RECORDER = "recorder"
    PAN_FLUTE = "pan_flute"
    OCARINA = "ocarina"
    OBOE = "oboe"
    ENGLISH_HORN = "english_horn"
    BASSOON = "bassoon"
    CLARINET = "clarinet"
    # Saxophone family
    SOPRANO_SAX = "soprano_sax"
    ALTO_SAX = "alto_sax"
    TENOR_SAX = "tenor_sax"
    BARITONE_SAX = "baritone_sax"
    # World
    BAGPIPE = "bagpipe"
    SHAKUHACHI = "shakuhachi"
    SHANAI = "shanai"
    KALIMBA = "kalimba"
    STEEL_DRUMS = "steel_drums"
    AGOGO = "agogo"
    WOODBLOCK = "woodblock"
    TAIKO = "taiko"
    # Percussion kit — no melodic program; renders through GM channel 10
    # where the note pitch IS the drum piece. Composes a piano
    # accompaniment under a mood-driven rhythm-pattern voice (see
    # `saimc.compose.percussion`).
    DRUM_SET = "drum_set"
    # Dedicated-font instrument — GM has no harmonium voice; it renders
    # from Wetthasinghe's Harmonium (CC-BY 4.0) via `FONT_PRESETS`.
    HARMONIUM = "harmonium"
    # Dedicated-font instruments — GM has no voice for any of these; they
    # render from MFA Boston 1 (CC-BY 3.0, museum-sampled) via
    # `FONT_PRESETS`.
    BANSURI = "bansuri"
    SARANGI = "sarangi"
    RUDRA_VEENA = "rudra_veena"
    SARASVATI_VEENA = "sarasvati_veena"
    QANOON = "qanoon"
    UD = "ud"
    KORA = "kora"


class WesternKey(StrEnum):
    """Bounded to common Western keys. `None` in the spec means engine chooses."""

    C_MAJOR = "C"
    G_MAJOR = "G"
    D_MAJOR = "D"
    A_MAJOR = "A"
    E_MAJOR = "E"
    B_MAJOR = "B"
    F_SHARP_MAJOR = "F#"
    F_MAJOR = "F"
    B_FLAT_MAJOR = "Bb"
    E_FLAT_MAJOR = "Eb"
    A_FLAT_MAJOR = "Ab"
    D_FLAT_MAJOR = "Db"
    G_FLAT_MAJOR = "Gb"
    A_MINOR = "Am"
    E_MINOR = "Em"
    B_MINOR = "Bm"
    F_SHARP_MINOR = "F#m"
    C_SHARP_MINOR = "C#m"
    G_SHARP_MINOR = "G#m"
    D_MINOR = "Dm"
    G_MINOR = "Gm"
    C_MINOR = "Cm"
    F_MINOR = "Fm"
    B_FLAT_MINOR = "Bbm"
    E_FLAT_MINOR = "Ebm"


class TimeSignature(StrEnum):
    """Bounded to common signatures. New values require a schema_version bump."""

    FOUR_FOUR = "4/4"
    THREE_FOUR = "3/4"
    SIX_EIGHT = "6/8"
    TWO_FOUR = "2/4"
    FIVE_FOUR = "5/4"
    SEVEN_EIGHT = "7/8"


class VoiceRole(StrEnum):
    """What an instrument does in the ensemble. One role per engine voice."""

    MELODY = "melody"
    HARMONY = "harmony"
    BASS = "bass"
    PERCUSSION = "percussion"


# Canonical entry order for the `instrumentation` list. The spec's
# validator re-sorts entries into this order so dumps, corpus labels,
# and canonical hashes are independent of the order the parser emitted.
ROLE_ORDER: tuple[VoiceRole, ...] = (
    VoiceRole.MELODY,
    VoiceRole.HARMONY,
    VoiceRole.BASS,
    VoiceRole.PERCUSSION,
)

# Ensemble size ceiling: 1 melody + <=2 harmony + <=1 bass + <=1
# percussion. Also the MIDI-channel budget: the render stage maps each
# voice to its own channel, and melodic voices must stay off channel 10.
ENSEMBLE_MAX_VOICES: int = 5


class InstrumentationEntry(BaseModel):
    """One instrument in the ensemble, tagged with its role."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    role: VoiceRole = Field(description="What this instrument does in the ensemble.")
    instrument: Instrument = Field(description="Which instrument plays the role.")


# Instruments with a dedicated soundfont in `FONT_PRESETS`
# (`saimc.render.instruments` — a drift-guard test keeps the enum and
# the font table honest). Only one dedicated font can load per job, so
# ensemble rules treat them specially: such a melody's accompaniment
# stays inside its own font (harmony doubles the melody instrument,
# separated by channel gain/pan at render time), and duplicate
# instruments are only tolerated among these.
DEDICATED_FONT_INSTRUMENTS: frozenset[Instrument] = frozenset(
    {
        Instrument.SITAR,
        Instrument.KOTO,
        Instrument.SHAMISEN,
        Instrument.HARMONIUM,
        Instrument.BANSURI,
        Instrument.SARANGI,
        Instrument.RUDRA_VEENA,
        Instrument.SARASVATI_VEENA,
        Instrument.QANOON,
        Instrument.UD,
        Instrument.KORA,
    }
)


class CompositionSpec(BaseModel):
    """The Phase 1 parser contract.

    Produced by the LLM adapter (with up to two schema-in-context repairs)
    or by the fallback parser. Validated locally against this same model
    before any downstream stage consumes it.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=False,
        str_strip_whitespace=True,
    )

    schema_version: SchemaVersion = Field(
        default=SPEC_SCHEMA_VERSION,
        description=(
            "Schema version of this CompositionSpec. Older-version specs "
            "remain valid input; new specs are written as the current "
            "version."
        ),
    )
    request_kind: Literal[RequestKind.MOOD_GENERATION] = Field(
        default=RequestKind.MOOD_GENERATION,
        description="Phase 1: only mood_generation. Reserved: famous_piece.",
    )
    duration_seconds: Annotated[
        int,
        Field(
            ge=DURATION_SECONDS_MIN,
            le=DURATION_SECONDS_MAX,
            description=f"Target duration in seconds ({DURATION_SECONDS_MIN}-{DURATION_SECONDS_MAX}).",
        ),
    ] = DURATION_SECONDS_DEFAULT
    tempo_bpm: int | None = Field(
        default=None,
        ge=TEMPO_BPM_MIN,
        le=TEMPO_BPM_MAX,
        description="Tempo in BPM. None means engine derives from mood.",
    )
    key: WesternKey | None = Field(
        default=None,
        description="Western key. None means engine chooses.",
    )
    time_signature: TimeSignature = Field(
        default=TimeSignature.FOUR_FOUR,
        description="Time signature.",
    )
    mood: Mood = Field(
        description="Phase 1 mood vocabulary: calming | electrifying | sleep.",
    )
    instrumentation: list[InstrumentationEntry] = Field(
        default_factory=lambda: [InstrumentationEntry(role=VoiceRole.MELODY, instrument=Instrument.PIANO)],
        min_length=1,
        max_length=ENSEMBLE_MAX_VOICES,
        description=(
            "The ensemble the piece is written for: role-tagged entries "
            "sorted melody-first. Exactly one melody; at most one bass, "
            "one percussion (drum_set only), and two harmony voices; "
            "instruments must be distinct. A bare instrument string is "
            "accepted for backwards compatibility and expands to the "
            "mood's default ensemble."
        ),
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description=(
            "RNG seed for reproducibility. None does not randomise the "
            "piece: the engine composes against a fixed default seed of 0, "
            "and the manifest records this field's None rather than that "
            "resolved value. So None reproduces exactly, and seed=0 and "
            "seed=None yield identical music. Pass an explicit seed to vary "
            "a piece."
        ),
    )
    humanization: Literal["none", "light", "expressive"] = Field(
        default="light",
        description=(
            "How much human variation the performance layer applies to the "
            "realized plan (the engraved score always stays on the grid): "
            "'none' is machine-perfect timing; 'light' adds small timing "
            "and velocity variation; 'expressive' widens both further and "
            "shortens repeated notes into staccato."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_and_normalize_instrumentation(cls, data: Any) -> Any:
        """Coerce legacy scalar instrumentation into a normalized ensemble.

        A bare instrument string (the version-1/2 shape) expands to the
        mood's default ensemble; a missing field becomes the default
        ensemble for the spec's mood; explicit entry lists get missing
        harmony/bass roles filled and all entries sorted into
        `ROLE_ORDER` — so dumps, corpus labels, and canonical hashes are
        independent of what the parser emitted. Malformed shapes (a list
        of bare strings, unknown roles) are left for field validation to
        reject with pydantic's own message, except the bare-string list
        which gets a targeted, repairable error. The mood's tables live
        in `saimc.compose.ensemble` — imported lazily because that
        module imports this one.
        """
        if not isinstance(data, dict):
            return data
        mood = data.get("mood")
        raw = data.get("instrumentation")
        from saimc.compose.ensemble import normalize_instrumentation

        data["instrumentation"] = normalize_instrumentation(raw, mood)
        return data

    @model_validator(mode="after")
    def _validate_ensemble(self) -> CompositionSpec:
        """Enforce ensemble shape that field-level constraints cannot express."""
        entries = self.instrumentation
        roles = [entry.role for entry in entries]
        if roles.count(VoiceRole.MELODY) != 1:
            raise ValueError(
                "ensemble must have exactly one melody role; got "
                f"{roles.count(VoiceRole.MELODY)}"
            )
        if roles.count(VoiceRole.BASS) > 1:
            raise ValueError("ensemble supports at most one bass entry")
        if roles.count(VoiceRole.PERCUSSION) > 1:
            raise ValueError("ensemble supports at most one percussion entry")
        for entry in entries:
            if entry.role == VoiceRole.PERCUSSION and entry.instrument != Instrument.DRUM_SET:
                raise ValueError(
                    "percussion role requires the drum_set instrument; "
                    f"got {entry.instrument.value}"
                )
        harmony = roles.count(VoiceRole.HARMONY)
        if harmony > 2:
            raise ValueError(f"ensemble supports at most two harmony entries; got {harmony}")
        instruments = [entry.instrument for entry in entries]
        duplicates = {i for i in instruments if instruments.count(i) > 1}
        if duplicates:
            font_only_ok = duplicates <= DEDICATED_FONT_INSTRUMENTS
            # The drum-set layout is the other exception: its melody
            # and accompaniment are both the piano (one Salamander
            # voice under the kit), exactly as the Phase 2 drum-set
            # branch composed it.
            drum_set_ok = (
                duplicates == {Instrument.PIANO}
                and roles == [VoiceRole.MELODY, VoiceRole.BASS, VoiceRole.PERCUSSION]
            )
            if not (font_only_ok or drum_set_ok):
                names = ", ".join(sorted(i.value for i in duplicates))
                raise ValueError(
                    f"each instrument may appear at most once in an ensemble: {names}"
                )
        return self


class SpecError(BaseModel):
    """Structured error returned when a prompt cannot be parsed.

    `error_code` is a stable identifier; downstream tooling can branch on it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    error_code: str
    message: str
    stage: Literal["parsing", "validating", "fallback"]
    attempts: NonNegativeInt = 0


__all__ = [
    "DEDICATED_FONT_INSTRUMENTS",
    "DURATION_SECONDS_DEFAULT",
    "DURATION_SECONDS_MAX",
    "DURATION_SECONDS_MIN",
    "ENSEMBLE_MAX_VOICES",
    "ROLE_ORDER",
    "SPEC_SCHEMA_VERSION",
    "TEMPO_BPM_MAX",
    "TEMPO_BPM_MIN",
    "CompositionSpec",
    "Instrument",
    "InstrumentationEntry",
    "Mood",
    "RequestKind",
    "SpecError",
    "TimeSignature",
    "VoiceRole",
    "WesternKey",
]
