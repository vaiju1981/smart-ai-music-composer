"""CompositionSpec — the Phase 1 parser contract.

This is the canonical, versioned schema that the prompt parser (LLM +
fallback) must produce. Every downstream component consumes and emits
artifacts shaped against this schema. See `docs/roadmap.md` §6.

The schema is intentionally narrow in Phase 1: three moods, one
instrument per piece, no famous-piece catalog. `instrumentation` was
widened from piano-only to the `Instrument` enum for Phase 2 — that is
an additive change (every previously-valid spec value is still valid),
so `SPEC_SCHEMA_VERSION` is unchanged. `request_kind` and
`humanization` widen under a new `schema_version`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

SPEC_SCHEMA_VERSION: int = 1
"""Bump on any breaking change to CompositionSpec."""

# Type-level companion of SPEC_SCHEMA_VERSION so Pydantic / mypy accept the Literal.
SchemaVersion1 = Literal[1]

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

    schema_version: SchemaVersion1 = Field(
        default=1,
        description="Schema version of this CompositionSpec. Must equal the saimc constant.",
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
    instrumentation: Instrument = Field(
        default=Instrument.PIANO,
        description=(
            "The instrument the piece is written for. One instrument per "
            "piece in Phase 2; per-voice orchestration arrives later."
        ),
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="RNG seed for reproducibility. None means engine chooses and reports.",
    )
    humanization: Literal["none"] = Field(
        default="none",
        description="Phase 1: only 'none'. Phase 2+ will add light | expressive.",
    )


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
    "DURATION_SECONDS_DEFAULT",
    "DURATION_SECONDS_MAX",
    "DURATION_SECONDS_MIN",
    "SPEC_SCHEMA_VERSION",
    "TEMPO_BPM_MAX",
    "TEMPO_BPM_MIN",
    "CompositionSpec",
    "Instrument",
    "Mood",
    "RequestKind",
    "SpecError",
    "TimeSignature",
    "WesternKey",
]
