"""Phase 1 chord templates per mood.

Per `docs/roadmap.md` §5, Phase 1 uses fixed forms (8/16/32 bars) with
predefined harmonic templates per mood. There are 3 templates per mood,
9 total. Templates are hand-coded I-V-vi-IV-style progressions that
match each mood's idiom; they are *not* algorithmic.

Each template is a list of (scale_degree, duration_bars) tuples where
scale_degree is in the key's diatonic scale:
    0 = I (tonic)
    1 = ii / ii- in minor
    2 = iii / iii- in minor
    3 = IV / iv in minor
    4 = V / v in minor
    5 = vi / VI in minor
    6 = vii (in major: vii dim; in minor: VII major)

Mode-specific: in major keys, `ii/iii/vi/vii` are minor; in minor
keys, `ii°/III/VI/VII` are the proper minor-key chords. The
`chord_for_degree()` helper in this module handles the mode-aware
mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

from saimc.compose.score import KeySignature
from saimc.spec import WesternKey

PHRASE_BARS: int = 4
"""The phrase unit: the melody breathes at least once per phrase, and
the CC11 swells ride one rise-and-fall per phrase."""


class ChordSlot(NamedTuple):
    """One chord of a template: (degree, bars) plus colour options.

    `seventh` adds the diatonic 7th to the chord (the engine renders
    the mode-aware 4-tone table: Imaj7, ii7, V7, ...).
    `borrowed` renders the chord from the opposite mode's table — in a
    major key that is the parallel-minor colour (bVII instead of vii°),
    in a minor key the raised (harmonic-minor) dominant instead of v.
    `bass_degree` pins the bass to a scale degree (root-position
    cadences); None lets the bass walk to the nearest chord tone.
    """

    degree: int
    bars: int
    seventh: bool = False
    borrowed: bool = False
    bass_degree: int | None = None


@dataclass(frozen=True)
class ChordTemplate:
    """A single chord-progression template.

    `bars` is the total length of the template in bars; `chords` is a
    list of ChordSlot entries whose `bars` must sum to exactly `bars`.

    `name` is a stable identifier used in logs and manifest records.
    """

    name: str
    bars: int
    chords: tuple[ChordSlot, ...]

    def __post_init__(self) -> None:
        # The engine advances one `bars`-length block per section while
        # `_generate_section` emits one chord-bar per bar of `chords`;
        # any drift between the two desynchronises every later section
        # (chord_bars no longer match the notes). Reject it at import.
        total = sum(slot.bars for slot in self.chords)
        if total != self.bars:
            raise ValueError(
                f"template {self.name!r} declares {self.bars} bars "
                f"but its chord slots sum to {total}"
            )


@dataclass(frozen=True)
class MoodProfile:
    """Everything the composer needs to know about one mood.

    `MOOD_PROFILES` below is the single source of truth: adding a mood
    for Phase 2 means adding one `MoodProfile` entry (plus templates),
    not editing the lookups scattered across duration.py and engine.py.
    """

    name: str
    tempo_range_bpm: tuple[int, int]
    templates: tuple[ChordTemplate, ...]


# ---------------------------------------------------------------------------
# The mood registry (calming 50-80 BPM I-V-vi-IV-ish idioms, electrifying
# 100-160 BPM driving changes, sleep 40-64 BPM sparse/drone-ish)
# ---------------------------------------------------------------------------
MOOD_PROFILES: Mapping[str, MoodProfile] = {
    "calming": MoodProfile(
        name="calming",
        tempo_range_bpm=(50, 80),
        templates=(
            ChordTemplate(
                name="calming_50s_progression",
                bars=8,
                chords=(
                    ChordSlot(0, 2, seventh=True),  # Imaj7
                    ChordSlot(5, 2, seventh=True),  # vi7
                    ChordSlot(3, 2, seventh=True),  # IVmaj7
                    ChordSlot(4, 2),  # V
                ),
            ),
            ChordTemplate(
                name="calming_iii_substitute",
                bars=8,
                chords=(
                    ChordSlot(0, 2, seventh=True),
                    ChordSlot(2, 2, seventh=True),  # iii7
                    ChordSlot(3, 2, seventh=True),
                    ChordSlot(4, 2),
                ),
            ),
            ChordTemplate(
                name="calming_extended_16",
                bars=16,
                chords=(
                    ChordSlot(0, 2, seventh=True),
                    ChordSlot(5, 2, seventh=True),
                    ChordSlot(3, 2, seventh=True),
                    ChordSlot(4, 2),
                    ChordSlot(0, 2, seventh=True),
                    ChordSlot(5, 2, seventh=True),
                    ChordSlot(3, 2, seventh=True, borrowed=True),  # iv, minor colour
                    ChordSlot(1, 2, seventh=True),  # ii7 closes the last phrase
                ),
            ),
        ),
    ),
    "electrifying": MoodProfile(
        name="electrifying",
        tempo_range_bpm=(100, 160),
        templates=(
            ChordTemplate(
                name="electrifying_anthemic",
                bars=8,
                chords=(
                    ChordSlot(0, 2),  # I
                    ChordSlot(4, 2, seventh=True),  # V7
                    ChordSlot(5, 2),  # vi
                    ChordSlot(3, 2),  # IV
                ),
            ),
            ChordTemplate(
                name="electrifying_vi_first",
                bars=8,
                chords=(
                    ChordSlot(5, 2),
                    ChordSlot(3, 2, seventh=True),  # IVmaj7
                    ChordSlot(0, 2),
                    ChordSlot(4, 2, seventh=True),  # V7
                ),
            ),
            ChordTemplate(
                name="electrifying_extended_16",
                bars=16,
                chords=(
                    ChordSlot(0, 2),
                    ChordSlot(4, 2, seventh=True),
                    ChordSlot(5, 2),
                    ChordSlot(3, 2),
                    ChordSlot(6, 2, borrowed=True),  # bVII, borrowed major
                    ChordSlot(3, 2),
                    ChordSlot(0, 2),
                    ChordSlot(4, 2, seventh=True),
                ),
            ),
        ),
    ),
    "sleep": MoodProfile(
        name="sleep",
        tempo_range_bpm=(40, 64),
        templates=(
            ChordTemplate(
                name="sleep_lullaby",
                bars=8,
                chords=(
                    ChordSlot(0, 2, seventh=True),  # Imaj7
                    ChordSlot(5, 2, seventh=True),  # vi7
                    ChordSlot(2, 2, seventh=True),  # iii7
                    ChordSlot(3, 2, seventh=True),  # IVmaj7
                ),
            ),
            ChordTemplate(
                name="sleep_drone",
                bars=8,
                chords=(
                    ChordSlot(0, 4, seventh=True),
                    ChordSlot(3, 4, seventh=True),
                ),
            ),
            ChordTemplate(
                name="sleep_extended_16",
                bars=16,
                chords=(
                    ChordSlot(0, 2, seventh=True),
                    ChordSlot(5, 2, seventh=True),
                    ChordSlot(2, 2, seventh=True),
                    ChordSlot(3, 2, seventh=True, borrowed=True),
                    ChordSlot(0, 2, seventh=True),
                    ChordSlot(5, 2, seventh=True),
                    ChordSlot(2, 2, seventh=True),
                    ChordSlot(3, 2, seventh=True),
                ),
            ),
        ),
    ),
}


def get_mood_profile(mood: str) -> MoodProfile:
    """Return the mood's profile, or a KeyError that names the known moods."""
    try:
        return MOOD_PROFILES[mood]
    except KeyError:
        known = ", ".join(sorted(MOOD_PROFILES))
        raise KeyError(f"unknown mood {mood!r}; known moods: {known}") from None


# Tempo ranges per mood, derived from the registry. Phase 1 fine-tunes
# within these bounds to hit the duration target per §10 #1.
TEMPO_RANGE_BPM: Mapping[str, tuple[int, int]] = {
    name: profile.tempo_range_bpm for name, profile in MOOD_PROFILES.items()
}

# Phrase sizes. Phase 1 offers three fixed forms (8, 16, 32 bars).
PHRASE_SIZES: tuple[int, ...] = (8, 16, 32)


def get_template_for_form(
    mood: str,
    form_bars: int,
    *,
    variant_index: int = 0,
) -> ChordTemplate:
    """Return the chord template for `mood` and `form_bars`.

    If no template of the requested length exists, extends the
    shortest available template by repeating it. `variant_index`
    selects among the templates available for the mood (modulo
    available count).
    """
    templates = get_mood_profile(mood).templates
    template = templates[variant_index % len(templates)]
    if template.bars == form_bars:
        return template
    if template.bars > form_bars:
        # Truncate the template to the requested length by keeping
        # the first `form_bars` bars.
        return _truncate_template(template, form_bars)
    # Extend by repeating the template.
    return _extend_template(template, form_bars)


def _truncate_template(template: ChordTemplate, target_bars: int) -> ChordTemplate:
    """Keep the first `target_bars` bars; partial final chord is allowed."""
    kept: list[ChordSlot] = []
    consumed = 0
    for slot in template.chords:
        if consumed + slot.bars > target_bars:
            kept.append(slot._replace(bars=target_bars - consumed))
            break
        kept.append(slot)
        consumed += slot.bars
    return ChordTemplate(name=f"{template.name}_truncated", bars=target_bars, chords=tuple(kept))


def _extend_template(template: ChordTemplate, target_bars: int) -> ChordTemplate:
    """Repeat the template until `target_bars` is reached; truncate the final repeat."""
    if target_bars <= template.bars:
        return _truncate_template(template, target_bars)
    full_repeats = target_bars // template.bars
    remainder = target_bars % template.bars
    chords = list(template.chords) * full_repeats
    if remainder:
        chords.extend(_truncate_template(template, remainder).chords)
    return ChordTemplate(
        name=f"{template.name}_x{target_bars}", bars=target_bars, chords=tuple(chords)
    )


# Cadential close per mood: electrifying resolves with an authentic
# dominant cadence (V -> I); calming and sleep with a plagal one
# (IV -> I). The final section's last two bars are rewritten to this
# cadence, so every piece ends at home instead of on whatever chord the
# template's tail happens to land on.
_CADENCE_DEGREE: Mapping[str, int] = {
    "calming": 3,  # IV
    "electrifying": 4,  # V
    "sleep": 3,  # IV
}


def apply_final_cadence(template: ChordTemplate, mood: str) -> ChordTemplate:
    """Rewrite a template's last two bars as its mood's cadence.

    The bar before the final tonic becomes the dominant (V) for
    electrifying — as a V7 — or the subdominant (IV) for calming/sleep,
    giving the melody a harmonic target to resolve onto. Both cadence
    chords are pinned to root-position bass (`bass_degree`) so the
    final close lands on the tonic's root, not on a walking inversion.
    Bar count is preserved: the cadence bars replace the template's
    last two bars, so duration arithmetic is unaffected. Templates
    shorter than three bars are returned unchanged (there is no room
    for a 2-bar close).
    """
    cadence_degree = _CADENCE_DEGREE.get(mood, 4)
    if template.bars < 3:
        return template
    kept = _truncate_template(template, template.bars - 2).chords
    cadence = (
        ChordSlot(cadence_degree, 1, seventh=(mood == "electrifying"), bass_degree=cadence_degree),
        ChordSlot(0, 1, bass_degree=0),
    )
    return ChordTemplate(
        name=f"{template.name}_cad",
        bars=template.bars,
        chords=(*kept, *cadence),
    )


# Root-relative chord-tone intervals per (mode, degree). Degree is
# 0-based (0 = I). Triads are 3-tone; sevenths are 4-tone diatonic
# sevenths (Imaj7, ii7, iii7, IVmaj7, V7, vi7, vii m7b5 in major;
# i7, ii m7b5, IIImaj7, iv7, v7, VImaj7, VII7 in minor).
_TRIAD_TABLES: Mapping[str, tuple[tuple[int, ...], ...]] = {
    "major": (
        (0, 4, 7),
        (0, 3, 7),
        (0, 3, 7),
        (0, 4, 7),
        (0, 4, 7),
        (0, 3, 7),
        (0, 3, 6),
    ),
    "minor": (
        (0, 3, 7),
        (0, 3, 6),
        (0, 4, 7),
        (0, 3, 7),
        (0, 3, 7),
        (0, 4, 7),
        (0, 4, 7),
    ),
}

_SEVENTH_TABLES: Mapping[str, tuple[tuple[int, ...], ...]] = {
    "major": (
        (0, 4, 7, 11),  # Imaj7
        (0, 3, 7, 10),  # ii7
        (0, 3, 7, 10),  # iii7
        (0, 4, 7, 11),  # IVmaj7
        (0, 4, 7, 10),  # V7
        (0, 3, 7, 10),  # vi7
        (0, 3, 6, 10),  # vii m7b5
    ),
    "minor": (
        (0, 3, 7, 10),  # i7
        (0, 3, 6, 10),  # ii m7b5
        (0, 4, 7, 11),  # IIImaj7
        (0, 3, 7, 10),  # iv7
        (0, 3, 7, 10),  # v7
        (0, 4, 7, 11),  # VImaj7
        (0, 4, 7, 10),  # VII7
    ),
}


def chord_intervals(
    degree: int,
    key: KeySignature,
    *,
    seventh: bool = False,
    borrowed: bool = False,
) -> tuple[int, ...]:
    """Return the root-relative chord-tone intervals for a diatonic chord.

    Major key degrees: I, ii, iii, IV, V, vi, vii -> major, minor,
    minor, major, major, minor, dim triads (maj7/min7/dom7/m7b5 as
    sevenths). Minor key: i, ii°, III, iv, v, VI, VII. Each entry is an
    interval above the chord root — callers add these to the chord
    root, not to the scale-degree root, to get absolute pitches.

    `borrowed` renders the chord from the opposite mode's table: in a
    major key that is the parallel-minor colour (degree 6 becomes a
    bVII major triad instead of vii°); in a minor key it is the raised
    dominant colour (degree 4 becomes a major V instead of v).
    """
    mode = key.mode
    if borrowed:
        mode = "minor" if mode == "major" else "major"
    table = _SEVENTH_TABLES[mode] if seventh else _TRIAD_TABLES[mode]
    return table[degree % 7]


# Root-to-semitone mapping for major/minor keys. The key name in
# CompositionSpec.WesternKey uses a compact form ("C", "G", "Am",
# "F#m"); we normalize to a (root-name, mode) tuple.
_KEY_ROOTS: Mapping[str, int] = {
    "C": 0,
    "G": 7,
    "D": 2,
    "A": 9,
    "E": 4,
    "B": 11,
    "F#": 6,
    "C#": 1,
    "F": 5,
    "Bb": 10,
    "Eb": 3,
    "Ab": 8,
    "Db": 1,
    "G#": 8,
    "Gb": 6,  # enharmonic with F#
}


def key_signature_from_spec_key(spec_key: WesternKey | None) -> KeySignature:
    """Resolve a CompositionSpec WesternKey to a KeySignature.

    If the spec key is None, return C major (the Phase 1 default).
    """
    if spec_key is None:
        return KeySignature(root="C", mode="major")
    value = spec_key.value
    if value.endswith("m"):
        root = value[:-1]
        mode = "minor"
    else:
        root = value
        mode = "major"
    if root not in _KEY_ROOTS:
        raise ValueError(f"Unknown key root: {root!r}")
    mode_lit = mode if mode in ("major", "minor") else "major"
    return KeySignature(root=root, mode=mode_lit)  # type: ignore[arg-type]


def key_signature_from_spec(spec) -> KeySignature:  # type: ignore[no-untyped-def]
    """Resolve a CompositionSpec's key to a KeySignature.

    Convenience wrapper for `key_signature_from_spec_key` that takes
    the whole spec. Imported only in the engine; the type is left
    loose so this module doesn't need to import CompositionSpec.
    """
    return key_signature_from_spec_key(spec.key)


def key_root_midi(key: KeySignature) -> int:
    """Return the MIDI note number of the key's tonic (one octave above middle C)."""
    base = _KEY_ROOTS.get(key.root)
    if base is None:
        raise ValueError(f"Unknown key root: {key.root!r}")
    return 60 + base  # middle C is 60


# A step is a semitone or a whole tone. This is the shared definition of
# conjunct motion, and it lives here rather than in either consumer: the
# linter's passing-tone licence and the quality scorecard's `step_ratio`
# both measure against it. If they disagreed on what a step is, a score
# could satisfy the licence and still be scored as having no stepwise
# motion, and the mismatch would read as a composition bug.
STEP_MAX_SEMITONES: int = 2

# Diatonic scale degrees as semitone offsets from the tonic, indexed by
# `degree % 7`. One octave only — the octave is the caller's business
# (`degree_to_midi` carries it, `scale_pitch_offset` deliberately does
# not).
_SCALE_TABLES: Mapping[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
}


def scale_semitones(mode: str) -> tuple[int, ...]:
    """The mode's scale as semitone offsets from the tonic, one octave."""
    return _SCALE_TABLES[mode]


def scale_pitch_offset(degree: int, mode: str) -> int:
    """A scale degree's semitone offset from the tonic, within one octave.

    Any integer degree is accepted and wraps into the octave: degree 7 is
    the tonic again, not the tonic an octave up. Callers that need the
    octave should use `degree_to_midi`, which carries it explicitly.
    """
    return _SCALE_TABLES[mode][degree % 7]


def degree_to_midi(degree: int, tonic_midi: int, mode: str) -> int:
    """A scale degree as an absolute MIDI pitch, octave included.

    Floor division means negative degrees descend correctly: degree -1 is
    the leading tone *below* the tonic, so a melody can walk under its
    starting note without the modulo flipping it up an octave.
    """
    return tonic_midi + 12 * (degree // 7) + _SCALE_TABLES[mode][degree % 7]


def key_scale_pcs(key: KeySignature) -> frozenset[int]:
    """The key's diatonic pitch classes, for a diatonic-membership test."""
    tonic_pc = key_root_midi(key) % 12
    return frozenset((tonic_pc + offset) % 12 for offset in _SCALE_TABLES[key.mode])


__all__ = [
    "MOOD_PROFILES",
    "PHRASE_BARS",
    "PHRASE_SIZES",
    "STEP_MAX_SEMITONES",
    "TEMPO_RANGE_BPM",
    "ChordSlot",
    "ChordTemplate",
    "MoodProfile",
    "apply_final_cadence",
    "degree_to_midi",
    "get_mood_profile",
    "get_template_for_form",
    "key_root_midi",
    "key_scale_pcs",
    "key_signature_from_spec",
    "key_signature_from_spec_key",
    "scale_pitch_offset",
    "scale_semitones",
]
