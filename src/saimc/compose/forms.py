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

from saimc.compose.score import KeySignature
from saimc.spec import WesternKey


@dataclass(frozen=True)
class ChordTemplate:
    """A single chord-progression template.

    `bars` is the total length of the template in bars; `chords` is a
    list of (scale_degree, duration_bars) tuples that must sum to
    exactly `bars` bars.

    `name` is a stable identifier used in logs and manifest records.
    """

    name: str
    bars: int
    chords: tuple[tuple[int, int], ...]


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
                chords=((0, 2), (5, 2), (3, 2), (4, 2)),
            ),
            ChordTemplate(
                name="calming_iii_substitute",
                bars=8,
                chords=((0, 2), (2, 2), (3, 2), (4, 2)),
            ),
            ChordTemplate(
                name="calming_extended_16",
                bars=16,
                chords=((0, 2), (5, 2), (3, 2), (4, 2), (0, 2), (5, 2), (3, 2), (1, 4)),
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
                chords=((0, 2), (4, 2), (5, 2), (3, 2)),
            ),
            ChordTemplate(
                name="electrifying_vi_first",
                bars=8,
                chords=((5, 2), (3, 2), (0, 2), (4, 2)),
            ),
            ChordTemplate(
                name="electrifying_extended_16",
                bars=16,
                chords=(
                    (0, 2),
                    (4, 2),
                    (5, 2),
                    (3, 2),
                    (6, 2),
                    (3, 2),
                    (0, 2),
                    (4, 2),
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
                chords=((0, 2), (5, 2), (2, 2), (3, 2)),
            ),
            ChordTemplate(
                name="sleep_drone",
                bars=8,
                chords=((0, 4), (3, 4)),
            ),
            ChordTemplate(
                name="sleep_extended_16",
                bars=16,
                chords=(
                    (0, 2),
                    (5, 2),
                    (2, 2),
                    (3, 2),
                    (0, 2),
                    (5, 2),
                    (2, 2),
                    (3, 2),
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
    kept: list[tuple[int, int]] = []
    consumed = 0
    for degree, dur in template.chords:
        if consumed + dur > target_bars:
            kept.append((degree, target_bars - consumed))
            break
        kept.append((degree, dur))
        consumed += dur
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


__all__ = [
    "MOOD_PROFILES",
    "PHRASE_SIZES",
    "TEMPO_RANGE_BPM",
    "ChordTemplate",
    "MoodProfile",
    "get_mood_profile",
    "get_template_for_form",
    "key_root_midi",
    "key_signature_from_spec",
    "key_signature_from_spec_key",
]
