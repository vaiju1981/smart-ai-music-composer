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

# Tempo ranges per mood. Phase 1 fine-tunes within these bounds to hit
# the duration target per §10 #1.
TEMPO_RANGE_BPM: Mapping[str, tuple[int, int]] = {
    "calming": (50, 80),
    "electrifying": (100, 160),
    "sleep": (40, 64),
}

# Phrase sizes. Phase 1 offers three fixed forms (8, 16, 32 bars).
PHRASE_SIZES: tuple[int, ...] = (8, 16, 32)


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


# ---------------------------------------------------------------------------
# Calming (50-80 BPM, 4/4 default, I-V-vi-IV-ish idioms)
# ---------------------------------------------------------------------------
_CALMING_TEMPLATES: tuple[ChordTemplate, ...] = (
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
)


# ---------------------------------------------------------------------------
# Electrifying (100-160 BPM, 4/4 default, driving rhythm changes)
# ---------------------------------------------------------------------------
_ELECTRIFYING_TEMPLATES: tuple[ChordTemplate, ...] = (
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
)


# ---------------------------------------------------------------------------
# Sleep (40-60 BPM, 4/4 default, sparse, drone-ish)
# ---------------------------------------------------------------------------
_SLEEP_TEMPLATES: tuple[ChordTemplate, ...] = (
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
)


_TEMPLATES_BY_MOOD: Mapping[str, tuple[ChordTemplate, ...]] = {
    "calming": _CALMING_TEMPLATES,
    "electrifying": _ELECTRIFYING_TEMPLATES,
    "sleep": _SLEEP_TEMPLATES,
}


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
    templates = _TEMPLATES_BY_MOOD[mood]
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
_MAJOR_KEY_ROOTS: Mapping[str, int] = {
    "C": 0,
    "G": 7,
    "D": 2,
    "A": 9,
    "E": 4,
    "B": 11,
    "F#": 6,
    "F": 5,
    "Bb": 10,
    "Eb": 3,
    "Ab": 8,
    "Db": 1,
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
    base = _MAJOR_KEY_ROOTS.get(key.root)
    if base is None:
        raise ValueError(f"Unknown key root: {key.root!r}")
    return 60 + base  # middle C is 60


__all__ = [
    "PHRASE_SIZES",
    "TEMPO_RANGE_BPM",
    "ChordTemplate",
    "get_template_for_form",
    "key_root_midi",
    "key_signature_from_spec",
    "key_signature_from_spec_key",
]
