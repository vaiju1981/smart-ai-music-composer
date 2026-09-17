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

from collections.abc import Mapping, Sequence
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
CADENCE_DEGREE: Mapping[str, int] = {
    "calming": 3,  # IV
    "electrifying": 4,  # V
    "sleep": 3,  # IV
}

DEFAULT_CADENCE_DEGREE: int = 4
"""The cadence an unlisted mood gets: a dominant, the stronger close."""

CADENCE_SEVENTH: Mapping[str, bool] = {
    "electrifying": True,
}
"""Whether the cadence chord is a seventh.

Only electrifying's dominant is a V7; a seventh under the plagal close
would be heard as an added sixth over the tonic that follows it. The
other moods take `DEFAULT_CADENCE_SEVENTH`, which is the triad.
"""

DEFAULT_CADENCE_SEVENTH: bool = False
"""The cadence an unlisted mood gets: the triad, which is the plain close."""

MODULATION_OFFSET: int = 2
"""Semitones a long piece's final repetition is lifted by.

A whole step, and the lift *is* the ending: the piece arrives in the new
key and stays there through the coda. It lives here, with `transposed_key`
and the rest of the key arithmetic, rather than in `engine.py` where it was
declared — a plan has to be able to read it, and `engine.py` imports
`plan.py`, so the dependency cannot run the other way.
"""


def cadence_degree_for(mood: str) -> int:
    """The scale degree this mood's final cadence approaches the tonic from."""
    return CADENCE_DEGREE.get(mood, DEFAULT_CADENCE_DEGREE)


def cadence_seventh_for(mood: str) -> bool:
    """Whether this mood's final cadence chord is a seventh."""
    return CADENCE_SEVENTH.get(mood, DEFAULT_CADENCE_SEVENTH)


HALF_CADENCE_APPROACH_DEGREE: int = 1
"""The predominant a half cadence leaves from: the supertonic.

ii in major, ii° in minor — the degree whose function is to lead to the
dominant, which is what makes the close a cadence rather than two chords.
"""

HALF_CADENCE_TARGET_DEGREE: int = 4
"""The degree a half cadence stops on: the dominant, unresolved."""


SECTION_CLOSES: tuple[str, ...] = ("hold", "half", "full")
"""How a section that is not the piece's last one closes.

A closed vocabulary rather than a free string, because the choice reaches
the notes through `apply_section_close` and a name outside it would be a
section that silently closed some third way. `hold` is what the engine did
before the choice existed — the template's last chord sounds to the end of
the form — and it is kept as a legal value rather than dropped: it is the
shape a piece has when every section runs on, and a plan that asks for it
should get it rather than be told the request is unknown.
"""

DEFAULT_SECTION_CLOSE: str = "half"
"""The close an interior section gets: a half cadence.

Measured before it was chosen. Over 108 pieces the engine wrote 384 interior
sections and **not one of them closed**: every one ended with the last chord
of the template's tail sounding into the final bar, 56.3% of them on a
predominant and 43.8% on a dominant, and 18.8% of the seams landed on a
motion (IV -> vi) that resolves nothing. So a piece had exactly one cadence,
at its end, and the sections before it stopped rather than closed.

A half cadence is the classical answer and the one the melody pass already
anticipated — `_generate_section`'s breath rule lifts onto "the dominant's
root when the chord there is the V (a half cadence)" — because it points
home without arriving: the V of the close resolves into the I that the next
section opens on. It also moves the harmony where the phrase ends, which is
the *only* place a rhythm of uniform chords can move without inventing
chords, so it is what makes "varied harmonic rhythm" true as well.
"""


def apply_section_close(
    template: ChordTemplate,
    *,
    close: str,
    cadence_degree: int,
    seventh: bool,
) -> ChordTemplate:
    """Rewrite a template's last two bars as the close the plan names.

    The three values are the three ways a section can end: `hold` returns
    the template untouched (the tail sounds into the final bar), `half`
    writes the half cadence below, and `full` writes the plan's own
    cadence through `apply_final_cadence`. One function rather than two
    call sites, so the policy is the thing an engine read names and the
    thing a critic's hint can point at.

    `cadence_degree` and `seventh` are read only by the `full` arm: a half
    cadence's two chords are what a half cadence *is* rather than a taste a
    plan holds, so they are the module's constants and not fields.
    """
    if close == "hold":
        return template
    if close == "half":
        return apply_half_cadence(template)
    if close == "full":
        return apply_final_cadence(template, cadence_degree=cadence_degree, seventh=seventh)
    # Unreachable through a plan, whose validator refuses a name the
    # vocabulary does not hold; a direct call can still make it, and a
    # template silently left alone is not a named refusal.
    raise ValueError(f"unknown section close {close!r}; expected one of {SECTION_CLOSES}")


def apply_half_cadence(template: ChordTemplate) -> ChordTemplate:
    """Rewrite a template's last two bars as a ii-V half cadence.

    The classical phrase ending: a predominant for one bar, then the
    dominant for one, both in root position so the bass states the two
    chords where they change. It approaches the dominant rather than the
    tonic — the close points home and does not arrive, which is what makes
    it a *half* cadence, and what lets the resolution happen across the
    section boundary where the next section's tonic opens.

    The two degrees are the definition of the cadence and live beside it
    rather than in the plan, the way `apply_final_cadence` takes its pair
    as arguments while `CADENCE_DEGREE` maps a mood to one. Bar count is
    preserved and templates shorter than three bars are returned unchanged,
    both for `apply_final_cadence`'s reasons.
    """
    if template.bars < 3:
        return template
    kept = _truncate_template(template, template.bars - 2).chords
    cadence = (
        ChordSlot(HALF_CADENCE_APPROACH_DEGREE, 1, bass_degree=HALF_CADENCE_APPROACH_DEGREE),
        ChordSlot(HALF_CADENCE_TARGET_DEGREE, 1, bass_degree=HALF_CADENCE_TARGET_DEGREE),
    )
    return ChordTemplate(
        name=f"{template.name}_half",
        bars=template.bars,
        chords=(*kept, *cadence),
    )


def apply_final_cadence(
    template: ChordTemplate, *, cadence_degree: int, seventh: bool
) -> ChordTemplate:
    """Rewrite a template's last two bars as the cadence it is given.

    The bar before the final tonic becomes the degree named — the dominant
    (V) for electrifying, the subdominant (IV) for calming/sleep — giving
    the melody a harmonic target to resolve onto, as a seventh chord when
    `seventh`. Both cadence chords are pinned to root-position bass
    (`bass_degree`) so the final close lands on the tonic's root, not on a
    walking inversion. Bar count is preserved: the cadence bars replace the
    template's last two bars, so duration arithmetic is unaffected.
    Templates shorter than three bars are returned unchanged (there is no
    room for a 2-bar close).

    The degree and the seventh arrive as arguments rather than as a mood
    name so the whole cadence is a value a plan carries; `CADENCE_DEGREE`
    and `cadence_degree_for` remain how the *default* plan finds this
    mood's pair.
    """
    if template.bars < 3:
        return template
    kept = _truncate_template(template, template.bars - 2).chords
    cadence = (
        ChordSlot(cadence_degree, 1, seventh=seventh, bass_degree=cadence_degree),
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


def chord_root_offset(degree: int, key: KeySignature, *, borrowed: bool = False) -> int:
    """A chord's root, in semitones above the key's tonic.

    The companion to `chord_intervals`, which gives the tones *above* a
    root this one locates. A borrowed chord is drawn from the parallel
    mode's table, so its root is that mode's degree and not this key's:
    the two modes disagree on exactly the three degrees the minor scale
    flattens (2, 5 and 6), a semitone below in minor and therefore a
    semitone above in major. Reading the root off the borrowed table is
    the same swap `chord_intervals` makes, which is what makes the
    invariant `bar_scale_intervals` documents an identity rather than a
    coincidence — a bar's chord *is* the tones its own scale spells at
    degrees 0, 2 and 4 (and 6 for a seventh).

    Both directions are load-bearing. Written as an adjustment to the
    key's own degree and applied only when the key is major, a borrowed
    root in a minor key sat a semitone below the root of the scale the
    line over it walks, so the melody's degree-0 tone was not the chord's
    root and the triad was spelled off the bar's own scale. Reachable
    through any explicitly named minor key — the extended electrifying
    template's borrowed degree-6 slot, which the golden corpus's own A
    minor cells already sound — and widened by the mood key pools, two of
    whose six electrifying entries are minor.
    """
    mode = key.mode
    if borrowed:
        mode = "minor" if mode == "major" else "major"
    return scale_pitch_offset(degree, mode)


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


# The keys a mood may be written in when the spec names none, in pool order,
# and the pool an unlisted mood gets. **A musical judgement, stated as one.**
# Nothing measures these: every one of the 25 keys `WesternKey` admits
# composes clean in every mood, in every ensemble, at every duration (a
# 225-cell sweep), so no key is the engine's to refuse and the choice is
# taste rather than a finding. It is pinned as a literal, with a ratchet on
# the table's shape, the way the arbiter's metric order is. Each pool varies
# on the two axes a key varies on — the root and the mode — which is the
# pair the golden corpus holds fixed.
#
# **The first entry is the key the mood falls back to**, because the engine
# picks `pool[seed % len(pool)]` and a spec naming no seed resolves to seed
# 0 the way the engine's other seed reads do. So the order is a decision,
# not a formatting accident.
MOOD_KEY_POOLS: Mapping[str, tuple[str, ...]] = {
    # Warm and open, on the plain side of the signature: the majors a slow
    # I-vi-IV-V idiom sits in without effort, with the two minor keys that
    # colour a calm piece rather than darken it.
    "calming": ("C", "F", "G", "Bb", "Am", "Dm"),
    # The flat side, for the sparse, drone-adjacent end of the vocabulary.
    "sleep": ("F", "Bb", "Eb", "Ab", "Dm", "Gm"),
    # Sharp side, majors and minors alike: the keys a driving tempo and a
    # guitar-shaped idiom live in.
    "electrifying": ("A", "E", "D", "G", "Am", "Em"),
}

DEFAULT_KEY_POOL: tuple[str, ...] = ("C", "G", "F", "D", "Em", "Am")
"""The pool an unlisted mood gets: the commonest keys, majors leading."""


def key_pool_for(mood: str) -> tuple[str, ...]:
    """The keys this mood may be written in, when the spec names none."""
    return MOOD_KEY_POOLS.get(mood, DEFAULT_KEY_POOL)


def key_signature_from_spec_key(spec_key: WesternKey) -> KeySignature:
    """Resolve the key a spec names to a KeySignature.

    The parameter is not optional, and that is the point of it: an unset
    key is not this function's to answer. It used to return C major for
    `None` — which is exactly the constant the engine's "engine chooses"
    used to be — and `chosen_key` below is the path that makes that choice
    now. Narrowing the type rather than keeping a `None` branch leaves no
    call site that can quietly reinstate it.
    """
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


def chosen_key(
    spec_key: WesternKey | None, *, pool: Sequence[str], seed: int | None
) -> KeySignature:
    """The key a piece is written in: the spec's, or the seed's pick.

    A spec that names a key gets it, and the pool is dead for that piece. A
    spec that does not gets `pool[seed % len(pool)]` — so varying the seed
    *walks* the pool rather than rolling against it, consecutive seeds of a
    fan-out land on different keys, and a caller that names no seed (which
    resolves to seed 0, as the engine's other seed reads do) gets the pool's
    first entry, which each pool declares as the mood's fallback.

    The pool arrives as an argument rather than being read from
    `MOOD_KEY_POOLS` here, because the plan is materialized: a stored piece
    has to replay against the pool it was written under and not against this
    build's table. `key_pool_for` is how the *default* plan finds one.
    """
    if spec_key is not None:
        return key_signature_from_spec_key(spec_key)
    if not pool:
        # The plan refuses an empty pool, so an engine call cannot reach
        # this; a direct call can, and a bare ZeroDivisionError is not a
        # named refusal. Same rule, and the same shape, as an empty bass
        # vocabulary.
        raise ValueError("a key pool must name at least one key")
    return key_signature_from_spec_key(WesternKey(pool[(seed or 0) % len(pool)]))


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

# A leap is a fourth or wider — the interval at which a listener hears a
# gap that wants closing, and so the interval a melody must answer with a
# step. Shared for the same reason as `STEP_MAX_SEMITONES`: the
# generator's recovery pass and the scorecard's `leap_recovery_ratio`
# have to agree on which intervals are leaps. Thirds sit between the two
# definitions and are neither.
LEAP_MIN_SEMITONES: int = 5

# Diatonic scale degrees as semitone offsets from the tonic, indexed by
# `degree % 7`. One octave only — the octave is the caller's business
# (`scale_walk` carries it, `scale_pitch_offset` deliberately does not).
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
    octave should use `scale_walk`, which carries it explicitly.
    """
    return _SCALE_TABLES[mode][degree % 7]


def scale_intervals(degree: int, mode: str) -> tuple[int, ...]:
    """The mode's scale spelled from `degree` as its root, one octave.

    Rotation only — the seven intervals cover the same pitch classes as
    the mode rooted on the tonic, in the order a line walking from
    `degree` would meet them. Degree 4 of major gives the mixolydian
    rotation (0, 2, 4, 5, 7, 9, 10), which is what a bar on the V sounds.
    """
    table = _SCALE_TABLES[mode]
    root = table[degree % 7]
    return tuple(
        table[(degree + step) % 7] - root + (12 if degree + step >= 7 else 0)
        for step in range(7)
    )


def scale_walk(degree: int, root_midi: int, intervals: tuple[int, ...]) -> int:
    """A scale degree above `root_midi` as an absolute MIDI pitch.

    `intervals` is one octave of the scale, spelled from `root_midi` (see
    `scale_intervals`). Floor division means negative degrees descend
    correctly: degree -1 is the scale tone *below* the root, so a melody
    can walk under its starting note without the modulo flipping it up an
    octave.
    """
    return root_midi + 12 * (degree // 7) + intervals[degree % 7]


def bar_scale_intervals(
    degree: int,
    key: KeySignature,
    *,
    borrowed: bool = False,
) -> tuple[int, ...]:
    """The scale a bar's chord is built from, spelled from the chord root.

    A diatonic bar walks the key's own mode; a borrowed bar walks the
    parallel mode, which is the same table `chord_intervals` takes its
    borrowed chord from. The two must agree, or a stepwise line would
    leave the scale its own chord belongs to: rotating a table into a
    scale always yields a chord whose tones are its degrees 0, 2, 4 (and
    6 for a seventh), which is what makes a chord-tone index a scale
    degree of exactly `2 * index`.
    """
    mode = key.mode
    if borrowed:
        mode = "minor" if mode == "major" else "major"
    return scale_intervals(degree, mode)


def key_scale_pcs(key: KeySignature) -> frozenset[int]:
    """The key's diatonic pitch classes, for a diatonic-membership test."""
    tonic_pc = key_root_midi(key) % 12
    return frozenset((tonic_pc + offset) % 12 for offset in _SCALE_TABLES[key.mode])


# `_KEY_ROOTS` read backwards, for naming the key a modulation lands on.
# Three pitch classes have two spellings (C#/Db, F#/Gb, Ab/G#); the table
# already prefers the sharp for the first two and `Ab` for the third, and
# the inverse keeps those choices, so a lift changes no spelling by itself.
_KEY_ROOT_NAMES: Mapping[int, str] = {
    0: "C",
    1: "C#",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}


def transposed_key(key: KeySignature, semitones: int) -> KeySignature:
    """The key `semitones` above `key`, mode unchanged.

    The modulation lift takes a long piece's final repetition a whole step
    up, and every bar the lift carries belongs to the *new* key. Which key
    that is cannot be recovered from the bar's chord — a lifted IV is the
    home key's V, spelled identically — so the engine publishes it per bar
    and the linter's passing-tone licence reads the bar it was written
    against rather than guessing and refusing the new key's own notes.
    """
    if semitones == 0:
        return key
    root = (_KEY_ROOTS[key.root] + semitones) % 12
    return KeySignature(root=_KEY_ROOT_NAMES[root], mode=key.mode)


# Every major and natural-minor scale, for recovering which scale a bar's
# harmony belongs to when the key alone does not say (see
# `bar_diatonic_pcs`). Two per root: the mode is part of the reading.
_CANDIDATE_SCALES: tuple[frozenset[int], ...] = tuple(
    frozenset((root + offset) % 12 for offset in table)
    for table in _SCALE_TABLES.values()
    for root in range(12)
)


def chord_tone_degrees(tone_count: int) -> tuple[int, ...]:
    """The scale degrees a chord of `tone_count` tones occupies.

    A bar's chord is spelled from the same scale the melody walks
    (`bar_scale_intervals`), which is a rotation of the mode's table
    rooted on the chord root — and the chord tables take their tones from
    that same rotation. So a triad's tones are degrees 0, 2 and 4 of the
    bar's scale and a seventh chord's are 0, 2, 4 and 6, whatever the
    mode, the degree or whether the chord is borrowed. Callers use this
    to tell a chord tone from a tone that needs the passing-tone licence,
    which is a question about scale degrees and not about pitch classes.
    """
    return tuple(range(0, 2 * tone_count, 2))


def bar_diatonic_pcs(chord_pcs: tuple[int, ...], key: KeySignature) -> frozenset[int]:
    """The pitch classes a passing tone may use in a bar sounding `chord_pcs`.

    The key's own scale is the wrong answer for the two cases where a
    bar's harmony leaves the key: the modulation lift on a long piece's
    final repetition, and the borrowed chords. Both are legible from the
    chord, so the scale is recovered from it.

    Every major and natural-minor scale containing the whole chord is a
    reading of the bar. When one of them is the key's own scale, that is
    the answer, and a diatonic chord therefore reads exactly — a C major
    triad in C major admits the key's seven tones and no others, which is
    what keeps the licence strict where it matters. A foreign chord has
    no single parent scale (a D major triad fits D, G and A major alike),
    so the readings are unioned: the honest statement is "a scale this
    bar's harmony belongs to", and the engine's own walk in such a bar is
    always one of the readings, so nothing it plays is refused.

    Computed once per bar by the caller, not once per note.
    """
    pcs = frozenset(chord_pcs)
    readings = [scale for scale in _CANDIDATE_SCALES if pcs <= scale]
    home = key_scale_pcs(key)
    if home in readings or not readings:
        return home
    return frozenset().union(*readings)


__all__ = [
    "CADENCE_DEGREE",
    "CADENCE_SEVENTH",
    "DEFAULT_CADENCE_DEGREE",
    "DEFAULT_CADENCE_SEVENTH",
    "DEFAULT_KEY_POOL",
    "DEFAULT_SECTION_CLOSE",
    "HALF_CADENCE_APPROACH_DEGREE",
    "HALF_CADENCE_TARGET_DEGREE",
    "LEAP_MIN_SEMITONES",
    "MODULATION_OFFSET",
    "MOOD_KEY_POOLS",
    "MOOD_PROFILES",
    "PHRASE_BARS",
    "PHRASE_SIZES",
    "SECTION_CLOSES",
    "STEP_MAX_SEMITONES",
    "TEMPO_RANGE_BPM",
    "ChordSlot",
    "ChordTemplate",
    "MoodProfile",
    "apply_final_cadence",
    "apply_half_cadence",
    "apply_section_close",
    "bar_diatonic_pcs",
    "bar_scale_intervals",
    "cadence_degree_for",
    "cadence_seventh_for",
    "chord_intervals",
    "chord_root_offset",
    "chord_tone_degrees",
    "chosen_key",
    "get_mood_profile",
    "get_template_for_form",
    "key_pool_for",
    "key_root_midi",
    "key_scale_pcs",
    "key_signature_from_spec_key",
    "scale_intervals",
    "scale_pitch_offset",
    "scale_semitones",
    "scale_walk",
    "transposed_key",
]
