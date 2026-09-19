"""The performance plan: notation turned into integer-microsecond events.

The last stage, and the only one that adds anything not in the notation —
humanized timing, legato overlap, pedal, ghost notes. All of it is seeded, so
`(score, seed) -> plan` is as reproducible as the notes it reads.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace
from itertools import pairwise
from typing import Final

from saimc.compose.duration import (
    DEFAULT_SECTION_ARC,
    DurationArrangement,
    SectionArc,
)
from saimc.compose.dynamics import phrase_swell, section_velocity_scale, velocity_arc
from saimc.compose.forms import (
    PHRASE_BARS,
)
from saimc.compose.score import (
    VOICE_HARMONY,
    VOICE_MELODY,
    VOICE_PERCUSSION,
    ControllerEvent,
    NotationScore,
    NoteEvent,
    PerformanceNoteEvent,
    PerformancePlan,
    microseconds_at_tick,
)

# Sustained instruments: their notes blur into one another, so the
# performance layer lets each melody note ring slightly into the next
# (legato). Percussive, plucked, and mallet instruments keep their
# notated durations — that attack gap IS their articulation.
SUSTAINED_INSTRUMENTS: frozenset[str] = frozenset(
    {
        "violin",
        "viola",
        "cello",
        "contrabass",
        "fiddle",
        "strings",
        "tremolo_strings",
        "flute",
        "piccolo",
        "recorder",
        "pan_flute",
        "ocarina",
        "oboe",
        "english_horn",
        "bassoon",
        "clarinet",
        "soprano_sax",
        "alto_sax",
        "tenor_sax",
        "baritone_sax",
        "french_horn",
        "brass_section",
        "trumpet",
        "muted_trumpet",
        "trombone",
        "tuba",
        "choir",
        "pipe_organ",
        "accordion",
        "harmonica",
        "sitar",
        "harmonium",
        "bansuri",
        "sarangi",
        "rudra_veena",
        "sarasvati_veena",
        "qanoon",
        "ud",
        "kora",
        "shakuhachi",
        "shanai",
        "bagpipe",
    }
)
# Keyboard instruments that read a sustain pedal; organ voices sustain
# by themselves and gain nothing from CC64.
PEDAL_INSTRUMENTS: frozenset[str] = frozenset({"piano", "harpsichord", "celesta", "music_box"})
# How far a legato note rings past its written end (never past the next
# note's start: a same-pitch retrigger would re-attack the line).
LEGATO_OVERLAP_US: int = 35_000
# Pedal lifts a moment before each bar line so chords do not blur across
# the bar; the lift is a fixed 40 ms before the next downbeat.
PEDAL_RELEASE_LEAD_US: int = 40_000
# Humanization profiles (spec.humanization): timing scatter on the
# realized timestamps and how far velocities spread around their neutral
# 64 centre. 'none' applies neither.
HUMANIZE_TIMING_US: dict[str, int] = {"light": 10_000, "expressive": 25_000}
HUMANIZE_VELOCITY_SPAN: dict[str, float] = {"light": 1.15, "expressive": 1.35}
PERCUSSION_TIMING_US: dict[str, int] = {"light": 5_000, "expressive": 15_000}
# Percussion ghost notes: quiet extra hits that make the kit feel played
# rather than sequenced.
GHOST_NOTE_PROBABILITY: float = 0.08
GHOST_NOTE_VELOCITY_RANGE: tuple[int, int] = (20, 35)
# The realization is drawn from two streams, one per group of voices: the
# melodic group (the tune and the bed under it) draws from one and the kit
# from the other. One stream could not do, and the measurement is the
# reason. The draws are taken in event order, so a voice that writes one
# more note shifts every draw the other voice makes after it:
# `bass_root_motion` shortened this piece's tune by one note and moved the
# kit's realized timing in **all 207** of its hits — its fills with it. A
# coupling between two voices neither of which reads the other is one no
# plan knob can express and no critic can attribute, so each voice's
# realized timing is a function of its own notes and the piece's seed and
# of nothing that happens in the other stream.
#
# Only that arrow was observable — the melodic events are built before the
# kit's, so the kit's length could never shift the tune's draws — but which
# order `events` is built in is an accident of this function rather than a
# property of the design, and a shared stream is a coupling waiting to point
# the other way.
_MELODIC_REALIZATION_SALT: Final[int] = 11
_KIT_REALIZATION_SALT: Final[int] = 12


def _realization_stream(seed: int | None, salt: int) -> random.Random:
    """The stream the voices of one group draw their realization from.

    The salt separates the groups; `None` seeds the piece's own default,
    which is what a caller that names no seed gets.
    """
    return random.Random(((seed or 0) * 2654435761 + salt) % (2**31))


# CC11 (expression) rides the dynamic arch so phrases swell and relax
# even inside a held chord. 96 is near-full expression at the arch peak.
EXPRESSION_BASE: int = 96


def _merged_tie_runs(
    notes: tuple[NoteEvent, ...] | list[NoteEvent],
) -> tuple[set[int], dict[int, int]]:
    """Which tied notes fold into their predecessor, and the merged spans.

    A note with `tie=True` connects to the next same-voice, same-pitch,
    contiguous note — and that continuation may itself be tied onward,
    so a run of tied noteheads collapses into one sounded note. Returned
    as (indices to skip, duration in ticks per surviving index). The
    walk stays on the tick grid, where contiguity is exact.
    """
    by_voice: dict[int, list[int]] = {}
    for idx, note in enumerate(notes):
        by_voice.setdefault(note.voice_id, []).append(idx)

    skip: set[int] = set()
    durations: dict[int, int] = {}
    for voice_indices in by_voice.values():
        k = 0
        while k < len(voice_indices):
            head = voice_indices[k]
            if head in skip:
                k += 1
                continue
            span = notes[head].duration_ticks
            cur = head
            j = k + 1
            while (
                j < len(voice_indices)
                and notes[cur].tie
                and notes[voice_indices[j]].pitch_midi == notes[cur].pitch_midi
                and notes[cur].tick + notes[cur].duration_ticks == notes[voice_indices[j]].tick
            ):
                span += notes[voice_indices[j]].duration_ticks
                skip.add(voice_indices[j])
                cur = voice_indices[j]
                j += 1
            durations[head] = span
            k = j
    return skip, durations


def build_performance_plan(
    score: NotationScore,
    *,
    voice_instruments: Mapping[int, str],
    humanization: str,
    seed: int | None,
    arrangement: DurationArrangement,
    arc: SectionArc = DEFAULT_SECTION_ARC,
    drum_set_legacy: bool = False,
) -> PerformancePlan:
    """Lay the expression layer on the notated surface, per voice.

    `voice_instruments` maps engine voice ids to instrument names; the
    plan's legato, CC11 swells, sustain pedal, and humanization are
    chosen per voice from that instrument's family (sustained vs
    percussive vs pedal-reading). The bass is the engine's own walking
    line and is deliberately left un-humanized. `drum_set_legacy` pins
    the scalar drum-set spec's plan to its pre-ensemble behaviour: the
    melody voice is treated as the kit's lead ("drum_set") — no legato,
    no pedal — so its render is byte-identical.
    """
    # Ties play as one sound: a tied note's continuation never
    # re-attacks — the predecessor rings through it. Runs are resolved
    # on the notation grid first (contiguity in ticks is exact, and a
    # chain of tied noteheads collapses into a single sounded note);
    # humanization then scatters the surviving events freely.
    tie_skips, tie_spans = _merged_tie_runs(score.notes)
    events: list[PerformanceNoteEvent] = []
    for idx, note in enumerate(score.notes):
        if idx in tie_skips:
            continue
        start_us = microseconds_at_tick(note.tick, score.tempo)
        duration_us = (
            microseconds_at_tick(tie_spans.get(idx, note.duration_ticks) + note.tick, score.tempo)
            - start_us
        )
        events.append(
            PerformanceNoteEvent(
                voice_id=note.voice_id,
                pitch_midi=note.pitch_midi,
                start_us=start_us,
                duration_us=duration_us,
                velocity=note.velocity,
                tie=note.tie,
            )
        )

    melody_sorted_idx = sorted(
        (i for i, e in enumerate(events) if e.voice_id == VOICE_MELODY),
        key=lambda i: events[i].start_us,
    )
    has_melody = bool(melody_sorted_idx)
    melody_instrument = (
        "drum_set" if drum_set_legacy else voice_instruments.get(VOICE_MELODY, "piano")
    )
    harmony_instruments = {
        voice: instrument
        for voice, instrument in voice_instruments.items()
        if voice >= VOICE_HARMONY
    }

    # Legato: sustained instruments let each note of a line ring a
    # little past the next attack so the release tail blurs into the
    # next note. A same-pitch neighbour is capped at its start: a late
    # note_off on the same key would re-attack or cut the line.
    legato_extended: set[int] = set()
    legato_voices = [
        voice
        for voice, instrument in (
            (VOICE_MELODY, melody_instrument),
            *harmony_instruments.items(),
        )
        if instrument in SUSTAINED_INSTRUMENTS
    ]
    for voice in legato_voices:
        voice_sorted_idx = sorted(
            (i for i, e in enumerate(events) if e.voice_id == voice),
            key=lambda i: events[i].start_us,
        )
        for a, b in pairwise(voice_sorted_idx):
            prev, curr = events[a], events[b]
            gap = curr.start_us - prev.start_us
            limit = gap if prev.pitch_midi == curr.pitch_midi else gap + LEGATO_OVERLAP_US
            extended = min(prev.duration_us + LEGATO_OVERLAP_US, limit)
            if extended > prev.duration_us:
                legato_extended.add(a)
                events[a] = PerformanceNoteEvent(
                    voice_id=prev.voice_id,
                    pitch_midi=prev.pitch_midi,
                    start_us=prev.start_us,
                    duration_us=extended,
                    velocity=prev.velocity,
                    tie=prev.tie,
                )

    controllers: list[ControllerEvent] = []
    if has_melody:
        # Expression swells ride the same arch as the velocities, one
        # value per bar, so phrases breathe in the controller layer too.
        # On top of the piece-long arch, each 4-bar phrase swells once
        # (rise into its middle, relax at its end) and the section
        # terracing sits the whole step of the arc. The melody always
        # swells; the harmony voice follows when its instrument is
        # sustained (a pad wants to breathe; a plucked arpeggio has no
        # sustain to shape).
        total_ticks = max(1, score.total_ticks())
        ticks_per_bar = max(1, score.measures[0].end_tick - score.measures[0].start_tick)
        phrase_ticks = PHRASE_BARS * ticks_per_bar
        swell_voices = [VOICE_MELODY]
        swell_voices.extend(
            voice
            for voice, instrument in harmony_instruments.items()
            if instrument in SUSTAINED_INSTRUMENTS
        )
        for measure in score.measures:
            position = measure.start_tick / total_ticks
            phrase_position = (measure.start_tick % phrase_ticks) / phrase_ticks
            section_idx = (
                (measure.start_tick // (arrangement.form_bars * ticks_per_bar))
                if (arrangement.form_bars > 0)
                else 0
            )
            value = min(
                127,
                round(
                    EXPRESSION_BASE
                    * velocity_arc(position)
                    * phrase_swell(phrase_position)
                    * section_velocity_scale(section_idx, arrangement.repetition_count, arc)
                ),
            )
            for voice in swell_voices:
                controllers.append(
                    ControllerEvent(
                        voice_id=voice,
                        control=11,
                        value=value,
                        start_us=microseconds_at_tick(measure.start_tick, score.tempo),
                    )
                )
        # Per-bar pedaling: press on the downbeat, lift just before the
        # next one so chords do not wash across the bar line. Applied
        # per voice whose instrument reads a pedal.
        pedal_voices = [
            voice
            for voice, instrument in (
                (VOICE_MELODY, melody_instrument),
                *harmony_instruments.items(),
            )
            if instrument in PEDAL_INSTRUMENTS
        ]
        for voice in pedal_voices:
            for i, measure in enumerate(score.measures):
                press_us = microseconds_at_tick(measure.start_tick, score.tempo)
                controllers.append(
                    ControllerEvent(voice_id=voice, control=64, value=127, start_us=press_us)
                )
                if i + 1 < len(score.measures):
                    next_downbeat = microseconds_at_tick(
                        score.measures[i + 1].start_tick, score.tempo
                    )
                    controllers.append(
                        ControllerEvent(
                            voice_id=voice,
                            control=64,
                            value=0,
                            start_us=max(0, next_downbeat - PEDAL_RELEASE_LEAD_US),
                        )
                    )

    # Humanization touches only this plan — the NotationScore (and the
    # engraved sheet) keeps its grid-perfect timing.
    if humanization != "none":
        rng = _realization_stream(seed, _MELODIC_REALIZATION_SALT)
        kit_rng = _realization_stream(seed, _KIT_REALIZATION_SALT)
        timing_us = HUMANIZE_TIMING_US.get(humanization, HUMANIZE_TIMING_US["light"])
        velocity_span = HUMANIZE_VELOCITY_SPAN.get(humanization, HUMANIZE_VELOCITY_SPAN["light"])
        perc_timing_us = PERCUSSION_TIMING_US.get(humanization, PERCUSSION_TIMING_US["light"])
        humanized: list[PerformanceNoteEvent] = []
        for event in events:
            if event.voice_id == VOICE_MELODY or event.voice_id in harmony_instruments:
                offset = round(rng.uniform(-1.0, 1.0) * timing_us)
                velocity = max(1, min(127, round(64 + (event.velocity - 64) * velocity_span)))
                humanized.append(
                    PerformanceNoteEvent(
                        voice_id=event.voice_id,
                        pitch_midi=event.pitch_midi,
                        start_us=max(0, event.start_us + offset),
                        duration_us=event.duration_us,
                        velocity=velocity,
                        tie=event.tie,
                    )
                )
            elif event.voice_id == VOICE_PERCUSSION:
                offset = round(kit_rng.uniform(-1.0, 1.0) * perc_timing_us)
                humanized.append(
                    PerformanceNoteEvent(
                        voice_id=event.voice_id,
                        pitch_midi=event.pitch_midi,
                        start_us=max(0, event.start_us + offset),
                        duration_us=event.duration_us,
                        velocity=event.velocity,
                        tie=event.tie,
                    )
                )
            else:
                humanized.append(event)
        events = humanized

        # Timing scatter can push one note past its neighbour's start on
        # back-to-back melody lines; trim the release so the line never
        # overlaps itself. Percussion hits are left alone — a few
        # milliseconds of overlap between drum voices is inaudible.
        melody_idx = sorted(
            (i for i, e in enumerate(events) if e.voice_id == VOICE_MELODY),
            key=lambda i: events[i].start_us,
        )
        for a, b in pairwise(melody_idx):
            prev, curr = events[a], events[b]
            gap = curr.start_us - prev.start_us
            if a not in legato_extended and 0 < gap < prev.duration_us:
                events[a] = replace(prev, duration_us=max(1, gap))

        if humanization == "expressive":
            # Repeated melody notes shorten into a light staccato
            # instead of two identical full-length hits.
            melody_events = sorted(
                (e for e in events if e.voice_id == VOICE_MELODY), key=lambda e: e.start_us
            )
            shortened = {
                id(prev)
                for prev, curr in pairwise(melody_events)
                if prev.pitch_midi == curr.pitch_midi
                and curr.start_us - prev.start_us < prev.duration_us * 2
            }
            events = [
                (
                    PerformanceNoteEvent(
                        voice_id=e.voice_id,
                        pitch_midi=e.pitch_midi,
                        start_us=e.start_us,
                        duration_us=round(e.duration_us * 0.8),
                        velocity=e.velocity,
                        tie=e.tie,
                    )
                    if id(e) in shortened
                    else e
                )
                for e in events
            ]

        # Ghost notes: a quiet extra hit a 16th after some percussion
        # notes, skipped when a real hit already occupies the slot. Both
        # of the kit's draws come from the kit's stream, so nothing the
        # tune or the bed writes can move them.
        percussion = [e for e in events if e.voice_id == VOICE_PERCUSSION]
        if percussion:
            sixteenth_us = round(60_000_000 / score.tempo.bpm / 4)
            ghosts: list[PerformanceNoteEvent] = []
            for event in percussion:
                if kit_rng.random() >= GHOST_NOTE_PROBABILITY:
                    continue
                ghost_start = event.start_us + sixteenth_us
                if any(
                    other.pitch_midi == event.pitch_midi
                    and abs(other.start_us - ghost_start) < 20_000
                    for other in percussion
                ):
                    continue
                ghosts.append(
                    PerformanceNoteEvent(
                        voice_id=VOICE_PERCUSSION,
                        pitch_midi=event.pitch_midi,
                        start_us=ghost_start,
                        duration_us=event.duration_us,
                        velocity=kit_rng.randint(*GHOST_NOTE_VELOCITY_RANGE),
                        tie=False,
                    )
                )
            events.extend(ghosts)

    # Canonical order for the plan: realized time, then voice, then pitch.
    events.sort(key=lambda e: (e.start_us, e.voice_id, e.pitch_midi))
    controllers.sort(key=lambda c: (c.start_us, c.control, c.voice_id))
    return PerformancePlan.make(
        sample_rate=44100, notes=events, controllers=controllers, pitch_bends=[]
    )


__all__ = [
    "EXPRESSION_BASE",
    "GHOST_NOTE_PROBABILITY",
    "GHOST_NOTE_VELOCITY_RANGE",
    "HUMANIZE_TIMING_US",
    "HUMANIZE_VELOCITY_SPAN",
    "LEGATO_OVERLAP_US",
    "PEDAL_INSTRUMENTS",
    "PEDAL_RELEASE_LEAD_US",
    "PERCUSSION_TIMING_US",
    "SUSTAINED_INSTRUMENTS",
    "build_performance_plan",
]
