"""The sketch render, against the real binaries.

`render_sketch` exists because the interactive loop needs a preview in
seconds while publishing takes tens of seconds, and that asymmetry is what
makes the loop work at all. `tests/unit/test_render_audio.py` mocks every
external process, so it can prove the sketch's encode command differs from
the full render's — it cannot prove what a sketch *is* relative to the
delivered artifact, which is the property the product depends on.

That property is: **the sketch and the full render produce the same mix, and
the delivered WAV is that mix mastered.** Same SMF, same font, same gain, so
what a user approves from a preview is the piece they get; the difference is
mastering, which happens to the WAV and is therefore carried by every
deliverable — the WAV, the OGG encoded from it, and the animation that muxes
it. The sketch is the one artifact that is deliberately *not* mastered, and
this module measures how far below the deliverable that leaves it.

**The OGG cannot be compared by digest.** An Ogg Opus stream is not
byte-reproducible: ffmpeg picks a random stream serial per encode, so the
same WAV encoded twice with identical arguments produces different bytes.
Anything that asserts "the two OGGs differ" is therefore vacuous — it passes
however the encodes were configured. This module asserts on the *decoded
level* instead (`_levels`), which is the only thing that can attribute a
difference to mastering. `AudioArtifact.ogg_sha256` is still a useful digest
of the shipped file; it is just not a digest of the *encode*, and re-rendering
a job will never reproduce it.

Measured on the reference machine (darwin-arm64, FFmpeg 8.1.2 LGPL from
`scripts/build_ffmpeg.sh`, FluidR3_GM, a calming piano ensemble):

    piece   sketch    full    full/sketch
     30 s    0.65 s   1.99 s      3.1x
    120 s    1.27 s   6.30 s      5.0x
    600 s    4.81 s  29.64 s      6.2x

So a sketch is well inside the "a few seconds" the UI promises. It is not
flat in piece length, though: 20x the piece is 7x the sketch, so the
design's "~2-3 s regardless of piece length" holds at 120 s and understates
600 s. The bounds asserted below are the design's numbers rather than these
measurements, so the margin stays visible to whoever reads them.

The same machine measured the two downloads a full render produces. Before
F6 the OGG normalised to the -16 LUFS target on its own while the WAV was
written straight from FluidSynth, so they sat **16 dB apart** — OGG peak
-1.25 / mean -19.25 dBFS against the WAV's -17.24 / -35.94 — with the
primary deliverable the quiet one. F6 moved the master onto the WAV, and the
same 120 s calming piece now measures:

    full    WAV peak  -1.50  mean -18.80     OGG peak  -1.20  mean -18.80
    sketch  WAV peak -14.00  mean -35.70     OGG peak -14.50  mean -35.70

So both downloads land within a third of a dB and inside a decibel of the
-1.5 dBTP the master aims at, and a sketch remains ~12 dB below them — which
is the intended difference, not a defect. `TestTheTwoDeliverablesMatchInLevel`
holds the first of those; the second is what a preview is for.

This module skips unless a **release-gate FFmpeg** is available, which means
it does not run in CI: `.github/workflows/ci.yml` installs Python
dependencies and nothing else, and the Homebrew FFmpeg on a developer
machine is GPL and correctly fails the audit. Point `SAIMC_RENDER_FFMPEG`
at a binary from `scripts/build_ffmpeg.sh` to run it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest

from saimc.compose.engine import EngineOutput, compose
from saimc.compose.score import PerformancePlan, TempoPoint, realized_duration_seconds
from saimc.release.gates import gate_render_time_budget
from saimc.render.audio import (
    MASTER_TRUE_PEAK_DBTP,
    AudioArtifact,
    AudioRenderError,
    find_ffmpeg,
    find_fluidsynth,
    render_audio,
    render_sketch,
)
from saimc.render.ffmpeg_audit import audit_ffmpeg
from saimc.render.instruments import resolve_job_soundfont
from saimc.spec import DURATION_SECONDS_MAX, CompositionSpec

SKETCH_PIECE_SECONDS = 120
"""The length the design's ~2-3 s sketch figure was quoted for."""


def _unavailable_reason() -> str | None:
    """Why the real-binary path cannot run here, or None if it can.

    The audit is part of the check on purpose: a GPL FFmpeg on $PATH means
    the renderer would refuse the job in production, so a test that accepted
    it would be exercising a path no job ever takes.
    """
    try:
        ffmpeg = find_ffmpeg()
    except AudioRenderError as exc:
        return f"no ffmpeg available ({exc.message})"
    audit = audit_ffmpeg(ffmpeg)
    if not audit.ok:
        return (
            f"the ffmpeg at {ffmpeg} fails the release audit ({'; '.join(audit.reasons)}); "
            "build one with scripts/build_ffmpeg.sh and point SAIMC_RENDER_FFMPEG at it"
        )
    try:
        find_fluidsynth()
    except AudioRenderError as exc:
        return f"no fluidsynth available ({exc.message})"
    return None


_UNAVAILABLE = _unavailable_reason()

pytestmark = pytest.mark.skipif(_UNAVAILABLE is not None, reason=_UNAVAILABLE or "")


@dataclass(frozen=True)
class _Render:
    """One render of one piece, with what it cost."""

    artifact: AudioArtifact
    elapsed_s: float
    piece_seconds: float

    def clears_the_release_budget(self) -> tuple[bool, str]:
        """The §8/§10 #5 bar, applied to this measurement."""
        result = gate_render_time_budget(self.elapsed_s, self.piece_seconds)
        return result.passed, result.detail


def _compose(duration_seconds: int) -> tuple[EngineOutput, PerformancePlan, dict[int, str], Path]:
    """A piece, its plan, and the font the whole ensemble renders under."""
    output = compose(CompositionSpec(mood="calming", duration_seconds=duration_seconds, seed=7))
    voice_instruments = {vi.voice_id: vi.instrument for vi in output.voice_instruments}
    soundfont = resolve_job_soundfont(voice_instruments)
    if not soundfont.exists():
        pytest.skip(f"the soundfont for the default ensemble is not installed: {soundfont}")
    return output, output.performance_plan, voice_instruments, soundfont


def _render(
    renderer: Callable[..., AudioArtifact],
    *,
    plan: PerformancePlan,
    piece_seconds: float,
    bpm: float,
    tempo_changes: tuple[TempoPoint, ...],
    voice_instruments: dict[int, str],
    soundfont: Path,
    out_dir: Path,
) -> _Render:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    artifact = renderer(
        plan,
        bpm=bpm,
        tempo_changes=tempo_changes,
        voice_instruments=voice_instruments,
        soundfont_path=soundfont,
        out_dir=out_dir,
        job_id="sketch-probe",
    )
    return _Render(
        artifact=artifact,
        elapsed_s=time.perf_counter() - started,
        piece_seconds=piece_seconds,
    )


@dataclass(frozen=True)
class _Renders:
    """One piece rendered three ways.

    Three, not two, because a sketch and a full render differ in *two*
    settings — the master and the bitrate — and a comparison between the two
    end points cannot say which one caused what. `unmastered` is the control
    that isolates them: the same plan, the same bitrate as `full`, with only
    the master switched off. It is also the render whose WAV *is* the sketch's
    WAV, which is how the preview claim is checked.
    """

    sketch: _Render
    full: _Render
    unmastered: _Render


@pytest.fixture(scope="module")
def renders_120s(tmp_path_factory: pytest.TempPathFactory) -> _Renders:
    """A 120 s piece rendered every way, once for the whole module.

    Module-scoped because a mastered render is seconds of work and four tests
    read the same artifacts; rendering per test would multiply the cost of the
    slowest thing in the suite for no extra coverage.
    """
    output, plan, voices, soundfont = _compose(SKETCH_PIECE_SECONDS)
    piece_seconds = realized_duration_seconds(plan)

    def _into(name: str, renderer: Callable[..., AudioArtifact]) -> _Render:
        return _render(
            renderer,
            plan=plan,
            piece_seconds=piece_seconds,
            bpm=output.arrangement.tempo_bpm,
            tempo_changes=output.notation_score.tempo.changes,
            voice_instruments=voices,
            soundfont=soundfont,
            out_dir=tmp_path_factory.mktemp(f"120s-{name}"),
        )

    return _Renders(
        sketch=_into("sketch", render_sketch),
        full=_into("full", render_audio),
        # `render_audio` with the master off and nothing else changed — the
        # control that attributes the difference to mastering rather than to
        # the bitrate that also differs between `sketch` and `full`.
        unmastered=_into("unmastered", partial(render_audio, master=False)),
    )


def _levels(ogg_path: Path) -> tuple[float, float]:
    """(peak, mean) volume of an encoded artifact, in dBFS.

    Read back through `volumedetect` rather than by comparing digests: an
    Ogg Opus stream is **not byte-reproducible** — ffmpeg picks a random
    stream serial per encode, so the same WAV encoded twice with identical
    arguments produces different bytes. An `ogg_a != ogg_b` assertion
    therefore passes whatever the settings were, which is exactly the trap
    this module fell into once already. Only a measurement can say whether
    the master changed anything.
    """
    import re

    from saimc.jobs.stages import safe_run

    proc = safe_run(
        [
            find_ffmpeg(),
            "-hide_banner",
            "-i",
            str(ogg_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        timeout_s=120.0,
    )
    text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    found = {
        key: float(value) for key, value in re.findall(r"(max|mean)_volume:\s*(-?[\d.]+) dB", text)
    }
    return found["max"], found["mean"]


class TestTheSketchPreviewsTheDeliveredAudio:
    """What the user approves is what they get."""

    def test_the_sketch_delivers_the_same_mix_the_full_render_masters(
        self, renders_120s: _Renders
    ) -> None:
        """Same notes, same font, same gain — nothing about the music differs.

        Neither the master nor the bitrate touches the mix, which is what
        makes a preview a preview: what the user hears while choosing is the
        audio they receive, not a rendering of the same score. The comparison
        is against the `unmastered` control rather than against `full`,
        because `full`'s WAV is the mastered one and the two *should* differ
        there — that difference is the master, and it is asserted below.
        """
        hashes = {
            "sketch": renders_120s.sketch.artifact.primary_sha256,
            "unmastered": renders_120s.unmastered.artifact.primary_sha256,
        }
        assert len(set(hashes.values())) == 1, hashes
        assert (
            renders_120s.sketch.artifact.primary_size_bytes
            == renders_120s.unmastered.artifact.primary_size_bytes
        )
        # The master is not a no-op, so the delivered WAV is a different file
        # from the mix. Without this the test above would be satisfied by a
        # build that mastered nothing at all.
        assert renders_120s.full.artifact.primary_sha256 != hashes["sketch"]

    def test_mastering_makes_the_audio_materially_louder(self, renders_120s: _Renders) -> None:
        """`master=False` is not a no-op — the same bitrate, a quieter mix.

        The control render differs from `full` in exactly one setting, so this
        is the assertion that mastering does something. Measured on the
        reference machine: peak -1.25 dBFS and mean -19.25 dBFS mastered,
        against -17.24 and -35.94 unmastered — the master is worth about
        16 dB, which is most of the difference between a preview and a
        deliverable.

        The 6 dB bar is well under that so a loaded machine or a different
        font cannot make this flap, while still being far more than the
        ~0.1 dB an ignored flag would show.
        """
        mastered_peak, mastered_mean = _levels(renders_120s.full.artifact.ogg_path)
        raw_peak, raw_mean = _levels(renders_120s.unmastered.artifact.ogg_path)
        assert mastered_peak > raw_peak + 6.0, (mastered_peak, raw_peak)
        assert mastered_mean > raw_mean + 6.0, (mastered_mean, raw_mean)

    def test_the_sketch_encodes_to_a_smaller_file(self, renders_120s: _Renders) -> None:
        """Half the bitrate is the other half of what a sketch saves.

        Sizes rather than digests, for the reason in `_levels`. This conflates
        the bitrate with the master, which is fine here — the user-facing fact
        is that a preview is cheaper to serve, and the unit test pins the
        mechanism by asserting `64k` reaches the command line.
        """
        sketch = renders_120s.sketch.artifact
        unmastered = renders_120s.unmastered.artifact
        assert sketch.ogg_codec == unmastered.ogg_codec == "opus"
        assert sketch.ogg_size_bytes is not None
        assert unmastered.ogg_size_bytes is not None
        assert sketch.ogg_size_bytes < unmastered.ogg_size_bytes


class TestTheTwoDeliverablesMatchInLevel:
    """The WAV and the OGG a user downloads, measured against each other.

    Roadmap §8 states media reproducibility as "same audio levels within
    tolerance", and this is that claim about the two files one render
    produces. It is the bar F6 was opened for: measured before the fix, the
    delivered WAV sat at peak -17.24 / mean -35.94 dBFS against the delivered
    OGG's -1.25 / -19.25 — **16 dB apart, with the primary deliverable the
    quiet one.** They are together now because the OGG is encoded from the
    mastered WAV rather than normalising to the target separately, and the
    same piece measures 0.30 dB apart on the peak and 0.00 dB on the mean.

    The animation is the third deliverable and is not measured here: it muxes
    `AudioArtifact.primary_path` (`jobs/stages.py`), so it carries whatever
    this WAV measures, and a WebM render is minutes of VP9 for a fact the WAV
    already states.
    """

    TOLERANCE_DB = 1.0
    """Lossy coding of a loud mix moves peak and mean by hundredths of a dB.

    The failure this guards — an encode that normalises to its own target, or
    a WAV that skipped its master — is 6 dB or more, so 1 dB separates the two
    without being tight enough for a font or a machine to flap it.
    """

    def test_the_wav_and_the_ogg_are_at_the_same_level(self, renders_120s: _Renders) -> None:
        wav_peak, wav_mean = _levels(renders_120s.full.artifact.primary_path)
        ogg_peak, ogg_mean = _levels(renders_120s.full.artifact.ogg_path)
        assert abs(wav_peak - ogg_peak) < self.TOLERANCE_DB, (wav_peak, ogg_peak)
        assert abs(wav_mean - ogg_mean) < self.TOLERANCE_DB, (wav_mean, ogg_mean)

    def test_the_pair_is_at_the_master_target(self, renders_120s: _Renders) -> None:
        """Both files land where the master aimed, not merely near each other.

        Two files 20 dB below the target would satisfy the assertion above,
        so this pins the direction and the distance. The delivered WAV
        measured -1.50 dBFS, exactly `MASTER_TRUE_PEAK_DBTP`: loudnorm's
        `linear=true` still applies a true-peak limiter at that value, and a
        sample peak read back with `volumedetect` cannot exceed a bounded true
        peak. So the ceiling is exact rather than approximate, and the floor is
        the part with room in it.
        """
        wav_peak, _ = _levels(renders_120s.full.artifact.primary_path)
        assert wav_peak > MASTER_TRUE_PEAK_DBTP - 4.0, wav_peak
        assert wav_peak <= MASTER_TRUE_PEAK_DBTP, wav_peak


class TestTheSketchIsFastEnoughToWaitOn:
    def test_it_costs_a_fraction_of_a_full_render(self, renders_120s: _Renders) -> None:
        """The asymmetry the loop is built on, asserted relatively.

        A ratio rather than a duration, so this says the same thing on a slow
        machine as on a fast one: mastering is what the sketch skips, and
        skipping it has to be worth something substantial.
        """
        assert renders_120s.full.elapsed_s >= 2 * renders_120s.sketch.elapsed_s, (
            f"sketch {renders_120s.sketch.elapsed_s:.2f}s vs "
            f"full {renders_120s.full.elapsed_s:.2f}s — the master is not the cost "
            "it was assumed to be, so the sketch needs re-measuring"
        )

    def test_it_stays_inside_the_design_budget_at_120s(self, renders_120s: _Renders) -> None:
        """~2-3 s is the figure the interaction design assumes, and the reason
        a four-wide fan-out fits in a turn. 5 s is the ceiling here; the margin
        is for a loaded machine, not slack in the design.
        """
        assert renders_120s.sketch.elapsed_s < 5.0, (
            f"a {SKETCH_PIECE_SECONDS}s sketch took {renders_120s.sketch.elapsed_s:.2f}s; "
            "the fan-out budget assumes a few seconds per draft"
        )


class TestTheFullRenderFitsTheReleaseBudget:
    def test_the_gate_passes_on_a_real_measurement(self, renders_120s: _Renders) -> None:
        """`gate_render_time_budget` wired to the render it was written for.

        The gate was exported and never called with a measured elapsed time —
        only with hand-made numbers in `test_release_gates.py`. This is the
        §8/§10 #5 bar, measured.
        """
        passed, detail = renders_120s.full.clears_the_release_budget()
        assert passed, detail


class TestTheSketchSurvivesTheDurationCap:
    """The longest piece the spec allows still previews in seconds.

    A separate case because the sketch is not flat in piece length — it grows
    sub-linearly, and the cap is where that growth is felt.
    """

    @pytest.fixture(scope="class")
    def cap(self, tmp_path_factory: pytest.TempPathFactory) -> _Render:
        output, plan, voices, soundfont = _compose(DURATION_SECONDS_MAX)
        return _render(
            render_sketch,
            plan=plan,
            piece_seconds=realized_duration_seconds(plan),
            bpm=output.arrangement.tempo_bpm,
            tempo_changes=output.notation_score.tempo.changes,
            voice_instruments=voices,
            soundfont=soundfont,
            out_dir=tmp_path_factory.mktemp("sketch-cap"),
        )

    def test_it_stays_inside_the_design_budget_at_the_cap(self, cap: _Render) -> None:
        """~4.8 s measured on the reference machine; 12 s is the ceiling, for
        the same reason as the 120 s bound.
        """
        assert cap.elapsed_s < 12.0, f"a {DURATION_SECONDS_MAX}s sketch took {cap.elapsed_s:.2f}s"

    def test_the_gate_passes_on_it_too(self, cap: _Render) -> None:
        passed, detail = cap.clears_the_release_budget()
        assert passed, detail
