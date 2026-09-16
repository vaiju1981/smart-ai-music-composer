"""Release-gate harness per `docs/roadmap.md` §8/§10.

Each gate is a function over real engine/renderer objects that returns
a `GateResult`; the acceptance suite in `tests/acceptance/` runs them
against the pinned spec matrix. Gates that need real external binaries
(FluidSynth, the render-service, the audited FFmpeg) are exercised by
the end-to-end run (task: real-binary verification) using these same
functions.
"""

from saimc.release.gates import (
    GateResult,
    gate_canonical_reproducibility,
    gate_composition_correctness,
    gate_duration_tolerance,
    gate_midi_parseable_and_onsets,
    gate_musical_quality,
    gate_musicxml_structural,
    gate_render_time_budget,
    gate_spec_round_trip,
)

__all__ = [
    "GateResult",
    "gate_canonical_reproducibility",
    "gate_composition_correctness",
    "gate_duration_tolerance",
    "gate_midi_parseable_and_onsets",
    "gate_musical_quality",
    "gate_musicxml_structural",
    "gate_render_time_budget",
    "gate_spec_round_trip",
]
