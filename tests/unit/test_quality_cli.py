"""Unit tests for the quality CLI's pure helpers.

`main()` itself composes the whole matrix (four pieces of three minutes
each), so it is exercised end to end by hand rather than per-test; the
parts with branching worth pinning are the spec loading, the labels, and
the threshold rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from saimc.quality import QUALITY_THRESHOLDS, QualityThreshold
from saimc.quality_cli import DEFAULT_MATRIX, _bar_text, _default_specs, _label, _load_spec
from saimc.spec import CompositionSpec, Mood


class TestDefaultMatrix:
    def test_the_matrix_spans_every_mood(self) -> None:
        moods = {spec.mood for spec in _default_specs()}
        assert moods == set(Mood)

    def test_the_matrix_includes_a_full_ensemble(self) -> None:
        # One entry must exercise the multi-voice path, or a regression
        # that only shows with a kit and a bass line goes unmeasured.
        roles = {
            frozenset(entry.role for entry in spec.instrumentation) for spec in _default_specs()
        }
        assert frozenset({"melody", "harmony", "bass", "percussion"}) in roles

    def test_every_entry_is_a_valid_spec(self) -> None:
        assert len(_default_specs()) == len(DEFAULT_MATRIX)
        for spec in _default_specs():
            assert spec.duration_seconds > 0


class TestLoadSpec:
    def test_it_reads_a_raw_spec(self, tmp_path: Path) -> None:
        path = tmp_path / "spec.json"
        path.write_text(json.dumps({"mood": "calming", "seed": 11}), encoding="utf-8")
        spec = _load_spec(path)
        assert isinstance(spec, CompositionSpec)
        assert spec.mood is Mood.CALMING
        assert spec.seed == 11

    def test_it_reads_the_spec_out_of_a_manifest(self, tmp_path: Path) -> None:
        # A manifest wraps the spec under input_spec.spec, so pointing the
        # CLI at a manifest measures the piece that job actually rendered.
        path = tmp_path / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "job_id": "abc",
                    "input_spec": {"spec": {"mood": "sleep", "seed": 3}},
                }
            ),
            encoding="utf-8",
        )
        assert _load_spec(path).mood is Mood.SLEEP

    def test_it_rejects_a_payload_that_is_not_a_spec(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.json"
        path.write_text(json.dumps({"mood": "not-a-mood"}), encoding="utf-8")
        with pytest.raises(ValueError):
            _load_spec(path)


class TestLabel:
    def test_the_label_names_the_mood_duration_and_voices(self) -> None:
        spec = CompositionSpec(mood=Mood.ELECTRIFYING, seed=4)
        label = _label(2, spec)
        voices = "+".join(entry.instrument.value for entry in spec.instrumentation)
        assert label == f"2:electrifying:{spec.duration_seconds}s:{voices}"

    def test_labels_are_unique_across_the_default_matrix(self) -> None:
        labels = [_label(i, spec) for i, spec in enumerate(_default_specs())]
        assert len(set(labels)) == len(labels)


class TestBarText:
    # Synthetic bars: the axis is a field of the table rather than of the
    # rendering, so any of the three does for a test of the text.
    def test_a_one_sided_minimum_reads_as_at_least(self) -> None:
        bar = QualityThreshold(
            metric="x", minimum=0.45, maximum=None, rationale="r", hint="h", axis="melody"
        )
        assert _bar_text(bar) == ">= 0.45"

    def test_a_one_sided_maximum_reads_as_at_most(self) -> None:
        bar = QualityThreshold(
            metric="x", minimum=None, maximum=0.25, rationale="r", hint="h", axis="melody"
        )
        assert _bar_text(bar) == "<= 0.25"

    def test_a_two_sided_bar_reads_as_a_band(self) -> None:
        bar = QualityThreshold(
            metric="x", minimum=7.0, maximum=24.0, rationale="r", hint="h", axis="melody"
        )
        assert _bar_text(bar) == "in [7.00, 24.00]"

    def test_every_shipped_threshold_renders(self) -> None:
        for threshold in QUALITY_THRESHOLDS:
            assert _bar_text(threshold)
