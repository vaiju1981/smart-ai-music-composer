"""Unit tests for `saimc-preferences`.

The proposal itself is `tests/unit/test_session_retune.py`'s subject. What is
pinned here is what a CLI owes on top of a pure report: that the log is opened
where the deployment says it is, that a log it cannot read is reported rather
than traced, that the report a person is handed shows the diff it is asking
them to approve, and that exit status stays 0 — a proposal is not a gate, and
nothing about reading a log is a failure.

The rows go through a real `PreferenceLog` rather than being handed to `_lines`
directly, so what is tested is the report a reader of the file actually gets.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from saimc.preferences_cli import _evidence, _plural, _rate, app
from saimc.session.deltas import Delta, SetAccompanimentDensity, SetHarmonyTexture, SetMelodyBand
from saimc.session.models import VerdictValue
from saimc.session.preferences import Preference, PreferenceLog
from saimc.session.retune import Candidate, proposal

runner = CliRunner()

_NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
_PAD = SetHarmonyTexture(broken_chord=False)
_DENSITY = SetAccompanimentDensity(step_ticks=1440)
_OFFERED = SetMelodyBand(semitones=12)


def _judged(
    draft_id: str,
    requests: Sequence[Delta],
    *,
    verdict: VerdictValue = "like",
    at: datetime = _NOW,
    source: str | None = None,
) -> tuple[Preference, ...]:
    """The rows one verdict writes for a chain, as the writer writes them."""
    return tuple(
        Preference(
            draft_id=draft_id,
            at=at,
            plan_hash="0" * 64,
            delta=request,
            verdict=verdict,
            requests_source=source,  # type: ignore[arg-type]
        )
        for request in requests
    )


@pytest.fixture
def log(tmp_path: Path) -> PreferenceLog:
    return PreferenceLog(tmp_path / "preferences")


class TestTheReport:
    def test_it_reads_the_log_the_writer_wrote(self, log: PreferenceLog) -> None:
        log.append(_judged("draft-one", (_PAD, _DENSITY)))
        result = runner.invoke(app, ["--log", str(log.root)])
        assert result.exit_code == 0
        assert f"Log: {log.path}" in result.stdout
        assert "1 judgement over 2 requests." in result.stdout
        assert "harmony_pad_coverage" in result.stdout

    def test_the_two_counts_are_printed_because_they_differ(self, log: PreferenceLog) -> None:
        """The line that makes one-opinion-per-judgement visible.

        A reader who only saw "3 requests" would read the rates as three
        opinions, which is the arithmetic this reader exists not to make.
        """
        log.append(_judged("draft-one", (_PAD, _DENSITY, _OFFERED)))
        result = runner.invoke(app, ["--log", str(log.root)])
        assert "1 judgement over 3 requests." in result.stdout

    def test_it_prints_the_tables_order_beside_the_proposed_one(self, log: PreferenceLog) -> None:
        """A reorder is the one thing a reader must be able to see without diffing.

        The second request of the bed is the only one any user liked, so the
        evidence puts it first and the table has it second. Asserted as the whole
        three-line block rather than as two substring searches, and that is not
        tidiness: `texture_hierarchy` holds the same two requests in the opposite
        order, so its `proposed:` line *is* the string this bar's `table:` line
        carries — an unanchored assertion here is satisfied by the bar below it,
        and a report that printed the table's order twice would pass.
        """
        log.append(
            (
                *_judged("draft-one", (_PAD,), verdict="dislike"),
                *_judged("draft-two", (_DENSITY,), at=_NOW + timedelta(minutes=1)),
            )
        )
        result = runner.invoke(app, ["--log", str(log.root)])
        assert (
            "harmony_pad_coverage   [reordered]\n"
            "  table:    SetHarmonyTexture(broken_chord=False),"
            " SetAccompanimentDensity(step_ticks=1440)\n"
            "  proposed: SetAccompanimentDensity(step_ticks=1440),"
            " SetHarmonyTexture(broken_chord=False)\n"
        ) in result.stdout

    def test_only_the_bar_the_evidence_disagrees_with_is_marked(self, log: PreferenceLog) -> None:
        """One request liked, and the two bars holding it part ways.

        `SetAccompanimentDensity(step_ticks=1440)` is second under
        `harmony_pad_coverage` and first under `texture_hierarchy`, so the same
        single judgement reorders one bar and leaves the other alone. A report
        that marked both — or neither — would be printing the evidence rather
        than the diff it is asking a person to approve.
        """
        log.append(_judged("draft-one", (_DENSITY,)))
        result = runner.invoke(app, ["--log", str(log.root)])
        assert "harmony_pad_coverage   [reordered]" in result.stdout
        assert "texture_hierarchy   [reordered]" not in result.stdout

    def test_a_request_no_user_has_seen_is_not_printed_as_a_rate(
        self, log: PreferenceLog
    ) -> None:
        """`0%` would read as an opinion, and there is none."""
        log.append(_judged("draft-one", (_PAD,)))
        result = runner.invoke(app, ["--log", str(log.root)])
        assert "SetAccompanimentDensity(step_ticks=1440)  no evidence" in result.stdout

    def test_a_request_a_repair_served_and_the_table_lacks_is_listed(
        self, log: PreferenceLog
    ) -> None:
        log.append(_judged("draft-one", (_OFFERED,), source="repair"))
        result = runner.invoke(app, ["--log", str(log.root)])
        assert "Requests a repair served that the table does not hold (1):" in result.stdout
        assert "SetMelodyBand(semitones=12)  100% (1 like, 0 dislike)" in result.stdout

    def test_a_request_the_user_typed_is_not_offered_as_a_candidate(
        self, log: PreferenceLog
    ) -> None:
        log.append(_judged("draft-one", (_OFFERED,), source="typed"))
        result = runner.invoke(app, ["--log", str(log.root)])
        assert "Requests a repair served that the table does not hold (0):" in result.stdout
        assert "    (none)" in result.stdout

    def test_the_unproven_entries_name_their_bar(self, log: PreferenceLog) -> None:
        """Six of the table's eight entries, because the pad is proven twice.

        `SetHarmonyTexture(broken_chord=False)` is held for two bars, so the one
        judgement carrying it lifts *two* entries out of this list, and the count
        is a statement about the table rather than about the request vocabulary.
        """
        log.append(_judged("draft-one", (_PAD,)))
        result = runner.invoke(app, ["--log", str(log.root)])
        assert "Table entries no judgement carries (6):" in result.stdout
        assert "    harmony_pad_coverage: SetAccompanimentDensity(step_ticks=1440)" in result.stdout
        assert "    harmony_pad_coverage: SetHarmonyTexture(broken_chord=False)" not in result.stdout

    def test_an_absent_log_says_so_instead_of_reporting_an_empty_one(
        self, log: PreferenceLog
    ) -> None:
        result = runner.invoke(app, ["--log", str(log.root)])
        assert result.exit_code == 0
        assert "(absent: nothing has been recorded)" in result.stdout
        assert "0 judgements over 0 requests." in result.stdout

    def test_the_default_root_is_the_deployments(
        self, log: PreferenceLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run without `--log`, the log is found where the app puts it."""
        log.append(_judged("draft-one", (_PAD,)))
        monkeypatch.setenv("SAIMC_PREFERENCES_DIR", str(log.root))
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "1 judgement over 1 request." in result.stdout

    def test_a_proposal_is_not_a_gate(self, log: PreferenceLog) -> None:
        """Every request in the log disliked is still a report, not a failure."""
        log.append(
            (
                *_judged("draft-one", (_PAD,), verdict="dislike"),
                *_judged("draft-two", (_DENSITY,), verdict="dislike", at=_NOW + timedelta(minutes=1)),
            )
        )
        assert runner.invoke(app, ["--log", str(log.root)]).exit_code == 0

    def test_the_same_log_prints_the_same_report(self, log: PreferenceLog) -> None:
        log.append(
            (
                *_judged("draft-one", (_OFFERED,), source="repair"),
                *_judged("draft-two", (SetMelodyBand(semitones=13),), source="repair", at=_NOW + timedelta(minutes=1)),
            )
        )
        first = runner.invoke(app, ["--log", str(log.root)]).stdout
        second = runner.invoke(app, ["--log", str(log.root)]).stdout
        assert first == second

    def test_an_empty_log_reports_the_whole_table_and_moves_nothing(
        self, log: PreferenceLog
    ) -> None:
        """The report for a deployment nothing has judged yet, pinned whole.

        A golden rather than a set of `in` assertions, because the layout is the
        deliverable here: a change to it should be a declared edit, not a line
        that quietly stopped being printed.
        """
        log.root.mkdir(parents=True, exist_ok=True)
        log.path.write_text("", encoding="utf-8")
        result = runner.invoke(app, ["--log", str(log.root)])
        assert result.stdout == (
            f"Log: {log.path}\n"
            "0 judgements over 0 requests.\n"
            "\n"
            "Every bar of the repair table, with the evidence the log holds for it.\n"
            "\n"
            "harmony_pad_coverage\n"
            "  table:    SetHarmonyTexture(broken_chord=False),"
            " SetAccompanimentDensity(step_ticks=1440)\n"
            "  proposed: SetHarmonyTexture(broken_chord=False),"
            " SetAccompanimentDensity(step_ticks=1440)\n"
            "    SetHarmonyTexture(broken_chord=False)     no evidence\n"
            "    SetAccompanimentDensity(step_ticks=1440)  no evidence\n"
            "\n"
            "texture_hierarchy\n"
            "  table:    SetAccompanimentDensity(step_ticks=1440),"
            " SetHarmonyTexture(broken_chord=False)\n"
            "  proposed: SetAccompanimentDensity(step_ticks=1440),"
            " SetHarmonyTexture(broken_chord=False)\n"
            "    SetAccompanimentDensity(step_ticks=1440)  no evidence\n"
            "    SetHarmonyTexture(broken_chord=False)     no evidence\n"
            "\n"
            "max_leap_semitones\n"
            "  table:    SetMelodyBand(semitones=9), SetMotifVariation(factor=0.5)\n"
            "  proposed: SetMelodyBand(semitones=9), SetMotifVariation(factor=0.5)\n"
            "    SetMelodyBand(semitones=9)     no evidence\n"
            "    SetMotifVariation(factor=0.5)  no evidence\n"
            "\n"
            "leap_recovery_ratio\n"
            "  table:    SetMelodyBand(semitones=14), SetMelodyBand(semitones=9)\n"
            "  proposed: SetMelodyBand(semitones=14), SetMelodyBand(semitones=9)\n"
            "    SetMelodyBand(semitones=14)  no evidence\n"
            "    SetMelodyBand(semitones=9)   no evidence\n"
            "\n"
            "Requests a repair served that the table does not hold (0):\n"
            "    (none)\n"
            "\n"
            "Table entries no judgement carries (8):\n"
            "    harmony_pad_coverage: SetHarmonyTexture(broken_chord=False)\n"
            "    harmony_pad_coverage: SetAccompanimentDensity(step_ticks=1440)\n"
            "    texture_hierarchy: SetAccompanimentDensity(step_ticks=1440)\n"
            "    texture_hierarchy: SetHarmonyTexture(broken_chord=False)\n"
            "    max_leap_semitones: SetMelodyBand(semitones=9)\n"
            "    max_leap_semitones: SetMotifVariation(factor=0.5)\n"
            "    leap_recovery_ratio: SetMelodyBand(semitones=14)\n"
            "    leap_recovery_ratio: SetMelodyBand(semitones=9)\n"
        )


class TestTheMalformedLog:
    def test_a_line_that_is_not_a_row_is_reported_rather_than_traced(
        self, log: PreferenceLog
    ) -> None:
        """The reader names the path and the line; the traceback is noise.

        Still a failure — the exit status is the one thing that changes — so a
        caller in a script is not told a report it could not read was fine.
        """
        log.append(_judged("draft-one", (_PAD,)))
        with log.path.open("a", encoding="utf-8") as handle:
            handle.write("this is not a row\n")
        result = runner.invoke(app, ["--log", str(log.root)])
        assert result.exit_code == 1
        assert f"{log.path} line 2 is not a preference row" in result.output
        assert "Traceback" not in result.output

    def test_a_judgement_whose_rows_disagree_is_refused_by_name(
        self, log: PreferenceLog
    ) -> None:
        """A hand-edited log, refused where it is read rather than counted."""
        log.append(_judged("draft-one", (_PAD,)))
        lines = log.path.read_text(encoding="utf-8").splitlines()
        torn = json.loads(lines[0])
        torn["verdict"] = "dislike" if torn["verdict"] == "like" else "like"
        lines.append(json.dumps(torn))
        log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = runner.invoke(app, ["--log", str(log.root)])
        assert result.exit_code == 1
        assert "a judgement has one verdict" in result.output


class TestTheLineHelpers:
    def test_a_rate_names_both_counts(self) -> None:
        assert _rate(Candidate(delta=_PAD, likes=3, dislikes=1)) == "75% (3 like, 1 dislike)"

    def test_a_candidate_with_no_evidence_is_named_as_such(self) -> None:
        assert _rate(Candidate(delta=_PAD, likes=0, dislikes=0)) == "no evidence"

    def test_a_count_of_one_is_not_a_plural(self) -> None:
        assert _plural(1, "judgement") == "1 judgement"
        assert _plural(0, "judgement") == "0 judgements"
        assert _plural(2, "request") == "2 requests"

    def test_the_rates_line_up_in_one_column(self) -> None:
        """The number is what a reader compares down the block, so it is aligned."""
        lines = _evidence(
            (
                Candidate(delta=SetMelodyBand(semitones=9), likes=1, dislikes=0),
                Candidate(delta=SetHarmonyTexture(broken_chord=False), likes=0, dislikes=1),
            ),
            indent="  ",
        )
        assert lines == [
            "  SetMelodyBand(semitones=9)             100% (1 like, 0 dislike)",
            "  SetHarmonyTexture(broken_chord=False)  0% (0 like, 1 dislike)",
        ]

    def test_an_empty_block_is_a_block_with_no_lines(self) -> None:
        assert _evidence((), indent="  ") == []


class TestTheProposalItReads:
    def test_the_report_is_the_proposals_own_numbers(self, log: PreferenceLog) -> None:
        """The CLI adds no arithmetic, so the two cannot disagree."""
        rows = _judged("draft-one", (_PAD, _DENSITY, _OFFERED), source="repair")
        log.append(rows)
        assert proposal(log.rows()) == proposal(rows)
