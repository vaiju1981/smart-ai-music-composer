"""The served page, driven in a browser.

What these hold is the page's *behaviour*: that a brief becomes a job, that
the status panel follows the job's state, and — the one with teeth — that the
score artifact cannot put script into the document. The last is the reason
this file exists at all: `adoptScore` is a security control, and a control
with no test that can fail is a comment.
"""

from __future__ import annotations

from typing import Any

from saimc.jobs.state import JobState
from saimc.jobs.storage import ArtifactRecord
from tests.ui.conftest import Studio

# An SVG with three ways to run script, all of which `innerHTML` would have
# honoured: an `onload` on the root, an `<image>` whose broken href fires
# `onerror`, and a `<script>`. The notes around them are ordinary, so a page
# that renders the score correctly *and* runs none of this is the pass.
HOSTILE_SVG = """<div id="osmdCanvasPage1"><svg id="osmdSvgPage1" viewBox="0 0 100 40"
     xmlns="http://www.w3.org/2000/svg" onload="window.__pwned = 'root-onload'">
  <script>window.__pwned = 'script-element';</script>
  <image href="https://example.invalid/nope.png" onerror="window.__pwned = 'image-onerror'"/>
  <foreignObject width="10" height="10"><body xmlns="http://www.w3.org/1999/xhtml">
    <img src="x" onerror="window.__pwned = 'foreign-object'"/></body></foreignObject>
  <g class="vf-stave"><path d="M2 20 L98 20" stroke="#000"/>
    <text x="4" y="14" class="vf-text">Andante</text></g>
</svg></div>"""


def _complete_job_with_score(studio: Studio, markup: str) -> str:
    """Seed a finished job whose sheet artifact is `markup`."""
    job = studio.jobs.create("a calm piano piece")
    studio.jobs.ensure_artifact_dir(job.job_id).joinpath("sheet.svg").write_text(
        markup, encoding="utf-8"
    )
    job = studio.jobs.attach_artifact(
        job,
        ArtifactRecord(
            kind="sheet",
            container="svg",
            codec="svg",
            path="sheet.svg",
            sha256="0" * 64,
            size_bytes=len(markup.encode("utf-8")),
        ),
    )
    job.state = JobState.COMPLETE
    job.progress = 1.0
    job.current_stage = "complete"
    studio.jobs.save(job)
    return job.job_id


class TestThePageLoads:
    def test_the_brief_form_is_the_front_door(self, studio: Studio, page: Any) -> None:
        page.goto(studio.url)
        assert page.locator("#prompt").is_visible()
        assert page.locator("#compose").is_visible()
        # The status panel is for a job that exists; nothing has been asked yet.
        assert page.locator("#status-panel").is_hidden()

    def test_the_service_pill_reports_the_studio(self, studio: Studio, page: Any) -> None:
        page.goto(studio.url)
        # `/health` answers with no worker running, so the pill must settle on
        # a *reported* state rather than staying on its "Checking…" placeholder.
        page.wait_for_function(
            "() => !document.getElementById('service-text').textContent.startsWith('Checking')"
        )

    def test_a_brief_becomes_a_job(self, studio: Studio, page: Any) -> None:
        page.goto(studio.url)
        page.fill("#prompt", "a calming piano piece in C major")
        page.click("#compose")
        page.wait_for_selector("#status-panel:visible")
        assert "JOB" in page.locator("#job-id").inner_text()
        assert len(list(studio.jobs.list_all())) == 1


class TestTheScoreCannotCarryScript:
    def test_a_hostile_score_renders_its_notes_and_runs_nothing(
        self, studio: Studio, page: Any
    ) -> None:
        job_id = _complete_job_with_score(studio, HOSTILE_SVG)
        page.goto(f"{studio.url}/#/job/{job_id}")
        page.wait_for_selector("#sheet-holder svg")

        assert page.evaluate("() => window.__pwned") is None
        # The music survived the strip: this is what separates sanitising from
        # refusing to render.
        assert page.locator("#sheet-holder svg .vf-text").text_content() == "Andante"
        assert page.locator("#sheet-holder svg path").count() == 1
        # ...and the ways in did not. `<script>` and `<foreignObject>` are
        # removed outright, because neither has a use in a notation SVG.
        for gone in ("script", "foreignObject", "img"):
            assert page.locator(f"#sheet-holder svg {gone}").count() == 0
        # `<image>` is *kept and disarmed* rather than deleted: it is legal
        # SVG that a future renderer could emit for a raster glyph, and what
        # makes it dangerous is the handler and the outbound href, not the
        # element. An image with neither fetches nothing and runs nothing.
        assert page.locator("#sheet-holder svg image").count() == 1
        assert (
            page.evaluate(
                "() => Array.from(document.querySelectorAll('#sheet-holder svg image'))"
                "      .flatMap(n => Array.from(n.attributes).map(a => a.name))"
            )
            == []
        )
        assert (
            page.evaluate(
                "() => document.querySelector('#sheet-holder svg').hasAttribute('onload')"
            )
            is False
        )

    def test_an_unparseable_score_says_so_instead_of_throwing(
        self, studio: Studio, page: Any
    ) -> None:
        job_id = _complete_job_with_score(studio, "this is not markup at all")
        page.goto(f"{studio.url}/#/job/{job_id}")
        page.wait_for_function(
            "() => document.getElementById('sheet-holder').textContent.indexOf('could not be loaded') !== -1"
        )


class TestTheCeilingIsStated:
    """What the engine will not do, said before the brief is typed.

    Reported from a real run: a brief asking for "10 to 15 instruments" came
    back with three or four, and nothing anywhere said why. It is not a parse
    failure — `ROLE_LIMITS` is one melody, two harmony, one bass and one kit,
    so five is the most this engine writes and six entries are refused by the
    schema. An enforced limit nobody is told about reads, from outside, as the
    product ignoring what was asked.
    """

    def test_the_page_says_how_many_voices_it_writes(self, studio: Studio, page: Any) -> None:
        page.goto(studio.url)
        page.wait_for_function(
            "() => document.getElementById('ensemble-note').textContent.length > 0"
        )
        note = page.locator("#ensemble-note").inner_text()
        # The numbers are the next test's business — this one holds that the
        # sentence is there, names every role, and says what it is about.
        # Pinning "up to 5 voices" here is what this test did first, and the
        # ceiling moving to fifteen broke it for no reason worth a failure.
        assert "voices" in note
        for role in ("melody", "harmony", "bass", "percussion"):
            assert role in note

    def test_the_sentence_is_built_from_what_the_schema_enforces(
        self, studio: Studio, page: Any
    ) -> None:
        """Read off `/meta`, so it cannot drift from the validator.

        The numbers in `ROLE_LIMITS` are what `_validate_ensemble` refuses
        against and what `/meta` publishes; the page renders them rather than
        repeating them, so a raised ceiling changes the sentence for free.
        """
        served = page.request.get(studio.url + "/meta").json()["ensemble"]
        page.goto(studio.url)
        page.wait_for_function(
            "() => document.getElementById('ensemble-note').textContent.length > 0"
        )
        note = page.locator("#ensemble-note").inner_text()
        assert f"up to {served['max_voices']} voices" in note
        for role, limit in served["max_by_role"].items():
            assert f"{limit} {role}" in note
