"""The session workspace, driven in a browser.

What this holds is the claim the page makes: that the harness is *visible*.
Before it, `session/` was seven thousand lines reachable only by curl — the
conductor's turns, the candidates it drafted, the scorecard behind each one,
the verdict, the edit and the publish. The page now shows all of it, and every
test below is one of those things being on screen and working.

The model is scripted; nothing under it is. The engine composes, the linter
lints and the scorecard measures for real, so what the cards display is what
the pipeline produced.
"""

from __future__ import annotations

from typing import Any

from saimc.llm.base import ChatResult, ToolCall
from tests.ui.conftest import Studio


def _open_a_session(studio: Studio, page: Any) -> None:
    """Type a brief and wait for the workspace the first turn fills."""
    page.goto(studio.url)
    page.wait_for_function(
        "() => !document.getElementById('conductor-text').textContent.startsWith('Checking')"
    )
    page.fill("#prompt", "a calming piano piece, about thirty seconds")
    page.click("#compose")
    page.wait_for_selector(".draft-card")


class TestTheFrontDoorIsTheOneTheMachineHas:
    def test_a_conductor_is_announced_before_the_brief_is_typed(
        self, conducted: Studio, page: Any
    ) -> None:
        page.goto(conducted.url)
        page.wait_for_function(
            "() => !document.getElementById('conductor-text').textContent.startsWith('Checking')"
        )
        # The identifier is the one the client reports, not one the page guessed.
        assert "scripted-conductor" in page.locator("#conductor-text").inner_text()
        assert page.locator("#compose").inner_text() == "Open a session"

    def test_no_conductor_means_the_one_shot_door_and_says_so(
        self, studio: Studio, page: Any
    ) -> None:
        page.goto(studio.url)
        page.wait_for_function(
            "() => document.getElementById('conductor-text').textContent === 'No conductor'"
        )
        assert page.locator("#compose").inner_text() == "Create piece"
        assert "composed once" in page.locator("#compose-kicker").inner_text()


class TestTheDraftsAreOnScreen:
    def test_the_conductor_s_candidates_become_cards(self, conducted: Studio, page: Any) -> None:
        _open_a_session(conducted, page)
        # The script drafts two, so the page shows two.
        assert page.locator(".draft-card").count() == 2
        assert page.locator(".draft-card").first.locator(".draft-name").inner_text() == "Draft 1"
        # ...and the session is a place you can come back to.
        assert page.url.find("#/session/") != -1

    def test_a_card_carries_its_sketch_its_numbers_and_its_misses(
        self, conducted: Studio, page: Any
    ) -> None:
        _open_a_session(conducted, page)
        card = page.locator(".draft-card").first
        # The audio points at the session's own sketch route.
        src = card.locator("audio").get_attribute("src")
        assert src.startswith("/sessions/")
        assert src.endswith("/ogg")
        assert page.request.get(conducted.url + src).ok

        # All thirteen measurements, every time — a card that showed only the
        # misses could not be told from one whose critics never ran.
        assert card.locator(".metric").count() == 13
        # And the misses are marked rather than merely listed somewhere.
        missed = card.locator(".metric.miss").count()
        findings = card.locator(".finding").count()
        assert missed == findings
        if findings:
            assert "How the engine would move it" in card.locator(".finding").first.inner_text()
        else:
            assert card.locator(".clean-note").count() == 1

    def test_the_spec_the_conductor_read_is_shown(self, conducted: Studio, page: Any) -> None:
        _open_a_session(conducted, page)
        chips = page.locator("#spec-chips").inner_text()
        assert "calming" in chips
        assert "30 sec" in chips


class TestTheConductorLogIsReadable:
    def test_each_turn_shows_what_it_said_and_what_it_called(
        self, conducted: Studio, page: Any
    ) -> None:
        _open_a_session(conducted, page)
        turn = page.locator(".turn").first
        assert "drafting two candidates" in turn.locator(".turn-narration").inner_text()
        called = [
            turn.locator(".call").nth(i).inner_text() for i in range(turn.locator(".call").count())
        ]
        assert [name.split(" ·")[0] for name in called] == [
            "parse_brief",
            "draft",
            "sketch",
            "sketch",
        ]

    def test_the_digest_shown_is_the_one_the_server_sent(
        self, conducted: Studio, page: Any
    ) -> None:
        _open_a_session(conducted, page)
        session_id = page.url.split("#/session/")[1]
        served = page.request.get(f"{conducted.url}/sessions/{session_id}").json()
        assert page.locator("#digest").text_content() == served["digest"]


class TestTheUserCanAct:
    def test_a_verdict_sticks_to_the_card_and_reaches_the_session(
        self, conducted: Studio, page: Any
    ) -> None:
        _open_a_session(conducted, page)
        card = page.locator(".draft-card").first
        card.get_by_role("button", name="Keep this one").click()
        page.wait_for_selector(".draft-card:first-child button.vote.on")

        session_id = page.url.split("#/session/")[1]
        served = page.request.get(f"{conducted.url}/sessions/{session_id}").json()
        assert [v["value"] for v in served["verdicts"]] == ["like"]

    def test_publishing_queues_the_job_and_hands_over_to_the_progress_view(
        self, conducted: Studio, page: Any
    ) -> None:
        _open_a_session(conducted, page)
        page.locator(".draft-card").first.get_by_role("button", name="Publish").click()
        page.wait_for_selector("#status-panel:visible")

        jobs = list(conducted.jobs.list_all())
        assert len(jobs) == 1
        assert "JOB" in page.locator("#job-id").inner_text()
        # A session publishes once, so the buttons stop offering it.
        page.wait_for_function(
            "() => Array.from(document.querySelectorAll('.draft-card button'))"
            "        .filter(b => b.textContent === 'Publish')"
            "        .every(b => b.disabled)"
        )
        assert page.locator(".draft-card.is-published .tag.ok").count() > 0

    def test_asking_the_conductor_takes_another_turn(self, conducted: Studio, page: Any) -> None:
        _open_a_session(conducted, page)
        conducted.model.replies.append(
            ChatResult(
                content="Trying one more with a sparser bass.",
                tool_calls=(ToolCall("draft", {"n": 1}),),
            )
        )
        page.fill("#revise", "try something with a sparser bass")
        page.click("#revise-send")
        page.wait_for_function("() => document.querySelectorAll('.draft-card').length === 3")
        assert page.locator(".turn").count() == 2
        assert page.locator("#revise").input_value() == ""

    def test_a_turn_whose_model_call_failed_says_so_rather_than_going_quiet(
        self, conducted: Studio, page: Any
    ) -> None:
        _open_a_session(conducted, page)
        # The script is empty now, so the next turn's model call fails.
        page.fill("#revise", "make it louder")
        page.click("#revise-send")
        page.wait_for_selector(".turn-error")
        assert "ran no tools" in page.locator(".turn-error").inner_text()
