"""The session's HTTP surface, and `POST /jobs` as an alias over it.

The harness is the tools' one, one layer out: the same scripted model standing
in for the adapter, the same fake at the renderer's boundary, and `TestClient`
over an app built on `tmp_path` roots. What is new is that the *request* is the
unit — a turn runs inside it — so the assertions are about what a caller gets
back and what a refusal costs them.

Three properties carry the weight.

- **A refusal is a status code that names the recovery.** A session that has
  published refuses everything that could add candidates, and says which job it
  published; a broker outage leaves the work intact and points at `/finalize`.
  Neither is a 500, because neither is a server fault.
- **The harness decides only what the harness may decide.** A lone draft is not
  a choice so the harness publishes it; several are a choice so the harness
  stops and says so. Both are asserted through the API, because that is where a
  caller meets them.
- **`/jobs` still answers.** Every path through the alias — no model, nothing
  published, an explicit seed, a session that published — returns a `job_id`,
  and exactly one job exists per request on each of them.

The compositions are real and short (30 s); only the outer edges are faked, as
in `test_session_tools.py`: the model, FluidSynth and the font resolver, and the
broker.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import saimc.jobs.worker
import saimc.session.api
import saimc.session.tools
from saimc.jobs.api import create_app
from saimc.jobs.storage import JobStorage
from saimc.llm.base import (
    ChatRequest,
    ChatResult,
    LLMError,
    ParseRequest,
    ParseResult,
    ToolCall,
)
from saimc.render.audio import AudioArtifact
from saimc.session import tools
from saimc.session.api import http_from_refusal
from saimc.session.models import Session
from saimc.session.store import SessionStorage
from saimc.session.tools import ToolRefusal
from saimc.spec import CompositionSpec, Mood, SpecError

_SPEC = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=5)

# The ids `draft` hands out are positional and therefore knowable at script
# time, which is what lets a script name the draft it wants sketched.
_FIRST = "draft-0"


class ScriptedModel:
    """A model with a script: one reply per turn, in order, and then silence.

    It satisfies both protocols on one object, which is what every real build
    does (C1) and what the app's single state slot assumes. `parse` is
    separate from `chat` because they are separate protocols: `parse_brief`
    goes through `ctx.llm` and the conductor through the client, and one test
    here is about what happens when only the second is on offer.

    Running out of script is not silence-by-accident: it answers with a named
    error, so a test that scripts too few replies fails as an
    `llm_unreachable` turn rather than as a mysterious empty one.
    """

    def __init__(self, *, spec: CompositionSpec | None = _SPEC) -> None:
        self.spec = spec
        self.replies: list[ChatResult] = []
        self.requests: list[ChatRequest] = []
        self.parse_requests: list[ParseRequest] = []

    async def parse(self, request: ParseRequest) -> ParseResult:
        self.parse_requests.append(request)
        if self.spec is None:
            return ParseResult(
                parser_source="llm",
                error=SpecError(error_code="llm_unreachable", message="no host", stage="parsing"),
            )
        return ParseResult(parser_source="llm", spec=self.spec)

    async def chat(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        if not self.replies:
            return ChatResult(error=LLMError("llm_unreachable", "the script is empty"))
        return self.replies.pop(0)

    async def aclose(self) -> None:
        return None


class ChatOnlyModel:
    """A client that can hold a conversation and cannot read a brief.

    `ChatClient` without `LLMClient`, which is the degradation `_parser` is
    there to handle: every deterministic tool still runs and `parse_brief`
    refuses with a reason, recorded in the turn like any other answer.
    """

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        return ChatResult(
            content="Let me read the brief first.",
            tool_calls=(ToolCall("parse_brief", {}),),
        )

    async def aclose(self) -> None:
        return None


def _reply(*calls: ToolCall, content: str = "") -> ChatResult:
    return ChatResult(content=content, tool_calls=calls)


def _call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


def _prompt(model: ScriptedModel, index: int) -> str:
    """The user message of one request the conductor made."""
    return model.requests[index].messages[1].content


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """A stand-in for the FluidSynth pass, writing the files it claims to.

    Copied in spirit from `test_session_tools.py`: faked at the renderer's
    boundary so the paths the draft records are the paths a real render
    writes, and so a change to that naming shows up as a 410 here.
    """
    calls: list[dict[str, Any]] = []

    def _fake(plan: Any, **kwargs: Any) -> AudioArtifact:
        out_dir: Path = kwargs["out_dir"]
        job_id: str = kwargs["job_id"]
        calls.append({"plan": plan, "job_id": job_id, **kwargs})
        (out_dir / f"{job_id}.mid").write_bytes(b"MThd")
        wav = out_dir / "audio.wav"
        wav.write_bytes(b"RIFF")
        ogg = out_dir / "audio.ogg"
        ogg.write_bytes(b"OggS")
        return AudioArtifact(
            primary_path=wav,
            primary_container="wav",
            primary_codec="pcm_s16le",
            primary_sha256="a" * 64,
            primary_size_bytes=4,
            ogg_path=ogg,
            ogg_codec="opus",
            ogg_sha256="b" * 64,
            ogg_size_bytes=4,
        )

    monkeypatch.setattr(tools, "render_sketch", _fake)
    monkeypatch.setattr(tools, "resolve_job_soundfont", lambda _voices: Path("/tmp/one.sf2"))
    return calls


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "jobs", tmp_path / "sessions"


@pytest.fixture
def model() -> ScriptedModel:
    return ScriptedModel()


@pytest.fixture
def client(roots: tuple[Path, Path], model: ScriptedModel, monkeypatch: pytest.MonkeyPatch):
    """An app with a model, over `tmp_path`, touching no broker."""
    monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", lambda job_id, **_: f"rq:{job_id}")
    app = create_app(
        jobs_root=roots[0],
        sessions_root=roots[1],
        session_llm=model,
    )
    return TestClient(app)


def _drafts(client: TestClient, model: ScriptedModel, n: int = 2, **body: Any) -> dict[str, Any]:
    """Open a session that drafts `n` candidates and does not publish.

    Two drafts is what makes it *not* publish — one would be a lone candidate,
    which the harness masters. `auto_finalize` is off in every caller that only
    wants the drafts, so the continuations do not burn scripted replies.
    """
    model.replies = [_reply(_call("parse_brief"), _call("draft", n=n))]
    response = client.post(
        "/sessions",
        json={"brief": "something for a rainy day", "auto_finalize": False, **body},
    )
    assert response.status_code == 201, response.text
    return response.json()


class TestCreateSession:
    def test_a_brief_becomes_a_session_with_its_first_turn(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        body = _drafts(client, model)
        assert body["brief"] == "something for a rainy day"
        assert [turn["trigger"] for turn in body["turns"]] == ["brief"]
        assert [call["name"] for call in body["turns"][0]["calls"]] == ["parse_brief", "draft"]
        assert body["spec"]["mood"] == Mood.CALMING.value
        assert len(body["drafts"]) == 2

    def test_a_specific_brief_publishes_without_being_asked(
        self, client: TestClient, model: ScriptedModel, roots: tuple[Path, Path]
    ) -> None:
        """The one-shot promise, kept by the harness rather than by a rule table.

        One candidate is not a choice, so the pass ends by mastering it — and
        because the conductor asked for no continuation and no second turn, the
        whole request cost exactly one model call.
        """
        model.replies = [_reply(_call("parse_brief"), _call("draft", n=1))]
        body = client.post("/sessions", json={"brief": "30s of calm piano in C"}).json()

        job_id = body["finalized_job_id"]
        assert job_id is not None
        assert [turn["trigger"] for turn in body["turns"]] == ["brief"]
        assert len(model.requests) == 1

        job = JobStorage(roots[0]).get(job_id)
        assert job.parser_source == "from-spec"
        assert job.input_plan is not None
        assert job.seed == body["drafts"][0]["seed"]

    def test_an_open_brief_ends_with_candidates_and_a_bounded_loop(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        """Several candidates are a choice, and choosing is not the harness's.

        The continuations run because the pass published nothing and left a
        decision — and they stop at `MAX_AUTO_TURNS`, which is the bound that
        keeps a turn's length the harness's rather than the model's.
        """
        model.replies = [
            _reply(_call("parse_brief"), _call("draft", n=2)),
            _reply(content="Two candidates, then."),
            _reply(content="Still nothing to publish."),
        ]
        body = client.post("/sessions", json={"brief": "something for a rainy day"}).json()

        assert body["finalized_job_id"] is None
        assert [turn["trigger"] for turn in body["turns"]] == ["brief"] + ["auto"] * (
            saimc.session.api.MAX_AUTO_TURNS
        )
        assert len(body["drafts"]) == 2
        assert "publish it" in _prompt(model, 1)

    def test_auto_finalize_off_takes_one_turn_and_no_continuations(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        model.replies = [_reply(_call("parse_brief"), _call("draft", n=1))]
        body = client.post(
            "/sessions",
            json={"brief": "30s of calm piano in C", "auto_finalize": False},
        ).json()

        assert body["finalized_job_id"] is None
        assert [turn["trigger"] for turn in body["turns"]] == ["brief"]
        assert len(model.requests) == 1

    def test_a_failed_model_call_ends_the_pass(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        """No script at all: the turn fails, and nothing is invented to follow it."""
        body = client.post("/sessions", json={"brief": "anything"}).json()

        assert body["finalized_job_id"] is None
        assert body["drafts"] == []
        assert len(body["turns"]) == 1
        assert body["turns"][0]["llm_error"]["error_code"] == "llm_unreachable"
        assert body["turns"][0]["calls"] == []

    def test_a_failed_continuation_is_not_followed_by_another(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        model.replies = [_reply(_call("parse_brief"), _call("draft", n=2))]
        body = client.post("/sessions", json={"brief": "something for a rainy day"}).json()

        assert [turn["trigger"] for turn in body["turns"]] == ["brief", "auto"]
        assert body["turns"][1]["llm_error"] is not None
        assert len(body["drafts"]) == 2

    def test_a_chat_only_model_degrades_instead_of_crashing(
        self, roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`parse_brief` needs `LLMClient`; a client that is only a `ChatClient` refuses.

        The refusal is recorded like any other answer — the conductor is told
        what is missing and can say so — which is what makes the two protocols
        a real boundary rather than a wiring assumption.
        """
        monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", lambda job_id, **_: f"rq:{job_id}")
        app = create_app(jobs_root=roots[0], sessions_root=roots[1], session_llm=ChatOnlyModel())
        body = TestClient(app).post("/sessions", json={"brief": "anything"}).json()

        calls = body["turns"][0]["calls"]
        assert [call["name"] for call in calls] == ["parse_brief"]
        assert calls[0]["outcome"] == "refused"
        assert calls[0]["error_code"] == "llm_not_configured"
        assert body["spec"] is None

    def test_a_session_without_a_model_is_refused_before_it_is_written(
        self, roots: tuple[Path, Path]
    ) -> None:
        """503 rather than an empty session the caller has to notice."""
        app = create_app(jobs_root=roots[0], sessions_root=roots[1])
        response = TestClient(app).post("/sessions", json={"brief": "anything"})

        assert response.status_code == 503
        assert "no language model is configured" in response.json()["detail"]
        assert list(SessionStorage(roots[1]).list_all()) == []

    def test_the_digest_names_what_has_been_published(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        """The response carries the text the next turn would be given.

        Asserted through the published line rather than by re-deriving the
        digest here, which would only restate the implementation: the claim is
        that a caller can read what the conductor will be told.
        """
        unpublished = _drafts(client, model)
        assert "published: not yet" in unpublished["digest"]

        model.replies = [_reply(_call("parse_brief"), _call("draft", n=1))]
        published = client.post("/sessions", json={"brief": "30s of calm piano in C"}).json()
        assert f"published: job {published['finalized_job_id']}" in published["digest"]


class TestPublishing:
    """Who starts the render, and what a broker that refuses costs.

    The conductor may publish from inside a turn — `finalize` is one of its
    tools — and the harness may publish after one. Both end in the same place,
    and both have to survive a broker that will not take the job.
    """

    def test_a_turn_that_publishes_ends_the_pass(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        """A continuation after the piece is mastered would have nothing to decide.

        The candidate the conductor publishes is one of two, so the harness
        would not have chosen it — the conductor did, which is the whole point
        of it being the conductor's tool.
        """
        model.replies = [
            _reply(_call("parse_brief"), _call("draft", n=2)),
            _reply(_call("finalize", draft_id="draft-1"), content="This one."),
        ]
        body = client.post("/sessions", json={"brief": "something for a rainy day"}).json()

        assert [turn["trigger"] for turn in body["turns"]] == ["brief", "auto"]
        assert body["finalized_job_id"] is not None
        assert body["drafts"][1]["draft_id"] == "draft-1"

    def test_a_broker_outage_while_the_harness_publishes_is_503(
        self, client: TestClient, model: ScriptedModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The session did not publish, and the 503 says which session to retry on."""
        model.replies = [_reply(_call("parse_brief"), _call("draft", n=1))]

        def fail(job_id: str, **_: object) -> str:
            raise ConnectionError("broker down")

        monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", fail)
        response = client.post("/sessions", json={"brief": "30s of calm piano in C"})

        assert response.status_code == 503
        sessions = list(SessionStorage(client.app.state.session_storage.root).list_all())
        assert len(sessions) == 1
        assert sessions[0].finalized_job_id is None
        assert sessions[0].session_id in response.json()["detail"]

    def test_a_broker_outage_during_a_message_turn_is_503(
        self, client: TestClient, model: ScriptedModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model.replies = [_reply(_call("parse_brief"))]
        created = client.post(
            "/sessions", json={"brief": "something", "auto_finalize": False}
        ).json()

        def fail(job_id: str, **_: object) -> str:
            raise ConnectionError("broker down")

        model.replies = [_reply(_call("draft", n=1))]
        monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", fail)
        response = client.post(
            f"/sessions/{created['session_id']}/message", json={"message": "go on then"}
        )

        assert response.status_code == 503
        assert created["session_id"] in response.json()["detail"]


class TestGetSession:
    def test_a_session_is_readable_by_its_id(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        created = _drafts(client, model)
        fetched = client.get(f"/sessions/{created['session_id']}").json()
        assert fetched["session_id"] == created["session_id"]
        assert fetched["drafts"] == created["drafts"]

    def test_an_unknown_session_is_404(self, client: TestClient) -> None:
        response = client.get("/sessions/does-not-exist")
        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

    def test_a_malformed_id_is_400_rather_than_404(self, client: TestClient) -> None:
        """A typo and a stale link are different situations with different advice."""
        response = client.get("/sessions/a%5Cb")
        assert response.status_code == 400
        assert "not a session id" in response.json()["detail"]


class TestMessage:
    def test_a_message_reaches_the_conductor_in_the_users_words(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        created = _drafts(client, model)
        model.replies = [_reply(content="Understood.")]
        response = client.post(
            f"/sessions/{created['session_id']}/message",
            json={"message": "make it slower", "auto_finalize": False},
        )

        assert response.status_code == 200
        assert [turn["trigger"] for turn in response.json()["turns"]] == ["brief", "message"]
        assert "The user says: make it slower" in _prompt(model, 1)

    def test_a_message_that_leaves_one_draft_publishes_it(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        model.replies = [_reply(_call("parse_brief"))]
        created = client.post(
            "/sessions", json={"brief": "something", "auto_finalize": False}
        ).json()
        assert created["drafts"] == []

        model.replies = [_reply(_call("draft", n=1))]
        body = client.post(
            f"/sessions/{created['session_id']}/message", json={"message": "go on then"}
        ).json()

        assert body["finalized_job_id"] is not None
        assert len(body["drafts"]) == 1

    def test_a_message_after_publishing_is_refused_and_names_the_job(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        """Refused rather than accepted, because the answer could never be mastered."""
        model.replies = [_reply(_call("parse_brief"), _call("draft", n=1))]
        created = client.post("/sessions", json={"brief": "30s of calm piano in C"}).json()
        job_id = created["finalized_job_id"]
        calls_before = len(model.requests)

        response = client.post(
            f"/sessions/{created['session_id']}/message", json={"message": "again"}
        )

        assert response.status_code == 409
        assert job_id in response.json()["detail"]
        assert len(model.requests) == calls_before


class TestVerdict:
    def test_a_verdict_is_recorded_and_survives_a_reload(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        created = _drafts(client, model)
        body = client.post(
            f"/sessions/{created['session_id']}/verdict",
            json={"draft_id": _FIRST, "value": "like", "feedback": "warmer, please"},
        ).json()

        assert body["verdicts"] == [
            {
                "draft_id": _FIRST,
                "at": body["verdicts"][0]["at"],
                "value": "like",
                "feedback": "warmer, please",
            }
        ]
        reloaded = client.get(f"/sessions/{created['session_id']}").json()
        assert reloaded["verdicts"] == body["verdicts"]

    def test_verdicts_accumulate_rather_than_replace(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        """A user changing their mind is the most informative thing they do.

        A record keeping only the latest verdict would throw exactly that away,
        and it is also what makes each `(plan_hash, verdict)` pair available to
        the slow loop.
        """
        created = _drafts(client, model)
        url = f"/sessions/{created['session_id']}/verdict"
        client.post(url, json={"draft_id": _FIRST, "value": "dislike"})
        body = client.post(url, json={"draft_id": _FIRST, "value": "like"}).json()

        assert [verdict["value"] for verdict in body["verdicts"]] == ["dislike", "like"]

    def test_a_verdict_takes_no_turn_but_is_read_by_the_next_one(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        created = _drafts(client, model)
        calls_before = len(model.requests)
        client.post(
            f"/sessions/{created['session_id']}/verdict",
            json={"draft_id": _FIRST, "feedback": "less busy in the left hand"},
        )
        assert len(model.requests) == calls_before

        model.replies = [_reply(content="Noted.")]
        client.post(
            f"/sessions/{created['session_id']}/message",
            json={"message": "carry on", "auto_finalize": False},
        )
        assert "less busy in the left hand" in _prompt(model, calls_before)

    def test_an_empty_verdict_is_refused_in_the_records_own_words(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        created = _drafts(client, model)
        response = client.post(
            f"/sessions/{created['session_id']}/verdict", json={"draft_id": _FIRST}
        )

        assert response.status_code == 422
        assert "a verdict must carry a like, a dislike, or feedback" in response.json()["detail"]

    def test_a_verdict_for_an_unknown_draft_lists_the_ones_it_has(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        created = _drafts(client, model)
        response = client.post(
            f"/sessions/{created['session_id']}/verdict",
            json={"draft_id": "draft-99", "value": "like"},
        )

        assert response.status_code == 404
        detail = response.json()["detail"]
        assert "draft-99" in detail
        assert "draft-0" in detail
        assert "draft-1" in detail


class TestFinalize:
    def test_finalizing_a_draft_publishes_it_and_queues_the_render(
        self, client: TestClient, model: ScriptedModel, roots: tuple[Path, Path]
    ) -> None:
        created = _drafts(client, model)
        response = client.post(
            f"/sessions/{created['session_id']}/finalize", json={"draft_id": _FIRST}
        )

        assert response.status_code == 202
        body = response.json()
        assert body["draft_id"] == _FIRST
        assert body["state"] == "queued"
        assert (
            client.get(f"/sessions/{created['session_id']}").json()["finalized_job_id"]
            == body["job_id"]
        )

        job = JobStorage(roots[0]).get(body["job_id"])
        assert job.input_spec is not None
        assert job.input_spec.mood == Mood.CALMING
        assert job.input_plan is not None
        assert job.seed == created["drafts"][0]["seed"]

    def test_finalizing_needs_no_model_in_the_process(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        """Publishing consults nothing that thinks: the draft is already composed.

        The model slot is emptied on the way in, so this is the claim itself
        rather than an inference from the fact that the fake was not asked —
        a process with no configuration at all can still master a draft.
        """
        created = _drafts(client, model)
        client.app.state.session_llm = None

        response = client.post(
            f"/sessions/{created['session_id']}/finalize", json={"draft_id": _FIRST}
        )
        assert response.status_code == 202

    def test_finalizing_twice_is_refused(self, client: TestClient, model: ScriptedModel) -> None:
        created = _drafts(client, model)
        url = f"/sessions/{created['session_id']}/finalize"
        first = client.post(url, json={"draft_id": _FIRST})
        assert first.status_code == 202

        second = client.post(url, json={"draft_id": "draft-1"})
        assert second.status_code == 409
        assert first.json()["job_id"] in second.json()["detail"]

    def test_finalizing_an_unknown_draft_lists_the_ones_it_has(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        created = _drafts(client, model)
        response = client.post(
            f"/sessions/{created['session_id']}/finalize", json={"draft_id": "draft-99"}
        )

        assert response.status_code == 404
        assert "draft-1" in response.json()["detail"]

    def test_a_broker_outage_leaves_the_session_unpublished(
        self, client: TestClient, model: ScriptedModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """503, the work intact, and the message naming the session to retry on."""
        created = _drafts(client, model)
        session_id = created["session_id"]

        def fail(job_id: str, **_: object) -> str:
            raise ConnectionError("broker down")

        monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", fail)
        response = client.post(f"/sessions/{session_id}/finalize", json={"draft_id": _FIRST})

        assert response.status_code == 503
        assert session_id in response.json()["detail"]
        assert client.get(f"/sessions/{session_id}").json()["finalized_job_id"] is None


class TestUndo:
    def test_undo_steps_back_over_the_turn_that_was_taken(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        created = _drafts(client, model)
        response = client.post(f"/sessions/{created['session_id']}/undo")

        assert response.status_code == 200
        body = response.json()
        assert body["turns"] == []
        assert body["drafts"] == []
        assert client.get(f"/sessions/{created['session_id']}").json()["turns"] == []

    def test_undo_with_nothing_behind_it_is_refused(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        created = _drafts(client, model)
        url = f"/sessions/{created['session_id']}/undo"
        assert client.post(url).status_code == 200

        response = client.post(url)
        assert response.status_code == 409
        assert "no earlier state" in response.json()["detail"]

    def test_undo_after_publishing_is_refused(
        self, client: TestClient, model: ScriptedModel
    ) -> None:
        """A render cannot be unwound, so the session cannot pretend it did not happen."""
        model.replies = [_reply(_call("parse_brief"), _call("draft", n=1))]
        created = client.post("/sessions", json={"brief": "30s of calm piano in C"}).json()

        response = client.post(f"/sessions/{created['session_id']}/undo")
        assert response.status_code == 409
        assert "cannot be unpublished" in response.json()["detail"]


class TestSketch:
    @pytest.fixture
    def sketched(self, client: TestClient, model: ScriptedModel, rendered) -> dict[str, Any]:
        """A session holding one draft with a sketch and one without."""
        model.replies = [
            _reply(
                _call("parse_brief"),
                _call("draft", n=2),
                _call("sketch", draft_id=_FIRST),
            )
        ]
        return client.post(
            "/sessions", json={"brief": "something for a rainy day", "auto_finalize": False}
        ).json()

    def test_a_sketch_is_served_as_ogg_and_points_at_itself(
        self, client: TestClient, sketched: dict[str, Any]
    ) -> None:
        sketch = sketched["drafts"][0]["sketch"]
        assert sketch is not None
        assert sketch["url"] == f"/sessions/{sketched['session_id']}/sketch/{_FIRST}/ogg"
        assert sketch["ogg_size_bytes"] == 4

        response = client.get(sketch["url"])
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/ogg"
        assert response.content == b"OggS"

    def test_the_wav_is_served_too(self, client: TestClient, sketched: dict[str, Any]) -> None:
        """The render writes two files and both are useful; neither is decoration."""
        response = client.get(f"/sessions/{sketched['session_id']}/sketch/{_FIRST}/wav")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content == b"RIFF"

    def test_an_unknown_kind_is_404_and_lists_the_kinds(
        self, client: TestClient, sketched: dict[str, Any]
    ) -> None:
        response = client.get(f"/sessions/{sketched['session_id']}/sketch/{_FIRST}/flac")
        assert response.status_code == 404
        assert "ogg" in response.json()["detail"]

    def test_a_draft_with_no_sketch_is_404(
        self, client: TestClient, sketched: dict[str, Any]
    ) -> None:
        assert sketched["drafts"][1]["sketch"] is None
        response = client.get(f"/sessions/{sketched['session_id']}/sketch/draft-1/ogg")
        assert response.status_code == 404
        assert "no sketch yet" in response.json()["detail"]

    def test_an_unknown_draft_is_404(self, client: TestClient, sketched: dict[str, Any]) -> None:
        response = client.get(f"/sessions/{sketched['session_id']}/sketch/draft-9/ogg")
        assert response.status_code == 404

    def test_a_sketch_whose_file_is_gone_is_410(
        self, client: TestClient, sketched: dict[str, Any], roots: tuple[Path, Path]
    ) -> None:
        """Rendered and then lost on disk is a different thing to be told than 'no such sketch'."""
        sessions = SessionStorage(roots[1])
        draft = sessions.get(sketched["session_id"]).draft(_FIRST)
        assert draft.sketch is not None
        sessions.sketch_path(sketched["session_id"], draft.sketch.ogg_path).unlink()

        response = client.get(f"/sessions/{sketched['session_id']}/sketch/{_FIRST}/ogg")
        assert response.status_code == 410


class TestUnconfiguredStorage:
    """A misconfigured app says so, rather than crashing inside a route.

    Both mirrors of `jobs/api._storage`'s guard, which is the one place this
    module repeats rather than imports: `jobs/api` imports *this* module for the
    alias, so the shared helper cannot point the other way.
    """

    def test_a_missing_session_store_is_500(self, client: TestClient) -> None:
        client.app.state.session_storage = None
        response = client.get("/sessions/anything")
        assert response.status_code == 500
        assert response.json()["detail"] == "Session storage not configured."

    def test_a_missing_job_store_is_500(self, client: TestClient, model: ScriptedModel) -> None:
        """Reached through a route that needs both stores, so the job guard fires."""
        created = _drafts(client, model)
        client.app.state.job_storage = None
        response = client.post(
            f"/sessions/{created['session_id']}/finalize", json={"draft_id": _FIRST}
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "Job storage not configured."


class TestRefusalTranslation:
    """The two mapped refusals, and the unmapped one that must not be guessed."""

    @staticmethod
    def _session() -> Session:
        now = datetime.now(UTC)
        return Session(session_id="s1", created_at=now, updated_at=now, brief="b")

    def test_a_mapped_refusal_becomes_its_status(self) -> None:
        exc = http_from_refusal(ToolRefusal("already_finalized", "no"), self._session())
        assert exc.status_code == 409
        assert "s1" in exc.detail
        assert "no" in exc.detail

    def test_an_unmapped_refusal_is_a_500_carrying_its_own_words(self) -> None:
        """A code with no mapping is a wiring mistake, not a 4xx the caller can fix."""
        exc = http_from_refusal(ToolRefusal("nonsense", "something odd"), self._session())
        assert exc.status_code == 500
        assert "nonsense" in exc.detail
        assert "s1" in exc.detail
