"""Unit tests for the FastAPI job routes (no live server, no RQ)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import saimc.jobs.api
import saimc.jobs.worker
from saimc.jobs.api import create_app
from saimc.jobs.state import JobState
from saimc.jobs.storage import ArtifactRecord, JobStorage
from saimc.llm.base import (
    ChatRequest,
    ChatResult,
    LLMError,
    ParseRequest,
    ParseResult,
    ToolCall,
)
from saimc.spec import CompositionSpec, Mood


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Hermetic: never touch a live broker. The enqueue contract (JSON
    # payload with primitive args) is the Valkey gate's job.
    monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", lambda job_id, **_: f"rq:{job_id}")
    app = create_app(jobs_root=tmp_path)
    return TestClient(app)


def _spec(**overrides) -> CompositionSpec:
    base = {"mood": Mood.CALMING.value}
    base.update(overrides)
    return CompositionSpec.model_validate(base)


class TestCreateJob:
    def test_create_returns_202_and_job_id(self, client: TestClient) -> None:
        resp = client.post("/jobs", json={"prompt": "calming piano music"})
        assert resp.status_code == 202
        body = resp.json()
        assert body["state"] == "queued"
        assert body["input_prompt"] == "calming piano music"
        assert body["progress"] == 0.0
        assert "job_id" in body

    def test_create_enqueues_the_job(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        enqueued: list[str] = []
        monkeypatch.setattr(
            saimc.jobs.worker, "enqueue_job", lambda job_id, **_: enqueued.append(job_id)
        )
        create = client.post("/jobs", json={"prompt": "x"}).json()
        assert enqueued == [create["job_id"]]

    def test_broker_outage_returns_503(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(job_id: str, **_: object) -> str:
            raise ConnectionError("broker down")

        monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", fail)
        client = TestClient(create_app(jobs_root=tmp_path), raise_server_exceptions=False)
        resp = client.post("/jobs", json={"prompt": "x"})
        assert resp.status_code == 503
        assert "was not queued" in resp.json()["detail"]
        jobs = list(JobStorage(tmp_path).list_all())
        assert len(jobs) == 1
        assert jobs[0].state == JobState.FAILED
        assert jobs[0].error is not None
        assert jobs[0].error.error_code == "queue_unavailable"

    def test_empty_prompt_rejected(self, client: TestClient) -> None:
        resp = client.post("/jobs", json={"prompt": ""})
        assert resp.status_code == 422

    def test_missing_prompt_rejected(self, client: TestClient) -> None:
        resp = client.post("/jobs", json={})
        assert resp.status_code == 422


class TestCreateFromSpec:
    def test_internal_endpoint_accepts_spec(self, client: TestClient) -> None:
        spec = _spec()
        resp = client.post("/jobs/from-spec", json={"spec": spec.model_dump(mode="json")})
        assert resp.status_code == 202
        body = resp.json()
        assert body["parser_source"] == "from-spec"
        assert body["artifacts"] == {}

    def test_internal_endpoint_rejects_invalid_spec(self, client: TestClient) -> None:
        resp = client.post("/jobs/from-spec", json={"spec": {"mood": "angsty"}})
        assert resp.status_code == 422


class TestGetJob:
    def test_returns_persisted_job(self, client: TestClient) -> None:
        create = client.post("/jobs", json={"prompt": "x"}).json()
        resp = client.get(f"/jobs/{create['job_id']}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == create["job_id"]
        assert body["state"] == "queued"

    def test_unknown_job_returns_404(self, client: TestClient) -> None:
        resp = client.get("/jobs/does-not-exist")
        assert resp.status_code == 404


class TestJobEvents:
    def test_terminal_job_streams_one_update_and_closes(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        create = client.post("/jobs", json={"prompt": "x"}).json()
        job_id = create["job_id"]
        storage = JobStorage(tmp_path)
        job = storage.get(job_id)
        job.state = JobState.COMPLETE
        job.progress = 1.0
        job.current_stage = JobState.COMPLETE.value
        storage.save(job)

        with client.stream("GET", f"/jobs/{job_id}/events") as resp:
            payload = "".join(resp.iter_text())

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["cache-control"] == "no-cache, no-transform"
        assert payload.startswith("data: ")
        assert f'"job_id":"{job_id}"' in payload
        assert '"state":"complete"' in payload

    def test_unknown_job_stream_returns_404(self, client: TestClient) -> None:
        assert client.get("/jobs/does-not-exist/events").status_code == 404


class TestArtifactEndpoint:
    def test_streams_existing_artifact(self, client: TestClient, tmp_path: Path) -> None:
        create = client.post("/jobs", json={"prompt": "x"}).json()
        job_id = create["job_id"]
        storage = JobStorage(tmp_path)
        storage.ensure_artifact_dir(job_id).joinpath("audio.wav").write_bytes(b"RIFF\x00\x00")

        job = storage.get(job_id)
        storage.attach_artifact(
            job,
            ArtifactRecord(
                kind="audio",
                container="wav",
                codec="pcm_s24le",
                path="audio.wav",
                sha256="0" * 64,
                size_bytes=6,
            ),
        )
        storage.save(job)
        resp = client.get(f"/jobs/{job_id}/artifact/audio")
        assert resp.status_code == 200
        assert resp.content == b"RIFF\x00\x00"
        assert resp.headers["content-type"].startswith("audio/")

    def test_missing_artifact_returns_404(self, client: TestClient) -> None:
        create = client.post("/jobs", json={"prompt": "x"}).json()
        resp = client.get(f"/jobs/{create['job_id']}/artifact/audio")
        assert resp.status_code == 404

    def test_unknown_kind_returns_404(self, client: TestClient) -> None:
        create = client.post("/jobs", json={"prompt": "x"}).json()
        resp = client.get(f"/jobs/{create['job_id']}/artifact/video")
        assert resp.status_code == 404


class TestCancel:
    def test_cancel_queued_succeeds(self, client: TestClient) -> None:
        create = client.post("/jobs", json={"prompt": "x"}).json()
        resp = client.post(f"/jobs/{create['job_id']}/cancel")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "cancelled"

    def test_cancel_complete_returns_409(self, client: TestClient, tmp_path: Path) -> None:
        create = client.post(
            "/jobs/from-spec", json={"spec": _spec().model_dump(mode="json")}
        ).json()
        job_id = create["job_id"]
        storage = JobStorage(tmp_path)
        job = storage.get(job_id)
        job.state = JobState.COMPLETE
        storage.save(job)
        resp = client.post(f"/jobs/{job_id}/cancel")
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "cancel_not_allowed"

    def test_cancel_rendering_returns_409(self, client: TestClient, tmp_path: Path) -> None:
        create = client.post(
            "/jobs/from-spec", json={"spec": _spec().model_dump(mode="json")}
        ).json()
        job_id = create["job_id"]
        storage = JobStorage(tmp_path)
        job = storage.get(job_id)
        job.state = JobState.RENDERING_AUDIO
        storage.save(job)
        resp = client.post(f"/jobs/{job_id}/cancel")
        assert resp.status_code == 409

    def test_cancel_unknown_returns_404(self, client: TestClient) -> None:
        resp = client.post("/jobs/does-not-exist/cancel")
        assert resp.status_code == 404


class TestIndexPage:
    """The web UI route (roadmap §2: prompt box, job status, preview, downloads)."""

    def test_index_serves_the_single_page_ui(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert resp.headers["cache-control"] == "no-store, max-age=0"
        text = resp.text
        # The page drives the same JSON routes the API exposes.
        assert "/jobs" in text
        assert "artifact/" in text
        assert 'id="prompt"' in text  # prompt box
        assert 'id="progress-bar"' in text  # job status
        assert "audio_ogg" in text  # preview
        assert "animation" in text  # preview
        assert "download" in text  # downloads
        assert "EventSource" in text  # live progress does not rely on timers

    def test_index_does_not_shadow_api_docs(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200


class TestListJobs:
    def test_lists_jobs_newest_first(self, client: TestClient) -> None:
        first = client.post("/jobs", json={"prompt": "calming piano"}).json()
        second = client.post("/jobs", json={"prompt": "electrifying piano"}).json()
        listed = client.get("/jobs").json()
        ids = [j["job_id"] for j in listed]
        assert ids.count(first["job_id"]) == 1
        assert ids.index(second["job_id"]) < ids.index(first["job_id"])

    def test_limit_caps_the_listing(self, client: TestClient) -> None:
        for _ in range(3):
            client.post("/jobs", json={"prompt": "calming piano"})
        listed = client.get("/jobs", params={"limit": 2}).json()
        assert len(listed) == 2

    def test_list_includes_spec_and_seed_once_parsed(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        create = client.post("/jobs", json={"prompt": "calming piano"}).json()
        storage = JobStorage(tmp_path)
        job = storage.get(create["job_id"])
        job.input_spec = _spec(seed=7)
        job.seed = 7  # parse_stage mirrors spec.seed onto the job
        storage.save(job)
        listed = client.get("/jobs").json()
        mine = next(j for j in listed if j["job_id"] == create["job_id"])
        assert mine["input_spec"]["mood"] == "calming"
        assert mine["seed"] == 7


class TestCreateJobSeed:
    def test_explicit_seed_is_persisted(self, client: TestClient, tmp_path: Path) -> None:
        body = client.post("/jobs", json={"prompt": "calming", "seed": 123}).json()
        assert body["seed"] == 123
        job = JobStorage(tmp_path).get(body["job_id"])
        assert job.seed == 123

    def test_negative_seed_rejected(self, client: TestClient) -> None:
        resp = client.post("/jobs", json={"prompt": "calming", "seed": -1})
        assert resp.status_code == 422


class TestMeta:
    def test_meta_lists_vocabulary(self, client: TestClient) -> None:
        meta = client.get("/meta").json()
        assert set(meta["moods"]) == {"calming", "electrifying", "sleep"}
        assert "4/4" in meta["time_signatures"]
        assert "piano" in meta["instruments"]
        assert meta["duration_seconds"] == {"min": 30, "max": 600, "default": 180}

    def test_meta_lists_ensemble_vocabulary(self, client: TestClient) -> None:
        """Roles, default ensembles, and font-only instruments surface
        for the UI to advertise what a spec's instrumentation accepts."""
        from saimc.compose.ensemble import SCALAR_BASS, SCALAR_HARMONY
        from saimc.render.instruments import FONT_ONLY_INSTRUMENTS
        from saimc.spec import ROLE_ORDER

        meta = client.get("/meta").json()
        assert meta["roles"] == [r.value for r in ROLE_ORDER]
        assert meta["default_ensembles"] == {
            mood: {"harmony": SCALAR_HARMONY[mood], "bass": SCALAR_BASS[mood]}
            for mood in SCALAR_HARMONY
        }
        assert meta["dedicated_font_only"] == sorted(FONT_ONLY_INSTRUMENTS)

    def test_meta_instruments_come_from_the_registry(self, client: TestClient) -> None:
        """Adding an instrument to SUPPORTED_INSTRUMENTS surfaces it in /meta."""
        from saimc.render.instruments import SUPPORTED_INSTRUMENTS

        meta = client.get("/meta").json()
        assert meta["instruments"] == sorted(SUPPORTED_INSTRUMENTS)

    def test_spec_instrumentations_are_all_registered(self) -> None:
        """Drift guard: the spec vocabulary and the render registry must
        agree in both directions. Spec-only names would fail at render
        time; registry-only names would never be requestable."""
        from saimc.render.instruments import (
            INSTRUMENT_FAMILIES,
            SUPPORTED_INSTRUMENTS,
            soundfont_for_instrument,
        )
        from saimc.spec import Instrument

        expected = {member.value for member in Instrument}
        assert set(SUPPORTED_INSTRUMENTS) == expected
        assert set(INSTRUMENT_FAMILIES) == expected
        for instrument in expected:
            soundfont_for_instrument(instrument)  # must resolve without error


class TestHealth:
    def test_health_exposes_worker_readiness(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            saimc.jobs.api,
            "_service_health",
            lambda: {"status": "ready", "broker": "ready", "workers": 1, "queue_depth": 2},
        )
        assert client.get("/health").json() == {
            "status": "ready",
            "broker": "ready",
            "workers": 1,
            "queue_depth": 2,
        }

    def test_worker_pid_liveness_rejects_missing_process(self) -> None:
        assert saimc.jobs.api._local_worker_process_alive(None) is False
        assert saimc.jobs.api._local_worker_process_alive(2_147_483_647) is False

    def test_worker_pid_liveness_accepts_current_process(self) -> None:
        import os

        assert saimc.jobs.api._local_worker_process_alive(os.getpid()) is True


class _Scripted:
    """A model with a script, satisfying both protocols on one object.

    Small enough to keep local: this module needs a conductor that asks for a
    named tool and nothing else, and `test_session_api.py` has the version that
    also records what it was asked.
    """

    def __init__(self) -> None:
        self.replies: list[ChatResult] = []
        self.requests: list[ChatRequest] = []

    async def parse(self, request: ParseRequest) -> ParseResult:
        return ParseResult(
            parser_source="llm",
            spec=CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=5),
        )

    async def chat(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        if not self.replies:
            return ChatResult(error=LLMError("llm_unreachable", "the script is empty"))
        return self.replies.pop(0)

    async def aclose(self) -> None:
        return None


def _reply(*calls: ToolCall, content: str = "") -> ChatResult:
    return ChatResult(content=content, tool_calls=calls)


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(name=name, arguments=dict(arguments))


class TestJobsAsASessionAlias:
    """`POST /jobs` over a session, and the three ways it falls back.

    The alias is the whole of what "the harness replaces one-shot" means to a
    caller who never learns there is a harness: one prompt in, one `job_id`
    out, whichever path answered. `test_create_*` above already covers the
    model-less path — it is what every test in this module takes — so what is
    here is the session path and the two deliberate ways past it.
    """

    @staticmethod
    def _app(roots: Path, model: object) -> TestClient:
        return TestClient(create_app(jobs_root=roots, session_llm=model))

    @staticmethod
    def _model(*replies: ChatResult) -> _Scripted:
        model = _Scripted()
        model.replies = list(replies)
        return model

    @staticmethod
    def _queue(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Record every `enqueue_job` call, and return the ids it was given.

        A job *record* is written before the broker is asked, and a record whose
        enqueue never happened sits in `queued` for ever, rendering nothing —
        which is the one failure a caller cannot see from the response, because
        its shape is identical either way. So the fallback is asserted to have
        asked, not merely to have written.
        """
        enqueued: list[str] = []

        def _record(job_id: str, **_: object) -> str:
            enqueued.append(job_id)
            return f"rq:{job_id}"

        monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", _record)
        return enqueued

    def test_a_prompt_through_a_session_returns_the_job_it_published(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The piece the conductor mastered, not a prompt queued for the worker.

        `parser_source` is what tells the two paths apart from the outside: a
        job the alias published already carries its spec, because the session
        composed and measured it before publishing.
        """
        enqueued = self._queue(monkeypatch)
        model = self._model(_reply(_call("parse_brief"), _call("draft", n=1)))
        client = self._app(tmp_path, model)

        body = client.post("/jobs", json={"prompt": "30s of calm piano in C"}).json()

        assert body["state"] == "queued"
        assert body["parser_source"] == "from-spec"
        assert body["input_spec"]["mood"] == "calming"
        assert len(list(JobStorage(tmp_path).list_all())) == 1
        assert enqueued == [body["job_id"]]

    def test_a_pass_that_publishes_nothing_queues_the_prompt_instead(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The alias still promises a job, and the session keeps its candidates.

        An open brief ends with several candidates and no choice made; the old
        path answers the request, and the session is left where the user can
        come back to it. Exactly one job exists, on either path.
        """
        enqueued = self._queue(monkeypatch)
        model = self._model(
            _reply(_call("parse_brief"), _call("draft", n=2)),
            _reply(content="Two candidates."),
            _reply(content="Still two."),
        )
        client = self._app(tmp_path, model)

        body = client.post("/jobs", json={"prompt": "something for a rainy day"}).json()

        assert body["parser_source"] is None
        assert body["input_spec"] is None
        assert len(list(JobStorage(tmp_path).list_all())) == 1
        assert enqueued == [body["job_id"]]

    def test_an_explicit_seed_takes_the_phase_one_path_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A seed asks for one exact piece, so there is nothing for a conductor to choose."""
        enqueued = self._queue(monkeypatch)
        model = self._model(_reply(_call("parse_brief"), _call("draft", n=1)))
        client = self._app(tmp_path, model)

        body = client.post("/jobs", json={"prompt": "anything", "seed": 7}).json()

        assert body["seed"] == 7
        assert body["parser_source"] is None
        assert model.requests == []
        assert enqueued == [body["job_id"]]

    def test_a_broker_outage_on_the_session_path_is_503(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal the session raised is translated, not swallowed into a fallback.

        Falling back here would queue a *second* job for a piece that was
        already composed, which is the one thing the alias must not do.
        """
        model = self._model(_reply(_call("parse_brief"), _call("draft", n=1)))
        app = create_app(jobs_root=tmp_path, session_llm=model)

        def fail(job_id: str, **_: object) -> str:
            raise ConnectionError("broker down")

        monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", fail)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/jobs", json={"prompt": "anything"})

        assert response.status_code == 503
        assert "not queued" in response.json()["detail"]

    def test_the_alias_names_a_session_and_a_real_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`create_served_app` is the only factory that reads configuration.

        Every other caller gets inert defaults, which is what keeps the tests
        above off the network — so the wiring itself is asserted here rather
        than assumed.
        """
        monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", lambda job_id, **_: f"rq:{job_id}")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
        monkeypatch.setenv("OLLAMA_MODEL", "a-model")

        app = saimc.jobs.api.create_served_app()

        assert app.state.session_llm is not None
        assert app.state.session_storage is not None
        assert create_app(jobs_root=tmp_path).state.session_llm is None
