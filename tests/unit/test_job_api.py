"""Unit tests for the FastAPI job routes (no live server, no RQ)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import saimc.jobs.api
from saimc.jobs.api import create_app
from saimc.jobs.state import JobState
from saimc.jobs.storage import ArtifactRecord, JobStorage
from saimc.spec import CompositionSpec, Mood


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Hermetic: never touch a live broker. The enqueue contract (JSON
    # payload with primitive args) is the Valkey gate's job.
    monkeypatch.setattr(saimc.jobs.api, "enqueue_job", lambda job_id: f"rq:{job_id}")
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
        monkeypatch.setattr(saimc.jobs.api, "enqueue_job", lambda job_id: enqueued.append(job_id))
        create = client.post("/jobs", json={"prompt": "x"}).json()
        assert enqueued == [create["job_id"]]

    def test_broker_outage_returns_503(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(job_id: str) -> str:
            raise ConnectionError("broker down")

        monkeypatch.setattr(saimc.jobs.api, "enqueue_job", fail)
        client = TestClient(create_app(jobs_root=tmp_path), raise_server_exceptions=False)
        resp = client.post("/jobs", json={"prompt": "x"})
        assert resp.status_code == 503
        assert "retry" in resp.json()["detail"]

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
        text = resp.text
        # The page drives the same JSON routes the API exposes.
        assert "/jobs" in text
        assert "artifact/" in text
        assert 'id="prompt"' in text  # prompt box
        assert 'id="progress-bar"' in text  # job status
        assert "audio_ogg" in text  # preview
        assert "animation" in text  # preview
        assert "download" in text  # downloads

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

    def test_meta_instruments_come_from_the_registry(self, client: TestClient) -> None:
        """Adding an instrument to INSTRUMENT_PROGRAMS surfaces it in /meta."""
        from saimc.render.instruments import INSTRUMENT_PROGRAMS

        meta = client.get("/meta").json()
        assert meta["instruments"] == sorted(INSTRUMENT_PROGRAMS)

    def test_spec_instrumentations_are_all_registered(self) -> None:
        """Drift guard: the spec vocabulary and the render registry must
        agree in both directions. Spec-only names would fail at render
        time; registry-only names would never be requestable."""
        from saimc.render.instruments import (
            INSTRUMENT_FAMILIES,
            INSTRUMENT_PROGRAMS,
            soundfont_for_instrument,
        )
        from saimc.spec import Instrument

        expected = {member.value for member in Instrument}
        assert set(INSTRUMENT_PROGRAMS) == expected
        assert set(INSTRUMENT_FAMILIES) == expected
        for instrument in expected:
            soundfont_for_instrument(instrument)  # must resolve without error
