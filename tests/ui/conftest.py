"""A browser, a live server, and the page under test.

`docs/open-items.md` B1 recorded the reason the session workspace was not
built: `src/saimc/jobs/static/index.html` had **no test witness in the repo
at all** — nothing read it, and its behaviour (fetch, render, poll) could not
be observed because the project had neither a browser nor jsdom. This package
is that witness, and it is the thing that had to land before the page could
grow.

It is a real browser against a real server: uvicorn on an ephemeral port with
the app `create_served_app` would build, minus the model, and Chromium driving
the page the way a person does. Nothing is stubbed on the page's side — the
fetches are real HTTP to the real routes — so a test here fails for the same
reasons a user would see a broken page.

The browser is shared with `render-service`, which already depends on
Playwright 1.61.0 for the notation renderer; pinning the Python binding to the
same version means one browser revision on disk for the whole repo rather than
two. The suite self-skips when that browser is absent, in the shape
`tests/integration/test_sketch_render.py` established for a missing FFmpeg: a
gate that cannot run says so rather than passing quietly.
"""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import uvicorn

import saimc.jobs.worker
import saimc.session.tools as session_tools
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
from saimc.session.store import SessionStorage
from saimc.spec import CompositionSpec, Mood

try:  # pragma: no cover - the import is the thing being reported on
    from playwright import sync_api as playwright_api
except ImportError:  # pragma: no cover
    playwright_api = None  # type: ignore[assignment]


def _unavailable() -> str | None:
    """Why this suite cannot run here, or None when it can.

    Checked without launching anything: `executable_path` is the browser
    Playwright would start, and its absence is the whole of the "no browser"
    case. Launching one to find out would cost a second on every full run,
    including the runs that never reach these tests.
    """
    if playwright_api is None:
        return "Playwright is not installed; `pip install -e '.[ui]'` to run the UI suite"
    try:
        with playwright_api.sync_playwright() as p:
            if not Path(p.chromium.executable_path).exists():
                raise FileNotFoundError(p.chromium.executable_path)
    except Exception as exc:  # pragma: no cover - environment, not logic
        return f"Playwright has no Chromium to drive ({exc}); run `playwright install chromium`"
    return None


REQUIRE_BROWSER_ENV = "SAIMC_UI_REQUIRE_BROWSER"
"""Set it when the browser is known to be installed and a skip would be a lie."""

_SKIP_REASON = _unavailable()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip this directory's tests when there is no browser, and say why.

    The shape `tests/integration/test_sketch_render.py` uses for a missing
    release-gate FFmpeg: a suite that cannot run reports the reason and the
    command that fixes it, rather than passing quietly and being mistaken for
    coverage.
    """
    if _SKIP_REASON is None:
        return
    if os.environ.get(REQUIRE_BROWSER_ENV):
        # CI installs the browser one step earlier, so a skip there would be a
        # green build over a suite that never ran. The caller that knows the
        # browser is present says so, and the skip becomes a failure.
        pytest.exit(f"{REQUIRE_BROWSER_ENV} is set but {_SKIP_REASON}", returncode=1)
    here = Path(__file__).parent
    mark = pytest.mark.skip(reason=_SKIP_REASON)
    for item in items:
        if here in Path(str(item.fspath)).parents:
            item.add_marker(mark)


@dataclass
class Studio:
    """The running app: where to point a browser, and what it is serving."""

    url: str
    jobs: JobStorage
    sessions: SessionStorage
    app: Any
    model: Any


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ScriptedModel:
    """A conductor with a script: one reply per turn, in order, then a refusal.

    It satisfies `LLMClient` and `ChatClient` on one object, which is what a
    real build has and what the app's single model slot assumes. The default
    script is the shape a first turn actually takes — read the brief, draft
    twice, sketch both — so a test that only wants a session to look at gets
    one without writing a script of its own.

    Running out of script answers with a named error rather than silence, so a
    test that scripts too few turns fails as an `llm_unreachable` turn instead
    of as a page that mysteriously stopped.
    """

    model_identifier = "scripted-conductor"

    def __init__(self) -> None:
        self.spec = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=5)
        self.replies: list[ChatResult] = list(self.opening_script())
        self.requests: list[ChatRequest] = []

    @staticmethod
    def opening_script() -> tuple[ChatResult, ...]:
        return (
            ChatResult(
                content="Reading the brief, then drafting two candidates to compare.",
                tool_calls=(
                    ToolCall("parse_brief", {}),
                    ToolCall("draft", {"n": 2}),
                    ToolCall("sketch", {"draft_id": "draft-0"}),
                    ToolCall("sketch", {"draft_id": "draft-1"}),
                ),
            ),
        )

    async def parse(self, request: ParseRequest) -> ParseResult:
        return ParseResult(parser_source="llm", spec=self.spec)

    async def chat(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        if not self.replies:
            return ChatResult(error=LLMError("llm_unreachable", "the script is empty"))
        return self.replies.pop(0)

    async def aclose(self) -> None:
        return None


@pytest.fixture
def sketching(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stand in for the FluidSynth pass, writing the files it claims to.

    Faked at the renderer's boundary, as `test_session_tools.py` does, so the
    paths a draft records are the paths a real render writes — which is what
    makes the page's `<audio src>` a real request to a real route rather than
    a URL nobody resolves. The bytes are not audio; what is under test is that
    the sketch is reachable, not that Chromium can decode it.
    """
    calls: list[dict[str, Any]] = []

    def _fake(plan: Any, **kwargs: Any) -> AudioArtifact:
        out_dir: Path = kwargs["out_dir"]
        job_id: str = kwargs["job_id"]
        calls.append({"job_id": job_id, **kwargs})
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

    monkeypatch.setattr(session_tools, "render_sketch", _fake)
    monkeypatch.setattr(
        session_tools, "resolve_job_soundfont", lambda _voices: Path("/tmp/one.sf2")
    )
    return calls


@contextmanager
def _serving(app: Any) -> Iterator[str]:
    """Run `app` on a loopback port for the length of the block."""
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
            # The page uses SSE, never a socket. Leaving the WebSocket
            # protocol on makes uvicorn import `websockets.legacy`, whose
            # import-time DeprecationWarning is an error under this project's
            # `filterwarnings` — raised on the server thread, where it reads
            # as a thread crash rather than as the third-party deprecation it
            # is.
            ws="none",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # `started` flips after the socket is listening, so a page opened on the
    # next line cannot race the bind.
    for _ in range(600):
        if server.started:
            break
        threading.Event().wait(0.02)
    else:  # pragma: no cover - only on a machine that cannot bind at all
        pytest.fail("the test server never started")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: Any) -> Any:
    """The real app, with its three storages under `tmp_path` and no broker.

    Hermetic in the two ways that matter: `enqueue_job` is replaced, as every
    other job test does, and nothing written here can outlive the test or
    reach the developer's own `var/`.
    """
    monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", lambda job_id, **_: f"rq:{job_id}")
    return create_app(
        jobs_root=tmp_path / "jobs",
        sessions_root=tmp_path / "sessions",
        preferences_root=tmp_path / "preferences",
        session_llm=model,
    )


@pytest.fixture
def studio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Studio]:
    """The app as a machine with no model has it: the one-shot front door."""
    app = _app(tmp_path, monkeypatch, None)
    with _serving(app) as url:
        yield Studio(
            url=url,
            jobs=app.state.job_storage,
            sessions=app.state.session_storage,
            app=app,
            model=None,
        )


@pytest.fixture
def conducted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sketching: list[dict[str, Any]]
) -> Iterator[Studio]:
    """The app with a conductor: the session front door, on a scripted model.

    The model is scripted rather than live for the reason every other session
    test scripts one — a turn's content is the thing under test, and a real
    host would make it the thing least under control. Everything below the
    model is real: the engine composes, the linter lints, the scorecard
    measures, and the page reads what they produced.
    """
    model = ScriptedModel()
    app = _app(tmp_path, monkeypatch, model)
    with _serving(app) as url:
        yield Studio(
            url=url,
            jobs=app.state.job_storage,
            sessions=app.state.session_storage,
            app=app,
            model=model,
        )


@pytest.fixture
def browser() -> Iterator[Any]:
    with playwright_api.sync_playwright() as p:
        instance = p.chromium.launch()
        try:
            yield instance
        finally:
            instance.close()


@pytest.fixture
def page(browser: Any) -> Iterator[Any]:
    """A fresh page whose console errors are a test failure, not a mystery.

    A page that throws on load still renders most of its markup, so a test
    asserting on text can pass over a broken page. Collecting the errors and
    asserting on them at teardown is what makes "the page works" mean it.
    """
    context = browser.new_context(viewport={"width": 1400, "height": 1000})
    page = context.new_page()
    failures: list[str] = []
    page.on("pageerror", lambda exc: failures.append(str(exc)))
    page.on(
        "console",
        lambda msg: failures.append(msg.text) if msg.type == "error" else None,
    )
    try:
        yield page
        assert not failures, f"the page reported errors: {failures}"
    finally:
        context.close()
