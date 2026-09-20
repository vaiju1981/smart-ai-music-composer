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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import uvicorn

import saimc.jobs.worker
from saimc.jobs.api import create_app
from saimc.jobs.storage import JobStorage
from saimc.session.store import SessionStorage

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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def studio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Studio]:
    """Serve the real app on a loopback port for the length of one test.

    Hermetic in the two ways that matter: the broker is never touched
    (`enqueue_job` is replaced, as every other job test does), and the three
    storages are under `tmp_path`, so a test's jobs and sessions cannot
    outlive it or reach the developer's own `var/`.
    """
    monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", lambda job_id, **_: f"rq:{job_id}")
    app = create_app(
        jobs_root=tmp_path / "jobs",
        sessions_root=tmp_path / "sessions",
        preferences_root=tmp_path / "preferences",
    )
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
            # import-time DeprecationWarning is an error under this
            # project's `filterwarnings` — raised on the server thread,
            # where it reads as a thread crash rather than as the
            # third-party deprecation it is.
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
        yield Studio(
            url=f"http://127.0.0.1:{port}",
            jobs=app.state.job_storage,
            sessions=app.state.session_storage,
            app=app,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)


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
