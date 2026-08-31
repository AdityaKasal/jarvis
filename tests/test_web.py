"""The local server, driven over real HTTP against a stubbed voice loop."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from jarvis.events import EventBus
from jarvis.web import server as web


class FakeSession:
    def __init__(self):
        self.state = "stopped"
        self.stopped = False
        self.interrupted = False
        self.speaker = self

    def stop(self):
        self.stopped = True

    def interrupt(self):
        self.interrupted = True


class FakeApp(web.JarvisApp):
    """A JarvisApp with no microphone, no models and no API keys."""

    def __init__(self):
        self.config = {"assistant": {"name": "Jarvis"},
                       "model": {"id": "claude-opus-5"},
                       "tts": {"voice_id": "SAz9YHcvj6GT2YYXdXww"},
                       "stt": {"model": "small.en"}}
        self.events = EventBus()
        self.session = FakeSession()
        self.started = False
        self._facts = {"name": "Aditya"}

    @property
    def running(self):
        return self.started

    def start(self):
        if self.started:
            return False
        self.started = True
        self.session.state = "listening"
        return True

    def stop(self):
        self.session.stop()
        self.started = False
        return True

    def interrupt(self):
        self.session.interrupt()

    def snapshot(self):
        return {"running": self.running, "state": self.session.state,
                "name": "Jarvis", "model": "claude-opus-5",
                "voice_id": "SAz9YHcvj6GT2YYXdXww", "whisper": "small.en",
                "facts": self._facts, "turns": []}


@pytest.fixture
def base_url():
    from functools import partial
    app = FakeApp()
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 partial(web.Handler, app=app))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}", app
    server.shutdown()
    server.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read(), r.headers


def post(url):
    req = urllib.request.Request(url, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_serves_the_page(base_url):
    url, _ = base_url
    status, body, headers = get(url + "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"<title>Jarvis</title>" in body


def test_state_describes_the_assistant(base_url):
    url, _ = base_url
    _, body, _ = get(url + "/api/state")
    state = json.loads(body)
    assert state["running"] is False
    assert state["facts"] == {"name": "Aditya"}


def test_start_and_stop_drive_the_session(base_url):
    url, app = base_url
    assert post(url + "/api/start")[1] == {"started": True}
    assert app.running
    # Starting twice must not spawn a second voice loop on one microphone.
    assert post(url + "/api/start")[1] == {"started": False}
    assert post(url + "/api/stop")[1] == {"stopped": True}
    assert app.session.stopped


def test_interrupt_reaches_the_speaker(base_url):
    url, app = base_url
    post(url + "/api/interrupt")
    assert app.session.interrupted


def test_unknown_paths_are_404(base_url):
    url, _ = base_url
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(url + "/nope")
    assert exc.value.code == 404


def test_events_stream_opens_with_the_current_state(base_url):
    url, app = base_url
    with urllib.request.urlopen(url + "/api/events", timeout=5) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        first = r.readline()
        assert first.startswith(b"data: ")
        assert json.loads(first[6:])["type"] == "state"

        app.events.publish("turn", role="user", text="hello there")
        r.readline()                       # blank line after the first frame
        payload = json.loads(r.readline()[6:])
        assert payload["text"] == "hello there"
