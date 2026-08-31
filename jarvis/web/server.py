"""A local web front end for the voice loop.

Standard library only, deliberately. This is the piece that has to work the
same on macOS, Windows and Linux, and every dependency it does not have is one
that cannot fail to install on somebody else's machine. Server-sent events -
one long-lived response per browser - carry the loop's state outward; three
small POSTs carry control back. No websockets, no framework.

The audio never touches the browser. Microphone, whisper, ElevenLabs and the
speaker all stay in this process; the page is a view and a set of buttons.
"""

from __future__ import annotations

import json
import logging
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from jarvis.assistant import Assistant, VoiceSession
from jarvis.events import EventBus

log = logging.getLogger("jarvis.web")

UI = Path(__file__).resolve().parent / "ui.html"


class JarvisApp:
    """Owns the assistant, the voice loop thread, and the event bus."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.events = EventBus()
        self.assistant = Assistant(config)
        # console=False: the browser is the interface, and duplicating the
        # transcript into a terminal nobody is reading just obscures the logs.
        self.session = VoiceSession(self.assistant, config,
                                    events=self.events, console=False)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the voice loop. False if it was already running."""
        with self._lock:
            if self.running:
                return False
            self._thread = threading.Thread(target=self._run, name="voice",
                                            daemon=True)
            self._thread.start()
            return True

    def _run(self) -> None:
        try:
            self.session.run()
        except Exception:
            # Already logged, and published as an error event for the page.
            # The session's own `finally` publishes the stopped state, so
            # there is nothing to add here.
            pass

    def stop(self) -> bool:
        if not self.running:
            return False
        self.session.stop()
        return True

    def interrupt(self) -> None:
        self.session.speaker.interrupt()

    def snapshot(self) -> dict[str, Any]:
        """Everything a page needs to render itself on first load."""
        return {
            "running": self.running,
            "state": self.session.state,
            "name": self.config["assistant"]["name"],
            "model": self.config["model"]["id"],
            "voice_id": self.config["tts"]["voice_id"],
            "whisper": self.config["stt"]["model"],
            "facts": self.assistant.memory.facts(),
            "turns": self.assistant.memory.turns_after(
                self.assistant.session_id, 0, limit=50),
        }

    def shutdown(self) -> None:
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.events.close()
        self.assistant.finish()


class Handler(BaseHTTPRequestHandler):
    server_version = "Jarvis"

    def __init__(self, *args, app: JarvisApp, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        # One line per SSE keep-alive would drown the real logs.
        log.debug("%s - %s", self.address_string(), fmt % args)

    # --- helpers ---

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The page is served to, and used by, exactly one local browser.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    # --- routes ---

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(self.app.snapshot())
        elif self.path == "/api/events":
            self._stream_events()
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        actions = {
            "/api/start": lambda: {"started": self.app.start()},
            "/api/stop": lambda: {"stopped": self.app.stop()},
            "/api/interrupt": lambda: (self.app.interrupt(), {"ok": True})[1],
        }
        action = actions.get(self.path)
        if action is None:
            self._json({"error": "not found"}, 404)
            return
        self._json(action())

    def _stream_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        with self.app.events.subscribe() as subscription:
            try:
                self._sse({"type": "state", "value": self.app.session.state})
                for event in subscription.stream(timeout=15.0):
                    if event is None:
                        # A comment frame. Without it a dropped connection is
                        # invisible until the next real event, which during a
                        # quiet conversation could be minutes.
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        continue
                    self._sse(event.to_dict())
            except (BrokenPipeError, ConnectionResetError):
                log.debug("browser disconnected")

    def _sse(self, payload: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()


def serve(config: dict[str, Any], host: str = "127.0.0.1", port: int = 8765
          ) -> tuple[ThreadingHTTPServer, JarvisApp]:
    """Build the app and its server. Caller runs and shuts them down.

    Bound to loopback: this process holds a microphone, an Anthropic key and an
    ElevenLabs key, and its control endpoints are unauthenticated. It has no
    business accepting connections from the network.
    """
    app = JarvisApp(config)
    server = ThreadingHTTPServer((host, port), partial(Handler, app=app))
    server.daemon_threads = True
    return server, app
