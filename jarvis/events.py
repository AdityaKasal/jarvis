"""A tiny publish/subscribe bus, so the voice loop can be watched.

The loop runs on its own thread and knows nothing about who is listening: the
console printer and every connected browser are just subscribers. Each gets its
own bounded queue, so a browser tab that stops reading cannot block the audio
loop - it drops events instead, which for a live view is the right trade.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "at": self.at, **self.data}


class EventBus:
    def __init__(self, queue_size: int = 200):
        self._subscribers: list[queue.Queue[Optional[Event]]] = []
        self._lock = threading.Lock()
        self._queue_size = queue_size
        self._last: dict[str, Event] = {}

    def publish(self, type: str, **data: Any) -> Event:
        event = Event(type, data)
        with self._lock:
            self._last[type] = event
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                # A subscriber that has stopped reading loses events rather
                # than stalling the loop that produced them.
                pass
        return event

    def latest(self, type: str) -> Optional[Event]:
        with self._lock:
            return self._last.get(type)

    def subscribe(self) -> "Subscription":
        q: queue.Queue[Optional[Event]] = queue.Queue(maxsize=self._queue_size)
        with self._lock:
            self._subscribers.append(q)
        return Subscription(self, q)

    def _unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def close(self) -> None:
        """Wake every subscriber so its stream can end."""
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


class Subscription:
    def __init__(self, bus: EventBus, q: queue.Queue[Optional[Event]]):
        self._bus = bus
        self._queue = q

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._bus._unsubscribe(self._queue)

    def stream(self, timeout: float = 1.0) -> Iterator[Optional[Event]]:
        """Yield events as they arrive; yield None on idle so callers can
        send a keep-alive and notice a dropped connection."""
        while True:
            try:
                event = self._queue.get(timeout=timeout)
            except queue.Empty:
                yield None
                continue
            if event is None:
                return
            yield event
