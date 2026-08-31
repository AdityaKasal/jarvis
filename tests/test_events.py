"""The event bus that lets the browser watch the voice loop."""

from __future__ import annotations

import threading

from jarvis.events import EventBus


def drain(subscription, expected):
    got = []
    for event in subscription.stream(timeout=1.0):
        if event is None:
            break
        got.append(event)
        if len(got) >= expected:
            break
    return got


def test_a_subscriber_receives_what_is_published():
    bus = EventBus()
    with bus.subscribe() as sub:
        bus.publish("state", value="listening")
        assert [e.to_dict()["value"] for e in drain(sub, 1)] == ["listening"]


def test_every_subscriber_gets_its_own_copy():
    bus = EventBus()
    with bus.subscribe() as a, bus.subscribe() as b:
        bus.publish("turn", role="user", text="hello")
        assert drain(a, 1)[0].data["text"] == "hello"
        assert drain(b, 1)[0].data["text"] == "hello"


def test_publishing_with_nobody_listening_is_fine():
    EventBus().publish("state", value="stopped")


def test_a_stalled_subscriber_cannot_block_the_publisher():
    """A browser tab that stops reading must lose events, not wedge the audio
    loop that produces them."""
    bus = EventBus(queue_size=4)
    with bus.subscribe():
        for i in range(500):
            bus.publish("level", value=i)   # never read
    assert True  # reaching here at all is the assertion


def test_unsubscribing_stops_delivery():
    bus = EventBus()
    sub = bus.subscribe()
    sub.close()
    assert bus.subscriber_count == 0
    bus.publish("state", value="stopped")


def test_latest_remembers_the_most_recent_of_each_type():
    bus = EventBus()
    bus.publish("state", value="listening")
    bus.publish("state", value="speaking")
    bus.publish("turn", role="user", text="hi")
    assert bus.latest("state").data["value"] == "speaking"
    assert bus.latest("turn").data["text"] == "hi"
    assert bus.latest("nothing") is None


def test_close_ends_every_open_stream():
    bus = EventBus()
    sub = bus.subscribe()
    ended = threading.Event()

    def consume():
        for _ in sub.stream(timeout=5.0):
            pass
        ended.set()

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    bus.close()
    assert ended.wait(timeout=3.0), "stream did not end when the bus closed"
