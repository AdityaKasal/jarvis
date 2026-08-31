"""The streaming tool loop, against a stubbed API.

These are the parts that are awkward to check by talking to it: that sentences
reach TTS in order and before the turn is over, that tool results are returned
in one message, and that the loop stops.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis.brain import MAX_TOOL_TURNS, Brain
from jarvis.memory import Memory
from jarvis.tools import Toolbox


def usage(**kw):
    return SimpleNamespace(input_tokens=10, output_tokens=5,
                           cache_read_input_tokens=0, **kw)


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(name, args, block_id="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=block_id)


class FakeStream:
    def __init__(self, chunks, final):
        self._chunks = chunks
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return iter(self._chunks)

    def get_final_message(self):
        return self._final


class FakeClient:
    """Replays a scripted list of (text chunks, final message) turns."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = SimpleNamespace(stream=self._stream, create=self._create)

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        chunks, final = self.script.pop(0)
        return FakeStream(chunks, final)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[text_block("a summary")], usage=usage())


@pytest.fixture
def make_brain(tmp_path, monkeypatch):
    def build(script):
        client = FakeClient(script)
        monkeypatch.setattr("anthropic.Anthropic", lambda *a, **k: client)
        toolbox = Toolbox(Memory(tmp_path / "brain.db"))
        brain = Brain({"model": {"id": "claude-opus-5", "effort": "low",
                                 "max_tokens": 4096}}, toolbox)
        return brain, client, toolbox
    return build


def final(stop_reason, content):
    return SimpleNamespace(stop_reason=stop_reason, content=content,
                           usage=usage(), stop_details=None)


def test_speaks_sentences_as_they_complete(make_brain):
    brain, _, _ = make_brain([
        (["Sure. ", "It is ", "four o'clock. "],
         final("end_turn", [text_block("Sure. It is four o'clock.")])),
    ])

    spoken = []
    reply = brain.respond([], [{"role": "user", "content": "time?"}], spoken.append)

    assert spoken == ["Sure.", "It is four o'clock."]
    assert reply == "Sure. It is four o'clock."


def test_runs_a_tool_and_continues(make_brain):
    brain, client, _ = make_brain([
        (["Let me check. "],
         final("tool_use", [text_block("Let me check. "),
                            tool_block("get_current_time", {})])),
        (["It's just gone four. "],
         final("end_turn", [text_block("It's just gone four.")])),
    ])

    spoken = []
    brain.respond([], [{"role": "user", "content": "time?"}], spoken.append)

    # The filler is spoken before the tool runs, not queued behind it.
    assert spoken == ["Let me check.", "It's just gone four."]

    second_call = client.calls[1]["messages"]
    assert second_call[-2]["role"] == "assistant"
    results = second_call[-1]["content"]
    assert second_call[-1]["role"] == "user"
    assert [r["type"] for r in results] == ["tool_result"]
    assert results[0]["tool_use_id"] == "tu_1"


def test_parallel_tool_results_go_back_in_one_message(make_brain):
    """Splitting them teaches the model to stop calling tools in parallel."""
    brain, client, _ = make_brain([
        ([], final("tool_use", [
            tool_block("remember_fact", {"key": "name", "value": "Aditya"}, "a"),
            tool_block("get_current_time", {}, "b"),
        ])),
        (["Noted. "], final("end_turn", [text_block("Noted.")])),
    ])

    brain.respond([], [{"role": "user", "content": "I'm Aditya"}], lambda s: None)

    results = client.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]


def test_a_failing_tool_still_returns_a_result(make_brain):
    """A tool_use with no matching tool_result is a 400 on the next request."""
    brain, client, _ = make_brain([
        ([], final("tool_use", [tool_block("no_such_tool", {}, "x")])),
        (["Sorry. "], final("end_turn", [text_block("Sorry.")])),
    ])

    brain.respond([], [{"role": "user", "content": "hi"}], lambda s: None)

    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "x"
    assert result["is_error"] is True


def test_end_conversation_sets_the_flag(make_brain):
    brain, _, toolbox = make_brain([
        ([], final("tool_use", [tool_block("end_conversation", {"reason": "bye"})])),
        (["Goodnight. "], final("end_turn", [text_block("Goodnight.")])),
    ])

    brain.respond([], [{"role": "user", "content": "goodnight"}], lambda s: None)
    assert toolbox.should_end is True


def test_refusal_says_something_rather_than_nothing(make_brain):
    brain, _, _ = make_brain([
        ([], SimpleNamespace(
            stop_reason="refusal", content=[], usage=usage(),
            stop_details=SimpleNamespace(category="cyber", explanation=""))),
    ])

    spoken = []
    reply = brain.respond([], [{"role": "user", "content": "..."}], spoken.append)
    assert spoken and reply


def test_a_tool_loop_that_never_settles_is_cut_off(make_brain):
    brain, client, _ = make_brain([
        ([], final("tool_use", [tool_block("get_current_time", {}, f"t{i}")]))
        for i in range(MAX_TOOL_TURNS + 2)
    ])

    reply = brain.respond([], [{"role": "user", "content": "hi"}], lambda s: None)
    assert len(client.calls) == MAX_TOOL_TURNS
    assert reply  # says something rather than going silent


def test_unterminated_text_is_still_spoken(make_brain):
    brain, _, _ = make_brain([
        (["No full stop here"], final("end_turn", [text_block("No full stop here")])),
    ])
    spoken = []
    brain.respond([], [{"role": "user", "content": "hi"}], spoken.append)
    assert spoken == ["No full stop here"]
