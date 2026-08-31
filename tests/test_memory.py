"""Persistence: facts, turns, and the summary window."""

from __future__ import annotations

import pytest

from jarvis.memory import Memory


@pytest.fixture
def memory(tmp_path):
    return Memory(tmp_path / "test.db")


def test_facts_round_trip(memory):
    memory.remember("name", "Aditya")
    assert memory.facts() == {"name": "Aditya"}


def test_remembering_the_same_key_overwrites(memory):
    memory.remember("city", "Tempe")
    memory.remember("city", "Phoenix")
    assert memory.facts()["city"] == "Phoenix"


def test_fact_keys_are_normalised(memory):
    memory.remember("  Coffee Order  ", "  oat flat white  ")
    assert memory.facts() == {"coffee order": "oat flat white"}


def test_forget_reports_whether_anything_was_there(memory):
    memory.remember("k", "v")
    assert memory.forget("k") is True
    assert memory.forget("k") is False
    assert memory.facts() == {}


def test_turns_come_back_in_spoken_order(memory):
    session = memory.start_session()
    for role, content in [("user", "one"), ("assistant", "two"), ("user", "three")]:
        memory.add_turn(session, role, content)

    turns = memory.turns_after(session, 0, limit=10)
    assert [t["content"] for t in turns] == ["one", "two", "three"]


def test_the_window_keeps_the_most_recent_turns(memory):
    session = memory.start_session()
    for i in range(10):
        memory.add_turn(session, "user", f"turn {i}")

    turns = memory.turns_after(session, 0, limit=3)
    assert [t["content"] for t in turns] == ["turn 7", "turn 8", "turn 9"]


def test_summarised_turns_drop_out_of_the_window(memory):
    session = memory.start_session()
    ids = [memory.add_turn(session, "user", f"turn {i}") for i in range(6)]

    memory.set_session_summary(session, "talked about six things", ids[3])
    summary, through = memory.session_summary(session)
    assert summary == "talked about six things"
    assert through == ids[3]

    remaining = memory.turns_after(session, through, limit=10)
    assert [t["content"] for t in remaining] == ["turn 4", "turn 5"]
    assert memory.turn_count(session, through) == 2


def test_sessions_are_isolated_from_each_other(memory):
    first = memory.start_session()
    memory.add_turn(first, "user", "in the first")
    second = memory.start_session()
    memory.add_turn(second, "user", "in the second")

    assert len(memory.turns_after(second, 0, limit=10)) == 1


def test_past_summaries_carry_forward_oldest_first(memory):
    older = memory.start_session()
    memory.end_session(older, "we talked about tea")
    recent = memory.start_session()
    memory.end_session(recent, "we talked about coffee")
    current = memory.start_session()

    history = memory.previous_summaries(current, limit=5)
    assert [text for _, text in history] == ["we talked about tea",
                                             "we talked about coffee"]


def test_unsummarised_sessions_are_not_carried(memory):
    memory.end_session(memory.start_session())  # ended, but nothing was said
    current = memory.start_session()
    assert memory.previous_summaries(current, limit=5) == []
