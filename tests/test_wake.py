"""Wake word matching, including the ways speech recognition mangles it."""

from __future__ import annotations

import pytest

from jarvis.wake import WakeWord, normalise, similar

CONFIG = {"wake": {"enabled": True, "words": ["jarvis", "hey jarvis"]}}


@pytest.fixture
def wake():
    return WakeWord.from_config(CONFIG)


def test_wake_word_alone_wakes_with_nothing_to_answer(wake):
    # Empty string, not None: it was addressed, it just was not asked anything.
    assert wake.find("Jarvis") == ""
    assert wake.find("Jarvis.") == ""


def test_the_question_survives_the_wake_word(wake):
    assert wake.find("Jarvis, what time is it?") == "what time is it?"
    assert wake.find("Hey Jarvis, set a timer") == "set a timer"


def test_unaddressed_speech_is_ignored(wake):
    assert wake.find("what time is it?") is None
    assert wake.find("I was watching television all evening") is None


@pytest.mark.parametrize("heard", [
    "Javis, what time is it",
    "jarvis what time is it",
    "JARVIS! what time is it",
    "Jarvis - what time is it",
])
def test_mishearings_of_the_wake_word_still_wake_it(wake, heard):
    """Whisper does not spell it right every time, and a wake word that needs
    perfect transcription is a wake word that does not work."""
    assert wake.find(heard) == "what time is it"


@pytest.mark.parametrize("heard", [
    "the harvest was good this year",
    "service is slow today",
    "carve this pumpkin for me",
    "the javelin landed short",
])
def test_similar_sounding_words_do_not_wake_it(wake, heard):
    """All of these come out of ordinary speech. None is a summons."""
    assert wake.find(heard) is None


def test_wake_word_may_follow_a_false_start(wake):
    assert wake.find("um, Jarvis, what time is it?") == "what time is it?"


def test_wake_word_late_in_a_sentence_is_not_a_summons(wake):
    """"I was telling Bob about Jarvis yesterday" is talk *about* it."""
    assert wake.find("I was telling Bob all about Jarvis yesterday") is None


def test_longest_phrase_wins(wake):
    """"hey jarvis" must be consumed whole, not leave "hey" on the question."""
    assert wake.find("hey jarvis how are you") == "how are you"


def test_disabled_passes_everything_through():
    off = WakeWord.from_config({"wake": {"enabled": False}})
    assert off.find("what time is it?") == "what time is it?"


def test_normalise_strips_punctuation_and_case():
    assert normalise("JARVIS,") == "jarvis"
    assert normalise("don't") == "don't"


def test_similar_is_tight_enough_to_be_useful():
    assert similar("jarvis", "jarvis")
    assert similar("jarvis", "javis")
    assert not similar("jarvis", "harvest")
    assert not similar("jarvis", "service")


# --- how the voice loop uses it ---

@pytest.fixture
def session(tmp_path, monkeypatch):
    """A real VoiceSession with no hardware and no network.

    Nothing in __init__ opens a device - Microphone and Speaker only touch
    PortAudio on open(), and the whisper model loads on first use - so the
    genuine object can be built and its wake logic exercised directly.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    from jarvis.assistant import Assistant, VoiceSession

    config = {
        "_root": tmp_path,
        "model": {"id": "claude-opus-5", "effort": "low", "max_tokens": 4096},
        "audio": {"sample_rate": 16000, "frame_ms": 20},
        "vad": {}, "stt": {}, "tts": {"voice_id": "v", "model_id": "m"},
        "memory": {"db": "test.db"},
        "assistant": {"name": "Jarvis", "greeting": "", "farewells": ["goodbye jarvis"]},
        "wake": {"enabled": True, "words": ["jarvis"], "stay_awake_s": 45,
                 "acknowledgement": "Yes?"},
    }
    return VoiceSession(Assistant(config), config, console=False)


def test_starts_asleep(session):
    assert not session.awake
    assert session._address("what time is it") is None


def test_the_wake_word_opens_the_window(session):
    assert session._address("jarvis what time is it") == "what time is it"
    session._stay_awake()
    # Follow-ups inside the window need no wake word.
    assert session.awake
    assert session._address("and tomorrow?") == "and tomorrow?"


def test_the_window_expires(session, monkeypatch):
    import time as time_module
    session._stay_awake()
    assert session.awake

    real = time_module.monotonic()
    monkeypatch.setattr(time_module, "monotonic", lambda: real + 46)
    assert not session.awake
    assert session._address("still there?") is None


def test_sleeping_closes_the_window_immediately(session):
    session._stay_awake()
    assert session.awake
    session._sleep()
    assert not session.awake


def test_disabling_the_wake_word_means_always_awake(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    from jarvis.assistant import Assistant, VoiceSession
    config = {
        "_root": tmp_path,
        "model": {"id": "claude-opus-5", "effort": "low", "max_tokens": 4096},
        "audio": {"sample_rate": 16000, "frame_ms": 20},
        "vad": {}, "stt": {}, "tts": {"voice_id": "v", "model_id": "m"},
        "memory": {"db": "test.db"}, "assistant": {"name": "Jarvis"},
        "wake": {"enabled": False},
    }
    session = VoiceSession(Assistant(config), config, console=False)
    assert session.awake
    assert session._address("what time is it") == "what time is it"
