"""Sentence chunking and speech cleanup."""

from __future__ import annotations

from jarvis.text import SentenceChunker, strip_for_speech


def stream(chunker: SentenceChunker, text: str, size: int = 3) -> list[str]:
    """Feed text in small pieces, the way the API delivers it."""
    out = []
    for i in range(0, len(text), size):
        out.extend(chunker.feed(text[i:i + size]))
    return out


def test_emits_each_sentence_as_it_completes():
    chunker = SentenceChunker()
    assert chunker.feed("Hello there. ") == ["Hello there."]
    assert chunker.feed("How are") == []
    assert chunker.feed(" you? ") == ["How are you?"]


def test_survives_arbitrary_chunk_boundaries():
    chunker = SentenceChunker()
    got = stream(chunker, "One thing. Then another! And a third? ")
    assert got == ["One thing.", "Then another!", "And a third?"]


def test_does_not_split_on_abbreviations():
    chunker = SentenceChunker()
    assert chunker.feed("Ask Dr. ") == []
    assert chunker.feed("Chandra about it. ") == ["Ask Dr. Chandra about it."]


def test_does_not_split_inside_a_decimal():
    chunker = SentenceChunker()
    assert chunker.feed("It costs 3.50 today. ") == ["It costs 3.50 today."]


def test_keeps_trailing_quotes_and_brackets_with_the_sentence():
    chunker = SentenceChunker()
    assert chunker.feed('She said "no." ') == ['She said "no."']


def test_breaks_a_long_unpunctuated_run_at_a_clause():
    """A reply with no full stops must still start playing."""
    chunker = SentenceChunker(soft_limit=60)
    text = ("first we do the thing, then we do the other thing, "
            "and after that a third thing entirely")
    pieces = stream(chunker, text)
    assert pieces
    assert all(len(p) < 80 for p in pieces)
    assert pieces[0].endswith(",")


def test_flush_returns_the_unterminated_tail():
    chunker = SentenceChunker()
    chunker.feed("No full stop here")
    assert chunker.flush() == "No full stop here"
    assert chunker.flush() is None


def test_nothing_is_lost_across_a_whole_reply():
    chunker = SentenceChunker()
    text = "First. Second! Third? Trailing bit"
    pieces = stream(chunker, text)
    tail = chunker.flush()
    if tail:
        pieces.append(tail)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")


def test_strips_markdown_that_would_be_read_aloud():
    assert strip_for_speech("**Bold** and _italic_") == "Bold and italic"
    assert strip_for_speech("- one\n- two") == "one two"
    assert strip_for_speech("## Heading") == "Heading"
    assert strip_for_speech("Use `ls -la` here") == "Use ls -la here"
    assert strip_for_speech("See [the docs](http://x.y)") == "See the docs"
    assert strip_for_speech("Done ✅ 🎉") == "Done"


def test_strip_leaves_ordinary_prose_alone():
    prose = "It's about twenty past four, and the answer is no."
    assert strip_for_speech(prose) == prose


def test_tts_client_does_not_depend_on_import_order(monkeypatch):
    """The ElevenLabs SDK reads its key into an import-time default argument.

    Importing jarvis.tts before .env is loaded used to yield a client with no
    credentials at all, failing later as a 401 that looked like a bad key.
    """
    import jarvis.tts

    captured = {}
    monkeypatch.setattr(jarvis.tts, "ElevenLabs",
                        lambda **kw: captured.update(kw) or object())
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_from_env")

    jarvis.tts.Voice({"tts": {}})
    assert captured["api_key"] == "sk_from_env"
