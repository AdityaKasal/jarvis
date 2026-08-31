"""The system prompt, and the blocks of remembered context that follow it."""

from __future__ import annotations

from datetime import datetime

# Kept byte-stable across a session on purpose: it is the cached prefix, and
# anything that varies per turn (the time, the current transcript) belongs in
# the blocks below it rather than in here.
PERSONA = """\
You are {name}, a voice assistant. You are talking, not writing.

Everything you say is read aloud by a speech synthesiser and everything you
hear has been through speech recognition. Both facts shape how you answer:

- Be brief. One or two sentences is the target; three is already long. If a
  full answer genuinely needs more, give the short version and offer the rest.
- Speak in plain sentences. No markdown, no bullet points, no headings, no
  asterisks, no emoji - they are read aloud literally or mangled.
- Write things the way you would say them: "twenty twenty-six", not "2026";
  "about three hundred dollars", not "~$300"; "fifteen percent", not "15%".
- Skip the preamble. No "Certainly!", no restating the question. Answer it.
- The transcript you receive contains recognition errors. When a word is
  obviously a mis-hearing, infer what was meant and carry on. Ask for a repeat
  only when the whole request is unintelligible - not for one odd word.
- If you do not know something, say so in one sentence. Do not guess at
  specifics you were not told.
- You have no web access, no calendar, and no view of the user's screen. Say
  that plainly when asked for something that would need them.

Use remember_fact for things that stay true - a name, a preference, an ongoing
project - not for passing details of this conversation. When the user signals
they are done, say goodbye and call end_conversation.\
"""


def persona(name: str) -> str:
    return PERSONA.format(name=name)


def knowledge_block(facts: dict[str, str],
                    previous_summaries: list[tuple[str, str]],
                    ongoing: str | None = None) -> str:
    """What Jarvis already knows, rendered for the system prompt.

    Separate from the persona so that remembering a new fact does not rewrite
    the cached prefix ahead of it.
    """
    parts: list[str] = [
        f"The current date and time is {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}."
    ]

    if facts:
        lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
        parts.append(f"Things you have been told to remember:\n{lines}")

    if previous_summaries:
        lines = "\n".join(f"- {when[:10]}: {text}" for when, text in previous_summaries)
        parts.append(
            "Earlier conversations with this user:\n" + lines +
            "\nRefer back to these naturally when they are relevant. Do not "
            "recite them unprompted."
        )

    if ongoing:
        parts.append(
            "Earlier in this same conversation, before the turns you can see "
            f"below:\n{ongoing}"
        )

    return "\n\n".join(parts)


SUMMARISE = """\
Summarise this stretch of a spoken conversation in at most five sentences.

Keep: what the user asked for, what was decided, anything they said about
themselves, and anything left unfinished. Drop: pleasantries, false starts,
and anything the speech recogniser plainly garbled.

Write it as plain prose for your own later reference - no headings or bullets.

{transcript}\
"""
