"""Tools Claude can call mid-conversation.

Kept to four, deliberately. Every tool definition is in the prompt on every
request, and a voice turn is judged on how fast it starts talking.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from jarvis.memory import Memory

log = logging.getLogger("jarvis.tools")

# `strict` guarantees the input validates against the schema, which is what
# lets the handlers below index into `block.input` without defensive gets.
DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "remember_fact",
        "description": (
            "Store something about the user that should still be true in a "
            "week: their name, where they work, a standing preference, a "
            "project they are in the middle of. Not for details that only "
            "matter inside this conversation - you can already see those."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Short lowercase identifier, e.g. 'name', "
                                   "'employer', 'coffee order'. Reusing an "
                                   "existing key overwrites it.",
                },
                "value": {
                    "type": "string",
                    "description": "The fact, in one short sentence.",
                },
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "forget_fact",
        "description": "Delete a remembered fact, by key, when the user asks "
                       "you to forget it or tells you it is no longer true.",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_current_time",
        "description": "The current local date and time. The prompt's timestamp "
                       "is from the start of the conversation, so use this for "
                       "anything time-sensitive in a long session.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "end_conversation",
        "description": (
            "Stop listening and end the session. Call this when the user says "
            "goodbye, says they are done, or asks you to stop. Say your "
            "farewell in the same turn - it is spoken before the session ends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Briefly, why the conversation is ending.",
                }
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class Toolbox:
    """Executes tool calls and tracks the one side effect the loop cares about."""

    def __init__(self, memory: Memory):
        self.memory = memory
        self.should_end = False

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return DEFINITIONS

    def run(self, name: str, args: dict[str, Any]) -> str:
        try:
            return self._dispatch(name, args)
        except Exception as exc:
            # A failed tool is reported back to Claude as a result, not raised.
            # Mid-conversation, "I could not look that up" is a far better
            # outcome than a traceback that drops the call.
            log.warning("tool %s failed: %s: %s", name, type(exc).__name__, exc)
            raise

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        if name == "remember_fact":
            self.memory.remember(args["key"], args["value"])
            log.info("remembered %s = %s", args["key"], args["value"])
            return f"Remembered. {args['key']}: {args['value']}"

        if name == "forget_fact":
            if self.memory.forget(args["key"]):
                log.info("forgot %s", args["key"])
                return f"Forgotten: {args['key']}"
            known = ", ".join(self.memory.facts()) or "nothing"
            return (f"No fact stored under {args['key']!r}. "
                    f"Currently remembered: {known}.")

        if name == "get_current_time":
            return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

        if name == "end_conversation":
            self.should_end = True
            log.info("ending conversation: %s", args.get("reason", ""))
            return ("The session will end once you finish speaking. "
                    "Say goodbye now.")

        raise ValueError(f"unknown tool: {name}")
