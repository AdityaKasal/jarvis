"""The Claude side of the loop: streaming replies and running tools."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import anthropic

from jarvis import prompts
from jarvis.text import SentenceChunker, strip_for_speech
from jarvis.tools import Toolbox

log = logging.getLogger("jarvis.brain")

# Tool calls chain: remember a fact, check the time, then answer. A cap keeps a
# confused model from looping until the rate limit stops it.
MAX_TOOL_TURNS = 6


class Brain:
    """Wraps one Claude conversation.

    The manual tool loop here rather than the SDK's tool runner: the runner
    hides per-token output, and per-token output is the entire point - each
    finished sentence is handed to TTS while Claude is still writing the next.
    """

    def __init__(self, config: dict[str, Any], toolbox: Toolbox):
        model_cfg = config.get("model", {}) or {}
        self.client = anthropic.Anthropic()
        self.model = model_cfg.get("id", "claude-opus-5")
        self.effort = model_cfg.get("effort", "low")
        self.max_tokens = model_cfg.get("max_tokens", 4096)
        self.name = (config.get("assistant", {}) or {}).get("name", "Jarvis")
        self.toolbox = toolbox

    def system(self, facts: dict[str, str],
               previous_summaries: list[tuple[str, str]],
               ongoing: str | None = None) -> list[dict[str, Any]]:
        """System prompt as two blocks: the frozen persona, then what's known.

        The cache breakpoint sits on the second block so both are cached
        together, and the persona ahead of it stays byte-identical all session.
        Short sessions will not see a cache hit at all - the minimum cacheable
        prefix is larger than this - but a long one reads the whole thing back
        at a tenth of the price.
        """
        return [
            {"type": "text", "text": prompts.persona(self.name)},
            {
                "type": "text",
                "text": prompts.knowledge_block(facts, previous_summaries, ongoing),
                "cache_control": {"type": "ephemeral"},
            },
        ]

    def respond(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        on_sentence: Optional[Callable[[str], Any]] = None,
    ) -> str:
        """Run one turn to completion, tools included. Returns the spoken text.

        `on_sentence` is called with each finished sentence as it arrives. It
        may raise `Interrupted` to abandon the turn - that is how barge-in
        stops Claude mid-answer instead of only stopping the speaker.
        """
        spoken: list[str] = []

        for turn in range(1, MAX_TOOL_TURNS + 1):
            chunker = SentenceChunker()

            def emit(text: str) -> None:
                clean = strip_for_speech(text)
                if not clean:
                    return
                spoken.append(clean)
                if on_sentence is not None:
                    on_sentence(clean)

            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=messages,
                tools=self.toolbox.definitions,
                output_config={"effort": self.effort},
            ) as stream:
                for chunk in stream.text_stream:
                    for sentence in chunker.feed(chunk):
                        emit(sentence)
                final = stream.get_final_message()

            # Flush before the tool runs: "let me check" should be spoken while
            # the lookup happens, not queued behind it.
            tail = chunker.flush()
            if tail:
                emit(tail)

            log.debug("turn %d: stop_reason=%s in=%d out=%d cached=%d",
                      turn, final.stop_reason, final.usage.input_tokens,
                      final.usage.output_tokens,
                      final.usage.cache_read_input_tokens or 0)

            if final.stop_reason == "refusal":
                detail = getattr(final, "stop_details", None)
                log.warning("refused: %s", getattr(detail, "category", "unknown"))
                if not spoken:
                    emit("I'd rather not answer that one.")
                break

            if final.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": final.content})
                continue

            if final.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": final.content})
                messages.append({"role": "user", "content": self._run_tools(final)})
                continue

            if final.stop_reason == "max_tokens":
                log.warning("reply hit max_tokens (%d) and was cut off",
                            self.max_tokens)
            break
        else:
            log.warning("gave up after %d tool turns", MAX_TOOL_TURNS)
            if not spoken:
                emit("I got stuck working that out. Ask me again?")

        return " ".join(spoken).strip()

    def _run_tools(self, message: Any) -> list[dict[str, Any]]:
        """Execute every tool_use block and return all results in one message.

        All of them, in one message: splitting results across several user
        messages teaches the model to stop making parallel calls, and dropping
        a failed one leaves a tool_use with no matching result, which is a 400.
        """
        results: list[dict[str, Any]] = []
        for block in message.content:
            if block.type != "tool_use":
                continue
            try:
                output = self.toolbox.run(block.name, dict(block.input))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
            except Exception as exc:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"{type(exc).__name__}: {exc}",
                    "is_error": True,
                })
        return results

    def summarise(self, transcript: str) -> str:
        """Condense a stretch of conversation. Used to keep the prompt bounded."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            output_config={"effort": "low"},
            messages=[{"role": "user",
                       "content": prompts.SUMMARISE.format(transcript=transcript)}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()
