"""Orchestration: one class that holds a conversation, one that speaks it."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from jarvis.audio import Microphone, Speaker, Vad, VadConfig, listen
from jarvis.brain import Brain
from jarvis.events import EventBus
from jarvis.wake import WakeWord
from jarvis.memory import Memory
from jarvis.stt import Transcriber
from jarvis.tools import Toolbox
from jarvis.tts import Voice

log = logging.getLogger("jarvis")


class Interrupted(Exception):
    """Raised out of a sentence callback when the user talks over the reply."""


class Assistant:
    """A conversation with memory. Knows nothing about audio.

    Everything here works over text, which is what makes `run.py chat` a real
    test of the brain and the memory rather than a separate code path.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        mem_cfg = config.get("memory", {}) or {}
        self.recent_turns = mem_cfg.get("recent_turns", 20)
        self.summarise_after = mem_cfg.get("summarize_after", 40)
        self.carry_sessions = mem_cfg.get("carry_sessions", 3)

        self.memory = Memory(config["_root"] / mem_cfg.get("db", "jarvis.db"))
        self.toolbox = Toolbox(self.memory)
        self.brain = Brain(config, self.toolbox)
        self.session_id = self.memory.start_session()
        log.debug("session %d started", self.session_id)

    @property
    def should_end(self) -> bool:
        return self.toolbox.should_end

    @should_end.setter
    def should_end(self, value: bool) -> None:
        # Cleared when a farewell only puts Jarvis back to sleep, so the next
        # conversation does not inherit the last one's decision to end.
        self.toolbox.should_end = value

    def turn(self, user_text: str,
             on_sentence: Optional[Callable[[str], Any]] = None) -> str:
        """One exchange: record what was said, answer it, record the answer."""
        self.memory.add_turn(self.session_id, "user", user_text)

        summary, _ = self.memory.session_summary(self.session_id)
        system = self.brain.system(
            facts=self.memory.facts(),
            previous_summaries=self.memory.previous_summaries(
                self.session_id, self.carry_sessions),
            ongoing=summary,
        )

        reply = self.brain.respond(system, self._history(), on_sentence)
        if reply:
            self.memory.add_turn(self.session_id, "assistant", reply)
        self._maybe_summarise()
        return reply

    def finish(self) -> None:
        """Close the session, leaving behind a summary for the next one."""
        try:
            summary = self._summarise_all()
        except Exception as exc:
            # Failing to summarise must not lose the session record itself.
            log.warning("could not summarise session: %s: %s",
                        type(exc).__name__, exc)
            summary = None
        self.memory.end_session(self.session_id, summary)
        log.debug("session %d ended", self.session_id)

    def _history(self) -> list[dict[str, Any]]:
        """Recent turns as API messages.

        Only what was actually said is stored, not the tool_use blocks that
        produced it: the transcript is the conversation's memory, and replaying
        old tool mechanics into it buys nothing.
        """
        _, through = self.memory.session_summary(self.session_id)
        turns = self.memory.turns_after(self.session_id, through, self.recent_turns)

        # The window can open on an assistant turn if the last exchange fell on
        # the boundary; the API requires the first message to be from the user.
        while turns and turns[0]["role"] != "user":
            turns.pop(0)

        return [{"role": t["role"], "content": t["content"]} for t in turns]

    def _maybe_summarise(self) -> None:
        summary, through = self.memory.session_summary(self.session_id)
        if self.memory.turn_count(self.session_id, through) < self.summarise_after:
            return

        turns = self.memory.turns_after(self.session_id, through, limit=100_000)
        fold = turns[: -self.recent_turns] if self.recent_turns else turns
        if not fold:
            return

        log.debug("folding %d turn(s) into the session summary", len(fold))
        try:
            self.memory.set_session_summary(
                self.session_id,
                self.brain.summarise(self._transcript(fold, summary)),
                fold[-1]["id"],
            )
        except Exception as exc:
            # Worst case the prompt stays long for another few turns.
            log.warning("summarisation failed: %s: %s", type(exc).__name__, exc)

    def _summarise_all(self) -> Optional[str]:
        summary, through = self.memory.session_summary(self.session_id)
        turns = self.memory.turns_after(self.session_id, through, limit=100_000)
        if not turns and not summary:
            return None
        if not turns:
            return summary
        return self.brain.summarise(self._transcript(turns, summary))

    @staticmethod
    def _transcript(turns: list[dict], prior: Optional[str]) -> str:
        body = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
        return f"Summary so far: {prior}\n\n{body}" if prior else body


class VoiceSession:
    """The audio loop: listen, transcribe, answer, speak, repeat.

    Publishes what it is doing to an `EventBus`. The console printer and every
    connected browser subscribe to that; the loop itself has no idea whether
    anyone is watching, which is what lets the same code back both `run.py
    talk` and the app without a second implementation.
    """

    STATES = ("starting", "asleep", "listening", "hearing", "transcribing",
              "thinking", "speaking", "stopped")

    def __init__(self, assistant: Assistant, config: dict[str, Any],
                 events: Optional[EventBus] = None, console: bool = True):
        self.assistant = assistant
        self.config = config
        self.events = events or EventBus()
        self.console = console

        audio = config.get("audio", {}) or {}
        self.sample_rate = audio.get("sample_rate", 16000)
        self.vad = Vad(VadConfig.from_config(config))

        self.mic = Microphone(
            sample_rate=self.sample_rate,
            frame_ms=audio.get("frame_ms", 20),
            device=audio.get("input_device"),
        )
        self.speaker = Speaker(
            sample_rate=self.sample_rate,
            device=audio.get("output_device"),
        )
        self.transcriber = Transcriber(config, sample_rate=self.sample_rate)
        self.voice = Voice(config)

        barge = config.get("barge_in", {}) or {}
        self.barge_in = bool(barge.get("enabled", False))
        self.barge_ratio = barge.get("ratio", 6.0)
        self._stop_watching = threading.Event()
        self._stopping = threading.Event()

        wake_cfg = config.get("wake", {}) or {}
        self.wake = WakeWord.from_config(config)
        self.stay_awake_s = wake_cfg.get("stay_awake_s", 45)
        self.acknowledgement = wake_cfg.get("acknowledgement", "Yes?")
        # Monotonic deadline; 0 means asleep. Not a bool, because "awake" is a
        # window that expires rather than a state something has to clear.
        self._awake_until = 0.0

        assistant_cfg = config.get("assistant", {}) or {}
        self.greeting = assistant_cfg.get("greeting", "I'm listening.")
        self.farewells = [f.lower() for f in assistant_cfg.get("farewells", [])]

        self.state = "stopped"
        # Levels arrive 50x a second; a meter needs about ten.
        self._last_level_at = 0.0

    # --- observation ---

    def _set_state(self, state: str) -> None:
        self.state = state
        self.events.publish("state", value=state)
        if self.console:
            labels = {"listening": "[listening]", "hearing": "[hearing you]",
                      "asleep": f"[asleep - say \"{self.wake.phrases[0][0]}\"]"
                                if self.wake.phrases else "[asleep]",
                      "transcribing": "[transcribing]", "thinking": "[thinking]",
                      "speaking": "[speaking]", "stopped": "[stopped]"}
            if state in labels:
                print(labels[state], flush=True)

    def _say_turn(self, role: str, text: str) -> None:
        self.events.publish("turn", role=role, text=text)
        if self.console:
            who = "you" if role == "user" else self.assistant.brain.name.lower()
            print(f"{who}: {text}")

    def _report_level(self, level: float) -> None:
        now = time.monotonic()
        if now - self._last_level_at < 0.1:
            return
        self._last_level_at = now
        self.events.publish("level", value=round(level, 5),
                            threshold=round(max(self.vad.noise_floor
                                                * self.vad.config.start_ratio,
                                                self.vad.config.absolute_floor), 5))

    @property
    def awake(self) -> bool:
        if not self.wake.enabled:
            return True
        return time.monotonic() < self._awake_until

    def _stay_awake(self) -> None:
        self._awake_until = time.monotonic() + self.stay_awake_s

    def _sleep(self) -> None:
        self._awake_until = 0.0

    # --- control ---

    def stop(self) -> None:
        """Ask the loop to finish. Safe from another thread."""
        self._stopping.set()
        self.speaker.interrupt()

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    # --- the loop ---

    def run(self) -> None:
        self._stopping.clear()
        self._set_state("starting")
        try:
            # Load whisper before opening the mic: the first inference is
            # several times slower than the rest, and it should not land in the
            # pause after the user's opening sentence.
            self.transcriber.warm_up()
            with self.mic, self.speaker:
                if self.greeting:
                    self.say(self.greeting)
                self._loop()
        except Exception as exc:
            log.error("voice loop failed: %s: %s", type(exc).__name__, exc)
            self.events.publish("error", message=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._set_state("stopped")

    def _loop(self) -> None:
        while not self.stopping:
            self._set_state("listening" if self.awake else "asleep")
            audio = listen(self.mic, self.vad,
                           on_speech_start=lambda: self._set_state("hearing"),
                           stop_check=lambda: self._stopping.is_set(),
                           on_level=self._report_level)
            if audio is None:
                continue

            self._set_state("transcribing")
            heard = self.transcriber.transcribe(audio)
            if not heard:
                log.debug("nothing intelligible in %.1fs of audio",
                          audio.size / self.sample_rate)
                continue

            asked = self._address(heard)
            if asked is None:
                # Heard, understood, and not for us. Published so the app can
                # show why nothing happened, rather than looking broken.
                log.debug("ignored (no wake word): %r", heard)
                self.events.publish("ignored", text=heard)
                continue

            self._stay_awake()
            self._say_turn("user", asked or heard)

            if not asked:
                # Woken by name with nothing else said.
                if self.acknowledgement:
                    self.say(self.acknowledgement)
                continue

            if self._is_farewell(asked):
                self.say("Goodbye.")
                # With a wake word there is somewhere to go back to, so a
                # farewell means "stop listening", not "quit".
                if self.wake.enabled:
                    self._sleep()
                    continue
                return

            reply = self._answer(asked)
            if reply:
                self._say_turn("assistant", reply)
            self.events.publish("facts", facts=self.assistant.memory.facts())
            self._stay_awake()
            if self.assistant.should_end:
                self.assistant.should_end = False
                if self.wake.enabled:
                    self._sleep()
                    continue
                return

    def _answer(self, heard: str) -> str:
        """Run a turn, speaking each sentence as it is written."""
        self.speaker.reset()
        self._set_state("thinking")
        watcher = self._start_barge_in_watch()
        try:
            return self.assistant.turn(heard, on_sentence=self.speak_sentence)
        except Interrupted:
            self.events.publish("interrupted")
            if self.console:
                print("[interrupted]")
            return ""
        finally:
            self._stop_watching.set()
            if watcher is not None:
                watcher.join(timeout=1.0)
            self.speaker.drain()

    def speak_sentence(self, sentence: str) -> None:
        self._set_state("speaking")
        self.events.publish("sentence", text=sentence)
        if not self.speaker.play(self.voice.stream(sentence)):
            raise Interrupted(sentence)

    def say(self, text: str) -> None:
        """Speak one line outside a conversation turn (greeting, goodbye)."""
        self.speaker.reset()
        self._set_state("speaking")
        self._say_turn("assistant", text)
        self.speaker.play(self.voice.stream(text))
        self.speaker.drain()

    def _address(self, heard: str) -> Optional[str]:
        """What was actually asked, or None if this was not addressed to us.

        An empty string means the wake word was said on its own.
        """
        if self.awake:
            return heard
        return self.wake.find(heard)

    def _is_farewell(self, heard: str) -> bool:
        cleaned = heard.lower().strip().strip(".!?")
        return any(phrase in cleaned for phrase in self.farewells)

    def _start_barge_in_watch(self) -> Optional[threading.Thread]:
        """Watch the mic during playback and cut the reply short if you speak.

        Off unless configured. Without acoustic echo cancellation, open
        speakers put Jarvis's own voice into the mic at full strength and it
        interrupts itself mid-sentence; on headphones it works properly.
        """
        if not self.barge_in or self.vad.noise_floor <= 0:
            return None

        threshold = max(self.vad.noise_floor * self.barge_ratio,
                        self.vad.config.absolute_floor * 2)
        self._stop_watching.clear()

        def watch() -> None:
            frame_s = self.vad.config.frame_ms / 1000
            needed = self.vad.config.min_speech_frames
            run = 0
            while not self._stop_watching.wait(frame_s):
                if self.mic.level >= threshold:
                    run += 1
                    if run >= needed:
                        log.debug("barge-in at %.4f (threshold %.4f)",
                                  self.mic.level, threshold)
                        self.speaker.interrupt()
                        return
                else:
                    run = 0

        thread = threading.Thread(target=watch, name="barge-in", daemon=True)
        thread.start()
        return thread


def text_session(assistant: Assistant, name: str = "jarvis") -> None:
    """Keyboard conversation against the same brain and the same memory."""
    print("Type to talk. Ctrl-D or 'quit' to end.\n")
    while True:
        try:
            heard = input("you: ").strip()
        except EOFError:
            print()
            return
        if not heard:
            continue
        if heard.lower() in {"quit", "exit"}:
            return

        started = time.monotonic()
        printed = False

        def show(sentence: str) -> None:
            nonlocal printed
            if not printed:
                print(f"{name}: ", end="", flush=True)
                printed = True
            print(sentence, end=" ", flush=True)

        assistant.turn(heard, on_sentence=show)
        print(f"\n[{time.monotonic() - started:.1f}s]\n")
        if assistant.should_end:
            return
