# Jarvis

A conversational voice assistant. You talk, it answers out loud, and it
remembers the conversation the next time you start it.

Audio in → [faster-whisper](https://github.com/SYSTRAN/faster-whisper) →
Claude → [ElevenLabs](https://elevenlabs.io) → audio out.

## How it works

```
 mic ──► VAD ──► whisper ──► Claude (streaming) ──► ElevenLabs ──► speaker
         │                     │      │                  ▲
   turn boundaries      tool calls  sentences ───────────┘
                             │      as they complete
                             ▼
                       SQLite memory
```

Three decisions shape everything else:

**Turn-taking is energy-based.** Whisper transcribes whatever buffer you hand
it, so something has to decide where a spoken turn starts and stops. That is
`jarvis/audio/vad.py`: it measures the room's noise floor at the start of every
listen and compares each 20 ms frame against it, so the thresholds in
`config.yaml` are ratios that hold up in a quiet room and a loud one. A
pre-roll buffer keeps the 300 ms *before* the trigger, which is what stops the
first syllable being clipped off the front.

**Replies are spoken sentence by sentence.** Claude's output is streamed, cut
into sentences as they complete (`jarvis/text.py`), and each one is sent to
ElevenLabs while the next is still being written. Waiting for the full reply
before synthesising would add its whole generation time to the pause before
Jarvis says anything.

**Memory has three layers.** Every turn is written to SQLite verbatim. When a
session outgrows the prompt budget its older turns are folded into a summary.
Durable things about you — a name, a preference, a project — are stored as
*facts*, which Claude writes deliberately through the `remember_fact` tool and
which are in the prompt of every future session.

## Setup

Needs Python 3.11–3.13. (`ctranslate2`, under faster-whisper, has no 3.14
wheels yet, so 3.14 will fail to install.)

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then add your two API keys
```

Check everything before you talk to it:

```bash
.venv/bin/python run.py doctor
```

That verifies both keys, finds your audio devices, downloads and loads the
whisper model, and makes one small Claude request. The first run downloads
~500 MB for `small.en`.

macOS will ask for microphone permission the first time you run `talk` or
`listen`. If it never asks and there is no input device, grant it under System
Settings → Privacy & Security → Microphone for your terminal.

## Using it

```bash
.venv/bin/python run.py talk
```

Speak; pause; it answers. Say "goodbye Jarvis" or ask it to stop, and it ends
the session and writes a summary for next time. Ctrl-C also exits cleanly.

Other commands:

| Command | What it does |
|---|---|
| `chat` | Same brain, same memory, over the keyboard. No mic, no API spend on TTS. |
| `say "text"` | Speak one line. Checks ElevenLabs and your speakers. |
| `listen -o out.wav` | Capture one utterance the way `talk` does, and transcribe it. |
| `transcribe f.wav` | Run STT over a file. |
| `calibrate` | Measure your room and print suggested VAD settings. |
| `devices` | List audio devices, with the defaults marked. |
| `voices` | List the ElevenLabs voices on your account. |
| `memory` | Show remembered facts and past session summaries. `--forget KEY` deletes one. |
| `doctor` | Check keys, audio, whisper, and Claude. |

Add `-v` to any of them for timings, token counts, and VAD decisions.

## Tuning

Everything lives in `config.yaml`; secrets live in `.env`.

**It does not hear me / it triggers on nothing.** Run `run.py calibrate`. It
measures your noise floor and speaking level and prints the `vad:` block to
paste in.

**It cuts me off mid-sentence.** Raise `vad.silence_hangover_ms`. That is how
long a pause may run before the turn is considered over — 800 ms suits brisk
speech, 1200 ms suits thinking out loud.

**It is too slow to start talking.** Drop `stt.model` to `base.en` or
`tiny.en`; whisper's load is usually the biggest share. `model.effort` is
already `low`. Cutting `vad.silence_hangover_ms` also helps, at the cost of
being interrupted more.

**Wrong microphone.** `run.py devices`, then set `audio.input_device` to the
index or name.

**Interrupting it.** `barge_in.enabled` is off by default. With open speakers
and no echo cancellation, Jarvis hears its own voice and interrupts itself
after half a sentence. Turn it on if you wear headphones.

## Model and cost

`claude-opus-5` at `effort: low` — low effort rather than thinking disabled,
because with thinking off Opus 5 will occasionally narrate a tool call as plain
text instead of emitting it, which in a spoken loop sounds exactly like Jarvis
lying about what it just did. Opus 5 is $5/$25 per million tokens; a voice turn
is small, and the system prompt is cached. Switch to `claude-sonnet-5` in
`config.yaml` if you would rather trade quality for cost.

ElevenLabs `eleven_flash_v2_5` is their lowest-latency model, and audio comes
back as raw 16 kHz PCM so there is no decode step and no ffmpeg dependency.

## Layout

```
run.py                 CLI: every command above
config.yaml            behaviour            .env  secrets
jarvis/
  assistant.py         Assistant (a conversation) + VoiceSession (the audio loop)
  brain.py             Claude: streaming, the tool loop, summarisation
  memory.py            SQLite: turns, session summaries, facts
  tools.py             remember_fact, forget_fact, get_current_time, end_conversation
  prompts.py           system prompt and the remembered-context blocks
  stt.py               faster-whisper, with hallucination filtering
  tts.py               ElevenLabs
  text.py              sentence chunking, markdown stripping
  audio/
    vad.py             turn detection
    recorder.py        microphone capture
    player.py          interruptible playback
tests/                 pytest; no API keys or audio hardware needed
```

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

They cover the parts that are painful to check by talking to it: turn detection
against synthetic audio, sentence chunking across arbitrary stream boundaries,
the memory window, and the streaming tool loop against a stubbed API.
