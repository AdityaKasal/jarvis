#!/usr/bin/env python3
"""Jarvis, a voice assistant.

    python run.py talk           # the thing itself: speak, and it answers
    python run.py chat           # same brain and memory, over the keyboard
    python run.py say "hello"    # speak one line (checks TTS + speakers)
    python run.py listen -o a.wav  # capture one utterance the way talk does
    python run.py transcribe a.wav # run STT over a file
    python run.py calibrate      # live mic levels, for tuning the VAD
    python run.py devices        # list audio devices
    python run.py voices         # list ElevenLabs voices in your account
    python run.py memory         # what Jarvis remembers about you
    python run.py doctor         # check keys, audio, and the whisper model
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

from jarvis.config import load_config, require_env

log = logging.getLogger("jarvis")

# These log a line per HTTP request at INFO, which in a voice loop is a line
# per sentence spoken. Useful at -v, noise otherwise. "httpx2" is the one that
# matters: anthropic 1.x is built on httpx2, not httpx.
_CHATTY = ("httpx", "httpx2", "httpcore", "urllib3", "anthropic", "elevenlabs",
           "faster_whisper", "huggingface_hub")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in _CHATTY:
        logging.getLogger(name).setLevel(
            logging.DEBUG if verbose else logging.WARNING)


def read_wav(path: Path, target_rate: int = 16000) -> np.ndarray:
    """Load a wav as mono float32 at `target_rate`."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise RuntimeError(f"{path}: need 16-bit PCM, got "
                               f"{w.getsampwidth() * 8}-bit")
        rate, channels = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != target_rate:
        # Linear resampling is not hi-fi, but this path exists for debugging
        # recordings, not for the live loop, which is already at 16 kHz.
        n = int(round(audio.size * target_rate / rate))
        audio = np.interp(np.linspace(0, audio.size - 1, n),
                          np.arange(audio.size), audio).astype(np.float32)
    return audio


def write_wav(path: Path, audio: np.ndarray, rate: int = 16000) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def cmd_talk(cfg: dict) -> int:
    from jarvis.assistant import Assistant, VoiceSession

    require_env("ANTHROPIC_API_KEY")
    require_env("ELEVENLABS_API_KEY")

    assistant = Assistant(cfg)
    session = VoiceSession(assistant, cfg)
    try:
        session.run()
    except KeyboardInterrupt:
        print("\n[stopped]")
    finally:
        # Always close the session out: the summary written here is what the
        # next conversation opens with.
        assistant.finish()
    return 0


def cmd_chat(cfg: dict) -> int:
    from jarvis.assistant import Assistant, text_session

    require_env("ANTHROPIC_API_KEY")
    assistant = Assistant(cfg)
    try:
        text_session(assistant, name=cfg["assistant"]["name"].lower())
    except KeyboardInterrupt:
        print()
    finally:
        assistant.finish()
    return 0


def cmd_say(cfg: dict, args) -> int:
    from jarvis.audio import Speaker
    from jarvis.tts import Voice

    require_env("ELEVENLABS_API_KEY")
    voice = Voice(cfg)
    started = time.monotonic()
    with Speaker(cfg["audio"]["sample_rate"],
                 device=cfg["audio"].get("output_device")) as speaker:
        speaker.play(voice.stream(args.text))
        speaker.drain()
    print(f"[{time.monotonic() - started:.1f}s]")
    return 0


def cmd_listen(cfg: dict, args) -> int:
    from jarvis.audio import Microphone, Vad, VadConfig, listen

    vad = Vad(VadConfig.from_config(cfg))
    with Microphone(cfg["audio"]["sample_rate"], cfg["audio"]["frame_ms"],
                    device=cfg["audio"].get("input_device")) as mic:
        print(f"Speak. Recording stops when you do (or after {args.timeout:.0f}s "
              "of silence).")
        audio = listen(mic, vad, on_speech_start=lambda: print("[hearing you]"),
                       timeout_s=args.timeout)

    if audio is None or audio.size == 0:
        print("Heard nothing.")
        return 1

    seconds = audio.size / cfg["audio"]["sample_rate"]
    print(f"Captured {seconds:.1f}s (noise floor {vad.noise_floor:.4f})")
    if args.out:
        write_wav(Path(args.out), audio, cfg["audio"]["sample_rate"])
        print(f"Wrote {args.out}")
    if not args.no_transcribe:
        from jarvis.stt import Transcriber
        print(f"Heard: {Transcriber(cfg).transcribe(audio)!r}")
    return 0


def cmd_transcribe(cfg: dict, args) -> int:
    from jarvis.stt import Transcriber

    path = Path(args.path)
    if not path.exists():
        log.error("no such file: %s", path)
        return 1
    audio = read_wav(path, cfg["audio"]["sample_rate"])
    print(Transcriber(cfg).transcribe(audio))
    return 0


def cmd_calibrate(cfg: dict, args) -> int:
    """Measure the room, then measure your voice. Prints a `vad:` block.

    The speech phase waits for you rather than running on a timer: a fixed
    window means whoever is at the mic has to be watching the terminal at the
    right moment, and if they are not, the measurement comes back with speech
    quieter than silence and no sign that anything went wrong.
    """
    from jarvis.audio import Microphone, rms

    device = cfg["audio"].get("input_device")
    quiet: list[float] = []
    loud: list[float] = []

    with Microphone(cfg["audio"]["sample_rate"], cfg["audio"]["frame_ms"],
                    device=device) as mic:
        print(f"Stay quiet for {args.seconds:.0f}s...")
        deadline = time.monotonic() + args.seconds
        for frame in mic.frames():
            quiet.append(rms(frame))
            if time.monotonic() >= deadline:
                break

        floor = float(np.median(quiet))
        if floor <= 0.00005:
            print(f"\nThe microphone is delivering silence (level {floor:.5f}).")
            print("Check macOS microphone permission for your terminal, or pick "
                  "a different audio.input_device - `run.py devices` lists them.")
            return 1

        print(f"Room measured ({floor:.4f}). Now talk normally - I'll wait, and "
              "stop once I've heard enough.")
        mic.drain()
        threshold = max(floor * 2.0, 0.002)
        needed = int(args.seconds * 1000 / cfg["audio"]["frame_ms"] / 2)
        give_up = time.monotonic() + 60.0

        for frame in mic.frames():
            level = rms(frame)
            if level >= threshold:
                loud.append(level)
                bar = "#" * min(50, int(level * 500))
                print(f"\r  {len(loud) * 100 // needed:>3}%  {bar:<50}",
                      end="", flush=True)
            if len(loud) >= needed:
                break
            if time.monotonic() >= give_up:
                print("\n\nHeard nothing above the room noise in 60s.")
                print("Is the right device selected, and is it unmuted?")
                return 1

    speech = float(np.percentile(loud, 75))
    ratio = speech / floor
    print(f"\n\nnoise floor : {floor:.4f}")
    print(f"speech level: {speech:.4f}")
    print(f"ratio       : {ratio:.1f}x")

    if ratio < 3:
        print("\nThat is a narrow margin - the mic is far away, or the room is "
              "loud. Turn detection will be unreliable; a closer mic helps more "
              "than any setting below.")

    print(f"\nSuggested config.yaml:\n  vad:"
          f"\n    start_ratio: {max(2.0, ratio / 3):.1f}"
          f"\n    stop_ratio: {max(1.5, ratio / 5):.1f}"
          f"\n    absolute_floor: {max(0.002, floor * 2):.4f}")
    return 0


def cmd_devices(cfg: dict) -> int:
    import sounddevice as sd

    default_in, default_out = sd.default.device
    for i, dev in enumerate(sd.query_devices()):
        marks = []
        if dev["max_input_channels"]:
            marks.append("in")
        if dev["max_output_channels"]:
            marks.append("out")
        flag = ""
        if i == default_in:
            flag += " <- default input"
        if i == default_out:
            flag += " <- default output"
        print(f"{i:>3}  {dev['name']:<40} {'/'.join(marks):<8}"
              f"{int(dev['default_samplerate'])} Hz{flag}")
    print("\nSet audio.input_device / audio.output_device in config.yaml to an "
          "index or a name.")
    return 0


def cmd_voices(cfg: dict) -> int:
    from jarvis.tts import Voice

    require_env("ELEVENLABS_API_KEY")
    current = cfg["tts"]["voice_id"]
    for voice_id, name in Voice(cfg).voices():
        mark = "  <- current" if voice_id == current else ""
        print(f"{voice_id}  {name}{mark}")
    return 0


def cmd_memory(cfg: dict, args) -> int:
    from jarvis.memory import Memory

    memory = Memory(cfg["_root"] / cfg["memory"]["db"])

    if args.forget:
        print("Forgotten." if memory.forget(args.forget)
              else f"Nothing stored under {args.forget!r}.")
        return 0

    facts = memory.facts()
    print("Facts:" if facts else "Facts: (none yet)")
    for key, value in facts.items():
        print(f"  {key}: {value}")

    # previous_summaries takes an exclusive upper bound; there is no session
    # in progress here, so ask for everything below a sentinel above any id.
    history = memory.previous_summaries(before_session=2**31, limit=args.sessions)
    print(f"\nLast {len(history)} session(s):" if history else "\nNo past sessions.")
    for when, summary in history:
        print(f"  {when}\n    {summary}")
    return 0


def _why(exc: Exception) -> str:
    """One readable line from an SDK exception.

    Both SDKs put the useful sentence inside a `body` dict and stringify to a
    full dump of response headers, which buries it.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, dict) and detail.get("message"):
            return f"{type(exc).__name__}: {detail['message']}"
        if body.get("error", {}).get("message"):
            return f"{type(exc).__name__}: {body['error']['message']}"
    return f"{type(exc).__name__}: {exc}"


def cmd_doctor(cfg: dict) -> int:
    """Check everything that can be checked without saying a word."""
    ok = True

    for key in ("ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY"):
        if os.environ.get(key):
            print(f"  ok    {key} is set")
        else:
            print(f"  FAIL  {key} is not set (copy .env.example to .env)")
            ok = False

    try:
        import sounddevice as sd
        devices = sd.query_devices()
        ins = [d for d in devices if d["max_input_channels"]]
        outs = [d for d in devices if d["max_output_channels"]]
        print(f"  ok    audio: {len(ins)} input, {len(outs)} output device(s)")
        if not ins:
            print("  FAIL  no input device - check macOS microphone permission")
            ok = False
    except Exception as exc:
        print(f"  FAIL  audio: {type(exc).__name__}: {exc}")
        ok = False

    try:
        from jarvis.stt import Transcriber
        started = time.monotonic()
        Transcriber(cfg).warm_up()
        print(f"  ok    whisper {cfg['stt']['model']} loaded "
              f"({time.monotonic() - started:.1f}s)")
    except Exception as exc:
        print(f"  FAIL  whisper: {type(exc).__name__}: {exc}")
        ok = False

    if os.environ.get("ELEVENLABS_API_KEY"):
        try:
            from jarvis.tts import Voice
            # Synthesise one word and count the bytes. This is what catches a
            # key that is set but wrong, or one whose Text to Speech permission
            # was left at No Access - neither shows up until you try to speak.
            audio = b"".join(Voice(cfg).stream("ok"))
            print(f"  ok    ElevenLabs {cfg['tts']['model_id']} synthesised "
                  f"{len(audio) / (cfg['audio']['sample_rate'] * 2):.1f}s of audio")
        except Exception as exc:
            print(f"  FAIL  ElevenLabs: {_why(exc)}")
            print(f"        (voice {cfg['tts']['voice_id']}; "
                  "`run.py voices` lists the ones on your account)")
            ok = False

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            reply = anthropic.Anthropic().messages.create(
                model=cfg["model"]["id"], max_tokens=16,
                messages=[{"role": "user", "content": "Reply with the word ok."}],
            )
            print(f"  ok    {cfg['model']['id']} reachable "
                  f"({reply.usage.input_tokens} in / "
                  f"{reply.usage.output_tokens} out)")
        except Exception as exc:
            print(f"  FAIL  Claude: {_why(exc)}")
            ok = False

    print("\nAll good." if ok else "\nSomething above needs fixing.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-c", "--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("talk", help="the voice loop")
    sub.add_parser("chat", help="same assistant, over the keyboard")
    sub.add_parser("devices", help="list audio devices")
    sub.add_parser("voices", help="list ElevenLabs voices")
    sub.add_parser("doctor", help="check keys, audio and models")

    p_say = sub.add_parser("say", help="speak one line")
    p_say.add_argument("text")

    p_listen = sub.add_parser("listen", help="capture one utterance")
    p_listen.add_argument("-o", "--out", help="write the captured audio here")
    p_listen.add_argument("--no-transcribe", action="store_true")
    p_listen.add_argument("-t", "--timeout", type=float, default=15.0,
                          help="give up after this many seconds of silence")

    p_tr = sub.add_parser("transcribe", help="transcribe a wav file")
    p_tr.add_argument("path")

    p_cal = sub.add_parser("calibrate", help="measure your room for the VAD")
    p_cal.add_argument("-s", "--seconds", type=float, default=5.0)

    p_mem = sub.add_parser("memory", help="show what Jarvis remembers")
    p_mem.add_argument("--forget", metavar="KEY", help="delete one fact")
    p_mem.add_argument("--sessions", type=int, default=5)

    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = load_config(args.config)

    try:
        if args.command == "talk":
            return cmd_talk(cfg)
        if args.command == "chat":
            return cmd_chat(cfg)
        if args.command == "say":
            return cmd_say(cfg, args)
        if args.command == "listen":
            return cmd_listen(cfg, args)
        if args.command == "transcribe":
            return cmd_transcribe(cfg, args)
        if args.command == "calibrate":
            return cmd_calibrate(cfg, args)
        if args.command == "devices":
            return cmd_devices(cfg)
        if args.command == "voices":
            return cmd_voices(cfg)
        if args.command == "memory":
            return cmd_memory(cfg, args)
        if args.command == "doctor":
            return cmd_doctor(cfg)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        if args.verbose:
            raise
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
